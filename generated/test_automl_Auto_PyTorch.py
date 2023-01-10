import sys
_module = sys.modules[__name__]
del sys
autoPyTorch = _module
__version__ = _module
api = _module
base_task = _module
tabular_classification = _module
tabular_regression = _module
time_series_forecasting = _module
configs = _module
constants = _module
data = _module
base_feature_validator = _module
base_target_validator = _module
base_validator = _module
tabular_feature_validator = _module
tabular_target_validator = _module
tabular_validator = _module
time_series_feature_validator = _module
time_series_forecasting_validator = _module
time_series_target_validator = _module
utils = _module
datasets = _module
base_dataset = _module
image_dataset = _module
resampling_strategy = _module
tabular_dataset = _module
time_series_dataset = _module
ensemble = _module
abstract_ensemble = _module
ensemble_builder = _module
ensemble_selection = _module
singlebest_ensemble = _module
evaluation = _module
abstract_evaluator = _module
tae = _module
test_evaluator = _module
time_series_forecasting_train_evaluator = _module
train_evaluator = _module
utils_extra = _module
metrics = _module
optimizer = _module
smbo = _module
pipeline = _module
base_pipeline = _module
components = _module
base_choice = _module
base_component = _module
preprocessing = _module
base_preprocessing = _module
image_preprocessing = _module
base_image_preprocessor = _module
ImageNormalizer = _module
NoNormalizer = _module
normalise = _module
base_normalizer = _module
TabularColumnTransformer = _module
tabular_preprocessing = _module
base_tabular_preprocessing = _module
MinorityCoalescer = _module
NoCoalescer = _module
coalescer = _module
base_coalescer = _module
NoEncoder = _module
OneHotEncoder = _module
encoding = _module
base_encoder = _module
ExtraTreesPreprocessorClassification = _module
ExtraTreesPreprocessorRegression = _module
FastICA = _module
FeatureAgglomeration = _module
KernelPCA = _module
LibLinearSVCPreprocessor = _module
NoFeaturePreprocessor = _module
Nystroem = _module
PCA = _module
PolynomialFeatures = _module
RandomKitchenSinks = _module
RandomTreesEmbedding = _module
SelectPercentileClassification = _module
SelectPercentileRegression = _module
SelectRatesClassification = _module
SelectRatesRegression = _module
TruncatedSVD = _module
feature_preprocessing = _module
base_feature_preprocessor = _module
SimpleImputer = _module
imputation = _module
base_imputer = _module
MinMaxScaler = _module
NoScaler = _module
Normalizer = _module
PowerTransformer = _module
QuantileTransformer = _module
RobustScaler = _module
StandardScaler = _module
scaling = _module
base_scaler = _module
VarianceThreshold = _module
variance_thresholding = _module
TimeSeriesTransformer = _module
time_series_preprocessing = _module
base_time_series_preprocessing = _module
time_series_base_encoder = _module
TimeSeriesImputer = _module
setup = _module
augmentation = _module
GaussianBlur = _module
GaussianNoise = _module
HorizontalFlip = _module
ImageAugmenter = _module
RandomAffine = _module
RandomCutout = _module
Resize = _module
VerticalFlip = _module
ZeroPadAndCrop = _module
image = _module
base_image_augmenter = _module
base_setup = _module
EarlyPreprocessing = _module
TimeSeriesEarlyPreProcessing = _module
early_preprocessor = _module
forecasting_target_scaling = _module
base_target_scaler = _module
utils = _module
DistributionLoss = _module
QuantileLoss = _module
RegressionLoss = _module
forecasting_training_loss = _module
base_forecasting_loss = _module
CosineAnnealingLR = _module
CosineAnnealingWarmRestarts = _module
CyclicLR = _module
ExponentialLR = _module
NoScheduler = _module
ReduceLROnPlateau = _module
StepLR = _module
lr_scheduler = _module
base_scheduler = _module
network = _module
base_network = _module
forecasting_architecture = _module
forecasting_network = _module
ConvNetImageBackbone = _module
DenseNetImageBackbone = _module
MLPBackbone = _module
ResNetBackbone = _module
ShapedMLPBackbone = _module
ShapedResNetBackbone = _module
network_backbone = _module
base_network_backbone = _module
forecasting_backbone = _module
cells = _module
components_util = _module
MLPDecoder = _module
NBEATSDecoder = _module
RNNDecoder = _module
TransformerDecoder = _module
forecasting_decoder = _module
base_forecasting_decoder = _module
components = _module
forecasting_encoder = _module
base_forecasting_encoder = _module
components = _module
MLPEncoder = _module
NBEATSEncoder = _module
flat_encoder = _module
InceptionTimeEncoder = _module
RNNEncoder = _module
TCNEncoder = _module
TransformerEncoder = _module
seq_encoder = _module
TemporalFusion = _module
other_components = _module
utils = _module
LearnedEntityEmbedding = _module
NoEmbedding = _module
network_embedding = _module
base_network_embedding = _module
network_head = _module
base_network_head = _module
NBEATS_head = _module
forecasting_network_head = _module
distribution = _module
forecasting_head = _module
fully_connected = _module
fully_convolutional = _module
utils = _module
KaimingInit = _module
NoInit = _module
OrthogonalInit = _module
SparseInit = _module
XavierInit = _module
network_initializer = _module
base_network_initializer = _module
AdamOptimizer = _module
AdamWOptimizer = _module
RMSpropOptimizer = _module
SGDOptimizer = _module
base_optimizer = _module
traditional_ml = _module
base_model = _module
tabular_traditional_model = _module
traditional_learner = _module
base_traditional_learner = _module
learners = _module
training = _module
base_training = _module
data_loader = _module
base_data_loader = _module
feature_data_loader = _module
image_data_loader = _module
time_series_forecasting_data_loader = _module
time_series_util = _module
losses = _module
base = _module
MixUpTrainer = _module
StandardTrainer = _module
trainer = _module
base_trainer = _module
ForecastingMixUpTrainer = _module
ForecastingStandardTrainer = _module
forecasting_trainer = _module
forecasting_base_trainer = _module
create_searchspace_util = _module
image_classification = _module
tabular_classification = _module
tabular_regression = _module
time_series_forecasting = _module
traditional_tabular_classification = _module
traditional_tabular_regression = _module
common = _module
hyperparameter_search_space_update = _module
implementations = _module
logging_ = _module
parallel = _module
parallel_model_runner = _module
results_manager = _module
results_visualizer = _module
single_thread_client = _module
stopwatch = _module
test_preselected_configs = _module
conf = _module
example_image_classification = _module
example_tabular_classification = _module
example_tabular_regression = _module
example_time_series_forecasting = _module
example_custom_configuration_space = _module
example_parallel_n_jobs = _module
example_pass_feature_types = _module
example_plot_over_time = _module
example_resampling_strategy = _module
example_run_with_portfolio = _module
example_single_configuration = _module
example_visualization = _module
test = _module
conftest = _module
test_api = _module
test_base_api = _module
test_data = _module
test_feature_validator = _module
test_forecasting_feature_validator = _module
test_forecasting_input_validator = _module
test_forecasting_target_validator = _module
test_target_validator = _module
test_utils = _module
test_validation = _module
test_base_dataset = _module
test_image_dataset = _module
test_resampling_strategies = _module
test_tabular_dataset = _module
test_time_series_datasets = _module
ensemble_utils = _module
test_ensemble = _module
test_evaluation = _module
evaluation_util = _module
test_abstract_evaluator = _module
test_evaluators = _module
test_forecasting_evaluators = _module
test_pipeline = _module
forecasting = _module
test_encoder_choice = _module
test_encoders = _module
test_imputer = _module
test_scaling = _module
test_time_series_transformer = _module
test_coalescer = _module
test_feature_preprocessor = _module
test_feature_preprocessor_choice = _module
test_imputers = _module
test_normalizer_choice = _module
test_normalizers = _module
test_scaler_choice = _module
test_scalers = _module
test_tabular_column_transformer = _module
test_variance_thresholding = _module
forecasting_networks = _module
test_base_components = _module
test_flat_backbones = _module
test_forecasting_architecture = _module
test_seq_encoder = _module
test_forecasting_target_scaling = _module
test_forecasting_training_losses = _module
test_setup = _module
test_setup_image_augmenter = _module
test_setup_networks = _module
test_setup_preprocessing_node = _module
test_setup_traditional_models = _module
base = _module
test_feature_data_loader = _module
test_forecasting_training = _module
test_image_data_loader = _module
test_time_series_data_loader = _module
test_training = _module
test_base_component = _module
test_losses = _module
test_metrics = _module
test_tabular_classification = _module
test_tabular_regression = _module
test_time_series_forecasting_pipeline = _module
test_traditional_pipeline = _module
test_coalescer_transformer = _module
test_common = _module
test_hyperparameter_search_space_update = _module
test_parallel_model_runner = _module
test_results_manager = _module
test_results_visualizer = _module
test_single_thread_client = _module

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


import uuid


from abc import ABCMeta


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


from typing import cast


import numpy as np


from scipy.sparse import issparse


from sklearn.utils.multiclass import type_of_target


from torch.utils.data import Dataset


from torch.utils.data import Subset


import torchvision


import torch


from torch.utils.data import TensorDataset


import torchvision.transforms


from torchvision.transforms import functional as TF


import copy


import warnings


import pandas as pd


from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data.dataset import Dataset


from collections import Counter


from sklearn.pipeline import Pipeline


from sklearn.utils.validation import check_random_state


from scipy.sparse import spmatrix


from sklearn.base import BaseEstimator


from sklearn.compose import ColumnTransformer


from sklearn.pipeline import make_pipeline


import torch.optim.lr_scheduler


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch import nn


from abc import abstractmethod


from torch.distributions import AffineTransform


from torch.distributions import TransformedDistribution


from typing import Iterable


import math


from collections import OrderedDict


from torch.nn import functional as F


from typing import Callable


from typing import NamedTuple


from enum import Enum


from torch.nn.utils import weight_norm


from torch.autograd import Function


import torch.nn as nn


from typing import Type


import torch.nn.functional as F


from torch.distributions import Beta


from torch.distributions import Distribution


from torch.distributions import Gamma


from torch.distributions import Normal


from torch.distributions import Poisson


from torch.distributions import StudentT


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import RMSprop


from torch.optim import SGD


import logging.handlers


from sklearn.utils import check_random_state


from sklearn.utils import check_array


from functools import partial


from typing import Iterator


import collections


from typing import Mapping


from typing import Sized


from torch._six import string_classes


from torch.utils.data._utils.collate import default_collate


from torch.utils.data._utils.collate import default_collate_err_msg_format


from torch.utils.data._utils.collate import np_str_obj_array_pattern


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import SubsetRandomSampler


from torch.nn.modules.loss import BCEWithLogitsLoss


from torch.nn.modules.loss import CrossEntropyLoss


from torch.nn.modules.loss import L1Loss


from torch.nn.modules.loss import MSELoss


from torch.nn.modules.loss import _Loss as Loss


import time


from torch.utils.tensorboard.writer import SummaryWriter


from abc import ABC


from sklearn.base import ClassifierMixin


from sklearn.base import RegressorMixin


from torch.utils.data.dataloader import default_collate


from scipy import sparse


from sklearn.base import TransformerMixin


import random


import sklearn.datasets


import re


from sklearn.datasets import fetch_openml


from sklearn.datasets import make_classification


from sklearn.datasets import make_regression


import itertools


from itertools import product


from sklearn.base import clone


import logging


from sklearn.preprocessing import StandardScaler


ALL_NET_OUTPUT = Union[torch.Tensor, List[torch.Tensor], torch.distributions.Distribution]


BaseDatasetPropertiesType = Union[int, float, str, List, bool, Tuple]


HyperparameterValueType = Union[int, str, float]


class HyperparameterSearchSpace(NamedTuple):
    """
    A class that holds the search space for an individual hyperparameter.
    Attributes:
        hyperparameter (str):
            name of the hyperparameter
        value_range (Sequence[HyperparameterValueType]):
            range of the hyperparameter, can be defined as min and
            max values for Numerical hyperparameter or a list of
            choices for a Categorical hyperparameter
        default_value (HyperparameterValueType):
            default value of the hyperparameter
        log (bool):
            whether to sample hyperparameter on a log scale
    """
    hyperparameter: str
    value_range: Sequence[HyperparameterValueType]
    default_value: HyperparameterValueType
    log: bool = False

    def __str__(self) ->str:
        """
        String representation for the Search Space
        """
        return 'Hyperparameter: %s | Range: %s | Default: %s | log: %s' % (self.hyperparameter, self.value_range, self.default_value, self.log)


VERY_SMALL_VALUE = 1e-12


class TargetScaler(BaseEstimator):
    """
    To accelerate training, this scaler is only applied under trainer (after the data is loaded by dataloader)
    """

    def __init__(self, mode: str):
        self.mode = mode

    def fit(self, X: Dict, y: Any=None) ->'TargetScaler':
        return self

    def transform(self, past_targets: torch.Tensor, past_observed_values: torch.BoolTensor, future_targets: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if past_observed_values is None or torch.all(past_observed_values):
            if self.mode == 'standard':
                loc = torch.mean(past_targets, dim=1, keepdim=True)
                scale = torch.std(past_targets, dim=1, keepdim=True)
                offset_targets = past_targets - loc
                scale = torch.where(torch.logical_or(scale == 0.0, scale == torch.nan), offset_targets[:, [-1]], scale)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                return (past_targets - loc) / scale, future_targets, loc, scale
            elif self.mode == 'min_max':
                min_ = torch.min(past_targets, dim=1, keepdim=True)[0]
                max_ = torch.max(past_targets, dim=1, keepdim=True)[0]
                diff_ = max_ - min_
                loc = min_
                scale = torch.where(diff_ == 0, past_targets[:, [-1]], diff_)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                return (past_targets - loc) / scale, future_targets, loc, scale
            elif self.mode == 'max_abs':
                max_abs_ = torch.max(torch.abs(past_targets), dim=1, keepdim=True)[0]
                max_abs_[max_abs_ < VERY_SMALL_VALUE] = 1.0
                scale = max_abs_
                if future_targets is not None:
                    future_targets = future_targets / scale
                return past_targets / scale, future_targets, None, scale
            elif self.mode == 'mean_abs':
                mean_abs = torch.mean(torch.abs(past_targets), dim=1, keepdim=True)
                scale = torch.where(mean_abs == 0.0, past_targets[:, [-1]], mean_abs)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = future_targets / scale
                return past_targets / scale, future_targets, None, scale
            elif self.mode == 'none':
                return past_targets, future_targets, None, None
            else:
                raise ValueError(f'Unknown mode {self.mode} for Forecasting scaler')
        else:
            valid_past_targets = past_observed_values * past_targets
            valid_past_obs = torch.sum(past_observed_values, dim=1, keepdim=True)
            if self.mode == 'standard':
                dfredom = 1
                loc = torch.sum(valid_past_targets, dim=1, keepdim=True) / valid_past_obs
                scale = torch.sum(torch.square(valid_past_targets - loc * past_observed_values), dim=1, keepdim=True)
                scale /= valid_past_obs - dfredom
                scale = torch.sqrt(scale)
                offset_targets = past_targets - loc
                scale = torch.where(torch.logical_or(scale == 0.0, scale == torch.nan), offset_targets[:, [-1]], scale)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                scaled_past_targets = torch.where(past_observed_values, offset_targets / scale, past_targets)
                return scaled_past_targets, future_targets, loc, scale
            elif self.mode == 'min_max':
                obs_mask = ~past_observed_values
                min_masked_past_targets = past_targets.masked_fill(obs_mask, value=torch.inf)
                max_masked_past_targets = past_targets.masked_fill(obs_mask, value=-torch.inf)
                min_ = torch.min(min_masked_past_targets, dim=1, keepdim=True)[0]
                max_ = torch.max(max_masked_past_targets, dim=1, keepdim=True)[0]
                diff_ = max_ - min_
                loc = min_
                scale = torch.where(diff_ == 0, past_targets[:, [-1]], diff_)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = (future_targets - loc) / scale
                scaled_past_targets = torch.where(past_observed_values, (past_targets - loc) / scale, past_targets)
                return scaled_past_targets, future_targets, loc, scale
            elif self.mode == 'max_abs':
                max_abs_ = torch.max(torch.abs(valid_past_targets), dim=1, keepdim=True)[0]
                max_abs_[max_abs_ < VERY_SMALL_VALUE] = 1.0
                scale = max_abs_
                if future_targets is not None:
                    future_targets = future_targets / scale
                scaled_past_targets = torch.where(past_observed_values, past_targets / scale, past_targets)
                return scaled_past_targets, future_targets, None, scale
            elif self.mode == 'mean_abs':
                mean_abs = torch.sum(torch.abs(valid_past_targets), dim=1, keepdim=True) / valid_past_obs
                scale = torch.where(mean_abs == 0.0, valid_past_targets[:, [-1]], mean_abs)
                scale[scale < VERY_SMALL_VALUE] = 1.0
                if future_targets is not None:
                    future_targets = future_targets / scale
                scaled_past_targets = torch.where(past_observed_values, past_targets / scale, past_targets)
                return scaled_past_targets, future_targets, None, scale
            elif self.mode == 'none':
                return past_targets, future_targets, None, None
            else:
                raise ValueError(f'Unknown mode {self.mode} for Forecasting scaler')


class FitRequirement(NamedTuple):
    """
    A class that holds inputs required to fit a pipeline. Also indicates whether
    requirements have to be user specified or are generated by the pipeline itself.

    Attributes:
        name (str): The name of the variable expected in the input dictionary
        supported_types (Iterable[Type]): An iterable of all types that are supported
        user_defined (bool): If false, this requirement does not have to be given to the pipeline
        dataset_property (bool): If True, this requirement is automatically inferred
            by the Dataset class
    """
    name: str
    supported_types: Iterable[Type]
    user_defined: bool
    dataset_property: bool

    def __str__(self) ->str:
        """
        String representation for the requirements
        """
        return 'Name: %s | Supported types: %s | User defined: %s | Dataset property: %s' % (self.name, self.supported_types, self.user_defined, self.dataset_property)


class HyperparameterSearchSpaceUpdate:
    """
    Allows specifying update to the search space of a
    particular hyperparameter.

    Args:
        node_name (str):
            The name of the node in the pipeline
        hyperparameter (str):
            The name of the hyperparameter
        value_range (Sequence[HyperparameterValueType]):
            In case of categorical hyperparameter, defines the new categorical choices.
            In case of numerical hyperparameter, defines the new range
            in the form of (LOWER, UPPER)
        default_value (HyperparameterValueType):
            New default value for the hyperparameter
        log (bool) (default=False):
            In case of numerical hyperparameters, whether to sample on a log scale
    """

    def __init__(self, node_name: str, hyperparameter: str, value_range: Sequence[HyperparameterValueType], default_value: HyperparameterValueType, log: bool=False) ->None:
        self.node_name = node_name
        self.hyperparameter = hyperparameter
        if len(value_range) == 0:
            raise ValueError('The new value range needs at least one value')
        self.value_range = value_range
        self.log = log
        self.default_value = default_value

    def apply(self, pipeline: List[Tuple[str, BaseEstimator]]) ->None:
        """
        Applies the update to the appropriate hyperparameter of the pipeline
        Args:
            pipeline (List[Tuple[str, BaseEstimator]]):
                The named steps of the current autopytorch pipeline

        Returns:
            None
        """
        [node[1]._apply_search_space_update(self) for node in pipeline if node[0] == self.node_name]

    def __str__(self) ->str:
        return '{}, {}, {}, {}, {}'.format(self.node_name, self.hyperparameter, str(self.value_range), self.default_value if isinstance(self.default_value, str) else self.default_value, ' log' if self.log else '')

    def get_search_space(self, remove_prefix: Optional[str]=None) ->HyperparameterSearchSpace:
        """
        Get Update as a HyperparameterSearchSpace object.

        Args:
            remove_prefix (Optional[str]):
                if specified, remove given prefix from hyperparameter name

        Returns:
            HyperparameterSearchSpace
        """
        hyperparameter_name = self.hyperparameter
        if remove_prefix is not None:
            if remove_prefix in self.hyperparameter:
                hyperparameter_name = hyperparameter_name.replace(f'{remove_prefix}:', '')
        return HyperparameterSearchSpace(hyperparameter=hyperparameter_name, value_range=self.value_range, default_value=self.default_value, log=self.log)


class DecoderProperties(NamedTuple):
    """
    Decoder properties

    Args:
        has_hidden_states (bool):
            if the decoder has hidden states. A decoder with hidden states might have additional output and requires
            additional inputs
        has_local_layer (bool):
            if the decoder has local layer, in which case the output is also a 3D sequential feature
        recurrent (bool):
            if the decoder is recurrent. This determines if decoders can be auto-regressive
        lagged_input (bool):
            if the decoder accepts past targets as additional features
        multi_blocks (bool):
            If the decoder is stacked by multiple blocks (only for N-BEATS)
    """
    has_hidden_states: bool = False
    has_local_layer: bool = True
    recurrent: bool = False
    lagged_input: bool = False
    multi_blocks: bool = False


class DecoderBlockInfo(NamedTuple):
    """
    Decoder block infos

    Args:
        decoder (nn.Module):
            decoder network
        decoder_properties (EncoderProperties):
            decoder properties
        decoder_output_shape (Tuple[int, ...]):
            output shape that the decoder ought to output

        decoder_input_shape (Tuple[int, ...]):
            requried input shape of the decoder

    """
    decoder: nn.Module
    decoder_properties: DecoderProperties
    decoder_output_shape: Tuple[int, ...]
    decoder_input_shape: Tuple[int, ...]


class EncoderProperties(NamedTuple):
    """
    Encoder properties

    Args:
        has_hidden_states (bool):
            if the encoder has hidden states. An encoder with hidden states might have additional output
        bijective_seq_output (bool):
            if the encoder's output sequence has the same length as its input sequence's length
        fixed_input_seq_length (bool):
            if the encoder requries a fixed length of input (for instance, MLP)
        lagged_input (bool):
            if the encoder accepts past targets as additional features
        is_casual (bool):
            If the output of the encoder only depends on the past targets
    """
    has_hidden_states: bool = False
    bijective_seq_output: bool = True
    fixed_input_seq_length: bool = False
    lagged_input: bool = False
    is_casual: bool = True


class EncoderBlockInfo(NamedTuple):
    """
    Encoder block infos

    Args:
        encoder (nn.Module):
            encoder network
        encoder_properties (EncoderProperties):
            encoder properties
        encoder_input_shape (Tuple[int, ...]):
            requried input shape of the encoder
        encoder_output_shape (Tuple[int, ...]):
            output shape that the encoder ought to output
        n_hidden_states (int):
            number of hidden states
    """
    encoder: nn.Module
    encoder_properties: EncoderProperties
    encoder_input_shape: Tuple[int, ...]
    encoder_output_shape: Tuple[int, ...]
    n_hidden_states: int


class NetworkStructure(NamedTuple):
    num_blocks: int = 1
    variable_selection: bool = False
    share_single_variable_networks: bool = False
    use_temporal_fusion: bool = False
    skip_connection: bool = False
    skip_connection_type: str = 'add'
    grn_dropout_rate: float = 0.0


class AddLayer(nn.Module):

    def __init__(self, input_size: int, skip_size: int):
        super().__init__()
        if input_size == skip_size:
            self.fc = nn.Linear(skip_size, input_size)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) ->torch.Tensor:
        if hasattr(self, 'fc'):
            return self.norm(input + self.fc(skip))
        else:
            return self.norm(input)


class StackedDecoder(nn.Module):
    """
    Decoder network that is stacked by several decoders. Skip-connections can be applied to each stack. It decodes the
    encoded features (encoder2decoder) from each corresponding stacks and known_future_features to generate the decoded
    output features that will be further fed to the network decoder.
    """

    def __init__(self, network_structure: NetworkStructure, encoder: nn.ModuleDict, encoder_info: Dict[str, EncoderBlockInfo], decoder_info: Dict[str, DecoderBlockInfo]):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.first_block = -1
        self.skip_connection = network_structure.skip_connection
        self.decoder_has_hidden_states = []
        decoder = nn.ModuleDict()
        for i in range(1, self.num_blocks + 1):
            block_id = f'block_{i}'
            if block_id in decoder_info:
                self.first_block = i if self.first_block == -1 else self.first_block
                decoder[block_id] = decoder_info[block_id].decoder
                if decoder_info[block_id].decoder_properties.has_hidden_states:
                    self.decoder_has_hidden_states.append(True)
                else:
                    self.decoder_has_hidden_states.append(False)
                if self.skip_connection:
                    input_size_encoder = encoder_info[block_id].encoder_output_shape[-1]
                    skip_size_encoder = encoder_info[block_id].encoder_input_shape[-1]
                    input_size_decoder = decoder_info[block_id].decoder_output_shape[-1]
                    skip_size_decoder = decoder_info[block_id].decoder_input_shape[-1]
                    if skip_size_decoder > 0:
                        if input_size_encoder == input_size_decoder and skip_size_encoder == skip_size_decoder:
                            decoder[f'skip_connection_{i}'] = encoder[f'skip_connection_{i}']
                        elif network_structure.skip_connection_type == 'add':
                            decoder[f'skip_connection_{i}'] = AddLayer(input_size_decoder, skip_size_decoder)
                        elif network_structure.skip_connection_type == 'gate_add_norm':
                            decoder[f'skip_connection_{i}'] = GateAddNorm(input_size_decoder, hidden_size=input_size_decoder, skip_size=skip_size_decoder, dropout=network_structure.grn_dropout_rate)
        self.cached_intermediate_state = [torch.empty(0) for _ in range(self.num_blocks + 1 - self.first_block)]
        self.decoder = decoder

    def forward(self, x_future: Optional[torch.Tensor], encoder_output: List[torch.Tensor], pos_idx: Optional[Tuple[int]]=None, cache_intermediate_state: bool=False, incremental_update: bool=False) ->torch.Tensor:
        """
        A forward pass through the decoder

        Args:
            x_future (Optional[torch.Tensor]):
                known future features
            encoder_output (List[torch.Tensor])
                encoded features, stored as List, whereas each element in the list indicates encoded features from an
                encoder stack
            pos_idx (int)
                position index of the current x_future. This is applied to transformer decoder
            cache_intermediate_state (bool):
                if the intermediate values are cached
            incremental_update (bool):
                if an incremental update is applied, this is normally applied for auto-regressive model

        Returns:
            decoder_output (torch.Tensor):
                decoder output that will be passed to the network head
        """
        x = x_future
        for i, block_id in enumerate(range(self.first_block, self.num_blocks + 1)):
            decoder_i = self.decoder[f'block_{block_id}']
            if self.decoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = decoder_i(x_future=x, encoder_output=hx, pos_idx=pos_idx)
                else:
                    fx, hx = decoder_i(x_future=x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            elif incremental_update:
                fx = decoder_i(x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            else:
                fx = decoder_i(x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            skip_id = f'skip_connection_{block_id}'
            if self.skip_connection and skip_id in self.decoder and x is not None:
                fx = self.decoder[skip_id](fx, x)
            if cache_intermediate_state:
                if self.decoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
            x = fx
        return x


class EncoderOutputForm(Enum):
    NoOutput = 0
    HiddenStates = 1
    Sequence = 2
    SequenceLast = 3


class StackedEncoder(nn.Module):
    """
    Encoder network that is stacked by several encoders. Skip-connections can be applied to each stack. Each stack
    needs to generate a sequence of encoded features passed to the next stack and the
    corresponding decoder (encoder2decoder) that is located at the same layer.Additionally, if temporal fusion
    transformer is applied, the last encoder also needs to output the full encoded feature sequence
    """

    def __init__(self, network_structure: NetworkStructure, has_temporal_fusion: bool, encoder_info: Dict[str, EncoderBlockInfo], decoder_info: Dict[str, DecoderBlockInfo]):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.skip_connection = network_structure.skip_connection
        self.has_temporal_fusion = has_temporal_fusion
        self.encoder_output_type = [EncoderOutputForm.NoOutput] * self.num_blocks
        self.encoder_has_hidden_states = [False] * self.num_blocks
        len_cached_intermediate_states = self.num_blocks + 1 if self.has_temporal_fusion else self.num_blocks
        self.cached_intermediate_state = [torch.empty(0) for _ in range(len_cached_intermediate_states)]
        self.encoder_num_hidden_states = [0] * self.num_blocks
        encoder = nn.ModuleDict()
        for i, block_idx in enumerate(range(1, self.num_blocks + 1)):
            block_id = f'block_{block_idx}'
            encoder[block_id] = encoder_info[block_id].encoder
            if self.skip_connection:
                input_size = encoder_info[block_id].encoder_output_shape[-1]
                skip_size = encoder_info[block_id].encoder_input_shape[-1]
                if network_structure.skip_connection_type == 'add':
                    encoder[f'skip_connection_{block_idx}'] = AddLayer(input_size, skip_size)
                elif network_structure.skip_connection_type == 'gate_add_norm':
                    encoder[f'skip_connection_{block_idx}'] = GateAddNorm(input_size, hidden_size=input_size, skip_size=skip_size, dropout=network_structure.grn_dropout_rate)
            if block_id in decoder_info:
                if decoder_info[block_id].decoder_properties.recurrent:
                    if decoder_info[block_id].decoder_properties.has_hidden_states:
                        self.encoder_output_type[i] = EncoderOutputForm.HiddenStates
                    else:
                        self.encoder_output_type[i] = EncoderOutputForm.Sequence
                else:
                    self.encoder_output_type[i] = EncoderOutputForm.SequenceLast
            if encoder_info[block_id].encoder_properties.has_hidden_states:
                self.encoder_has_hidden_states[i] = True
                self.encoder_num_hidden_states[i] = encoder_info[block_id].n_hidden_states
            else:
                self.encoder_has_hidden_states[i] = False
        self.encoder = encoder

    def forward(self, encoder_input: torch.Tensor, additional_input: List[Optional[torch.Tensor]], output_seq: bool=False, cache_intermediate_state: bool=False, incremental_update: bool=False) ->Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        A forward pass through the encoder

        Args:
            encoder_input (torch.Tensor):
                encoder input
            additional_input (List[Optional[torch.Tensor]])
                additional input to the encoder, e.g., initial hidden states
            output_seq (bool)
                if the encoder want to generate a sequence of multiple time steps or a single time step
            cache_intermediate_state (bool):
                if the intermediate values are cached
            incremental_update (bool):
                if an incremental update is applied, this is normally applied for
                auto-regressive model, however, ony deepAR requires incremental update in encoder

        Returns:
            encoder2decoder ([List[torch.Tensor]]):
                encoder output that will be passed to decoders
            encoder_output (torch.Tensor):
                full sequential encoded features from the last encoder layer. Applied to temporal transformer
        """
        encoder2decoder = []
        x = encoder_input
        for i, block_id in enumerate(range(1, self.num_blocks + 1)):
            output_seq_i = output_seq or self.has_temporal_fusion or block_id < self.num_blocks
            encoder_i = self.encoder[f'block_{block_id}']
            if self.encoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = encoder_i(x, output_seq=False, hx=hx)
                else:
                    rnn_num_layers = encoder_i.config['num_layers']
                    hx = additional_input[i]
                    if hx is None:
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
                    elif self.encoder_num_hidden_states[i] == 1:
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx[0].expand((rnn_num_layers, -1, -1)).contiguous())
                    else:
                        hx = tuple(hx_i.expand(rnn_num_layers, -1, -1).contiguous() for hx_i in hx)
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
            elif incremental_update:
                x_all = torch.cat([self.cached_intermediate_state[i], x], dim=1)
                fx = encoder_i(x_all, output_seq=False)
            else:
                fx = encoder_i(x, output_seq=output_seq_i)
            if self.skip_connection:
                if output_seq_i:
                    fx = self.encoder[f'skip_connection_{block_id}'](fx, x)
                else:
                    fx = self.encoder[f'skip_connection_{block_id}'](fx, x[:, -1:])
            if self.encoder_output_type[i] == EncoderOutputForm.HiddenStates:
                encoder2decoder.append(hx)
            elif self.encoder_output_type[i] == EncoderOutputForm.Sequence:
                encoder2decoder.append(fx)
            elif self.encoder_output_type[i] == EncoderOutputForm.SequenceLast:
                if output_seq_i and not output_seq:
                    encoder2decoder.append(encoder_i.get_last_seq_value(fx).squeeze(1))
                else:
                    encoder2decoder.append(fx)
            else:
                raise NotImplementedError
            if cache_intermediate_state:
                if self.encoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
                elif incremental_update:
                    self.cached_intermediate_state[i] = x_all
                else:
                    self.cached_intermediate_state[i] = x
            x = fx
        if self.has_temporal_fusion:
            if incremental_update:
                self.cached_intermediate_state[i + 1] = torch.cat([self.cached_intermediate_state[i + 1], x], dim=1)
            else:
                self.cached_intermediate_state[i + 1] = x
            return encoder2decoder, x
        else:
            return encoder2decoder, None


class TemporalFusionLayer(nn.Module):
    """
    (Lim et al.
    Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,
    https://arxiv.org/abs/1912.09363)
    we follow the implementation from pytorch forecasting:
    https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py
    """

    def __init__(self, window_size: int, network_structure: NetworkStructure, network_encoder: Dict[str, EncoderBlockInfo], n_decoder_output_features: int, d_model: int, n_head: int, dropout: Optional[float]=None):
        super().__init__()
        num_blocks = network_structure.num_blocks
        last_block = f'block_{num_blocks}'
        n_encoder_output = network_encoder[last_block].encoder_output_shape[-1]
        self.window_size = window_size
        if n_decoder_output_features != n_encoder_output:
            self.decoder_proj_layer = nn.Linear(n_decoder_output_features, n_encoder_output, bias=False)
        else:
            self.decoder_proj_layer = None
        if network_structure.variable_selection:
            if network_structure.skip_connection:
                n_encoder_output_first = network_encoder['block_1'].encoder_output_shape[-1]
                self.static_context_enrichment = GatedResidualNetwork(n_encoder_output_first, n_encoder_output_first, n_encoder_output_first, dropout)
                self.enrichment = GatedResidualNetwork(input_size=n_encoder_output, hidden_size=n_encoder_output, output_size=d_model, dropout=dropout, context_size=n_encoder_output_first, residual=False)
                self.enrich_with_static = True
        if not hasattr(self, 'enrichment'):
            self.enrichment = GatedResidualNetwork(input_size=n_encoder_output, hidden_size=n_encoder_output, output_size=d_model, dropout=dropout, residual=False)
            self.enrich_with_static = False
        self.attention_fusion = InterpretableMultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout or 0.0)
        self.post_attn_gate_norm = GateAddNorm(d_model, dropout=dropout, trainable_add=False)
        self.pos_wise_ff = GatedResidualNetwork(input_size=d_model, hidden_size=d_model, output_size=d_model, dropout=dropout)
        self.network_structure = network_structure
        if network_structure.skip_connection:
            if network_structure.skip_connection_type == 'add':
                self.residual_connection = AddLayer(d_model, n_encoder_output)
            elif network_structure.skip_connection_type == 'gate_add_norm':
                self.residual_connection = GateAddNorm(d_model, skip_size=n_encoder_output, dropout=None, trainable_add=False)
        self._device = 'cpu'

    def forward(self, encoder_output: torch.Tensor, decoder_output: torch.Tensor, past_observed_targets: torch.BoolTensor, decoder_length: int, static_embedding: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            encoder_output (torch.Tensor):
                the output of the last layer of encoder network
            decoder_output (torch.Tensor):
                the output of the last layer of decoder network
            past_observed_targets (torch.BoolTensor):
                observed values in the past
            decoder_length (int):
                length of decoder network
            static_embedding Optional[torch.Tensor]:
                embeddings of static features  (if available)
        """
        if self.decoder_proj_layer is not None:
            decoder_output = self.decoder_proj_layer(decoder_output)
        network_output = torch.cat([encoder_output, decoder_output], dim=1)
        if self.enrich_with_static and static_embedding is not None:
            static_context_enrichment = self.static_context_enrichment(static_embedding)
            attn_input = self.enrichment(network_output, static_context_enrichment[:, None].expand(-1, network_output.shape[1], -1))
        else:
            attn_input = self.enrichment(network_output)
        encoder_out_length = encoder_output.shape[1]
        past_observed_targets = past_observed_targets[:, -encoder_out_length:]
        past_observed_targets = past_observed_targets
        mask = self.get_attention_mask(past_observed_targets=past_observed_targets, decoder_length=decoder_length)
        if mask.shape[-1] < attn_input.shape[1]:
            mask = torch.cat([mask.new_full((*mask.shape[:-1], attn_input.shape[1] - mask.shape[-1]), True), mask], dim=-1)
        attn_output, attn_output_weights = self.attention_fusion(q=attn_input[:, -decoder_length:], k=attn_input, v=attn_input, mask=mask)
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, -decoder_length:])
        output = self.pos_wise_ff(attn_output)
        if self.network_structure.skip_connection:
            return self.residual_connection(output, decoder_output)
        else:
            return output

    @property
    def device(self) ->torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) ->None:
        self
        self._device = device

    def get_attention_mask(self, past_observed_targets: torch.BoolTensor, decoder_length: int) ->torch.Tensor:
        """
        https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/
        temporal_fusion_transformer/__init__.py
        """
        attend_step = torch.arange(decoder_length, device=self.device)
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        decoder_mask = attend_step >= predict_step
        encoder_mask = ~past_observed_targets.squeeze(-1)
        mask = torch.cat((encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1), decoder_mask.unsqueeze(0).expand(encoder_mask.size(0), -1, -1)), dim=2)
        return mask


class TransformedDistribution_(TransformedDistribution):
    """
    We implement the mean function such that we do not need to enquire base mean every time
    """

    @property
    def mean(self) ->torch.Tensor:
        mean = self.base_dist.mean
        for transform in self.transforms:
            mean = transform(mean)
        return mean


class VariableSelector(nn.Module):

    def __init__(self, network_structure: NetworkStructure, dataset_properties: Dict[str, Any], network_encoder: Dict[str, EncoderBlockInfo], auto_regressive: bool=False, feature_names: Union[Tuple[str], Tuple[()]]=(), known_future_features: Union[Tuple[str], Tuple[()]]=(), feature_shapes: Dict[str, int]={}, static_features: Union[Tuple[Union[str, int]], Tuple[()]]=(), time_feature_names: Union[Tuple[str], Tuple[()]]=()):
        """
        Variable Selector. This models follows the implementation from
        pytorch_forecasting.models.temporal_fusion_transformer.sub_modules.VariableSelectionNetwork
        However, we adjust the structure to fit the data extracted from our dataloader: we record the feature index from
        each feature names and break the input features on the fly.

        The order of the input variables is as follows:
        [features (from the dataset), time_features (from time feature transformers), targets]
        Args:
            network_structure (NetworkStructure):
                contains the information of the overall architecture information
            dataset_properties (Dict):
                dataset properties
            network_encoder(Dict[str, EncoderBlockInfo]):
                Network encoders
            auto_regressive (bool):
                if it belongs to an auto-regressive model
            feature_names (Tuple[str]):
                feature names, used to construct the selection network
            known_future_features (Tuple[str]):
                known future features
            feature_shapes (Dict[str, int]):
                shapes of each features
            time_feature_names (Tuple[str]):
                time feature names, used to complement feature_shapes
        """
        super().__init__()
        first_encoder_output_shape = network_encoder['block_1'].encoder_output_shape[-1]
        self.hidden_size = first_encoder_output_shape
        assert set(feature_names) == set(feature_shapes.keys()), f'feature_names and feature_shapes must have the same variable names but they are differentat {set(feature_names) ^ set(feature_shapes.keys())}'
        pre_scalar = {'past_targets': nn.Linear(dataset_properties['output_shape'][-1], self.hidden_size)}
        encoder_input_sizes = {'past_targets': self.hidden_size}
        decoder_input_sizes = {}
        future_feature_name2tensor_idx = {}
        feature_names2tensor_idx = {}
        idx_tracker = 0
        idx_tracker_future = 0
        static_features = set(static_features)
        static_features_input_size = {}
        known_future_features = tuple(known_future_features)
        feature_names = tuple(feature_names)
        time_feature_names = tuple(time_feature_names)
        if feature_names:
            for name in feature_names:
                feature_shape = feature_shapes[name]
                feature_names2tensor_idx[name] = [idx_tracker, idx_tracker + feature_shape]
                idx_tracker += feature_shape
                pre_scalar[name] = nn.Linear(feature_shape, self.hidden_size)
                if name in static_features:
                    static_features_input_size[name] = self.hidden_size
                else:
                    encoder_input_sizes[name] = self.hidden_size
                    if name in known_future_features:
                        decoder_input_sizes[name] = self.hidden_size
        for future_name in known_future_features:
            feature_shape = feature_shapes[future_name]
            future_feature_name2tensor_idx[future_name] = [idx_tracker_future, idx_tracker_future + feature_shape]
            idx_tracker_future += feature_shape
        if time_feature_names:
            for name in time_feature_names:
                feature_names2tensor_idx[name] = [idx_tracker, idx_tracker + 1]
                future_feature_name2tensor_idx[name] = [idx_tracker_future, idx_tracker_future + 1]
                idx_tracker += 1
                idx_tracker_future += 1
                pre_scalar[name] = nn.Linear(1, self.hidden_size)
                encoder_input_sizes[name] = self.hidden_size
                decoder_input_sizes[name] = self.hidden_size
        if not feature_names or not known_future_features:
            placeholder_features = 'placeholder_features'
            i = 0
            self.placeholder_features: List[str] = []
            while placeholder_features in feature_names or placeholder_features in self.placeholder_features:
                i += 1
                placeholder_features = f'placeholder_features_{i}'
                if i == 5000:
                    raise RuntimeError('Cannot assign name to placeholder features, please considering rename your features')
            name = placeholder_features
            pre_scalar[name] = nn.Linear(1, self.hidden_size)
            encoder_input_sizes[name] = self.hidden_size
            decoder_input_sizes[name] = self.hidden_size
            self.placeholder_features.append(placeholder_features)
        feature_names = time_feature_names + feature_names
        known_future_features = time_feature_names + known_future_features
        self.feature_names = feature_names
        self.feature_names2tensor_idx = feature_names2tensor_idx
        self.future_feature_name2tensor_idx = future_feature_name2tensor_idx
        self.known_future_features = known_future_features
        if auto_regressive:
            pre_scalar.update({'future_prediction': nn.Linear(dataset_properties['output_shape'][-1], self.hidden_size)})
            decoder_input_sizes.update({'future_prediction': self.hidden_size})
        self.pre_scalars = nn.ModuleDict(pre_scalar)
        self._device = torch.device('cpu')
        if not dataset_properties['uni_variant']:
            self.static_variable_selection = VariableSelectionNetwork(input_sizes=static_features_input_size, hidden_size=self.hidden_size, input_embedding_flags={}, dropout=network_structure.grn_dropout_rate, prescalers=self.pre_scalars)
        self.static_input_sizes = static_features_input_size
        self.static_features = static_features
        self.auto_regressive = auto_regressive
        if network_structure.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(input_size, min(input_size, self.hidden_size), self.hidden_size, network_structure.grn_dropout_rate)
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(input_size, min(input_size, self.hidden_size), self.hidden_size, network_structure.grn_dropout_rate)
        self.encoder_variable_selection = VariableSelectionNetwork(input_sizes=encoder_input_sizes, hidden_size=self.hidden_size, input_embedding_flags={}, dropout=network_structure.grn_dropout_rate, context_size=self.hidden_size, single_variable_grns={} if not network_structure.share_single_variable_networks else self.shared_single_variable_grns, prescalers=self.pre_scalars)
        self.decoder_variable_selection = VariableSelectionNetwork(input_sizes=decoder_input_sizes, hidden_size=self.hidden_size, input_embedding_flags={}, dropout=network_structure.grn_dropout_rate, context_size=self.hidden_size, single_variable_grns={} if not network_structure.share_single_variable_networks else self.shared_single_variable_grns, prescalers=self.pre_scalars)
        self.static_context_variable_selection = GatedResidualNetwork(input_size=self.hidden_size, hidden_size=self.hidden_size, output_size=self.hidden_size, dropout=network_structure.grn_dropout_rate)
        n_hidden_states = 0
        if network_encoder['block_1'].encoder_properties.has_hidden_states:
            n_hidden_states = network_encoder['block_1'].n_hidden_states
        static_context_initial_hidden = [GatedResidualNetwork(input_size=self.hidden_size, hidden_size=self.hidden_size, output_size=self.hidden_size, dropout=network_structure.grn_dropout_rate) for _ in range(n_hidden_states)]
        self.static_context_initial_hidden = nn.ModuleList(static_context_initial_hidden)
        self.cached_static_contex: Optional[torch.Tensor] = None
        self.cached_static_embedding: Optional[torch.Tensor] = None

    @property
    def device(self) ->torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) ->None:
        self
        self._device = device

    def forward(self, x_past: Optional[Dict[str, torch.Tensor]], x_future: Optional[Dict[str, torch.Tensor]], x_static: Optional[Dict[str, torch.Tensor]], length_past: int=0, length_future: int=0, batch_size: int=0, cache_static_contex: bool=False, use_cached_static_contex: bool=False) ->Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if x_past is None and x_future is None:
            raise ValueError('Either past input or future inputs need to be given!')
        if length_past == 0 and length_future == 0:
            raise ValueError('Either length_past or length_future must be given!')
        timesteps = length_past + length_future
        if not use_cached_static_contex:
            if len(self.static_input_sizes) > 0:
                static_embedding, _ = self.static_variable_selection(x_static)
            else:
                if length_past > 0:
                    assert x_past is not None, 'x_past must be given when length_past is greater than 0!'
                    model_dtype = next(iter(x_past.values())).dtype
                else:
                    assert x_future is not None, 'x_future must be given when length_future is greater than 0!'
                    model_dtype = next(iter(x_future.values())).dtype
                static_embedding = torch.zeros((batch_size, self.hidden_size), dtype=model_dtype, device=self.device)
            static_context_variable_selection = self.static_context_variable_selection(static_embedding)[:, None]
            static_context_initial_hidden: Optional[Tuple[torch.Tensor, ...]] = tuple(init_hidden(static_embedding) for init_hidden in self.static_context_initial_hidden)
            if cache_static_contex:
                self.cached_static_contex = static_context_variable_selection
                self.cached_static_embedding = static_embedding
        else:
            static_embedding = self.cached_static_embedding
            static_context_initial_hidden = None
            static_context_variable_selection = self.cached_static_contex
        static_context_variable_selection = static_context_variable_selection.expand(-1, timesteps, -1)
        if x_past is not None:
            embeddings_varying_encoder, _ = self.encoder_variable_selection(x_past, static_context_variable_selection[:, :length_past])
        else:
            embeddings_varying_encoder = None
        if x_future is not None:
            embeddings_varying_decoder, _ = self.decoder_variable_selection(x_future, static_context_variable_selection[:, length_past:])
        else:
            embeddings_varying_decoder = None
        return embeddings_varying_encoder, embeddings_varying_decoder, static_embedding, static_context_initial_hidden


class _NoEmbedding(nn.Module):

    def get_partial_models(self, subset_features: List[int]) ->'_NoEmbedding':
        return self

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x


def get_lagged_subsequences(sequence: torch.Tensor, subsequences_length: int, lags_seq: Optional[List[int]]=None, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns lagged subsequences of a given sequence, this allows the model to receive the input from the past targets
    outside the sliding windows. This implementation is similar to gluonTS's implementation
     the only difference is that we pad the sequence that is not long enough

    Args:
        sequence (torch.Tensor):
            the sequence from which lagged subsequences should be extracted, Shape: (N, T, C).
        subsequences_length (int):
            length of the subsequences to be extracted.
        lags_seq (Optional[List[int]]):
            lags of the sequence, indicating the sequence that needs to be extracted
        mask (Optional[torch.Tensor]):
            a mask tensor indicating, it is a cached mask tensor that allows the model to quickly extract the desired
            lagged values

    Returns:
        lagged (Tensor)
            A tensor of shape (N, S, I * C), where S = subsequences_length and I = len(indices),
             containing lagged subsequences.
        mask (torch.Tensor):
            cached mask
    """
    batch_size = sequence.shape[0]
    num_features = sequence.shape[2]
    if mask is None:
        if lags_seq is None:
            warnings.warn('Neither lag_mask or lags_seq is given, we simply return the input value')
            return sequence, None
        num_lags = len(lags_seq)
        mask_length = max(lags_seq) + subsequences_length
        mask = torch.zeros((num_lags, mask_length), dtype=torch.bool)
        for i, lag_index in enumerate(lags_seq):
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            mask[i, begin_index:end_index] = True
    else:
        num_lags = mask.shape[0]
        mask_length = mask.shape[1]
    mask_extend = mask.clone()
    if mask_length > sequence.shape[1]:
        sequence = torch.cat([sequence.new_zeros([batch_size, mask_length - sequence.shape[1], num_features]), sequence], dim=1)
    elif mask_length < sequence.shape[1]:
        mask_extend = torch.cat([mask.new_zeros([num_lags, sequence.shape[1] - mask_length]), mask_extend], dim=1)
    sequence = sequence.unsqueeze(1)
    mask_extend = mask_extend.unsqueeze(-1)
    lagged_seq = torch.masked_select(sequence, mask_extend).reshape(batch_size, num_lags, subsequences_length, -1)
    lagged_seq = torch.transpose(lagged_seq, 1, 2).reshape(batch_size, subsequences_length, -1)
    return lagged_seq, mask


def get_lagged_subsequences_inference(sequence: torch.Tensor, subsequences_length: int, lags_seq: List[int]) ->torch.Tensor:
    """
    this function works exactly the same as get_lagged_subsequences. However, this implementation is faster when no
    cached value is available, thus it is applied during inference times.

    Args:
        sequence (torch.Tensor):
            the sequence from which lagged subsequences should be extracted, Shape: (N, T, C).
        subsequences_length (int):
            length of the subsequences to be extracted.
        lags_seq (Optional[List[int]]):
            lags of the sequence, indicating the sequence that needs to be extracted

    Returns:
        lagged (Tensor)
            A tensor of shape (N, S, I * C), where S = subsequences_length and I = len(indices),
             containing lagged subsequences.
    """
    sequence_length = sequence.shape[1]
    batch_size = sequence.shape[0]
    lagged_values = []
    for lag_index in lags_seq:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        if end_index is not None and end_index < -sequence_length:
            lagged_values.append(torch.zeros([batch_size, subsequences_length, *sequence.shape[2:]]))
            continue
        if begin_index < -sequence_length:
            if end_index is not None:
                pad_shape = [batch_size, subsequences_length - sequence_length - end_index, *sequence.shape[2:]]
                lagged_values.append(torch.cat([torch.zeros(pad_shape), sequence[:, :end_index, ...]], dim=1))
            else:
                pad_shape = [batch_size, subsequences_length - sequence_length, *sequence.shape[2:]]
                lagged_values.append(torch.cat([torch.zeros(pad_shape), sequence], dim=1))
            continue
        else:
            lagged_values.append(sequence[:, begin_index:end_index, ...])
    lagged_seq = torch.stack(lagged_values, -1).transpose(-1, -2).reshape(batch_size, subsequences_length, -1)
    return lagged_seq


_activations = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid}


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features: int, activation: str, growth_rate: int, bn_size: int, drop_rate: float, bn_args: Dict[str, Any]):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, **bn_args)),
        self.add_module('relu1', _activations[activation]()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate, **bn_args)),
        self.add_module('relu2', _activations[activation]()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers: int, num_input_features: int, activation: str, bn_size: int, growth_rate: int, drop_rate: float, bn_args: Dict[str, Any]):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features=num_input_features + i * growth_rate, activation=activation, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, bn_args=bn_args)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features: int, activation: str, num_output_features: int, pool_size: int, bn_args: Dict[str, Any]):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, **bn_args))
        self.add_module('relu', _activations[activation]())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))


class ShakeDropFunction(Function):
    """
    References:
        Title: ShakeDrop Regularization for Deep Residual Learning
        Authors: Yoshihiro Yamada et. al.
        URL: https://arxiv.org/pdf/1802.02375.pdf
        Title: ShakeDrop Regularization
        Authors: Yoshihiro Yamada et. al.
        URL: https://openreview.net/pdf?id=S1NHaMW0b
        Github URL: https://github.com/owruby/shake-drop_pytorch/blob/master/models/shakedrop.py
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, bl: torch.Tensor) ->torch.Tensor:
        ctx.save_for_backward(x, alpha, beta, bl)
        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, alpha, beta, bl = ctx.saved_tensors
        grad_x = grad_alpha = grad_beta = grad_bl = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)
        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_drop_get_bl(block_index: int, min_prob_no_shake: float, num_blocks: int, is_training: bool, is_cuda: bool) ->torch.Tensor:
    """
    The sampling of Bernoulli random variable
    based on Eq. (4) in the paper

    Args:
        block_index (int): The index of the block from the input layer
        min_prob_no_shake (float): The initial shake probability
        num_blocks (int): The total number of building blocks
        is_training (bool): Whether it is training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        bl (torch.Tensor): a Bernoulli random variable in {0, 1}

    Reference:
        ShakeDrop Regularization for Deep Residual Learning
        Yoshihiro Yamada et. al. (2020)
        paper: https://arxiv.org/pdf/1802.02375.pdf
        implementation: https://github.com/imenurok/ShakeDrop
    """
    pl = 1 - (block_index + 1) / num_blocks * (1 - min_prob_no_shake)
    if is_training:
        bl = torch.as_tensor(1.0) if torch.rand(1) <= pl else torch.as_tensor(0.0)
    else:
        bl = torch.as_tensor(pl)
    if is_cuda:
        bl = bl
    return bl


def shake_get_alpha_beta(is_training: bool, is_cuda: bool) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    The methods used in this function have been introduced in 'ShakeShake Regularisation'
    Currently, this function supports `shake-shake`.

    Args:
        is_training (bool): Whether the computation for the training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        alpha, beta (Tuple[float, float]):
            alpha (in [0, 1]) is the weight coefficient  for the forward pass
            beta (in [0, 1]) is the weight coefficient for the backward pass

    Reference:
        Title: Shake-shake regularization
        Author: Xavier Gastaldi
        URL: https://arxiv.org/abs/1705.07485

    Note:
        The names have been taken from the paper as well.
        Currently, this function supports `shake-shake`.
    """
    if not is_training:
        result = torch.FloatTensor([0.5]), torch.FloatTensor([0.5])
        return result if not is_cuda else (result[0], result[1])
    alpha = torch.rand(1)
    beta = torch.rand(1)
    if is_cuda:
        alpha = alpha
        beta = beta
    return alpha, beta


class ShakeShakeFunction(Function):
    """
    References:
        Title: Shake-Shake regularization
        Authors: Xavier Gastaldi
        URL: https://arxiv.org/pdf/1705.07485.pdf
        Github URL: https://github.com/hysts/pytorch_shake_shake/blob/master/functions/shake_shake_function.py
    """

    @staticmethod
    def forward(ctx: Any, x1: torch.Tensor, x2: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) ->torch.Tensor:
        ctx.save_for_backward(x1, x2, alpha, beta)
        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, alpha, beta = ctx.saved_tensors
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)
        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_shake = ShakeShakeFunction.apply


class ResBlock(nn.Module):
    """
    __author__ = "Max Dippel, Michael Burkart and Matthias Urban"
    """

    def __init__(self, config: Dict[str, Any], in_features: int, out_features: int, blocks_per_group: int, block_index: int, dropout: Optional[float], activation: nn.Module):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation
        self.shortcut = None
        self.start_norm: Optional[Callable] = None
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            self.start_norm = nn.Sequential(nn.BatchNorm1d(in_features), self.activation())
        self.block_index = block_index
        self.num_blocks = blocks_per_group * self.config['num_groups']
        self.layers = self._build_block(in_features, out_features)
        if config['use_shake_shake']:
            self.shake_shake_layers = self._build_block(in_features, out_features)

    def _build_block(self, in_features: int, out_features: int) ->nn.Module:
        layers = list()
        if self.start_norm is None:
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) ->torch.FloatTensor:
        residual = x
        if self.shortcut is not None and self.start_norm is not None:
            x = self.start_norm(x)
            residual = self.shortcut(x)
        if self.config['use_shake_shake']:
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            x = shake_shake(x1, x2, alpha, beta)
        else:
            x = self.layers(x)
        if self.config['use_shake_drop']:
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            bl = shake_drop_get_bl(self.block_index, 1 - self.config['max_shake_drop_probability'], self.num_blocks, self.training, x.is_cuda)
            x = shake_drop(x, alpha, beta, bl)
        x = x + residual
        return x


class PositionalEncoding(nn.Module):
    """https://github.com/pytorch/examples/blob/master/word_language_model/model.py

        NOTE: different from the raw implementation, this model is designed for the batch_first inputs!
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model (int):
            the embed dim (required).
        dropout(float):
            the dropout value (default=0.1).
        max_len(int):
            the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, pos_idx: Optional[Tuple[int]]=None) ->torch.Tensor:
        """Inputs of forward function
        Args:
            x (torch.Tensor(B, L, N)):
                the sequence fed to the positional encoder model (required).
            pos_idx (Tuple[int]):
                position idx indicating the start (first) and end (last) time index of x in a sequence

        Examples:
            >>> output = pos_encoder(x)
        """
        if pos_idx is None:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, pos_idx[0]:pos_idx[1], :]
        return self.dropout(x)


class DecoderNetwork(nn.Module):

    def forward(self, x_future: torch.Tensor, encoder_output: torch.Tensor, pos_idx: Optional[Tuple[int]]=None) ->torch.Tensor:
        """
        Base forecasting Decoder Network, its output needs to be a 3-d Tensor:


        Args:
            x_future: torch.Tensor(B, L_future, N_out), the future features
            encoder_output: torch.Tensor(B, L_encoder, N), output of the encoder network, or the hidden states
            pos_idx: positional index, indicating the position of the forecasted tensor, used for transformer
        Returns:
            net_output: torch.Tensor with shape either (B, L_future, N)

        """
        raise NotImplementedError


class EncoderNetwork(nn.Module):

    def forward(self, x: torch.Tensor, output_seq: bool=False) ->torch.Tensor:
        """
        Base forecasting network, its output needs to be a 2-d or 3-d Tensor:
        When the decoder is an auto-regressive model, then it needs to output a 3-d Tensor, in which case, output_seq
         needs to be set as True
        When the decoder is a seq2seq model, the network needs to output a 2-d Tensor (B, N), in which case,
        output_seq needs to be set as False

        Args:
            x: torch.Tensor(B, L_in, N)
                input data
            output_seq (bool): if the network outputs a sequence tensor. If it is set True,
                output will be a 3-d Tensor (B, L_out, N). L_out = L_in if encoder_properties['recurrent'] is True.
                If this value is set as False, the network only returns the last item of the sequence.
            hx (Optional[torch.Tensor]): addational input to the network, this could be a hidden states or a sequence
                from previous inputs

        Returns:
            net_output: torch.Tensor with shape either (B, N) or (B, L_out, N)

        """
        raise NotImplementedError

    def get_last_seq_value(self, x: torch.Tensor) ->torch.Tensor:
        """
        get the last value of the sequential output
        Args:
            x (torch.Tensor(B, L, N)):
                a sequential value output by the network, usually this value needs to be fed to the decoder
                (or a 2D tensor for a flat encoder)
        Returns:
            output (torch.Tensor(B, 1, M)):
                last element of the sequential value (or a 2D tensor for flat encoder)

        """
        raise NotImplementedError


class TimeSeriesMLP(EncoderNetwork):

    def __init__(self, window_size: int, network: Optional[nn.Module]=None):
        """
        Transform the input features (B, T, N) to fit the requirement of MLP
        Args:
            window_size (int): T
            fill_lower_resolution_seq: if sequence with lower resolution needs to be filled with 0
        (for multi-fidelity problems with resolution as fidelity)
        """
        super().__init__()
        self.window_size = window_size
        self.network = network

    def forward(self, x: torch.Tensor, output_seq: bool=False) ->torch.Tensor:
        """

        Args:
            x: torch.Tensor(B, L_in, N)
            output_seq (bool), if the MLP outputs a squence, in which case, the input will be rolled to fit the size of
            the network. For Instance if self.window_size = 3, and we obtain a squence with [1, 2, 3, 4, 5]
            the input of this mlp is rolled as :
            [[1, 2, 3]
            [2, 3, 4]
            [3, 4 ,5]]

        Returns:

        """
        if output_seq:
            x = x.unfold(1, self.window_size, 1).transpose(-1, -2)
        elif x.shape[1] > self.window_size:
            x = x[:, -self.window_size:]
        x = x.flatten(-2)
        return x if self.network is None else self.network(x)

    def get_last_seq_value(self, x: torch.Tensor) ->torch.Tensor:
        return x


class _InceptionBlock(nn.Module):

    def __init__(self, n_inputs: int, n_filters: int, kernel_size: int, bottleneck: int=None):
        super(_InceptionBlock, self).__init__()
        self.n_filters = n_filters
        self.bottleneck = None if bottleneck is None else nn.Conv1d(n_inputs, bottleneck, kernel_size=1)
        kernel_sizes = [(kernel_size // 2 ** i) for i in range(3)]
        n_inputs = n_inputs if bottleneck is None else bottleneck
        self.pad1 = nn.ConstantPad1d(padding=self._padding(kernel_sizes[0]), value=0)
        self.conv1 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[0])
        self.pad2 = nn.ConstantPad1d(padding=self._padding(kernel_sizes[1]), value=0)
        self.conv2 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[1])
        self.pad3 = nn.ConstantPad1d(padding=self._padding(kernel_sizes[2]), value=0)
        self.conv3 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[2])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv1d(n_inputs, n_filters, 1)
        self.bn = nn.BatchNorm1d(4 * n_filters)

    def _padding(self, kernel_size: int) ->Tuple[int, int]:
        if kernel_size % 2 == 0:
            return kernel_size // 2, kernel_size // 2 - 1
        else:
            return kernel_size // 2, kernel_size // 2

    def get_n_outputs(self) ->int:
        return 4 * self.n_filters

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x1 = self.conv1(self.pad1(x))
        x2 = self.conv2(self.pad2(x))
        x3 = self.conv3(self.pad3(x))
        x4 = self.convpool(self.maxpool(x))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        return torch.relu(x)


class _ResidualBlock(nn.Module):

    def __init__(self, n_res_inputs: int, n_outputs: int):
        super(_ResidualBlock, self).__init__()
        self.shortcut = nn.Conv1d(n_res_inputs, n_outputs, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_outputs)

    def forward(self, x: torch.Tensor, res: torch.Tensor) ->torch.Tensor:
        shortcut = self.shortcut(res)
        shortcut = self.bn(shortcut)
        x = x + shortcut
        return torch.relu(x)


class _InceptionTime(nn.Module):

    def __init__(self, in_features: int, config: Dict[str, Any]) ->None:
        super().__init__()
        self.config = config
        n_inputs = in_features
        n_filters = self.config['num_filters']
        bottleneck_size = self.config['bottleneck_size']
        kernel_size = self.config['kernel_size']
        n_res_inputs = in_features
        receptive_field = 1
        for i in range(self.config['num_blocks']):
            block = _InceptionBlock(n_inputs=n_inputs, n_filters=n_filters, bottleneck=bottleneck_size, kernel_size=kernel_size)
            receptive_field += max(kernel_size, 3) - 1
            self.__setattr__(f'inception_block_{i}', block)
            if i % 3 == 2:
                n_res_outputs = block.get_n_outputs()
                self.__setattr__(f'residual_block_{i}', _ResidualBlock(n_res_inputs=n_res_inputs, n_outputs=n_res_outputs))
                n_res_inputs = n_res_outputs
            n_inputs = block.get_n_outputs()
        self.receptive_field = receptive_field

    def forward(self, x: torch.Tensor, output_seq: bool=False) ->torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        res = x
        for i in range(self.config['num_blocks']):
            x = self.__getattr__(f'inception_block_{i}')(x)
            if i % 3 == 2:
                x = self.__getattr__(f'residual_block_{i}')(x, res)
                res = x
        x = x.transpose(1, 2).contiguous()
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: torch.Tensor) ->torch.Tensor:
        return x[:, -1:, :]


class _RNN(EncoderNetwork):

    def __init__(self, in_features: int, config: Dict[str, Any], lagged_value: Optional[List[int]]=None):
        super().__init__()
        if lagged_value is None:
            self.lagged_value = [0]
        else:
            self.lagged_value = lagged_value
        self.config = config
        if config['cell_type'] == 'lstm':
            cell_type = nn.LSTM
        else:
            cell_type = nn.GRU
        self.lstm = cell_type(input_size=in_features, hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config.get('dropout', 0.0), bidirectional=config['bidirectional'], batch_first=True)
        self.cell_type = config['cell_type']

    def forward(self, x: torch.Tensor, output_seq: bool=False, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) ->Tuple[torch.Tensor, ...]:
        B, T, _ = x.shape
        x, hidden_state = self.lstm(x, hx)
        if output_seq:
            return x, hidden_state
        else:
            return self.get_last_seq_value(x), hidden_state

    def get_last_seq_value(self, x: torch.Tensor) ->torch.Tensor:
        B, T, _ = x.shape
        if not self.config['bidirectional']:
            return x[:, -1:]
        else:
            x_by_direction = x.view(B, T, 2, self.config['hidden_size'])
            x = torch.cat([x_by_direction[:, -1, [0], :], x_by_direction[:, 0, [1], :]], dim=-1)
            return x


class _Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float=0.2):
        super(_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_weights(self) ->None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalConvNet(EncoderNetwork):

    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: List[int], dropout: float=0.2):
        super(_TemporalConvNet, self).__init__()
        layers: List[Any] = []
        num_levels = len(num_channels)
        receptive_field = 1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            stride = 1
            layers += [_TemporalBlock(in_channels, out_channels, kernel_size[i], stride=stride, dilation=dilation_size, padding=(kernel_size[i] - 1) * dilation_size, dropout=dropout)]
            receptive_field_block = 1 + 2 * (kernel_size[i] - 1) * dilation_size
            receptive_field += receptive_field_block
        self.receptive_field = receptive_field
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, output_seq: bool=False) ->torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: torch.Tensor) ->torch.Tensor:
        return x[:, -1:]


class _TransformerEncoder(EncoderNetwork):

    def __init__(self, in_features: int, d_model: int, num_layers: int, transformer_encoder_layers: nn.Module, use_positional_encoder: bool, use_layer_norm_output: bool, dropout_pe: float=0.0, layer_norm_eps_output: Optional[float]=None, lagged_value: Optional[List[int]]=None):
        super().__init__()
        if lagged_value is None:
            self.lagged_value = [0]
        else:
            self.lagged_value = lagged_value
        if in_features != d_model:
            input_layer = [nn.Linear(in_features, d_model, bias=False)]
        else:
            input_layer = []
        if use_positional_encoder:
            input_layer.append(PositionalEncoding(d_model, dropout_pe))
        self.input_layer = nn.Sequential(*input_layer)
        self.use_layer_norm_output = use_layer_norm_output
        if use_layer_norm_output:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps_output)
        else:
            norm = None
        self.transformer_encoder_layers = nn.TransformerEncoder(encoder_layer=transformer_encoder_layers, num_layers=num_layers, norm=norm)

    def forward(self, x: torch.Tensor, output_seq: bool=False, mask: Optional[torch.Tensor]=None, src_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        x = self.input_layer(x)
        x = self.transformer_encoder_layers(x)
        if output_seq:
            return x
        else:
            return self.get_last_seq_value(x)

    def get_last_seq_value(self, x: torch.Tensor) ->torch.Tensor:
        return x[:, -1:]


class _LearnedEntityEmbedding(nn.Module):
    """ Learned entity embedding module for categorical features"""

    def __init__(self, config: Dict[str, Any], num_input_features: np.ndarray, num_numerical_features: int):
        """
        Args:
            config (Dict[str, Any]): The configuration sampled by the hyperparameter optimizer
            num_input_features (np.ndarray): column wise information of number of output columns after transformation
                for each categorical column and 0 for numerical columns
            num_numerical_features (int): number of numerical features in X
        """
        super().__init__()
        self.config = config
        self.num_numerical = num_numerical_features
        self.num_input_features = num_input_features
        categorical_features: np.ndarray = self.num_input_features > 0
        self.num_categorical_features = self.num_input_features[categorical_features]
        self.embed_features = [(num_in >= config['min_unique_values_for_embedding']) for num_in in self.num_input_features]
        self.num_output_dimensions = [0] * num_numerical_features
        self.num_output_dimensions.extend([(config['dimension_reduction_' + str(i)] * num_in) for i, num_in in enumerate(self.num_categorical_features)])
        self.num_output_dimensions = [int(np.clip(num_out, 1, num_in - 1)) for num_out, num_in in zip(self.num_output_dimensions, self.num_input_features)]
        self.num_output_dimensions = [(num_out if embed else num_in) for num_out, embed, num_in in zip(self.num_output_dimensions, self.embed_features, self.num_input_features)]
        self.num_out_feats = self.num_numerical + sum(self.num_output_dimensions)
        self.ee_layers = self._create_ee_layers()

    def get_partial_models(self, subset_features: List[int]) ->'_LearnedEntityEmbedding':
        """
        extract a partial models that only works on a subset of the data that ought to be passed to the embedding
        network, this function is implemented for time series forecasting tasks where the known future features is only
        a subset of the past features
        Args:
            subset_features (List[int]):
                a set of index identifying which features will pass through the partial model

        Returns:
            partial_model (_LearnedEntityEmbedding)
                a new partial model
        """
        num_input_features = self.num_input_features[subset_features]
        num_numerical_features = sum([(sf < self.num_numerical) for sf in subset_features])
        num_output_dimensions = [self.num_output_dimensions[sf] for sf in subset_features]
        embed_features = [self.embed_features[sf] for sf in subset_features]
        ee_layers = []
        ee_layer_tracker = 0
        for sf in subset_features:
            if self.embed_features[sf]:
                ee_layers.append(self.ee_layers[ee_layer_tracker])
                ee_layer_tracker += 1
        ee_layers = nn.ModuleList(ee_layers)
        return PartialLearnedEntityEmbedding(num_input_features, num_numerical_features, embed_features, num_output_dimensions, ee_layers)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        concat_seq = []
        last_concat = 0
        x_pointer = 0
        layer_pointer = 0
        for num_in, embed in zip(self.num_input_features, self.embed_features):
            if not embed:
                x_pointer += 1
                continue
            if x_pointer > last_concat:
                concat_seq.append(x[..., last_concat:x_pointer])
            categorical_feature_slice = x[..., x_pointer:x_pointer + num_in]
            concat_seq.append(self.ee_layers[layer_pointer](categorical_feature_slice))
            layer_pointer += 1
            x_pointer += num_in
            last_concat = x_pointer
        concat_seq.append(x[..., last_concat:])
        return torch.cat(concat_seq, dim=-1)

    def _create_ee_layers(self) ->nn.ModuleList:
        layers = nn.ModuleList()
        for i, (num_in, embed, num_out) in enumerate(zip(self.num_input_features, self.embed_features, self.num_output_dimensions)):
            if not embed:
                continue
            layers.append(nn.Linear(num_in, num_out))
        return layers


class TransposeLinear(nn.Module):

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x.mm(self.weights)


class ProjectionLayer(nn.Module):
    """
    A projection layer that project features to a torch distribution
    """
    value_in_support = 0.0

    def __init__(self, num_in_features: int, output_shape: Tuple[int, ...], n_prediction_heads: int, decoder_has_local_layer: bool, **kwargs: Any):
        super().__init__(**kwargs)

        def build_single_proj_layer(arg_dim: int) ->nn.Module:
            """
            build a single proj layer given the input dims, the output is unflattened to fit the required output_shape
            and n_prediction_steps.
            we note that output_shape's first dimensions is always n_prediction_steps
            Args:
                arg_dim (int):
                    dimension of the target distribution

            Returns:
                proj_layer (nn.Module):
                    projection layer that maps the decoder output to parameterize distributions
            """
            if decoder_has_local_layer:
                return nn.Sequential(nn.Linear(num_in_features, np.prod(output_shape).item() * arg_dim), nn.Unflatten(-1, (*output_shape, arg_dim)))
            else:
                return nn.Sequential(nn.Linear(num_in_features, n_prediction_heads * np.prod(output_shape).item() * arg_dim), nn.Unflatten(-1, (n_prediction_heads, *output_shape, arg_dim)))
        self.proj = nn.ModuleList([build_single_proj_layer(dim) for dim in self.arg_dims.values()])

    def forward(self, x: torch.Tensor) ->torch.distributions:
        """
        get a target distribution
        Args:
            x: input tensor ([batch_size, in_features]):
                input tensor, acquired by the base header, have the shape [batch_size, in_features]

        Returns:
            dist: torch.distributions ([batch_size, n_prediction_steps, output_shape]):
                an output torch distribution with shape (batch_size, n_prediction_steps, output_shape)
        """
        params_unbounded = [proj(x) for proj in self.proj]
        return self.dist_cls(*self.domain_map(*params_unbounded))

    @property
    @abstractmethod
    def arg_dims(self) ->Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def domain_map(self, *args: torch.Tensor) ->Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dist_cls(self) ->Type[Distribution]:
        raise NotImplementedError


class NormalOutput(ProjectionLayer):

    @property
    def arg_dims(self) ->Dict[str, int]:
        return {'loc': 1, 'scale': 1}

    def domain_map(self, loc: torch.Tensor, scale: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        scale = F.softplus(scale) + 1e-10
        return loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) ->Type[Distribution]:
        return Normal


class StudentTOutput(ProjectionLayer):

    @property
    def arg_dims(self) ->Dict[str, int]:
        return {'df': 1, 'loc': 1, 'scale': 1}

    def domain_map(self, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = F.softplus(scale) + 1e-10
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def dist_cls(self) ->Type[Distribution]:
        return StudentT


class BetaOutput(ProjectionLayer):
    value_in_support = 0.5

    @property
    def arg_dims(self) ->Dict[str, int]:
        return {'concentration1': 1, 'concentration0': 1}

    def domain_map(self, concentration1: torch.Tensor, concentration0: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        epsilon = 1e-10
        concentration1 = F.softplus(concentration1) + epsilon
        concentration0 = F.softplus(concentration0) + epsilon
        return concentration1.squeeze(-1), concentration0.squeeze(-1)

    @property
    def dist_cls(self) ->Type[Distribution]:
        return Beta


class GammaOutput(ProjectionLayer):
    value_in_support = 0.5

    @property
    def arg_dims(self) ->Dict[str, int]:
        return {'concentration': 1, 'rate': 1}

    def domain_map(self, concentration: torch.Tensor, rate: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        epsilon = 1e-10
        concentration = F.softplus(concentration) + epsilon
        rate = F.softplus(rate) + epsilon
        return concentration.squeeze(-1), rate.squeeze(-1)

    @property
    def dist_cls(self) ->Type[Distribution]:
        return Gamma


class PoissonOutput(ProjectionLayer):

    @property
    def arg_dims(self) ->Dict[str, int]:
        return {'rate': 1}

    def domain_map(self, rate: torch.Tensor) ->Tuple[torch.Tensor]:
        rate_pos = F.softplus(rate).clone()
        return rate_pos.squeeze(-1),

    @property
    def dist_cls(self) ->Type[Distribution]:
        return Poisson


class QuantileHead(nn.Module):

    def __init__(self, head_components: List[nn.Module]):
        super().__init__()
        self.net = nn.ModuleList(head_components)

    def forward(self, x: torch.Tensor) ->List[torch.Tensor]:
        return [net(x) for net in self.net]


class _FullyConvolutional2DHead(nn.Module):

    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], pooling_method: str, activation: str, num_layers: int, num_channels: List[int]):
        super().__init__()
        layers = []
        in_channels = input_shape[0]
        for i in range(1, num_layers):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=num_channels[i - 1], kernel_size=1))
            layers.append(_activations[activation]())
            in_channels = num_channels[i - 1]
        out_channels = output_shape[0]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        if pooling_method == 'average':
            layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        else:
            layers.append(nn.AdaptiveMaxPool2d(output_size=1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        B, C, H, W = x.size()
        return self.head(x).view(B, -1)


class AbstractForecastingLoss(Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str='mean') ->None:
        super(AbstractForecastingLoss, self).__init__(reduction=reduction)

    def aggregate_loss(self, loss_values: torch.Tensor) ->torch.Tensor:
        if self.reduction == 'mean':
            return loss_values.mean()
        elif self.reduction == 'sum':
            return loss_values.sum()
        else:
            return loss_values


class LogProbLoss(AbstractForecastingLoss):

    def forward(self, input_dist: torch.distributions.Distribution, target_tensor: torch.Tensor) ->torch.Tensor:
        scores = input_dist.log_prob(target_tensor)
        return self.aggregate_loss(-scores)


class MAPELoss(AbstractForecastingLoss):

    def forward(self, predictions: torch.Tensor, target_tensor: torch.Tensor) ->torch.Tensor:
        denominator = torch.abs(target_tensor)
        diff = torch.abs(predictions - target_tensor)
        flag = (denominator == 0).float()
        mape = diff * (1 - flag) / (denominator + flag)
        return self.aggregate_loss(mape)


class MASELoss(AbstractForecastingLoss):

    def __init__(self, reduction: str='mean') ->None:
        super(MASELoss, self).__init__(reduction=reduction)
        self._mase_coefficient: Union[float, torch.Tensor] = 1.0

    def set_mase_coefficient(self, mase_coefficient: torch.Tensor) ->'MASELoss':
        """
        set mase coefficient for computing MASE losses
        Args:
            mase_coefficient (torch.Tensor): mase coefficient, its dimensions corresponds to [B, L, N] and can be
                broadcasted

        Returns:

        """
        if len(mase_coefficient.shape) == 2:
            mase_coefficient = mase_coefficient.unsqueeze(1)
        self._mase_coefficient = mase_coefficient
        return self

    def forward(self, predictions: torch.Tensor, target_tensor: torch.Tensor) ->torch.Tensor:
        if isinstance(self._mase_coefficient, torch.Tensor):
            mase_shape = self._mase_coefficient.shape
            pred_shape = predictions.shape
            if len(mase_shape) == len(pred_shape):
                if mase_shape[0] != pred_shape[0] or mase_shape[-1] != pred_shape[-1]:
                    raise ValueError(f'If self._mase_coefficient is a Tensor, it must have the same batch size and num_targets as the predictions, However, their shapes are {mase_shape}(self._mase_coefficient) and {pred_shape}(pred_shape)')
        loss_values = torch.abs(predictions - target_tensor) * self._mase_coefficient
        return self.aggregate_loss(loss_values)


class QuantileLoss(AbstractForecastingLoss):

    def __init__(self, reduction: str='mean', quantiles: List[float]=[0.5]) ->None:
        super(QuantileLoss, self).__init__(reduction=reduction)
        self.quantiles = quantiles

    def set_quantiles(self, quantiles: List[float]) ->None:
        self.quantiles = quantiles

    def forward(self, predictions: List[torch.Tensor], target_tensor: torch.Tensor) ->torch.Tensor:
        assert len(self.quantiles) == len(predictions)
        losses_all = []
        for q, y_pred in zip(self.quantiles, predictions):
            diff = target_tensor - y_pred
            loss_q = torch.max(q * diff, (q - 1) * diff)
            losses_all.append(loss_q.unsqueeze(-1))
        losses_all = torch.mean(torch.concat(losses_all, dim=-1), dim=-1)
        return self.aggregate_loss(losses_all)


class DummyEmbedding(torch.nn.Module):

    def forward(self, x):
        if x.shape[-1] > 10:
            return x[..., :-10]
        return x


class DummyEncoderNetwork(EncoderNetwork):

    def forward(self, x, output_seq=False):
        if output_seq:
            return torch.ones((*x.shape[:-1], 10))
        return torch.ones((*x.shape[:-2], 1, 10))


class ReducedEmbedding(torch.nn.Module):

    def __init__(self, num_input_features, num_numerical_features: int):
        super(ReducedEmbedding, self).__init__()
        self.num_input_features = num_input_features
        self.num_numerical_features = num_numerical_features
        self.n_cat_features = len(num_input_features) - num_numerical_features

    def forward(self, x):
        x = x[..., :-self.n_cat_features]
        return x

    def get_partial_models(self, subset_features):
        num_numerical_features = sum([(sf < self.num_numerical_features) for sf in subset_features])
        num_input_features = [self.num_input_features[sf] for sf in subset_features]
        return ReducedEmbedding(num_input_features, num_numerical_features)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddLayer,
     lambda: ([], {'input_size': 4, 'skip_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyEncoderNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MAPELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MASELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantileHead,
     lambda: ([], {'head_components': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReducedEmbedding,
     lambda: ([], {'num_input_features': [4, 4], 'num_numerical_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'config': _mock_config(num_groups=1, use_dropout=0.5, use_shake_shake=4, use_shake_drop=4, max_shake_drop_probability=4), 'in_features': 4, 'out_features': 4, 'blocks_per_group': 4, 'block_index': 4, 'dropout': 0.5, 'activation': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TimeSeriesMLP,
     lambda: ([], {'window_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_FullyConvolutional2DHead,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4], 'pooling_method': 4, 'activation': 4, 'num_layers': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_InceptionBlock,
     lambda: ([], {'n_inputs': 4, 'n_filters': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (_InceptionTime,
     lambda: ([], {'in_features': 4, 'config': _mock_config(num_filters=4, bottleneck_size=4, kernel_size=4, num_blocks=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (_NoEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ResidualBlock,
     lambda: ([], {'n_res_inputs': 4, 'n_outputs': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (_TemporalConvNet,
     lambda: ([], {'num_inputs': 4, 'num_channels': [4, 4], 'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_automl_Auto_PyTorch(_paritybench_base):
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

