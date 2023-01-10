import sys
_module = sys.modules[__name__]
del sys
setup = _module
autogluon = _module
common = _module
features = _module
feature_metadata = _module
infer_types = _module
types = _module
loaders = _module
_utils = _module
load_json = _module
load_pd = _module
load_pkl = _module
load_pointer = _module
load_s3 = _module
load_str = _module
load_zip = _module
savers = _module
save_json = _module
save_pd = _module
save_pkl = _module
save_pointer = _module
save_str = _module
utils = _module
compression_utils = _module
deprecated = _module
file_utils = _module
log_utils = _module
multiprocessing_utils = _module
pandas_utils = _module
s3_utils = _module
conftest = _module
test_check_style = _module
test_compression_utils = _module
test_import_version = _module
test_log_utils = _module
test_setup_outputdir = _module
test_compare_autogluon_metadata = _module
src = _module
core = _module
_setup_utils = _module
augmentation = _module
distill_utils = _module
calibrate = _module
conformity_score = _module
temperature_scaling = _module
constants = _module
data = _module
cleaner = _module
label_cleaner = _module
dataset = _module
hpo = _module
exceptions = _module
executors = _module
ray_hpo = _module
ray_tune_constants = _module
ray_tune_scheduler = _module
ray_tune_scheduler_factory = _module
ray_tune_searcher_factory = _module
space_converter = _module
learner = _module
abstract_learner = _module
locks = _module
metrics = _module
classification_metrics = _module
quantile_metrics = _module
softclass_metrics = _module
models = _module
abstract = _module
_tags = _module
abstract_model = _module
abstract_nn_model = _module
model_trial = _module
ensemble = _module
bagged_ensemble_model = _module
fold_fitting_strategy = _module
ray_parallel_fold_fitting_strategy = _module
stacker_ensemble_model = _module
weighted_ensemble_model = _module
greedy_ensemble = _module
ensemble_selection = _module
greedy_weighted_ensemble_model = _module
pseudolabeling = _module
ray = _module
resources_calculator = _module
scheduler = _module
reporter = _module
scheduler_factory = _module
seq_scheduler = _module
searcher = _module
dummy_searcher = _module
local_grid_searcher = _module
local_random_searcher = _module
local_searcher = _module
searcher_factory = _module
space = _module
task = _module
base = _module
base_task = _module
trainer = _module
abstract_trainer = _module
custom_process = _module
custom_queue = _module
decorators = _module
default_arguments = _module
defaultdict = _module
early_stopping = _module
feature_selection = _module
files = _module
infer_utils = _module
miscs = _module
nvutil = _module
plots = _module
serialization = _module
sync_remote = _module
time = _module
try_import = _module
utils = _module
version_utils = _module
test_scheduler_factory = _module
test_ray_hpo = _module
test_space_converter = _module
test_classification_metrics = _module
test_metric_kwargs = _module
test_metrics = _module
test_quantile_metrics = _module
test_bagged_ensemble_model = _module
test_resource_calculator = _module
test_scheduler = _module
test_seq_scheduler = _module
test_local_grid_searcher = _module
test_local_random_searcher = _module
test_local_searcher = _module
test_feature_selection = _module
test_parallel_local_folding = _module
test_search_space = _module
test_presets = _module
test_time = _module
test_utils = _module
test_version_utils = _module
test = _module
eda = _module
analysis = _module
interaction = _module
missing = _module
model = _module
shift = _module
transform = _module
auto = _module
simple = _module
state = _module
visualization = _module
jupyter = _module
layouts = _module
unittests = _module
test_base = _module
test_dataset = _module
test_interaction = _module
test_missing = _module
test_model = _module
test_transform = _module
test_simple = _module
test_shift = _module
test_state = _module
example_cancer_survival = _module
automm_distillation_glue = _module
automm_distillation_pawsx = _module
eval_pawsx = _module
example_kaggle_house = _module
kaggle_feedback_prize_preprocess = _module
kaggle_feedback_prize_train = _module
kaggle_pawpularity_train = _module
detection_eval = _module
detection_inference = _module
detection_load = _module
detection_train = _module
eval_pretrained_coco_format = _module
eval_pretrained_voc_format = _module
finetune_coco_format = _module
inference_pretrained_coco_format = _module
inference_pretrained_voc_format = _module
load_predictor = _module
quick_start_on_a_tiny_dataset = _module
visualize_results = _module
feature_extraction_example = _module
onnx_text = _module
example_tabular = _module
demo = _module
example_distill_binary = _module
example_advanced_tabular = _module
example_custom_feature_generator = _module
example_custom_model_tabular = _module
example_quantile_regression = _module
example_simple_tabular = _module
generate_submission = _module
prepare_glue = _module
run_competition = _module
run_text_prediction = _module
fair = _module
learners = _module
efficient_compute = _module
fair_frontier = _module
group_metric_classes = _module
group_metrics = _module
test_fair = _module
binning = _module
generators = _module
astype = _module
auto_ml_pipeline = _module
binned = _module
bulk = _module
category = _module
datetime = _module
drop_duplicates = _module
drop_unique = _module
dummy = _module
fillna = _module
identity = _module
isnan = _module
label_encoder = _module
memory_minimize = _module
one_hot_encoder = _module
pipeline = _module
rename = _module
text_ngram = _module
text_special = _module
vectorizers = _module
test_auto_ml_pipeline = _module
test_bulk = _module
test_category = _module
test_datetime = _module
test_dummy = _module
test_fillna = _module
test_identity = _module
test_isnan = _module
test_label_encoder = _module
test_one_hot_encoder = _module
test_pipeline = _module
test_rename = _module
test_text_ngram = _module
test_text_special = _module
test_feature_metadata = _module
setup = _module
multimodal = _module
prepare_detection_dataset = _module
voc2coco = _module
collator = _module
datamodule = _module
dataset = _module
labelencoder_ner = _module
mixup = _module
preprocess_dataframe = _module
process_categorical = _module
process_image = _module
process_label = _module
process_mmlab = _module
process_mmdet = _module
process_mmlab_base = _module
process_mmocr = _module
process_ner = _module
process_numerical = _module
process_text = _module
randaug = _module
template_engine = _module
templates = _module
trivial_augmenter = _module
matcher = _module
adaptation_layers = _module
categorical_mlp = _module
categorical_transformer = _module
clip = _module
ft_transformer = _module
fusion = _module
huggingface_text = _module
mlp = _module
mmdet_image = _module
mmocr_text_detection = _module
mmocr_text_recognition = _module
ner_text = _module
numerical_mlp = _module
numerical_transformer = _module
t_few = _module
timm_image = _module
utils = _module
optimization = _module
deepspeed = _module
lit_distiller = _module
lit_matcher = _module
lit_mmdet = _module
lit_module = _module
lit_ner = _module
losses = _module
lr_scheduler = _module
utils = _module
predictor = _module
presets = _module
problem_types = _module
registry = _module
cache = _module
checkpoint = _module
colormap = _module
config = _module
data = _module
download = _module
environment = _module
inference = _module
load = _module
log = _module
map = _module
matcher = _module
metric = _module
misc = _module
mmcv = _module
model = _module
object_detection = _module
object_detection_visualizer = _module
onnx = _module
save = _module
others = _module
test_auto_model = _module
test_backward_compatibility = _module
test_classification = _module
test_config = _module
test_data_augmentation = _module
test_data_collator = _module
test_data_process_image = _module
test_deployment = _module
test_distiller = _module
test_hpo = _module
test_matcher = _module
test_metrics = _module
test_ner = _module
test_ner_standalone = _module
test_object_detection = _module
test_pipeline_feature_extraction = _module
test_predictor_advanced = _module
test_problem_types = _module
test_registry = _module
test_save_path = _module
test_text_detection = _module
test_text_recognition = _module
test_zero_shot = _module
unittest_datasets = _module
test_predictor = _module
setup = _module
tabular = _module
configs = _module
config_helper = _module
feature_generator_presets = _module
hyperparameter_configs = _module
presets_configs = _module
default_learner = _module
models = _module
rapids_utils = _module
torch_utils = _module
automm = _module
automm_model = _module
catboost = _module
callbacks = _module
catboost_model = _module
catboost_softclass_utils = _module
catboost_utils = _module
hyperparameters = _module
parameters = _module
searchspaces = _module
fastainn = _module
fastai_helpers = _module
imports_helper = _module
quantile_helpers = _module
tabular_nn_fastai = _module
fasttext = _module
fasttext_model = _module
image_prediction = _module
image_predictor = _module
imodels = _module
imodels_models = _module
knn = _module
_knn_loo_variants = _module
knn_model = _module
knn_rapids_model = _module
knn_utils = _module
lgb = _module
lgb_model = _module
lgb_utils = _module
lr = _module
lr_model = _module
lr_preprocessing_utils = _module
lr_rapids_model = _module
rf = _module
compilers = _module
native = _module
rf_model = _module
rf_quantile = _module
rf_rapids_model = _module
tab_transformer = _module
parameters = _module
modified_transformer = _module
pretexts = _module
tab_model_base = _module
tab_transformer = _module
tab_transformer_encoder = _module
tab_transformer_model = _module
utils = _module
tabular_nn = _module
mxnet = _module
embednet = _module
tabular_nn_dataset = _module
tabular_nn_mxnet = _module
tabular_nn_torch = _module
tabular_torch_dataset = _module
torch_network_modules = _module
categorical_encoders = _module
data_preprocessor = _module
nn_architecture_utils = _module
text_prediction = _module
text_prediction_v1_model = _module
vowpalwabbit = _module
vowpalwabbit_model = _module
vowpalwabbit_utils = _module
xgboost = _module
xgboost_model = _module
xgboost_utils = _module
xt = _module
xt_model = _module
auto_trainer = _module
model_presets = _module
presets_custom = _module
presets_distill = _module
tuning = _module
feature_pruner = _module
test_tabular_regression = _module
test_config_helper = _module
test_label_cleaner = _module
test_cascade = _module
test_catboost = _module
test_image_predictor = _module
test_knn = _module
test_lightgbm = _module
test_linear = _module
test_rf = _module
test_tabular_nn = _module
test_tabular_nn_fastai = _module
test_text_prediction_v1_model = _module
test_vowpalwabbit = _module
test_xgboost = _module
test_xt = _module
pseudo_filter = _module
test_bagging_resource_allocation = _module
test_hpo_resource_allocation = _module
test_resource_allocation_combined = _module
test_resources_mocking = _module
test_total_resource_allocation = _module
test_tabular = _module
text = _module
legacy_presets = _module
mx = _module
modules = _module
preprocessing = _module
mx_predictor = _module
test_modules = _module
test_preprocessing = _module
test_legacy_metrics = _module
test_predictor_pytorch = _module
test_text_presets = _module
setup = _module
timeseries = _module
ts_dataframe = _module
evaluator = _module
abstract_timeseries_model = _module
autogluon_tabular = _module
tabular_model = _module
gluonts = _module
abstract_gluonts = _module
callback = _module
prophet = _module
models = _module
local = _module
abstract_local_model = _module
naive = _module
statsmodels = _module
sktime = _module
abstract_sktime = _module
splitter = _module
forecast = _module
hashing = _module
metadata = _module
random = _module
seasonality = _module
warning_filters = _module
test_features_and_covariates = _module
test_es_callback = _module
test_mx = _module
test_gluonts = _module
test_prophet = _module
test_autogluon_tabular = _module
test_ensemble = _module
test_local = _module
test_models = _module
test_sktime = _module
test_evaluator = _module
test_hashing = _module
test_learner = _module
test_splitter = _module
test_trainer = _module
test_ts_dataset = _module
vision = _module
_gluoncv = _module
image_classification = _module
detector = _module
predictor = _module
error_handler = _module
learning_rate = _module
pickle = _module
space_sanitizer = _module
test_gluoncv_image_classification = _module
test_gluoncv_object_detection = _module
test_gluoncv_torch = _module
test_image_classification = _module
test_image_regression = _module

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


import pandas as pd


from pandas import DataFrame


import numpy as np


from sklearn.neighbors import NearestNeighbors


import copy


import time


from collections import defaultdict


from typing import Dict


from typing import List


from typing import Union


from typing import Tuple


from types import ModuleType


import math


import random


from typing import Callable


import scipy.stats


from pandas import Series


from sklearn.model_selection import RepeatedKFold


from sklearn.model_selection import RepeatedStratifiedKFold


from sklearn.model_selection import LeaveOneGroupOut


from sklearn.model_selection import train_test_split


import torch as th


import warnings


from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import mean_squared_error


import torch


from sklearn.metrics.pairwise import paired_cosine_distances


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from torch import tensor


from typing import Optional


from torch.utils.data import DataLoader


from typing import Any


from torch import nn


from torchvision import transforms


from copy import deepcopy


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


import enum


from typing import cast


import collections


from functools import lru_cache


import re


from torch.nn.modules.loss import _Loss


from typing import Generator


from typing import Mapping


import functools


from torch.optim.lr_scheduler import LambdaLR


from torch import optim


from torch.nn import functional as F


from collections import OrderedDict


from sklearn.preprocessing import LabelEncoder


from typing import Sequence


from torch import IntTensor


from scipy.special import softmax


from typing import Type


from typing import Iterable


from sklearn.metrics import f1_score


from sklearn.metrics import log_loss


import numpy.testing as npt


import uuid


from sklearn.isotonic import IsotonicRegression


import sklearn


from torch.nn import Module


from torch.nn import init


from torch.nn.functional import linear


from torch.nn.functional import softmax


from torch.nn.functional import dropout


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.modules.dropout import Dropout


from torch.nn.modules.normalization import LayerNorm


from torch.nn.parameter import Parameter


from collections import Counter


from functools import partial


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.preprocessing import RobustScaler


from sklearn.preprocessing import PowerTransformer


from sklearn.preprocessing import QuantileTransformer


from sklearn.preprocessing import KBinsDiscretizer


from torch.utils.data import Dataset


from sklearn.datasets import make_classification


from sklearn.datasets import make_regression


from typing import Iterator


from pandas.tseries.frequencies import to_offset


def identity(x):
    return x


class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(self, in_features: int, out_features: int, r=8, lora_alpha=8, lora_dropout: float=0.0, fan_in_fan_out=False, merge_weights=False, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool=True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(self, in_features: int, out_features: int, merge_weights: (False), **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights)
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool=True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, in_features: int, out_features: int, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, fan_in_fan_out: bool=False, merge_weights: bool=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool=True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, r: int=0, lora_alpha: int=1, merge_weights: bool=True, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool=True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.lora_B @ self.lora_A * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(x, self.lora_A.T, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
                result += after_A @ self.lora_B.T * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, in_features: int, out_features: int, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, enable_lora: List[bool]=[False], fan_in_fan_out: bool=False, merge_weights: bool=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool=True):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, merge_weights: bool=True, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool=True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(x, self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return nn.Conv2d.forward(self, x)


CATEGORICAL = 'categorical'


FEATURES = 'features'


LABEL = '__label__'


LOGITS = 'logits'


ALL_ACT_LAYERS = {'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'relu': nn.ReLU}


class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization.
    It allows the use of large batch sizes,
    but with batch normalization parameters calculated on smaller sub-batches.

    [1] Train longer, generalize better: closing the generalization gap in large batch training of neural networks : https://arxiv.org/abs/1705.08741
    [2] Simple Modifications to Improve Tabular Neural Networks: https://arxiv.org/pdf/2108.03214
    """

    def __init__(self, input_dim: int, virtual_batch_size: Optional[int]=64, momentum: Optional[float]=0.01):
        super(GhostBatchNorm, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class Unit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """

    def __init__(self, normalization: str, in_features: int, out_features: int, activation: str, dropout_prob: float):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == 'layer_norm':
            self.norm = nn.LayerNorm(in_features)
        elif normalization == 'batch_norm':
            self.norm = nn.BatchNorm1d(in_features)
        elif normalization == 'ghost_batch_norm':
            self.norm = GhostBatchNorm(in_features)
        else:
            raise ValueError(f'unknown normalization: {normalization}')
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP). If the hidden or output feature dimension is
    not provided, we assign it the input feature dimension.
    """

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, num_layers: Optional[int]=1, activation: Optional[str]='gelu', dropout_prob: Optional[float]=0.5, normalization: Optional[str]='layer_norm'):
        """
        Parameters
        ----------
        in_features
            Dimension of input features.
        hidden_features
            Dimension of hidden features.
        out_features
            Dimension of output features.
        num_layers
            Number of layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        layers = []
        for _ in range(num_layers):
            per_unit = Unit(normalization=normalization, in_features=in_features, out_features=hidden_features, activation=activation, dropout_prob=dropout_prob)
            in_features = hidden_features
            layers.append(per_unit)
        if out_features != hidden_features:
            self.fc_out = nn.Linear(hidden_features, out_features)
        else:
            self.fc_out = None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.fc_out is not None:
            return self.fc_out(x)
        else:
            return x


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class CategoricalMLP(nn.Module):
    """
    MLP for categorical input. The input dimension is automatically computed based on
    the number of categories in each categorical column.
    """

    def __init__(self, prefix: str, num_categories: List[int], out_features: Optional[int]=None, num_layers: Optional[int]=1, activation: Optional[str]='gelu', dropout_prob: Optional[float]=0.5, normalization: Optional[str]='layer_norm', num_classes: Optional[int]=0):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        num_categories
            A list of integers. Each one is the number of categories in one categorical column.
        out_features
            Dimension of output features.
        num_layers
            Number of MLP layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        num_classes
            Number of classes. 1 for a regression task.
        """
        super().__init__()
        self.out_features = out_features
        max_embedding_dim = 100
        embed_exponent = 0.56
        size_factor = 1.0
        self.column_embeddings = nn.ModuleList()
        self.column_mlps = nn.ModuleList()
        assert isinstance(num_categories, list)
        for num_categories_per_col in num_categories:
            embedding_dim_per_col = int(size_factor * max(2, min(max_embedding_dim, 1.6 * num_categories_per_col ** embed_exponent)))
            self.column_embeddings.append(nn.Embedding(num_embeddings=num_categories_per_col, embedding_dim=embedding_dim_per_col))
            self.column_mlps.append(MLP(in_features=embedding_dim_per_col, hidden_features=out_features, out_features=out_features, num_layers=num_layers, activation=activation, dropout_prob=dropout_prob, normalization=normalization))
        self.aggregator_mlp = MLP(in_features=out_features * len(num_categories), hidden_features=out_features * len(num_categories), out_features=out_features, num_layers=num_layers, activation=activation, dropout_prob=dropout_prob, normalization=normalization)
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(init_weights)
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def categorical_key(self):
        return f'{self.prefix}_{CATEGORICAL}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: dict):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        assert len(batch[self.categorical_key]) == len(self.column_embeddings)
        features = []
        for categorical_id, embed, mlp in zip(batch[self.categorical_key], self.column_embeddings, self.column_mlps):
            features.append(mlp(embed(categorical_id)))
        cat_features = torch.cat(features, dim=1)
        features = self.aggregator_mlp(cat_features)
        logits = self.head(features)
        return {self.prefix: {LOGITS: logits, FEATURES: features}}

    def get_layer_ids(self):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) ->'_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) ->None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Feature tokenizer for categorical features in tabular data.
    It transforms the input categorical features to tokens (embeddings).

    The categorical features usually refers to discrete features.
    """

    def __init__(self, num_categories: List[int], d_token: int, bias: Optional[bool]=True, initialization: Optional[str]='normal') ->None:
        """
        Parameters
        ----------
        num_categories:
            A list of integers. Each one is the number of categories in one categorical column.
        d_token:
            The size of one token.
        bias:
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization:
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`.

        References
        ----------
        1. Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        2. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        """
        super().__init__()
        self.num_categories = num_categories
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(num_categories), d_token)
        self.bias = nn.Parameter(Tensor(len(num_categories), d_token)) if bias else None
        initialization_ = _TokenInitialization.from_str(initialization)
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return len(self.num_categories)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) ->Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [1].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    References:
    ----------
    [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) ->None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) ->Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `_CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) ->Tensor:
        """Append self **to the end** of each item in the batch (see `_CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class AdditiveAttention(nn.Module):
    """Additive Attention with linear complexity to input sequence length.

    Additive attention was proposed and used in FastFormer.
    See Ref. [1] for details.
    This implementation is motivated by: https://github.com/jrzaurin/pytorch-widedeep.git

    References:
    ----------
    [1] Wu, Chuhan, et al. "Fastformer: Additive attention can be all you need." arXiv preprint arXiv:2108.09084 (2021).
    """

    def __init__(self, *, d_token: int, n_heads: int, dropout: float, bias: bool, share_qv_weights: bool, initialization: str) ->None:
        """
        Parameters
        ----------
        d_token:
            the token size. Must be a multiple of :code:`n_heads`.
        n_heads:
            the number of heads. If greater than 1, then the module will have
            an addition output layer (so called "mixing" layer).
        dropout:
            dropout rate for the attention map. The dropout is applied to
            *probabilities* and do not affect logits.
        bias:
            if `True`, then input (and output, if presented) layers also have bias.
            `True` is a reasonable default choice.
        share_qv_weights:
            if 'True', then value and query transformation parameters are shared.
        initialization:
            initialization for input projection layers. Must be one of
            :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        """
        super().__init__()
        assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']
        self.head_dim = d_token // n_heads
        self.n_heads = n_heads
        self.share_qv_weights = share_qv_weights
        self.dropout = nn.Dropout(dropout)
        trainable = []
        if share_qv_weights:
            self.qv_proj = nn.Linear(d_token, d_token, bias=bias)
            trainable.extend([self.qv_proj])
        else:
            self.q_proj = nn.Linear(d_token, d_token, bias=bias)
            self.v_proj = nn.Linear(d_token, d_token, bias=bias)
            trainable.extend([self.q_proj, self.v_proj])
        self.k_proj = nn.Linear(d_token, d_token, bias=bias)
        self.W_q = nn.Linear(d_token, n_heads)
        self.W_k = nn.Linear(d_token, n_heads)
        self.r_out = nn.Linear(d_token, d_token)
        trainable.extend([self.k_proj, self.W_q, self.W_k, self.r_out])
        if initialization == 'xavier':
            self.apply(init_weights)
        else:
            for m in trainable:
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_q: Tensor, x_kv: Tensor, *args) ->Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, n_q_tokens, d_token = x_q.shape
        batch_size, n_k_tokens, d_token = x_kv.shape
        q = self.qv_proj(x_q) if self.share_qv_weights else self.q_proj(x_q)
        v = self.qv_proj(x_kv) if self.share_qv_weights else self.v_proj(x_kv)
        k = self.k_proj(x_kv)
        alphas = (self.W_q(q) / math.sqrt(self.head_dim)).softmax(dim=1)
        q_r = q.reshape(batch_size, n_q_tokens, self.n_heads, self.head_dim)
        global_query = torch.einsum(' b s h, b s h d -> b h d', alphas, q_r)
        global_query = global_query.reshape(batch_size, self.n_heads * self.head_dim).unsqueeze(1)
        p = k * global_query
        betas = (self.W_k(p) / math.sqrt(self.head_dim)).softmax(dim=1)
        p_r = p.reshape(batch_size, n_k_tokens, self.n_heads, self.head_dim)
        global_key = torch.einsum(' b s h, b s h d -> b h d', betas, p_r)
        global_key = global_key.reshape(batch_size, self.n_heads * self.head_dim).unsqueeze(1)
        u = v * global_key
        output = q + self.dropout(self.r_out(u))
        return output, {'query_weight': alphas, 'key_weight': betas}


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \\text{in\\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \\text{out\\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                :math:`k = \\frac{1}{\\text{in\\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool=True) ->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class _LinearWithBias(Linear):
    bias: torch.Tensor

    def __init__(self, in_features: int, out_features: int) ->None:
        super().__init__(in_features, out_features, bias=True)


def multi_head_attention_forward(self, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, fixed_k=None, fixed_q=None, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
        if any([(type(t) is not torch.Tensor) for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(self.multi_head_attention_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    v = linear(query, in_proj_weight, in_proj_bias)
    k = torch.cat([fixed_k.unsqueeze(1) for _ in range(key.shape[1])], dim=1)
    q = torch.cat([fixed_q.unsqueeze(1) for _ in range(key.shape[1])], dim=1)
    q = q * scaling
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        text{MultiHead}(Q, K, V) = text{Concat}(head_1,dots,head_h)W^O
        text{where} head_i = text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {'bias_k': torch._jit_internal.Optional[torch.Tensor], 'bias_v': torch._jit_internal.Optional[torch.Tensor]}

    def __init__(self, embed_dim, n_cat_embeddings, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
            self.register_parameter('fixed_k', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(embed_dim, embed_dim))
            self.fixed_k = Parameter(torch.empty(n_cat_embeddings, embed_dim))
            self.fixed_q = Parameter(torch.empty(n_cat_embeddings, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
            xavier_uniform_(self.fixed_k)
            xavier_uniform_(self.fixed_q)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super().__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        return multi_head_attention_forward(self, query=query, key=key, value=value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads, in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=self.bias_k, bias_v=self.bias_v, add_zero_attn=self.add_zero_attn, dropout_p=self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, fixed_k=self.fixed_k, fixed_q=self.fixed_q, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)


_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


def geglu(x: Tensor) ->Tensor:
    """The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class GEGLU(nn.Module):
    """
    The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) ->Tensor:
        return geglu(x)


def reglu(x: Tensor) ->Tensor:
    """The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    """
    The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) ->Tensor:
        return reglu(x)


def _is_glu_activation(activation: ModuleType):
    return isinstance(activation, str) and activation.endswith('glu') or activation in [ReGLU, GEGLU]


def _make_nn_module(module_type: ModuleType, *args) ->nn.Module:
    if isinstance(module_type, str):
        if module_type == 'reglu':
            return ReGLU()
        elif module_type == 'geglu':
            return GEGLU()
        elif module_type == 'gelu':
            return nn.GELU()
        elif module_type == 'relu':
            return nn.ReLU()
        elif module_type == 'leaky_relu':
            return nn.LeakyReLU()
        elif module_type == 'layer_norm':
            return nn.LayerNorm(*args)
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(f'Failed to construct the module {module_type} with the arguments {args}') from err
            return cls(*args)
    else:
        return module_type(*args)


class FT_Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""
    WARNINGS = {'first_prenormalization': True, 'prenormalization': True}


    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(self, *, d_token: int, d_hidden: int, bias_first: bool, bias_second: bool, dropout: float, activation: ModuleType):
            super().__init__()
            self.linear_first = nn.Linear(d_token, d_hidden * (2 if _is_glu_activation(activation) else 1), bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) ->Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(self, *, d_in: int, bias: bool, activation: ModuleType, normalization: ModuleType, d_out: int):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) ->Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(self, *, d_token: int, n_blocks: int, attention_n_heads: int, attention_dropout: float, attention_initialization: str, attention_normalization: str, ffn_d_hidden: int, ffn_dropout: float, ffn_activation: str, ffn_normalization: str, residual_dropout: float, prenormalization: bool, first_prenormalization: bool, last_layer_query_idx: Union[None, List[int], slice], n_tokens: Optional[int], kv_compression_ratio: Optional[float], kv_compression_sharing: Optional[str], head_activation: ModuleType, head_normalization: ModuleType, d_out: int, projection: Optional[bool]=False, additive_attention: Optional[bool]=False, share_qv_weights: Optional[bool]=False) ->None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        n_tokens
            Number of tokens of the input sequence.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        d_out
            Output dimension.
        projection
            Whether to use a project head.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        """
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(f'last_layer_query_idx must be None, list[int] or slice. Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?')
        if not prenormalization:
            assert not first_prenormalization, 'If `prenormalization` is False, then `first_prenormalization` must be False'
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), 'If any of the following arguments is (not) None, then all of them must (not) be None: n_tokens, kv_compression_ratio, kv_compression_sharing'
        assert additive_attention or not share_qv_weights, 'If `share_qv_weights` is True, then `additive_attention` must be True'
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn('prenormalization is set to False. Are you sure about this? The training can become less stable. You can turn off this warning by tweaking the rtdl.Transformer.WARNINGS dictionary.', UserWarning)
            assert not first_prenormalization, 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        if prenormalization and first_prenormalization and self.WARNINGS['first_prenormalization']:
            warnings.warn('first_prenormalization is set to True. Are you sure about this? For example, the vanilla FTTransformer with first_prenormalization=True performs SIGNIFICANTLY worse. You can turn off this warning by tweaking the rtdl.Transformer.WARNINGS dictionary.', UserWarning)

        def make_kv_compression():
            assert n_tokens and kv_compression_ratio, _INTERNAL_ERROR_MESSAGE
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)
        self.shared_kv_compression = make_kv_compression() if kv_compression_ratio and kv_compression_sharing == 'layerwise' else None
        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx
        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict({'attention': AdditiveAttention(d_token=d_token, n_heads=attention_n_heads, dropout=attention_dropout, bias=True, share_qv_weights=share_qv_weights, initialization=attention_initialization) if additive_attention else MultiheadAttention(d_token=d_token, n_heads=attention_n_heads, dropout=attention_dropout, bias=True, initialization=attention_initialization), 'ffn': FT_Transformer.FFN(d_token=d_token, d_hidden=ffn_d_hidden, bias_first=True, bias_second=True, dropout=ffn_dropout, activation=ffn_activation), 'attention_residual_dropout': nn.Dropout(residual_dropout), 'ffn_residual_dropout': nn.Dropout(residual_dropout), 'output': nn.Identity()})
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(attention_normalization, d_token)
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value', _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)
        self.head = FT_Transformer.Head(d_in=d_token, d_out=d_out, bias=True, activation=head_activation, normalization=head_normalization if prenormalization else 'Identity') if projection else nn.Identity()

    def _get_kv_compressions(self, layer):
        return (self.shared_kv_compression, self.shared_kv_compression) if self.shared_kv_compression is not None else (layer['key_compression'], layer['value_compression']) if 'key_compression' in layer and 'value_compression' in layer else (layer['key_compression'], layer['key_compression']) if 'key_compression' in layer else (None, None)

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        assert x.ndim == 3, 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)
            query_idx = self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](x_residual if query_idx is None else x_residual[:, query_idx], x_residual, *self._get_kv_compressions(layer))
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)
            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)
        x = self.head(x)
        return x


class CategoricalTransformer(nn.Module):
    """
    FT-Transformer for categorical tabular features.
    The input dimension is automatically computed based on
    the number of categories in each categorical column.
    """

    def __init__(self, prefix: str, num_categories: List[int], d_token: int, cls_token: Optional[bool]=False, out_features: Optional[int]=None, num_classes: Optional[int]=0, token_bias: Optional[bool]=True, token_initialization: Optional[str]='normal', n_blocks: Optional[int]=0, attention_n_heads: Optional[int]=8, attention_initialization: Optional[str]='kaiming', attention_normalization: Optional[str]='layer_norm', attention_dropout: Optional[str]=0.2, residual_dropout: Optional[str]=0.0, ffn_activation: Optional[str]='reglu', ffn_normalization: Optional[str]='layer_norm', ffn_d_hidden: Optional[str]=6, ffn_dropout: Optional[str]=0.0, prenormalization: Optional[bool]=True, first_prenormalization: Optional[bool]=False, kv_compression_ratio: Optional[float]=None, kv_compression_sharing: Optional[str]=None, head_activation: Optional[str]='relu', head_normalization: Optional[str]='layer_norm', additive_attention: Optional[bool]=False, share_qv_weights: Optional[bool]=False) ->None:
        """
        Parameters
        ----------
        prefix
            The model prefix.
        num_categories
            A list of integers. Each one is the number of categories in one categorical column.
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        cls_token
            If `True`, cls token will be added to the token embeddings.
        out_features
            Dimension of output features.
        num_classes
            Number of classes. 1 for a regression task.
        token_bias
            If `True`, for each feature, an additional trainable vector will be added in `_CategoricalFeatureTokenizer`
            to the embedding regardless of feature value. Notably, the bias are not shared between features.
        token_initialization
            Initialization policy for parameters in `_CategoricalFeatureTokenizer` and `_CLSToke`.
            Must be one of `['uniform', 'normal']`.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.

        References
        ----------
        1. Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        2. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        """
        super().__init__()
        assert num_categories, 'num_categories must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        assert n_blocks >= 0, 'n_blocks must be non-negative'
        assert attention_n_heads > 0, 'attention_n_heads must be positive'
        assert token_initialization in ['uniform', 'normal'], 'initialization must be uniform or normal'
        self.num_categories = num_categories
        self.prefix = prefix
        self.out_features = out_features
        self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(num_categories=num_categories, d_token=d_token, bias=token_bias, initialization=token_initialization)
        self.cls_token = CLSToken(d_token=d_token, initialization=token_initialization) if cls_token else nn.Identity()
        if kv_compression_ratio is not None:
            if cls_token:
                n_tokens = self.categorical_feature_tokenizer.n_tokens + 1
            else:
                n_tokens = self.categorical_feature_tokenizer.n_tokens
        else:
            n_tokens = None
        self.transformer = FT_Transformer(d_token=d_token, n_blocks=n_blocks, attention_n_heads=attention_n_heads, attention_dropout=attention_dropout, attention_initialization=attention_initialization, attention_normalization=attention_normalization, ffn_d_hidden=ffn_d_hidden, ffn_dropout=ffn_dropout, ffn_activation=ffn_activation, ffn_normalization=ffn_normalization, residual_dropout=residual_dropout, prenormalization=prenormalization, first_prenormalization=first_prenormalization, last_layer_query_idx=None, n_tokens=n_tokens, kv_compression_ratio=kv_compression_ratio, kv_compression_sharing=kv_compression_sharing, head_activation=head_activation, head_normalization=head_normalization, d_out=out_features, additive_attention=additive_attention, share_qv_weights=share_qv_weights)
        self.head = FT_Transformer.Head(d_in=d_token, d_out=num_classes, bias=True, activation=head_activation, normalization=head_normalization if prenormalization else 'Identity')
        self.name_to_id = self.get_layer_ids()

    @property
    def categorical_key(self):
        return f'{self.prefix}_{CATEGORICAL}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: dict):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        categorical_features = []
        for categorical_feature in batch[self.categorical_key]:
            categorical_features.append(categorical_feature)
        categorical_features = torch.stack(categorical_features, dim=1)
        features = self.categorical_feature_tokenizer(categorical_features)
        features = self.cls_token(features)
        features = self.transformer(features)
        logits = self.head(features)
        return {self.prefix: {LOGITS: logits, FEATURES: features}}

    def get_layer_ids(self):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


COLUMN = 'column'


COLUMN_FEATURES = 'column_features'


IMAGE = 'image'


IMAGE_VALID_NUM = 'image_valid_num'


LOGIT_SCALE = 'logit_scale'


MASKS = 'masks'


TEXT_TOKEN_IDS = 'text_token_ids'


TEXT_VALID_LENGTH = 'text_valid_length'


def assign_encoder_layer_ids(encoder_names: List[List[str]]):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split('.')
            for split in n_splits:
                if split.isdigit():
                    inferred_id = int(split)
                    if inferred_id != last_inferred_id:
                        cur_id += 1
                        last_inferred_id = inferred_id
                    name_to_id[n] = cur_id
                    detect_id = True
                    break
            if detect_id is False:
                raise ValueError(f'parameter name: {n} not has no id inside')
    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(non_encoder_names: List[str], layer_id: int):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str]):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        for i in n.split('.'):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)
    return encoder_names, non_encoder_names


def group_param_names(names: List[str], pre_encoder_patterns: Tuple[str, ...], post_encoder_patterns: Tuple[str, ...], model_prefix: Optional[str]=None):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)
    left_names = []
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)
    children_prefix = []
    for n in selected_names:
        child_name = n[len(model_prefix) + 1:].split('.')[0]
        child_prefix = f'{model_prefix}.{child_name}'
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)
    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)
    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f'parameter name: {n} belong to neither pre or post encoder names')
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


logger = logging.getLogger(__name__)


def reverse_layer_ids(encoder_name_to_id: dict, pre_enocder_name_to_id: dict, post_enocder_name_to_id: dict):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id
    return name_to_id


def assign_layer_ids(names: List[str], pre_encoder_patterns: Tuple[str, ...], post_encoder_patterns: Tuple[str, ...], model_pre: Optional[str]=None):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_prefix=model_pre)
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f'encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}')
        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(encoder_names=encoder_names)
        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)
        post_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1)
        name_to_id = reverse_layer_ids(encoder_name_to_id=encoder_name_to_id, pre_enocder_name_to_id=pre_encoder_name_to_id, post_enocder_name_to_id=post_encoder_name_to_id)
    except Exception as e:
        logger.debug(f'When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id.')
        name_to_id = dict()
        left_names = names
    return name_to_id, left_names


def get_column_features(batch: Dict[str, torch.Tensor], column_name_prefix: str, features: torch.Tensor, valid_lengths: torch.Tensor, cls_feature: Optional[torch.Tensor]=None):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}
    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        joint_mask = torch.zeros(features.shape[0])
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0])
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)
            feature_masks[column_name] = per_col_masks
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)
    if cls_feature is not None and len(all_column_names) > 0:
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)
        joint_column_name = '_'.join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask
    return column_features, feature_masks


def get_hf_config_and_model(checkpoint_name: str, pretrained: Optional[bool]=True, low_cpu_mem_usage: Optional[bool]=False):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)
    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage)
    else:
        model = AutoModel.from_config(config)
    return config, model


class CLIPForImageText(nn.Module):
    """
    Support the CLIP model.
    Refer to https://huggingface.co/docs/transformers/model_doc/clip
    """

    def __init__(self, prefix: str, checkpoint_name: str, num_classes: Optional[int]=None, pretrained: Optional[bool]=True):
        """
        Load the pretrained CLIP from huggingface transformers.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint.
        num_classes
            The number of classes. 1 for a regression task.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.config, self.model = get_hf_config_and_model(checkpoint_name=checkpoint_name, pretrained=pretrained)
        self.out_features = self.model.config.projection_dim
        self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
        self.head.apply(init_weights)
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def text_token_ids_key(self):
        return f'{self.prefix}_{TEXT_TOKEN_IDS}'

    @property
    def text_valid_length_key(self):
        return f'{self.prefix}_{TEXT_VALID_LENGTH}'

    @property
    def image_key(self):
        return f'{self.prefix}_{IMAGE}'

    @property
    def image_valid_num_key(self):
        return f'{self.prefix}_{IMAGE_VALID_NUM}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def text_column_prefix(self):
        return f'{self.text_token_ids_key}_{COLUMN}'

    @property
    def image_column_prefix(self):
        return f'{self.image_key}_{COLUMN}'

    @property
    def text_feature_dim(self):
        return self.model.config.text_config.hidden_size

    @property
    def image_feature_dim(self):
        return self.model.config.vision_config.hidden_size

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        has_image = self.image_key in batch
        has_text = self.text_token_ids_key in batch
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if has_image:
            images = batch[self.image_key]
            image_valid_num = batch[self.image_valid_num_key]
            assert images.dim() == 5
            b, n, c, h, w = images.shape
            vision_outputs = self.model.vision_model(pixel_values=images.reshape((b * n, c, h, w)), output_attentions=True, output_hidden_states=True, return_dict=True)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = (steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))).type_as(image_features)
            image_features = image_features.reshape((b, n, -1)) * image_masks[:, :, None]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_column_features, image_column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.image_column_prefix, features=image_features, valid_lengths=image_valid_num)
            ret[COLUMN_FEATURES][FEATURES].update(image_column_features)
            ret[COLUMN_FEATURES][MASKS].update(image_column_feature_masks)
            image_features = image_features.mean(dim=1)
            ret[FEATURES] = image_features
        if has_text:
            text_token_ids = batch[self.text_token_ids_key]
            text_valid_length = batch[self.text_valid_length_key]
            steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
            text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
            assert torch.equal(text_valid_length, text_masks.sum(dim=-1))
            text_outputs = self.model.text_model(input_ids=text_token_ids, attention_mask=text_masks, output_attentions=True, output_hidden_states=True, return_dict=True)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_column_features, text_column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=self.model.text_projection(text_outputs.last_hidden_state), valid_lengths=text_valid_length, cls_feature=text_features)
            ret[COLUMN_FEATURES][FEATURES].update(text_column_features)
            ret[COLUMN_FEATURES][MASKS].update(text_column_feature_masks)
            ret[FEATURES] = text_features
        if has_image and has_text:
            if self.num_classes:
                features = image_features + text_features
                logits = self.head(features)
                ret[FEATURES] = features
            else:
                logits = torch.sum(image_features * text_features, dim=-1)
            ret[LOGITS] = logits
        ret[LOGIT_SCALE] = self.model.logit_scale.exp()
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefixes = ['model.text_model', 'model.vision_model', 'model']
        for i, model_pre in enumerate(model_prefixes):
            for model_pre2 in model_prefixes[i + 1:]:
                if model_pre2.startswith(model_pre):
                    raise ValueError(f'{model_pre} is a substring of {model_pre2}. Need to swap them in {model_prefixes}.')
        pre_encoder_patterns = 'embeddings', 'pre'
        post_encoder_patterns = 'head', 'final', 'post', 'logit', 'project'
        names = [n for n, _ in self.named_parameters()]
        name_to_id = {}
        for per_prefix in model_prefixes:
            per_model_name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=per_prefix)
            name_to_id.update(per_model_name_to_id)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0
        return name_to_id


WEIGHT = 'weight'


class MultimodalFusionMLP(nn.Module):
    """
    Use MLP to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through MLP.
    """

    def __init__(self, prefix: str, models: list, hidden_features: List[int], num_classes: int, adapt_in_features: Optional[str]=None, activation: Optional[str]='gelu', dropout_prob: Optional[float]=0.5, normalization: Optional[str]='layer_norm', loss_weight: Optional[float]=None):
        """
        Parameters
        ----------
        prefix
            The fusion model's prefix
        models
            The individual models whose output features will be fused.
        hidden_features
            A list of integers representing the hidden feature dimensions. For example,
            [512, 128, 64] indicates three hidden MLP layers with their corresponding output
            feature dimensions.
        num_classes
            The number of classes.
        adapt_in_features
            Choice of how to adapt the features of each model. We now support
            - min
                Adapt all features to the minimum dimension. For example, if three models have
                feature dimensions [512, 768, 64], it will linearly map all the features to
                dimension 64.
            - max
                Adapt all features to the maximum dimension. For example, if three models have
                feature dimensions are [512, 768, 64], it will linearly map all the features to
                dimension 768.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxiliary loss for each individual model.
        """
        super().__init__()
        logger.debug('initializing MultimodalFusionMLP')
        if loss_weight is not None:
            assert loss_weight > 0
        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)
        self.num_classes = num_classes
        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == 'min':
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == 'max':
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f'unknown adapt_in_features: {adapt_in_features}')
            self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])
            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList([nn.Identity() for _ in range(len(raw_in_features))])
            in_features = sum(raw_in_features)
        assert len(self.adapter) == len(self.model)
        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(MLP(in_features=in_features, hidden_features=per_hidden_features, out_features=per_hidden_features, num_layers=1, activation=activation, dropout_prob=dropout_prob, normalization=normalization))
            in_features = per_hidden_features
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        self.head = nn.Linear(in_features, num_classes)
        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)
        self.out_features = in_features
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: dict):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data. The fusion model doesn't need to
            directly access the mini-batch data since it aims to fuse the individual models'
            output features.

        Returns
        -------
        If "loss_weight" is None, it returns dictionary containing the fusion model's logits and
        features. Otherwise, it returns a list of dictionaries collecting all the models' output,
        including the fusion model's.
        """
        multimodal_features = []
        output = {}
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_features.append(per_adapter(per_output[per_model.prefix][FEATURES]))
            if self.loss_weight is not None:
                per_output[per_model.prefix].update({WEIGHT: self.loss_weight})
                output.update(per_output)
        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        logits = self.head(features)
        fusion_output = {self.prefix: {LOGITS: logits, FEATURES: features}}
        if self.loss_weight is not None:
            fusion_output[self.prefix].update({WEIGHT: 1})
            output.update(fusion_output)
            return output
        else:
            return fusion_output

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        names = [n for n, _ in self.named_parameters()]
        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]
        name_to_id = {}
        logger.debug(f'outer layers are treated as head: {outer_layer_names}')
        for n in outer_layer_names:
            name_to_id[n] = 0
        for i, per_model in enumerate(self.model):
            per_model_prefix = f'{model_prefix}.{i}'
            if not hasattr(per_model, 'name_to_id'):
                raise ValueError(f'name_to_id attribute is missing in model: {per_model.__class__.__name__}')
            for n, layer_id in per_model.name_to_id.items():
                full_n = f'{per_model_prefix}.{n}'
                name_to_id[full_n] = layer_id
        for n in names:
            assert n in name_to_id
        return name_to_id


NER_ANNOTATION = 'ner_annotation'


NER_TEXT = 'ner_text'


TOKEN_WORD_MAPPING = 'token_word_mapping'


WORD_OFFSETS = 'word_offsets'


class MultimodalFusionNER(MultimodalFusionMLP):
    """
    Use MLP to fuse different models' features (single-modal and multimodal) for NER.
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through MLP.
    """

    def __init__(self, prefix: str, models: list, hidden_features: List[int], num_classes: int, adapt_in_features: str='max', activation: Optional[str]='gelu', dropout_prob: Optional[float]=0.5, normalization: Optional[str]='layer_norm', loss_weight: Optional[float]=None):
        """
        Parameters
        ----------
        prefix
            The fusion model's prefix
        models
            The individual models whose output features will be fused.
        hidden_features
            A list of integers representing the hidden feature dimensions. For example,
            [512, 128, 64] indicates three hidden MLP layers with their corresponding output
            feature dimensions.
        num_classes
            The number of classes.
        adapt_in_features
            Choice of how to adapt the features of each model. We now support
            - min
                Adapt all features to the minimum dimension. For example, if three models have
                feature dimensions [512, 768, 64], it will linearly map all the features to
                dimension 64.
            - max
                Adapt all features to the maximum dimension. For example, if three models have
                feature dimensions are [512, 768, 64], it will linearly map all the features to
                dimension 768.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        loss_weight
            The weight of individual models.
        """
        super().__init__(prefix=prefix, models=models, hidden_features=hidden_features, num_classes=num_classes, adapt_in_features=adapt_in_features, activation=activation, dropout_prob=dropout_prob, normalization=normalization, loss_weight=None)
        logger.debug('initializing MultimodalFusionNER')
        self.ner_model = None
        self.tokenizer = None
        other_models = []
        for per_model in models:
            if per_model.prefix != NER_TEXT:
                other_models.append(per_model)
            else:
                self.ner_model = per_model
                self.tokenizer = per_model.tokenizer
        self.other_models = nn.ModuleList(other_models)
        raw_in_features = [per_model.out_features for per_model in models if per_model.prefix != NER_TEXT]
        if adapt_in_features is not None:
            if adapt_in_features == 'min':
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == 'max':
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f'unknown adapt_in_features: {adapt_in_features}')
            self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])
            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList([nn.Identity() for _ in range(len(raw_in_features))])
            in_features = sum(raw_in_features)
        assert len(self.adapter) == len(self.other_models)
        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(MLP(in_features=in_features, hidden_features=per_hidden_features, out_features=per_hidden_features, num_layers=1, activation=activation, dropout_prob=dropout_prob, normalization=normalization))
            in_features = per_hidden_features
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        self.head = nn.Linear(in_features + self.ner_model.out_features, num_classes)

    @property
    def label_key(self):
        return f'{NER_TEXT}_{LABEL}'

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data. The fusion model doesn't need to
            directly access the mini-batch data since it aims to fuse the individual models'
            output features.
        Returns
        -------
        It returns dictionary containing the fusion model's logits and features.
        """
        multimodal_features = []
        output = {}
        ner_output = self.ner_model(batch)
        for per_model, per_adapter in zip(self.other_models, self.adapter):
            per_output = per_model(batch)
            multimodal_features.append(per_adapter(per_output[per_model.prefix][FEATURES]))
        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        features = features.unsqueeze(dim=1).repeat(1, ner_output[self.ner_model.prefix][FEATURES].size()[1], 1)
        features = torch.cat((ner_output[self.ner_model.prefix][FEATURES], features), dim=-1)
        logits = self.head(features)
        logits_label = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
        fusion_output = {self.prefix: {LOGITS: logits, FEATURES: features, NER_ANNOTATION: logits_label, TOKEN_WORD_MAPPING: ner_output[self.ner_model.prefix][TOKEN_WORD_MAPPING], WORD_OFFSETS: ner_output[self.ner_model.prefix][WORD_OFFSETS]}}
        return fusion_output


class MultimodalFusionTransformer(nn.Module):
    """
    Use Transformer to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through Transformer.
    """

    def __init__(self, prefix: str, models: list, hidden_features: int, num_classes: int, n_blocks: Optional[int]=0, attention_n_heads: Optional[int]=8, attention_initialization: Optional[str]='kaiming', attention_normalization: Optional[str]='layer_norm', attention_dropout: Optional[str]=0.2, residual_dropout: Optional[str]=0.0, ffn_activation: Optional[str]='reglu', ffn_normalization: Optional[str]='layer_norm', ffn_d_hidden: Optional[str]=192, ffn_dropout: Optional[str]=0.0, prenormalization: Optional[bool]=True, first_prenormalization: Optional[bool]=False, kv_compression_ratio: Optional[float]=None, kv_compression_sharing: Optional[str]=None, head_activation: Optional[str]='relu', head_normalization: Optional[str]='layer_norm', adapt_in_features: Optional[str]=None, loss_weight: Optional[float]=None, additive_attention: Optional[bool]=False, share_qv_weights: Optional[bool]=False):
        """
        Parameters
        ----------
        prefix
            The fusion model's prefix
        models
            The individual models whose output features will be fused.
        hidden_features
            A list of integers representing the hidden feature dimensions. For example,
            [512, 128, 64] indicates three hidden MLP layers with their corresponding output
            feature dimensions.
        num_classes
            The number of classes.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        adapt_in_features
            Choice of how to adapt the features of each model. We now support
            - min
                Adapt all features to the minimum dimension. For example, if three models have
                feature dimensions [512, 768, 64], it will linearly map all the features to
                dimension 64.
            - max
                Adapt all features to the maximum dimension. For example, if three models have
                feature dimensions are [512, 768, 64], it will linearly map all the features to
                dimension 768.
        loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxiliary loss for each individual model.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        """
        super().__init__()
        logger.debug('initializing MultimodalFusionTransformer')
        if loss_weight is not None:
            assert loss_weight > 0
        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)
        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features == 'min':
            base_in_feat = min(raw_in_features)
        elif adapt_in_features == 'max':
            base_in_feat = max(raw_in_features)
        else:
            raise ValueError(f'unknown adapt_in_features: {adapt_in_features}')
        self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])
        in_features = base_in_feat
        assert len(self.adapter) == len(self.model)
        self.fusion_transformer = FT_Transformer(d_token=in_features, n_blocks=n_blocks, attention_n_heads=attention_n_heads, attention_dropout=attention_dropout, attention_initialization=attention_initialization, attention_normalization=attention_normalization, ffn_d_hidden=ffn_d_hidden, ffn_dropout=ffn_dropout, ffn_activation=ffn_activation, ffn_normalization=ffn_normalization, residual_dropout=residual_dropout, prenormalization=prenormalization, first_prenormalization=first_prenormalization, last_layer_query_idx=None, n_tokens=None, kv_compression_ratio=kv_compression_ratio, kv_compression_sharing=kv_compression_sharing, head_activation=head_activation, head_normalization=head_normalization, d_out=hidden_features, projection=False, additive_attention=additive_attention, share_qv_weights=share_qv_weights)
        self.head = FT_Transformer.Head(d_in=in_features, d_out=num_classes, bias=True, activation=head_activation, normalization=head_normalization)
        self.cls_token = CLSToken(d_token=in_features, initialization='uniform')
        self.out_features = in_features
        self.adapter.apply(init_weights)
        self.head.apply(init_weights)
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: dict):
        multimodal_features = []
        output = {}
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)
            if self.loss_weight is not None:
                per_output[per_model.prefix].update({WEIGHT: self.loss_weight})
                output.update(per_output)
        multimodal_features = torch.cat(multimodal_features, dim=1)
        multimodal_features = self.cls_token(multimodal_features)
        features = self.fusion_transformer(multimodal_features)
        logits = self.head(features)
        fusion_output = {self.prefix: {LOGITS: logits, FEATURES: features}}
        if self.loss_weight is not None:
            fusion_output[self.prefix].update({WEIGHT: 1})
            output.update(fusion_output)
            return output
        else:
            return fusion_output

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        names = [n for n, _ in self.named_parameters()]
        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]
        name_to_id = {}
        logger.debug(f'outer layers are treated as head: {outer_layer_names}')
        for n in outer_layer_names:
            name_to_id[n] = 0
        for i, per_model in enumerate(self.model):
            per_model_prefix = f'{model_prefix}.{i}'
            if not hasattr(per_model, 'name_to_id'):
                raise ValueError(f'name_to_id attribute is missing in model: {per_model.__class__.__name__}')
            for n, layer_id in per_model.name_to_id.items():
                full_n = f'{per_model_prefix}.{n}'
                name_to_id[full_n] = layer_id
        for n in names:
            assert n in name_to_id
        return name_to_id


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias - self.dummy_bias


TEXT_SEGMENT_IDS = 'text_segment_ids'


class HFAutoModelForTextPrediction(nn.Module):
    """
    Support huggingface text backbones.
    Refer to https://github.com/huggingface/transformers
    """

    def __init__(self, prefix: str, checkpoint_name: str='microsoft/deberta-v3-base', num_classes: Optional[int]=0, pooling_mode: Optional[str]='cls', gradient_checkpointing: Optional[bool]=False, low_cpu_mem_usage: Optional[bool]=False, pretrained: Optional[bool]=True):
        """
        Load a pretrained huggingface text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'microsoft/deberta-v3-base'
                    - 'bert-base-uncased'
                    - 'google/electra-base-discriminator'
                    - 'distilroberta-base'
                Multilingual backbones:
                    - 'microsoft/mdeberta-v3-base'
                    - 'xlm-roberta-base'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode for the Transformer. Can be "cls", or "mean"
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.config, self.model = get_hf_config_and_model(checkpoint_name=checkpoint_name, pretrained=pretrained, low_cpu_mem_usage=low_cpu_mem_usage)
        if isinstance(self.model, T5PreTrainedModel):
            self.is_t5 = True
            del self.model.decoder
        else:
            self.is_t5 = False
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_t5:
                self.dummy_layer = DummyLayer()
        self.out_features = self.model.config.hidden_size
        self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
        self.head.apply(init_weights)
        self.prefix = prefix
        self.pooling_mode = pooling_mode
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]
        if hasattr(self.model.config, 'type_vocab_size') and self.model.config.type_vocab_size <= 1:
            self.disable_seg_ids = True
        else:
            self.disable_seg_ids = False

    @property
    def text_token_ids_key(self):
        return f'{self.prefix}_{TEXT_TOKEN_IDS}'

    @property
    def text_segment_ids_key(self):
        return f'{self.prefix}_{TEXT_SEGMENT_IDS}'

    @property
    def text_valid_length_key(self):
        return f'{self.prefix}_{TEXT_VALID_LENGTH}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def text_column_prefix(self):
        return f'{self.text_token_ids_key}_{COLUMN}'

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        text_token_ids = batch[self.text_token_ids_key]
        if self.disable_seg_ids:
            text_segment_ids = None
        else:
            text_segment_ids = batch[self.text_segment_ids_key]
        text_valid_length = batch[self.text_valid_length_key]
        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
        if self.is_t5:
            inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
            if self.gradient_checkpointing:
                inputs_embeds = self.dummy_layer(inputs_embeds)
            outputs = self.model.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)
        else:
            outputs = self.model(input_ids=text_token_ids, token_type_ids=text_segment_ids, attention_mask=text_masks)
        if self.pooling_mode == 'cls':
            pooled_features = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_mode == 'mean':
            pooled_features = (outputs.last_hidden_state * text_masks.unsqueeze(-1)).sum(1)
            sum_mask = text_masks.unsqueeze(-1).sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-09)
            pooled_features = pooled_features / sum_mask
        else:
            raise NotImplementedError(f'Pooling mode={self.pooling_mode} is not supported.')
        logits = self.head(pooled_features)
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        column_features, column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=outputs.last_hidden_state, valid_lengths=text_valid_length, cls_feature=pooled_features)
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret.update({LOGITS: logits, FEATURES: pooled_features})
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        pre_encoder_patterns = 'embeddings', 'LayerNorm', 'wte', 'wpe', 'shared.weight', 'encoder.conv.conv', 'relative_attention_bias', 'dummy_layer'
        post_encoder_patterns = 'head', 'pooler', 'ln_f', 'final_layer_norm'
        names = [n for n, _ in self.named_parameters()]
        name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=model_prefix)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0
        return name_to_id


BBOX = 'bbox'


def lookup_mmdet_config(key, config):
    if key in config:
        return config[key]
    for subconfig in config.values():
        if isinstance(subconfig, dict):
            result = lookup_mmdet_config(key, subconfig)
            if result is not None:
                return result
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    result = lookup_mmdet_config(key, subsubconfig)
                    if result is not None:
                        return result
    return None


def update_mmdet_config(key, value, config):
    for k, subconfig in config.items():
        if key == k:
            config[k] = value
        elif isinstance(subconfig, dict):
            update_mmdet_config(key, value, subconfig)
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    update_mmdet_config(key, value, subsubconfig)


class MMDetAutoModelForObjectDetection(nn.Module):
    """
    Support MMDET object detection models.
    Refer to https://github.com/open-mmlab/mmdetection
    """

    def __init__(self, prefix: str, checkpoint_name: str, num_classes: Optional[int]=None, classes: Optional[list]=None, pretrained: Optional[bool]=True):
        """
        Load a pretrained object detector from MMdetection.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModelForObjectDetection model.
        checkpoint_name
            Name of the mmdet checkpoint.
        num_classes
            The number of classes.
        classes
            All classes in this dataset.
        pretrained
            Whether using the pretrained mmdet models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.classes = classes
        if self.classes:
            if self.num_classes:
                assert len(self.classes) == self.num_classes
            else:
                self.num_classes = len(self.classes)
        checkpoint, config_file = self._load_checkpoint_and_config()
        assert mmcv is not None, 'Please install mmcv-full by: mim install mmcv-full.'
        if isinstance(config_file, str):
            self.config = mmcv.Config.fromfile(config_file)
        if self.num_classes:
            update_mmdet_config(key='num_classes', value=self.num_classes, config=self.config)
        else:
            self.num_classes = lookup_mmdet_config(key='num_classes', config=self.config)
            if not self.num_classes:
                raise ValueError('Cannot retrieve num_classes for current model structure.')
        self.id2label = dict(zip(range(self.num_classes), range(self.num_classes)))
        assert mmdet is not None, 'Please install MMDetection by: pip install mmdet.'
        self.model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))
        if self.pretrained and checkpoint is not None:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
        if self.classes:
            self.model.CLASSES = self.classes
        elif num_classes == 20:
            warnings.simplefilter('once')
            warnings.warn(f'Using VOC classes because num_classes = {num_classes}. Provide data while init MultiModalPredictor if this is not VOC.')
            self.model.CLASSES = get_classes('voc')
        elif num_classes == 80:
            warnings.simplefilter('once')
            warnings.warn(f'Using COCO classes because num_classes = {num_classes}. Provide data while init MultiModalPredictor if this is not COCO.')
            self.model.CLASSES = get_classes('coco')
        elif 'CLASSES' in checkpoint.get('meta', {}):
            warnings.simplefilter('once')
            warnings.warn(f"Using classes provided in checkpoints: {checkpoint['meta']['CLASSES']}. Provide data while init MultiModalPredictor if this is not expected.")
            self.model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            raise ValueError('Classes need to be specified.')
        self.model.cfg = self.config
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def _load_checkpoint_and_config(self, checkpoint_name=None):
        if not checkpoint_name:
            checkpoint_name = self.checkpoint_name
        if checkpoint_name == 'faster_rcnn_r50_fpn_1x_voc0712':
            if not os.path.exists('voc_config'):
                os.makedirs('voc_config')
            checkpoint = download(url='https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth')
            config_file = download(url='https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712.py')
            download(url='https://automl-mm-bench.s3.amazonaws.com/voc_script/default_runtime.py', path='voc_config')
            download(url='https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn.py', path='voc_config')
            download(url='https://automl-mm-bench.s3.amazonaws.com/voc_script/voc0712.py', path='voc_config')
        else:
            checkpoint = download(package='mmdet', configs=[checkpoint_name], dest_root='.')[0]
            config_file = checkpoint_name + '.py'
        return checkpoint, config_file

    def dump_config(self, path):
        self.config.dump(path)
        self.name_to_id = self.get_layer_ids()

    @property
    def image_key(self):
        return f'{self.prefix}_{IMAGE}'

    @property
    def image_valid_num_key(self):
        return f'{self.prefix}_{IMAGE_VALID_NUM}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def image_column_prefix(self):
        return f'{self.image_key}_{COLUMN}'

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with bounding boxes.
        """
        logger.warning('MMDetAutoModelForObjectDetection.forward() is deprecated since it does not support multi gpu.')
        data = batch[self.image_key]
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        device = next(self.model.parameters()).device
        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(m, RoIPool), 'CPU inference with RoIPool is not supported currently.'
        results = self.model(return_loss=False, rescale=True, **data)
        ret = {BBOX: results}
        return {self.prefix: ret}

    def forward_test(self, imgs, img_metas, rescale=True):
        return self.model.forward_test(imgs=imgs, img_metas=img_metas, rescale=rescale)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        return self.model.forward_train(img=img, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

    def _parse_losses(self, losses):
        return self.model._parse_losses(losses)

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Setting all layers as the same id 0 for now.
        TODO: Need to investigate mmdetection's model definitions
        Currently only head to 0 others to 1.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        registered_head_layers_patterns = ['bbox_head.fc_cls', 'bbox_head.fc_reg', 'bbox_head.convs_pred', 'bbox_head.cls_branches', 'bbox_head.multi_level_conv_cls', 'bbox_head.vfnet_cls', 'bbox_head.heatmap_head', 'bbox_head.atss_cls', 'bbox_head.cls_convs']
        default_head_layers_patterns = ['bbox_head']
        head_registered = False
        for n, _ in self.named_parameters():
            name_to_id[n] = 1
            for pattern in registered_head_layers_patterns:
                if pattern in n:
                    name_to_id[n] = 0
                    head_registered = True
        if not head_registered:
            for n, _ in self.named_parameters():
                name_to_id[n] = 1
                for pattern in default_head_layers_patterns:
                    if pattern in n:
                        name_to_id[n] = 0
        return name_to_id


class MMOCRAutoModelForTextDetection(nn.Module):
    """
    Support MMOCR object detection models.
    Refer to https://github.com/open-mmlab/mmocr
    """

    def __init__(self, prefix: str, checkpoint_name: str, num_classes: Optional[int]=None, pretrained: Optional[bool]=True):
        """
        Load a pretrained ocr text detection detector from MMOCR.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModelForTextDetection model.
        checkpoint_name
            Name of the mmdet checkpoint.
        num_classes
            The number of classes.
        pretrained
            Whether using the pretrained mmdet models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.pretrained = pretrained
        self.config, self.model = get_mmocr_config_and_model(checkpoint_name)
        self.model = revert_sync_batchnorm(self.model)
        self.model.cfg = self.config
        self.prefix = prefix

    @property
    def image_key(self):
        return f'{self.prefix}_{IMAGE}'

    @property
    def image_valid_num_key(self):
        return f'{self.prefix}_{IMAGE_VALID_NUM}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def image_column_prefix(self):
        return f'{self.image_key}_{COLUMN}'

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with bounding boxes.
        """
        data = batch[self.image_key]
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        device = next(self.model.parameters()).device
        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        results = self.model(return_loss=False, rescale=True, **data)
        ret = {BBOX: results[0]['boundary_result']}
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Setting all layers as the same id 0 for now.
        TODO: Need to investigate mmocr's model definitions

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


SCORE = 'score'


TEXT = 'text'


class MMOCRAutoModelForTextRecognition(nn.Module):
    """
    Support MMOCR text recognition models.
    Refer to https://github.com/open-mmlab/mmocr
    """

    def __init__(self, prefix: str, checkpoint_name: str, num_classes: Optional[int]=None, pretrained: Optional[bool]=True):
        """
        Load a pretrained ocr text recognition detector from MMOCR.

        Parameters
        ----------
        prefix
            The prefix of the MMdetAutoModelForTextRecognition model.
        checkpoint_name
            Name of the mmdet checkpoint.
        num_classes
            The number of classes.
        pretrained
            Whether using the pretrained mmocr models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.pretrained = pretrained
        self.config, self.model = get_mmocr_config_and_model(checkpoint_name)
        self.model = revert_sync_batchnorm(self.model)
        self.model.cfg = self.config
        self.prefix = prefix

    @property
    def image_key(self):
        return f'{self.prefix}_{IMAGE}'

    @property
    def image_valid_num_key(self):
        return f'{self.prefix}_{IMAGE_VALID_NUM}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def image_column_prefix(self):
        return f'{self.image_key}_{COLUMN}'

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with bounding boxes.
        """
        data = batch[self.image_key]
        if isinstance(data['img_metas'], List):
            data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        else:
            data['img_metas'] = data['img_metas'].data
        if isinstance(data['img'], List):
            data['img'] = [img.data[0] for img in data['img']]
        else:
            data['img'] = data['img'].data
        device = next(self.model.parameters()).device
        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        results = self.model(return_loss=False, rescale=True, **data)
        ret = {TEXT: results[0]['text'], SCORE: results[0]['score']}
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Setting all layers as the same id 0 for now.
        TODO: Need to investigate mmocr's model definitions

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


class HFAutoModelForNER(HFAutoModelForTextPrediction):
    """
    Named entity recognition with huggingface backbones. Inherit from HFAutoModelForTextPrediction.
    """

    def __init__(self, prefix: str, checkpoint_name: str='microsoft/deberta-v3-base', num_classes: Optional[int]=0, pooling_mode: Optional[str]='cls', gradient_checkpointing: Optional[bool]=False, low_cpu_mem_usage: Optional[bool]=False, pretrained: Optional[bool]=True):
        """
        Load a pretrained huggingface text transformer backbone.
        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'bert-base-cased'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode to be used, it is not used in the NER task.
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__(prefix=prefix, checkpoint_name=checkpoint_name, num_classes=num_classes, pooling_mode=pooling_mode, gradient_checkpointing=gradient_checkpointing, low_cpu_mem_usage=low_cpu_mem_usage, pretrained=pretrained)
        logger.debug(f'initializing {checkpoint_name}')
        if self.config.model_type in {'gpt2', 'roberta'}:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        if hasattr(self.model.config, 'max_position_embeddings'):
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        if hasattr(self.model.config, 'n_positions'):
            self.tokenizer.model_max_length = self.model.config.n_positions

    @property
    def text_token_word_mapping_key(self):
        return f'{self.prefix}_{TOKEN_WORD_MAPPING}'

    @property
    def text_word_offsets_key(self):
        return f'{self.prefix}_{WORD_OFFSETS}'

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        text_token_ids = batch[self.text_token_ids_key]
        if self.disable_seg_ids:
            text_segment_ids = None
        else:
            text_segment_ids = batch[self.text_segment_ids_key]
        text_valid_length = batch[self.text_valid_length_key]
        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
        if self.is_t5:
            inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
            outputs = self.model.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)
        else:
            outputs = self.model(input_ids=text_token_ids, token_type_ids=text_segment_ids, attention_mask=text_masks)
        sequence_output = outputs.last_hidden_state
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        pooled_features = outputs.last_hidden_state[:, 0, :]
        logits = self.head(sequence_output)
        logits_label = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        column_features, column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=outputs.last_hidden_state, valid_lengths=text_valid_length, cls_feature=pooled_features)
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret.update({LOGITS: logits, FEATURES: sequence_output, NER_ANNOTATION: logits_label, TOKEN_WORD_MAPPING: batch[self.text_token_word_mapping_key], WORD_OFFSETS: batch[self.text_word_offsets_key]})
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        pre_encoder_patterns = 'embeddings', 'LayerNorm', 'wte', 'wpe', 'shared.weight', 'encoder.conv.conv', 'relative_attention_bias', 'dummy_layer', 'mask_emb', 'word_embedding.weight'
        post_encoder_patterns = 'head', 'pooler', 'ln_f', 'final_layer_norm'
        names = [n for n, _ in self.named_parameters()]
        name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=model_prefix)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0
        return name_to_id


NUMERICAL = 'numerical'


class NLinear(nn.Module):

    def __init__(self, n: int, d_in: int, d_out: int, bias: bool=True):
        super().__init__()
        self.weight = nn.Parameter(Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3, 'Error input dimension, should be 3, but given {}'.format(x.ndim)
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NumericalFeatureTokenizer(nn.Module):
    """
    Numerical tokenizer for numerical features in tabular data.
    It transforms the input numerical features to tokens (embeddings).

    The numerical features usually refers to continuous features.

    It consists of two steps:
        1. each feature is multiplied by a trainable vector i.e., weights,
        2. another trainable vector is added i.e., bias.

    Note that each feature has its separate pair of trainable vectors,
    i.e. the vectors are not shared between features.
    """

    def __init__(self, in_features: int, d_token: int, bias: Optional[bool]=True, initialization: Optional[str]='normal'):
        """
        Parameters
        ----------
        in_features:
            Dimension of input features i.e. the number of continuous (scalar) features
        d_token:
            The size of one token.
        bias:
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization:
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`.

        References
        ----------
        Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(in_features, d_token))
        self.bias = nn.Parameter(Tensor(in_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) ->Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class AutoDis(nn.Module):
    """
    Paper (the version is important): https://arxiv.org/abs/2012.08986v2
    Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    We borrow the implementations from: https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/train4.py
    The paper is significantly different from the code (it looks like the code
    implements the first version of the paper). We implement the second version
    here. Not all technical details are given for the second version, so what we do
    here can be different from what authors actually did.
    Anyway, AutoDis (v2) is essentially the following sequence of layers (applied from
    left to right): [Linear(no bias), LeakyReLU, Linear(no bias), Softmax, Linear]
    """

    def __init__(self, in_features: int, d_embedding: int, n_meta_embeddings: int, temperature: Optional[float]=3.0):
        super().__init__()
        self.first_layer = NumericalFeatureTokenizer(in_features=in_features, d_token=n_meta_embeddings, bias=False, initialization='uniform')
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(in_features, n_meta_embeddings, n_meta_embeddings, False)
        self.softmax = nn.Softmax(-1)
        self.temperature = temperature
        self.third_layer = NLinear(in_features, n_meta_embeddings, d_embedding, False)
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: Tensor):
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x / self.temperature)
        x = self.third_layer(x)
        return x


class NLayerNorm(nn.Module):

    def __init__(self, n_features: int, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_features, d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x: Tensor):
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


class NLinearMemoryEfficient(nn.Module):

    def __init__(self, n: int, d_in: int, d_out: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class Periodic(nn.Module):

    def __init__(self, in_features: int, d_embedding: int, trainable: Optional[bool]=True, initialization: Optional[str]='normal', sigma: Optional[float]=1.0):
        """
        Parameters
        ----------
        in_features
            Input feature size.
        d_embedding
            Output feature size, should be an even number.
        trainable
            Determine whether the coefficients needed to be updated.
        initialization
            Initialization scheme.
        sigma
            Standard deviation used for initialization='normal'

        Reference:
        ----------
        1. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        """
        super().__init__()
        assert d_embedding % 2 == 0, 'd_embedding mod 2 should be 0, current d_embedding is {}'.format(d_embedding)
        if initialization == 'log-linear':
            coefficients = sigma ** (torch.arange(d_embedding // 2) / (d_embedding // 2))
            coefficients = coefficients[None].repeat(in_features, 1)
        elif initialization == 'normal':
            coefficients = torch.normal(0.0, sigma, (in_features, d_embedding // 2))
        if trainable:
            self.coefficients = nn.Parameter(coefficients)
        else:
            self.register_buffer('coefficients', coefficients)

    def cos_sin(self, x: Tensor):
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

    def forward(self, x: Tensor):
        assert x.ndim == 2, 'Periodic should only be applied to first layer i.e. ndim==2'
        return self.cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


class NumEmbeddings(nn.Module):

    def __init__(self, in_features: int, embedding_arch: List[str], d_embedding: Optional[int]=None, memory_efficient: Optional[bool]=False):
        """
        Parameters
        ----------
        in_features
            Input feature size.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
                {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'leaky_relu', 'layernorm'}
            To use the embedding schemes summarized in Table 3 of 'On Embeddings for Numerical Features in Tabular Deep Learning' (https://arxiv.org/abs/2203.05556)
            By setting the embedding_arch as follows:
                1. `L`: ['linear']
                2. `LR`: ['linear', 'relu']
                3. `LRLR`: ['linear', 'relu', 'linear', 'relu']
                4. `P`: ['positional']
                5. `PL`: ['positional', 'linear']
                6. `PLR`: ['positional', 'linear', 'relu']
                7. `PLRLR`: ['positional', 'linear', 'relu', 'linear', 'relu']
                8. `AutoDis`: ['autodis']
                9. `Leaky Gates` in [ref.3]: ['linear', 'leaky_relu']
            Notably, in `L` (i.e. embedding_arch=['linear']) for numerical transformer,
            it identical as the original feature_tokenzier in FT_Transformer (c.f. Figure 2.a in https://arxiv.org/pdf/2106.11959.pdf).
        d_embedding:
            Dimension of the embeddings.
            The output shape should be [batch_size, number_of_numerical_featurs, d_embedding]
        memory_efficient:
            Use efficient linear layer scheme if True. Default is False.

        Reference:
        ----------
        1. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        3. Paper: Simple Modifications to Improve Tabular Neural Networks: https://arxiv.org/pdf/2108.03214
        """
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'leaky_relu', 'layernorm'}
        if any(x in embedding_arch for x in ['linear', 'shared_linear', 'autodis']):
            assert d_embedding is not None
        assert embedding_arch.count('positional') <= 1
        if 'autodis' in embedding_arch:
            embedding_arch = ['autodis']
        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: list[nn.Module] = []
        if embedding_arch[0] == 'linear':
            layers.append(NumericalFeatureTokenizer(in_features=in_features, d_token=d_embedding, bias=True, initialization='normal'))
        elif embedding_arch[0] == 'positional':
            layers.append(Periodic(in_features=in_features, d_embedding=d_embedding, trainable=True, initialization='normal', sigma=1.0))
        elif embedding_arch[0] == 'autodis':
            layers.append(AutoDis(in_features=in_features, d_embedding=d_embedding, n_meta_embeddings=d_embedding, temperature=3.0))
        else:
            layers.append(nn.Identity())
        for x in embedding_arch[1:]:
            layers.append(nn.ReLU() if x == 'relu' else nn.LeakyReLU() if x == 'leaky_relu' else NLinear_(in_features, d_embedding, d_embedding) if x == 'linear' else nn.Linear(d_embedding, d_embedding) if x == 'shared_linear' else NLayerNorm(in_features, d_embedding) if x == 'layernorm' else nn.Identity())
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_embedding
        self.in_features = in_features
        self.layers = nn.Sequential(*layers)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        y = self.forward(torch.ones(1, self.in_features))
        return y.shape[1]

    @property
    def d_token(self) ->int:
        """The size of one token."""
        y = self.forward(torch.ones(1, self.in_features))
        return y.shape[-1]

    def forward(self, x):
        return self.layers(x)


class NumericalMLP(nn.Module):
    """
    MLP for numerical input.
    """

    def __init__(self, prefix: str, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, num_layers: Optional[int]=1, activation: Optional[str]='leaky_relu', dropout_prob: Optional[float]=0.5, normalization: Optional[str]='layer_norm', num_classes: Optional[int]=0, d_token: Optional[int]=8, embedding_arch: Optional[List[str]]=None):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        in_features
            Dimension of input features.
        hidden_features
            Dimension of hidden features.
        out_features
            Dimension of output features.
        num_layers
            Number of MLP layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        num_classes
            Number of classes. 1 for a regression task.
        d_token
            The size of one token for `NumericalEmbedding`.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
            {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}
        """
        super().__init__()
        self.out_features = out_features
        self.numerical_feature_tokenizer = NumEmbeddings(in_features=in_features, d_embedding=d_token, embedding_arch=embedding_arch) if embedding_arch is not None else nn.Identity()
        in_features = in_features * d_token if embedding_arch is not None else in_features
        self.mlp = MLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features, num_layers=num_layers, activation=activation, dropout_prob=dropout_prob, normalization=normalization)
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(init_weights)
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def numerical_key(self):
        return f'{self.prefix}_{NUMERICAL}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: dict):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        features = self.numerical_feature_tokenizer(batch[self.numerical_key])
        features = features.flatten(1, 2) if features.ndim == 3 else features
        features = self.mlp(features)
        logits = self.head(features)
        return {self.prefix: {LOGITS: logits, FEATURES: features}}

    def get_layer_ids(self):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


class NumericalTransformer(nn.Module):
    """
    FT-Transformer for numerical tabular features.
    """

    def __init__(self, prefix: str, in_features: int, d_token: int, cls_token: Optional[bool]=False, out_features: Optional[int]=None, num_classes: Optional[int]=0, token_initialization: Optional[str]='normal', n_blocks: Optional[int]=0, attention_n_heads: Optional[int]=8, attention_initialization: Optional[str]='kaiming', attention_normalization: Optional[str]='layer_norm', attention_dropout: Optional[str]=0.2, residual_dropout: Optional[str]=0.0, ffn_activation: Optional[str]='reglu', ffn_normalization: Optional[str]='layer_norm', ffn_d_hidden: Optional[str]=192, ffn_dropout: Optional[str]=0.0, prenormalization: Optional[bool]=True, first_prenormalization: Optional[bool]=False, kv_compression_ratio: Optional[float]=None, kv_compression_sharing: Optional[str]=None, head_activation: Optional[str]='relu', head_normalization: Optional[str]='layer_norm', embedding_arch: Optional[List[str]]=['linear'], additive_attention: Optional[bool]=False, share_qv_weights: Optional[bool]=False):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        in_features
            Dimension of input features.
        d_token
            The size of one token for `NumericalEmbedding`.
        cls_token
            If `True`, [cls] token will be added to the token embeddings.
        out_features
            Dimension of output features.
        num_classes
            Number of classes. 1 for a regression task.
        token_bias
            If `True`, for each feature, an additional trainable vector will be added in `_CategoricalFeatureTokenizer`
            to the embedding regardless of feature value. Notably, the bias are not shared between features.
        token_initialization
            Initialization policy for parameters in `_CategoricalFeatureTokenizer` and `_CLSToke`.
            Must be one of `['uniform', 'normal']`.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
            {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.

        References
        ----------
        1. Paper: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021 https://arxiv.org/pdf/2106.11959.pdf
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        3. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        4. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        """
        super().__init__()
        assert d_token > 0, 'd_token must be positive'
        assert n_blocks >= 0, 'n_blocks must be non-negative'
        assert attention_n_heads > 0, 'attention_n_heads must be positive'
        assert token_initialization in ['uniform', 'normal'], 'initialization must be uniform or normal'
        self.prefix = prefix
        self.out_features = out_features
        self.numerical_feature_tokenizer = NumEmbeddings(in_features=in_features, d_embedding=d_token, embedding_arch=embedding_arch)
        self.cls_token = CLSToken(d_token=d_token, initialization=token_initialization) if cls_token else nn.Identity()
        if kv_compression_ratio is not None:
            if self.cls_token:
                n_tokens = self.numerical_feature_tokenizer.n_tokens + 1
            else:
                n_tokens = self.numerical_feature_tokenizer.n_tokens
        else:
            n_tokens = None
        self.transformer = FT_Transformer(d_token=d_token, n_blocks=n_blocks, attention_n_heads=attention_n_heads, attention_dropout=attention_dropout, attention_initialization=attention_initialization, attention_normalization=attention_normalization, ffn_d_hidden=ffn_d_hidden, ffn_dropout=ffn_dropout, ffn_activation=ffn_activation, ffn_normalization=ffn_normalization, residual_dropout=residual_dropout, prenormalization=prenormalization, first_prenormalization=first_prenormalization, last_layer_query_idx=None, n_tokens=n_tokens, kv_compression_ratio=kv_compression_ratio, kv_compression_sharing=kv_compression_sharing, head_activation=head_activation, head_normalization=head_normalization, d_out=out_features, additive_attention=additive_attention, share_qv_weights=share_qv_weights)
        self.head = FT_Transformer.Head(d_in=d_token, d_out=num_classes, bias=True, activation=head_activation, normalization=head_normalization if prenormalization else 'Identity')
        self.name_to_id = self.get_layer_ids()

    @property
    def numerical_key(self):
        return f'{self.prefix}_{NUMERICAL}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    def forward(self, batch: dict):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        features = self.numerical_feature_tokenizer(batch[self.numerical_key])
        features = self.cls_token(features)
        features = self.transformer(features)
        logits = self.head(features)
        return {self.prefix: {LOGITS: logits, FEATURES: features}}

    def get_layer_ids(self):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id


CHOICES_IDS = 'choices_ids'


LM_TARGET = 'lm_target'


TEMPLATE_LOGITS = 'template_logits'


@lru_cache(None)
def warn_once(logger, msg: str):
    logger.warning(msg)


class TFewModel(nn.Module):
    """
    Implementation of T-Few (https://arxiv.org/pdf/2205.05638.pdf).
    Refer to https://github.com/r-three/t-few
    """

    def __init__(self, prefix: str, checkpoint_name: str='bigscience/T0_3B', num_classes: Optional[int]=0, length_norm: float=1.0, unlikely_loss: float=1.0, mc_loss: float=1.0, gradient_checkpointing: Optional[bool]=False, low_cpu_mem_usage: Optional[bool]=False, pretrained: Optional[bool]=True):
        """
        Load a pretrained T5-based text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading T5ForConditionalGeneration checkpoints from
            Huggingface Models list: https://huggingface.co/models.
            We recommend using T0 backbones. For example, you may use
                - 'bigscience/T0_3B'
                - 'bigscience/T0p'
                - 'bigscience/T0pp'
        num_classes
            The number of classes. 1 for a regression task.
        length_norm
             Normalizes length to adjust for length bias in target template
        unlikely_loss
            Adds loss term that lowers probability of incorrect outputs
        mc_loss
            Adds multiple choice cross entropy loss
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.config = AutoConfig.from_pretrained(checkpoint_name)
        if pretrained:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage)
        else:
            self.model = AutoModelForSeq2SeqLM.from_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        self.eos_token = self.tokenizer.eos_token
        self.out_features = self.model.config.hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.dummy_layer = DummyLayer()
        self.prefix = prefix
        self.mc_loss = mc_loss
        self.unlikely_loss = unlikely_loss
        self.length_norm = length_norm
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def text_token_ids_key(self):
        return f'{self.prefix}_{TEXT_TOKEN_IDS}'

    @property
    def text_segment_ids_key(self):
        return f'{self.prefix}_{TEXT_SEGMENT_IDS}'

    @property
    def text_valid_length_key(self):
        return f'{self.prefix}_{TEXT_VALID_LENGTH}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def choices_key(self):
        return f'{self.prefix}_{CHOICES_IDS}'

    @property
    def text_column_prefix(self):
        return f'{self.text_token_ids_key}_{COLUMN}'

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        if not batch[self.choices_key].numel():
            warn_once(logger, msg="No target choices found in batch. Ensure that 'data.templates_turn_on=True' and that a valid preset or custom templates are provided.")
            warn_once(logger, msg='Fallback to numerical representation of classes...')
            batch[self.choices_key] = self.tokenizer([str(i) for i in range(self.num_classes)], return_tensors='pt', padding=True)['input_ids'].repeat(batch[self.text_token_ids_key].size(0), 1, 1)
        assert batch[self.choices_key].size(1) == self.num_classes, f'Number of target choices is different from number of classes, but they must be the same. Please check template.'
        text_token_ids = batch[self.text_token_ids_key]
        bs = text_token_ids.size(0)
        choices_ids = batch[self.choices_key]
        bs, num_choices = choices_ids.size()[:2]
        flat_choices_ids = choices_ids.flatten(0, 1)
        text_valid_length = batch[self.text_valid_length_key]
        text_masks = (text_token_ids != self.tokenizer.pad_token_id).float()
        inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
        if self.gradient_checkpointing:
            inputs_embeds = self.dummy_layer(inputs_embeds)
        encoder_hidden_states_or = self.model.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)[0]
        encoder_hidden_states = encoder_hidden_states_or.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
        attention_mask = text_masks.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
        decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
        decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
        model_output = self.model(attention_mask=attention_mask, encoder_outputs=[encoder_hidden_states], decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        model_output = model_output.logits
        target_template_logits = model_output
        lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()
        choices_scores = F.cross_entropy(target_template_logits.flatten(0, 1), lm_target.flatten(0, 1), reduction='none').view(bs, num_choices, -1).sum(dim=-1)
        if self.length_norm > 0:
            choices_scores = choices_scores / torch.pow((choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.length_norm)
        choices_scores = -choices_scores
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        column_features, column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.text_column_prefix, features=model_output, valid_lengths=text_valid_length)
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret.update({LOGITS: choices_scores, TEMPLATE_LOGITS: target_template_logits, LM_TARGET: lm_target, FEATURES: encoder_hidden_states_or[:, 0, :]})
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        pre_encoder_patterns = 'embeddings', 'LayerNorm', 'wte', 'wpe', 'shared.weight', 'encoder.conv.conv', 'dummy_layer'
        post_encoder_patterns = 'head', 'pooler', 'ln_f', 'final_layer_norm'
        names = [n for n, _ in self.named_parameters()]
        name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=model_prefix)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 1
        for name, id in name_to_id.items():
            if id == 0:
                name_to_id[name] = 1
        return name_to_id


CATEGORICAL_MLP = 'categorical_mlp'


CATEGORICAL_TRANSFORMER = 'categorical_transformer'


CLIP = 'clip'


FUSION = 'fusion'


FUSION_MLP = f'{FUSION}_mlp'


NER = 'ner'


FUSION_NER = f'{FUSION}_{NER}'


FUSION_TRANSFORMER = f'{FUSION}_transformer'


HF_TEXT = 'hf_text'


MMDET_IMAGE = 'mmdet_image'


MMOCR_TEXT_DET = 'mmocr_text_detection'


MMOCR_TEXT_RECOG = 'mmocr_text_recognition'


NUMERICAL_MLP = 'numerical_mlp'


NUMERICAL_TRANSFORMER = 'numerical_transformer'


TIMM_IMAGE = 'timm_image'


T_FEW = 't_few'


def get_model_head(model: nn.Module):
    """
    Return the model's head. Different models may have different head names.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    The model's head.
    """
    if hasattr(model, 'head'):
        head = model.head
    elif hasattr(model, 'last_linear'):
        head = model.last_linear
    elif hasattr(model, 'fc'):
        head = model.fc
    elif hasattr(model, 'classifier'):
        head = model.classifier
    else:
        raise ValueError(f"Model {type(model)} doesn't have head. Need to check its implementation.")
    return head.fc if hasattr(head, 'fc') else head


class TimmAutoModelForImagePrediction(nn.Module):
    """
    Support TIMM image backbones.
    Refer to https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, prefix: str, checkpoint_name: str, num_classes: Optional[int]=0, mix_choice: Optional[str]='all_logits', pretrained: Optional[bool]=True):
        """
        Load a pretrained image backbone from TIMM.

        Parameters
        ----------
        prefix
            The prefix of the TimmAutoModelForImagePrediction model.
        checkpoint_name
            Name of the timm checkpoint.
        num_classes
            The number of classes. 1 for a regression task.
        mix_choice
            Choice used for mixing multiple images. We now support.
            - all_images
                The images are directly averaged and passed to the model.
            - all_logits
                The logits output from individual images are averaged to generate the final output.
        pretrained
            Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f'initializing {checkpoint_name}')
        self.checkpoint_name = checkpoint_name
        self.pretrained = pretrained
        self.model = create_model(checkpoint_name, pretrained=pretrained, num_classes=num_classes)
        self.config = self.model.default_cfg
        self.num_classes = self.model.num_classes
        self.out_features = self.model.num_features
        self.head = get_model_head(model=self.model)
        self.model.reset_classifier(0)
        self.mix_choice = mix_choice
        logger.debug(f'mix_choice: {mix_choice}')
        self.prefix = prefix
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def image_key(self):
        return f'{self.prefix}_{IMAGE}'

    @property
    def image_valid_num_key(self):
        return f'{self.prefix}_{IMAGE_VALID_NUM}'

    @property
    def label_key(self):
        return f'{self.prefix}_{LABEL}'

    @property
    def image_column_prefix(self):
        return f'{self.image_key}_{COLUMN}'

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def forward(self, batch: dict):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        images = batch[self.image_key]
        image_valid_num = batch[self.image_valid_num_key]
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if self.mix_choice == 'all_images':
            mixed_images = images.sum(dim=1) / torch.clamp(image_valid_num, min=1e-06)[:, None, None, None]
            features = self.model(mixed_images)
            if self.num_classes > 0:
                logits = self.head(features)
        elif self.mix_choice == 'all_logits':
            b, n, c, h, w = images.shape
            features = self.model(images.reshape((b * n, c, h, w)))
            if self.num_classes > 0:
                logits = self.head(features)
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = (steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))).type_as(features)
            features = features.reshape((b, n, -1)) * image_masks[:, :, None]
            column_features, column_feature_masks = get_column_features(batch=batch, column_name_prefix=self.image_column_prefix, features=features, valid_lengths=image_valid_num)
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
            features = features.sum(dim=1) / torch.clamp(image_valid_num, min=1e-06)[:, None]
            if self.num_classes > 0:
                logits = logits.reshape((b, n, -1)) * image_masks[:, :, None]
                logits = logits.sum(dim=1) / torch.clamp(image_valid_num, min=1e-06)[:, None]
        else:
            raise ValueError(f'unknown mix_choice: {self.mix_choice}')
        ret[FEATURES] = features
        if self.num_classes > 0:
            ret[LOGITS] = logits
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Due to different backbone architectures in TIMM, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = 'model'
        pre_encoder_patterns = 'embed', 'cls_token', 'stem', 'bn1', 'conv1'
        post_encoder_patterns = 'head', 'norm', 'bn2'
        names = [n for n, _ in self.named_parameters()]
        name_to_id, names = assign_layer_ids(names=names, pre_encoder_patterns=pre_encoder_patterns, post_encoder_patterns=post_encoder_patterns, model_pre=model_prefix)
        if len(names) > 0:
            logger.debug(f'outer layers are treated as head: {names}')
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0
        return name_to_id


class RKDLoss(nn.Module):
    """
    Compute RKD Distance Loss.
    Paper Refer to: Relational Knowledge Disitllation, CVPR2019. https://arxiv.org/abs/1904.05068
    Code Refer to: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/RKD.py
    and https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    """

    def __init__(self, distance_loss_weight: Optional[float]=25.0, angle_loss_weight: Optional[float]=50.0):
        """
        Parameters
        ----------
        distance_loss_weight
            Weight of RKD distance loss
        angle_loss_weight
            Weight of RKD angle loss
        Returns
        -------
        """
        super(RKDLoss, self).__init__()
        self.distance_loss_weight = distance_loss_weight
        self.angle_loss_weight = angle_loss_weight

    def forward(self, feature_student: Optional[torch.Tensor], feature_teacher: Optional[torch.Tensor]):
        """
        Parameters
        ----------
        feature_student
            Output feature of student model, shape: (N, D)
        feature_teacher
            Output feature of teacher model, shape: (N, D)
        Returns
        -------
        The RKD Loss between teacher and student
        """
        if self.distance_loss_weight > 0:
            with torch.no_grad():
                t_dist = self.pdist(feature_teacher, squared=False)
                mean_td = t_dist[t_dist > 0].mean()
                t_dist = t_dist / mean_td
            s_dist = self.pdist(feature_student, squared=False)
            mean_d = s_dist[s_dist > 0].mean()
            s_dist = s_dist / mean_d
            loss_distance = F.smooth_l1_loss(s_dist, t_dist)
        if self.angle_loss_weight > 0:
            with torch.no_grad():
                td = feature_teacher.unsqueeze(0) - feature_teacher.unsqueeze(1)
                norm_td = F.normalize(td, p=2, dim=2)
                t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
            sd = feature_student.unsqueeze(0) - feature_student.unsqueeze(1)
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
            loss_angle = F.smooth_l1_loss(s_angle, t_angle)
        loss = (self.distance_loss_weight * loss_distance if self.distance_loss_weight > 0 else 0) + (self.angle_loss_weight * loss_angle if self.angle_loss_weight > 0 else 0)
        return loss

    @staticmethod
    def pdist(embeddings: Optional[torch.Tensor], squared: Optional[bool]=False, eps: Optional[float]=1e-12):
        """
        Compute pairwise Euclidean distances between embeddings in n-dimensional space.

        Parameters
        ----------
        embeddings
            The embeddings to compute pairwise distance between. Shape: (N,D)
        squared
            If the result is square of Euclidean distance.
        eps
            Min value of each entry.

        Returns
        -------
        Pairwise Euclidean distances. Shape: (N,N)
        """
        e_square = embeddings.pow(2).sum(dim=1)
        prod = embeddings @ embeddings.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(embeddings)), range(len(embeddings))] = 0
        return res


class SoftTargetCrossEntropy(nn.Module):
    """
    The soft target CrossEntropy from timm.
    https://github.com/rwightman/pytorch-image-models/blob/e4360e6125bb0bb4279785810c8eb33b40af3ebd/timm/loss/cross_entropy.py
    It works under the mixup.
    It can calculate the crossentropy of input and label with one-hot.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(input, dim=-1), dim=-1)
        return loss.mean()


def gather_features(image_features, text_features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False):
    """
    Gather features across GPUs.

    Parameters
    ----------
    image_features
        image features of the current process.
    text_features
        text features of the current process.
    local_loss
        If False, make sure the features on the current GPU have gradients.
    gather_with_grad
        Whether to gather all features with gradients enabled.
    rank
        Rank of the current process (it should be a number between 0 and world_size-1).
    world_size
        Number of processes participating in the job.
    use_horovod
        Whether to use horovod.

    Returns
    -------
    Gathered image and text features from all processes.
    """
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    elif gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features


class MultiNegativesSoftmaxLoss(nn.Module):
    """
    This loss expects as input a batch consisting of pairs (a_1, p_1), (a_2, p_2)…, (a_n, p_n) where
        we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i,
        we have 1 positive example (p_i) and n-1 negative examples (p_j).
        It then minimizes the negative log-likehood for softmax normalized scores.
        It can also support gather negatives across processes.
    """

    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, use_horovod=False):
        """
        Parameters
        ----------
        local_loss
            Whether to compute the loss only for the current process's samples.
        gather_with_grad
            Whether to gather all features with gradients enabled.
        cache_labels
            Whether to cache labels for loss in next iterations.
        use_horovod
            Whether to use horovod.
        """
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, features_a, features_b, logit_scale, rank=0, world_size=1):
        device = features_a.device
        if world_size > 1:
            all_features_a, all_features_b = gather_features(features_a, features_b, self.local_loss, self.gather_with_grad, rank, world_size, self.use_horovod)
            if self.local_loss:
                logits_per_a = logit_scale * features_a @ all_features_b.T
                logits_per_b = logit_scale * features_b @ all_features_a.T
            else:
                logits_per_a = logit_scale * all_features_a @ all_features_b.T
                logits_per_b = logits_per_a.T
        else:
            logits_per_a = logit_scale * features_a @ features_b.T
            logits_per_b = logit_scale * features_b @ features_a.T
        num_logits = logits_per_a.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (F.cross_entropy(logits_per_a, labels) + F.cross_entropy(logits_per_b, labels)) / 2
        return total_loss


class HuberPinballLoss(nn.Module):
    __name__ = 'huber_pinball_loss'

    def __init__(self, quantile_levels, alpha=0.01):
        super(HuberPinballLoss, self).__init__()
        if quantile_levels is not None:
            self.quantile_levels = torch.Tensor(quantile_levels).contiguous().reshape(1, -1)
        else:
            self.quantile_levels = None
        self.alpha = alpha

    def forward(self, predict_data, target_data):
        if self.quantile_levels is None:
            return None
        target_data = target_data.contiguous().reshape(-1, 1)
        batch_size = target_data.size()[0]
        predict_data = predict_data.contiguous().reshape(batch_size, -1)
        error_data = target_data - predict_data
        if self.alpha == 0.0:
            loss_data = torch.max(self.quantile_levels * error_data, (self.quantile_levels - 1) * error_data)
        else:
            loss_data = torch.where(torch.abs(error_data) < self.alpha, 0.5 * error_data * error_data, self.alpha * (torch.abs(error_data) - 0.5 * self.alpha))
            loss_data = loss_data / self.alpha
            scale = torch.where(error_data >= 0, torch.ones_like(error_data) * self.quantile_levels, torch.ones_like(error_data) * (1 - self.quantile_levels))
            loss_data *= scale
        return loss_data.mean()


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))


class TransformerEncoderLayerModified(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, n_cat_embeddings, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_cat_embeddings, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]=None, src_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


REGRESSION = 'regression'


class SupervisedPretext(nn.Module):

    def __init__(self, problem_type, device):
        super().__init__()
        self.device = device
        self.loss_funct = nn.MSELoss() if problem_type == REGRESSION else nn.CrossEntropyLoss()

    def forward(self, out, target):
        loss = self.loss_funct(out, target)
        pred = out.max(dim=1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        correct = correct.float()
        correct = correct / out.shape[0]
        return loss, correct

    def get(self, data, target):
        data = data
        target = target
        return data, target


class BERTPretext(nn.Module):
    """
    This is the current default pretext task module.

    Functionality:

        self.get:
            inputs: (data, target) (target will often be None)
            outputs: (pretext_data, pretext_target)

            called before the forward pass through the TabTransformer to create the
            input and label for a BERT-style pretext task.

        self.forward:
            inputs: out (embedding for TabTransformer), target (pretext task label)
            outputs: loss, % accuracy on pretext task

            given the embedding it passes it through a classifier which learns to
            predict the pretext label.
    """

    def __init__(self, cat_feat_origin_cards, device, hidden_dim, replacement_noise='random', p_replace=0.3):
        super().__init__()
        self.cat_feat_origin_cards = cat_feat_origin_cards
        self.device = device
        self.hidden_dim = hidden_dim
        self.loss_funct = nn.CrossEntropyLoss()
        self.p_replace = p_replace
        self.predicters = nn.ModuleList()
        self.n_cat_feats = len(cat_feat_origin_cards)
        self.replacement_noise = replacement_noise
        for col in range(self.n_cat_feats):
            lin = nn.Linear(self.hidden_dim, 2)
            self.predicters.append(lin)

    def forward(self, out, target):
        prob = torch.cat([self.predicters[col](out[:, col, :]).unsqueeze(1) for col in range(self.n_cat_feats)], dim=1)
        prob = prob.view(-1, 2)
        target = target.view(-1)
        loss = self.loss_funct(prob, target)
        pred = prob.max(dim=1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        correct = correct.float()
        correct = correct / pred.shape[0]
        return loss, correct

    def get(self, data, target):
        cat_feats = data
        orig_cat_feats = deepcopy(cat_feats.detach())
        if self.replacement_noise == 'swap':
            n_cat = cat_feats.shape[1]
            cols_to_shuffle = np.random.choice(n_cat, int(self.p_replace * n_cat), replace=False)
            for col in cols_to_shuffle:
                cat_feats[:, col] = cat_feats[:, col][torch.randperm(cat_feats.shape[0])]
        elif self.replacement_noise == 'random':
            locs_to_replace = torch.empty_like(cat_feats, dtype=float).uniform_() < self.p_replace
            col_cardinalities = torch.LongTensor([i[1] for i in self.cat_feat_origin_cards])
            col_cardinalities = col_cardinalities.unsqueeze(0).expand_as(cat_feats)
            unif = torch.rand(cat_feats.shape, device=col_cardinalities.device)
            random_feats = (unif * col_cardinalities).floor() + 1
            extra_replace = torch.mul((cat_feats == random_feats).to(int), locs_to_replace.to(int))
            cat_feats[locs_to_replace] = random_feats[locs_to_replace]
            assert torch.all(cat_feats[extra_replace] == orig_cat_feats[extra_replace]).item() is True
            extra_plus1 = cat_feats[extra_replace] + 1
            extra_minus1 = cat_feats[extra_replace] - 1
            extra_zero_padd_idx = extra_minus1 == 0
            extra_minus1[extra_zero_padd_idx] = extra_plus1[extra_zero_padd_idx]
            cat_feats[extra_replace] = extra_minus1
            assert torch.all(~(cat_feats[extra_replace] == orig_cat_feats[extra_replace])).item() is True
        elif self.replacement_noise == 'low_rank':
            assert self.p_replace + 0.2 <= 1, 'p_replace too big, lower it!'
            weights = torch.tensor([self.p_replace, 0.1, 0.9 - self.p_replace], dtype=torch.float)
            locs_to_change = torch.multinomial(weights, np.prod(cat_feats.shape), replacement=True).view(cat_feats.shape)
            col_cardinalities = torch.LongTensor([i[1] for i in self.cat_feat_origin_cards])
            col_cardinalities = col_cardinalities.unsqueeze(0).expand_as(cat_feats)
            unif = torch.rand(cat_feats.shape, device=col_cardinalities.device)
            random_feats = (unif * col_cardinalities).floor() + 1
            extra_replace = torch.mul((cat_feats == random_feats).to(int), (locs_to_change == 1).to(int))
            cat_feats[locs_to_change == 1] = random_feats[locs_to_change == 1]
            cat_feats[locs_to_change == 0] = 0
            assert torch.all(cat_feats[extra_replace] == orig_cat_feats[extra_replace]).item() is True
            extra_plus1 = cat_feats[extra_replace] + 1
            extra_minus1 = cat_feats[extra_replace] - 1
            extra_zero_padd_idx = extra_minus1 == 0
            extra_minus1[extra_zero_padd_idx] = extra_plus1[extra_zero_padd_idx]
            cat_feats[extra_replace] = extra_minus1
            assert torch.all(~(cat_feats[extra_replace] == orig_cat_feats[extra_replace])).item() is True
        pretext_label = (cat_feats != orig_cat_feats).long()
        pretext_data = cat_feats
        pretext_data = pretext_data
        pretext_label = pretext_label
        return pretext_data, pretext_label


class TabNet(nn.Module):

    def __init__(self, num_class, feature_dim, num_output_layers, device, params):
        """
        Internal torch model that uses TabTransformer as an embedding.
        This is where we are passing through activations and neurons.

        Parameters
        ----------
        num_class (int): Number of classes identified.
        cat_feat_origin_cards (list): List of categorical features
        """
        super().__init__()
        import torch.nn as nn
        self.embed = TabTransformer(**params)
        relu = nn.ReLU()
        in_dim = 2 * feature_dim
        lin = nn.Linear(in_dim, in_dim, bias=True)
        lin_out = nn.Linear(in_dim, num_class, bias=True)
        self.fc = [nn.Sequential(*[relu, lin])] * (num_output_layers - 1) + [nn.Sequential(*[relu, lin_out])]
        if device.type == 'cuda':
            for layer in range(num_output_layers):
                self.fc[layer] = self.fc[layer]

    def forward(self, data):
        features = self.embed(data)
        out = features.mean(dim=1)
        for layer in range(len(self.fc)):
            out = self.fc[layer](out)
        return out, features


class TabModelBase(nn.Module):

    def __init__(self, n_cont_features, norm_class_name, cat_feat_origin_cards, max_emb_dim, p_dropout, one_hot_embeddings, drop_whole_embeddings):
        super().__init__()
        """
        Base class for all TabTransformer models
        
        Parameters
        ----------
        max_emb_dim (int): Maximum allowable amount of embeddings.
        n_cont_features (int): How many continuous features to concatenate onto the categorical features.
        cat_feat_origin_cards (list): Categorical features to turn into embeddings.
        norm_class: What normalization to use for continuous features.
        p_dropout (float): How much dropout to apply.
        drop_whole_embeddings (bool): If True, dropout pretends the embedding was a missing value. If false, dropout sets embed features to 0
        one_hot_embeddings (bool): If True, one-hot encode variables whose cardinality is < max_emb_dim.
        cat_initializers (dict): Structure to hold the initial embeddings for categorical features.
        """
        self.max_emb_dim = max_emb_dim
        self.n_cont_features = n_cont_features
        self.cat_feat_origin_cards = cat_feat_origin_cards
        self.norm_class = nn.__dict__[norm_class_name]
        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        self.one_hot_embeddings = one_hot_embeddings
        self.cat_initializers = nn.ModuleDict()
        if isinstance(self.cat_feat_origin_cards, list):
            for col_name, card in self.cat_feat_origin_cards:
                self.cat_initializers[col_name] = EmbeddingInitializer(card, max_emb_dim, p_dropout, drop_whole_embeddings=drop_whole_embeddings, one_hot=one_hot_embeddings)
            self.init_feat_dim = sum(i.emb_dim for i in self.cat_initializers.values()) + self.n_cont_features

    def forward(self, input):
        raise NotImplementedError

    def get_norm(self, num_feats):
        return self.norm_class(num_feats)

    def pred_from_output(self, output):
        return output.max(dim=1, keepdim=True)[1]


class TabTransformer(TabModelBase):
    """
    Transformer model for tabular data, can also be used for semi-supervised learning.
    This is the internal transformer model embedding that will have further fully connected layers (TabNet) to
    actually produce predictions.
    """

    def __init__(self, n_cont_embeddings, n_layers, n_heads, hidden_dim, tab_readout, column_embedding, orig_emb_resid, fix_attention, n_shared_embs=8, shared_embedding_added=False, **kwargs):
        super().__init__(n_cont_features=kwargs['n_cont_features'], norm_class_name=kwargs['norm_class_name'], cat_feat_origin_cards=kwargs['cat_feat_origin_cards'], max_emb_dim=kwargs['max_emb_dim'], p_dropout=kwargs['p_dropout'], one_hot_embeddings=kwargs['one_hot_embeddings'], drop_whole_embeddings=kwargs['drop_whole_embeddings'])
        self.n_cont_embeddings = n_cont_embeddings
        self.hidden_dim = hidden_dim
        self.readout = tab_readout
        self.orig_emb_resid = orig_emb_resid
        self.cat_initializers = nn.ModuleDict()
        if isinstance(self.cat_feat_origin_cards, list):
            self.n_embeddings = len(self.cat_feat_origin_cards) + (n_cont_embeddings if self.n_cont_features else 0)
        else:
            self.n_embeddings = None
        self.cat_initializers = nn.ModuleDict()
        for col_name, card in self.cat_feat_origin_cards:
            self.cat_initializers[col_name] = EmbeddingInitializer(num_embeddings=card, max_emb_dim=self.max_emb_dim, p_dropout=self.p_dropout, minimize_emb_dim=False, drop_whole_embeddings=self.drop_whole_embeddings, one_hot=False, out_dim=self.hidden_dim, shared_embedding=column_embedding, n_shared_embs=n_shared_embs, shared_embedding_added=shared_embedding_added)
        if self.n_cont_features:
            self.cont_norm = self.get_norm(self.n_cont_features)
            self.cont_initializer = nn.Linear(self.n_cont_features, hidden_dim * n_cont_embeddings)
            self.cont_init_norm = self.get_norm(hidden_dim * n_cont_embeddings)
        if self.readout == 'readout_emb':
            self.readout_emb = nn.Parameter(torch.zeros(1, hidden_dim).uniform_(-1, 1))
            self.n_embeddings += 1
        if fix_attention is True:
            self.n_cat_embeddings = len(self.cat_feat_origin_cards)
            self.tfmr_layers = nn.ModuleList([TransformerEncoderLayerModified(d_model=hidden_dim, n_cat_embeddings=self.n_cat_embeddings, nhead=n_heads, dim_feedforward=4 * hidden_dim, dropout=self.p_dropout, activation='gelu') for _ in range(n_layers)])
        else:
            self.tfmr_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim, dropout=self.p_dropout, activation='gelu') for _ in range(n_layers)])

    def init_input(self, input):
        feats = [init(input[:, i]) for i, init in enumerate(self.cat_initializers.values())]
        if self.readout == 'readout_emb':
            readout_emb = self.readout_emb.expand_as(feats[0])
            feat_embs = torch.stack([readout_emb] + feats, dim=0)
        else:
            feat_embs = torch.stack(feats, dim=0)
        return feat_embs

    def run_tfmr(self, feat_embs):
        orig_feat_embs = feat_embs
        all_feat_embs = [feat_embs]
        for layer in self.tfmr_layers:
            feat_embs = layer(feat_embs)
            all_feat_embs.append(feat_embs)
            if self.orig_emb_resid:
                feat_embs = feat_embs + orig_feat_embs
        if self.readout == 'readout_emb':
            out = self.fc_out(feat_embs[0])
        elif self.readout == 'mean':
            out = torch.mean(feat_embs, dim=0)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool':
            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)
            last_layer = feat_embs.transpose(0, 1).reshape(feat_embs.shape[1], -1)
            out = torch.cat((last_layer, max, mean), dim=1)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool_all':
            feat_embs_all_layers = []
            for each_feat_embs in [all_feat_embs[0], all_feat_embs[-1]]:
                feat_embs_all_layers.append(each_feat_embs.transpose(0, 1).reshape(each_feat_embs.shape[1], -1))
            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)
            feat_embs_all_layers.append(max)
            feat_embs_all_layers.append(mean)
            out = torch.cat(feat_embs_all_layers, dim=1)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool_add':
            orig_feat_embs_cp = copy.deepcopy(orig_feat_embs.detach())
            last_layer = feat_embs.transpose(0, 1).reshape(feat_embs.shape[1], -1)
            last_layer += orig_feat_embs_cp.transpose(0, 1).reshape(orig_feat_embs_cp.shape[1], -1)
            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)
            out = torch.cat([last_layer, max, mean], dim=1)
        elif self.readout == 'all_feat_embs':
            out = feat_embs
        elif self.readout == 'mean_feat_embs':
            out = feat_embs.mean(dim=0)
        elif self.readout == 'none':
            out = feat_embs.transpose(1, 0)
        return out

    def forward(self, input):
        """
        Returns logits for output classes
        """
        feat_embs = self.init_input(input)
        out = self.run_tfmr(feat_embs)
        return out


class EmbeddingInitializer(nn.Module):

    def __init__(self, num_embeddings, max_emb_dim, p_dropout, minimize_emb_dim=True, drop_whole_embeddings=False, one_hot=False, out_dim=None, shared_embedding=False, n_shared_embs=8, shared_embedding_added=False):
        """
        :param minimize_emb_dim:
            Whether to set embedding_dim = max_emb_dim or to make embedding_dim smaller is num_embeddings is small
        :param drop_whole_embeddings:
            If True, dropout pretends the embedding was a missing value. If false, dropout sets embed features to 0
        :param one_hot:
            If True, one-hot encode variables whose cardinality is < max_emb_dim. Also, set reqiures_grad = False
        :param out_dim:
            If None, return the embedding straight from self.embed.  If another dimension, put the embedding through a
            Linear layer to make it size (batch x out_dim).
         :param shared_embedding:
            If True, 1/(n_shared_embs)th of every embedding will be reserved for a learned parameter that's common to all embeddings.
            This is useful for transformers to identify which column an embedding came from.
            Mutually exclusive with one_hot.

        Note: the 0 embedding is reserved for padding and masking.  The various encoders use 1 for missing values.

        """
        super().__init__()
        assert not (one_hot and out_dim is not None)
        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        self.shared_embedding = shared_embedding
        self.shared_embedding_added = shared_embedding_added
        if minimize_emb_dim or one_hot:
            self.emb_dim = min(max_emb_dim, num_embeddings)
        else:
            self.emb_dim = max_emb_dim
        self.reshape_out = nn.Identity()
        if out_dim is not None:
            assert self.emb_dim <= out_dim, 'Makes no sense: just set max_emb_dim = out_dim and out_dim = None'
            if num_embeddings > self.emb_dim:
                self.reshape_out = nn.Linear(self.emb_dim, out_dim, bias=True)
            else:
                self.emb_dim = out_dim
        self.embed = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=self.emb_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if one_hot:
            self.embed.weight.requires_grad = False
            if num_embeddings <= max_emb_dim:
                self.embed.weight.data[1:, :] = torch.eye(self.emb_dim)
        if shared_embedding:
            assert not one_hot
            ce_dim = self.emb_dim if shared_embedding_added else (out_dim if out_dim else self.emb_dim) // n_shared_embs
            self.shared_emb = nn.Parameter(torch.empty(1, ce_dim).uniform_(-1, 1))
        self.do = nn.Dropout(p=p_dropout)

    def forward(self, input):
        if self.drop_whole_embeddings and self.training:
            mask = torch.zeros_like(input).bernoulli_(1 - self.p_dropout)
            input = input * mask
        out = self.embed(input)
        if not self.drop_whole_embeddings:
            out = self.do(out)
        out = self.reshape_out(out)
        if self.shared_embedding:
            shared_emb = self.shared_emb.expand(out.shape[0], -1)
            if not self.shared_embedding_added:
                out[:, :shared_emb.shape[1]] = shared_emb
            else:
                out += shared_emb
        return out


BINARY = 'binary'


MULTICLASS = 'multiclass'


QUANTILE = 'quantile'


SOFTCLASS = 'softclass'


def get_embed_sizes(train_dataset, params, num_categs_per_feature):
    """ Returns list of embedding sizes for each categorical variable.
        Selects this adaptively based on training_dataset.
        Note: Assumes there is at least one embed feature.
    """
    max_embedding_dim = params['max_embedding_dim']
    embed_exponent = params['embed_exponent']
    size_factor = params['embedding_size_factor']
    embed_dims = [int(size_factor * max(2, min(max_embedding_dim, 1.6 * num_categs_per_feature[i] ** embed_exponent))) for i in range(len(num_categs_per_feature))]
    return embed_dims


class EmbedNet(nn.Module):
    """
    y_range: Used specifically for regression. = None for classification.
    """

    def __init__(self, problem_type, num_net_outputs=None, quantile_levels=None, train_dataset=None, architecture_desc=None, device=None, **kwargs):
        if architecture_desc is None and train_dataset is None:
            raise ValueError('train_dataset cannot = None if architecture_desc=None')
        super().__init__()
        self.problem_type = problem_type
        if self.problem_type == QUANTILE:
            self.register_buffer('quantile_levels', torch.Tensor(quantile_levels).float().reshape(1, -1))
        self.device = torch.device('cpu') if device is None else device
        if architecture_desc is None:
            params = self._set_params(**kwargs)
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features
            self.has_embed_features = train_dataset.has_embed_features
            if self.has_embed_features:
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = get_embed_sizes(train_dataset, params, num_categs_per_feature)
            if self.has_vector_features:
                vector_dims = train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        else:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc['has_vector_features']
            self.has_embed_features = architecture_desc['has_embed_features']
            self.from_logits = architecture_desc['from_logits']
            params = architecture_desc['params']
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc['num_categs_per_feature']
                embed_dims = architecture_desc['embed_dims']
            if self.has_vector_features:
                vector_dims = architecture_desc['vector_dims']
        input_size = 0
        if self.has_embed_features:
            self.embed_blocks = nn.ModuleList()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.append(nn.Embedding(num_embeddings=num_categs_per_feature[i], embedding_dim=embed_dims[i]))
                input_size += embed_dims[i]
        if self.has_vector_features:
            input_size += vector_dims
        act_fn = nn.Identity()
        if params['activation'] == 'elu':
            act_fn = nn.ELU()
        elif params['activation'] == 'relu':
            act_fn = nn.ReLU()
        elif params['activation'] == 'tanh':
            act_fn = nn.Tanh()
        layers = []
        if params['use_batchnorm']:
            layers.append(nn.BatchNorm1d(input_size))
        layers.append(nn.Linear(input_size, params['hidden_size']))
        layers.append(act_fn)
        for _ in range(params['num_layers'] - 1):
            if params['use_batchnorm']:
                layers.append(nn.BatchNorm1d(params['hidden_size']))
            layers.append(nn.Dropout(params['dropout_prob']))
            layers.append(nn.Linear(params['hidden_size'], params['hidden_size']))
            layers.append(act_fn)
        layers.append(nn.Linear(params['hidden_size'], num_net_outputs))
        self.main_block = nn.Sequential(*layers)
        if self.problem_type in [REGRESSION, QUANTILE]:
            y_range = params['y_range']
            self.y_constraint = None
            if y_range is not None:
                if y_range[0] == -np.inf and y_range[1] == np.inf:
                    self.y_constraint = None
                elif y_range[0] >= 0 and y_range[1] == np.inf:
                    self.y_constraint = 'nonnegative'
                elif y_range[0] == -np.inf and y_range[1] <= 0:
                    self.y_constraint = 'nonpositive'
                else:
                    self.y_constraint = 'bounded'
                self.y_lower = y_range[0]
                self.y_upper = y_range[1]
                self.y_span = self.y_upper - self.y_lower
        if self.problem_type == QUANTILE:
            self.alpha = params['alpha']
        if self.problem_type == SOFTCLASS:
            self.log_softmax = torch.nn.LogSoftmax(dim=1)
        if self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
            self.softmax = torch.nn.Softmax(dim=1)
        if architecture_desc is None:
            self.architecture_desc = {'has_vector_features': self.has_vector_features, 'has_embed_features': self.has_embed_features, 'params': params, 'num_net_outputs': num_net_outputs, 'from_logits': self.from_logits}
            if self.has_embed_features:
                self.architecture_desc['num_categs_per_feature'] = num_categs_per_feature
                self.architecture_desc['embed_dims'] = embed_dims
            if self.has_vector_features:
                self.architecture_desc['vector_dims'] = vector_dims

    def _set_params(self, num_layers=4, hidden_size=128, activation='relu', use_batchnorm=False, dropout_prob=0.1, y_range=None, alpha=0.01, max_embedding_dim=100, embed_exponent=0.56, embedding_size_factor=1.0):
        return dict(num_layers=num_layers, hidden_size=hidden_size, activation=activation, use_batchnorm=use_batchnorm, dropout_prob=dropout_prob, y_range=y_range, alpha=alpha, max_embedding_dim=max_embedding_dim, embed_exponent=embed_exponent, embedding_size_factor=embedding_size_factor)

    def init_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data_batch):
        input_data = []
        input_offset = 0
        if self.has_vector_features:
            input_data.append(data_batch[0])
            input_offset += 1
        if self.has_embed_features:
            embed_data = data_batch[input_offset]
            for i in range(len(self.embed_blocks)):
                input_data.append(self.embed_blocks[i](embed_data[i]))
        if len(input_data) > 1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]
        output_data = self.main_block(input_data)
        if self.problem_type in [REGRESSION, QUANTILE]:
            if self.y_constraint is None:
                return output_data
            elif self.y_constraint == 'nonnegative':
                return self.y_lower + torch.abs(output_data)
            elif self.y_constraint == 'nonpositive':
                return self.y_upper - torch.abs(output_data)
            else:
                return torch.sigmoid(output_data) * self.y_span + self.y_lower
        elif self.problem_type == SOFTCLASS:
            return self.log_softmax(output_data)
        else:
            return output_data

    def huber_pinball_loss(self, input_data, target_data):
        error_data = target_data.contiguous().reshape(-1, 1) - input_data
        if self.alpha == 0.0:
            loss_data = torch.max(self.quantile_levels * error_data, (self.quantile_levels - 1) * error_data)
            return loss_data.mean()
        loss_data = torch.where(torch.abs(error_data) < self.alpha, 0.5 * error_data * error_data, self.alpha * (torch.abs(error_data) - 0.5 * self.alpha))
        loss_data /= self.alpha
        scale = torch.where(error_data >= 0, torch.ones_like(error_data) * self.quantile_levels, torch.ones_like(error_data) * (1 - self.quantile_levels))
        loss_data *= scale
        return loss_data.mean()

    def margin_loss(self, input_data, margin_scale=0.0001):
        batch_size, num_quantiles = input_data.size()
        error_data = input_data.unsqueeze(1) - input_data.unsqueeze(2)
        margin_data = self.quantile_levels.permute(1, 0) - self.quantile_levels
        margin_data = torch.tril(margin_data, -1) * margin_scale
        loss_data = torch.tril(error_data + margin_data, diagonal=-1)
        loss_data = loss_data.relu()
        loss_data = loss_data.sum() / float(batch_size * (num_quantiles * num_quantiles - num_quantiles) * 0.5)
        return loss_data

    def quantile_loss(self, predict_data, target_data, margin):
        if margin > 0.0:
            m_loss = self.margin_loss(predict_data)
        else:
            m_loss = 0.0
        h_loss = self.huber_pinball_loss(predict_data, target_data).mean()
        return h_loss + margin * m_loss

    def compute_loss(self, data_batch, loss_function=None, gamma=None):
        self.train()
        predict_data = self(data_batch)
        target_data = data_batch[-1]
        if self.problem_type in [BINARY, MULTICLASS]:
            target_data = target_data.type(torch.long)
        if self.problem_type == QUANTILE:
            return self.quantile_loss(predict_data, target_data, margin=gamma)
        if self.problem_type == SOFTCLASS:
            return loss_function(predict_data, target_data)
        else:
            target_data = target_data.flatten()
            if self.problem_type == REGRESSION:
                predict_data = predict_data.flatten()
            return loss_function(predict_data, target_data)

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data)
            if self.problem_type == QUANTILE:
                predict_data = torch.sort(predict_data, -1)[0]
            elif self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
                predict_data = self.softmax(predict_data)
            elif self.problem_type == REGRESSION:
                predict_data = predict_data.flatten()
            if self.problem_type == BINARY:
                predict_data = predict_data[:, 1]
            return predict_data.data.cpu().numpy()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdditiveAttention,
     lambda: ([], {'d_token': 4, 'n_heads': 4, 'dropout': 0.5, 'bias': 4, 'share_qv_weights': 4, 'initialization': 'kaiming'}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (AutoDis,
     lambda: ([], {'in_features': 4, 'd_embedding': 4, 'n_meta_embeddings': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DummyLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostBatchNorm,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (IA3Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'merge_weights': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IA3LoRALinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LoRAConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LoRALinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LoRAMergedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiNegativesSoftmaxLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (NLayerNorm,
     lambda: ([], {'n_features': 4, 'd': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (NLinear,
     lambda: ([], {'n': 4, 'd_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (NLinearMemoryEfficient,
     lambda: ([], {'n': 4, 'd_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NumericalFeatureTokenizer,
     lambda: ([], {'in_features': 4, 'd_token': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Periodic,
     lambda: ([], {'in_features': 4, 'd_embedding': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (RKDLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (ReGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftTargetCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_LinearWithBias,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_autogluon_autogluon(_paritybench_base):
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

