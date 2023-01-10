import sys
_module = sys.modules[__name__]
del sys
check_docs = _module
conf = _module
mock_docs = _module
demo0 = _module
demo1 = _module
demo10 = _module
demo11 = _module
demo12 = _module
demo2 = _module
demo3 = _module
demo4 = _module
demo5 = _module
demo6 = _module
demo7 = _module
demo8 = _module
demo9 = _module
conditional_parameters = _module
custom_search_space = _module
sequential_parameter_search = _module
simple_tabular_classification = _module
lightautoml = _module
addons = _module
interpretation = _module
data_process = _module
l2x = _module
l2x_model = _module
lime = _module
utils = _module
uplift = _module
base = _module
metalearners = _module
metrics = _module
utils = _module
utilization = _module
automl = _module
blend = _module
presets = _module
base = _module
image_presets = _module
tabular_presets = _module
text_presets = _module
whitebox_presets = _module
dataset = _module
np_pd_dataset = _module
roles = _module
image = _module
image = _module
ml_algo = _module
boost_cb = _module
boost_lgbm = _module
dl_model = _module
linear_sklearn = _module
torch_based = _module
linear_model = _module
tuning = _module
optuna = _module
whitebox = _module
pipelines = _module
features = _module
image_pipeline = _module
lgb_pipeline = _module
linear_pipeline = _module
text_pipeline = _module
wb_pipeline = _module
ml = _module
nested_ml_pipe = _module
whitebox_ml_pipe = _module
selection = _module
importance_based = _module
linear_selector = _module
permutation_importance_based = _module
reader = _module
guess_roles = _module
tabular_batch_generator = _module
report = _module
report_deco = _module
tasks = _module
base = _module
common_metric = _module
losses = _module
cb = _module
cb_custom = _module
lgb = _module
lgb_custom = _module
sklearn = _module
text = _module
dl_transformers = _module
dp_utils = _module
embed_dataset = _module
nn_model = _module
sentence_pooling = _module
tokenizer = _module
trainer = _module
utils = _module
weighted_average_transformer = _module
transformers = _module
categorical = _module
datetime = _module
decomposition = _module
image = _module
numeric = _module
text = _module
installation = _module
logging = _module
timer = _module
validation = _module
np_iterators = _module
conftest = _module
test_default_tabular = _module
test_simple_integration = _module
test_wb_preset = _module
test_addons = _module
test_automl = _module
test_dataset = _module
test_image = _module
test_ml_algo = _module
test_optuna_tuner = _module
test_pipelines = _module
test_reader = _module
test_report = _module
test_tasks = _module
test_text = _module
test_transformers = _module
test_utils = _module
test_logging = _module
test_validation = _module

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


from collections import Counter


from collections import defaultdict


from random import shuffle


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import numpy as np


import pandas as pd


import torch


from torch import nn


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import Sampler


import logging


from numbers import Number


from typing import Type


from typing import Union


import torch.nn as nn


from torch.optim import Adam


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as F


from torch.distributions.utils import clamp_probs


import itertools


import matplotlib.pyplot as plt


from matplotlib.colors import Colormap


from typing import Sequence


from typing import Iterable


from pandas import DataFrame


from copy import copy


from copy import deepcopy


from typing import cast


from sklearn.base import TransformerMixin


import uuid


from torch.optim import lr_scheduler


from sklearn.linear_model import ElasticNet


from sklearn.linear_model import Lasso


from sklearn.linear_model import LogisticRegression


from scipy import sparse


from torch import optim


import inspect


from functools import partial


from typing import TYPE_CHECKING


from itertools import chain


from torch._utils import ExceptionWrapper


from torch.cuda._utils import _get_device_index


import random


from sklearn.utils.murmurhash import murmurhash3_32


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.linear_model import SGDClassifier


from sklearn.linear_model import SGDRegressor


def create_emb_layer(weights_matrix=None, voc_size=None, embed_dim=None, trainable_embeds=True) ->torch.nn.Embedding:
    """Create initialized embedding layer.

    Args:
        weights_matrix: Weights of embedding layer.
        voc_size: Size of vocabulary.
        embed_dim: Size of embeddings.
        trainable_embeds: To optimize layer when training model.

    Returns:
        Initialized embedding layer.

    """
    assert weights_matrix is not None or voc_size is not None and embed_dim is not None, 'Please define anything: weights_matrix or voc_size & embed_dim'
    if weights_matrix is not None:
        voc_size, embed_dim = weights_matrix.size()
    emb_layer = nn.Embedding(voc_size, embed_dim)
    if weights_matrix is not None:
        emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable_embeds:
        emb_layer.weight.requires_grad = False
    return emb_layer


class TIModel(nn.Module):

    def __init__(self, voc_size: int, embed_dim: int=50, conv_filters: int=100, conv_ksize: int=3, drop_rate: float=0.2, hidden_dim: int=100, weights_matrix: Optional[torch.FloatTensor]=None, trainable_embeds: bool=False):
        super(TIModel, self).__init__()
        self.lookup = create_emb_layer(weights_matrix, voc_size, embed_dim, trainable_embeds)
        embed_dim = self.lookup.embedding_dim
        self.drop1 = nn.Dropout(p=drop_rate)
        self.conv1 = nn.Conv1d(embed_dim, conv_filters, conv_ksize, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool1d(output_size=1)
        self.global_info = nn.Linear(conv_filters, hidden_dim)
        self.global_act = nn.ReLU()
        self.conv2 = nn.Conv1d(conv_filters, hidden_dim, conv_ksize, padding=1)
        self.act2 = nn.ReLU()
        self.local_info = nn.Conv1d(hidden_dim, hidden_dim, conv_ksize, padding=1)
        self.local_act = nn.ReLU()
        self.drop3 = nn.Dropout(p=drop_rate)
        self.conv3 = nn.Conv1d(2 * hidden_dim, conv_filters, 1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv1d(conv_filters, 1, 1)

    def get_embedding(self, input):
        return self.lookup(input)

    def get_embedding_layer(self):
        return self.lookup

    def freeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(False)

    def unfreeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(True)

    def predict(self, x):
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(self.drop1(x)))
        global_info = self.global_act(self.global_info(self.pool1(x).squeeze(2)))
        local_info = self.local_act(self.local_info(self.act2(self.conv2(x))))
        global_info = global_info.unsqueeze(-1).expand_as(local_info)
        z = torch.cat([global_info, local_info], dim=1)
        z = self.act3(self.conv3(self.drop3(z)))
        logits = self.conv4(z)
        return logits

    def forward(self, x):
        embed = self.get_embedding(x)
        logits = self.predict(embed)
        return logits


class GumbelTopKSampler(nn.Module):

    def __init__(self, T, k):
        super(GumbelTopKSampler, self).__init__()
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.int32), requires_grad=False)

    def sample_continous(self, logits):
        l_shape = logits.shape[0], self.k, logits.shape[2]
        u = clamp_probs(torch.rand(l_shape, device=logits.device))
        gumbel = -torch.log(-torch.log(u))
        noisy_logits = (gumbel + logits) / self.T
        samples = F.softmax(noisy_logits, dim=-1)
        samples = torch.max(samples, dim=1)[0]
        return samples

    def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()
        return samples

    def forward(self, logits):
        csamples = None
        if self.training:
            csamples = self.sample_continous(logits)
        dsamples = self.sample_discrete(logits)
        return dsamples, csamples


class SoftSubSampler(nn.Module):

    def __init__(self, T, k):
        super(SoftSubSampler, self).__init__()
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.int32), requires_grad=False)

    def inject_noise(self, logits):
        u = clamp_probs(torch.rand_like(logits))
        z = -torch.log(-torch.log(u))
        noisy_logits = logits + z
        return noisy_logits

    def continuous_topk(self, w, separate=False):
        khot_list = []
        onehot_approx = torch.zeros_like(w, dtype=torch.float32)
        for _ in range(self.k):
            khot_mask = clamp_probs(1.0 - onehot_approx)
            w += torch.log(khot_mask)
            onehot_approx = F.softmax(w / self.T, dim=-1)
            khot_list.append(onehot_approx)
        if separate:
            return khot_list
        else:
            return torch.stack(khot_list, dim=-1).sum(-1).squeeze(1)

    def sample_continous(self, logits):
        return self.continuous_topk(self.inject_noise(logits))

    def sample_discrete(self, logits):
        threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
        samples = torch.ge(logits.squeeze(1), threshold).float()
        return samples

    def forward(self, logits):
        csamples = None
        if self.training:
            csamples = self.sample_continous(logits)
        dsamples = self.sample_discrete(logits)
        return dsamples, csamples


class DistilPredictor(nn.Module):

    def __init__(self, task_name: str, n_outs: int, voc_size: int, embed_dim: int=300, hidden_dim: int=100, weights_matrix: Optional[torch.FloatTensor]=None, trainable_embeds: bool=False):
        super(DistilPredictor, self).__init__()
        self.lookup = create_emb_layer(weights_matrix, voc_size, embed_dim, trainable_embeds)
        embed_dim = self.lookup.embedding_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        if task_name == 'reg':
            self.head = nn.Linear(hidden_dim, n_outs)
        elif task_name == 'binary':
            self.head = nn.Sequential(nn.Linear(hidden_dim, n_outs), nn.Sigmoid())
        elif task_name == 'multiclass':
            self.head = nn.Sequential(nn.Linear(hidden_dim, n_outs), nn.Softmax(dim=-1))

    def get_embedding(self, input):
        return self.lookup(input)

    def get_embedding_layer(self):
        return self.lookup

    def freeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(False)

    def unfreeze_embedding(self):
        embeds = self.get_embedding_layer()
        embeds.weight.requires_grad_(True)

    def predict(self, embed, T):
        out = torch.mean(embed * T.unsqueeze(2), axis=1)
        out = self.act(self.fc1(out))
        out = self.head(out)
        return out

    def forward(self, x, T):
        embed = self.get_embedding(x)
        out = self.predict(embed, T)
        return out


class L2XModel(nn.Module):

    def __init__(self, task_name: str, n_outs: int, voc_size: int=1000, embed_dim: int=100, conv_filters: int=100, conv_ksize: int=3, drop_rate: float=0.2, hidden_dim: int=100, T: float=0.3, k: int=5, weights_matrix: Optional[torch.FloatTensor]=None, trainable_embeds: bool=False, sampler: str='gumbeltopk', anneal_factor: float=1.0):
        super(L2XModel, self).__init__()
        self.ti_model = TIModel(voc_size, embed_dim, conv_filters, conv_ksize, drop_rate, hidden_dim, weights_matrix, trainable_embeds)
        self.T = T
        self.anneal_factor = anneal_factor
        if sampler == 'gumbeltopk':
            self.sampler = GumbelTopKSampler(T, k)
        else:
            self.sampler = SoftSubSampler(T, k)
        self.distil_model = DistilPredictor(task_name, n_outs, voc_size, embed_dim, hidden_dim, weights_matrix, trainable_embeds)

    def forward(self, x):
        """Forward pass."""
        logits = self.ti_model(x)
        dsamples, csamples = self.sampler(logits)
        if self.training:
            T = csamples
        else:
            T = dsamples
        out = self.distil_model(x, T)
        return out, T

    def anneal(self):
        """Temperature annealing."""
        self.sampler.T *= self.anneal_factor


class EffNetImageEmbedder(nn.Module):
    """Class to compute EfficientNet embeddings."""

    def __init__(self, model_name: str='efficientnet-b0', weights_path: Optional[str]=None, is_advprop: bool=True, device=torch.device('cuda:0')):
        """Pytorch module for image embeddings based on efficient-net model.

        Args:
            model_name: Name of effnet model.
            weights_path: Path to saved weights.
            is_advprop: Use adversarial training.
            device: Device to use.

        """
        super(EffNetImageEmbedder, self).__init__()
        self.device = device
        self.model = EfficientNet.from_pretrained(model_name, weights_path=weights_path, advprop=is_advprop, include_top=False).eval()
        self.feature_shape = self.get_shape()
        self.is_advprop = is_advprop
        self.model_name = model_name

    @torch.no_grad()
    def get_shape(self) ->int:
        """Calculate output embedding shape.

        Returns:
            Shape of embedding.

        """
        return self.model(torch.randn(1, 3, 224, 224)).squeeze().shape[0]

    def forward(self, x) ->torch.Tensor:
        """Forward pass."""
        out = self.model(x)
        return out[:, :, 0, 0]


class CatLinear(nn.Module):
    """Simple linear model to handle numeric and categorical features.

    Args:
        numeric_size: Number of numeric features.
        embed_sizes: Embedding sizes.
        output_size: Size of output layer.

    """

    def __init__(self, numeric_size: int=0, embed_sizes: Sequence[int]=(), output_size: int=1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.linear = None
        if numeric_size > 0:
            self.linear = nn.Linear(in_features=numeric_size, out_features=output_size, bias=False)
            nn.init.zeros_(self.linear.weight)
        self.cat_params = None
        if len(embed_sizes) > 0:
            self.cat_params = nn.Parameter(torch.zeros(sum(embed_sizes), output_size))
            self.embed_idx = torch.LongTensor(embed_sizes).cumsum(dim=0) - torch.LongTensor(embed_sizes)

    def forward(self, numbers: Optional[torch.Tensor]=None, categories: Optional[torch.Tensor]=None):
        """Forward-pass.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        Returns:
            Linear prediction.

        """
        x = self.bias
        if self.linear is not None:
            x = x + self.linear(numbers)
        if self.cat_params is not None:
            x = x + self.cat_params[categories + self.embed_idx].sum(dim=1)
        return x


class CatLogisticRegression(CatLinear):
    """Realisation of torch-based logistic regression."""

    def __init__(self, numeric_size: int, embed_sizes: Sequence[int]=(), output_size: int=1):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, numbers: Optional[torch.Tensor]=None, categories: Optional[torch.Tensor]=None):
        """Forward-pass. Sigmoid func at the end of linear layer.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        Returns:
            Probabilitics.

        """
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.sigmoid(x)
        return x


class CatRegression(CatLinear):
    """Realisation of torch-based linear regreession."""

    def __init__(self, numeric_size: int, embed_sizes: Sequence[int]=(), output_size: int=1):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)


class CatMulticlass(CatLinear):
    """Realisation of multi-class linear classifier."""

    def __init__(self, numeric_size: int, embed_sizes: Sequence[int]=(), output_size: int=1):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, numbers: Optional[torch.Tensor]=None, categories: Optional[torch.Tensor]=None):
        """Forward-pass.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        Returns:
            Linear prediction.

        """
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.softmax(x)
        return x


class TorchLossWrapper(nn.Module):
    """Customize PyTorch-based loss.

    Args:
        func: loss to customize. Example: `torch.nn.MSELoss`.
        **kwargs: additional parameters.

    Returns:
        callable loss, uses format (y_true, y_pred, sample_weight).

    """

    def __init__(self, func: Callable, flatten=False, log=False, **kwargs: Any):
        super(TorchLossWrapper, self).__init__()
        self.base_loss = func(reduction='none', **kwargs)
        self.flatten = flatten
        self.log = log

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None):
        """Forward-pass."""
        if self.flatten:
            y_true = y_true[:, 0].type(torch.int64)
        if self.log:
            y_pred = torch.log(y_pred)
        outp = self.base_loss(y_pred, y_true)
        if len(outp.shape) == 2:
            outp = outp.sum(dim=1)
        if sample_weight is not None:
            outp = outp * sample_weight
            return outp.mean() / sample_weight.mean()
        return outp.mean()


class SequenceAbstractPooler(nn.Module):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) ->torch.Tensor:
        x = x.masked_fill(~x_mask, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active.data
        return values


class SequenceClsPooler(SequenceAbstractPooler):
    """CLS token pooling."""

    def __init__(self):
        super(SequenceClsPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) ->torch.Tensor:
        return x[..., 0, :]


class SequenceIndentityPooler(SequenceAbstractPooler):
    """Identity pooling."""

    def __init__(self):
        super(SequenceIndentityPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) ->torch.Tensor:
        return x


class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) ->torch.Tensor:
        x = x.masked_fill(~x_mask, -float('inf'))
        values, _ = torch.max(x, dim=-2)
        return values


class SequenceSumPooler(SequenceAbstractPooler):
    """Sum value pooling."""

    def __init__(self):
        super(SequenceSumPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) ->torch.Tensor:
        x = x.masked_fill(~x_mask, 0)
        values = torch.sum(x, dim=-2)
        return values


pooling_by_name = {'mean': SequenceAvgPooler, 'sum': SequenceSumPooler, 'max': SequenceMaxPooler, 'cls': SequenceClsPooler, 'none': SequenceIndentityPooler}


def position_encoding_init(n_pos: int, embed_size: int) ->torch.Tensor:
    """Compute positional embedding matrix.

    Args:
        n_pos: Len of sequence.
        embed_size: Size of output sentence embedding.

    Returns:
        Torch tensor with all positional embeddings.

    """
    position_enc = np.array([([(pos / np.power(10000, 2 * (j // 2) / embed_size)) for j in range(embed_size)] if pos != 0 else np.zeros(embed_size)) for pos in range(n_pos)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).float()


def seed_everything(seed: int=42, deterministic: bool=True):
    """Set random seed and cudnn params.

    Args:
        seed: Random state.
        deterministic: cudnn backend.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True


class BOREP(nn.Module):
    """Class to compute Bag of Random Embedding Projections sentence embeddings from words embeddings.

    Bag of Random Embedding Projections sentence embeddings.

    Args:
        embed_size: Size of word embeddings.
        proj_size: Size of output sentence embedding.
        pooling: Pooling type.
        max_length: Maximum length of sentence.
        init: Type of weight initialization.
        pos_encoding: Add positional embedding.
        **kwargs: Ignored params.

    Note:
        There are several pooling types:

            - `'max'`: Maximum on seq_len dimension for non masked inputs.
            - `'mean'`: Mean on seq_len dimension for non masked inputs.
            - `'sum'`: Sum on seq_len dimension for non masked inputs.

        For init parameter there are several options:

            - `'orthogonal'`: Orthogonal init.
            - `'normal'`: Normal with std 0.1.
            - `'uniform'`: Uniform from -0.1 to 0.1.
            - `'kaiming'`: Uniform kaiming init.
            - `'xavier'`: Uniform xavier init.

    """
    name = 'BOREP'
    _poolers = {'max', 'mean', 'sum'}

    def __init__(self, embed_size: int=300, proj_size: int=300, pooling: str='mean', max_length: int=200, init: str='orthogonal', pos_encoding: bool=False, **kwargs: Any):
        super(BOREP, self).__init__()
        self.embed_size = embed_size
        self.proj_size = proj_size
        self.pos_encoding = pos_encoding
        seed_everything(42)
        if self.pos_encoding:
            self.pos_code = position_encoding_init(max_length, self.embed_size).view(1, max_length, self.embed_size)
        self.pooling = pooling_by_name[pooling]()
        self.proj = nn.Linear(self.embed_size, self.proj_size, bias=False)
        if init == 'orthogonal':
            nn.init.orthogonal_(self.proj.weight)
        elif init == 'normal':
            nn.init.normal_(self.proj.weight, std=0.1)
        elif init == 'uniform':
            nn.init.uniform_(self.proj.weight, a=-0.1, b=0.1)
        elif init == 'kaiming':
            nn.init.kaiming_uniform_(self.proj.weight)
        elif init == 'xavier':
            nn.init.xavier_uniform_(self.proj.weight)

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.proj_size

    def get_name(self) ->str:
        """Module name.

        Returns:
            String with module name.

        """
        return self.name

    @torch.no_grad()
    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        x = inp['text']
        batch_size, batch_max_length = x.shape[0], x.shape[1]
        if self.pos_encoding:
            x = x + self.pos_code[:, :batch_max_length, :]
        x = x.contiguous().view(batch_size * batch_max_length, -1)
        x = self.proj(x)
        out = x.contiguous().view(batch_size, batch_max_length, -1)
        x_length = (torch.arange(out.shape[1])[None, :] < inp['length'][:, None])[:, :, None]
        out = self.pooling(out, x_length)
        return out


class RandomLSTM(nn.Module):
    """Class to compute Random LSTM sentence embeddings from words embeddings.

    Args:
        embed_size: Size of word embeddings.
        hidden_size: Size of hidden dimensions of LSTM.
        pooling: Pooling type.
        num_layers: Number of lstm layers.
        **kwargs: Ignored params.

    Note:
        There are several pooling types:

            - `'max'`: Maximum on seq_len dimension for non masked inputs.
            - `'mean'`: Mean on seq_len dimension for non masked inputs.
            - `'sum'`: Sum on seq_len dimension for non masked inputs.

    """
    name = 'RandomLSTM'
    _poolers = 'max', 'mean', 'sum'

    def __init__(self, embed_size: int=300, hidden_size: int=256, pooling: str='mean', num_layers: int=1, **kwargs: Any):
        super(RandomLSTM, self).__init__()
        if pooling not in self._poolers:
            raise ValueError('pooling - {} - not in the list of available types {}'.format(pooling, self._poolers))
        seed_everything(42)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.hidden_size * 2

    def get_name(self) ->str:
        """Module name.

        Returns:
            String with module name.

        """
        return self.name

    @torch.no_grad()
    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        out, _ = self.lstm(inp['text'])
        x_length = (torch.arange(out.shape[1])[None, :] < inp['length'][:, None])[:, :, None]
        out = self.pooling(out, x_length)
        return out


def single_text_hash(x: str) ->str:
    """Get text hash.

    Args:
        x: Text.

    Returns:
        String text hash.

    """
    numhash = murmurhash3_32(x, seed=13)
    texthash = str(numhash) if numhash > 0 else 'm' + str(abs(numhash))
    return texthash


class BertEmbedder(nn.Module):
    """Class to compute `HuggingFace <https://huggingface.co>`_ transformers words or sentence embeddings.

    Bert sentence or word embeddings.

    Args:
        model_name: Name of transformers model.
        pooling: Pooling type.
        **kwargs: Ignored params.

    Note:
        There are several pooling types:

            - `'cls'`: Use CLS token for sentence embedding
                from last hidden state.
            - `'max'`: Maximum on seq_len dimension
                for non masked inputs from last hidden state.
            - `'mean'`: Mean on seq_len dimension for non masked
                inputs from last hidden state.
            - `'sum'`: Sum on seq_len dimension for non masked inputs
                from last hidden state.
            - `'none'`: Don't use pooling (for RandomLSTM pooling strategy).

    """
    name = 'BertEmb'
    _poolers = {'cls', 'max', 'mean', 'sum', 'none'}

    def __init__(self, model_name: str, pooling: str='none', **kwargs: Any):
        super(BertEmbedder, self).__init__()
        if pooling not in self._poolers:
            raise ValueError('pooling - {} - not in the list of available types {}'.format(pooling, self._poolers))
        self.pooling = pooling_by_name[pooling]()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)

    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        encoded_layers, _ = self.transformer(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], token_type_ids=inp.get('token_type_ids'), return_dict=False)
        encoded_layers = self.pooling(encoded_layers, inp['attention_mask'].unsqueeze(-1).bool())
        return encoded_layers

    def freeze(self):
        """Freeze module parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = False

    def get_name(self) ->str:
        """Module name.

        Returns:
            String with module name.

        """
        return self.name + single_text_hash(self.model_name)

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.transformer.config.hidden_size


class CustomDataParallel(nn.DataParallel):
    """Extension for nn.DataParallel for supporting predict method of DL model."""

    def __init__(self, module: nn.Module, device_ids: Optional[List[int]]=None, output_device: Optional[torch.device]=None, dim: Optional[int]=0):
        super(CustomDataParallel, self).__init__(module, device_ids, output_device, dim)
        try:
            self.n_out = module.n_out
        except:
            pass

    def predict(self, *inputs, **kwargs):
        """Predict."""
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError('module must have its parameters and buffers on device {} (device_ids[0]) but found one of them on device: {}'.format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.predict(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply_predict(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply_predict(self, replicas, inputs, kwargs):
        """Parrallel prediction."""
        return parallel_apply_predict(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class Clump(nn.Module):
    """Clipping input tensor.

    Args:
        min_v: Min value.
        max_v: Max value.

    """

    def __init__(self, min_v: int=-50, max_v: int=50):
        super(Clump, self).__init__()
        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Forward-pass."""
        x = torch.clamp(x, self.min_v, self.max_v)
        return x


class TextBert(nn.Module):
    """Text data model.

    Class for working with text data based on HuggingFace transformers.

    Args:
        model_name: Transformers model name.
        pooling: Pooling type.

    Note:
        There are different pooling types:

            - cls: Use CLS token for sentence embedding
                from last hidden state.
            - max: Maximum on seq_len dimension for non masked
                inputs from last hidden state.
            - mean: Mean on seq_len dimension for non masked
                inputs from last hidden state.
            - sum: Sum on seq_len dimension for non masked
                inputs from last hidden state.
            - none: Without pooling for seq2seq models.

    """
    _poolers = {'cls', 'max', 'mean', 'sum', 'none'}

    def __init__(self, model_name: str='bert-base-uncased', pooling: str='cls'):
        super(TextBert, self).__init__()
        if pooling not in self._poolers:
            raise ValueError('pooling - {} - not in the list of available types {}'.format(pooling, self._poolers))
        self.transformer = AutoModel.from_pretrained(model_name)
        self.n_out = self.transformer.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU(inplace=True)
        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        encoded_layers, _ = self.transformer(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], token_type_ids=inp.get('token_type_ids'), return_dict=False)
        encoded_layers = self.pooling(encoded_layers, inp['attention_mask'].unsqueeze(-1).bool())
        mean_last_hidden_state = self.activation(encoded_layers)
        mean_last_hidden_state = self.dropout(mean_last_hidden_state)
        return mean_last_hidden_state


class CatEmbedder(nn.Module):
    """Category data model.

    Args:
        cat_dims: Sequence with number of unique categories
            for category features.
        emb_dropout: Dropout probability.
        emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        max_emb_size: Max embedding size.

    """

    def __init__(self, cat_dims: Sequence[int], emb_dropout: bool=0.1, emb_ratio: int=3, max_emb_size: int=50):
        super(CatEmbedder, self).__init__()
        emb_dims = [(int(x), int(min(max_emb_size, max(1, (x + 1) // emb_ratio)))) for x in cat_dims]
        self.no_of_embs = sum([y for x, y in emb_dims])
        assert self.no_of_embs != 0, 'The input is empty.'
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.no_of_embs

    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        output = torch.cat([emb_layer(inp['cat'][:, i]) for i, emb_layer in enumerate(self.emb_layers)], dim=1)
        output = self.emb_dropout_layer(output)
        return output


class ContEmbedder(nn.Module):
    """Numeric data model.

    Class for working with numeric data.

    Args:
        num_dims: Sequence with number of numeric features.
        input_bn: Use 1d batch norm for input data.

    """

    def __init__(self, num_dims: int, input_bn: bool=True):
        super(ContEmbedder, self).__init__()
        self.n_out = num_dims
        self.bn = None
        if input_bn:
            self.bn = nn.BatchNorm1d(num_dims)
        assert num_dims != 0, 'The input is empty.'

    def get_out_shape(self) ->int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        output = inp['cont']
        if self.bn is not None:
            output = self.bn(output)
        return output


class ArgsWrapper:
    """Wrapper - ignore sample_weight if metric not accepts.

    Args:
        func: Metric function.
        metric_params: Additional metric parameters.

    """

    def __init__(self, func: Callable, metric_params: dict):
        keys = inspect.signature(func).parameters
        self.flg = 'sample_weight' in keys
        self.func = partial(func, **metric_params)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Calculate metric value.

        If the metric does not include weights, then they are ignored.

        Args:
            y_true: Ground truth target values.
            y_pred: Estimated target values.
            sample_weight: Sample weights.

        Returns:
            Metric value.

        """
        if self.flg:
            return self.func(y_true, y_pred, sample_weight=sample_weight)
        return self.func(y_true, y_pred)


class MetricFunc:
    """Wrapper for metric.

    Args:
        metric_func: Callable metric function.
        m: Multiplier for metric value.
        bw_func: Backward function.

    """

    def __init__(self, metric_func, m, bw_func):
        self.metric_func = metric_func
        self.m = m
        self.bw_func = bw_func

    def __call__(self, y_true, y_pred, sample_weight=None) ->float:
        """Calculate metric."""
        y_pred = self.bw_func(y_pred)
        try:
            val = self.metric_func(y_true, y_pred, sample_weight=sample_weight)
        except TypeError:
            val = self.metric_func(y_true, y_pred)
        return val * self.m


class BestClassBinaryWrapper:
    """Metric wrapper to get best class prediction instead of probs.

    There is cut-off for prediction by ``0.5``.

    Args:
        func: Metric function. Function format:
            func(y_pred, y_true, weights, \\*\\*kwargs).

    """

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None, **kwargs):
        """Calculate metric."""
        y_pred = (y_pred > 0.5).astype(np.float32)
        return self.func(y_true, y_pred, sample_weight=sample_weight, **kwargs)


class BestClassMulticlassWrapper:
    """Metric wrapper to get best class prediction instead of probs for multiclass.

    Prediction provides by argmax.

    Args:
        func: Metric function. Function format:
            func(y_pred, y_true, weights, \\*\\*kwargs)

    """

    def __init__(self, func):
        self.func = func

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None, **kwargs):
        """Calculate metric."""
        y_pred = y_pred.argmax(axis=1).astype(np.float32)
        return self.func(y_true, y_pred, sample_weight=sample_weight, **kwargs)


class F1Factory:
    """Wrapper for :func:`~sklearn.metrics.f1_score` function.

    Args:
        average: Averaging type ('micro', 'macro', 'weighted').

    """

    def __init__(self, average: str='micro'):
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->float:
        """Compute metric.

        Args:
            y_true: Ground truth target values.
            y_pred: Estimated target values.
            sample_weight: Sample weights.

        Returns:
            F1 score of the positive class in binary classification
            or weighted average of the F1 scores of each class
            for the multiclass task.

        """
        return f1_score(y_true, y_pred, sample_weight=sample_weight, average=self.average)


def auc_mu(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None, class_weights: Optional[np.ndarray]=None) ->float:
    """Compute multi-class metric AUC-Mu.

    We assume that confusion matrix full of ones, except diagonal elements.
    All diagonal elements are zeroes.
    By default, for averaging between classes scores we use simple mean.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Not used.
        class_weights: The between classes weight matrix. If ``None``,
            the standard mean will be used. It is expected to be a lower
            triangular matrix (diagonal is also full of zeroes).
            In position (i, j), i > j, there is a partial positive score
            between i-th and j-th classes. All elements must sum up to 1.

    Returns:
        Metric value.

    Note:
        Code was refactored from https://github.com/kleimanr/auc_mu/blob/master/auc_mu.py

    """
    if not isinstance(y_pred, np.ndarray):
        raise TypeError('Expected y_pred to be np.ndarray, got: {}'.format(type(y_pred)))
    if not y_pred.ndim == 2:
        raise ValueError('Expected array with predictions be a 2-dimentional array')
    if not isinstance(y_true, np.ndarray):
        raise TypeError('Expected y_true to be np.ndarray, got: {}'.format(type(y_true)))
    if not y_true.ndim == 1:
        raise ValueError('Expected array with ground truths be a 1-dimentional array')
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError('Expected number of samples in y_true and y_pred be same, got {} and {}, respectively'.format(y_true.shape[0], y_pred.shape[0]))
    uniq_labels = np.unique(y_true)
    n_samples, n_classes = y_pred.shape
    if not np.all(uniq_labels == np.arange(n_classes)):
        raise ValueError('Expected classes encoded values 0, ..., N_classes-1')
    if class_weights is None:
        class_weights = np.tri(n_classes, k=-1)
        class_weights /= class_weights.sum()
    if not isinstance(class_weights, np.ndarray):
        raise TypeError('Expected class_weights to be np.ndarray, got: {}'.format(type(class_weights)))
    if not class_weights.ndim == 2:
        raise ValueError('Expected class_weights to be a 2-dimentional array')
    if not class_weights.shape == (n_classes, n_classes):
        raise ValueError('Expected class_weights size: {}, got: {}'.format((n_classes, n_classes), class_weights.shape))
    confusion_matrix = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    auc_full = 0.0
    for class_i in range(n_classes):
        preds_i = y_pred[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_pred[y_true == class_j]
            n_j = preds_j.shape[0]
            n = n_i + n_j
            tmp_labels = np.zeros((n,), dtype=np.int32)
            tmp_labels[n_i:] = 1
            tmp_pres = np.vstack((preds_i, preds_j))
            v = confusion_matrix[class_i, :] - confusion_matrix[class_j, :]
            scores = np.dot(tmp_pres, v)
            score_ij = roc_auc_score(tmp_labels, scores)
            auc_full += class_weights[class_i, class_j] * score_ij
    return auc_full


def roc_auc_ovr(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None):
    """ROC-AUC One-Versus-Rest.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Weights of samples.

    Returns:
        Metric values.

    """
    return roc_auc_score(y_true, y_pred, sample_weight=sample_weight, multi_class='ovr')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None) ->float:
    """Computes Mean Absolute Percentage error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.

    Returns:
        Metric value.

    """
    err = (y_true - y_pred) / y_true
    err = np.abs(err)
    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()
    return err.mean()


def mean_fair_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None, c: float=0.9) ->float:
    """Computes Mean Fair Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        c: Metric coefficient.

    Returns:
        Metric value.

    """
    x = np.abs(y_pred - y_true) / c
    err = c ** 2 * (x - np.log(x + 1))
    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()
    return err.mean()


def mean_huber_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None, a: float=0.9) ->float:
    """Computes Mean Huber Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        a: Metric coefficient.

    Returns:
        Metric value.

    """
    err = y_pred - y_true
    s = np.abs(err) < a
    err = np.where(s, 0.5 * err ** 2, a * np.abs(err) - 0.5 * a ** 2)
    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()
    return err.mean()


def mean_quantile_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None, q: float=0.9) ->float:
    """Computes Mean Quantile Error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Specify weighted mean.
        q: Metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = np.sign(err)
    err = np.abs(err)
    err = np.where(s > 0, q, 1 - q) * err
    if sample_weight is not None:
        return (err * sample_weight).mean() / sample_weight.mean()
    return err.mean()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]=None):
    """Root mean squared log error.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        sample_weight: Weights of samples.

    Returns:
        Metric values.


    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight))


def infer_gib(metric: Callable) ->bool:
    """Infer greater is better from metric.

    Args:
        metric: Score or loss function.

    Returns:
        ```True``` if grater is better.

    Raises:
        AssertionError: If there is no way to order the predictions.

    """
    label = np.array([0, 1])
    pred = np.array([0.1, 0.9])
    g_val = metric(label, pred)
    b_val = metric(label, pred[::-1])
    assert g_val != b_val, 'Cannot infer greater is better from metric. Should be set manually.'
    return g_val > b_val


class Loss:
    """Loss function with target transformation."""

    @staticmethod
    def _fw_func(target: Any, weights: Any) ->Tuple[Any, Any]:
        """Forward transformation.

        Args:
            target: Ground truth target values.
            weights: Item weights.

        Returns:
            Tuple (target, weights) without transformation.

        """
        return target, weights

    @staticmethod
    def _bw_func(pred: Any) ->Any:
        """Backward transformation for predicted values.

        Args:
            pred: Predicted target values.

        Returns:
            Pred without transformation.

        """
        return pred

    @property
    def fw_func(self):
        """Forward transformation for target values and item weights.

        Returns:
            Callable transformation.

        """
        return self._fw_func

    @property
    def bw_func(self):
        """Backward transformation for predicted values.

        Returns:
            Callable transformation.

        """
        return self._bw_func

    def metric_wrapper(self, metric_func: Callable, greater_is_better: Optional[bool], metric_params: Optional[Dict]=None) ->Callable:
        """Customize metric.

        Args:
            metric_func: Callable metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.

        Returns:
            Callable metric.

        """
        if greater_is_better is None:
            greater_is_better = infer_gib(metric_func)
        m = 2 * float(greater_is_better) - 1
        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)
        return MetricFunc(metric_func, m, self._bw_func)

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool]=None, metric_params: Optional[Dict]=None, task_name: Optional[Dict]=None):
        """Callback metric setter.

        Args:
            metric: Callback metric
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        Note:
            Value of ``task_name`` should be one of following options:

            -  `'binary'`
            - `'reg'`
            - `'multiclass'`

        """
        assert task_name in ['binary', 'reg', 'multiclass'], 'Incorrect task name: {}'.format(task_name)
        self.metric = metric
        if metric_params is None:
            metric_params = {}
        if type(metric) is str:
            metric_dict = _valid_str_metric_names[task_name]
            self.metric_func = self.metric_wrapper(metric_dict[metric], greater_is_better, metric_params)
            self.metric_name = metric
        else:
            self.metric_func = self.metric_wrapper(metric, greater_is_better, metric_params)
            self.metric_name = None


def fw_rmsle(x, y):
    """Function wrapper for rmsle."""
    return np.log1p(x), y


_cb_loss_mapping = {'mse': ('RMSE', None, None), 'mae': ('MAE', None, None), 'logloss': ('Logloss', None, None), 'rmsle': ('RMSE', fw_rmsle, np.expm1), 'mape': ('MAPE', None, None), 'quantile': ('Quantile', None, None), 'fair': ('FairLoss', None, None), 'huber': ('Huber', None, None), 'crossentropy': ('MultiClass', None, None)}


_cb_loss_params_mapping = {'quantile': {'q': 'alpha'}, 'huber': {'a': 'delta'}, 'fair': {'c': 'smoothness'}}


_cb_metric_params_mapping = {'quantile': {'q': 'alpha'}, 'huber': {'a': 'delta'}, 'fair': {'c': 'smoothness'}}


_cb_binary_metrics_dict = {'auc': 'AUC', 'logloss': 'Logloss', 'accuracy': 'Accuracy'}


_cb_multiclass_metrics_dict = {'auc': 'AUC:type=Mu', 'auc_mu': 'AUC:type=Mu', 'accuracy': 'Accuracy', 'crossentropy': 'MultiClass', 'f1_macro': 'TotalF1:average=Macro', 'f1_micro': 'TotalF1:average=Micro', 'f1_weighted': 'TotalF1:average=Weighted'}


_cb_reg_metrics_dict = {'mse': 'RMSE', 'mae': 'MAE', 'r2': 'R2', 'rmsle': 'MSLE', 'mape': 'MAPE', 'quantile': 'Quantile', 'fair': 'FairLoss', 'huber': 'Huber'}


_cb_metrics_dict = {'binary': _cb_binary_metrics_dict, 'reg': _cb_reg_metrics_dict, 'multiclass': _cb_multiclass_metrics_dict}


def cb_str_loss_wrapper(name: str, **params: Optional[Dict]) ->str:
    """CatBoost loss name wrapper, if it has keyword args.  # noqa D403

    Args:
        name: One of CatBoost loss names.
        **params: Additional parameters.

    Returns:
        Wrapped CatBoost loss name.

    """
    return name + ':' + ';'.join([(k + '=' + str(v)) for k, v in params.items()])


class CBLoss(Loss):
    """Loss used for CatBoost.

    Args:
        loss: String with one of default losses.
        loss_params: additional loss parameters.
            Format like in :mod:`lightautoml.tasks.custom_metrics`.
        fw_func: Forward transformation.
            Used for transformation of target and item weights.
        bw_func: Backward transformation.
            Used for predict values transformation.

    """

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict]=None, fw_func: Optional[Callable]=None, bw_func: Optional[Callable]=None):
        self.loss_params = {}
        if loss_params is not None:
            self.loss_params = loss_params
        if type(loss) is str:
            if loss in _cb_loss_mapping:
                loss_name, fw_func, bw_func = _cb_loss_mapping[loss]
                if loss in _cb_loss_params_mapping:
                    mapped_params = {_cb_loss_params_mapping[loss][k]: v for k, v in self.loss_params.items()}
                    self.fobj = None
                    self.fobj_name = cb_str_loss_wrapper(loss_name, **mapped_params)
                else:
                    self.fobj = None
                    self.fobj_name = loss_name
            else:
                raise ValueError('Unexpected loss for catboost')
        else:
            self.fobj = loss
            self.fobj_name = None
        if fw_func is not None:
            self._fw_func = fw_func
        if bw_func is not None:
            self._bw_func = bw_func
        self.fobj_params = {}
        if loss_params is not None:
            self.fobj_params = loss_params
        self.metric = None
        self.metric_name = None

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool]=None, metric_params: Optional[Dict]=None, task_name: str=None):
        """Callback metric setter.

        Args:
            metric: Callback metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task. For now it omitted.

        """
        assert task_name in ['binary', 'reg', 'multiclass'], 'Unknown task name: {}'.format(task_name)
        self.metric_params = {}
        if metric_params is not None:
            self.metric_params = metric_params
        if type(metric) is str:
            self.metric = None
            _metric_dict = _cb_metrics_dict[task_name]
            if metric in _cb_metric_params_mapping:
                metric_params = {_cb_metric_params_mapping[metric][k]: v for k, v in self.metric_params.items()}
                self.metric_name = cb_str_loss_wrapper(_metric_dict[metric], **metric_params)
            else:
                self.metric_name = _metric_dict[metric]
        else:
            self.metric_name = self.fobj_name
            self.metric_params = self.fobj_params
            self.metric = None


class LAMLMetric:
    """Abstract class for metric.

    Metric should be called on dataset.

    """
    greater_is_better = True

    def __call__(self, dataset: 'LAMLDataset', dropna: bool=False):
        """Call metric on dataset.

        Args:
            dataset: Table with data.
            dropna: To ignore NaN in metric calculation.

        Returns:  # noqa DAR202
            Metric value.

        Raises:
            AttributeError: If metric isn't defined.

        """
        assert hasattr(dataset, 'target'), 'Dataset should have target to calculate metric'
        raise NotImplementedError


class LGBFunc:
    """Wrapper of metric function for LightGBM."""

    def __init__(self, metric_func, greater_is_better, bw_func):
        self.metric_func = metric_func
        self.greater_is_better = greater_is_better
        self.bw_func = bw_func

    def __call__(self, pred: np.ndarray, dtrain: lgb.Dataset) ->Tuple[str, float, bool]:
        """Calculate metric."""
        label = dtrain.get_label()
        weights = dtrain.get_weight()
        if label.shape[0] != pred.shape[0]:
            pred = pred.reshape((label.shape[0], -1), order='F')
            label = label.astype(np.int32)
        label = self.bw_func(label)
        pred = self.bw_func(pred)
        try:
            val = self.metric_func(label, pred, sample_weight=weights)
        except TypeError:
            val = self.metric_func(label, pred)
        return 'Opt metric', val, self.greater_is_better


_lgb_force_metric = {'rmsle': ('mse', None, None)}


def softmax_ax1(x: np.ndarray) ->np.ndarray:
    """Softmax columnwise.

    Args:
        x: input.

    Returns:
        softmax values.

    """
    return softmax(x, axis=1)


def lgb_f1_loss_multiclass(preds: np.ndarray, train_data: lgb.Dataset, clip: float=1e-05) ->Tuple[np.ndarray, np.ndarray]:
    """Custom loss for optimizing f1.

    Args:
        preds: Predctions.
        train_data: Dataset in LightGBM format.
        clip: Clump constant.

    Returns:
        Gradient, hessian.

    """
    y_true = train_data.get_label().astype(np.int32)
    preds = preds.reshape((y_true.shape[0], -1), order='F')
    preds = np.clip(softmax_ax1(preds), clip, 1 - clip)
    y_ohe = np.zeros_like(preds)
    np.add.at(y_ohe, (np.arange(y_true.shape[0]), y_true), 1)
    grad = (preds - y_ohe) * preds
    hess = (1 - preds) * preds * np.clip(2 * preds - y_ohe, 0.001, np.inf)
    return grad.reshape((-1,), order='F'), hess.reshape((-1,), order='F')


_lgb_loss_mapping = {'logloss': ('binary', None, None), 'mse': ('regression', None, None), 'mae': ('l1', None, None), 'mape': ('mape', None, None), 'crossentropy': ('multiclass', None, None), 'rmsle': ('mse', fw_rmsle, np.expm1), 'quantile': ('quantile', None, None), 'huber': ('huber', None, None), 'fair': ('fair', None, None), 'f1': (lgb_f1_loss_multiclass, None, softmax_ax1)}


_lgb_loss_params_mapping = {'quantile': {'q': 'alpha'}, 'huber': {'a': 'alpha'}, 'fair_c': {'c': 'fair_c'}}


_lgb_binary_metrics_dict = {'auc': 'auc', 'logloss': 'binary_logloss', 'accuracy': 'binary_error'}


_lgb_reg_metrics_dict = {'mse': 'mse', 'mae': 'mae', 'r2': 'mse', 'rmsle': 'mse', 'mape': 'mape', 'quantile': 'quantile', 'huber': 'huber', 'fair': 'fair'}


logger = logging.getLogger(__name__)


class LGBLoss(Loss):
    """Loss used for LightGBM.

    Args:
        loss: Objective to optimize.
        loss_params: additional loss parameters.
            Format like in :mod:`lightautoml.tasks.custom_metrics`.
        fw_func: forward transformation.
            Used for transformation of target and item weights.
        bw_func: backward transformation.
            Used for predict values transformation.

    Note:
        Loss can be one of the types:

            - Str: one of default losses
                ('auc', 'mse', 'mae', 'logloss', 'accuray', 'r2',
                'rmsle', 'mape', 'quantile', 'huber', 'fair')
                or another lightgbm objective.
            - Callable: custom lightgbm style objective.

    """

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict]=None, fw_func: Optional[Callable]=None, bw_func: Optional[Callable]=None):
        if loss in _lgb_loss_mapping:
            fobj, fw_func, bw_func = _lgb_loss_mapping[loss]
            if type(fobj) is str:
                self.fobj_name = fobj
                self.fobj = None
            else:
                self.fobj_name = None
                self.fobj = fobj
            if self.fobj_name in _lgb_loss_params_mapping:
                param_mapping = _lgb_loss_params_mapping[self.fobj_name]
                loss_params = {param_mapping[x]: loss_params[x] for x in loss_params}
        elif type(loss) is str:
            self.fobj_name = loss
            self.fobj = None
        else:
            self.fobj_name = None
            self.fobj = loss
        if fw_func is not None:
            self._fw_func = fw_func
        if bw_func is not None:
            self._bw_func = bw_func
        self.fobj_params = {}
        if loss_params is not None:
            self.fobj_params = loss_params
        self.metric = None

    def metric_wrapper(self, metric_func: Callable, greater_is_better: Optional[bool], metric_params: Optional[Dict]=None) ->Callable:
        """Customize metric.

        Args:
            metric_func: Callable metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.

        Returns:
            Callable metric, that returns ('Opt metric', value, greater_is_better).

        """
        if greater_is_better is None:
            greater_is_better = infer_gib(metric_func)
        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)
        return LGBFunc(metric_func, greater_is_better, self._bw_func)

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool]=None, metric_params: Optional[Dict]=None, task_name: Optional[str]=None):
        """Callback metric setter.

        Args:
            metric: Callback metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        Note:
            Value of ``task_name`` should be one of following options:

            - `'binary'`
            - `'reg'`
            - `'multiclass'`

        """
        if self.fobj_name in _lgb_force_metric:
            metric, greater_is_better, metric_params = _lgb_force_metric[self.fobj_name]
            logger.info2('For lgbm {0} callback metric switched to {1}'.format(self.fobj_name, metric), UserWarning)
        self.metric_params = {}
        self.metric = metric
        if type(metric) is str:
            if metric_params is not None:
                self.metric_params = metric_params
            _metric_dict = _lgb_metrics_dict[task_name]
            _metric = _metric_dict.get(metric)
            if type(_metric) is str:
                self.metric_name = _metric
                self.feval = None
            else:
                self.metric_name = None
                self.feval = self.metric_wrapper(_metric, greater_is_better, {})
        else:
            self.metric_name = None
            self.feval = self.metric_wrapper(metric, greater_is_better, self.metric_params)


_sk_force_metric = {'rmsle': ('mse', None, None)}


_sk_loss_mapping = {'rmsle': ('mse', fw_rmsle, np.expm1)}


class SKLoss(Loss):
    """Loss used for scikit-learn.

    Args:
        loss: One of default loss function.
            Valid are: 'logloss', 'mse', 'crossentropy', 'rmsle'.
        loss_params: Addtional loss parameters.
        fw_func: Forward transformation.
            Used for transformation of target and item weights.
        bw_func: backward transformation.
            Used for predict values transformation.

    """

    def __init__(self, loss: str, loss_params: Optional[Dict]=None, fw_func: Optional[Callable]=None, bw_func: Optional[Callable]=None):
        assert loss in ['logloss', 'mse', 'crossentropy', 'rmsle'], 'Not supported in sklearn in general case.'
        self.flg_regressor = loss in ['mse', 'rmsle']
        if loss in _sk_loss_mapping:
            self.loss, fw_func, bw_func = _sk_loss_mapping[loss]
        else:
            self.loss = loss
            if fw_func is not None:
                self._fw_func = fw_func
            if bw_func is not None:
                self._bw_func = bw_func
        self.loss_params = loss_params

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool]=None, metric_params: Optional[Dict]=None, task_name: Optional[str]=None):
        """Callback metric setter.

        Uses default callback of parent class `Loss`.

        Args:
            metric: Callback metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        """
        if self.loss in _sk_force_metric:
            metric, greater_is_better, metric_params = _sk_force_metric[self.loss]
            logger.info2('For sklearn {0} callback metric switched to {1}'.format(self.loss, metric))
        super().set_callback_metric(metric, greater_is_better, metric_params, task_name)


class SkMetric(LAMLMetric):
    """Abstract class for scikit-learn compatible metric.

    Implements metric calculation in sklearn format on numpy/pandas datasets.


    Args:
        metric: Specifies metric. Format:
            ``func(y_true, y_false, Optional[sample_weight], **kwargs)`` -> `float`.
        name: Name of metric.
        greater_is_better: Whether or not higher metric value is better.
        one_dim: `True` for single class, False for multiclass.
        weighted: Weights of classes.
        **kwargs: Other parameters for metric.

    """

    @property
    def metric(self) ->Callable:
        """Metric function."""
        assert self._metric is not None, 'Metric calculation is not defined'
        return self._metric

    @property
    def name(self) ->str:
        """Name of used metric."""
        if self._name is None:
            return 'AutoML Metric'
        else:
            return self._name

    def __init__(self, metric: Optional[Callable]=None, name: Optional[str]=None, greater_is_better: bool=True, one_dim: bool=True, **kwargs: Any):
        self._metric = metric
        self._name = name
        self.greater_is_better = greater_is_better
        self.one_dim = one_dim
        self.kwargs = kwargs

    def __call__(self, dataset: 'SklearnCompatible', dropna: bool=False) ->float:
        """Implement call sklearn metric on dataset.

        Args:
            dataset: Dataset in Numpy or Pandas format.
            dropna: To ignore NaN in metric calculation.

        Returns:
            Metric value.

        Raises:
            AssertionError: if dataset has no target or
                target specified as one-dimensioned, but it is not.

        """
        assert hasattr(dataset, 'target'), 'Dataset should have target to calculate metric'
        if self.one_dim:
            assert dataset.shape[1] == 1, 'Dataset should have single column if metric is one_dim'
        dataset = dataset.to_numpy()
        y_true = dataset.target
        y_pred = dataset.data
        sample_weight = dataset.weights
        if dropna:
            sl = ~np.isnan(y_pred).any(axis=1)
            y_pred = y_pred[sl]
            y_true = y_true[sl]
            if sample_weight is not None:
                sample_weight = sample_weight[sl]
        if self.one_dim:
            y_pred = y_pred[:, 0]
        value = self.metric(y_true, y_pred, sample_weight=sample_weight)
        sign = 2 * float(self.greater_is_better) - 1
        return value * sign


def torch_f1(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None):
    """Computes F1 macro.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    y_true = y_true[:, 0].type(torch.int64)
    y_true_ohe = torch.zeros_like(y_pred)
    y_true_ohe[range(y_true.shape[0]), y_true] = 1
    tp = y_true_ohe * y_pred
    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(-1)
        sm = sample_weight.mean()
        tp = (tp * sample_weight).mean(dim=0) / sm
        f1 = 2 * tp / ((y_pred * sample_weight).mean(dim=0) / sm + (y_true_ohe * sample_weight).mean(dim=0) / sm + 1e-07)
        return -f1.mean()
    tp = torch.mean(tp, dim=0)
    f1 = 2 * tp / (y_pred.mean(dim=0) + y_true_ohe.mean(dim=0) + 1e-07)
    f1[f1 != f1] = 0
    return -f1.mean()


def torch_fair(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None, c: float=0.9):
    """Computes Mean Fair Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        c: metric coefficient.

    Returns:
        metric value.

    """
    x = torch.abs(y_pred - y_true) / c
    err = c ** 2 * (x - torch.log(x + 1))
    if len(err.shape) == 2:
        err = err.sum(dim=1)
    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()
    return err.mean()


def torch_huber(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None, a: float=0.9):
    """Computes Mean Huber Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        a: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = torch.abs(err) < a
    err = torch.where(s, 0.5 * err ** 2, a * torch.abs(err) - 0.5 * a ** 2)
    if len(err.shape) == 2:
        err = err.sum(dim=1)
    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()
    return err.mean()


def torch_mape(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None):
    """Computes Mean Absolute Percentage Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    err = (y_true - y_pred) / y_true
    err = torch.abs(err)
    if len(err.shape) == 2:
        err = err.sum(dim=1)
    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()
    return err.mean()


def torch_quantile(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None, q: float=0.9):
    """Computes Mean Quantile Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        q: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = err < 0
    err = torch.abs(err)
    err = torch.where(s, err * (1 - q), err * q)
    if len(err.shape) == 2:
        err = err.sum(dim=1)
    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()
    return err.mean()


def torch_rmsle(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor]=None):
    """Computes Root Mean Squared Logarithmic Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    y_pred = torch.log1p(y_pred)
    y_true = torch.log1p(y_true)
    outp = (y_pred - y_true) ** 2
    if len(outp.shape) == 2:
        outp = outp.sum(dim=1)
    if sample_weight is not None:
        outp = outp * sample_weight
        return outp.mean() / sample_weight.mean()
    return outp.mean()


_torch_loss_dict = {'mse': (nn.MSELoss, False, False), 'mae': (nn.L1Loss, False, False), 'logloss': (nn.BCELoss, False, False), 'crossentropy': (nn.CrossEntropyLoss, True, True), 'rmsle': (torch_rmsle, False, False), 'mape': (torch_mape, False, False), 'quantile': (torch_quantile, False, False), 'fair': (torch_fair, False, False), 'huber': (torch_huber, False, False), 'f1': (torch_f1, False, False)}


class TORCHLoss(Loss):
    """Loss used for PyTorch.

    Args:
        loss: name or callable objective function.
        loss_params: additional loss parameters.

    """

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict]=None):
        self.loss_params = {}
        if loss_params is not None:
            self.loss_params = loss_params
        if loss in ['mse', 'mae', 'logloss', 'crossentropy']:
            self.loss = TorchLossWrapper(*_torch_loss_dict[loss], **self.loss_params)
        elif type(loss) is str:
            self.loss = partial(_torch_loss_dict[loss][0], **self.loss_params)
        else:
            self.loss = partial(loss, **self.loss_params)


_default_losses = {'binary': 'logloss', 'reg': 'mse', 'multiclass': 'crossentropy'}


_default_metrics = {'binary': 'auc', 'reg': 'mse', 'multiclass': 'crossentropy'}


_one_dim_output_tasks = ['binary', 'reg']


_valid_loss_args = {'quantile': ['q'], 'huber': ['a'], 'fair': ['c']}


_valid_loss_types = ['lgb', 'sklearn', 'torch', 'cb']


_valid_metric_args = {'quantile': ['q'], 'huber': ['a'], 'fair': ['c']}


_valid_str_loss_names = {'binary': ['logloss'], 'reg': ['mse', 'mae', 'mape', 'rmsle', 'quantile', 'huber', 'fair'], 'multiclass': ['crossentropy', 'f1']}


_valid_task_names = ['binary', 'reg', 'multiclass']


def infer_gib_multiclass(metric: Callable) ->bool:
    """Infer greater is better from metric.

    Args:
        metric: Metric function. It must take two arguments y_true, y_pred.

    Returns:
        ```True``` if grater is better.

    Raises:
        AssertionError: If there is no way to order the predictions.

    """
    label = np.array([0, 1, 2])
    pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    g_val = metric(label, pred)
    b_val = metric(label, pred[::-1])
    assert g_val != b_val, 'Cannot infer greater is better from metric. Should be set manually.'
    return g_val > b_val


class Task:
    """Specify task (binary classification, multiclass classification, regression), metrics, losses.

    Args:
        name: Task name.
        loss: Objective function or dict of functions.
        loss_params: Additional loss parameters,
            if dict there is no presence check for loss_params.
        metric: String name or callable.
        metric_params: Additional metric parameters.
        greater_is_better: Whether or not higher value is better.

    Note:
        There is 3 different task types:

            - `'binary'` - for binary classification.
            - `'reg'` - for regression.
            - `'multiclass'` - for multiclass classification.

        Avaliable losses for binary task:

            - `'logloss'` - (uses by default) Standard logistic loss.

        Avaliable losses for regression task:

            - `'mse'` - (uses by default) Mean Squared Error.
            - `'mae'` - Mean Absolute Error.
            - `'mape'` - Mean Absolute Percentage Error.
            - `'rmsle'` - Root Mean Squared Log Error.
            - `'huber'` - Huber loss, reqired params:
                ``a`` - threshold between MAE and MSE losses.
            - `'fair'` - Fair loss, required params:
                ``c`` - sets smoothness.
            - `'quantile'` - Quantile loss, required params:
                ``q`` - sets quantile.

        Avaliable losses for multi-classification task:

            - `'crossentropy'` - (uses by default) Standard crossentropy function.
            - `'f1'` - Optimizes F1-Macro Score, now avaliable for
                LightGBM and NN models. Here we implicitly assume
                that the prediction lies not in the set ``{0, 1}``,
                but in the interval ``[0, 1]``.

        Available metrics for binary task:

            - `'auc'` - (uses by default) ROC-AUC score.
            - `'accuracy'` - Accuracy score (uses argmax prediction).
            - `'logloss'` - Standard logistic loss.

        Avaliable metrics for regression task:

            - `'mse'` - (uses by default) Mean Squared Error.
            - `'mae'` - Mean Absolute Error.
            - `'mape'` - Mean Absolute Percentage Error.
            - `'rmsle'` - Root Mean Squared Log Error.
            - `'huber'` - Huber loss, reqired params:
                ``a`` - threshold between MAE and MSE losses.
            - `'fair'` - Fair loss, required params:
                ``c`` - sets smoothness.
            - `'quantile'` - Quantile loss, required params:
                ``q`` - sets quantile.

        Avaliable metrics for multi-classification task:

            - `'crossentropy'` - (uses by default) Standard cross-entropy loss.
            - `'auc'` - ROC-AUC of each class against the rest.
            - `'auc_mu'` - AUC-Mu. Multi-class extension of standard AUC
                for binary classification. In short,
                mean of n_classes * (n_classes - 1) / 2 binary AUCs.
                More info on http://proceedings.mlr.press/v97/kleiman19a/kleiman19a.pdf


    Example:
        >>> task = Task('binary', metric='auc')

    """

    @property
    def name(self) ->str:
        """Name of task."""
        return self._name

    def __init__(self, name: str, loss: Optional[Union[dict, str]]=None, loss_params: Optional[Dict]=None, metric: Optional[Union[str, Callable]]=None, metric_params: Optional[Dict]=None, greater_is_better: Optional[bool]=None):
        assert name in _valid_task_names, 'Invalid task name: {}, allowed task names: {}'.format(name, _valid_task_names)
        self._name = name
        self.losses = {}
        if loss is None:
            loss = _default_losses[self.name]
        if loss_params is None:
            loss_params = {}
        if type(loss) is str:
            if len(loss_params) > 0:
                self._check_loss_from_params(loss, loss_params)
                if loss == metric:
                    metric_params = loss_params
                    logger.info2('As loss and metric are equal, metric params are ignored.')
            else:
                assert loss not in _valid_loss_args, "Loss should be defined with arguments. Ex. loss='quantile', loss_params={'q': 0.7}."
                loss_params = None
            assert loss in _valid_str_loss_names[self.name], 'Invalid loss name: {} for task {}.'.format(loss, self.name)
            for loss_key, loss_factory in zip(['lgb', 'sklearn', 'torch', 'cb'], [LGBLoss, SKLoss, TORCHLoss, CBLoss]):
                try:
                    self.losses[loss_key] = loss_factory(loss, loss_params=loss_params)
                except (AssertionError, TypeError, ValueError):
                    logger.info2("{0} doesn't support in general case {1} and will not be used.".format(loss_key, loss))
            assert len(self.losses) > 0, 'None of frameworks supports {0} loss.'.format(loss)
        elif type(loss) is dict:
            assert len([key for key in loss.keys() if key in _valid_loss_types]) != len(loss), 'Invalid loss key.'
            self.losses = loss
        else:
            raise TypeError('Loss passed incorrectly.')
        if metric is None:
            metric = _default_metrics[self.name]
        self.metric_params = {}
        if metric_params is not None:
            self.metric_params = metric_params
        if type(metric) is str:
            self._check_metric_from_params(metric, self.metric_params)
            metric_func = _valid_str_metric_names[self.name][metric]
            metric_func = partial(metric_func, **self.metric_params)
            self.metric_func = metric_func
            self.metric_name = metric
        else:
            metric = ArgsWrapper(metric, self.metric_params)
            self.metric_params = {}
            self.metric_func = metric
            self.metric_name = None
        if greater_is_better is None:
            infer_gib_fn = infer_gib_multiclass if name == 'multiclass' else infer_gib
            greater_is_better = infer_gib_fn(self.metric_func)
        self.greater_is_better = greater_is_better
        for loss_key in self.losses:
            self.losses[loss_key].set_callback_metric(metric, greater_is_better, self.metric_params, self.name)

    def get_dataset_metric(self) ->LAMLMetric:
        """Create metric for dataset.

        Get metric that is called on dataset.

        Returns:
            Metric in scikit-learn compatible format.

        """
        one_dim = self.name in _one_dim_output_tasks
        dataset_metric = SkMetric(self.metric_func, name=self.metric_name, one_dim=one_dim, greater_is_better=self.greater_is_better)
        return dataset_metric

    @staticmethod
    def _check_loss_from_params(loss_name, loss_params):
        if loss_name in _valid_loss_args:
            required_params = set(_valid_loss_args[loss_name])
        else:
            required_params = set()
        given_params = set(loss_params)
        extra_params = given_params - required_params
        assert len(extra_params) == 0, 'For loss {0} given extra params {1}'.format(loss_name, extra_params)
        needed_params = required_params - given_params
        assert len(needed_params) == 0, 'For loss {0} required params {1} are not defined'.format(loss_name, needed_params)

    @staticmethod
    def _check_metric_from_params(metric_name, metric_params):
        if metric_name in _valid_metric_args:
            required_params = set(_valid_loss_args[metric_name])
        else:
            required_params = set()
        given_params = set(metric_params)
        extra_params = given_params - required_params
        assert len(extra_params) == 0, 'For metric {0} given extra params {1}'.format(metric_name, extra_params)
        needed_params = required_params - given_params
        assert len(needed_params) == 0, 'For metric {0} required params {1} are not defined'.format(metric_name, needed_params)


class TorchUniversalModel(nn.Module):
    """Mixed data model.

    Class for preparing input for DL model with mixed data.

    Args:
        loss: Callable torch loss with order of arguments (y_true, y_pred).
        task: Task object.
        n_out: Number of output dimensions.
        cont_embedder: Torch module for numeric data.
        cont_params: Dict with numeric model params.
        cat_embedder: Torch module for category data.
        cat_params: Dict with category model params.
        text_embedder: Torch module for text data.
        text_params: Dict with text model params.
        bias: Array with last hidden linear layer bias.

    """

    def __init__(self, loss: Callable, task: Task, n_out: int=1, cont_embedder: Optional[Any]=None, cont_params: Optional[Dict]=None, cat_embedder: Optional[Any]=None, cat_params: Optional[Dict]=None, text_embedder: Optional[Any]=None, text_params: Optional[Dict]=None, bias: Optional[Sequence]=None):
        super(TorchUniversalModel, self).__init__()
        self.n_out = n_out
        self.loss = loss
        self.task = task
        self.cont_embedder = None
        self.cat_embedder = None
        self.text_embedder = None
        n_in = 0
        if cont_embedder is not None:
            self.cont_embedder = cont_embedder(**cont_params)
            n_in += self.cont_embedder.get_out_shape()
        if cat_embedder is not None:
            self.cat_embedder = cat_embedder(**cat_params)
            n_in += self.cat_embedder.get_out_shape()
        if text_embedder is not None:
            self.text_embedder = text_embedder(**text_params)
            n_in += self.text_embedder.get_out_shape()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc = torch.nn.Linear(n_in, self.n_out)
        if bias is not None:
            bias = torch.Tensor(bias)
            self.fc.bias.data = nn.Parameter(bias)
            self.fc.weight.data = nn.Parameter(torch.zeros(self.n_out, n_in))
        if self.task.name == 'binary' or self.task.name == 'multilabel':
            self.fc = nn.Sequential(self.fc, Clump(), nn.Sigmoid())
        elif self.task.name == 'multiclass':
            self.fc = nn.Sequential(self.fc, Clump(), nn.Softmax(dim=1))

    def forward(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Forward-pass."""
        x = self.predict(inp)
        loss = self.loss(inp['label'].view(inp['label'].shape[0], -1), x, inp.get('weight', None))
        return loss

    def predict(self, inp: Dict[str, torch.Tensor]) ->torch.Tensor:
        """Prediction."""
        outputs = []
        if self.cont_embedder is not None:
            outputs.append(self.cont_embedder(inp))
        if self.cat_embedder is not None:
            outputs.append(self.cat_embedder(inp))
        if self.text_embedder is not None:
            outputs.append(self.text_embedder(inp))
        if len(outputs) > 1:
            output = torch.cat(outputs, dim=1)
        else:
            output = outputs[0]
        logits = self.fc(output)
        return logits.view(logits.shape[0], -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CatLinear,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (Clump,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GumbelTopKSampler,
     lambda: ([], {'T': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequenceClsPooler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequenceIndentityPooler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftSubSampler,
     lambda: ([], {'T': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_sberbank_ai_lab_LightAutoML(_paritybench_base):
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

