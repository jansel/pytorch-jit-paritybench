import sys
_module = sys.modules[__name__]
del sys
adult_script = _module
airbnb_data_preprocessing = _module
airbnb_script = _module
airbnb_script_multiclass = _module
download_images = _module
pytorch_widedeep = _module
callbacks = _module
initializers = _module
losses = _module
metrics = _module
models = _module
_multiple_lr_scheduler = _module
_multiple_optimizer = _module
_multiple_transforms = _module
_warmup = _module
_wd_dataset = _module
deep_dense = _module
deep_image = _module
deep_text = _module
wide = _module
wide_deep = _module
optim = _module
radam = _module
preprocessing = _module
_preprocessors = _module
utils = _module
dense_utils = _module
fastai_transforms = _module
image_utils = _module
text_utils = _module
version = _module
wdtypes = _module
setup = _module
test_du_deep_dense = _module
test_du_deep_image = _module
test_du_deep_text = _module
test_du_wide = _module
test_mc_deep_dense = _module
test_mc_deep_image = _module
test_mc_deep_text = _module
test_mc_wide = _module
test_callbacks = _module
test_data_inputs = _module
test_fit_methods = _module
test_focal_loss = _module
test_initializers = _module
test_metrics = _module
test_warm_up_routines = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


import warnings


from copy import deepcopy


import re


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from sklearn.utils import Bunch


from torch.utils.data import Dataset


from torchvision import models


from sklearn.model_selection import train_test_split


from torch.utils.data import DataLoader


import math


from torch.optim.optimizer import Optimizer


from torch.nn import Module


from torch import Tensor


from torchvision.transforms import CenterCrop


from torchvision.transforms import ColorJitter


from torchvision.transforms import Compose


from torchvision.transforms import FiveCrop


from torchvision.transforms import Grayscale


from torchvision.transforms import Lambda


from torchvision.transforms import LinearTransformation


from torchvision.transforms import Pad


from torchvision.transforms import RandomAffine


from torchvision.transforms import RandomApply


from torchvision.transforms import RandomChoice


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomGrayscale


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomOrder


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import RandomRotation


from torchvision.transforms import RandomSizedCrop


from torchvision.transforms import RandomVerticalFlip


from torchvision.transforms import Resize


from torchvision.transforms import Scale


from torchvision.transforms import TenCrop


from torchvision.transforms import ToPILImage


from torch.utils.data.dataloader import DataLoader


from torch.optim.lr_scheduler import _LRScheduler


from typing import List


from typing import Any


from typing import Union


from typing import Dict


from typing import Callable


from typing import Optional


from typing import Tuple


from typing import Generator


from typing import Collection


from typing import Iterable


from typing import Match


from typing import Iterator


from scipy.sparse.csr import csr_matrix as sparse_matrix


from types import SimpleNamespace


import string


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import CyclicLR


from itertools import chain


from copy import deepcopy as c


use_cuda = torch.cuda.is_available()


class FocalLoss(nn.Module):

    def __init__(self, alpha: float=0.25, gamma: float=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def get_weight(self, x: Tensor, t: Tensor) ->Tensor:
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (w * (1 - pt).pow(self.gamma)).detach()

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        if input.size(1) == 1:
            input = torch.cat([1 - input, input], axis=1)
            num_class = 2
        else:
            num_class = input.size(1)
        binary_target = torch.eye(num_class)[target.long()]
        if use_cuda:
            binary_target = binary_target
        binary_target = binary_target.contiguous()
        weight = self.get_weight(input, binary_target)
        return F.binary_cross_entropy_with_logits(input, binary_target, weight, reduction='mean')


def dense_layer(inp: int, out: int, p: float=0.0, bn=False):
    layers = [nn.Linear(inp, out), nn.LeakyReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm1d(out))
    layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)


class DeepDense(nn.Module):
    """Dense branch of the deep side of the model. This class combines embedding
    representations of the categorical features with numerical (aka
    continuous) features. These are then passed through a series of dense
    layers.

    Parameters
    ----------
    deep_column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the DeepDense model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    hidden_layers: List
        List with the number of neurons per dense layer. e.g: [64,32]
    batchnorm: Boolean
        Boolean indicating whether or not to include batch normalizatin in the
        dense layers
    dropout: List, Optional
        List with the dropout between the dense layers. e.g: [0.5,0.5]
    embeddings_input: List, Optional
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...]
    embed_p: float
        embeddings dropout
    continuous_cols: List, Optional
        List with the name of the numeric (aka continuous) columns

    **Either embeddings_input or continuous_cols (or both) should be passed to the
    model

    Attributes
    ----------
    dense: nn.Sequential
        model of dense layers that will receive the concatenation of the
        embeddings and the continuous columns
    embed_layers: nn.ModuleDict
        ModuleDict with the embedding layers
    output_dim: Int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import DeepDense
    >>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> deep_column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = DeepDense(hidden_layers=[8,4], deep_column_idx=deep_column_idx, embed_input=embed_input)
    >>> model(X_deep)
    tensor([[ 3.4470e-02, -2.0089e-03,  4.7983e-02,  3.3500e-01],
            [ 1.4329e-02, -1.3800e-03, -3.3617e-04,  4.1046e-01],
            [-3.3546e-04,  3.2413e-02, -4.1198e-03,  4.8717e-01],
            [-6.7882e-04,  7.9103e-03, -1.9960e-03,  4.2134e-01],
            [ 6.7187e-02, -1.2821e-03, -3.0960e-04,  3.6123e-01]],
           grad_fn=<LeakyReluBackward1>)
    """

    def __init__(self, deep_column_idx: Dict[str, int], hidden_layers: List[int], batchnorm: bool=False, dropout: Optional[List[float]]=None, embed_input: Optional[List[Tuple[str, int, int]]]=None, embed_p: float=0.0, continuous_cols: Optional[List[str]]=None):
        super(DeepDense, self).__init__()
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({('emb_layer_' + col): nn.Embedding(val, dim) for col, val, dim in self.embed_input})
            self.embed_dropout = nn.Dropout(embed_p)
            emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])
        else:
            emb_inp_dim = 0
        if self.continuous_cols is not None:
            cont_inp_dim = len(self.continuous_cols)
        else:
            cont_inp_dim = 0
        input_dim = emb_inp_dim + cont_inp_dim
        hidden_layers = [input_dim] + hidden_layers
        if not dropout:
            dropout = [0.0] * len(hidden_layers)
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module('dense_layer_{}'.format(i - 1), dense_layer(hidden_layers[i - 1], hidden_layers[i], dropout[i - 1], batchnorm))
        self.output_dim = hidden_layers[-1]

    def forward(self, X: Tensor) ->Tensor:
        if self.embed_input is not None:
            x = [self.embed_layers['emb_layer_' + col](X[:, (self.deep_column_idx[col])].long()) for col, _, _ in self.embed_input]
            x = torch.cat(x, 1)
            x = self.embed_dropout(x)
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            x_cont = X[:, (cont_idx)].float()
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont
        return self.dense(x)


def conv_layer(ni: int, nf: int, ks: int=3, stride: int=1, maxpool: bool=True, adaptiveavgpool: bool=False):
    layer = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=ks, bias=True, stride=stride, padding=ks // 2), nn.BatchNorm2d(nf, momentum=0.01), nn.LeakyReLU(negative_slope=0.1, inplace=True))
    if maxpool:
        layer.add_module('maxpool', nn.MaxPool2d(2, 2))
    if adaptiveavgpool:
        layer.add_module('adaptiveavgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return layer


class DeepImage(nn.Module):
    """
    Standard image classifier/regressor using a pretrained network freezing
    some of the first layers, or all layers. I use Resnets which have 9
    "components" before the last dense layers.
    The first 4 are: conv->batchnorm->relu->maxpool.
    After that we have 4 additional 'layers' (resnet blocks) (so 4+4=8)
    comprised by a series of convolutions and then the final AdaptiveAvgPool2d
    (8+1=9). The parameter freeze sets the layers to be frozen. For example,
    freeze=6 will freeze all but the last 2 Layers and AdaptiveAvgPool2d
    layer. If freeze='all' it freezes the entire network. In addition, there
    is the option to add a Fully Connected (FC) set of dense layers (FC-Head,
    referred as 'imagehead') on top of the stack of RNNs

    Parameters
    ----------
    pretrained: Boolean
        Indicates whether or not we use a pretrained Resnet network or a
        series of conv layers (see conv_layer function)
    resnet: Int
        The resnet architecture. One of 18, 34 or 50
    freeze: Int, Str
        number of layers to freeze. If int must be less than 8. The only
        string allowed is 'all' which will freeze the entire network
    head_layers: List, Optional
        List with the sizes of the stacked dense layers in the head
        e.g: [128, 64]
    head_dropout: List, Optional
        List with the dropout between the dense layers. e.g: [0.5, 0.5].
    head_batchnorm: Boolean, Optional
        Boolean indicating whether or not to include batch normalizatin in the
        dense layers that form the imagehead

    Attributes
    ----------
    backbone: nn.Sequential
        Sequential stack of CNNs comprising the 'backbone' of the network
    imagehead: nn.Sequential
        Sequential stack of dense layers comprising the FC-Head (aka imagehead)
    output_dim: Int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import DeepImage
    >>> X_img = torch.rand((2,3,224,224))
    >>> model = DeepImage(head_layers=[512, 64, 8])
    >>> model(X_img)
    tensor([[ 7.7234e-02,  8.0923e-02,  2.3077e-01, -5.1122e-03, -4.3018e-03,
              3.1193e-01,  3.0780e-01,  6.5098e-01],
            [ 4.6191e-02,  6.7856e-02, -3.0163e-04, -3.7670e-03, -2.1437e-03,
              1.5416e-01,  3.9227e-01,  5.5048e-01]], grad_fn=<LeakyReluBackward1>)
    """

    def __init__(self, pretrained: bool=True, resnet: int=18, freeze: Union[str, int]=6, head_layers: Optional[List[int]]=None, head_dropout: Optional[List[float]]=None, head_batchnorm: Optional[bool]=False):
        super(DeepImage, self).__init__()
        self.head_layers = head_layers
        if pretrained:
            if resnet == 18:
                vision_model = models.resnet18(pretrained=True)
            elif resnet == 34:
                vision_model = models.resnet34(pretrained=True)
            elif resnet == 50:
                vision_model = models.resnet50(pretrained=True)
            backbone_layers = list(vision_model.children())[:-1]
            if isinstance(freeze, str):
                frozen_layers = []
                for layer in backbone_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen_layers.append(layer)
                self.backbone = nn.Sequential(*frozen_layers)
            if isinstance(freeze, int):
                assert freeze < 8, "freeze' must be less than 8 when using resnet architectures"
                frozen_layers = []
                trainable_layers = backbone_layers[freeze:]
                for layer in backbone_layers[:freeze]:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen_layers.append(layer)
                backbone_layers = frozen_layers + trainable_layers
                self.backbone = nn.Sequential(*backbone_layers)
        else:
            self.backbone = nn.Sequential(conv_layer(3, 64, 3), conv_layer(64, 128, 1, maxpool=False), conv_layer(128, 256, 1, maxpool=False), conv_layer(256, 512, 1, maxpool=False, adaptiveavgpool=True))
        self.output_dim = 512
        if self.head_layers is not None:
            assert self.head_layers[0] == self.output_dim, 'The output dimension from the backbone ({}) is not consistent with the expected input dimension ({}) of the fc-head'.format(self.output_dim, self.head_layers[0])
            if not head_dropout:
                head_dropout = [0.0] * len(head_layers)
            self.imagehead = nn.Sequential()
            for i in range(1, len(head_layers)):
                self.imagehead.add_module('dense_layer_{}'.format(i - 1), dense_layer(head_layers[i - 1], head_layers[i], head_dropout[i - 1], head_batchnorm))
            self.output_dim = head_layers[-1]

    def forward(self, x: Tensor) ->Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        if self.head_layers is not None:
            out = self.imagehead(x)
            return out
        else:
            return x


class DeepText(nn.Module):
    """Standard text classifier/regressor comprised by a stack of RNNs (LSTMs).
    In addition, there is the option to add a Fully Connected (FC) set of dense
    layers (FC-Head, referred as 'texthead') on top of the stack of RNNs

    Parameters
    ----------
    vocab_size: Int
        number of words in the vocabulary
    hidden_dim: Int
        number of features in the hidden state h of the LSTM
    n_layers: Int
        number of recurrent layers
    rnn_dropout: Int
        dropout for the dropout layer on the outputs of each LSTM layer except
        the last layer
    bidirectional: Boolean
        indicates whether the staked RNNs are bidirectional
    padding_idx: Int
        index of the padding token in the padded-tokenised sequences. default:
        1. I use the fastai Tokenizer where the token index 0 is reserved for
        the  unknown word token
    embed_dim: Int, Optional
        Dimension of the word embedding matrix
    embedding_matrix: np.ndarray, Optional
         Pretrained word embeddings
    head_layers: List, Optional
        List with the sizes of the stacked dense layers in the head
        e.g: [128, 64]
    head_dropout: List, Optional
        List with the dropout between the dense layers. e.g: [0.5, 0.5].
    head_batchnorm: Boolean, Optional
        Whether or not to include batch normalizatin in the dense layers that
        form the texthead

    Attributes
    ----------
    word_embed: nn.Module
        word embedding matrix
    rnn: nn.Module
        Stack of LSTMs
    texthead: nn.Sequential, Optional
        Stack of dense layers
    output_dim: Int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import DeepText
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> model = DeepText(vocab_size=4, hidden_dim=4, n_layers=1, padding_idx=0, embed_dim=4)
    >>> model(X_text)
    tensor([[ 0.0315,  0.0393, -0.0618, -0.0561],
            [-0.0674,  0.0297, -0.1118, -0.0668],
            [-0.0446,  0.0814, -0.0921, -0.0338],
            [-0.0844,  0.0681, -0.1016, -0.0464],
            [-0.0268,  0.0294, -0.0988, -0.0666]], grad_fn=<SelectBackward>)
    """

    def __init__(self, vocab_size: int, hidden_dim: int=64, n_layers: int=3, rnn_dropout: float=0.0, bidirectional: bool=False, padding_idx: int=1, embed_dim: Optional[int]=None, embedding_matrix: Optional[np.ndarray]=None, head_layers: Optional[List[int]]=None, head_dropout: Optional[List[float]]=None, head_batchnorm: Optional[bool]=False):
        super(DeepText, self).__init__()
        if embed_dim is not None and embedding_matrix is not None and not embed_dim == embedding_matrix.shape[1]:
            warnings.warn('the input embedding dimension {} and the dimension of the pretrained embeddings {} do not match. The pretrained embeddings dimension ({}) will be used'.format(embed_dim, embedding_matrix.shape[1], embedding_matrix.shape[1]), UserWarning)
        self.bidirectional = bidirectional
        self.head_layers = head_layers
        if isinstance(embedding_matrix, np.ndarray):
            assert embedding_matrix.dtype == 'float32', "'embedding_matrix' must be of dtype 'float32', got dtype '{}'".format(str(embedding_matrix.dtype))
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx=padding_idx)
            self.word_embed.weight = nn.Parameter(torch.tensor(embedding_matrix), requires_grad=True)
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=rnn_dropout, batch_first=True)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if self.head_layers is not None:
            assert self.head_layers[0] == self.output_dim, 'The hidden dimension from the stack or RNNs ({}) is not consistent with the expected input dimension ({}) of the fc-head'.format(self.output_dim, self.head_layers[0])
            if not head_dropout:
                head_dropout = [0.0] * len(head_layers)
            self.texthead = nn.Sequential()
            for i in range(1, len(head_layers)):
                self.texthead.add_module('dense_layer_{}'.format(i - 1), dense_layer(head_layers[i - 1], head_layers[i], head_dropout[i - 1], head_batchnorm))
            self.output_dim = head_layers[-1]

    def forward(self, X: Tensor) ->Tensor:
        embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        if self.bidirectional:
            last_h = torch.cat((h[-2], h[-1]), dim=1)
        else:
            last_h = h[-1]
        if self.head_layers is not None:
            out = self.texthead(last_h)
            return out
        else:
            return last_h


class Wide(nn.Module):
    """simple linear layer between the one-hot encoded wide input and the output
    neuron.

    Parameters
    ----------
    wide_dim: Int
        size of the input tensor
    output_dim: Int
        size of the ouput tensor

    Attributes
    ----------
    wide_linear: nn.Module
        the linear layer that comprises the wide branch of the model

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Wide
    >>> X = torch.empty(4, 4).random_(2)
    >>> wide = Wide(wide_dim=X.size(0), output_dim=1)
    >>> wide(X)
    tensor([[-0.8841],
            [-0.8633],
            [-1.2713],
            [-0.4762]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, wide_dim: int, output_dim: int=1):
        super(Wide, self).__init__()
        self.wide_linear = nn.Linear(wide_dim, output_dim)

    def forward(self, X: Tensor) ->Tensor:
        out = self.wide_linear(X.float())
        return out


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model: Any):
        self.model = model

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict]=None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict]=None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict]=None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict]=None):
        pass

    def on_train_begin(self, logs: Optional[Dict]=None):
        pass

    def on_train_end(self, logs: Optional[Dict]=None):
        pass


def _get_current_time():
    return datetime.datetime.now().strftime('%B %d, %Y - %I:%M%p')


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """

    def __init__(self, callbacks: Optional[List]=None, queue_length: int=10):
        instantiated_callbacks = []
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    instantiated_callbacks.append(callback())
                else:
                    instantiated_callbacks.append(callback)
        self.callbacks = [c for c in instantiated_callbacks]
        self.queue_length = queue_length

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model: Any):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs: Optional[Dict]=None):
        logs = logs or {}
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


class History(Callback):
    """
    Callback that records events into a `History` object.
    """

    def on_train_begin(self, logs: Optional[Dict]=None):
        self.epoch: List[int] = []
        self._history: Dict[str, List[float]] = {}

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict]=None):
        logs = deepcopy(logs) or {}
        for k, v in logs.items():
            self._history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict]=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self._history.setdefault(k, []).append(v)


class Initializer(object):

    def __call__(self, model: nn.Module):
        raise NotImplementedError('Initializer must implement this method')


LRScheduler = _LRScheduler


class Metric(object):

    def __init__(self):
        self._name = ''

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        raise NotImplementedError('Custom Metrics must implement this function')


class MultipleMetrics(object):

    def __init__(self, metrics: List[Metric], prefix: str=''):
        instantiated_metrics = []
        for metric in metrics:
            if isinstance(metric, type):
                instantiated_metrics.append(metric())
            else:
                instantiated_metrics.append(metric)
        self._metrics = instantiated_metrics
        self.prefix = prefix

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) ->Dict:
        logs = {}
        for metric in self._metrics:
            logs[self.prefix + metric._name] = metric(y_pred, y_true)
        return logs


class MetricCallback(Callback):

    def __init__(self, container: MultipleMetrics):
        self.container = container

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict]=None):
        self.container.reset()


class MultipleInitializer(object):

    def __init__(self, initializers: Dict[str, Initializer], verbose=True):
        self.verbose = verbose
        instantiated_initializers = {}
        for model_name, initializer in initializers.items():
            if isinstance(initializer, type):
                instantiated_initializers[model_name] = initializer()
            else:
                instantiated_initializers[model_name] = initializer
        self._initializers = instantiated_initializers

    def apply(self, model: nn.Module):
        for name, child in model.named_children():
            try:
                self._initializers[name](child)
            except:
                if self.verbose:
                    warnings.warn('No initializer found for {}'.format(name))


class MultipleLRScheduler(object):

    def __init__(self, scheds: Dict[str, LRScheduler]):
        self._schedulers = scheds

    def step(self):
        for _, sc in self._schedulers.items():
            sc.step()


class MultipleOptimizer(object):

    def __init__(self, opts: Dict[str, Optimizer]):
        self._optimizers = opts

    def zero_grad(self):
        for _, op in self._optimizers.items():
            op.zero_grad()

    def step(self):
        for _, op in self._optimizers.items():
            op.step()


Transforms = Union[CenterCrop, ColorJitter, Compose, FiveCrop, Grayscale, Lambda, LinearTransformation, Normalize, Pad, RandomAffine, RandomApply, RandomChoice, RandomCrop, RandomGrayscale, RandomHorizontalFlip, RandomOrder, RandomResizedCrop, RandomRotation, RandomSizedCrop, RandomVerticalFlip, Resize, Scale, TenCrop, ToPILImage, ToTensor]


class MultipleTransforms(object):

    def __init__(self, transforms: List[Transforms]):
        instantiated_transforms = []
        for transform in transforms:
            if isinstance(transform, type):
                instantiated_transforms.append(transform())
            else:
                instantiated_transforms.append(transform)
        self._transforms = instantiated_transforms

    def __call__(self):
        return Compose(self._transforms)


class WarmUp(object):
    """
    'Warm up' methods to be applied to the individual models before the joined
    training. There are 3 warm up routines available:
    1) Warm up all trainable layers at once
    2) Gradual warm up inspired by the work of Felbo et al., 2017
    3) Gradual warm up inspired by the work of Howard & Ruder 2018

    The structure of the code in this class is designed to be instantiated within
    the class WideDeep. This is not ideal, but represents a compromise towards
    implementing a 'warm up' functionality for the current overall structure of
    the package without having to re-structure most of the existing code.

    Parameters
    ----------
    activation_fn: Any
       any function with the same strucure as '_activation_fn' in the main class
       WideDeep at pytorch_widedeep.models.wide_deep
    loss_fn: Any
       any function with the same strucure as '_loss_fn' in the main class WideDeep
       at pytorch_widedeep.models.wide_deep
    metric: Metric
       object of class Metric (see Metric in pytorch_widedeep.metrics)
    method: str
       one of 'binary', 'regression' or 'multiclass'
    verbose: Boolean
    """

    def __init__(self, activation_fn: Any, loss_fn: Any, metric: Union[Metric, MultipleMetrics], method: str, verbose: int):
        super(WarmUp, self).__init__()
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.metric = metric
        self.method = method
        self.verbose = verbose

    def warm_all(self, model: nn.Module, model_name: str, loader: DataLoader, n_epochs: int, max_lr: float):
        """
        Warm up all trainable layers in a model using a one cyclic learning rate
        with a triangular pattern. This is refereed as Slanted Triangular learing
        rate in Jeremy Howard & Sebastian Ruder 2018
        (https://arxiv.org/abs/1801.06146). The cycle is described as follows:
        1-The learning rate will gradually increase for 10% of the training steps
            from max_lr/10 to max_lr.
        2-It will then gradually decrease to max_lr/10 for the remaining 90% of the
            steps.
        The optimizer used in the process is AdamW

        Parameters:
        ----------
        model: nn.Module
            nn.Module object containing one the WideDeep model components (wide,
            deepdense, deeptext or deepimage)
        model_name: Str
            string indicating the model name to access the corresponding parameters.
            One of 'wide', 'deepdense', 'deeptext' or 'deepimage'
        loader: DataLoader
            Pytorch DataLoader containing the data used to warm up
        n_epochs: Int
            number of epochs used to warm up the model
        max_lr: Float
            maximum learning rate value during the triangular cycle.
        """
        if self.verbose:
            None
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr / 10.0)
        step_size_up, step_size_down = self._steps_up_down(len(loader), n_epochs)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=max_lr / 10.0, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, cycle_momentum=False)
        self._warm(model, model_name, loader, optimizer, scheduler, n_epochs=n_epochs)

    def warm_gradual(self, model: nn.Module, model_name: str, loader: DataLoader, last_layer_max_lr: float, layers: List[nn.Module], routine: str):
        """
        Warm up certain layers within the model following a gradual warm up routine.
        The approaches implemented in this method are inspired by the work of Felbo
        et al., 2017 in their DeepEmoji paper (https://arxiv.org/abs/1708.00524) and
        Howard & Sebastian Ruder 2018 ULMFit paper
        (https://arxiv.org/abs/1801.06146).

        A one cycle triangular learning rate is used. In both Felbo's and Howard's
        routines a gradually decreasing learning rate is used as we go deeper into
        the network. The 'closest' layer to the output neuron(s) will use a maximum
        learning rate of 'last_layer_max_lr'. The learning rate will then decrease by a factor
        of 2.5 per layer

        1) The 'Felbo' routine:
           warm up the first layer in 'layers' for one epoch. Then warm up the next
           layer in 'layers' for one epoch freezing the already warmed up layer(s).
           Repeat untill all individual layers are warmed. Then warm one last epoch
           with all warmed layers trainable
        2) The 'Howard' routine:
           warm up the first layer in 'layers' for one epoch. Then warm the next layer
           in the model for one epoch while keeping the already warmed up layer(s)
           trainable. Repeat.

        Parameters:
        ----------
        model: nn.Module
           nn.Module object containing one the WideDeep model components (wide,
           deepdense, deeptext or deepimage)
        model_name: Str
           string indicating the model name to access the corresponding parameters.
           One of 'wide', 'deepdense', 'deeptext' or 'deepimage'
        loader: DataLoader
           Pytorch DataLoader containing the data to warm up with.
        last_layer_max_lr: Float
           maximum learning rate value during the triangular cycle for the layer
           closest to the output neuron(s). Deeper layers in 'model' will be trained
           with a gradually descending learning rate. The descending factor is fixed
           and is 2.5
        layers: List
           List of nn.Module objects containing the layers that will be warmed up.
           This must be in 'WARM-UP ORDER'.
        routine: str
           one of 'howard' or 'felbo'
        """
        model.train()
        step_size_up, step_size_down = self._steps_up_down(len(loader))
        original_setup = {}
        for n, p in model.named_parameters():
            original_setup[n] = p.requires_grad
        layers_max_lr = [last_layer_max_lr] + [(last_layer_max_lr / (2.5 * n)) for n in range(1, len(layers))]
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False
        if routine == 'howard':
            params: List = []
            max_lr: List = []
            base_lr: List = []
        for i, (lr, layer) in enumerate(zip(layers_max_lr, layers)):
            if self.verbose:
                None
            for p in layer.parameters():
                p.requires_grad = True
            if routine == 'felbo':
                params, max_lr, base_lr = layer.parameters(), lr, lr / 10.0
            elif routine == 'howard':
                params += [{'params': layer.parameters(), 'lr': lr / 10.0}]
                max_lr += [lr]
                base_lr += [lr / 10.0]
            optimizer = torch.optim.AdamW(params)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, cycle_momentum=False)
            self._warm(model, model_name, loader, optimizer, scheduler)
            if routine == 'felbo':
                for p in layer.parameters():
                    p.requires_grad = False
        if routine == 'felbo':
            if self.verbose:
                None
            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True
            params, max_lr, base_lr = [], [], []
            for lr, layer in zip(layers_max_lr, layers):
                params += [{'params': layer.parameters(), 'lr': lr / 10.0}]
                max_lr += [lr]
                base_lr += [lr / 10.0]
            optimizer = torch.optim.AdamW(params)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, cycle_momentum=False)
            self._warm(model, model_name, loader, optimizer, scheduler)
        for n, p in model.named_parameters():
            p.requires_grad = original_setup[n]

    def _warm(self, model: nn.Module, model_name: str, loader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, n_epochs: int=1):
        """
        Standard Pytorch training loop
        """
        steps = len(loader)
        for epoch in range(n_epochs):
            running_loss = 0.0
            with trange(steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, target) in zip(t, loader):
                    t.set_description('epoch %i' % (epoch + 1))
                    X = data[model_name] if use_cuda else data[model_name]
                    y = target.float() if self.method != 'multiclass' else target
                    y = y if use_cuda else y
                    optimizer.zero_grad()
                    y_pred = self.activation_fn(model(X))
                    loss = self.loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    running_loss += loss.item()
                    avg_loss = running_loss / (batch_idx + 1)
                    if self.metric is not None:
                        acc = self.metric(y_pred, y)
                        t.set_postfix(metrics=acc, loss=avg_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(avg_loss))

    def _steps_up_down(self, steps: int, n_epochs: int=1) ->Tuple[int, int]:
        """
        Calculate the number of steps up and down during the one cycle warm up for a
        given number of epochs

        Parameters:
        ----------
        steps: Int
            steps per epoch
        n_epochs: Int. Default=1
            number of warm up epochs

        Returns:
        -------
        up, down: Tuple, Int
            number of steps increasing/decreasing the learning rate during the cycle
        """
        up = round(steps * n_epochs * 0.1)
        down = steps * n_epochs - up
        return up, down


class WideDeepDataset(Dataset):
    """Dataset object to load WideDeep data to the model

    Parameters
    ----------
    X_wide: np.ndarray, scipy csr sparse matrix.
        wide input.Note that if a sparse matrix is passed to the
        WideDeepDataset class, the loading process will be notably slow since
        the transformation to a dense matrix is done on an index basis 'on the
        fly'. At the moment this is the best option given the current support
        offered for sparse tensors for pytorch.
    X_deep: np.ndarray
        deepdense input
    X_text: np.ndarray
        deeptext input
    X_img: np.ndarray
        deepimage input
    target: np.ndarray
    transforms: MultipleTransforms() object (which is in itself a torchvision
        Compose). See in models/_multiple_transforms.py
    """

    def __init__(self, X_wide: Union[np.ndarray, sparse_matrix], X_deep: np.ndarray, target: Optional[np.ndarray]=None, X_text: Optional[np.ndarray]=None, X_img: Optional[np.ndarray]=None, transforms: Optional[Any]=None):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [tr.__class__.__name__ for tr in self.transforms.transforms]
        else:
            self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx: int):
        if isinstance(self.X_wide, sparse_matrix):
            X = Bunch(wide=np.array(self.X_wide[idx].todense()).squeeze())
        else:
            X = Bunch(wide=self.X_wide[idx])
        X.deepdense = self.X_deep[idx]
        if self.X_text is not None:
            X.deeptext = self.X_text[idx]
        if self.X_img is not None:
            xdi = self.X_img[idx]
            if 'int' in str(xdi.dtype) and 'uint8' != str(xdi.dtype):
                xdi = xdi.astype('uint8')
            if 'float' in str(xdi.dtype) and 'float32' != str(xdi.dtype):
                xdi = xdi.astype('float32')
            if not self.transforms or 'ToTensor' not in self.transforms_names:
                xdi = xdi.transpose(2, 0, 1)
                if 'int' in str(xdi.dtype):
                    xdi = (xdi / xdi.max()).astype('float32')
            if 'ToTensor' in self.transforms_names:
                xdi = self.transforms(xdi)
            elif self.transforms:
                xdi = self.transforms(torch.tensor(xdi))
            X.deepimage = xdi
        if self.Y is not None:
            y = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_deep)


class WideDeep(nn.Module):
    """ Main collector class to combine all Wide, DeepDense, DeepText and
    DeepImage models. There are two options to combine these models.
    1) Directly connecting the output of the models to an ouput neuron(s).
    2) Adding a FC-Head on top of the deep models. This FC-Head will combine
    the output form the DeepDense, DeepText and DeepImage and will be then
    connected to the output neuron(s)

    Parameters
    ----------
    wide: nn.Module
        Wide model. I recommend using the Wide class in this package. However,
        can a custom model as long as is  consistent with the required
        architecture.
    deepdense: nn.Module
        'Deep dense' model consisting in a series of categorical features
        represented by embeddings combined with numerical (aka continuous)
        features. I recommend using the DeepDense class in this package.
        However, a custom model as long as is  consistent with the required
        architecture.
    deeptext: nn.Module, Optional
        'Deep text' model for the text input. Must be an object of class
        DeepText or a custom model as long as is consistent with the required
        architecture.
    deepimage: nn.Module, Optional
        'Deep Image' model for the images input. Must be an object of class
        DeepImage or a custom model as long as is consistent with the required
        architecture.
    deephead: nn.Module, Optional
        Dense model consisting in a stack of dense layers. The FC-Head
    head_layers: List, Optional
        Sizes of the stacked dense layers in the fc-head e.g: [128, 64]
    head_dropout: List, Optional
        Dropout between the dense layers. e.g: [0.5, 0.5]
    head_batchnorm: Boolean, Optional
        Specifies if batch normalizatin should be included in the dense layers
        that form the texthead
    output_dim: Int
        Size of the final layer. 1 for regression and binary classification or
        'n_class' for multiclass classification

    ** While I recommend using the Wide and DeepDense classes within this
    package when building the corresponding model components, it is very likely
    that the user will want to use custom text and image models. That is perfectly
    possible. Simply, build them and pass them as the corresponding parameters.
    Note that the custom models MUST return a last layer of activations (i.e. not
    the final prediction) so that  these activations are collected by WideDeep and
    combined accordingly. In  addition, the models MUST also contain an attribute
    'output_dim' with the size of these last layers of activations.

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Wide, DeepDense, DeepText, DeepImage, WideDeep
    >>>
    >>> X_wide = torch.empty(5, 5).random_(2)
    >>> wide = Wide(wide_dim=X_wide.size(0), output_dim=1)
    >>>
    >>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> deep_column_idx = {k:v for v,k in enumerate(colnames)}
    >>> deepdense = DeepDense(hidden_layers=[8,4], deep_column_idx=deep_column_idx, embed_input=embed_input)
    >>>
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> deeptext = DeepText(vocab_size=4, hidden_dim=4, n_layers=1, padding_idx=0, embed_dim=4)
    >>>
    >>> X_img = torch.rand((5,3,224,224))
    >>> deepimage = DeepImage(head_layers=[512, 64, 8])
    >>>
    >>> model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage, output_dim=1)
    >>> input_dict = {'wide':X_wide, 'deepdense':X_deep, 'deeptext':X_text, 'deepimage':X_img}
    >>> model(X=input_dict)
    tensor([[-0.3779],
            [-0.5247],
            [-0.2773],
            [-0.2888],
            [-0.2010]], grad_fn=<AddBackward0>)
    """

    def __init__(self, wide: nn.Module, deepdense: nn.Module, output_dim: int=1, deeptext: Optional[nn.Module]=None, deepimage: Optional[nn.Module]=None, deephead: Optional[nn.Module]=None, head_layers: Optional[List[int]]=None, head_dropout: Optional[List]=None, head_batchnorm: Optional[bool]=None):
        super(WideDeep, self).__init__()
        self.wide = wide
        self.deepdense = deepdense
        self.deeptext = deeptext
        self.deepimage = deepimage
        self.deephead = deephead
        if self.deephead is None:
            if head_layers is not None:
                input_dim: int = self.deepdense.output_dim
                if self.deeptext is not None:
                    input_dim += self.deeptext.output_dim
                if self.deepimage is not None:
                    input_dim += self.deepimage.output_dim
                head_layers = [input_dim] + head_layers
                if not head_dropout:
                    head_dropout = [0.0] * (len(head_layers) - 1)
                self.deephead = nn.Sequential()
                for i in range(1, len(head_layers)):
                    self.deephead.add_module('head_layer_{}'.format(i - 1), dense_layer(head_layers[i - 1], head_layers[i], head_dropout[i - 1], head_batchnorm))
                self.deephead.add_module('head_out', nn.Linear(head_layers[-1], output_dim))
            else:
                self.deepdense = nn.Sequential(self.deepdense, nn.Linear(self.deepdense.output_dim, output_dim))
                if self.deeptext is not None:
                    self.deeptext = nn.Sequential(self.deeptext, nn.Linear(self.deeptext.output_dim, output_dim))
                if self.deepimage is not None:
                    self.deepimage = nn.Sequential(self.deepimage, nn.Linear(self.deepimage.output_dim, output_dim))

    def forward(self, X: Dict[str, Tensor]) ->Tensor:
        """
        Parameters
        ----------
        X: List
            List of Dict where the keys are the model names ('wide',
            'deepdense', 'deeptext' and 'deepimage') and the values are the
            corresponding Tensors
        """
        out = self.wide(X['wide'])
        if self.deephead:
            deepside = self.deepdense(X['deepdense'])
            if self.deeptext is not None:
                deepside = torch.cat([deepside, self.deeptext(X['deeptext'])], axis=1)
            if self.deepimage is not None:
                deepside = torch.cat([deepside, self.deepimage(X['deepimage'])], axis=1)
            deepside_out = self.deephead(deepside)
            return out.add_(deepside_out)
        else:
            out.add_(self.deepdense(X['deepdense']))
            if self.deeptext is not None:
                out.add_(self.deeptext(X['deeptext']))
            if self.deepimage is not None:
                out.add_(self.deepimage(X['deepimage']))
            return out

    def compile(self, method: str, optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]]=None, lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]]=None, initializers: Optional[Dict[str, Initializer]]=None, transforms: Optional[List[Transforms]]=None, callbacks: Optional[List[Callback]]=None, metrics: Optional[List[Metric]]=None, class_weight: Optional[Union[float, List[float], Tuple[float]]]=None, with_focal_loss: bool=False, alpha: float=0.25, gamma: float=2, verbose: int=1, seed: int=1):
        """
        Function to set a number of attributes that will be used during the
        training process.

        Parameters
        ----------
        method: Str
            One of ('regression', 'binary' or 'multiclass')
        optimizers: Optimizer, Dict. Optional, Default=AdamW
            Either an optimizers object (e.g. torch.optim.Adam()) or a
            dictionary where there keys are the model's children (i.e. 'wide',
            'deepdense', 'deeptext', 'deepimage' and/or 'deephead')  and the
            values are the corresponding optimizers. If multiple optimizers
            are used the  dictionary MUST contain an optimizer per child.
        lr_schedulers: LRScheduler, Dict. Optional. Default=None
            Either a LRScheduler object (e.g
            torch.optim.lr_scheduler.StepLR(opt, step_size=5)) or dictionary
            where there keys are the model's children (i.e. 'wide', 'deepdense',
            'deeptext', 'deepimage' and/or 'deephead') and the values are the
            corresponding learning rate schedulers.
        initializers: Dict, Optional. Default=None
            Dict where there keys are the model's children (i.e. 'wide',
            'deepdense', 'deeptext', 'deepimage' and/or 'deephead') and the
            values are the corresponding initializers.
        transforms: List, Optional. Default=None
            List with torchvision.transforms to be applied to the image
            component of the model (i.e. 'deepimage')
        callbacks: List, Optional. Default=None
            Callbacks available are: ModelCheckpoint, EarlyStopping, and
            LRHistory. The History callback is used by default.
        metrics: List, Optional. Default=None
            Metrics available are: BinaryAccuracy and CategoricalAccuracy
        class_weight: List, Tuple, Float. Optional. Default=None
            Can be one of: float indicating the weight of the minority class
            in binary classification problems (e.g. 9.) or a list or tuple
            with weights for the different classes in multiclass
            classification problems  (e.g. [1., 2., 3.]). The weights do not
            neccesarily need to be normalised. If your loss function uses
            reduction='mean', the loss will be normalized by the sum of the
            corresponding weights for each element. If you are using
            reduction='none', you would have to take care of the normalization
            yourself. See here:
            https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
        with_focal_loss: Boolean, Optional. Default=False
            Use the Focal Loss. https://arxiv.org/pdf/1708.02002.pdf
        alpha, gamma: Float. Default=0.25, 2
            Focal Loss parameters. See: https://arxiv.org/pdf/1708.02002.pdf
        verbose: Int
            Setting it to 0 will print nothing during training.
        seed: Int, Default=1
            Random seed to be used throughout all the methods

        Attributes
        ----------
        Attributes that are not direct assignations of parameters

        self.cyclic: Boolean
            Indicates if any of the lr_schedulers is cyclic (i.e. CyclicLR or
            OneCycleLR)

        Example
        --------
        Assuming you have already built the model components (wide, deepdense, etc...)

        >>> from pytorch_widedeep.models import WideDeep
        >>> from pytorch_widedeep.initializers import *
        >>> from pytorch_widedeep.callbacks import *
        >>> from pytorch_widedeep.optim import RAdam
        >>> model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
        >>> wide_opt = torch.optim.Adam(model.wide.parameters())
        >>> deep_opt = torch.optim.Adam(model.deepdense.parameters())
        >>> text_opt = RAdam(model.deeptext.parameters())
        >>> img_opt  = RAdam(model.deepimage.parameters())
        >>> wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
        >>> deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
        >>> text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
        >>> img_sch  = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)
        >>> optimizers = {'wide': wide_opt, 'deepdense':deep_opt, 'deeptext':text_opt, 'deepimage': img_opt}
        >>> schedulers = {'wide': wide_sch, 'deepdense':deep_sch, 'deeptext':text_sch, 'deepimage': img_sch}
        >>> initializers = {'wide': Uniform, 'deepdense':Normal, 'deeptext':KaimingNormal,
        >>> ... 'deepimage':KaimingUniform}
        >>> transforms = [ToTensor, Normalize(mean=mean, std=std)]
        >>> callbacks = [LRHistory, EarlyStopping, ModelCheckpoint(filepath='model_weights/wd_out.pt')]
        >>> model.compile(method='regression', initializers=initializers, optimizers=optimizers,
        >>> ... lr_schedulers=schedulers, callbacks=callbacks, transforms=transforms)
        """
        self.verbose = verbose
        self.seed = seed
        self.early_stop = False
        self.method = method
        self.with_focal_loss = with_focal_loss
        if self.with_focal_loss:
            self.alpha, self.gamma = alpha, gamma
        if isinstance(class_weight, float):
            self.class_weight = torch.tensor([1.0 - class_weight, class_weight])
        elif isinstance(class_weight, (tuple, list)):
            self.class_weight = torch.tensor(class_weight)
        else:
            self.class_weight = None
        if initializers is not None:
            self.initializer = MultipleInitializer(initializers, verbose=self.verbose)
            self.initializer.apply(self)
        if optimizers is not None:
            if isinstance(optimizers, Optimizer):
                self.optimizer: Union[Optimizer, MultipleOptimizer] = optimizers
            elif len(optimizers) > 1:
                opt_names = list(optimizers.keys())
                mod_names = [n for n, c in self.named_children()]
                for mn in mod_names:
                    assert mn in opt_names, 'No optimizer found for {}'.format(mn)
                self.optimizer = MultipleOptimizer(optimizers)
        else:
            self.optimizer = torch.optim.AdamW(self.parameters())
        if lr_schedulers is not None:
            if isinstance(lr_schedulers, LRScheduler):
                self.lr_scheduler: Union[LRScheduler, MultipleLRScheduler] = lr_schedulers
                self.cyclic = 'cycl' in self.lr_scheduler.__class__.__name__.lower()
            elif len(lr_schedulers) > 1:
                self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
                scheduler_names = [sc.__class__.__name__.lower() for _, sc in self.lr_scheduler._schedulers.items()]
                self.cyclic = any([('cycl' in sn) for sn in scheduler_names])
        else:
            self.lr_scheduler, self.cyclic = None, False
        if transforms is not None:
            self.transforms: MultipleTransforms = MultipleTransforms(transforms)()
        else:
            self.transforms = None
        self.history = History()
        self.callbacks: List = [self.history]
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    callback = callback()
                self.callbacks.append(callback)
        if metrics is not None:
            self.metric = MultipleMetrics(metrics)
            self.callbacks += [MetricCallback(self.metric)]
        else:
            self.metric = None
        self.callback_container = CallbackContainer(self.callbacks)
        self.callback_container.set_model(self)
        if use_cuda:
            self

    def fit(self, X_wide: Optional[np.ndarray]=None, X_deep: Optional[np.ndarray]=None, X_text: Optional[np.ndarray]=None, X_img: Optional[np.ndarray]=None, X_train: Optional[Dict[str, np.ndarray]]=None, X_val: Optional[Dict[str, np.ndarray]]=None, val_split: Optional[float]=None, target: Optional[np.ndarray]=None, n_epochs: int=1, validation_freq: int=1, batch_size: int=32, patience: int=10, warm_up: bool=False, warm_epochs: int=4, warm_max_lr: float=0.01, warm_deeptext_gradual: bool=False, warm_deeptext_max_lr: float=0.01, warm_deeptext_layers: Optional[List[nn.Module]]=None, warm_deepimage_gradual: bool=False, warm_deepimage_max_lr: float=0.01, warm_deepimage_layers: Optional[List[nn.Module]]=None, warm_routine: str='howard'):
        """
        fit method that must run after calling 'compile'

        Parameters
        ----------
        X_wide: np.ndarray, Optional. Default=None
            One hot encoded wide input.
        X_deep: np.ndarray, Optional. Default=None
            Input for the deepdense model
        X_text: np.ndarray, Optional. Default=None
            Input for the deeptext model
        X_img : np.ndarray, Optional. Default=None
            Input for the deepimage model
        X_train: Dict, Optional. Default=None
            Training dataset for the different model branches.  Keys are
            'X_wide', 'X_deep', 'X_text', 'X_img' and 'target' the values are
            the corresponding matrices e.g X_train = {'X_wide': X_wide,
            'X_wide': X_wide, 'X_text': X_text, 'X_img': X_img}
        X_val: Dict, Optional. Default=None
            Validation dataset for the different model branches.  Keys are
            'X_wide', 'X_deep', 'X_text', 'X_img' and 'target' the values are
            the corresponding matrices e.g X_val = {'X_wide': X_wide,
            'X_wide': X_wide, 'X_text': X_text, 'X_img': X_img}
        val_split: Float, Optional. Default=None
            train/val split
        target: np.ndarray, Optional. Default=None
            target values
        n_epochs: Int, Default=1
        validation_freq: Int, Default=1
        batch_size: Int, Default=32
        patience: Int, Default=10
            Number of epochs without improving the target metric before we
            stop the fit
        warm_up: Boolean, Default=False
            warm_up model components individually before the joined traininga
        warm_epochs: Int, Default=4
            Number of warm up epochs for those model componenst that will not
            be gradually warmed up
        warm_max_lr: Float, Default=0.01
            Maximum learning rate during the Triangular Learning rate cycle
            for those model componenst that will not be gradually warmed up
        warm_deeptext_gradual: Boolean, Default=False
            Boolean indicating if the deeptext component will be warmed
            up gradually
        warm_deeptext_max_lr: Float, Default=0.01
            Maximum learning rate during the Triangular Learning rate cycle
            for the deeptext component
        warm_deeptext_layers: Optional, List, Default=None
            List of nn.Modules that will be warmed up gradually. These have to
            be in 'warm-up-order': the layers or blocks close to the output
            neuron(s) first
        warm_deepimage_gradual: Boolean, Default=False
            Boolean indicating if the deepimage component will be warmed
            up gradually
        warm_deepimage_max_lr: Float, Default=0.01
            Maximum learning rate during the Triangular Learning rate cycle
            for the deepimage component
        warm_deepimage_layers: Optional, List, Default=None
            List of nn.Modules that will be warmed up gradually. These have to
            be in 'warm-up-order': the layers or blocks close to the output
            neuron(s) first
        warm_routine: Str, Default='felbo'
            Warm up routine. On of 'felbo' or 'howard'. See the WarmUp class
            documentation for details

        **WideDeep assumes that X_wide, X_deep and target ALWAYS exist, while
        X_text and X_img are optional
        **Either X_train or X_wide, X_deep and target must be passed to the
        fit method

        Example
        --------
        Assuming you have already built and compiled the model

        Ex 1. using train input arrays directly and no validation
        >>> model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=10, batch_size=256)

        Ex 2: using train input arrays directly and validation with val_split
        >>> model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=10, batch_size=256, val_split=0.2)

        Ex 3: using train dict and val_split
        >>> X_train = {'X_wide': X_wide, 'X_deep': X_deep, 'target': y}
        >>> model.fit(X_train, n_epochs=10, batch_size=256, val_split=0.2)

        Ex 4: validation using training and validation dicts
        >>> X_train = {'X_wide': X_wide_tr, 'X_deep': X_deep_tr, 'target': y_tr}
        >>> X_val = {'X_wide': X_wide_val, 'X_deep': X_deep_val, 'target': y_val}
        >>> model.fit(X_train=X_train, X_val=X_val n_epochs=10, batch_size=256)
        """
        if X_train is None and (X_wide is None or X_deep is None or target is None):
            raise ValueError('Training data is missing. Either a dictionary (X_train) with the training dataset or at least 3 arrays (X_wide, X_deep, target) must be passed to the fit method')
        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(X_wide, X_deep, X_text, X_img, X_train, X_val, val_split, target)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=n_cpus)
        if warm_up:
            self._warm_up(train_loader, warm_epochs, warm_max_lr, warm_deeptext_gradual, warm_deeptext_layers, warm_deeptext_max_lr, warm_deepimage_gradual, warm_deepimage_layers, warm_deepimage_max_lr, warm_routine)
        train_steps = len(train_loader)
        self.callback_container.on_train_begin({'batch_size': batch_size, 'train_steps': train_steps, 'n_epochs': n_epochs})
        if self.verbose:
            None
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)
            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, target) in zip(t, train_loader):
                    t.set_description('epoch %i' % (epoch + 1))
                    acc, train_loss = self._training_step(data, target, batch_idx)
                    if acc is not None:
                        t.set_postfix(metrics=acc, loss=train_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(train_loss))
                    if self.lr_scheduler:
                        self._lr_scheduler_step(step_location='on_batch_end')
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs['train_loss'] = train_loss
            if acc is not None:
                epoch_logs['train_acc'] = acc['acc']
            if epoch % validation_freq == validation_freq - 1:
                if eval_set is not None:
                    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, num_workers=n_cpus, shuffle=False)
                    eval_steps = len(eval_loader)
                    self.valid_running_loss = 0.0
                    with trange(eval_steps, disable=self.verbose != 1) as v:
                        for i, (data, target) in zip(v, eval_loader):
                            v.set_description('valid')
                            acc, val_loss = self._validation_step(data, target, i)
                            if acc is not None:
                                v.set_postfix(metrics=acc, loss=val_loss)
                            else:
                                v.set_postfix(loss=np.sqrt(val_loss))
                    epoch_logs['val_loss'] = val_loss
                    if acc is not None:
                        epoch_logs['val_acc'] = acc['acc']
            if self.lr_scheduler:
                self._lr_scheduler_step(step_location='on_epoch_end')
            self.callback_container.on_epoch_end(epoch, epoch_logs)
            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break
            self.callback_container.on_train_end(epoch_logs)
        self.train()

    def predict(self, X_wide: np.ndarray, X_deep: np.ndarray, X_text: Optional[np.ndarray]=None, X_img: Optional[np.ndarray]=None, X_test: Optional[Dict[str, np.ndarray]]=None) ->np.ndarray:
        """
        fit method that must run after calling 'compile'

        Parameters
        ----------
        X_wide: np.ndarray, Optional. Default=None
            One hot encoded wide input.
        X_deep: np.ndarray, Optional. Default=None
            Input for the deepdense model
        X_text: np.ndarray, Optional. Default=None
            Input for the deeptext model
        X_img : np.ndarray, Optional. Default=None
            Input for the deepimage model
        X_test: Dict, Optional. Default=None
            Testing dataset for the different model branches.  Keys are
            'X_wide', 'X_deep', 'X_text', 'X_img' and 'target' the values are
            the corresponding matrices e.g X_train = {'X_wide': X_wide,
            'X_wide': X_wide, 'X_text': X_text, 'X_img': X_img}

        **WideDeep assumes that X_wide, X_deep and target ALWAYS exist, while
        X_text and X_img are optional

        Returns
        -------
        preds: np.array with the predicted target for the test dataset.
        """
        preds_l = self._predict(X_wide, X_deep, X_text, X_img, X_test)
        if self.method == 'regression':
            return np.vstack(preds_l).squeeze(1)
        if self.method == 'binary':
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype('int')
        if self.method == 'multiclass':
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)

    def predict_proba(self, X_wide: np.ndarray, X_deep: np.ndarray, X_text: Optional[np.ndarray]=None, X_img: Optional[np.ndarray]=None, X_test: Optional[Dict[str, np.ndarray]]=None) ->np.ndarray:
        """
        Returns
        -------
        preds: np.ndarray
            Predicted probabilities of target for the test dataset for  binary
            and multiclass methods
        """
        preds_l = self._predict(X_wide, X_deep, X_text, X_img, X_test)
        if self.method == 'binary':
            preds = np.vstack(preds_l).squeeze(1)
            probs = np.zeros([preds.shape[0], 2])
            probs[:, (0)] = 1 - preds
            probs[:, (1)] = preds
            return probs
        if self.method == 'multiclass':
            return np.vstack(preds_l)

    def get_embeddings(self, col_name: str, cat_encoding_dict: Dict[str, Dict[str, int]]) ->Dict[str, np.ndarray]:
        """
        Get the learned embeddings for the categorical features passed through deepdense.

        Parameters
        ----------
        col_name: str,
            Column name of the feature we want to get the embeddings for
        cat_encoding_dict: Dict
            Categorical encodings. The function is designed to take the
            'encoding_dict' attribute from the DeepPreprocessor class. Any
            Dict with the same structure can be used

        Returns
        -------
        cat_embed_dict: Dict
            Categorical levels of the col_name feature and the corresponding
            embeddings

        Example:
        -------
        Assuming we have already train the model:

        >>> model.get_embeddings(col_name='education', cat_encoding_dict=deep_preprocessor.encoding_dict)
        {'11th': array([-0.42739448, -0.22282735,  0.36969638,  0.4445322 ,  0.2562272 ,
        0.11572784, -0.01648579,  0.09027119,  0.0457597 , -0.28337458], dtype=float32),
         'HS-grad': array([-0.10600474, -0.48775527,  0.3444158 ,  0.13818645, -0.16547225,
        0.27409762, -0.05006042, -0.0668492 , -0.11047247,  0.3280354 ], dtype=float32),
        ...
        }

        where:

        >>> deep_preprocessor.encoding_dict['education']
        {'11th': 0, 'HS-grad': 1, 'Assoc-acdm': 2, 'Some-college': 3, '10th': 4, 'Prof-school': 5,
        '7th-8th': 6, 'Bachelors': 7, 'Masters': 8, 'Doctorate': 9, '5th-6th': 10, 'Assoc-voc': 11,
        '9th': 12, '12th': 13, '1st-4th': 14, 'Preschool': 15}
        """
        for n, p in self.named_parameters():
            if 'embed_layers' in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v: k for k, v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx, value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict

    def _activation_fn(self, inp: Tensor) ->Tensor:
        if self.method == 'binary':
            return torch.sigmoid(inp)
        else:
            return inp

    def _loss_fn(self, y_pred: Tensor, y_true: Tensor) ->Tensor:
        if self.with_focal_loss:
            return FocalLoss(self.alpha, self.gamma)(y_pred, y_true)
        if self.method == 'regression':
            return F.mse_loss(y_pred, y_true.view(-1, 1))
        if self.method == 'binary':
            return F.binary_cross_entropy(y_pred, y_true.view(-1, 1), weight=self.class_weight)
        if self.method == 'multiclass':
            return F.cross_entropy(y_pred, y_true, weight=self.class_weight)

    def _train_val_split(self, X_wide: Optional[np.ndarray]=None, X_deep: Optional[np.ndarray]=None, X_text: Optional[np.ndarray]=None, X_img: Optional[np.ndarray]=None, X_train: Optional[Dict[str, np.ndarray]]=None, X_val: Optional[Dict[str, np.ndarray]]=None, val_split: Optional[float]=None, target: Optional[np.ndarray]=None):
        """
        If a validation set (X_val) is passed to the fit method, or val_split
        is specified, the train/val split will happen internally. A number of
        options are allowed in terms of data inputs. For parameter
        information, please, see the .fit() method documentation

        Returns
        -------
        train_set: WideDeepDataset
            WideDeepDataset object that will be loaded through
            torch.utils.data.DataLoader
        eval_set : WideDeepDataset
            WideDeepDataset object that will be loaded through
            torch.utils.data.DataLoader
        """
        if X_val is None and val_split is None:
            if X_train is not None:
                X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']
                if 'X_text' in X_train.keys():
                    X_text = X_train['X_text']
                if 'X_img' in X_train.keys():
                    X_img = X_train['X_img']
            X_train = {'X_wide': X_wide, 'X_deep': X_deep, 'target': target}
            try:
                X_train.update({'X_text': X_text})
            except:
                pass
            try:
                X_train.update({'X_img': X_img})
            except:
                pass
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)
            eval_set = None
        else:
            if X_val is not None:
                if X_train is None:
                    X_train = {'X_wide': X_wide, 'X_deep': X_deep, 'target': target}
                    if X_text is not None:
                        X_train.update({'X_text': X_text})
                    if X_img is not None:
                        X_train.update({'X_img': X_img})
            else:
                if X_train is not None:
                    X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']
                    if 'X_text' in X_train.keys():
                        X_text = X_train['X_text']
                    if 'X_img' in X_train.keys():
                        X_img = X_train['X_img']
                X_tr_wide, X_val_wide, X_tr_deep, X_val_deep, y_tr, y_val = train_test_split(X_wide, X_deep, target, test_size=val_split, random_state=self.seed, stratify=target if self.method != 'regression' else None)
                X_train = {'X_wide': X_tr_wide, 'X_deep': X_tr_deep, 'target': y_tr}
                X_val = {'X_wide': X_val_wide, 'X_deep': X_val_deep, 'target': y_val}
                try:
                    X_tr_text, X_val_text = train_test_split(X_text, test_size=val_split, random_state=self.seed, stratify=target if self.method != 'regression' else None)
                    X_train.update({'X_text': X_tr_text}), X_val.update({'X_text': X_val_text})
                except:
                    pass
                try:
                    X_tr_img, X_val_img = train_test_split(X_img, test_size=val_split, random_state=self.seed, stratify=target if self.method != 'regression' else None)
                    X_train.update({'X_img': X_tr_img}), X_val.update({'X_img': X_val_img})
                except:
                    pass
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)
            eval_set = WideDeepDataset(**X_val, transforms=self.transforms)
        return train_set, eval_set

    def _warm_up(self, loader: DataLoader, n_epochs: int, max_lr: float, deeptext_gradual: bool, deeptext_layers: List[nn.Module], deeptext_max_lr: float, deepimage_gradual: bool, deepimage_layers: List[nn.Module], deepimage_max_lr: float, routine: str='felbo'):
        """
        Simple wrappup to individually warm up model components
        """
        if self.deephead is not None:
            raise ValueError("Currently warming up is only supported without a fully connected 'DeepHead'")
        warmer = WarmUp(self._activation_fn, self._loss_fn, self.metric, self.method, self.verbose)
        warmer.warm_all(self.wide, 'wide', loader, n_epochs, max_lr)
        warmer.warm_all(self.deepdense, 'deepdense', loader, n_epochs, max_lr)
        if self.deeptext:
            if deeptext_gradual:
                warmer.warm_gradual(self.deeptext, 'deeptext', loader, deeptext_max_lr, deeptext_layers, routine)
            else:
                warmer.warm_all(self.deeptext, 'deeptext', loader, n_epochs, max_lr)
        if self.deepimage:
            if deepimage_gradual:
                warmer.warm_gradual(self.deepimage, 'deepimage', loader, deepimage_max_lr, deepimage_layers, routine)
            else:
                warmer.warm_all(self.deepimage, 'deepimage', loader, n_epochs, max_lr)

    def _lr_scheduler_step(self, step_location: str):
        """
        Function to execute the learning rate schedulers steps.
        If the lr_scheduler is Cyclic (i.e. CyclicLR or OneCycleLR), the step
        must happen after training each bach durig training. On the other
        hand, if the  scheduler is not Cyclic, is expected to be called after
        validation.

        Parameters
        ----------
        step_location: Str
            Indicates where to run the lr_scheduler step
        """
        if self.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler' and self.cyclic:
            if step_location == 'on_batch_end':
                for model_name, scheduler in self.lr_scheduler._schedulers.items():
                    if 'cycl' in scheduler.__class__.__name__.lower():
                        scheduler.step()
            elif step_location == 'on_epoch_end':
                for scheduler_name, scheduler in self.lr_scheduler._schedulers.items():
                    if 'cycl' not in scheduler.__class__.__name__.lower():
                        scheduler.step()
        elif self.cyclic:
            if step_location == 'on_batch_end':
                self.lr_scheduler.step()
            else:
                pass
        elif self.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler':
            if step_location == 'on_epoch_end':
                self.lr_scheduler.step()
            else:
                pass
        elif step_location == 'on_epoch_end':
            self.lr_scheduler.step()
        else:
            pass

    def _training_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):
        self.train()
        X = {k: v for k, v in data.items()} if use_cuda else data
        y = target.float() if self.method != 'multiclass' else target
        y = y if use_cuda else y
        self.optimizer.zero_grad()
        y_pred = self._activation_fn(self.forward(X))
        loss = self._loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)
        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def _validation_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):
        self.eval()
        with torch.no_grad():
            X = {k: v for k, v in data.items()} if use_cuda else data
            y = target.float() if self.method != 'multiclass' else target
            y = y if use_cuda else y
            y_pred = self._activation_fn(self.forward(X))
            loss = self._loss_fn(y_pred, y)
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)
        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def _predict(self, X_wide: np.ndarray, X_deep: np.ndarray, X_text: Optional[np.ndarray]=None, X_img: Optional[np.ndarray]=None, X_test: Optional[Dict[str, np.ndarray]]=None) ->List:
        """
        Hidden method to avoid code repetition in predict and predict_proba.
        For parameter information, please, see the .predict() method
        documentation
        """
        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            if X_text is not None:
                load_dict.update({'X_text': X_text})
            if X_img is not None:
                load_dict.update({'X_img': X_img})
            test_set = WideDeepDataset(**load_dict)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, num_workers=n_cpus, shuffle=False)
        test_steps = len(test_loader.dataset) // test_loader.batch_size + 1
        self.eval()
        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description('predict')
                    X = {k: v for k, v in data.items()} if use_cuda else data
                    preds = self._activation_fn(self.forward(X))
                    if self.method == 'multiclass':
                        preds = F.softmax(preds, dim=1)
                    preds = preds.cpu().data.numpy()
                    preds_l.append(preds)
        self.train()
        return preds_l


class TestDeepText(nn.Module):

    def __init__(self):
        super(TestDeepText, self).__init__()
        self.word_embed = nn.Embedding(5, 16, padding_idx=0)
        self.rnn = nn.LSTM(16, 8, batch_first=True)
        self.linear = nn.Linear(8, 1)

    def forward(self, X):
        embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        return self.linear(h).view(-1, 1)


class TestDeepImage(nn.Module):

    def __init__(self):
        super(TestDeepImage, self).__init__()
        self.conv_block = nn.Sequential(conv_layer(3, 64, 3), conv_layer(64, 128, 1, maxpool=False, adaptiveavgpool=True))
        self.linear = nn.Linear(128, 1)

    def forward(self, X):
        x = self.conv_block(X)
        x = x.view(x.size(0), -1)
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeepImage,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestDeepImage,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestDeepText,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Wide,
     lambda: ([], {'wide_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jrzaurin_pytorch_widedeep(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

