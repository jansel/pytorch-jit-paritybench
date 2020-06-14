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


import re


import warnings


from torch import nn


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.utils.data import DataLoader


from torch.nn import Module


from torch import Tensor


from torch.optim.optimizer import Optimizer


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


from torch.utils.data import Dataset


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
        return F.binary_cross_entropy_with_logits(input, binary_target,
            weight, reduction='mean')


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

    def __init__(self, deep_column_idx: Dict[str, int], hidden_layers: List
        [int], batchnorm: bool=False, dropout: Optional[List[float]]=None,
        embed_input: Optional[List[Tuple[str, int, int]]]=None, embed_p:
        float=0.0, continuous_cols: Optional[List[str]]=None):
        super(DeepDense, self).__init__()
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({('emb_layer_' + col): nn.
                Embedding(val, dim) for col, val, dim in self.embed_input})
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
            self.dense.add_module('dense_layer_{}'.format(i - 1),
                dense_layer(hidden_layers[i - 1], hidden_layers[i], dropout
                [i - 1], batchnorm))
        self.output_dim = hidden_layers[-1]

    def forward(self, X: Tensor) ->Tensor:
        if self.embed_input is not None:
            x = [self.embed_layers['emb_layer_' + col](X[:, (self.
                deep_column_idx[col])].long()) for col, _, _ in self.
                embed_input]
            x = torch.cat(x, 1)
            x = self.embed_dropout(x)
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.
                continuous_cols]
            x_cont = X[:, (cont_idx)].float()
            x = torch.cat([x, x_cont], 1
                ) if self.embed_input is not None else x_cont
        return self.dense(x)


def conv_layer(ni: int, nf: int, ks: int=3, stride: int=1, maxpool: bool=
    True, adaptiveavgpool: bool=False):
    layer = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=ks, bias=True,
        stride=stride, padding=ks // 2), nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))
    if maxpool:
        layer.add_module('maxpool', nn.MaxPool2d(2, 2))
    if adaptiveavgpool:
        layer.add_module('adaptiveavgpool', nn.AdaptiveAvgPool2d(
            output_size=(1, 1)))
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

    def __init__(self, pretrained: bool=True, resnet: int=18, freeze: Union
        [str, int]=6, head_layers: Optional[List[int]]=None, head_dropout:
        Optional[List[float]]=None, head_batchnorm: Optional[bool]=False):
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
            self.backbone = nn.Sequential(conv_layer(3, 64, 3), conv_layer(
                64, 128, 1, maxpool=False), conv_layer(128, 256, 1, maxpool
                =False), conv_layer(256, 512, 1, maxpool=False,
                adaptiveavgpool=True))
        self.output_dim = 512
        if self.head_layers is not None:
            assert self.head_layers[0
                ] == self.output_dim, 'The output dimension from the backbone ({}) is not consistent with the expected input dimension ({}) of the fc-head'.format(
                self.output_dim, self.head_layers[0])
            if not head_dropout:
                head_dropout = [0.0] * len(head_layers)
            self.imagehead = nn.Sequential()
            for i in range(1, len(head_layers)):
                self.imagehead.add_module('dense_layer_{}'.format(i - 1),
                    dense_layer(head_layers[i - 1], head_layers[i],
                    head_dropout[i - 1], head_batchnorm))
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

    def __init__(self, vocab_size: int, hidden_dim: int=64, n_layers: int=3,
        rnn_dropout: float=0.0, bidirectional: bool=False, padding_idx: int
        =1, embed_dim: Optional[int]=None, embedding_matrix: Optional[np.
        ndarray]=None, head_layers: Optional[List[int]]=None, head_dropout:
        Optional[List[float]]=None, head_batchnorm: Optional[bool]=False):
        super(DeepText, self).__init__()
        if (embed_dim is not None and embedding_matrix is not None and not 
            embed_dim == embedding_matrix.shape[1]):
            warnings.warn(
                'the input embedding dimension {} and the dimension of the pretrained embeddings {} do not match. The pretrained embeddings dimension ({}) will be used'
                .format(embed_dim, embedding_matrix.shape[1],
                embedding_matrix.shape[1]), UserWarning)
        self.bidirectional = bidirectional
        self.head_layers = head_layers
        if isinstance(embedding_matrix, np.ndarray):
            assert embedding_matrix.dtype == 'float32', "'embedding_matrix' must be of dtype 'float32', got dtype '{}'".format(
                str(embedding_matrix.dtype))
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.
                shape[1], padding_idx=padding_idx)
            self.word_embed.weight = nn.Parameter(torch.tensor(
                embedding_matrix), requires_grad=True)
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim,
                padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
            bidirectional=bidirectional, dropout=rnn_dropout, batch_first=True)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if self.head_layers is not None:
            assert self.head_layers[0
                ] == self.output_dim, 'The hidden dimension from the stack or RNNs ({}) is not consistent with the expected input dimension ({}) of the fc-head'.format(
                self.output_dim, self.head_layers[0])
            if not head_dropout:
                head_dropout = [0.0] * len(head_layers)
            self.texthead = nn.Sequential()
            for i in range(1, len(head_layers)):
                self.texthead.add_module('dense_layer_{}'.format(i - 1),
                    dense_layer(head_layers[i - 1], head_layers[i],
                    head_dropout[i - 1], head_batchnorm))
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
        self.conv_block = nn.Sequential(conv_layer(3, 64, 3), conv_layer(64,
            128, 1, maxpool=False, adaptiveavgpool=True))
        self.linear = nn.Linear(128, 1)

    def forward(self, X):
        x = self.conv_block(X)
        x = x.view(x.size(0), -1)
        return self.linear(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jrzaurin_pytorch_widedeep(_paritybench_base):
    pass
    def test_000(self):
        self._check(TestDeepImage(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(TestDeepText(*[], **{}), [torch.rand([4, 4])], {})

    def test_002(self):
        self._check(Wide(*[], **{'wide_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

