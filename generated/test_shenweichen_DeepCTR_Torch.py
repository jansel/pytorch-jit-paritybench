import sys
_module = sys.modules[__name__]
del sys
deepctr_torch = _module
callbacks = _module
inputs = _module
layers = _module
activation = _module
core = _module
interaction = _module
sequence = _module
utils = _module
models = _module
afm = _module
afn = _module
autoint = _module
basemodel = _module
ccpm = _module
dcn = _module
dcnmix = _module
deepfm = _module
dien = _module
difm = _module
din = _module
fibinet = _module
ifm = _module
mlr = _module
multitask = _module
esmm = _module
mmoe = _module
ple = _module
sharedbottom = _module
nfm = _module
onn = _module
pnn = _module
wdl = _module
xdeepfm = _module
utils = _module
conf = _module
run_classification_criteo = _module
run_dien = _module
run_din = _module
run_multitask_learning = _module
run_multivalue_movielens = _module
run_regression_movielens = _module
setup = _module
tests = _module
activation_test = _module
AFM_test = _module
AFN_test = _module
AutoInt_test = _module
CCPM_test = _module
DCNMix_test = _module
DCN_test = _module
DIEN_test = _module
DIFM_test = _module
DIN_test = _module
DeepFM_test = _module
FiBiNET_test = _module
IFM_test = _module
MLR_test = _module
NFM_test = _module
ONN_test = _module
PNN_test = _module
WDL_test = _module
ESMM_test = _module
MMOE_test = _module
PLE_test = _module
SharedBottom_test = _module
xDeepFM_test = _module
utils = _module
utils_mtl = _module

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


from tensorflow.python.keras.callbacks import EarlyStopping


from tensorflow.python.keras.callbacks import ModelCheckpoint


from tensorflow.python.keras.callbacks import History


from collections import OrderedDict


from collections import namedtuple


from collections import defaultdict


from itertools import chain


import torch.nn as nn


import numpy as np


import math


import torch.nn.functional as F


import itertools


from torch.nn.utils.rnn import PackedSequence


import time


import torch.utils.data as Data


from sklearn.metrics import *


from torch.utils.data import DataLoader


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import logging


import pandas as pd


from sklearn.metrics import log_loss


from sklearn.metrics import roc_auc_score


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import MinMaxScaler


from sklearn.metrics import mean_squared_error


import torch as torch


class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]

    Output shape:
        - Same shape as input.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-08, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError('hidden_units is empty!!')
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.activation_layers = nn.ModuleList([activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
        user interests varies adaptively given different candidate items.

    Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0, dice_dim=3, l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()
        self.dnn = DNN(inputs_dim=4 * embedding_dim, hidden_units=hidden_units, activation=activation, l2_reg=l2_reg, dropout_rate=dropout_rate, dice_dim=dice_dim, use_bn=use_bn)
        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        user_behavior_len = user_behavior.size(1)
        queries = query.expand(-1, user_behavior_len, -1)
        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior], dim=-1)
        attention_output = self.dnn(attention_input)
        attention_score = self.dense(attention_output)
        return attention_score


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError('task must be binary,multiclass or regression')
        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == 'binary':
            output = torch.sigmoid(output)
        return output


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term


class BiInteractionPooling(nn.Module):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """

    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term


class SENETLayer(nn.Module):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, reduction_ratio=3, seed=1024, device='cpu'):
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(nn.Linear(self.filed_size, self.reduction_size, bias=False), nn.ReLU(), nn.Linear(self.reduction_size, self.filed_size, bias=False), nn.ReLU())
        self

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError('Unexpected inputs dimensions %d, expect to be 3 dimensions' % len(inputs.shape))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))
        return V


class BilinearInteraction(nn.Module):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2, embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **embedding_size** : Positive integer, embedding size of sparse features.
        - **bilinear_type** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, embedding_size, bilinear_type='interaction', seed=1024, device='cpu'):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == 'all':
            self.bilinear = nn.Linear(embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == 'each':
            for _ in range(filed_size):
                self.bilinear.append(nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == 'interaction':
            for _, _ in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError
        self

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError('Unexpected inputs dimensions %d, expect to be 3 dimensions' % len(inputs.shape))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == 'all':
            p = [torch.mul(self.bilinear(v_i), v_j) for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == 'each':
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j]) for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == 'interaction':
            p = [torch.mul(bilinear(v[0]), v[1]) for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)


class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function name used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-05, seed=1024, device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError('layer_size must be a list(tuple) of length greater than 1')
        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed
        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))
            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError('layer_size must be even number except for the last layer when split_half=True')
                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)
        self

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError('Unexpected inputs dimensions %d, expect to be 3 dimensions' % len(inputs.shape))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []
        for i, size in enumerate(self.layer_size):
            x = torch.einsum('bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            x = x.reshape(batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            x = self.conv1ds[i](x)
            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)
            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        return result


class AFMLayer(nn.Module):
    """Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.
        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.
        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.
        - **seed** : A Python integer to use as random seed.
      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, in_features, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, device='cpu'):
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features
        self.attention_W = nn.Parameter(torch.Tensor(embedding_size, self.attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))
        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))
        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor)
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor)
        self.dropout = nn.Dropout(dropout_rate)
        self

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []
        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)
        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q
        bi_interaction = inner_product
        attention_temp = F.relu(torch.tensordot(bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(self.normalized_att_score * bi_interaction, dim=1)
        attention_output = self.dropout(attention_output)
        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out


class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed
        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        self

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError('Unexpected inputs dimensions %d, expect to be 3 dimensions' % len(inputs.shape))
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))
        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)
        result = torch.matmul(self.normalized_att_scores, values)
        result = torch.cat(torch.split(result, 1), dim=-1)
        result = torch.squeeze(result, dim=0)
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)
        return result


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])
        self

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)
                dot_ = xl_w + self.bias[i]
                x_l = x_0 * dot_ + x_l
            else:
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class CrossNetMix(nn.Module):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **device**: str, e.g. ``"cpu"`` or ``"cuda:0"``
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, low_rank=32, num_experts=4, layer_num=2, device='cpu'):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts
        self.U_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        self.V_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        self.C_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, low_rank, low_rank))
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        init_para_list = [self.U_list, self.V_list, self.C_list]
        for para in init_para_list:
            for i in range(self.layer_num):
                nn.init.xavier_normal_(para[i])
        for i in range(len(self.bias)):
            nn.init.zeros_(self.bias[i])
        self

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)
                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_
                output_of_experts.append(dot_.squeeze(2))
            output_of_experts = torch.stack(output_of_experts, 2)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l
        x_l = x_l.squeeze()
        return x_l


class InnerProductLayer(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.
      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape:
        ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.
      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product
      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//
            Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.]
            (https://arxiv.org/pdf/1611.00144.pdf)"""

    def __init__(self, reduce_sum=True, device='cpu'):
        super(InnerProductLayer, self).__init__()
        self.reduce_sum = reduce_sum
        self

    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)
        q = torch.cat([embed_list[idx] for idx in col], dim=1)
        inner_product = p * q
        if self.reduce_sum:
            inner_product = torch.sum(inner_product, dim=2, keepdim=True)
        return inner_product


class OutterProductLayer(nn.Module):
    """OutterProduct Layer used in PNN.This implemention is
    adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.
      Input shape
            - A list of N 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
            - 2D tensor with shape:``(batch_size,N*(N-1)/2 )``.
      Arguments
            - **filed_size** : Positive integer, number of feature groups.
            - **kernel_type**: str. The kernel weight matrix type to use,can be mat,vec or num
            - **seed**: A Python integer to use as random seed.
      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, field_size, embedding_size, kernel_type='mat', seed=1024, device='cpu'):
        super(OutterProductLayer, self).__init__()
        self.kernel_type = kernel_type
        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size
        if self.kernel_type == 'mat':
            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_pairs, embed_size))
        elif self.kernel_type == 'vec':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))
        elif self.kernel_type == 'num':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))
        nn.init.xavier_uniform_(self.kernel)
        self

    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)
        q = torch.cat([embed_list[idx] for idx in col], dim=1)
        if self.kernel_type == 'mat':
            p.unsqueeze_(dim=1)
            kp = torch.sum(torch.mul(torch.transpose(torch.sum(torch.mul(p, self.kernel), dim=-1), 2, 1), q), dim=-1)
        else:
            k = torch.unsqueeze(self.kernel, 0)
            kp = torch.sum(p * q * k, dim=-1)
        return kp


class KMaxPooling(nn.Module):
    """K Max pooling that selects the k biggest value along the specific axis.

      Input shape
        -  nD tensor with shape: ``(batch_size, ..., input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., output_dim)``.

      Arguments
        - **k**: positive integer, number of top elements to look for along the ``axis`` dimension.

        - **axis**: positive integer, the dimension to look for elements.

     """

    def __init__(self, k, axis, device='cpu'):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.axis = axis
        self

    def forward(self, inputs):
        if self.axis < 0 or self.axis >= len(inputs.shape):
            raise ValueError('axis must be 0~%d,now is %d' % (len(inputs.shape) - 1, self.axis))
        if self.k < 1 or self.k > inputs.shape[self.axis]:
            raise ValueError('k must be in 1 ~ %d,now k is %d' % (inputs.shape[self.axis], self.k))
        out = torch.topk(inputs, k=self.k, dim=self.axis, sorted=True)[0]
        return out


class ConvLayer(nn.Module):
    """Conv Layer used in CCPM.

      Input shape
            - A list of N 3D tensor with shape: ``(batch_size,1,filed_size,embedding_size)``.
      Output shape
            - A list of N 3D tensor with shape: ``(batch_size,last_filters,pooling_size,embedding_size)``.
      Arguments
            - **filed_size** : Positive integer, number of feature groups.
            - **conv_kernel_width**: list. list of positive integer or empty list,the width of filter in each conv layer.
            - **conv_filters**: list. list of positive integer or empty list,the number of filters in each conv layer.
      Reference:
            - Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.(http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)
    """

    def __init__(self, field_size, conv_kernel_width, conv_filters, device='cpu'):
        super(ConvLayer, self).__init__()
        self.device = device
        module_list = []
        n = int(field_size)
        l = len(conv_filters)
        filed_shape = n
        for i in range(1, l + 1):
            if i == 1:
                in_channels = 1
            else:
                in_channels = conv_filters[i - 2]
            out_channels = conv_filters[i - 1]
            width = conv_kernel_width[i - 1]
            k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3
            module_list.append(Conv2dSame(in_channels=in_channels, out_channels=out_channels, kernel_size=(width, 1), stride=1))
            module_list.append(torch.nn.Tanh())
            module_list.append(KMaxPooling(k=min(k, filed_shape), axis=2, device=self.device))
            filed_shape = min(k, filed_shape)
        self.conv_layer = nn.Sequential(*module_list)
        self
        self.filed_shape = filed_shape

    def forward(self, inputs):
        return self.conv_layer(inputs)


class LogTransformLayer(nn.Module):
    """Logarithmic Transformation Layer in Adaptive factorization network, which models arbitrary-order cross features.

      Input shape
        - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, ltl_hidden_size*embedding_size)``.
      Arguments
        - **field_size** : positive integer, number of feature groups
        - **embedding_size** : positive integer, embedding size of sparse features
        - **ltl_hidden_size** : integer, the number of logarithmic neurons in AFN
      References
        - Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
    """

    def __init__(self, field_size, embedding_size, ltl_hidden_size):
        super(LogTransformLayer, self).__init__()
        self.ltl_weights = nn.Parameter(torch.Tensor(field_size, ltl_hidden_size))
        self.ltl_biases = nn.Parameter(torch.Tensor(1, 1, ltl_hidden_size))
        self.bn = nn.ModuleList([nn.BatchNorm1d(embedding_size) for i in range(2)])
        nn.init.normal_(self.ltl_weights, mean=0.0, std=0.1)
        nn.init.zeros_(self.ltl_biases)

    def forward(self, inputs):
        afn_input = torch.clamp(torch.abs(inputs), min=1e-07, max=float('Inf'))
        afn_input_trans = torch.transpose(afn_input, 1, 2)
        ltl_result = torch.log(afn_input_trans)
        ltl_result = self.bn[0](ltl_result)
        ltl_result = torch.matmul(ltl_result, self.ltl_weights) + self.ltl_biases
        ltl_result = torch.exp(ltl_result)
        ltl_result = self.bn[1](ltl_result)
        ltl_result = torch.flatten(ltl_result, start_dim=1)
        return ltl_result


class SequencePoolingLayer(nn.Module):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

    """

    def __init__(self, mode='mean', supports_masking=False, device='cpu'):
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-08])
        self

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1], dtype=torch.float32)
            mask = torch.transpose(mask, 1, 2)
        embedding_size = uiseq_embed_list.shape[-1]
        mask = torch.repeat_interleave(mask, embedding_size, dim=2)
        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1000000000.0
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)
        if self.mode == 'mean':
            self.eps = self.eps
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)
        hist = torch.unsqueeze(hist, dim=1)
        return hist


class AttentionSequencePoolingLayer(nn.Module):
    """The Attentional sequence pooling operation used in DIN & DIEN.

        Arguments
          - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

          - **att_activation**: Activation function to use in attention net.

          - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

          - **supports_masking**:If True,the input need to support masking.

        References
          - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
      """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False, return_score=False, supports_masking=False, embedding_dim=4, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim, activation=att_activation, dropout_rate=0, use_bn=False)

    def forward(self, query, keys, keys_length, mask=None):
        """
        Input shape
          - A list of three tensor: [query,keys,keys_length]

          - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

          - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

          - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

        Output shape
          - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
        """
        batch_size, max_length, _ = keys.size()
        if self.supports_masking:
            if mask is None:
                raise ValueError('When supports_masking=True,input must support masking')
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device, dtype=keys_length.dtype).repeat(batch_size, 1)
            keys_masks = keys_masks < keys_length.view(-1, 1)
            keys_masks = keys_masks.unsqueeze(1)
        attention_score = self.local_att(query, keys)
        outputs = torch.transpose(attention_score, 1, 2)
        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(outputs)
        outputs = torch.where(keys_masks, outputs, paddings)
        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)
        if not self.return_score:
            outputs = torch.matmul(outputs, keys)
        return outputs


class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_state = torch.tanh(i_n + reset_gate * h_n)
        att_score = att_score.view(-1, 1)
        hy = (1.0 - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)
        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1.0 - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError('DynamicGRU only supports packed input and att_scores')
        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores
        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        outputs = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(inputs[begin:begin + batch], hx[0:batch], att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


DEFAULT_GROUP_NAME = 'default_group'


class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype='int32', embedding_name=None, group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            None
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype, embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner='mean', length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
    embedding_dict = nn.ModuleDict({feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse) for feat in sparse_feature_columns + varlen_sparse_feature_columns})
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
    return embedding_dict


def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.name]
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)([seq_emb, seq_mask])
        else:
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False, device=device)([seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def varlen_embedding_lookup(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](X[:, lookup_idx[0]:lookup_idx[1]].long())
    return varlen_embedding_vec_dict


class Linear(nn.Module):

    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        self.varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False, device=device)
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in self.sparse_feature_columns]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in self.dense_feature_columns]
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index, self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.varlen_sparse_feature_columns, self.device)
        sparse_embedding_list += varlen_embedding_list
        linear_logit = torch.zeros([X.shape[0], 1])
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit
        return linear_logit


def build_input_features(feature_columns):
    features = OrderedDict()
    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = start, start + 1
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = start, start + feat.dimension
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = start, start + feat.maxlen
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = start, start + 1
                start += 1
        else:
            raise TypeError('Invalid feature column type,got', type(feat))
    return features


def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """
    if arrays is None:
        return [None]
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [(None if x is None else x[start]) for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [(None if x is None else x[start:stop]) for x in arrays]
    elif hasattr(start, '__len__'):
        if hasattr(start, 'shape'):
            start = start.tolist()
        return arrays[start]
    elif hasattr(start, '__getitem__'):
        return arrays[start:stop]
    else:
        return [None]


class BaseModel(nn.Module):

    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-05, l2_reg_embedding=1e-05, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError('`gpus[0]` should be the same gpu with `device`')
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns
        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        self.linear_model = Linear(linear_feature_columns, self.feature_index, device=device)
        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)
        self.out = PredictionLayer(task)
        self
        self._is_graph_network = True
        self._ckpt_saved_epoch = False
        self.history = History()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.0, validation_data=None, shuffle=True, callbacks=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing a `validation_data` argument, it must contain either 2 items (x_val, y_val), or 3 items (x_val, y_val, val_sample_weights), or alternatively it could be a dataset or a dataset or a dataset iterator. However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        elif validation_split and 0.0 < validation_split < 1.0:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1.0 - validation_split))
            else:
                split_at = int(len(x[0]) * (1.0 - validation_split))
            x, val_x = slice_arrays(x, 0, split_at), slice_arrays(x, split_at)
            y, val_y = slice_arrays(y, 0, split_at), slice_arrays(y, split_at)
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        if self.gpus:
            None
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)
        else:
            None
        train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        callbacks = (callbacks or []) + [self.history]
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False
        None
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.float()
                        y = y_train.float()
                        y_pred = model(x).squeeze()
                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks, 'the length of `loss_func` should be equal with `self.num_tasks`'
                            loss = sum([loss_func[i](y_pred[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()
                        total_loss = loss + reg_loss + self.aux_loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype('float64')))
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_logs['loss'] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs['val_' + name] = result
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                None
                eval_str = '{0}s - loss: {1: .4f}'.format(epoch_time, epoch_logs['loss'])
                for name in self.metrics:
                    eval_str += ' - ' + name + ': {0: .4f}'.format(epoch_logs[name])
                if do_validation:
                    for name in self.metrics:
                        eval_str += ' - ' + 'val_' + name + ': {0: .4f}'.format(epoch_logs['val_' + name])
                None
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break
        callbacks.on_train_end()
        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)
        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].float()
                y_pred = model(x).cpu().data.numpy()
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans).astype('float64')

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError('DenseFeat is not supported in dnn_feature_columns')
        sparse_embedding_list = [embedding_dict[feat.embedding_name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in sparse_feature_columns]
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index, varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, varlen_sparse_feature_columns, self.device)
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in dense_feature_columns]
        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)
        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer, loss=None, metrics=None):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ['loss']
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == 'sgd':
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == 'adam':
                optim = torch.optim.Adam(self.parameters())
            elif optimizer == 'adagrad':
                optim = torch.optim.Adagrad(self.parameters())
            elif optimizer == 'rmsprop':
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [self._get_loss_func_single(loss_single) for loss_single in loss]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        if loss == 'binary_crossentropy':
            loss_func = F.binary_cross_entropy
        elif loss == 'mse':
            loss_func = F.mse_loss
        elif loss == 'mae':
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-07, normalize=True, sample_weight=None, labels=None):
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == 'binary_crossentropy' or metric == 'logloss':
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == 'auc':
                    metrics_[metric] = roc_auc_score
                if metric == 'mse':
                    metrics_[metric] = mean_squared_error
                if metric == 'accuracy' or metric == 'acc':
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        return None

    @property
    def embedding_size(self):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError('embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!')
        return list(embedding_size_set)[0]


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


class CCPM(BaseModel):
    """Instantiates the Convolutional Click Prediction Model architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, conv_kernel_width=(6, 5), conv_filters=(4, 4), dnn_hidden_units=(256,), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', dnn_use_bn=False, dnn_activation='relu', gpus=None):
        super(CCPM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        if len(conv_kernel_width) != len(conv_filters):
            raise ValueError('conv_kernel_width must have same element with conv_filters')
        filed_size = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        self.conv_layer = ConvLayer(field_size=filed_size, conv_kernel_width=conv_kernel_width, conv_filters=conv_filters, device=device)
        self.dnn_input_dim = self.conv_layer.filed_shape * self.embedding_size * conv_filters[-1]
        self.dnn = DNN(self.dnn_input_dim, dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        linear_logit = self.linear_model(X)
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict, support_dense=False)
        if len(sparse_embedding_list) == 0:
            raise ValueError('must have the embedding feature,now the embedding feature is None!')
        conv_input = concat_fun(sparse_embedding_list, axis=1)
        conv_input_concact = torch.unsqueeze(conv_input, 1)
        pooling_result = self.conv_layer(conv_input_concact)
        flatten_result = pooling_result.view(pooling_result.size(0), -1)
        dnn_output = self.dnn(flatten_result)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = linear_logit + dnn_logit
        y_pred = self.out(logit)
        return y_pred


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


class DCN(BaseModel):
    """Instantiates the Deep&Cross Network architecture. Including DCN-V (parameterization='vector')
    and DCN-M (parameterization='matrix').

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector', dnn_hidden_units=(128, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_cross=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(DCN, self).__init__(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, init_std=init_std, device=device)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns), layer_num=cross_num, parameterization=cross_parameterization, device=device)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)
        self

    def forward(self, X):
        logit = self.linear_model(X)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:
            pass
        y_pred = self.out(logit)
        return y_pred


class DCNMix(BaseModel):
    """Instantiates the DCN-Mix model.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param low_rank: Positive integer, dimensionality of low-rank sapce.
    :param num_experts: Positive integer, number of experts.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, dnn_hidden_units=(128, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_cross=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, low_rank=32, num_experts=4, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(DCNMix, self).__init__(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, init_std=init_std, device=device)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)
        self.crossnet = CrossNetMix(in_features=self.compute_input_dim(dnn_feature_columns), low_rank=low_rank, num_experts=num_experts, layer_num=cross_num, device=device)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        regularization_modules = [self.crossnet.U_list, self.crossnet.V_list, self.crossnet.C_list]
        for module in regularization_modules:
            self.add_regularization_weight(module, l2=l2_reg_cross)
        self

    def forward(self, X):
        logit = self.linear_model(X)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:
            pass
        y_pred = self.out(logit)
        return y_pred


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, use_fm=True, dnn_hidden_units=(256, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
        y_pred = self.out(logit)
        return y_pred


class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self, input_size, gru_type='GRU', use_neg=False, init_std=0.001, att_hidden_size=(64, 16), att_activation='sigmoid', att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError('gru_type: {gru_type} is not supported')
        self.gru_type = gru_type
        self.use_neg = use_neg
        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size, att_hidden_units=att_hidden_size, att_activation=att_activation, weight_normalization=att_weight_normalization, return_score=False)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size, att_hidden_units=att_hidden_size, att_activation=att_activation, weight_normalization=att_weight_normalization, return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size, att_hidden_units=att_hidden_size, att_activation=att_activation, weight_normalization=att_weight_normalization, return_score=True)
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size, gru_type=gru_type)
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        batch_size, max_seq_length, _ = states.size()
        mask = torch.arange(max_seq_length, device=keys_length.device).repeat(batch_size, 1) == keys_length.view(-1, 1) - 1
        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: (masked_interests), 3D tensor, [b, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        batch_size, dim = query.size()
        max_length = keys.size()[1]
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)
        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0, total_length=max_length)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))
            outputs = outputs.squeeze(1)
        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))
            interests = keys * att_scores.transpose(1, 2)
            packed_interests = pack_padded_sequence(interests, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)
            packed_interests = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=max_length)
            outputs = InterestEvolving._get_last_state(outputs, keys_length)
        zero_outputs[mask] = outputs
        return zero_outputs


class InterestExtractor(nn.Module):

    def __init__(self, input_size, use_neg=False, init_std=0.001, device='cpu'):
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], 'sigmoid', init_std=init_std, device=device)
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self

    def forward(self, keys, keys_length, neg_keys=None):
        """
        Parameters
        ----------
        keys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        neg_keys: 3D tensor, [B, T, H]

        Returns
        -------
        masked_interests: 2D tensor, [b, H]
        aux_loss: [1]
        """
        batch_size, max_length, dim = keys.size()
        zero_outputs = torch.zeros(batch_size, dim, device=keys.device)
        aux_loss = torch.zeros((1,), device=keys.device)
        mask = keys_length > 0
        masked_keys_length = keys_length[mask]
        if masked_keys_length.shape[0] == 0:
            return zero_outputs,
        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
        packed_keys = pack_padded_sequence(masked_keys, lengths=masked_keys_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0, total_length=max_length)
        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            aux_loss = self._cal_auxiliary_loss(interests[:, :-1, :], masked_keys[:, 1:, :], masked_neg_keys[:, 1:, :], masked_keys_length - 1)
        return interests, aux_loss

    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)
        _, max_seq_length, embedding_size = states.size()
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        batch_size = states.size()[0]
        mask = (torch.arange(max_seq_length, device=states.device).repeat(batch_size, 1) < keys_length.view(-1, 1)).float()
        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2
        click_p = self.auxiliary_net(click_input.view(batch_size * max_seq_length, embedding_size)).view(batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(click_p.size(), dtype=torch.float, device=click_p.device)
        noclick_p = self.auxiliary_net(noclick_input.view(batch_size * max_seq_length, embedding_size)).view(batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(noclick_p.size(), dtype=torch.float, device=noclick_p.device)
        loss = F.binary_cross_entropy(torch.cat([click_p, noclick_p], dim=0), torch.cat([click_target, noclick_target], dim=0))
        return loss


def embedding_lookup(X, sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=(), to_list=False):
    """
        Args:
            X: input Tensor [batch_size x hidden_dim]
            sparse_embedding_dict: nn.ModuleDict, {embedding_name: nn.Embedding}
            sparse_input_dict: OrderedDict, {feature_name:(start, start+dimension)}
            sparse_feature_columns: list, sparse features
            return_feat_list: list, names of feature to be returned, defualt () -> return all features
            mask_feat_list, list, names of feature to be masked in hash transform
        Return:
            group_embedding_dict: defaultdict(list)
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            lookup_idx = np.array(sparse_input_dict[feature_name])
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def get_dense_input(X, features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = np.array(features[fc.name])
        input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list


def maxlen_lookup(X, sparse_input_dict, maxlen_column):
    if maxlen_column is None or len(maxlen_column) == 0:
        raise ValueError('please add max length column for VarLenSparseFeat of DIN/DIEN input')
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0]:lookup_idx[1]].long()


class DIEN(BaseModel):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param gru_type: str,can be GRU AIGRU AUGRU AGRU
    :param use_negsampling: bool, whether or not use negtive sampling
    :param alpha: float ,weight of auxiliary_loss
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, gru_type='GRU', use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_units=(64, 16), att_activation='relu', att_weight_normalization=True, l2_reg_dnn=0, l2_reg_embedding=1e-06, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(DIEN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.item_features = history_feature_list
        self.use_negsampling = use_negsampling
        self.alpha = alpha
        self._split_columns()
        input_size = self._compute_interest_dim()
        self.interest_extractor = InterestExtractor(input_size=input_size, use_neg=use_negsampling, init_std=init_std)
        self.interest_evolution = InterestEvolving(input_size=input_size, gru_type=gru_type, use_neg=use_negsampling, init_std=init_std, att_hidden_size=att_hidden_units, att_activation=att_activation, att_weight_normalization=att_weight_normalization)
        dnn_input_size = self._compute_dnn_dim() + input_size
        self.dnn = DNN(dnn_input_size, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, init_std=init_std, seed=seed)
        self.linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self

    def forward(self, X):
        query_emb, keys_emb, neg_keys_emb, keys_length = self._get_emb(X)
        masked_interest, aux_loss = self.interest_extractor(keys_emb, keys_length, neg_keys_emb)
        self.add_auxiliary_loss(aux_loss, self.alpha)
        hist = self.interest_evolution(query_emb, masked_interest, keys_length)
        deep_input_emb = self._get_deep_input_emb(X)
        deep_input_emb = concat_fun([hist, deep_input_emb])
        dense_value_list = get_dense_input(X, self.feature_index, self.dense_feature_columns)
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        output = self.linear(self.dnn(dnn_input))
        y_pred = self.out(output)
        return y_pred

    def _get_emb(self, X):
        history_feature_columns = []
        neg_history_feature_columns = []
        sparse_varlen_feature_columns = []
        history_fc_names = list(map(lambda x: 'hist_' + x, self.item_features))
        neg_history_fc_names = list(map(lambda x: 'neg_' + x, history_fc_names))
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in history_fc_names:
                history_feature_columns.append(fc)
            elif feature_name in neg_history_fc_names:
                neg_history_feature_columns.append(fc)
            else:
                sparse_varlen_feature_columns.append(fc)
        features = self.feature_index
        query_emb_list = embedding_lookup(X, self.embedding_dict, features, self.sparse_feature_columns, return_feat_list=self.item_features, to_list=True)
        query_emb = torch.squeeze(concat_fun(query_emb_list), 1)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, features, history_feature_columns, return_feat_list=history_fc_names, to_list=True)
        keys_emb = concat_fun(keys_emb_list)
        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, features, keys_length_feature_name), 1)
        if self.use_negsampling:
            neg_keys_emb_list = embedding_lookup(X, self.embedding_dict, features, neg_history_feature_columns, return_feat_list=neg_history_fc_names, to_list=True)
            neg_keys_emb = concat_fun(neg_keys_emb_list)
        else:
            neg_keys_emb = None
        return query_emb, keys_emb, neg_keys_emb, keys_length

    def _split_columns(self):
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), self.dnn_feature_columns)) if len(self.dnn_feature_columns) else []
        self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), self.dnn_feature_columns)) if len(self.dnn_feature_columns) else []
        self.varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), self.dnn_feature_columns)) if len(self.dnn_feature_columns) else []

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.item_features:
                interest_dim += feat.embedding_dim
        return interest_dim

    def _compute_dnn_dim(self):
        dnn_input_dim = 0
        for fc in self.sparse_feature_columns:
            dnn_input_dim += fc.embedding_dim
        for fc in self.dense_feature_columns:
            dnn_input_dim += fc.dimension
        return dnn_input_dim

    def _get_deep_input_emb(self, X):
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns, mask_feat_list=self.item_features, to_list=True)
        dnn_input_emb = concat_fun(dnn_input_emb_list)
        return dnn_input_emb.squeeze(1)


class DIFM(BaseModel):
    """Instantiates the DIFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_head_num: int. The head number in multi-head  self-attention network.
    :param att_res: bool. Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on ``device`` . ``gpus[0]`` should be the same gpu with ``device`` .
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, att_head_num=4, att_res=True, dnn_hidden_units=(256, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(DIFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        if not len(dnn_hidden_units) > 0:
            raise ValueError('dnn_hidden_units is null!')
        self.fm = FM()
        self.vector_wise_net = InteractingLayer(self.embedding_size, att_head_num, att_res, scaling=True, device=device)
        self.bit_wise_net = DNN(self.compute_input_dim(dnn_feature_columns, include_dense=False), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat), dnn_feature_columns)))
        self.transform_matrix_P_vec = nn.Linear(self.sparse_feat_num * self.embedding_size, self.sparse_feat_num, bias=False)
        self.transform_matrix_P_bit = nn.Linear(dnn_hidden_units[-1], self.sparse_feat_num, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.vector_wise_net.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bit_wise_net.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.transform_matrix_P_vec.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.transform_matrix_P_bit.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        if not len(sparse_embedding_list) > 0:
            raise ValueError('there are no sparse features')
        att_input = concat_fun(sparse_embedding_list, axis=1)
        att_out = self.vector_wise_net(att_input)
        att_out = att_out.reshape(att_out.shape[0], -1)
        m_vec = self.transform_matrix_P_vec(att_out)
        dnn_input = combined_dnn_input(sparse_embedding_list, [])
        dnn_output = self.bit_wise_net(dnn_input)
        m_bit = self.transform_matrix_P_bit(dnn_output)
        m_x = m_vec + m_bit
        logit = self.linear_model(X, sparse_feat_refine_weight=m_x)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * m_x.unsqueeze(-1)
        logit += self.fm(refined_fm_input)
        y_pred = self.out(logit)
        return y_pred


class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False, dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16), att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0, l2_reg_embedding=1e-06, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.history_feature_list = history_feature_list
        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: 'hist_' + x, history_feature_list))
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)
        att_emb_dim = self._compute_interest_dim()
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, embedding_dim=att_emb_dim, att_activation=att_activation, return_score=False, supports_masking=False, weight_normalization=att_weight_normalization)
        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns), hidden_units=dnn_hidden_units, activation=dnn_activation, dropout_rate=dnn_dropout, l2_reg=l2_reg_dnn, use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self

    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns, return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns, return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns, to_list=True)
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_varlen_feature_columns)
        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.sparse_varlen_feature_columns, self.device)
        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
        query_emb = torch.cat(query_emb_list, dim=-1)
        keys_emb = torch.cat(keys_emb_list, dim=-1)
        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)
        hist = self.attention(query_emb, keys_emb, keys_length)
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        y_pred = self.out(dnn_logit)
        return y_pred

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim


class FiBiNET(BaseModel):
    """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bilinear_type: str,bilinear function type used in Bilinear Interaction Layer,can be ``'all'`` , ``'each'`` or ``'interaction'``
    :param reduction_ratio: integer in [1,inf), reduction ratio used in SENET Layer
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, bilinear_type='interaction', reduction_ratio=3, dnn_hidden_units=(128, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', task='binary', device='cpu', gpus=None):
        super(FiBiNET, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.field_size = len(self.embedding_dict)
        self.SE = SENETLayer(self.field_size, reduction_ratio, seed, device)
        self.Bilinear = BilinearInteraction(self.field_size, self.embedding_size, bilinear_type, seed, device)
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        field_size = len(sparse_feature_columns)
        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        embedding_size = sparse_feature_columns[0].embedding_dim
        sparse_input_dim = field_size * (field_size - 1) * embedding_size
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        sparse_embedding_input = torch.cat(sparse_embedding_list, dim=1)
        senet_output = self.SE(sparse_embedding_input)
        senet_bilinear_out = self.Bilinear(senet_output)
        bilinear_out = self.Bilinear(sparse_embedding_input)
        linear_logit = self.linear_model(X)
        temp = torch.split(torch.cat((senet_bilinear_out, bilinear_out), dim=1), 1, dim=1)
        dnn_input = combined_dnn_input(temp, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        if len(self.linear_feature_columns) > 0 and len(self.dnn_feature_columns) > 0:
            final_logit = linear_logit + dnn_logit
        elif len(self.linear_feature_columns) == 0:
            final_logit = dnn_logit
        elif len(self.dnn_feature_columns) == 0:
            final_logit = linear_logit
        else:
            raise NotImplementedError
        y_pred = self.out(final_logit)
        return y_pred


class IFM(BaseModel):
    """Instantiates the IFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on ``device`` .  ``gpus[0]``  should be the same gpu with ``device`` .
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(IFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        if not len(dnn_hidden_units) > 0:
            raise ValueError('dnn_hidden_units is null!')
        self.fm = FM()
        self.factor_estimating_net = DNN(self.compute_input_dim(dnn_feature_columns, include_dense=False), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat), dnn_feature_columns)))
        self.transform_weight_matrix_P = nn.Linear(dnn_hidden_units[-1], self.sparse_feat_num, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.factor_estimating_net.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.transform_weight_matrix_P.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        if not len(sparse_embedding_list) > 0:
            raise ValueError('there are no sparse features')
        dnn_input = combined_dnn_input(sparse_embedding_list, [])
        dnn_output = self.factor_estimating_net(dnn_input)
        dnn_output = self.transform_weight_matrix_P(dnn_output)
        input_aware_factor = self.sparse_feat_num * dnn_output.softmax(1)
        logit = self.linear_model(X, sparse_feat_refine_weight=input_aware_factor)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * input_aware_factor.unsqueeze(-1)
        logit += self.fm(refined_fm_input)
        y_pred = self.out(logit)
        return y_pred


class MLR(BaseModel):
    """Instantiates the Mixed Logistic Regression/Piece-wise Linear Model.

    :param region_feature_columns: An iterable containing all the features used by region part of the model.
    :param base_feature_columns: An iterable containing all the features used by base part of the model.
    :param region_num: integer > 1,indicate the piece number
    :param l2_reg_linear: float. L2 regularizer strength applied to weight
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param bias_feature_columns: An iterable containing all the features used by bias part of the model.
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, region_feature_columns, base_feature_columns=None, bias_feature_columns=None, region_num=4, l2_reg_linear=1e-05, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(MLR, self).__init__(region_feature_columns, region_feature_columns, task=task, device=device, gpus=gpus)
        if region_num <= 1:
            raise ValueError('region_num must > 1')
        self.l2_reg_linear = l2_reg_linear
        self.init_std = init_std
        self.seed = seed
        self.device = device
        self.region_num = region_num
        self.region_feature_columns = region_feature_columns
        self.base_feature_columns = base_feature_columns
        self.bias_feature_columns = bias_feature_columns
        if base_feature_columns is None or len(base_feature_columns) == 0:
            self.base_feature_columns = region_feature_columns
        if bias_feature_columns is None:
            self.bias_feature_columns = []
        self.feature_index = build_input_features(self.region_feature_columns + self.base_feature_columns + self.bias_feature_columns)
        self.region_linear_model = nn.ModuleList([Linear(self.region_feature_columns, self.feature_index, self.init_std, self.device) for i in range(self.region_num)])
        self.base_linear_model = nn.ModuleList([Linear(self.base_feature_columns, self.feature_index, self.init_std, self.device) for i in range(self.region_num)])
        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            self.bias_model = nn.Sequential(Linear(self.bias_feature_columns, self.feature_index, self.init_std, self.device), PredictionLayer(task='binary', use_bias=False))
        self.prediction_layer = PredictionLayer(task=task, use_bias=False)
        self

    def get_region_score(self, inputs, region_number):
        region_logit = torch.cat([self.region_linear_model[i](inputs) for i in range(region_number)], dim=-1)
        region_score = nn.Softmax(dim=-1)(region_logit)
        return region_score

    def get_learner_score(self, inputs, region_number):
        learner_score = self.prediction_layer(torch.cat([self.region_linear_model[i](inputs) for i in range(region_number)], dim=-1))
        return learner_score

    def forward(self, X):
        region_score = self.get_region_score(X, self.region_num)
        learner_score = self.get_learner_score(X, self.region_num)
        final_logit = torch.sum(region_score * learner_score, dim=-1, keepdim=True)
        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            bias_score = self.bias_model(X)
            final_logit = final_logit * bias_score
        return final_logit


class ESMM(BaseModel):
    """Instantiates the Entire Space Multi-Task Model architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, tower_dnn_hidden_units=(256, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(ESMM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task='binary', device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks != 2:
            raise ValueError('the length of task_names must be equal to 2')
        if len(dnn_feature_columns) == 0:
            raise ValueError('dnn_feature_columns is null!')
        if len(task_types) != self.num_tasks:
            raise ValueError('num_tasks must be equal to the length of task_types')
        for task_type in task_types:
            if task_type != 'binary':
                raise ValueError('task must be binary in ESMM, {} is illegal'.format(task_type))
        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.ctr_dnn = DNN(input_dim, tower_dnn_hidden_units, activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.cvr_dnn = DNN(input_dim, tower_dnn_hidden_units, activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.ctr_dnn_final_layer = nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
        self.cvr_dnn_final_layer = nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.ctr_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cvr_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.ctr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.cvr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        ctr_output = self.ctr_dnn(dnn_input)
        cvr_output = self.cvr_dnn(dnn_input)
        ctr_logit = self.ctr_dnn_final_layer(ctr_output)
        cvr_logit = self.cvr_dnn_final_layer(cvr_output)
        ctr_pred = self.out(ctr_logit)
        cvr_pred = self.out(cvr_logit)
        ctcvr_pred = ctr_pred * cvr_pred
        task_outs = torch.cat([ctr_pred, ctcvr_pred], -1)
        return task_outs


class MMOE(BaseModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128), gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError('num_tasks must be greater than 1')
        if num_experts <= 1:
            raise ValueError('num_experts must be greater than 1')
        if len(dnn_feature_columns) == 0:
            raise ValueError('dnn_feature_columns is null!')
        if len(task_types) != self.num_tasks:
            raise ValueError('num_tasks must be equal to the length of task_types')
        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError('task must be binary or regression, {} is illegal'.format(task_type))
        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, expert_dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for _ in range(self.num_experts)])
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()), l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList([nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim, self.num_experts, bias=False) for _ in range(self.num_tasks)])
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList([DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()), l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1, bias=False) for _ in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])
        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)
            mmoe_outs.append(gate_mul_expert.squeeze())
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs


class PLE(BaseModel):
    """Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param shared_expert_num: integer, number of task-shared experts.
    :param specific_expert_num: integer, number of task-specific experts.
    :param num_levels: integer, number of CGC levels.
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, shared_expert_num=1, specific_expert_num=1, num_levels=2, expert_dnn_hidden_units=(256, 128), gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(PLE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError('num_tasks must be greater than 1!')
        if len(dnn_feature_columns) == 0:
            raise ValueError('dnn_feature_columns is null!')
        if len(task_types) != self.num_tasks:
            raise ValueError('num_tasks must be equal to the length of task_types')
        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError('task must be binary or regression, {} is illegal'.format(task_type))
        self.specific_expert_num = specific_expert_num
        self.shared_expert_num = shared_expert_num
        self.num_levels = num_levels
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        def multi_module_list(num_level, num_tasks, expert_num, inputs_dim_level0, inputs_dim_not_level0, hidden_units):
            return nn.ModuleList([nn.ModuleList([nn.ModuleList([DNN(inputs_dim_level0 if level_num == 0 else inputs_dim_not_level0, hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for _ in range(expert_num)]) for _ in range(num_tasks)]) for level_num in range(num_level)])
        self.specific_experts = multi_module_list(self.num_levels, self.num_tasks, self.specific_expert_num, self.input_dim, expert_dnn_hidden_units[-1], expert_dnn_hidden_units)
        self.shared_experts = multi_module_list(self.num_levels, 1, self.specific_expert_num, self.input_dim, expert_dnn_hidden_units[-1], expert_dnn_hidden_units)
        specific_gate_output_dim = self.specific_expert_num + self.shared_expert_num
        if len(gate_dnn_hidden_units) > 0:
            self.specific_gate_dnn = multi_module_list(self.num_levels, self.num_tasks, 1, self.input_dim, expert_dnn_hidden_units[-1], gate_dnn_hidden_units)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.specific_gate_dnn.named_parameters()), l2=l2_reg_dnn)
        self.specific_gate_dnn_final_layer = nn.ModuleList([nn.ModuleList([nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim if level_num == 0 else expert_dnn_hidden_units[-1], specific_gate_output_dim, bias=False) for _ in range(self.num_tasks)]) for level_num in range(self.num_levels)])
        shared_gate_output_dim = self.num_tasks * self.specific_expert_num + self.shared_expert_num
        if len(gate_dnn_hidden_units) > 0:
            self.shared_gate_dnn = nn.ModuleList([DNN(self.input_dim if level_num == 0 else expert_dnn_hidden_units[-1], gate_dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for level_num in range(self.num_levels)])
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.shared_gate_dnn.named_parameters()), l2=l2_reg_dnn)
        self.shared_gate_dnn_final_layer = nn.ModuleList([nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim if level_num == 0 else expert_dnn_hidden_units[-1], shared_gate_output_dim, bias=False) for level_num in range(self.num_levels)])
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList([DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()), l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1, bias=False) for _ in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])
        regularization_modules = [self.specific_experts, self.shared_experts, self.specific_gate_dnn_final_layer, self.shared_gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self

    def cgc_net(self, inputs, level_num):
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_experts[level_num][i][j](inputs[i])
                specific_expert_outputs.append(specific_expert_output)
        shared_expert_outputs = []
        for k in range(self.shared_expert_num):
            shared_expert_output = self.shared_experts[level_num][0][k](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)
        cgc_outs = []
        for i in range(self.num_tasks):
            cur_experts_outputs = specific_expert_outputs[i * self.specific_expert_num:(i + 1) * self.specific_expert_num] + shared_expert_outputs
            cur_experts_outputs = torch.stack(cur_experts_outputs, 1)
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.specific_gate_dnn[level_num][i][0](inputs[i])
                gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](gate_dnn_out)
            else:
                gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](inputs[i])
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)
            cgc_outs.append(gate_mul_expert.squeeze())
        cur_experts_outputs = specific_expert_outputs + shared_expert_outputs
        cur_experts_outputs = torch.stack(cur_experts_outputs, 1)
        if len(self.gate_dnn_hidden_units) > 0:
            gate_dnn_out = self.shared_gate_dnn[level_num](inputs[-1])
            gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](gate_dnn_out)
        else:
            gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](inputs[-1])
        gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs)
        cgc_outs.append(gate_mul_expert.squeeze())
        return cgc_outs

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        ple_inputs = [dnn_input] * (self.num_tasks + 1)
        ple_outputs = []
        for i in range(self.num_levels):
            ple_outputs = self.cgc_net(inputs=ple_inputs, level_num=i)
            ple_inputs = ple_outputs
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](ple_outputs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](ple_outputs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs


class SharedBottom(BaseModel):
    """Instantiates the SharedBottom multi-task learning Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bottom_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of shared bottom DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN
    :param init_std: float, to use as the initialize std of embedding vector
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, bottom_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(SharedBottom, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError('num_tasks must be greater than 1')
        if len(dnn_feature_columns) == 0:
            raise ValueError('dnn_feature_columns is null!')
        if len(task_types) != self.num_tasks:
            raise ValueError('num_tasks must be equal to the length of task_types')
        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError('task must be binary or regression, {} is illegal'.format(task_type))
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.bottom_dnn_hidden_units = bottom_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.bottom_dnn = DNN(self.input_dim, bottom_dnn_hidden_units, activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList([DNN(bottom_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()), l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else bottom_dnn_hidden_units[-1], 1, bias=False) for _ in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bottom_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn_final_layer.named_parameters()), l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        shared_bottom_output = self.bottom_dnn(dnn_input)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](shared_bottom_output)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](shared_bottom_output)
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs


class NFM(BaseModel):
    """Instantiates the NFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param biout_dropout: When not ``None``, the probability we will drop out the output of BiInteractionPooling Layer.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in deep net
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-05, l2_reg_linear=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, bi_dropout=0, dnn_dropout=0, dnn_activation='relu', task='binary', device='cpu', gpus=None):
        super(NFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, include_sparse=False) + self.embedding_size, dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        linear_logit = self.linear_model(X)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)
        dnn_input = combined_dnn_input([bi_out], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = linear_logit + dnn_logit
        y_pred = self.out(logit)
        return y_pred


class Interac(nn.Module):

    def __init__(self, first_size, second_size, emb_size, init_std, sparse=False):
        super(Interac, self).__init__()
        self.emb1 = nn.Embedding(first_size, emb_size, sparse=sparse)
        self.emb2 = nn.Embedding(second_size, emb_size, sparse=sparse)
        self.__init_weight(init_std)

    def __init_weight(self, init_std):
        nn.init.normal_(self.emb1.weight, mean=0, std=init_std)

    def forward(self, first, second):
        """
        input:
            x batch_size * 2
        output:
            y batch_size * emb_size
        """
        first_emb = self.emb1(first)
        second_emb = self.emb2(second)
        y = first_emb * second_emb
        return y


class ONN(BaseModel):
    """Instantiates the Operation-aware Neural Networks  architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param use_bn: bool,whether use bn after ffm out or not
    :param reduce_sum: bool,whether apply reduce_sum on cross vector
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-05, l2_reg_linear=1e-05, l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001, seed=1024, dnn_use_bn=False, dnn_activation='relu', task='binary', device='cpu', gpus=None):
        super(ONN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        embedding_size = self.embedding_size
        self.second_order_embedding_dict = self.__create_second_order_embedding_matrix(dnn_feature_columns, embedding_size=embedding_size, sparse=False)
        self.add_regularization_weight(self.second_order_embedding_dict.parameters(), l2=l2_reg_embedding)
        dim = self.__compute_nffm_dnn_dim(feature_columns=dnn_feature_columns, embedding_size=embedding_size)
        self.dnn = DNN(inputs_dim=dim, hidden_units=dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self

    def __compute_nffm_dnn_dim(self, feature_columns, embedding_size):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        return int(len(sparse_feature_columns) * (len(sparse_feature_columns) - 1) / 2 * embedding_size + sum(map(lambda x: x.dimension, dense_feature_columns)))

    def __input_from_second_order_column(self, X, feature_columns, second_order_embedding_dict):
        """
        :param X: same as input_from_feature_columns
        :param feature_columns: same as input_from_feature_columns
        :param second_order_embedding_dict: ex: {'A1+A2': Interac model} created by function create_second_order_embedding_matrix
        :return:
        """
        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        second_order_embedding_list = []
        for first_index in range(len(sparse_feature_columns) - 1):
            for second_index in range(first_index + 1, len(sparse_feature_columns)):
                first_name = sparse_feature_columns[first_index].embedding_name
                second_name = sparse_feature_columns[second_index].embedding_name
                second_order_embedding_list.append(second_order_embedding_dict[first_name + '+' + second_name](X[:, self.feature_index[first_name][0]:self.feature_index[first_name][1]].long(), X[:, self.feature_index[second_name][0]:self.feature_index[second_name][1]].long()))
        return second_order_embedding_list

    def __create_second_order_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        temp_dict = {}
        for first_index in range(len(sparse_feature_columns) - 1):
            for second_index in range(first_index + 1, len(sparse_feature_columns)):
                first_name = sparse_feature_columns[first_index].embedding_name
                second_name = sparse_feature_columns[second_index].embedding_name
                temp_dict[first_name + '+' + second_name] = Interac(sparse_feature_columns[first_index].vocabulary_size, sparse_feature_columns[second_index].vocabulary_size, emb_size=embedding_size, init_std=init_std, sparse=sparse)
        return nn.ModuleDict(temp_dict)

    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        linear_logit = self.linear_model(X)
        spare_second_order_embedding_list = self.__input_from_second_order_column(X, self.dnn_feature_columns, self.second_order_embedding_dict)
        dnn_input = combined_dnn_input(spare_second_order_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        if len(self.dnn_feature_columns) > 0:
            final_logit = dnn_logit + linear_logit
        else:
            final_logit = linear_logit
        y_pred = self.out(final_logit)
        return y_pred


class PNN(BaseModel):
    """Instantiates the Product-based Neural Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param use_inner: bool,whether use inner-product or not.
    :param use_outter: bool,whether use outter-product or not.
    :param kernel_type: str,kernel_type used in outter-product,can be ``'mat'`` , ``'vec'`` or ``'num'``
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False, kernel_type='mat', task='binary', device='cpu', gpus=None):
        super(PNN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError('kernel_type must be mat,vec or num')
        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        self.task = task
        product_out_dim = 0
        num_inputs = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)
        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(num_inputs, self.embedding_size, kernel_type=kernel_type, device=device)
        self.dnn = DNN(product_out_dim + self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False, init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        linear_signal = torch.flatten(concat_fun(sparse_embedding_list), start_dim=1)
        if self.use_inner:
            inner_product = torch.flatten(self.innerproduct(sparse_embedding_list), start_dim=1)
        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)
        if self.use_outter and self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal
        dnn_input = combined_dnn_input([product_layer], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit
        y_pred = self.out(logit)
        return y_pred


class WDL(BaseModel):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128), l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(WDL, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
        y_pred = self.out(logit)
        return y_pred


class xDeepFM(BaseModel):
    """Instantiates the xDeepFM architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256), cin_layer_size=(256, 128), cin_split_half=True, cin_activation='relu', l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict)
            if cin_split_half == True:
                self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(field_num, cin_layer_size, cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()), l2=l2_reg_cin)
        self

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        linear_logit = self.linear_model(X)
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:
            final_logit = linear_logit + cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError
        y_pred = self.out(final_logit)
        return y_pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AFMLayer,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([4, 4])], {}),
     True),
    (AUGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([64, 4]), torch.rand([64, 4]), torch.rand([16, 4])], {}),
     True),
    (AttentionSequencePoolingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([16384, 4, 4]), torch.rand([16384, 4, 4]), torch.rand([256, 4, 4, 4])], {}),
     False),
    (BiInteractionPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BilinearInteraction,
     lambda: ([], {'filed_size': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CIN,
     lambda: ([], {'field_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayer,
     lambda: ([], {'field_size': 4, 'conv_kernel_width': [4, 4], 'conv_filters': [4, 4]}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     True),
    (CrossNet,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossNetMix,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DNN,
     lambda: ([], {'inputs_dim': 4, 'hidden_units': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dice,
     lambda: ([], {'emb_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (FM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InnerProductLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InteractingLayer,
     lambda: ([], {'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'feature_columns': [4, 4], 'feature_index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LocalActivationUnit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (LogTransformLayer,
     lambda: ([], {'field_size': 4, 'embedding_size': 4, 'ltl_hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PredictionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SENETLayer,
     lambda: ([], {'filed_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_shenweichen_DeepCTR_Torch(_paritybench_base):
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

