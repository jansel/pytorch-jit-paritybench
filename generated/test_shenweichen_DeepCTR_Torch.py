import sys
_module = sys.modules[__name__]
del sys
deepctr_torch = _module
inputs = _module
layers = _module
activation = _module
core = _module
interaction = _module
sequence = _module
utils = _module
models = _module
afm = _module
autoint = _module
basemodel = _module
ccpm = _module
dcn = _module
deepfm = _module
dien = _module
din = _module
fibinet = _module
mlr = _module
nfm = _module
onn = _module
pnn = _module
wdl = _module
xdeepfm = _module
conf = _module
run_classification_criteo = _module
run_dien = _module
run_din = _module
run_multivalue_movielens = _module
run_regression_movielens = _module
setup = _module
tests = _module
activation_test = _module
AFM_test = _module
AutoInt_test = _module
CCPM_test = _module
DCN_test = _module
DIEN_test = _module
DIN_test = _module
DeepFM_test = _module
FiBiNET_test = _module
MLR_test = _module
NFM_test = _module
ONN_test = _module
PNN_test = _module
WDL_test = _module
xDeepFM_test = _module

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


from collections import OrderedDict


from collections import namedtuple


from collections import defaultdict


from itertools import chain


import torch


import torch.nn as nn


import numpy as np


import math


import torch.nn.functional as F


import itertools


from torch.nn.utils.rnn import PackedSequence


import time


import torch.utils.data as Data


from torch.utils.data import DataLoader


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


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
            self.alpha = torch.zeros((emb_size,)).to(device)
        else:
            self.alpha = torch.zeros((emb_size, 1)).to(device)

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

    def forward(self, X):
        return X


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

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation=
        'sigmoid', dropout_rate=0, dice_dim=3, l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()
        self.dnn = DNN(inputs_dim=4 * embedding_dim, hidden_units=
            hidden_units, activation=activation, l2_reg=l2_reg,
            dropout_rate=dropout_rate, dice_dim=dice_dim, use_bn=use_bn)
        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        user_behavior_len = user_behavior.size(1)
        queries = query.expand(-1, user_behavior_len, -1)
        attention_input = torch.cat([queries, user_behavior, queries -
            user_behavior, queries * user_behavior], dim=-1)
        attention_output = self.dnn(attention_input)
        attention_score = self.dense(attention_output)
        return attention_score


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

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=
        0, dropout_rate=0, use_bn=False, init_std=0.0001, dice_dim=3, seed=
        1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError('hidden_units is empty!!')
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList([nn.Linear(hidden_units[i],
            hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_units[i + 1]) for
                i in range(len(hidden_units) - 1)])
        self.activation_layers = nn.ModuleList([activation_layer(activation,
            hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) -
            1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] +
            1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] +
            1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
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
        square_of_sum = torch.pow(torch.sum(concated_embeds_value, dim=1,
            keepdim=True), 2)
        sum_of_square = torch.sum(concated_embeds_value *
            concated_embeds_value, dim=1, keepdim=True)
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
        self.excitation = nn.Sequential(nn.Linear(self.filed_size, self.
            reduction_size, bias=False), nn.ReLU(), nn.Linear(self.
            reduction_size, self.filed_size, bias=False), nn.ReLU())
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                'Unexpected inputs dimensions %d, expect to be 3 dimensions' %
                len(inputs.shape))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))
        return V


class BilinearInteraction(nn.Module):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **str** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, embedding_size, bilinear_type=
        'interaction', seed=1024, device='cpu'):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == 'all':
            self.bilinear = nn.Linear(embedding_size, embedding_size, bias=
                False)
        elif self.bilinear_type == 'each':
            for i in range(filed_size):
                self.bilinear.append(nn.Linear(embedding_size,
                    embedding_size, bias=False))
        elif self.bilinear_type == 'interaction':
            for i, j in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(nn.Linear(embedding_size,
                    embedding_size, bias=False))
        else:
            raise NotImplementedError
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                'Unexpected inputs dimensions %d, expect to be 3 dimensions' %
                len(inputs.shape))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == 'all':
            p = [torch.mul(self.bilinear(v_i), v_j) for v_i, v_j in
                itertools.combinations(inputs, 2)]
        elif self.bilinear_type == 'each':
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j]) for i, j in
                itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == 'interaction':
            p = [torch.mul(bilinear(v[0]), v[1]) for v, bilinear in zip(
                itertools.combinations(inputs, 2), self.bilinear)]
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

    def __init__(self, field_size, layer_size=(128, 128), activation='relu',
        split_half=True, l2_reg=1e-05, seed=1024, device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                'layer_size must be a list(tuple) of length greater than 1')
        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed
        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(nn.Conv1d(self.field_nums[-1] * self.
                field_nums[0], size, 1))
            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        'layer_size must be even number except for the last layer when split_half=True'
                        )
                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                'Unexpected inputs dimensions %d, expect to be 3 dimensions' %
                len(inputs.shape))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []
        for i, size in enumerate(self.layer_size):
            x = torch.einsum('bhd,bmd->bhmd', hidden_nn_layers[-1],
                hidden_nn_layers[0])
            x = x.reshape(batch_size, hidden_nn_layers[-1].shape[1] *
                hidden_nn_layers[0].shape[1], dim)
            x = self.conv1ds[i](x)
            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)
            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(curr_out, 2 *
                        [size // 2], 1)
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

    def __init__(self, in_features, attention_factor=4, l2_reg_w=0,
        dropout_rate=0, seed=1024, device='cpu'):
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features
        self.attention_W = nn.Parameter(torch.Tensor(embedding_size, self.
            attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1)
            )
        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))
        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor)
        self.dropout = nn.Dropout(dropout_rate)
        self.to(device)

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
        attention_temp = F.relu(torch.tensordot(bi_interaction, self.
            attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(self.normalized_att_score *
            bi_interaction, dim=1)
        attention_output = self.dropout(attention_output)
        afm_out = torch.tensordot(attention_output, self.projection_p, dims
            =([-1], [0]))
        return afm_out


class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,att_embedding_size * head_num)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, in_features, att_embedding_size=8, head_num=2,
        use_res=True, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        embedding_size = in_features
        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, self.
            att_embedding_size * self.head_num))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, self.
            att_embedding_size * self.head_num))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, self.
            att_embedding_size * self.head_num))
        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, self.
                att_embedding_size * self.head_num))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                'Unexpected inputs dimensions %d, expect to be 3 dimensions' %
                len(inputs.shape))
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))
        querys = torch.stack(torch.split(querys, self.att_embedding_size,
            dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size,
            dim=2))
        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
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
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, in_features, layer_num=2, seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList([nn.Parameter(nn.init.
            xavier_normal_(torch.empty(in_features, 1))) for i in range(
            self.layer_num)])
        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(in_features, 1))) for i in range(self.layer_num)])
        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
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
        self.to(device)

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

    def __init__(self, field_size, embedding_size, kernel_type='mat', seed=
        1024, device='cpu'):
        super(OutterProductLayer, self).__init__()
        self.kernel_type = kernel_type
        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size
        if self.kernel_type == 'mat':
            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_pairs,
                embed_size))
        elif self.kernel_type == 'vec':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))
        elif self.kernel_type == 'num':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))
        nn.init.xavier_uniform_(self.kernel)
        self.to(device)

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
            kp = torch.sum(torch.mul(torch.transpose(torch.sum(torch.mul(p,
                self.kernel), dim=-1), 2, 1), q), dim=-1)
        else:
            k = torch.unsqueeze(self.kernel, 0)
            kp = torch.sum(p * q * k, dim=-1)
        return kp


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

    def __init__(self, field_size, conv_kernel_width, conv_filters, device=
        'cpu'):
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
            module_list.append(Conv2dSame(in_channels=in_channels,
                out_channels=out_channels, kernel_size=(width, 1), stride=1
                ).to(self.device))
            module_list.append(torch.nn.Tanh().to(self.device))
            module_list.append(KMaxPooling(k=min(k, filed_shape), axis=2,
                device=self.device).to(self.device))
            filed_shape = min(k, filed_shape)
        self.conv_layer = nn.Sequential(*module_list)
        self.to(device)
        self.filed_shape = filed_shape

    def forward(self, inputs):
        return self.conv_layer(inputs)


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
        self.eps = torch.FloatTensor([1e-08]).to(device)
        self.to(device)

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.device)
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
            mask = self._sequence_mask(user_behavior_length, maxlen=
                uiseq_embed_list.shape[1], dtype=torch.float32)
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
            hist = torch.div(hist, user_behavior_length.type(torch.float32) +
                self.eps)
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

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid',
        weight_normalization=False, return_score=False, supports_masking=
        False, embedding_dim=4, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units,
            embedding_dim=embedding_dim, activation=att_activation,
            dropout_rate=0, use_bn=False)

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
        batch_size, max_length, dim = keys.size()
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    'When supports_masking=True,input must support masking')
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device,
                dtype=keys_length.dtype).repeat(batch_size, 1)
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
        self.to(device)

    def forward(self, input):
        if self.axis < 0 or self.axis >= len(input.shape):
            raise ValueError('axis must be 0~%d,now is %d' % (len(input.
                shape) - 1, self.axis))
        if self.k < 1 or self.k > input.shape[self.axis]:
            raise ValueError('k must be in 1 ~ %d,now k is %d' % (input.
                shape[self.axis], self.k))
        out = torch.topk(input, k=self.k, dim=self.axis, sorted=True)[0]
        return out


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
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size)
            )
        self.register_parameter('weight_ih', self.weight_ih)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size,
            hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hx, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
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
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size)
            )
        self.register_parameter('weight_ih', self.weight_ih)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size,
            hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hx, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
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

    def forward(self, input, att_scores=None, hx=None):
        if not isinstance(input, PackedSequence) or not isinstance(att_scores,
            PackedSequence):
            raise NotImplementedError(
                'DynamicGRU only supports packed input and att_scores')
        input, batch_sizes, sorted_indices, unsorted_indices = input
        att_scores, _, _, _ = att_scores
        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size, dtype=input.
                dtype, device=input.device)
        outputs = torch.zeros(input.size(0), self.hidden_size, dtype=input.
            dtype, device=input.device)
        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(input[begin:begin + batch], hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices,
            unsorted_indices)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


DEFAULT_GROUP_NAME = 'default_group'


class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size',
    'embedding_dim', 'use_hash', 'dtype', 'embedding_name', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False,
        dtype='int32', embedding_name=None, group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                'Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!'
                )
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size,
            embedding_dim, use_hash, dtype, embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparsefeat',
    'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner='mean', length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen,
            combiner, length_name)

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


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False,
    sparse=False, device='cpu'):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat
        ), feature_columns)) if len(feature_columns) else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x,
        VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
    embedding_dict = nn.ModuleDict({feat.embedding_name: nn.Embedding(feat.
        vocabulary_size, feat.embedding_dim if not linear else 1, sparse=
        sparse) for feat in sparse_feature_columns +
        varlen_sparse_feature_columns})
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
    return embedding_dict.to(device)


def get_varlen_pooling_list(embedding_dict, features, feature_index,
    varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.embedding_name](features[:,
            feature_index[feat.name][0]:feature_index[feat.name][1]].long())
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:
                feature_index[feat.name][1]].long() != 0
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking
                =True, device=device)([seq_emb, seq_mask])
        else:
            seq_length = features[:, feature_index[feat.length_name][0]:
                feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking
                =False, device=device)([seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


class Linear(nn.Module):

    def __init__(self, feature_columns, feature_index, init_std=0.0001,
        device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x,
            SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(filter(lambda x: isinstance(x,
            DenseFeat), feature_columns)) if len(feature_columns) else []
        self.varlen_sparse_feature_columns = list(filter(lambda x:
            isinstance(x, VarLenSparseFeat), feature_columns)) if len(
            feature_columns) else []
        self.embedding_dict = create_embedding_matrix(feature_columns,
            init_std, linear=True, sparse=False, device=device)
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in
                self.dense_feature_columns), 1)).to(device)
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](X
            [:, self.feature_index[feat.name][0]:self.feature_index[feat.
            name][1]].long()) for feat in self.sparse_feature_columns]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.
            feature_index[feat.name][1]] for feat in self.dense_feature_columns
            ]
        varlen_embedding_list = get_varlen_pooling_list(self.embedding_dict,
            X, self.feature_index, self.varlen_sparse_feature_columns, self
            .device)
        sparse_embedding_list += varlen_embedding_list
        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(torch.cat(sparse_embedding_list,
                dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(dense_value_list, dim=-1).matmul(
                self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(torch.cat(sparse_embedding_list, dim=-
                1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(dense_value_list, dim=-1).matmul(self.
                weight)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
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
            if (feat.length_name is not None and feat.length_name not in
                features):
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
        raise ValueError(
            'The stop argument has to be None if the value of start is a list.'
            )
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

    def __init__(self, linear_feature_columns, dnn_feature_columns,
        dnn_hidden_units=(128, 128), l2_reg_linear=1e-05, l2_reg_embedding=
        1e-05, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
        dnn_activation='relu', task='binary', device='cpu'):
        super(BaseModel, self).__init__()
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.feature_index = build_input_features(linear_feature_columns +
            dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns
        self.embedding_dict = create_embedding_matrix(dnn_feature_columns,
            init_std, sparse=False, device=device)
        self.linear_model = Linear(linear_feature_columns, self.
            feature_index, device=device)
        self.add_regularization_loss(self.embedding_dict.parameters(),
            l2_reg_embedding)
        self.add_regularization_loss(self.linear_model.parameters(),
            l2_reg_linear)
        self.out = PredictionLayer(task)
        self.to(device)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
        initial_epoch=0, validation_split=0.0, validation_data=None,
        shuffle=True, use_double=False):
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
        :param use_double: Boolean. Whether to use double precision in metric calculation.

        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        if validation_data:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, it must contain either 2 items (x_val, y_val), or 3 items (x_val, y_val, val_sample_weights), or alternatively it could be a dataset or a dataset or a dataset iterator. However we received `validation_data=%s`'
                     % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        elif validation_split and 0.0 < validation_split < 1.0:
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
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.
            concatenate(x, axis=-1)), torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(dataset=train_tensor_data, shuffle=
            shuffle, batch_size=batch_size)
        None
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        None
        for epoch in range(initial_epoch, epochs):
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        y_pred = model(x).squeeze()
                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        total_loss = loss + self.reg_loss + self.aux_loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward(retain_graph=True)
                        optim.step()
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                if use_double:
                                    train_result[name].append(metric_fun(y.
                                        cpu().data.numpy(), y_pred.cpu().
                                        data.numpy().astype('float64')))
                                else:
                                    train_result[name].append(metric_fun(y.
                                        cpu().data.numpy(), y_pred.cpu().
                                        data.numpy()))
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_time = int(time.time() - start_time)
            if verbose > 0:
                None
                eval_str = '{0}s - loss: {1: .4f}'.format(epoch_time, 
                    total_loss_epoch / sample_num)
                for name, result in train_result.items():
                    eval_str += ' - ' + name + ': {0: .4f}'.format(np.sum(
                        result) / steps_per_epoch)
                if len(val_x) and len(val_y):
                    eval_result = self.evaluate(val_x, val_y, batch_size)
                    for name, result in eval_result.items():
                        eval_str += ' - val_' + name + ': {0: .4f}'.format(
                            result)
                None

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size:
        :return: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256, use_double=False):
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
        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x,
            axis=-1)))
        test_loader = DataLoader(dataset=tensor_data, shuffle=False,
            batch_size=batch_size)
        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).cpu().data.numpy()
                pred_ans.append(y_pred)
        if use_double:
            return np.concatenate(pred_ans).astype('float64')
        else:
            return np.concatenate(pred_ans)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict,
        support_dense=True):
        sparse_feature_columns = list(filter(lambda x: isinstance(x,
            SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x,
            DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x,
            VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                'DenseFeat is not supported in dnn_feature_columns')
        sparse_embedding_list = [embedding_dict[feat.embedding_name](X[:,
            self.feature_index[feat.name][0]:self.feature_index[feat.name][
            1]].long()) for feat in sparse_feature_columns]
        varlen_sparse_embedding_list = get_varlen_pooling_list(self.
            embedding_dict, X, self.feature_index,
            varlen_sparse_feature_columns, self.device)
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.
            feature_index[feat.name][1]] for feat in dense_feature_columns]
        return (sparse_embedding_list + varlen_sparse_embedding_list,
            dense_value_list)

    def compute_input_dim(self, feature_columns, include_sparse=True,
        include_dense=True, feature_group=False):
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (
            SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(filter(lambda x: isinstance(x,
            DenseFeat), feature_columns)) if len(feature_columns) else []
        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns)
            )
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in
                sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = torch.zeros((1,), device=self.device)
        for w in weight_list:
            if isinstance(w, tuple):
                l2_reg = torch.norm(w[1], p=p)
            else:
                l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        self.reg_loss = self.reg_loss + reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer, loss=None, metrics=None):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
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
            if loss == 'binary_crossentropy':
                loss_func = F.binary_cross_entropy
            elif loss == 'mse':
                loss_func = F.mse_loss
            elif loss == 'mae':
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-07, normalize=True,
        sample_weight=None, labels=None):
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

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
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
        return metrics_

    @property
    def embedding_size(self):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(filter(lambda x: isinstance(x, (
            SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in
            sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError(
                'embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!'
                )
        return list(embedding_size_set)[0]


class InterestExtractor(nn.Module):

    def __init__(self, input_size, use_neg=False, init_std=0.001, device='cpu'
        ):
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size,
            batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1],
                'sigmoid', init_std=init_std, device=device)
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

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
        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(-
            1, max_length, dim)
        packed_keys = pack_padded_sequence(masked_keys, lengths=
            masked_keys_length, batch_first=True, enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=
            True, padding_value=0.0, total_length=max_length)
        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)
                ).view(-1, max_length, dim)
            aux_loss = self._cal_auxiliary_loss(interests[:, :-1, :],
                masked_keys[:, 1:, :], masked_neg_keys[:, 1:, :], 
                masked_keys_length - 1)
        return interests, aux_loss

    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)
        _, max_seq_length, embedding_size = states.size()
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(
            -1, max_seq_length, embedding_size)
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)
            ).view(-1, max_seq_length, embedding_size)
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 
            1, 1)).view(-1, max_seq_length, embedding_size)
        batch_size = states.size()[0]
        mask = (torch.arange(max_seq_length, device=states.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1)).float()
        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2
        click_p = self.auxiliary_net(click_input.view(batch_size *
            max_seq_length, embedding_size)).view(batch_size, max_seq_length)[
            mask > 0].view(-1, 1)
        click_target = torch.ones(click_p.size(), dtype=torch.float, device
            =click_p.device)
        noclick_p = self.auxiliary_net(noclick_input.view(batch_size *
            max_seq_length, embedding_size)).view(batch_size, max_seq_length)[
            mask > 0].view(-1, 1)
        noclick_target = torch.zeros(noclick_p.size(), dtype=torch.float,
            device=noclick_p.device)
        loss = F.binary_cross_entropy(torch.cat([click_p, noclick_p], dim=0
            ), torch.cat([click_target, noclick_target], dim=0))
        return loss


class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self, input_size, gru_type='GRU', use_neg=False, init_std=
        0.001, att_hidden_size=(64, 16), att_activation='sigmoid',
        att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError('gru_type: {gru_type} is not supported')
        self.gru_type = gru_type
        self.use_neg = use_neg
        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=
                input_size, att_hidden_units=att_hidden_size,
                att_activation=att_activation, weight_normalization=
                att_weight_normalization, return_score=False)
            self.interest_evolution = nn.GRU(input_size=input_size,
                hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=
                input_size, att_hidden_units=att_hidden_size,
                att_activation=att_activation, weight_normalization=
                att_weight_normalization, return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size,
                hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=
                input_size, att_hidden_units=att_hidden_size,
                att_activation=att_activation, weight_normalization=
                att_weight_normalization, return_score=True)
            self.interest_evolution = DynamicGRU(input_size=input_size,
                hidden_size=input_size, gru_type=gru_type)
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        batch_size, max_seq_length, hidden_size = states.size()
        mask = torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == keys_length.view(-1, 1) - 1
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
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim
            ).unsqueeze(1)
        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length,
                batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests,
                batch_first=True, padding_value=0.0, total_length=max_length)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1)
                )
            outputs = outputs.squeeze(1)
        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))
            interests = keys * att_scores.transpose(1, 2)
            packed_interests = pack_padded_sequence(interests, lengths=
                keys_length, batch_first=True, enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)
                ).squeeze(1)
            packed_interests = pack_padded_sequence(keys, lengths=
                keys_length, batch_first=True, enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=
                keys_length, batch_first=True, enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True,
                padding_value=0.0, total_length=max_length)
            outputs = InterestEvolving._get_last_state(outputs, keys_length)
        zero_outputs[mask] = outputs
        return zero_outputs


class Interac(nn.Module):

    def __init__(self, first_size, second_size, emb_size, init_std, sparse=
        False):
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_shenweichen_DeepCTR_Torch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AFMLayer(*[], **{'in_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(AGRUCell(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([4, 4])], {})

    def test_002(self):
        self._check(AUGRUCell(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(AttentionSequencePoolingLayer(*[], **{}), [torch.rand([16384, 4, 4]), torch.rand([16384, 4, 4]), torch.rand([256, 4, 4, 4])], {})

    def test_004(self):
        self._check(BiInteractionPooling(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Conv2dSame(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ConvLayer(*[], **{'field_size': 4, 'conv_kernel_width': [4, 4], 'conv_filters': [4, 4]}), [torch.rand([4, 1, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(CrossNet(*[], **{'in_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(DNN(*[], **{'inputs_dim': 4, 'hidden_units': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(Dice(*[], **{'emb_size': 4}), [torch.rand([4, 4])], {})

    def test_010(self):
        self._check(FM(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(InnerProductLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Interac(*[], **{'first_size': 4, 'second_size': 4, 'emb_size': 4, 'init_std': 4}), [torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_014(self):
        self._check(Linear(*[], **{'feature_columns': [4, 4], 'feature_index': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(LocalActivationUnit(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_016(self):
        self._check(PredictionLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

