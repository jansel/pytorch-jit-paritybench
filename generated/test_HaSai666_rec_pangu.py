import sys
_module = sys.modules[__name__]
del sys
run_multi_task_benchmark_example = _module
run_multi_task_example = _module
inference_example = _module
run_ranking_benchmark_example = _module
run_ranking_example = _module
run_ranking_graph_example = _module
run_ranking_wandb_example = _module
run_set_pretrained_emb_example = _module
rec_pangu = _module
benchmark_trainer = _module
dataset = _module
base_dataset = _module
graph_dataset = _module
multi_task_dataset = _module
process_data = _module
model_pipeline = _module
models = _module
base_model = _module
LGConv = _module
layers = _module
activation = _module
attention = _module
deep = _module
embedding = _module
graph = _module
interaction = _module
sequence = _module
shallow = _module
multi_task = _module
aitm = _module
essm = _module
mlmmoe = _module
mmoe = _module
omoe = _module
sharebottom = _module
ranking = _module
afm = _module
afn = _module
aoanet = _module
autoint = _module
ccpm = _module
dcn = _module
deepfm = _module
fibinet = _module
fm = _module
lightgcn = _module
lr = _module
nfm = _module
wdl = _module
xdeepfm = _module
utils = _module
trainer = _module
check_version = _module
gpu_utils = _module
json_utils = _module
setup = _module

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


import pandas as pd


import numpy as np


import time


from torch.utils.data import Dataset


from collections import defaultdict


import torch.utils.data as D


import copy


from sklearn.metrics import roc_auc_score


from sklearn.metrics import log_loss


from torch import nn


from torch.nn.init import xavier_normal_


from torch.nn.init import constant_


import torch.nn as nn


import torch.nn.functional as F


from itertools import product


from itertools import combinations


import random


class EmbeddingLayer(nn.Module):

    def __init__(self, enc_dict=None, embedding_dim=None):
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()
        self.emb_feature = []
        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Embedding(self.enc_dict[col]['vocab_size'] + 1, self.embedding_dim)})

    def set_weights(self, col_name, embedding_matrix, trainable=True):
        self.embedding_layer[col_name].weight = embedding_matrix
        if not trainable:
            self.embedding_layer[col_name].weight.requires_grad = False

    def forward(self, X, name=None):
        if name == None:
            feature_emb_list = []
            for col in self.emb_feature:
                inp = X[col].long().view(-1, 1)
                feature_emb_list.append(self.embedding_layer[col](inp))
            return torch.stack(feature_emb_list, dim=1).squeeze(2)
        else:
            if 'seq' in name:
                inp = X[name].long()
                fea = self.embedding_layer[name.replace('_seq', '')](inp)
            else:
                inp = X[name].long().view(-1, 1)
                fea = self.embedding_layer[name](inp)
            return fea


class BaseModel(nn.Module):

    def __init__(self, enc_dict, embedding_dim):
        super(BaseModel, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def set_pretrained_weights(self, col_name, pretrained_dict, trainable=True):
        assert col_name in self.enc_dict.keys(), 'Pretrained Embedding Col: {} must be in the {}'.fotmat(col_name, self.enc_dict.keys())
        pretrained_emb_dim = len(list(pretrained_dict.values())[0])
        assert self.embedding_dim == pretrained_emb_dim, 'Pretrained Embedding Dim:{} must be equal to Model Embedding Dim:{}'.format(pretrained_emb_dim, self.embedding_dim)
        pretrained_emb = np.random.rand(self.enc_dict[col_name]['vocab_size'], pretrained_emb_dim)
        for k, v in self.enc_dict[col_name].items():
            if k == 'vocab_size':
                continue
            pretrained_emb[v, :] = pretrained_dict.get(k, np.random.rand(pretrained_emb_dim))
        embeddings = torch.from_numpy(pretrained_emb).float()
        embedding_matrix = torch.nn.Parameter(embeddings)
        self.embedding_layer.set_weights(col_name=col_name, embedding_matrix=embedding_matrix, trainable=trainable)
        logger.info('Successfully Set The Pretrained Embedding Weights for the column:{} With Trainable={}'.format(col_name, trainable))


class Dice(nn.Module):

    def __init__(self, input_dim, eps=1e-09):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + (1 - p) * self.alpha * X
        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, dropout_rate=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.0, use_residual=True, use_scale=False, layer_norm=False, align_to='input'):
        super(MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.align_to = align_to
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            if align_to == 'output':
                self.W_res = nn.Linear(input_dim, self.output_dim, bias=False)
            elif align_to == 'input':
                self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        output = output.view(batch_size, -1, self.output_dim)
        if self.W_res is not None:
            if self.align_to == 'output':
                residual = self.W_res(residual)
            elif self.align_to == 'input':
                output = self.W_res(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output, attention


class MultiHeadSelfAttention(MultiHeadAttention):

    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class SqueezeExcitationLayer(nn.Module):

    def __init__(self, num_fields, reduction_ratio=3):
        super(SqueezeExcitationLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False), nn.ReLU(), nn.Linear(reduced_size, num_fields, bias=False), nn.ReLU())

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation


class MLP_Layer(nn.Module):

    def __init__(self, input_dim, output_dim=None, hidden_units=[], hidden_activations='ReLU', output_activation=None, dropout_rates=0, batch_norm=False, use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.dnn = nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.dnn(inputs)


class GraphLayer(nn.Module):

    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNN_Layer(nn.Module):

    def __init__(self, num_fields, embedding_dim, gnn_layers=3, reuse_graph_layer=False, use_gru=True, use_residual=True, device=None):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim) for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    def build_graph_with_attention(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields)
        alpha = alpha.masked_fill(mask.byte(), float('-inf'))
        graph = F.softmax(alpha, dim=-1)
        return graph

    def forward(self, feature_emb):
        g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h


class InnerProductLayer(nn.Module):
    """ output: product_sum_pooling (bs x 1),
                Bi_interaction_pooling (bs * dim),
                inner_product (bs x f2/2),
                elementwise_product (bs x f2/2 x emb_dim)
    """

    def __init__(self, num_fields=None, output='product_sum_pooling'):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in ['product_sum_pooling', 'Bi_interaction_pooling', 'inner_product', 'elementwise_product']:
            raise ValueError('InnerProductLayer output={} is not supported.'.format(output))
        if num_fields is None:
            if output in ['inner_product', 'elementwise_product']:
                raise ValueError('num_fields is required when InnerProductLayer output={}.'.format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.ByteTensor), requires_grad=False)

    def forward(self, feature_emb):
        if self._output_type in ['product_sum_pooling', 'Bi_interaction_pooling']:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == 'Bi_interaction_pooling':
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == 'elementwise_product':
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == 'inner_product':
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)


class BilinearInteractionLayer(nn.Module):

    def __init__(self, num_fields, embedding_dim, bilinear_type='field_interaction'):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == 'field_all':
            self.bilinear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif self.bilinear_type == 'field_each':
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == 'field_interaction':
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == 'field_all':
            bilinear_list = [(self.bilinear_layer(v_i) * v_j) for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == 'field_each':
            bilinear_list = [(self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]) for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == 'field_interaction':
            bilinear_list = [(self.bilinear_layer[i](v[0]) * v[1]) for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class HolographicInteractionLayer(nn.Module):

    def __init__(self, num_fields, interaction_type='circular_convolution'):
        super(HolographicInteractionLayer, self).__init__()
        self.interaction_type = interaction_type
        if self.interaction_type == 'circular_correlation':
            self.conj_sign = nn.Parameter(torch.tensor([1.0, -1.0]), requires_grad=False)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
        self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)

    def forward(self, feature_emb):
        emb1 = torch.index_select(feature_emb, 1, self.field_p)
        emb2 = torch.index_select(feature_emb, 1, self.field_q)
        if self.interaction_type == 'hadamard_product':
            interact_tensor = emb1 * emb2
        elif self.interaction_type == 'circular_convolution':
            fft1 = torch.rfft(emb1, 1, onesided=False)
            fft2 = torch.rfft(emb2, 1, onesided=False)
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
            interact_tensor = torch.irfft(fft_product, 1, onesided=False)
        elif self.interaction_type == 'circular_correlation':
            fft1_emb = torch.rfft(emb1, 1, onesided=False)
            fft1 = fft1_emb * self.conj_sign.expand_as(fft1_emb)
            fft2 = torch.rfft(emb2, 1, onesided=False)
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
            interact_tensor = torch.irfft(fft_product, 1, onesided=False)
        else:
            raise ValueError('interaction_type={} not supported.'.format(self.interaction_type))
        return interact_tensor


class CrossInteractionLayer(nn.Module):

    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class CrossNet(nn.Module):

    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CompressedInteractionNet(nn.Module):

    def __init__(self, num_fields, cin_layer_units, output_dim=1):
        super(CompressedInteractionNet, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer['layer_' + str(i + 1)] = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_emb):
        pooling_outputs = []
        X_0 = feature_emb
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum('bhd,bmd->bhmd', X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer['layer_' + str(i + 1)](hadamard_tensor).view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        output = self.fc(concate_vec)
        return output


class InteractionMachine(nn.Module):

    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, 'order={} is not supported.'.format(order)
        self.order = order
        self.bn = nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = nn.Linear(order * embedding_dim, 1)

    def second_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2) + 8 * p1 * p3 - 6 * p4) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4 - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120

    def forward(self, X):
        out = []
        Q = X
        if self.order >= 1:
            p1 = Q.sum(dim=1)
            out.append(p1)
            if self.order >= 2:
                Q = Q * X
                p2 = Q.sum(dim=1)
                out.append(self.second_order(p1, p2))
                if self.order >= 3:
                    Q = Q * X
                    p3 = Q.sum(dim=1)
                    out.append(self.third_order(p1, p2, p3))
                    if self.order >= 4:
                        Q = Q * X
                        p4 = Q.sum(dim=1)
                        out.append(self.fourth_order(p1, p2, p3, p4))
                        if self.order == 5:
                            Q = Q * X
                            p5 = Q.sum(dim=1)
                            out.append(self.fifth_order(p1, p2, p3, p4, p5))
        out = torch.cat(out, dim=-1)
        if self.bn is not None:
            out = self.bn(out)
        y = self.fc(out)
        return y


class FM_Layer(nn.Module):

    def __init__(self, final_activation=None, use_bias=True):
        super(FM_Layer, self).__init__()
        self.inner_product_layer = InnerProductLayer(output='product_sum_pooling')
        self.final_activation = final_activation

    def forward(self, feature_emb_list):
        output = self.inner_product_layer(feature_emb_list)
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


class SENET_Layer(nn.Module):

    def __init__(self, num_fields, reduction_ratio=3):
        super(SENET_Layer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False), nn.ReLU(), nn.Linear(reduced_size, num_fields, bias=False), nn.ReLU())

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class MaskedAveragePooling(nn.Module):

    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix != 0).sum(dim=1)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return embedding_vec


class MaskedSumPooling(nn.Module):

    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        return torch.sum(embedding_matrix, dim=1)


class KMaxPooling(nn.Module):

    def __init__(self, k, dim):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, X):
        index = X.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        output = X.gather(self.dim, index)
        return output


def get_dnn_input_dim(enc_dict, embedding_dim):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse * embedding_dim + num_dense


def get_linear_input(enc_dict, data):
    res_data = []
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data, axis=1)
    return res_data


class LR_Layer(nn.Module):

    def __init__(self, enc_dict):
        super(LR_Layer, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, 1)
        self.fc = nn.Linear(self.dnn_input_dim, 1)

    def forward(self, data):
        sparse_emb = self.emb_layer(data).squeeze(-1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat((sparse_emb, dense_input), dim=1)
        out = self.fc(dnn_input)
        return out


def get_feature_num(enc_dict):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse, num_dense


class AITM(BaseModel):

    def __init__(self, embedding_dim=32, tower_dims=[400, 400, 400], drop_prob=[0.1, 0.1, 0.1], enc_dict=None, device=None):
        super(AITM, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.tower_dims = tower_dims
        self.drop_prob = drop_prob
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        self.tower_input_size = self.num_sparse_fea * self.embedding_dim
        self.click_tower = MLP_Layer(input_dim=self.tower_input_size, hidden_units=self.tower_dims, hidden_activations='relu', dropout_rates=self.drop_prob)
        self.conversion_tower = MLP_Layer(input_dim=self.tower_input_size, hidden_units=self.tower_dims, hidden_activations='relu', dropout_rates=self.drop_prob)
        self.attention_layer = MultiHeadSelfAttention(self.tower_dims[-1])
        self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], tower_dims[-1]), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_embedding = self.embedding_layer(data)
        feature_embedding = feature_embedding.flatten(start_dim=1)
        tower_click = self.click_tower(feature_embedding)
        tower_conversion = torch.unsqueeze(self.conversion_tower(feature_embedding), 1)
        info = torch.unsqueeze(self.info_layer(tower_click), 1)
        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))
        ait = torch.sum(ait, dim=1)
        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)
        if is_training:
            loss = self.loss(data['task1_label'], click, data['task2_label'], conversion)
            output_dict = {'task1_pred': click, 'task2_pred': conversion, 'loss': loss}
        else:
            output_dict = {'task1_pred': click, 'task2_pred': conversion}
        return output_dict

    def loss(self, click_label, click_pred, conversion_label, conversion_pred, constraint_weight=0.6):
        click_label = click_label
        conversion_label = conversion_label
        click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
        conversion_loss = nn.functional.binary_cross_entropy(conversion_pred, conversion_label)
        label_constraint = torch.maximum(conversion_pred - click_pred, torch.zeros_like(click_label))
        constraint_loss = torch.sum(label_constraint)
        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss


class ESSM(BaseModel):

    def __init__(self, embedding_dim=40, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(ESSM, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim
        self.ctr_layer = MLP_Layer(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim, hidden_activations='relu', dropout_rates=self.dropouts)
        self.cvr_layer = MLP_Layer(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim, hidden_activations='relu', dropout_rates=self.dropouts)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        click = self.sigmoid(self.ctr_layer(hidden))
        conversion = self.sigmoid(self.cvr_layer(hidden))
        pctrcvr = click * conversion
        if is_training:
            loss = self.loss(click, pctrcvr, data)
            output_dict = {'task1_pred': click, 'task2_pred': conversion, 'loss': loss}
        else:
            output_dict = {'task1_pred': click, 'task2_pred': conversion}
        return output_dict

    def loss(self, click, conversion, data, weight=0.5):
        ctr_loss = nn.functional.binary_cross_entropy(click.squeeze(-1), data['task1_label'])
        cvr_loss = nn.functional.binary_cross_entropy(conversion.squeeze(-1), data['task2_label'])
        loss = cvr_loss + weight * ctr_loss
        return loss


class MLMMOE(BaseModel):

    def __init__(self, num_task=2, n_expert=3, embedding_dim=40, mmoe_hidden_dim=128, expert_activation=None, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(MLMMOE, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        self.level_gates = [torch.nn.Parameter(torch.rand(n_expert, 1), requires_grad=True) for _ in range(n_expert)]
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.set_device(device)
        self.apply(self._init_weights)

    def set_device(self, device):
        for i in range(self.num_task):
            self.gates[i] = self.gates[i]
            self.gates_bias[i] = self.gates_bias[i]
        for i in range(self.n_expert):
            self.level_gates[i] = self.level_gates[i]
        None

    def forward(self, data, is_training=True):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        level_out = []
        for idx, gate in enumerate(self.level_gates):
            gate = nn.Softmax(dim=0)(gate)
            temp_out = torch.einsum('abc, cd -> abd', experts_out, gate)
            level_out.append(temp_out)
        level_out = torch.cat(level_out, axis=-1)
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)
        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = level_out * expanded_gate_output.expand_as(level_out)
            outs.append(torch.sum(weighted_expert_output, 2))
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight == None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i in range(len(task_outputs)):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1), data[f'task{i + 1}_label'])
        return loss


class MMOE(BaseModel):

    def __init__(self, num_task=2, n_expert=3, embedding_dim=40, mmoe_hidden_dim=128, expert_activation=None, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(MMOE, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.set_device(device)
        self.apply(self._init_weights)

    def set_device(self, device):
        for i in range(self.num_task):
            self.gates[i] = self.gates[i]
            self.gates_bias[i] = self.gates_bias[i]
        None

    def forward(self, data, is_training=True):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)
        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(experts_out)
            outs.append(torch.sum(weighted_expert_output, 2))
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight == None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i in range(len(task_outputs)):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1) + 1e-06, data[f'task{i + 1}_label'])
        return loss


class OMOE(BaseModel):

    def __init__(self, num_task=2, n_expert=3, embedding_dim=40, omoe_hidden_dim=128, expert_activation=None, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(OMOE, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.omoe_hidden_dim = omoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, omoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(omoe_hidden_dim, n_expert), requires_grad=True)
        self.gate = torch.nn.Parameter(torch.rand(n_expert, 1), requires_grad=True)
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [omoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        gate = nn.Softmax(dim=0)(self.gate)
        gate_out = torch.einsum('abc, cd -> abd', experts_out, gate).squeeze(-1)
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = gate_out
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight == None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i in range(len(task_outputs)):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1), data[f'task{i + 1}_label'])
        return loss


class ShareBottom(BaseModel):

    def __init__(self, num_task=2, embedding_dim=40, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(ShareBottom, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.apply(self._init_weights)
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        out = torch.cat([feature_emb, dense_fea], axis=-1)
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = out
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight == None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i in range(len(task_outputs)):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1), data[f'task{i + 1}_label'])
        return loss


class AFM(BaseModel):

    def __init__(self, embedding_dim=32, hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(AFM, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.senet_layer = SENET_Layer(self.num_sparse, 3)
        self.bilinear_interaction = BilinearInteractionLayer(self.num_sparse, embedding_dim, 'field_interaction')
        input_dim = self.num_sparse * (self.num_sparse - 1) * self.embedding_dim + self.num_dense
        self.dnn = MLP_Layer(input_dim=input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        y_pred = self.lr(data)
        feature_emb = self.embedding_layer(data)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        comb_out = torch.cat([comb_out, dense_input], dim=1)
        y_pred += self.dnn(comb_out)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class AFN(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], afn_hidden_units=[64, 64, 64], ensemble_dnn=True, loss_fun='torch.nn.BCELoss()', logarithmic_neurons=5, enc_dict=None):
        super(AFN, self).__init__(enc_dict, embedding_dim)
        self.dnn_hidden_units = dnn_hidden_units
        self.afn_hidden_units = afn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.coefficient_W = nn.Linear(self.num_sparse, logarithmic_neurons, bias=False)
        self.dense_layer = MLP_Layer(input_dim=embedding_dim * logarithmic_neurons, output_dim=1, hidden_units=afn_hidden_units, use_bias=True)
        self.log_batch_norm = nn.BatchNorm1d(self.num_sparse)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.ensemble_dnn = ensemble_dnn
        self.apply(self._init_weights)
        if ensemble_dnn:
            self.embedding_layer2 = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
            self.dnn = MLP_Layer(input_dim=embedding_dim * self.num_sparse, output_dim=1, hidden_units=dnn_hidden_units, use_bias=True)
            self.fc = nn.Linear(2, 1)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        dnn_input = self.logarithmic_net(feature_emb)
        afn_out = self.dense_layer(dnn_input)
        if self.ensemble_dnn:
            feature_emb2 = self.embedding_layer2(data)
            dnn_out = self.dnn(feature_emb2.flatten(start_dim=1))
            y_pred = self.fc(torch.cat([afn_out, dnn_out], dim=-1))
        else:
            y_pred = afn_out
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-05)
        log_feature_emb = torch.log(feature_emb)
        log_feature_emb = self.log_batch_norm(log_feature_emb)
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out)
        cross_out = self.exp_batch_norm(cross_out)
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out


class GeneralizedInteraction(nn.Module):

    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum('bnh,bnd->bnhd', B_0.repeat(1, self.input_subspaces, 1), B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim))
        fusion = torch.matmul(outer_product.permute(0, 2, 3, 1), self.alpha)
        fusion = self.W * fusion.permute(0, 3, 1, 2)
        B_i = torch.matmul(fusion, self.h).squeeze(-1)
        return B_i


class GeneralizedInteractionNet(nn.Module):

    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces, num_subspaces, num_fields, embedding_dim) for i in range(num_layers)])

    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i


class AOANet(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], num_interaction_layers=3, num_subspaces=4, loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(AOANet, self).__init__(enc_dict, embedding_dim)
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.dnn = MLP_Layer(input_dim=self.embedding_dim * self.num_sparse + self.num_dense, output_dim=None, hidden_units=self.dnn_hidden_units)
        self.gin = GeneralizedInteractionNet(num_interaction_layers, num_subspaces, self.num_sparse, self.embedding_dim)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * self.embedding_dim, 1)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        emb_flatten = feature_emb.flatten(start_dim=1)
        dnn_out = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
        interact_out = self.gin(feature_emb).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class AutoInt(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], attention_layers=1, num_heads=1, attention_dim=8, loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(AutoInt, self).__init__(enc_dict, embedding_dim)
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.lr_layer = LR_Layer(enc_dict=enc_dict)
        self.dnn = MLP_Layer(input_dim=self.embedding_dim * self.num_sparse + self.num_dense, output_dim=1, hidden_units=self.dnn_hidden_units)
        self.self_attention = nn.Sequential(*[MultiHeadSelfAttention(self.embedding_dim if i == 0 else num_heads * attention_dim, attention_dim=attention_dim, num_heads=num_heads, align_to='output') for i in range(attention_layers)])
        self.fc = nn.Linear(self.num_sparse * attention_dim * num_heads, 1)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        attention_out = self.self_attention(feature_emb)
        attention_out = attention_out.flatten(start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            y_pred += self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(data)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class CCPM_ConvLayer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """

    def __init__(self, num_fields, channels=[3], kernel_heights=[3], activation='Tanh'):
        super(CCPM_ConvLayer, self).__init__()
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        elif len(kernel_heights) != len(channels):
            raise ValueError('channels={} and kernel_heights={} should have the same length.'.format(channels, kernel_heights))
        module_list = []
        self.channels = [1] + channels
        layers = len(kernel_heights)
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i - 1]
            out_channels = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            module_list.append(nn.ZeroPad2d((0, 0, kernel_height - 1, kernel_height - 1)))
            module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, 1)))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(KMaxPooling(k, dim=2))
            module_list.append(get_activation(activation))
        self.conv_layer = nn.Sequential(*module_list)

    def forward(self, X):
        return self.conv_layer(X)


class CCPM(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], channels=[4, 4, 2], kernel_heights=[6, 5, 3], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(CCPM, self).__init__(enc_dict, embedding_dim)
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.conv_layer = CCPM_ConvLayer(self.num_sparse, channels=channels, kernel_heights=kernel_heights)
        conv_out_dim = 3 * embedding_dim * channels[-1]
        self.fc = nn.Linear(conv_out_dim, 1)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        conv_in = torch.unsqueeze(feature_emb, 1)
        conv_out = self.conv_layer(conv_in)
        flatten_out = torch.flatten(conv_out, start_dim=1)
        y_pred = self.fc(flatten_out)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class DCN(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', crossing_layers=3, enc_dict=None):
        super(DCN, self).__init__(enc_dict, embedding_dim)
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        input_dim = self.num_sparse * self.embedding_dim + self.num_dense
        self.crossnet = CrossNet(input_dim, crossing_layers)
        self.fc = nn.Linear(input_dim, 1)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(torch.cat([flat_feature_emb, dense_input], dim=1))
        y_pred = self.fc(cross_out).sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class DeepFM(BaseModel):

    def __init__(self, embedding_dim=32, hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(DeepFM, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.fm = FM_Layer()
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        sparse_embedding = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        fm_out = self.fm(sparse_embedding)
        emb_flatten = sparse_embedding.flatten(start_dim=1)
        dnn_input = torch.cat((emb_flatten, dense_input), dim=1)
        dnn_output = self.dnn(dnn_input)
        y_pred = torch.sigmoid(fm_out + dnn_output)
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class FiBiNet(BaseModel):

    def __init__(self, embedding_dim=32, hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(FiBiNet, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.senet_layer = SENET_Layer(self.num_sparse, 3)
        self.bilinear_interaction = BilinearInteractionLayer(self.num_sparse, embedding_dim, 'field_interaction')
        input_dim = self.num_sparse * (self.num_sparse - 1) * self.embedding_dim + self.num_dense
        self.dnn = MLP_Layer(input_dim=input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        y_pred = self.lr(data)
        feature_emb = self.embedding_layer(data)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        comb_out = torch.cat([comb_out, dense_input], dim=1)
        y_pred += self.dnn(comb_out)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class FM(BaseModel):

    def __init__(self, embedding_dim=32, loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(FM, self).__init__(enc_dict, embedding_dim)
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.fm = FM_Layer()
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        y_pred = self.fm(feature_emb)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class lightgcn(nn.Module):

    def __init__(self, num_nodes, embedding_dim, num_layers):
        super(lightgcn, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.loss_fun = nn.MSELoss()
        alpha = 1.0 / (num_layers + 1)
        self.alpha = torch.tensor([alpha] * (num_layers + 1))
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.decoder = torch.nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def get_embedding(self, edge_index):
        x = self.embedding.weight
        out = x * self.alpha[0]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]
        return out

    def forward(self, edge_index, edge_label_index, edge_label):
        """预测节点对的rating"""
        out = self.get_embedding(edge_index)
        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        pred = self.decoder(torch.cat([out_src, out_dst], dim=-1))
        target = edge_label.view(-1, 1)
        loss = self.loss_fun(pred, target)
        result = dict()
        result['pred'] = pred
        result['loss'] = loss
        return result


class LR(nn.Module):

    def __init__(self, loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(LR, self).__init__()
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)

    def forward(self, data, is_training=True):
        y_pred = self.lr_layer(data)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class NFM(BaseModel):

    def __init__(self, embedding_dim=32, hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(NFM, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.inner_product_layer = InnerProductLayer(output='Bi_interaction_pooling')
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP_Layer(input_dim=self.embedding_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        y_pred = self.lr(data)
        batch_size = y_pred.shape[0]
        sparse_embedding = self.embedding_layer(data)
        inner_product_tensor = self.inner_product_layer(sparse_embedding)
        bi_pooling_tensor = inner_product_tensor.view(batch_size, -1)
        y_pred += self.dnn(bi_pooling_tensor)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class WDL(BaseModel):

    def __init__(self, embedding_dim=32, hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(WDL, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        wide_logit = self.lr(data)
        sparse_emb = self.embedding_layer(data)
        sparse_emb = sparse_emb.flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat([sparse_emb, dense_input], dim=1)
        deep_logit = self.dnn(dnn_input)
        y_pred = (wide_logit + deep_logit).sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class xDeepFM(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], cin_layer_units=[16, 16, 16], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(xDeepFM, self).__init__(enc_dict, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.dnn = MLP_Layer(input_dim=self.num_sparse * self.embedding_dim + self.num_dense, output_dim=1, hidden_units=self.dnn_hidden_units)
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)
        self.cin = CompressedInteractionNet(self.num_sparse, cin_layer_units, output_dim=1)
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        feature_emb = self.embedding_layer(data)
        lr_logit = self.lr_layer(data)
        cin_logit = self.cin(feature_emb)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            dnn_logit = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
            y_pred = lr_logit + cin_logit + dnn_logit
        else:
            y_pred = lr_logit + cin_logit
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearInteractionLayer,
     lambda: ([], {'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CCPM_ConvLayer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     True),
    (CompressedInteractionNet,
     lambda: ([], {'num_fields': 4, 'cin_layer_units': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CrossInteractionLayer,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossNet,
     lambda: ([], {'input_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dice,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (FM_Layer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FiGNN_Layer,
     lambda: ([], {'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GeneralizedInteraction,
     lambda: ([], {'input_subspaces': 4, 'output_subspaces': 4, 'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (GeneralizedInteractionNet,
     lambda: ([], {'num_layers': 1, 'num_subspaces': 4, 'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GraphLayer,
     lambda: ([], {'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (InnerProductLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InteractionMachine,
     lambda: ([], {'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KMaxPooling,
     lambda: ([], {'k': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (MLP_Layer,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedAveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedSumPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SENET_Layer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SqueezeExcitationLayer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_HaSai666_rec_pangu(_paritybench_base):
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

