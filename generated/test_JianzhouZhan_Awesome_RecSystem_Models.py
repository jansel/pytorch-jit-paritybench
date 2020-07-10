import sys
_module = sys.modules[__name__]
del sys
DeepCrossNetwork_PyTorch = _module
DeepFM_PyTorch = _module
DeepFM_TensorFlow = _module
FFM_Multi_PyTorch = _module
FFM_PyTorch = _module
FM_Multi_PyTorch = _module
FM_PyTorch = _module
FM_TensorFlow = _module
PNN_PyTorch = _module
PNN_TensorFlow = _module
xDeepFM_PyTorch = _module
Criteo = _module
DCN_dataPreprocess = _module
forDCN = _module
forOtherModels = _module
dataPreprocess_PyTorch = _module
dataPreprocess_TensorFlow = _module
xDeepFM_dataPreprocess = _module
util = _module
Movielens100K = _module
data = _module
load_data_util = _module
train_model_util_PyTorch = _module
train_model_util_TensorFlow = _module

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


import re


import math


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from sklearn.metrics import roc_auc_score


from sklearn import preprocessing


import torch.utils.data as data


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DCN_layer(nn.Module):

    def __init__(self, num_dense_feat, num_sparse_feat_list, dropout_deep, deep_layer_sizes, reg_l1=0.01, reg_l2=0.01, num_cross_layers=4):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_dense_feat = num_dense_feat
        embedding_sizes = []
        self.sparse_feat_embeddings = nn.ModuleList()
        for i, num_sparse_feat in enumerate(num_sparse_feat_list):
            embedding_dim = min(num_sparse_feat, 6 * int(np.power(num_sparse_feat, 1 / 4)))
            embedding_sizes.append(embedding_dim)
            feat_embedding = nn.Embedding(num_sparse_feat, embedding_dim)
            nn.init.xavier_uniform_(feat_embedding.weight)
            feat_embedding
            self.sparse_feat_embeddings.append(feat_embedding)
        self.num_cross_layers = num_cross_layers
        self.deep_layer_sizes = deep_layer_sizes
        self.input_dim = num_dense_feat + sum(embedding_sizes)
        self.cross_bias = nn.Parameter(torch.randn(num_cross_layers, self.input_dim))
        nn.init.zeros_(self.cross_bias)
        self.cross_W = nn.Parameter(torch.randn(num_cross_layers, self.input_dim))
        nn.init.xavier_uniform_(self.cross_W)
        self.batchNorm_list = nn.ModuleList()
        for _ in range(num_cross_layers):
            self.batchNorm_list.append(nn.BatchNorm1d(self.input_dim))
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))
        self.fc = nn.Linear(self.input_dim + all_dims[-1], 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, feat_index_list, dense_x, use_dropout=True):
        x0 = dense_x
        for i, feat_index in enumerate(feat_index_list):
            sparse_x = self.sparse_feat_embeddings[i](feat_index)
            x0 = torch.cat((x0, sparse_x), dim=1)
        x_cross = x0
        for i in range(self.num_cross_layers):
            W = torch.unsqueeze(self.cross_W[(i), :].T, dim=1)
            xT_W = torch.mm(x_cross, W)
            x_cross = torch.mul(x0, xT_W) + self.cross_bias[(i), :] + x_cross
            x_cross = self.batchNorm_list[i](x_cross)
        x_deep = x0
        for i in range(1, len(self.deep_layer_sizes) + 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            x_deep = getattr(self, 'batchNorm_' + str(i))(x_deep)
            x_deep = F.relu(x_deep)
            if use_dropout:
                x_deep = getattr(self, 'dropout_' + str(i))(x_deep)
        x_stack = torch.cat((x_cross, x_deep), dim=1)
        output = self.fc(x_stack)
        return output


class DeepFM(nn.Module):

    def __init__(self, num_feat, num_field, dropout_deep, dropout_fm, reg_l1=0.01, reg_l2=0.01, layer_sizes=[400, 400, 400], embedding_size=10):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat
        self.num_field = num_field
        self.embedding_size = embedding_size
        self.layer_sizes = layer_sizes
        self.dropout_deep = dropout_deep
        self.dropout_fm = dropout_fm
        self.first_weights = nn.Embedding(num_feat, 1)
        nn.init.xavier_uniform_(self.first_weights.weight)
        self.feat_embeddings = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(self.feat_embeddings.weight)
        all_dims = [self.num_field * self.embedding_size] + layer_sizes
        for i in range(1, len(layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))
        self.fc = nn.Linear(num_field + embedding_size + all_dims[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=True):
        feat_value = torch.unsqueeze(feat_value, dim=2)
        first_weights = self.first_weights(feat_index)
        first_weight_value = torch.mul(first_weights, feat_value)
        y_first_order = torch.sum(first_weight_value, dim=2)
        if use_dropout:
            y_first_order = nn.Dropout(self.dropout_fm[0])(y_first_order)
        secd_feat_emb = self.feat_embeddings(feat_index)
        feat_emd_value = secd_feat_emb * feat_value
        summed_feat_emb = torch.sum(feat_emd_value, 1)
        interaction_part1 = torch.pow(summed_feat_emb, 2)
        squared_feat_emd_value = torch.pow(feat_emd_value, 2)
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)
        y_secd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)
        if use_dropout:
            y_secd_order = nn.Dropout(self.dropout_fm[1])(y_secd_order)
        y_deep = feat_emd_value.reshape(-1, self.num_field * self.embedding_size)
        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)
        for i in range(1, len(self.layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)
        concat_input = torch.cat((y_first_order, y_secd_order, y_deep), dim=1)
        output = self.fc(concat_input)
        return output


class FFM_layer(nn.Module):

    def __init__(self, num_feat, num_field, reg_l1=0.0001, reg_l2=0.0001, embedding_size=10):
        super(FFM_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.embedding_size = embedding_size
        pass


class FM_layer(nn.Module):

    def __init__(self, num_feat, num_field, reg_l1=0.01, reg_l2=0.01, embedding_size=16):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat
        self.num_field = num_field
        self.embedding_size = embedding_size
        self.first_weights = nn.Embedding(num_feat, 1)
        nn.init.xavier_uniform_(self.first_weights.weight)
        self.bias = nn.Parameter(torch.randn(1))
        self.feat_embeddings = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(self.feat_embeddings.weight)

    def forward(self, feat_index, feat_value):
        feat_value = torch.unsqueeze(feat_value, dim=2)
        first_weights = self.first_weights(feat_index)
        first_weight_value = torch.mul(first_weights, feat_value)
        first_weight_value = torch.squeeze(first_weight_value, dim=2)
        y_first_order = torch.sum(first_weight_value, dim=1)
        secd_feat_emb = self.feat_embeddings(feat_index)
        feat_emd_value = torch.mul(secd_feat_emb, feat_value)
        summed_feat_emb = torch.sum(feat_emd_value, 1)
        interaction_part1 = torch.pow(summed_feat_emb, 2)
        squared_feat_emd_value = torch.pow(feat_emd_value, 2)
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)
        y_secd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)
        y_secd_order = torch.sum(y_secd_order, dim=1)
        output = self.bias + y_first_order + y_secd_order
        output = torch.unsqueeze(output, dim=1)
        return output


class PNN_layer(nn.Module):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10, reg_l1=0.01, reg_l2=1e-05, embedding_size=10, product_type='outer'):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat
        self.num_field = num_field
        self.product_layer_dim = product_layer_dim
        self.dropout_deep = dropout_deep
        feat_embeddings = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embeddings.weight)
        self.feat_embeddings = feat_embeddings
        linear_weights = torch.randn((product_layer_dim, num_field, embedding_size))
        nn.init.xavier_uniform_(linear_weights)
        self.linear_weights = nn.Parameter(linear_weights)
        self.product_type = product_type
        if product_type == 'inner':
            theta = torch.randn((product_layer_dim, num_field))
            nn.init.xavier_uniform_(theta)
            self.theta = nn.Parameter(theta)
        else:
            quadratic_weights = torch.randn((product_layer_dim, embedding_size, embedding_size))
            nn.init.xavier_uniform_(quadratic_weights)
            self.quadratic_weights = nn.Parameter(quadratic_weights)
        self.deep_layer_sizes = deep_layer_sizes
        all_dims = [self.product_layer_dim + self.product_layer_dim] + deep_layer_sizes
        for i in range(1, len(deep_layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))
        self.fc = nn.Linear(deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=True):
        feat_embedding = self.feat_embeddings(feat_index)
        lz = torch.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)
        if self.product_type == 'inner':
            theta = torch.einsum('bnm,dn->bdnm', feat_embedding, self.theta)
            lp = torch.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            embed_sum = torch.sum(feat_embedding, dim=1)
            p = torch.einsum('bm,bn->bmn', embed_sum, embed_sum)
            lp = torch.einsum('bmn,dmn->bd', p, self.quadratic_weights)
        y_deep = torch.cat((lz, lp), dim=1)
        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)
        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)
        output = self.fc(y_deep)
        return output


class xDeepFM_layer(nn.Module):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, cin_layer_sizes, split_half=True, reg_l1=0.01, reg_l2=1e-05, embedding_size=10):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat
        self.num_field = num_field
        self.cin_layer_sizes = cin_layer_sizes
        self.deep_layer_sizes = deep_layer_sizes
        self.embedding_size = embedding_size
        self.dropout_deep = dropout_deep
        self.split_half = split_half
        self.input_dim = num_field * embedding_size
        feat_embedding = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embedding.weight)
        self.feat_embedding = feat_embedding
        cin_layer_dims = [self.num_field] + cin_layer_sizes
        prev_dim, fc_input_dim = self.num_field, 0
        self.conv1ds = nn.ModuleList()
        for k in range(1, len(cin_layer_dims)):
            conv1d = nn.Conv1d(cin_layer_dims[0] * prev_dim, cin_layer_dims[k], 1)
            nn.init.xavier_uniform_(conv1d.weight)
            self.conv1ds.append(conv1d)
            if self.split_half and k != len(self.cin_layer_sizes):
                prev_dim = cin_layer_dims[k] // 2
            else:
                prev_dim = cin_layer_dims[k]
            fc_input_dim += prev_dim
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))
        self.linear = nn.Linear(self.input_dim, 1)
        self.output_layer = nn.Linear(1 + fc_input_dim + deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=True):
        fea_embedding = self.feat_embedding(feat_index)
        x0 = fea_embedding
        linear_part = self.linear(fea_embedding.reshape(-1, self.input_dim))
        x_list = [x0]
        res = []
        for k in range(1, len(self.cin_layer_sizes) + 1):
            z_k = torch.einsum('bhd,bmd->bhmd', x_list[-1], x_list[0])
            z_k = z_k.reshape(x0.shape[0], x_list[-1].shape[1] * x0.shape[1], x0.shape[2])
            x_k = self.conv1ds[k - 1](z_k)
            x_k = torch.relu(x_k)
            if self.split_half and k != len(self.cin_layer_sizes):
                next_hidden, hi = torch.split(x_k, x_k.shape[1] // 2, 1)
            else:
                next_hidden, hi = x_k, x_k
            x_list.append(next_hidden)
            res.append(hi)
        res = torch.cat(res, dim=1)
        res = torch.sum(res, dim=2)
        y_deep = fea_embedding.reshape(-1, self.num_field * self.embedding_size)
        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)
        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)
        concat_input = torch.cat((linear_part, res, y_deep), dim=1)
        output = self.output_layer(concat_input)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FM_layer,
     lambda: ([], {'num_feat': 4, 'num_field': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.rand([4, 4])], {}),
     True),
]

class Test_JianzhouZhan_Awesome_RecSystem_Models(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

