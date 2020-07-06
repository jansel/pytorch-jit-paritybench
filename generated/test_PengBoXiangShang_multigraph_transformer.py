import sys
_module = sys.modules[__name__]
del sys
cnn_baselines = _module
train_inceptionv3 = _module
train_mobilenetv2 = _module
gcn_baselines = _module
rnn_baselines = _module
train_bigru = _module
QuickdrawDataset = _module
QuickdrawDataset4dict_2nn = _module
QuickdrawDataset4dict_2nn4nn = _module
QuickdrawDataset4dict_2nn4nn6nn = _module
QuickdrawDataset4dict_2nn4nnjnn = _module
QuickdrawDataset4dict_2nnjnn = _module
QuickdrawDataset4dict_4nn = _module
QuickdrawDataset4dict_4nnjnn = _module
QuickdrawDataset4dict_6nn = _module
QuickdrawDataset4dict_bigru = _module
QuickdrawDataset4dict_fully_connected_graph_attention_mask = _module
QuickdrawDataset4dict_fully_connected_stroke_attention_mask = _module
QuickdrawDataset4dict_jnn = _module
QuickdrawDataset4dict_random_attention_mask = _module
dataloader = _module
Bidirectional_GRU = _module
network = _module
gra_transf_inpt5_new_dropout_2layerMLP = _module
gra_transf_inpt5_new_dropout_2layerMLP_2_adj_mtx = _module
gra_transf_inpt5_new_dropout_2layerMLP_3_adj_mtx = _module
graph_attention_net = _module
graph_mlp_net = _module
graph_transformer_layers_new_dropout = _module
graph_transformer_layers_new_dropout_2_adj_mtx = _module
graph_transformer_layers_new_dropout_3_adj_mtx = _module
test_case_4_QuickdrawDataset4dict_2nn = _module
test_case_4_QuickdrawDataset4dict_4nn = _module
test_case_4_QuickdrawDataset4dict_6nn = _module
test_case_4_QuickdrawDataset4dict_fully_connected_stroke_attention_mask = _module
test_case_4_QuickdrawDataset4dict_jnn = _module
test_case_4_QuickdrawDataset4dict_random_attention_mask = _module
train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nnjnn_early_stop = _module
train_gra_transf_inpt5_new_dropout_2layerMLP_4nn_early_stop = _module
train_gra_transf_inpt5_new_dropout_2layerMLP_4nnjnn_early_stop = _module
train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_graph_early_stop = _module
AverageMeter = _module
EarlyStopping = _module
Logger = _module
utils = _module
accuracy = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import collections


import time


import numpy as np


import torch


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torchvision


import torchvision.transforms as transforms


import torchvision.models as models


from torch.autograd import Variable


from torch.nn.modules.module import Module


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


import torch.utils.data as data


import math


import torch.nn.functional as F


from torch import nn


import random


class GRUNet(nn.Module):

    def __init__(self, network_configs):
        super(GRUNet, self).__init__()
        self.coord_embed = nn.Linear(network_configs['coord_input_dim'], network_configs['embed_dim'], bias=False)
        self.feat_embed = nn.Embedding(network_configs['feat_dict_size'], network_configs['embed_dim'])
        self.hidden_size = network_configs['hidden_size']
        self.gru = nn.GRU(input_size=network_configs['embed_dim'], hidden_size=network_configs['hidden_size'], num_layers=network_configs['num_layers'], batch_first=True, dropout=network_configs['dropout'], bidirectional=True)
        self.out_layer = nn.Linear(network_configs['hidden_size'] * 2, network_configs['num_classes'])

    def forward(self, coordinate, flag_bits, position_encoding):
        x = self.coord_embed(coordinate) + self.feat_embed(flag_bits) + self.feat_embed(position_encoding)
        self.rnn_hidden_feature, h = self.gru(x)
        featur = torch.cat((self.rnn_hidden_feature[:, (-1), :self.hidden_size], self.rnn_hidden_feature[:, (-1), self.hidden_size:]), 1)
        x = self.out_layer(featur)
        return x, featur


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            assert embed_dim is not None, 'Provide either embed_dim or val_dim'
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Args:
            q: Input queries (batch_size, n_query, input_dim)
            h: Input data (batch_size, graph_size, input_dim)
            mask: Input attention mask (batch_size, n_query, graph_size)
                  or viewable as that (i.e. can be 2 dim if n_query == 1);
                  Mask should contain -inf if attention is not possible 
                  (i.e. mask is a negative adjacency matrix)
        
        Returns: 
            out: Updated data after attention (batch_size, graph_size, input_dim)
        """
        if h is None:
            h = q
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, 'Wrong embedding dimension of input'
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        shp = self.n_heads, batch_size, graph_size, -1
        shp_q = self.n_heads, batch_size, n_query, -1
        dropt1_qflat = self.dropout_1(qflat)
        Q = torch.matmul(dropt1_qflat, self.W_query).view(shp_q)
        dropt2_hflat = self.dropout_2(hflat)
        K = torch.matmul(dropt2_hflat, self.W_key).view(shp)
        dropt3_hflat = self.dropout_3(hflat)
        V = torch.matmul(dropt3_hflat, self.W_val).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility = compatibility + mask.type_as(compatibility)
        attn = F.softmax(compatibility, dim=-1)
        heads = torch.matmul(attn, V)
        out = torch.mm(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim), self.W_out.view(-1, self.embed_dim)).view(batch_size, n_query, self.embed_dim)
        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        normalizer_class = {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d}.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, affine=True)
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, 'Unknown normalizer type'
            return input


class PositionWiseFeedforward(nn.Module):

    def __init__(self, embed_dim, feedforward_dim=512, dropout=0.1):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim, bias=True), nn.ReLU())
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)


class MultiGraphTransformerLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim, normalization='batch', dropout=0.1):
        super(MultiGraphTransformerLayer, self).__init__()
        self.self_attention1 = SkipConnection(MultiHeadAttention(n_heads=n_heads, input_dim=embed_dim, embed_dim=embed_dim, dropout=dropout))
        self.self_attention2 = SkipConnection(MultiHeadAttention(n_heads=n_heads, input_dim=embed_dim, embed_dim=embed_dim, dropout=dropout))
        self.self_attention3 = SkipConnection(MultiHeadAttention(n_heads=n_heads, input_dim=embed_dim, embed_dim=embed_dim, dropout=dropout))
        self.tmp_linear_layer = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim * 3, embed_dim, bias=True), nn.ReLU())
        self.norm1 = Normalization(embed_dim, normalization)
        self.positionwise_ff = SkipConnection(PositionWiseFeedforward(embed_dim=embed_dim, feedforward_dim=feedforward_dim, dropout=dropout))
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, input, mask1, mask2, mask3):
        h1 = self.self_attention1(input, mask=mask1)
        h2 = self.self_attention2(input, mask=mask2)
        h3 = self.self_attention3(input, mask=mask3)
        hh = torch.cat((h1, h2, h3), dim=2)
        hh = self.tmp_linear_layer(hh)
        hh = self.norm1(hh, mask=mask1)
        hh = self.positionwise_ff(hh, mask=mask1)
        hh = self.norm2(hh, mask=mask1)
        return hh


class GraphTransformerEncoder(nn.Module):

    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=6, n_heads=8, embed_dim=512, feedforward_dim=2048, normalization='batch', dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        self.transformer_layers = nn.ModuleList([MultiGraphTransformerLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout) for _ in range(n_layers)])

    def forward(self, coord, flag, pos, attention_mask1=None, attention_mask2=None, attention_mask3=None):
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag)), dim=2)
        h = torch.cat((h, self.feat_embed(pos)), dim=2)
        for layer in self.transformer_layers:
            h = layer(h, mask1=attention_mask1, mask2=attention_mask2, mask3=attention_mask3)
        return h


class GraphTransformerClassifier(nn.Module):

    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=6, n_heads=8, embed_dim=512, feedforward_dim=2048, normalization='batch', dropout=0.1, mlp_classifier_dropout=0.1):
        super(GraphTransformerClassifier, self).__init__()
        self.encoder = GraphTransformerEncoder(coord_input_dim, feat_input_dim, feat_dict_size, n_layers, n_heads, embed_dim, feedforward_dim, normalization, dropout)
        self.mlp_classifier = nn.Sequential(nn.Dropout(mlp_classifier_dropout), nn.Linear(embed_dim * 3, feedforward_dim, bias=True), nn.ReLU(), nn.Dropout(mlp_classifier_dropout), nn.Linear(feedforward_dim, feedforward_dim, bias=True), nn.ReLU(), nn.Linear(feedforward_dim, n_classes, bias=True))

    def forward(self, coord, flag, pos, attention_mask1=None, attention_mask2=None, attention_mask3=None, padding_mask=None, true_seq_length=None):
        """
        Args:
            coord: Input coordinates (batch_size, seq_length, coord_input_dim)
            # TODO feat: Input features (batch_size, seq_length, feat_input_dim)
            attention_mask: Masks for attention computation (batch_size, seq_length, seq_length)
                            Attention mask should contain -inf if attention is not possible 
                            (i.e. mask is a negative adjacency matrix)
            padding_mask: Mask indicating padded elements in input (batch_size, seq_length)
                          Padding mask element should be 1 if valid element, 0 if padding
                          (i.e. mask is a boolean multiplicative mask)
            true_seq_length: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """
        h = self.encoder(coord, flag, pos, attention_mask1, attention_mask2, attention_mask3)
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim=1)
        else:
            g = h.sum(dim=1)
        logits = self.mlp_classifier(g)
        return logits


class GraphAttentionLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim, normalization='batch', dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.self_attention = SkipConnection(MultiHeadAttention(n_heads=n_heads, input_dim=embed_dim, embed_dim=embed_dim, dropout=dropout))
        self.norm = Normalization(embed_dim, normalization)

    def forward(self, input, mask):
        h = F.relu(self.self_attention(input, mask=mask))
        h = self.norm(h, mask=mask)
        return h


class GraphAttentionEncoder(nn.Module):

    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=3, n_heads=8, embed_dim=256, feedforward_dim=1024, normalization='batch', dropout=0.1):
        super(GraphAttentionEncoder, self).__init__()
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        self.attention_layers = nn.ModuleList([GraphAttentionLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout) for _ in range(n_layers)])

    def forward(self, coord, flag, pos, attention_mask=None):
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag), self.feat_embed(pos)), dim=2)
        for layer in self.attention_layers:
            h = layer(h, mask=attention_mask)
        return h


class GraphAttentionClassifier(nn.Module):

    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=3, n_heads=8, embed_dim=256, feedforward_dim=1024, normalization='batch', dropout=0.1):
        super(GraphAttentionClassifier, self).__init__()
        self.encoder = GraphAttentionEncoder(coord_input_dim, feat_input_dim, feat_dict_size, n_layers, n_heads, embed_dim, feedforward_dim, normalization, dropout)
        self.mlp_classifier = nn.Sequential(nn.Linear(embed_dim * 3, feedforward_dim, bias=True), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feedforward_dim, feedforward_dim, bias=True), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feedforward_dim, n_classes, bias=True))

    def forward(self, coord, flag, pos, attention_mask=None, padding_mask=None, true_seq_length=None):
        """
        Args:
            coord: Input coordinates (batch_size, seq_length, coord_input_dim)
            # TODO feat: Input features (batch_size, seq_length, feat_input_dim)
            attention_mask: Masks for attention computation (batch_size, seq_length, seq_length)
                            Attention mask should contain -inf if attention is not possible 
                            (i.e. mask is a negative adjacency matrix)
            padding_mask: Mask indicating padded elements in input (batch_size, seq_length)
                          Padding mask element should be 1 if valid element, 0 if padding
                          (i.e. mask is a boolean multiplicative mask)
            true_seq_length: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """
        h = self.encoder(coord, flag, pos, attention_mask)
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim=1)
        else:
            g = h.sum(dim=1)
        logits = self.mlp_classifier(g)
        return logits


class GraphMLPLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1, normalization='batch'):
        super(GraphMLPLayer, self).__init__()
        self.sub_layers = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=True), nn.ReLU(), nn.Dropout(dropout))
        self.norm = Normalization(embed_dim, normalization)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.norm(self.sub_layers(input))


class GraphMLPEncoder(nn.Module):

    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=3, embed_dim=256, dropout=0.1):
        super(GraphMLPEncoder, self).__init__()
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        self.mlp_layers = nn.ModuleList([GraphMLPLayer(embed_dim * 3, dropout) for _ in range(n_layers)])
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, coord, flag, pos):
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag), self.feat_embed(pos)), dim=2)
        for layer in self.mlp_layers:
            h = layer(h)
        return h


class GraphMLPClassifier(nn.Module):

    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=3, embed_dim=256, feedforward_dim=1024, dropout=0.1):
        super(GraphMLPClassifier, self).__init__()
        self.encoder = GraphMLPEncoder(coord_input_dim, feat_input_dim, feat_dict_size, n_layers, embed_dim, dropout)
        self.mlp_classifier = nn.Sequential(nn.Linear(embed_dim * 3, feedforward_dim, bias=True), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feedforward_dim, feedforward_dim, bias=True), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feedforward_dim, n_classes, bias=True))

    def forward(self, coord, flag, pos, attention_mask=None, padding_mask=None, true_seq_length=None):
        """
        Args:
            coord: Input coordinates (batch_size, seq_length, coord_input_dim)
            # TODO feat: Input features (batch_size, seq_length, feat_input_dim)
            attention_mask: Masks for attention computation (batch_size, seq_length, seq_length)
                            Attention mask should contain -inf if attention is not possible 
                            (i.e. mask is a negative adjacency matrix)
            padding_mask: Mask indicating padded elements in input (batch_size, seq_length)
                          Padding mask element should be 1 if valid element, 0 if padding
                          (i.e. mask is a boolean multiplicative mask)
            true_seq_length: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """
        h = self.encoder(coord, flag, pos)
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim=1)
        else:
            g = h.sum(dim=1)
        logits = self.mlp_classifier(g)
        return logits


class GraphTransformerLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim, normalization='batch', dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.self_attention = SkipConnection(MultiHeadAttention(n_heads=n_heads, input_dim=embed_dim, embed_dim=embed_dim, dropout=dropout))
        self.norm1 = Normalization(embed_dim, normalization)
        self.positionwise_ff = SkipConnection(PositionWiseFeedforward(embed_dim=embed_dim, feedforward_dim=feedforward_dim, dropout=dropout))
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, input, mask):
        h = self.self_attention(input, mask=mask)
        h = self.norm1(h, mask=mask)
        h = self.positionwise_ff(h, mask=mask)
        h = self.norm2(h, mask=mask)
        return h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GraphAttentionLayer,
     lambda: ([], {'n_heads': 4, 'embed_dim': 4, 'feedforward_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([1, 4, 4, 4])], {}),
     False),
    (GraphMLPLayer,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GraphTransformerLayer,
     lambda: ([], {'n_heads': 4, 'embed_dim': 4, 'feedforward_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([1, 4, 4, 4])], {}),
     False),
    (MultiGraphTransformerLayer,
     lambda: ([], {'n_heads': 4, 'embed_dim': 4, 'feedforward_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([1, 4, 4, 4]), torch.rand([1, 4, 4, 4]), torch.rand([1, 4, 4, 4])], {}),
     False),
    (Normalization,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionWiseFeedforward,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_PengBoXiangShang_multigraph_transformer(_paritybench_base):
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

