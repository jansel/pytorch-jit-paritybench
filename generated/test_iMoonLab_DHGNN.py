import sys
_module = sys.modules[__name__]
del sys
config = _module
datasets = _module
cora = _module
data_factory = _module
models = _module
layers = _module
model_factory = _module
models = _module
train = _module
utils = _module
construct_hypergraph = _module
layer_utils = _module

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


import math


import copy


import torch


import time


from torch import nn


from torch.nn.parameter import Parameter


import pandas as pd


import numpy as np


from sklearn.cluster import KMeans


from sklearn.metrics.pairwise import euclidean_distances


from torch.nn import Module


import random


import torch.optim as optim


import sklearn


from sklearn import neighbors


from sklearn.metrics.pairwise import cosine_distances as cos_dis


class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """

    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()
        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()
        conved = self.convKK(region_feats)
        multiplier = conved.view(N, k, k)
        multiplier = self.activation(multiplier)
        transformed_feats = torch.matmul(multiplier, region_feats)
        return transformed_feats


class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """

    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()
        self.trans = Transform(dim_in, k)
        self.convK1 = nn.Conv1d(k, 1, 1)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats


class GraphConvolution(nn.Module):
    """
    A GCN layer
    """

    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()
        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])
        return pooled_feats

    def forward(self, ids, feats, edge_dict, G, ite):
        """
        :param ids: compatible with `MultiClusterConvolution`
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats
        x = self.dropout(self.activation(self.fc(x)))
        x = self._region_aggregate(x, edge_dict)
        return x


class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """

    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = ft.size(1)
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        return (scores * ft).sum(1)


def sample_ids(ids, k):
    """
    sample `k` indexes from ids, must sample the centroid node itself
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    sampled_ids.append(ids[-1])
    return sampled_ids


def sample_ids_v2(ids, k):
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    return sampled_ids


class DHGLayer(GraphConvolution):
    """
    A Dynamic Hypergraph Convolution Layer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ks = kwargs['structured_neighbor']
        self.n_cluster = kwargs['n_cluster']
        self.n_center = kwargs['n_center']
        self.kn = kwargs['nearest_neighbor']
        self.kc = kwargs['cluster_neighbor']
        self.wu_knn = kwargs['wu_knn']
        self.wu_kmeans = kwargs['wu_kmeans']
        self.wu_struct = kwargs['wu_struct']
        self.vc_sn = VertexConv(self.dim_in, self.ks + self.kn)
        self.vc_s = VertexConv(self.dim_in, self.ks)
        self.vc_n = VertexConv(self.dim_in, self.kn)
        self.vc_c = VertexConv(self.dim_in, self.kc)
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in // 4)
        self.kmeans = None
        self.structure = None

    def _vertex_conv(self, func, x):
        return func(x)

    def _structure_select(self, ids, feats, edge_dict):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :param edge_dict: torch.LongTensor
        :return: mapped graph neighbors
        """
        if self.structure is None:
            _N = feats.size(0)
            idx = torch.LongTensor([sample_ids(edge_dict[i], self.ks) for i in range(_N)])
            self.structure = idx
        else:
            idx = self.structure
        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)
        return region_feats

    def _nearest_select(self, ids, feats):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: mapped nearest neighbors
        """
        dis = cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        idx = idx[ids]
        N = len(idx)
        d = feats.size(1)
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)
        return nearest_feature

    def _cluster_select(self, ids, feats):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        if self.kmeans is None:
            _N = feats.size(0)
            np_feats = feats.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0, n_jobs=-1).fit(np_feats)
            centers = kmeans.cluster_centers_
            dis = euclidean_distances(np_feats, centers)
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
            cluster_center_dict = cluster_center_dict.numpy()
            point_labels = kmeans.labels_
            point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]
            idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc) for i in range(self.n_center)] for point in range(_N)])
            self.kmeans = idx
        else:
            idx = self.kmeans
        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)
        return cluster_feats

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, G, ite):
        hyperedges = []
        if ite >= self.wu_kmeans:
            c_feat = self._cluster_select(ids, feats)
            for c_idx in range(c_feat.size(1)):
                xc = self._vertex_conv(self.vc_c, c_feat[:, c_idx, :, :])
                xc = xc.view(len(ids), 1, feats.size(1))
                hyperedges.append(xc)
        if ite >= self.wu_knn:
            n_feat = self._nearest_select(ids, feats)
            xn = self._vertex_conv(self.vc_n, n_feat)
            xn = xn.view(len(ids), 1, feats.size(1))
            hyperedges.append(xn)
        if ite >= self.wu_struct:
            s_feat = self._structure_select(ids, feats, edge_dict)
            xs = self._vertex_conv(self.vc_s, s_feat)
            xs = xs.view(len(ids), 1, feats.size(1))
            hyperedges.append(xs)
        x = torch.cat(hyperedges, dim=1)
        x = self._edge_conv(x)
        x = self._fc(x)
        return x


class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """

    def __init__(self, **kwargs):
        super(HGNN_conv, self).__init__()
        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def forward(self, ids, feats, edge_dict, G, ite):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x


class DHGNN_v1(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a GCN-style input layer
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(dim_in=self.dims_in[0], dim_out=self.dims_out[0], dropout_rate=kwargs['dropout_rate'], activation=activations[0], has_bias=kwargs['has_bias'])] + [DHGLayer(dim_in=self.dims_in[i], dim_out=self.dims_out[i], dropout_rate=kwargs['dropout_rate'], activation=activations[i], structured_neighbor=kwargs['k_structured'], nearest_neighbor=kwargs['k_nearest'], cluster_neighbor=kwargs['k_cluster'], wu_knn=kwargs['wu_knn'], wu_kmeans=kwargs['wu_kmeans'], wu_struct=kwargs['wu_struct'], n_cluster=kwargs['clusters'], n_center=kwargs['adjacent_centers'], has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        G = kwargs['G']
        ite = kwargs['ite']
        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](ids, x, edge_dict, G, ite)
        return x


class DHGNN_v2(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a HGNN-style input layer
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([HGNN_conv(dim_in=self.dims_in[0], dim_out=self.dims_out[0], dropout_rate=kwargs['dropout_rate'], activation=activations[0], has_bias=kwargs['has_bias'])] + [DHGLayer(dim_in=self.dims_in[i], dim_out=self.dims_out[i], dropout_rate=kwargs['dropout_rate'], activation=activations[i], structured_neighbor=kwargs['k_structured'], nearest_neighbor=kwargs['k_nearest'], cluster_neighbor=kwargs['k_cluster'], wu_knn=kwargs['wu_knn'], wu_kmeans=kwargs['wu_kmeans'], wu_struct=kwargs['wu_struct'], n_cluster=kwargs['clusters'], n_center=kwargs['adjacent_centers'], has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        G = kwargs['G']
        ite = kwargs['ite']
        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](ids, x, edge_dict, G, ite)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EdgeConv,
     lambda: ([], {'dim_ft': 4, 'hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transform,
     lambda: ([], {'dim_in': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (VertexConv,
     lambda: ([], {'dim_in': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_iMoonLab_DHGNN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

