import sys
_module = sys.modules[__name__]
del sys
data = _module
batch = _module
data = _module
dataloader = _module
dataset = _module
download = _module
extract = _module
in_memory_dataset = _module
makedirs = _module
datasets = _module
edgelist_label = _module
gatne = _module
matlab_matrix = _module
pyg = _module
layers = _module
maggregator = _module
se_layer = _module
models = _module
base_model = _module
emb = _module
deepwalk = _module
dngr = _module
gatne = _module
graph2vec = _module
grarep = _module
hope = _module
line = _module
netmf = _module
netsmf = _module
node2vec = _module
prone = _module
sdne = _module
spectral = _module
nn = _module
gat = _module
gcn = _module
gin = _module
graphsage = _module
infograph = _module
mlp = _module
pyg_cheb = _module
pyg_drgat = _module
pyg_drgcn = _module
pyg_gat = _module
pyg_gcn = _module
pyg_infomax = _module
pyg_unet = _module
options = _module
tasks = _module
base_task = _module
community_detection = _module
graph_classification = _module
influence_maximization = _module
link_prediction = _module
multiplex_link_prediction = _module
node_classification = _module
unsupervised_graph_classification = _module
unsupervised_node_classification = _module
conf = _module
display_data = _module
parallel_train = _module
train = _module
setup = _module
test_pyg = _module

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


import re


import torch


import torch.utils.data


from torch.utils.data.dataloader import default_collate


import collections


from itertools import repeat


from itertools import product


import numpy as np


import scipy.io


import torch.nn as nn


import time


import scipy.sparse as sp


from sklearn import preprocessing


import torch.nn.functional as F


from collections import defaultdict


import random


import math


from torch.nn.parameter import Parameter


from sklearn.linear_model import LogisticRegression


import copy


import warnings


from sklearn.cluster import KMeans


from sklearn.metrics import silhouette_score


from sklearn.metrics import normalized_mutual_info_score


from sklearn.metrics import auc


from sklearn.metrics import f1_score


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import roc_auc_score


from sklearn.utils import shuffle as skshuffle


from sklearn.svm import SVC


from sklearn.metrics import accuracy_score


from scipy import sparse as sp


from sklearn.multiclass import OneVsRestClassifier


import itertools


from collections import namedtuple


import torch.multiprocessing as mp


class MeanAggregator(torch.nn.Module):

    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True):
        super(MeanAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.linear = nn.Linear(in_channels, out_channels, bias)

    @staticmethod
    def norm(x, edge_index):
        deg = torch.sparse.sum(edge_index, 1)
        deg_inv = deg.pow(-1).to_dense()
        x = torch.matmul(edge_index, x)
        x = x.t() * deg_inv
        return x.t()

    def forward(self, x, edge_index, edge_weight=None, bias=True):
        """"""
        x = self.linear(x)
        x = self.norm(x, edge_index)
        return x

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.se_channels = se_channels
        self.encoder_decoder = nn.Sequential(nn.Linear(in_channels, se_channels), nn.ELU(), nn.Linear(se_channels, in_channels), nn.Sigmoid())
        self.reset_parameters()

    def forward(self, x):
        """"""
        x_global = torch.mean(x, dim=0)
        s = self.encoder_decoder(x_global)
        return x * s


class BaseModel(nn.Module):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError('Models must implement the build_model_from_args method')


MODEL_REGISTRY = {}


def register_model(name):
    """
    New model types can be added to cogdl with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError('Model ({}: {}) must extend BaseModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls


class DNGR(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--hidden-size1', type=int, default=1000, help='Hidden size in first layer of Auto-Encoder')
        parser.add_argument('--hidden-size2', type=int, default=128, help='Hidden size in second layer of Auto-Encoder')
        parser.add_argument('--noise', type=float, default=0.2, help='denoise rate of DAE')
        parser.add_argument('--alpha', type=float, default=0.98, help='alhpa is a hyperparameter in DNGR')
        parser.add_argument('--step', type=int, default=10, help='step is a hyperparameter in DNGR')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size1, args.hidden_size2, args.noise, args.alpha, args.step, args.max_epoch, args.lr)

    def __init__(self, hidden_size1, hidden_size2, noise, alpha, step, max_epoch, lr):
        super(DNGR, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.noise = noise
        self.alpha = alpha
        self.step = step
        self.max_epoch = max_epoch
        self.lr = lr

    def build_nn(self):
        self.encoder = nn.Sequential(nn.Linear(self.num_node, self.hidden_size1), nn.Tanh(), nn.Linear(self.hidden_size1, self.hidden_size2), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(self.hidden_size2, self.hidden_size1), nn.Tanh(), nn.Linear(self.hidden_size1, self.num_node), nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def scale_matrix(self, mat):
        mat = mat - np.diag(np.diag(mat))
        D_inv = np.diagflat(np.reciprocal(np.sum(mat, axis=0)))
        mat = np.dot(D_inv, mat)
        return mat

    def random_surfing(self, adj_matrix):
        adj_matrix = self.scale_matrix(adj_matrix)
        P0 = np.eye(self.num_node, dtype='float32')
        M = np.zeros((self.num_node, self.num_node), dtype='float32')
        P = np.eye(self.num_node, dtype='float32')
        for i in range(0, self.step):
            P = self.alpha * np.dot(P, adj_matrix) + (1 - self.alpha) * P0
            M = M + P
        return M

    def get_ppmi_matrix(self, mat):
        mat = self.random_surfing(mat)
        M = self.scale_matrix(mat)
        col_s = np.sum(M, axis=0).reshape(1, self.num_node)
        row_s = np.sum(M, axis=1).reshape(self.num_node, 1)
        D = np.sum(col_s)
        rowcol_s = np.dot(row_s, col_s)
        PPMI = np.log(np.divide(D * M, rowcol_s))
        PPMI[np.isnan(PPMI)] = 0.0
        PPMI[np.isinf(PPMI)] = 0.0
        PPMI[np.isneginf(PPMI)] = 0.0
        PPMI[PPMI < 0] = 0.0
        return PPMI

    def get_denoised_matrix(self, mat):
        return mat * (np.random.random(mat.shape) > self.noise)

    def get_emb(self, matrix):
        ut, s, _ = sp.linalg.svds(matrix, self.hidden_size2)
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, 'l2')
        return emb_matrix

    def train(self, G):
        self.num_node = G.number_of_nodes()
        A = nx.adjacency_matrix(G).todense()
        PPMI = self.get_ppmi_matrix(A)
        None
        input_mat = torch.from_numpy(self.get_denoised_matrix(PPMI).astype(np.float32))
        self.build_nn()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            opt.zero_grad()
            encoded, decoded = self.forward(input_mat)
            Loss = loss_func(decoded, input_mat)
            Loss.backward()
            epoch_iter.set_description(f'Epoch: {epoch:03d},  Loss: {Loss:.8f}')
            opt.step()
        embedding, _ = self.forward(input_mat)
        return embedding.detach().numpy()


class GATNEModel(nn.Module):

    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a
        self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.node_type_embeddings = Parameter(torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size))
        self.trans_weights = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.node_embeddings.data.uniform_(-1.0, 1.0)
        self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        node_embed = self.node_embeddings[train_inputs]
        node_embed_neighbors = self.node_type_embeddings[node_neigh]
        node_embed_tmp = torch.cat([node_embed_neighbors[:, (i), :, (i), :].unsqueeze(1) for i in range(self.edge_type_count)], dim=1)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)
        trans_w = self.trans_weights[train_types]
        trans_w_s1 = self.trans_weights_s1[train_types]
        trans_w_s2 = self.trans_weights_s2[train_types]
        attention = F.softmax(torch.matmul(F.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2).squeeze()).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze()
        last_node_embed = F.normalize(node_embed, dim=1)
        return last_node_embed


class NSLoss(nn.Module):

    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(torch.Tensor([((math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)) for k in range(num_nodes)]), dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1)))
        negs = torch.multinomial(self.sample_weights, self.num_sampled * n, replacement=True).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()
        loss = log_target + sum_log_sampled
        return -loss.sum() / n


def generate_pairs(all_walks, vocab, window_size=5):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)
    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)
    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i
    return vocab, index2word


class RWGraph:

    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        G = self.G
        rand = random.Random()
        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]
        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():
                if schema == None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, schema=None):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if schema is not None:
            schema_list = schema.split(',')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split('-')[0] == self.node_type[node]:
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter))
        return walks


def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = int(edge_key.split('_')[0])
        y = int(edge_key.split('_')[1])
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G


def generate_walks(network_data, num_walks, walk_length, schema=None):
    if schema is not None:
        pass
    else:
        node_type = None
    all_walks = []
    for layer_id in network_data:
        tmp_data = network_data[layer_id]
        layer_walker = RWGraph(get_G_from_edges(tmp_data))
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)
        all_walks.append(layer_walks)
    return all_walks


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size
    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


class GATNE(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--walk-length', type=int, default=10, help='Length of walk per source. Default is 10.')
        parser.add_argument('--walk-num', type=int, default=20, help='Number of walks per source. Default is 20.')
        parser.add_argument('--window-size', type=int, default=5, help='Window size of skip-gram model. Default is 5.')
        parser.add_argument('--worker', type=int, default=10, help='Number of parallel workers. Default is 10.')
        parser.add_argument('--iteration', type=int, default=10, help='Number of iterations. Default is 10.')
        parser.add_argument('--epoch', type=int, default=20, help='Number of epoch. Default is 20.')
        parser.add_argument('--batch-size', type=int, default=64, help='Number of batch_size. Default is 64.')
        parser.add_argument('--edge-dim', type=int, default=10, help='Number of edge embedding dimensions. Default is 10.')
        parser.add_argument('--att-dim', type=int, default=20, help='Number of attention dimensions. Default is 20.')
        parser.add_argument('--negative-samples', type=int, default=5, help='Negative samples for optimization. Default is 5.')
        parser.add_argument('--neighbor-samples', type=int, default=10, help='Neighbor samples for aggregation. Default is 10.')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.walk_length, args.walk_num, args.window_size, args.worker, args.iteration, args.epoch, args.batch_size, args.edge_dim, args.att_dim, args.negative_samples, args.neighbor_samples)

    def __init__(self, dimension, walk_length, walk_num, window_size, worker, iteration, epoch, batch_size, edge_dim, att_dim, negative_samples, neighbor_samples):
        super(GATNE, self).__init__()
        self.embedding_size = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.epochs = epoch
        self.batch_size = batch_size
        self.embedding_u_size = edge_dim
        self.dim_att = att_dim
        self.num_sampled = negative_samples
        self.neighbor_samples = neighbor_samples
        self.multiplicity = True
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self, network_data):
        all_walks = generate_walks(network_data, self.walk_num, self.walk_length)
        vocab, index2word = generate_vocab(all_walks)
        train_pairs = generate_pairs(all_walks, vocab)
        edge_types = list(network_data.keys())
        num_nodes = len(index2word)
        edge_type_count = len(edge_types)
        epochs = self.epochs
        batch_size = self.batch_size
        embedding_size = self.embedding_size
        embedding_u_size = self.embedding_u_size
        num_sampled = self.num_sampled
        dim_att = self.dim_att
        neighbor_samples = self.neighbor_samples
        neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
        for r in range(edge_type_count):
            g = network_data[edge_types[r]]
            for x, y in g:
                ix = vocab[x].index
                iy = vocab[y].index
                neighbors[ix][r].append(iy)
                neighbors[iy][r].append(ix)
            for i in range(num_nodes):
                if len(neighbors[i][r]) == 0:
                    neighbors[i][r] = [i] * neighbor_samples
                elif len(neighbors[i][r]) < neighbor_samples:
                    neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]))))
                elif len(neighbors[i][r]) > neighbor_samples:
                    neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
        model = GATNEModel(num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_att)
        nsloss = NSLoss(num_nodes, num_sampled, embedding_size)
        model
        nsloss
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': nsloss.parameters()}], lr=0.0001)
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)
            data_iter = tqdm.tqdm(batches, desc='epoch %d' % epoch, total=(len(train_pairs) + (batch_size - 1)) // batch_size, bar_format='{l_bar}{r_bar}')
            avg_loss = 0.0
            for i, data in enumerate(data_iter):
                optimizer.zero_grad()
                embs = model(data[0], data[2], data[3])
                loss = nsloss(data[0], embs, data[1])
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                if i % 5000 == 0:
                    post_fix = {'epoch': epoch, 'iter': i, 'avg_loss': avg_loss / (i + 1), 'loss': loss.item()}
                    data_iter.write(str(post_fix))
        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)])
            train_types = torch.tensor(list(range(edge_type_count)))
            node_neigh = torch.tensor([neighbors[i] for _ in range(edge_type_count)])
            node_emb = model(train_inputs, train_types, node_neigh)
            for j in range(edge_type_count):
                final_model[edge_types[j]][index2word[i]] = node_emb[j].cpu().detach().numpy()
        return final_model


class SDNE(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--hidden-size1', type=int, default=1000, help='Hidden size in first layer of Auto-Encoder')
        parser.add_argument('--hidden-size2', type=int, default=128, help='Hidden size in second layer of Auto-Encoder')
        parser.add_argument('--droput', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--alpha', type=float, default=0.1, help='alhpa is a hyperparameter in SDNE')
        parser.add_argument('--beta', type=float, default=5, help='beta is a hyperparameter in SDNE')
        parser.add_argument('--nu1', type=float, default=0.0001, help='nu1 is a hyperparameter in SDNE')
        parser.add_argument('--nu2', type=float, default=0.001, help='nu2 is a hyperparameter in SDNE')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size1, args.hidden_size2, args.droput, args.alpha, args.beta, args.nu1, args.nu2, args.max_epoch, args.lr)

    def __init__(self, hidden_size1, hidden_size2, droput, alpha, beta, nu1, nu2, max_epoch, lr):
        super(SDNE, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.droput = droput
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.max_epoch = max_epoch
        self.lr = lr

    def build_nn(self):
        self.encode0 = nn.Linear(self.num_node, self.hidden_size1)
        self.encode1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.decode0 = nn.Linear(self.hidden_size2, self.hidden_size1)
        self.decode1 = nn.Linear(self.hidden_size1, self.num_node)

    def forward(self, adj_mat, l_mat):
        t0 = F.leaky_relu(self.encode0(adj_mat))
        t0 = F.leaky_relu(self.encode1(t0))
        self.embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        L_1st = 2 * torch.trace(torch.mm(torch.mm(torch.t(self.embedding), l_mat), self.embedding))
        L_2nd = torch.sum((adj_mat - t0) * adj_mat * self.beta * ((adj_mat - t0) * adj_mat * self.beta))
        return self.alpha * L_1st, L_2nd, self.alpha * L_1st + L_2nd

    def get_emb(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0

    def train(self, G):
        self.num_node = G.number_of_nodes()
        self = self
        A = torch.from_numpy(nx.adjacency_matrix(G).todense().astype(np.float32))
        L = torch.from_numpy(nx.laplacian_matrix(G).todense().astype(np.float32))
        self.build_nn()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            opt.zero_grad()
            L_1st, L_2nd, L_all = self.forward(A, L)
            L_reg = 0
            for param in self.parameters():
                L_reg += self.nu1 * torch.sum(torch.abs(param)) + self.nu2 * torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            epoch_iter.set_description(f'Epoch: {epoch:03d}, L_1st: {L_1st:.4f}, L_2nd: {L_2nd:.4f}, L_reg: {L_reg:.4f}')
            opt.step()
        embedding = self.get_emb(A)
        return embedding.detach().numpy()


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[(0), :] * ctx.N + a._indices()[(1), :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):

    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, edge):
        N = input.size()[0]
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[(edge[(0), :]), :], h[(edge[(1), :]), :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class PetarVGAT(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.6)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--nheads', type=int, default=8)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout, args.alpha, args.nheads)

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(PetarVGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class PetarVSpGAT(PetarVGAT):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        BaseModel.__init__(self)
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, edge_index):
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).float(), (input.shape[0], input.shape[0]))
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class TKipfGCN(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--dropout', type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout)

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(TKipfGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=-1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class GINLayer(nn.Module):

    def __init__(self, apply_func=None, eps=0, train_eps=True):
        super(GINLayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = remove_self_loops(edge_index)
        edge_weight = torch.ones(edge_index.shape[1]) if edge_weight is None else edge_weight
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (x.shape[0], x.shape[0]))
        adj = adj
        out = (1 + self.eps) * x + torch.spmm(adj, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


class GINMLP(nn.Module):

    def __init__(self, in_feats, out_feats, hidden_dim, num_layers, use_bn=True, activation=None):
        super(GINMLP, self).__init__()
        self.use_bn = use_bn
        self.nn = nn.ModuleList()
        if use_bn:
            self.bn = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError('number of MLP layers should be positive')
        elif num_layers == 1:
            self.nn.append(nn.Linear(in_feats, out_feats))
        else:
            for i in range(num_layers - 1):
                if i == 0:
                    self.nn.append(nn.Linear(in_feats, hidden_dim))
                else:
                    self.nn.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    self.bn.append(nn.BatchNorm1d(hidden_dim))
            self.nn.append(nn.Linear(hidden_dim, out_feats))

    def forward(self, x):
        h = x
        for i in range(self.num_layers - 1):
            h = self.nn[i](h)
            if self.use_bn:
                h = self.bn[i](h)
            h = F.relu(h)
        return self.nn[self.num_layers - 1](h)


class Data(object):
    """A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos

    @staticmethod
    def from_dict(dictionary):
        """Creates a data object from a python dictionary."""
        data = Data()
        for key, item in dictionary.items():
            data[key] = item
        return data

    def __getitem__(self, key):
        """Gets the data of the attribute :obj:`key`."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        """Returns all names of graph attributes."""
        return [key for key in self.__dict__.keys() if self[key] is not None]

    def __len__(self):
        """Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        """Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        """Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        """Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in (sorted(self.keys) if not keys else keys):
            if self[key] is not None:
                yield key, self[key]

    def cat_dim(self, key, value):
        """Returns the dimension in which the attribute :obj:`key` with
        content :obj:`value` gets concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return -1 if bool(re.search('(index|face)', key)) else 0

    @property
    def num_edges(self):
        """Returns the number of edges in the graph."""
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.cat_dim(key, item))
        return None

    @property
    def num_features(self):
        """Returns the number of features per node in the graph."""
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_nodes(self):
        if self.x is not None:
            return self.x.shape[0]
        return torch.max(self.edge_index) + 1

    def is_coalesced(self):
        """Returns :obj:`True`, if edge indices are ordered and do not contain
        duplicate entries."""
        row, col = self.edge_index
        index = self.num_nodes * row + col
        return row.size(0) == torch.unique(index).size(0)

    def apply(self, func, *keys):
        """Applies the function :obj:`func` to all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, :obj:`func` is applied to all present
        attributes.
        """
        for key, item in self(*keys):
            self[key] = func(item)
        return self

    def contiguous(self, *keys):
        """Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        """Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x, *keys)

    def cuda(self, *keys):
        return self.apply(lambda x: x, *keys)

    def clone(self):
        return Data.from_dict({k: v.clone() for k, v in self})

    def __repr__(self):
        info = ['{}={}'.format(key, list(item.size())) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))


class Batch(Data):
    """A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`cogdl.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        """Constructs a batch object from a python list holding
        :class:`cogdl.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys
        batch = Batch()
        for key in keys:
            batch[key] = []
        batch.batch = []
        cumsum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                item = item + cumsum if batch.cumsum(key, item) else item
                batch[key].append(item)
            cumsum += num_nodes
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        """If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return bool(re.search('(index|face)', key))

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoader(torch.utils.data.DataLoader):
    """Data loader which merges data objects from a
    :class:`cogdl.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=lambda data_list: Batch.from_data_list(data_list), **kwargs)


class GIN(BaseModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--epsilon', type=float, default=0.0)
        parser.add_argument('--hidden-size', type=int, default=32)
        parser.add_argument('--num-layers', type=int, default=5)
        parser.add_argument('--num-mlp-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--train-epsilon', type=bool, default=True)
        parser.add_argument('--pooling', type=str, default='sum')
        parser.add_argument('--batch-size', type=int, default=128)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_layers, args.num_features, args.num_classes, args.hidden_size, args.num_mlp_layers, args.epsilon, args.pooling, args.train_epsilon, args.dropout)

    @classmethod
    def split_dataset(cls, dataset, args):
        test_index = random.sample(range(len(dataset)), len(dataset) // 10)
        train_index = [x for x in range(len(dataset)) if x not in test_index]
        train_dataset = [dataset[i] for i in train_index]
        test_dataset = [dataset[i] for i in test_index]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        return train_loader, test_loader, test_loader

    def __init__(self, num_layers, in_feats, out_feats, hidden_dim, num_mlp_layers, eps=0, pooling='sum', train_eps=False, dropout=0.5):
        super(GIN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            if i == 0:
                mlp = GINMLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers)
            else:
                mlp = GINMLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers)
            self.gin_layers.append(GINLayer(mlp, eps, train_eps))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim))
        self.linear_prediction = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_feats))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_feats))
        self.dropout = nn.Dropout(dropout)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch, edge_weight=None, label=None):
        h = x
        layer_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.gin_layers[i](h, edge_index)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            layer_rep.append(h)
        final_score = 0
        for i in range(self.num_layers):
            pooled = scatter_add(layer_rep[i], batch, dim=0)
            final_score += self.dropout(self.linear_prediction[i](pooled))
        final_score = F.softmax(final_score, dim=-1)
        if label is not None:
            loss = self.loss(final_score, label)
            return final_score, loss
        return final_score, None

    def loss(self, output, label=None):
        return self.criterion(output, label)


class Graphsage(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, nargs='+', default=[128])
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--sample-size', type=int, nargs='+', default=[10, 10])
        parser.add_argument('--dropout', type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_layers, args.sample_size, args.dropout)

    def sampler(self, edge_index, num_sample):
        if self.adjlist == {}:
            edge_index = edge_index.t().cpu().tolist()
            for i in edge_index:
                if not i[0] in self.adjlist:
                    self.adjlist[i[0]] = [i[1]]
                else:
                    self.adjlist[i[0]].append(i[1])
        sample_list = []
        for i in self.adjlist:
            list = [[i, j] for j in self.adjlist[i]]
            if len(list) > num_sample:
                list = random.sample(list, num_sample)
            sample_list.extend(list)
        edge_idx = torch.LongTensor(sample_list).t()
        return edge_idx

    def __init__(self, num_features, num_classes, hidden_size, num_layers, sample_size, dropout):
        super(Graphsage, self).__init__()
        self.adjlist = {}
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        shapes = [num_features] + hidden_size + [num_classes]
        None
        self.convs = nn.ModuleList([MeanAggregator(shapes[layer], shapes[layer + 1], cached=True) for layer in range(num_layers)])

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            edge_index_sp = self.sampler(edge_index, self.sample_size[i])
            adj_sp = torch.sparse_coo_tensor(edge_index_sp, torch.ones(edge_index_sp.shape[1]).float(), (x.shape[0], x.shape[0]))
            x = self.convs[i](x, adj_sp)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class SUPEncoder(torch.nn.Module):

    def __init__(self, num_features, dim, num_layers=1):
        super(SUPEncoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)
        nnu = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nnu, aggr='mean', root_weight=False)
        self.gru = nn.GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data, **kwargs):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map.append(out)
        out = self.set2set(out, data.batch)
        return out, feat_map[-1]


class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


class FF(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(FF, self).__init__()
        self.block = GINMLP(in_feats, out_feats, out_feats, num_layers=3, use_bn=False)
        self.shortcut = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        return F.relu(self.block(x)) + self.shortcut(x)


class InfoGraph(BaseModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--batch-size', type=int, default=20)
        parser.add_argument('--target', dest='target', type=int, default=0, help='')
        parser.add_argument('--train-num', dest='train_num', type=int, default=5000)
        parser.add_argument('--num-layers', type=int, default=3)
        parser.add_argument('--use-unsupervised', dest='use_unsup', action='store_true')
        parser.add_argument('--epochs', type=int, default=40)
        parser.add_argument('--nn', type=bool, default=True)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.num_layers, args.use_unsup)

    @classmethod
    def split_dataset(cls, dataset, args):
        if args.dataset == 'QM9':
            test_dataset = dataset[:10000]
            val_dataset = dataset[10000:20000]
            train_dataset = dataset[20000:20000 + args.train_num]
            return DataLoader(train_dataset, batch_size=args.batch_size), DataLoader(val_dataset, batch_size=args.batch_size), DataLoader(test_dataset, batch_size=args.batch_size)
        else:
            test_index = random.sample(range(len(dataset)), len(dataset) // 10)
            train_index = [x for x in range(len(dataset)) if x not in test_index]
            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            return train_loader, test_loader, test_loader

    def __init__(self, in_feats, hidden_dim, out_feats, num_layers=3, unsup=True):
        super(InfoGraph, self).__init__()
        self.unsup = unsup
        self.emb_dim = hidden_dim
        self.out_feats = out_feats
        self.sem_fc1 = nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.sem_fc2 = nn.Linear(hidden_dim, out_feats)
        if unsup:
            self.unsup_encoder = Encoder(in_feats, hidden_dim, num_layers)
            self.register_parameter('sem_encoder', None)
        else:
            self.unsup_encoder = SUPEncoder(in_feats, hidden_dim, num_layers)
            self.sem_encoder = Encoder(in_feats, hidden_dim, num_layers)
        self._fc1 = FF(num_layers * hidden_dim, hidden_dim)
        self._fc2 = FF(num_layers * hidden_dim, hidden_dim)
        self.local_dis = FF(num_layers * hidden_dim, hidden_dim)
        self.global_dis = FF(num_layers * hidden_dim, hidden_dim)
        self.criterion = nn.MSELoss()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x, edge_index=None, batch=None, label=None):
        if self.unsup:
            return self.unsup_forward(x, edge_index, batch)
        else:
            return self.sup_forward(x, edge_index, batch, label)

    def sup_forward(self, x, edge_index=None, batch=None, label=None):
        node_feat, graph_feat = self.sem_encoder(x, edge_index, batch)
        node_feat = F.relu(self.sem_fc1(node_feat))
        node_feat = self.sem_fc2(node_feat)
        prediction = F.softmax(node_feat, dim=1)
        if label is not None:
            loss = self.sup_loss(prediction, label)
            loss += self.unsup_loss(x, edge_index, batch)
            loss += self.unsup_sup_loss(x, edge_index, batch)
            return prediction, loss
        return prediction, None

    def unsup_forward(self, x, edge_index=None, batch=None):
        return self.unsup_loss(x, edge_index, batch)

    def sup_loss(self, prediction, label=None):
        sup_loss = self.criterion(prediction, label)
        return sup_loss

    def unsup_loss(self, x, edge_index=None, batch=None):
        graph_feat, node_feat = self.unsup_encoder(x, edge_index, batch)
        local_encode = self.local_dis(node_feat)
        global_encode = self.global_dis(graph_feat)
        num_graphs = graph_feat.shape[0]
        num_nodes = node_feat.shape[0]
        pos_mask = torch.zeros((num_nodes, num_graphs))
        neg_mask = torch.ones((num_nodes, num_graphs))
        for nid, gid in enumerate(batch):
            pos_mask[nid][gid] = 1
            neg_mask[nid][gid] = 0
        glob_local_mi = torch.mm(local_encode, global_encode.t())
        loss = InfoGraph.mi_loss(pos_mask, neg_mask, glob_local_mi, num_nodes, num_nodes * (num_graphs - 1))
        return graph_feat, loss

    def unsup_sup_loss(self, x, edge_index, batch):
        sem_g_feat, _ = self.sem_encoder(x, edge_index, batch)
        un_g_feat, _ = self.unsup_encoder(x, edge_index, batch)
        sem_encode = self._fc1(sem_g_feat)
        un_encode = self._fc2(un_g_feat)
        num_graphs = sem_encode.shape[1]
        pos_mask = torch.eye(num_graphs)
        neg_mask = 1 - pos_mask
        mi = torch.mm(sem_encode, un_encode.t())
        loss = InfoGraph.mi_loss(pos_mask, neg_mask, mi, pos_mask.sum(), neg_mask.sum())
        return loss

    @staticmethod
    def mi_loss(pos_mask, neg_mask, mi, pos_div, neg_div):
        pos_mi = pos_mask * mi
        neg_mi = neg_mask * mi
        pos_loss = F.softplus(-pos_mi).sum()
        neg_loss = F.softplus(neg_mi).sum()
        return pos_loss / pos_div + neg_loss / neg_div


class MLP(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=16)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_layers, args.dropout)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(MLP, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.mlp = nn.ModuleList([nn.Linear(shapes[layer], shapes[layer + 1]) for layer in range(num_layers)])

    def forward(self, x, edge_index):
        for fc in self.mlp[:-1]:
            x = F.relu(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class Chebyshev(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--filter-size', type=int, default=5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_layers, args.dropout, args.filter_size)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout, filter_size):
        super(Chebyshev, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.filter_size = filter_size
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList([ChebConv(shapes[layer], shapes[layer + 1], filter_size) for layer in range(num_layers)])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class DrGAT(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument('--num-heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.6)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_heads, args.dropout)

    def __init__(self, num_features, num_classes, hidden_size, num_heads, dropout):
        super(DrGAT, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv1 = GATConv(num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, num_classes, dropout=dropout)
        self.se1 = SELayer(num_features, se_channels=int(np.sqrt(num_features)))
        self.se2 = SELayer(hidden_size * num_heads, se_channels=int(np.sqrt(hidden_size * num_heads)))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.se1(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.se2(x)
        x = F.elu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class DrGCN(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=16)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_layers, args.dropout)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(DrGCN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList([GCNConv(shapes[layer], shapes[layer + 1], cached=True) for layer in range(num_layers)])
        self.ses = nn.ModuleList([SELayer(shapes[layer], se_channels=int(np.sqrt(shapes[layer]))) for layer in range(num_layers)])

    def forward(self, x, edge_index):
        x = self.ses[0](x)
        for se, conv in zip(self.ses[1:], self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = se(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class GAT(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument('--num-heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.6)
        parser.add_argument('--lr', type=float, default=0.005)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_heads, args.dropout)

    def __init__(self, num_features, num_classes, hidden_size, num_heads, dropout):
        super(GAT, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv1 = GATConv(num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, num_classes, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


class GCN(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_layers, args.dropout)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList([GCNConv(shapes[layer], shapes[layer + 1], cached=True) for layer in range(num_layers)])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class Infomax(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=512)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size)

    def __init__(self, num_features, num_classes, hidden_size):
        super(Infomax, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.model = DeepGraphInfomax(hidden_channels=hidden_size, encoder=Encoder(num_features, hidden_size), summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)), corruption=corruption)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def loss(self, data):
        pos_z, neg_z, summary = self.forward(data.x, data.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)
        return loss

    def predict(self, data):
        z, _, _ = self.forward(data.x, data.edge_index)
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150)
        clf.fit(z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
        logits = torch.Tensor(clf.predict_proba(z.detach().cpu().numpy()))
        if z.is_cuda:
            logits = logits
        return logits


class UNet(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.num_layers, args.dropout)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(UNet, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.unet = GraphUNet(num_features, hidden_size, num_classes, depth=3, pool_ratios=[0.5, 0.5])

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=0.2, force_undirected=True, num_nodes=x.shape[0], training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(self.forward(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FF,
     lambda: ([], {'in_feats': 4, 'out_feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GATNEModel,
     lambda: ([], {'num_nodes': 4, 'embedding_size': 4, 'embedding_u_size': 4, 'edge_type_count': 4, 'dim_a': 4}),
     lambda: ([torch.zeros([4, 4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64), torch.zeros([4, 4, 4], dtype=torch.int64)], {}),
     True),
    (GINMLP,
     lambda: ([], {'in_feats': 4, 'out_feats': 4, 'hidden_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GraphAttentionLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'dropout': 0.5, 'alpha': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'num_features': 4, 'num_classes': 4, 'hidden_size': 4, 'num_layers': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PetarVGAT,
     lambda: ([], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5, 'alpha': 4, 'nheads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_THUDM_cogdl(_paritybench_base):
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

