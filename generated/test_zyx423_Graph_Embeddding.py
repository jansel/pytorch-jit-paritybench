import sys
_module = sys.modules[__name__]
del sys
model_AE = _module
my_AE = _module
my_SAE = _module
my_VAE = _module
data_process = _module
data_process_pubmed = _module
model_GAE = _module
my_GAE = _module
my_VGAE = _module
model_LGAE = _module
my_LGAE = _module
my_LVGAE = _module
model_SDNE = _module
my_SDNE = _module
data_process = _module
graph_pre = _module
graph_pre_pubmed = _module
link_prediction = _module
metrics = _module
my_SVM_test = _module

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


from sklearn.cluster import KMeans


import numpy as np


import time


import warnings


import scipy.sparse as sp


import scipy.io as scio


from sklearn.datasets import make_moons


import random


import matplotlib.pyplot as plt


from sklearn.manifold import TSNE


from sklearn import svm


from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score


class MyAE(torch.nn.Module):

    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(MyAE, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1), torch.nn.ReLU(inplace=True))
        self.conv2 = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.conv3 = torch.nn.Sequential(torch.nn.Linear(d_2, d_3), torch.nn.ReLU(inplace=True))
        self.conv4 = torch.nn.Sequential(torch.nn.Linear(d_3, d_4), torch.nn.Tanh())

    def encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2 = self.conv2(H_1)
        return H_2

    def decoder(self, H_2):
        H_3 = self.conv3(H_2)
        H_4 = self.conv4(H_3)
        return H_4

    def forward(self, H_0):
        Latent_Representation = self.encoder(H_0)
        Features_Reconstrction = self.decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstrction


class MyVAE(torch.nn.Module):

    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(MyVAE, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1), torch.nn.ReLU(inplace=True))
        self.conv2_mean = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.conv2_std = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.conv3 = torch.nn.Sequential(torch.nn.Linear(d_2, d_3), torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(torch.nn.Linear(d_3, d_4), torch.nn.Tanh())

    def encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2_mean = self.conv2_mean(H_1)
        H_2_std = self.conv2_std(H_1)
        return H_2_mean, H_2_std

    def reparametrization(self, H_2_mean, H_2_std):
        eps = torch.rand_like(H_2_std)
        std = torch.exp(H_2_std)
        latent_representation = eps * std + H_2_mean
        return latent_representation

    def decoder(self, Latent_Representation):
        H_3 = self.conv3(Latent_Representation)
        Features_Reconstruction = self.conv4(H_3)
        return Features_Reconstruction

    def forward(self, H_0):
        H_2_mean, H_2_std = self.encoder(H_0)
        Latent_Representation = self.reparametrization(H_2_mean, H_2_std)
        Features_Reconstruction = self.decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstruction, H_2_mean, H_2_std


class MySAE(torch.nn.Module):

    def __init__(self, input_dim, middle_dim, output_dim, bias=False):
        super(MySAE, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(input_dim, middle_dim), torch.nn.ReLU(inplace=True))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(middle_dim, output_dim), torch.nn.ReLU(inplace=True))

    def forward(self, H_0):
        latent_representation = self.encoder(H_0)
        features_reconstrction = self.decoder(latent_representation)
        return latent_representation, features_reconstrction


def l2_distance(A, B):
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True)
    AB = A.dot(B.T)
    D = AA.repeat(BB.shape[0], axis=1) + BB.T.repeat(AA.shape[0], axis=0) - 2 * AB
    D = np.abs(D)
    return D


class GraphConstruction:

    def __init__(self, X):
        self.X = X

    def middle(self):
        Inner_product = self.X.mm(self.X.T)
        Graph_middle = torch.sigmoid(Inner_product)
        return Graph_middle

    def knn(self, k=9, issymmetry=False):
        n = self.X.shape[0]
        D = l2_distance(self.X, self.X)
        idx = np.argsort(D, axis=1)
        S = np.zeros((n, n))
        for i in range(n):
            id = idx[i][1:k + 1]
            S[i][id] = 1
        if issymmetry:
            S = (S + S.T) / 2
        return S

    def can(self, k=9, issymmetry=True):
        n = self.X.shape[0]
        D = l2_distance(self.X, self.X)
        idx = np.argsort(D, axis=1)
        S = np.zeros((n, n))
        for i in range(n):
            id = idx[i][1:k + 1 + 1]
            di = D[i][id]
            S[i][id] = (di[k].repeat(di.shape[0]) - di) / (k * di[k] - np.sum(di[0:k]) + 0.0001)
        if issymmetry:
            S = (S + S.T) / 2
        return S

    def adjacency_incomplete(self, scale):
        Adjacency = np.array(self.adjacency)
        raw, col = np.nonzero(Adjacency)
        Num_nozero = len(raw)
        Num_setzero = round(scale * Num_nozero)
        Index_setzero = random.sample(range(0, Num_nozero), Num_setzero)
        for i in range(Num_setzero):
            raw_0 = raw[Index_setzero[i]]
            col_0 = col[Index_setzero[i]]
            Adjacency[raw_0][col_0] = 0
        left_0, _ = np.nonzero(Adjacency)
        None
        return torch.Tensor(Adjacency)


def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2 * bound * torch.rand(d1, d2)
    return torch.Tensor(nor_W)


class myGAE(torch.nn.Module):

    def __init__(self, d_0, d_1, d_2):
        super(myGAE, self).__init__()
        self.gconv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1), torch.nn.ReLU(inplace=True))
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)
        self.gconv2 = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.gconv2[0].weight.data = get_weight_initial(d_2, d_1)

    def encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2 = self.gconv2(torch.matmul(Adjacency_Modified, H_1))
        return H_2

    def graph_decoder(self, H_2):
        graph_re = GraphConstruction(H_2)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction

    def forward(self, adjacency_modified, H_0):
        latent_representation = self.encoder(adjacency_modified, H_0)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return graph_reconstruction, latent_representation


class myVGAE(torch.nn.Module):

    def __init__(self, d_0, d_1, d_2):
        super(myVGAE, self).__init__()
        self.gconv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1), torch.nn.ReLU(inplace=True))
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)
        self.gconv2_mean = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.gconv2_mean[0].weight.data = get_weight_initial(d_2, d_1)
        self.gconv2_std = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.gconv2_std[0].weight.data = get_weight_initial(d_2, d_1)

    def encoder(self, Adjacency_Convolution, H_0):
        H_1 = self.gconv1(Adjacency_Convolution.mm(H_0))
        H_2_mean = self.gconv2_mean(torch.matmul(Adjacency_Convolution, H_1))
        H_2_std = self.gconv2_std(torch.matmul(Adjacency_Convolution, H_1))
        return H_2_mean, H_2_std

    def reparametrization(self, H_2_mean, H_2_std):
        eps = torch.randn_like(H_2_std)
        std = torch.exp(H_2_std)
        latent_representation = eps.mul(std) + H_2_mean
        return latent_representation

    def graph_decoder(self, latent_representation):
        graph_re = GraphConstruction(latent_representation)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction

    def forward(self, adjacency_convolution, H_0):
        H_2_mean, H_2_std = self.encoder(adjacency_convolution, H_0)
        latent_representation = self.reparametrization(H_2_mean, H_2_std)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return latent_representation, graph_reconstruction, H_2_mean, H_2_std


class myLGAE(torch.nn.Module):

    def __init__(self, d_0, d_1):
        super(myLGAE, self).__init__()
        self.gconv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1))
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)

    def encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        return H_1

    def graph_decoder(self, H_1):
        graph_re = GraphConstruction(H_1)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction

    def forward(self, Adjacency_Modified, H_0):
        latent_representation = self.encoder(Adjacency_Modified, H_0)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return graph_reconstruction, latent_representation


class myLVGAE(torch.nn.Module):

    def __init__(self, d_0, d_1):
        super(myLVGAE, self).__init__()
        self.gconv1_mean = torch.nn.Sequential(torch.nn.Linear(d_0, d_1))
        self.gconv1_mean[0].weight.data = get_weight_initial(d_1, d_0)
        self.gconv1_std = torch.nn.Sequential(torch.nn.Linear(d_0, d_1))
        self.gconv1_std[0].weight.data = get_weight_initial(d_1, d_0)

    def encoder(self, adjacency_convolution, H_0):
        H_1_mean = self.gconv1_mean(torch.matmul(adjacency_convolution, H_0))
        H_1_std = self.gconv1_std(torch.matmul(adjacency_convolution, H_0))
        return H_1_mean, H_1_std

    def reparametrization(self, H_1_mean, H_1_std):
        eps = torch.randn_like(H_1_std)
        std = torch.exp(H_1_std)
        latent_representation = eps.mul(std) + H_1_mean
        return latent_representation

    def graph_decoder(self, latent_representation):
        graph_re = GraphConstruction(latent_representation)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction

    def forward(self, adjacency_convolution, H_0):
        H_1_mean, H_1_std = self.encoder(adjacency_convolution, H_0)
        latent_representation = self.reparametrization(H_1_mean, H_1_std)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return latent_representation, graph_reconstruction, H_1_mean, H_1_std


class mySDNE(torch.nn.Module):

    def __init__(self, d_0, d_1, d_2):
        super(mySDNE, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1), torch.nn.ReLU(inplace=True))
        self.conv2 = torch.nn.Sequential(torch.nn.Linear(d_1, d_2))
        self.conv3 = torch.nn.Sequential(torch.nn.Linear(d_2, d_1), torch.nn.ReLU(inplace=True))
        self.conv4 = torch.nn.Sequential(torch.nn.Linear(d_1, d_0), torch.nn.Sigmoid())

    def Encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2 = self.conv2(H_1)
        return H_2

    def Decoder(self, H_2):
        H_3 = self.conv3(H_2)
        H_4 = self.conv4(H_3)
        return H_4

    def forward(self, G_0):
        Latent_Representation = self.Encoder(G_0)
        Graph_Reconstrction = self.Decoder(Latent_Representation)
        return Latent_Representation, Graph_Reconstrction


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MyAE,
     lambda: ([], {'d_0': 4, 'd_1': 4, 'd_2': 4, 'd_3': 4, 'd_4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MySAE,
     lambda: ([], {'input_dim': 4, 'middle_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MyVAE,
     lambda: ([], {'d_0': 4, 'd_1': 4, 'd_2': 4, 'd_3': 4, 'd_4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (myGAE,
     lambda: ([], {'d_0': 4, 'd_1': 4, 'd_2': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (myLGAE,
     lambda: ([], {'d_0': 4, 'd_1': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (myLVGAE,
     lambda: ([], {'d_0': 4, 'd_1': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (mySDNE,
     lambda: ([], {'d_0': 4, 'd_1': 4, 'd_2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (myVGAE,
     lambda: ([], {'d_0': 4, 'd_1': 4, 'd_2': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_zyx423_Graph_Embeddding(_paritybench_base):
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

