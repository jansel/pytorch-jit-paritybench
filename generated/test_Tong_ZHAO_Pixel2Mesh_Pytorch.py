import sys
_module = sys.modules[__name__]
del sys
ShapeNet = _module
evaluate = _module
dist_chamfer = _module
setup = _module
gcn_layers = _module
loss = _module
model = _module
train = _module
utils = _module

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


import numpy as np


import torchvision


import torch.utils.data as data


from torchvision.transforms import Resize


import torch


import torch.optim as optim


import time


import random


import math


from torch import nn


from torch.autograd import Function


from numbers import Number


from collections import Set


from collections import Mapping


from collections import deque


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from scipy.sparse import coo_matrix


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


class chamferFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        dist1 = dist1
        dist2 = dist2
        idx1 = idx1
        idx2 = idx2
        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        gradxyz1 = gradxyz1
        gradxyz2 = gradxyz2
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class chamferDist(nn.Module):

    def __init__(self):
        super(chamferDist, self).__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        res = x.mm(y)
    else:
        res = torch.matmul(x, y)
    return res


def torch_sparse_tensor(indice, value, size, use_cuda):
    coo = coo_matrix((value, (indice[:, 0], indice[:, 1])), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    if use_cuda:
        return torch.sparse.FloatTensor(i, v, shape)
    else:
        return torch.sparse.FloatTensor(i, v, shape)


class GraphConvolution(Module):
    """Simple GCN layer
    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adjs, bias=True, use_cuda=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        adj0 = torch_sparse_tensor(*adjs[0], use_cuda)
        adj1 = torch_sparse_tensor(*adjs[1], use_cuda)
        self.adjs = [adj0, adj1]
        self.weight_1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_2 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_1.size(1))
        self.weight_1.data.uniform_(-stdv, stdv)
        self.weight_2.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support_1 = torch.matmul(input, self.weight_1)
        support_2 = torch.matmul(input, self.weight_2)
        output1 = dot(self.adjs[0], support_1, True)
        output2 = dot(self.adjs[1], support_2, True)
        output = output1 + output2
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphPooling(Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.

    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """

    def __init__(self, pool_idx):
        super(GraphPooling, self).__init__()
        self.pool_idx = pool_idx
        self.in_num = np.max(pool_idx)
        self.out_num = self.in_num + len(pool_idx)

    def forward(self, input):
        new_features = input[self.pool_idx].clone()
        new_vertices = 0.5 * new_features.sum(1)
        output = torch.cat((input, new_vertices), 0)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_num) + ' -> ' + str(self.out_num) + ')'


class GraphProjection(Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use 
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(GraphProjection, self).__init__()

    def forward(self, img_features, input):
        self.img_feats = img_features
        h = 248 * torch.div(input[:, 1], input[:, 2]) + 111.5
        w = 248 * torch.div(input[:, 0], -input[:, 2]) + 111.5
        h = torch.clamp(h, min=0, max=223)
        w = torch.clamp(w, min=0, max=223)
        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = [input]
        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], out_dims[i])
            feats.append(out)
        output = torch.cat(feats, 1)
        return output

    def project(self, index, h, w, img_size, out_dim):
        img_feat = self.img_feats[index]
        x = h / (224.0 / img_size)
        y = w / (224.0 / img_size)
        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
        x2 = torch.clamp(x2, max=img_size - 1)
        y2 = torch.clamp(y2, max=img_size - 1)
        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()
        x, y = x.long(), y.long()
        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))
        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0, 1))
        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))
        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))
        output = Q11 + Q21 + Q12 + Q22
        return output


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adjs, use_cuda):
        super(GResBlock, self).__init__()
        self.conv1 = GraphConvolution(in_features=in_dim, out_features=hidden_dim, adjs=adjs, use_cuda=use_cuda)
        self.conv2 = GraphConvolution(in_features=in_dim, out_features=hidden_dim, adjs=adjs, use_cuda=use_cuda)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return (input + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adjs, use_cuda):
        super(GBottleneck, self).__init__()
        blocks = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adjs=adjs, use_cuda=use_cuda)]
        for _ in range(block_num - 1):
            blocks.append(GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adjs=adjs, use_cuda=use_cuda))
        self.blocks = nn.Sequential(*blocks)
        self.conv1 = GraphConvolution(in_features=in_dim, out_features=hidden_dim, adjs=adjs, use_cuda=use_cuda)
        self.conv2 = GraphConvolution(in_features=hidden_dim, out_features=out_dim, adjs=adjs, use_cuda=use_cuda)

    def forward(self, input):
        x = self.conv1(input)
        x_cat = self.blocks(x)
        x_out = self.conv2(x_cat)
        return x_out, x_cat


class VGG16_Decoder(nn.Module):

    def __init__(self, input_dim=512, image_channel=3):
        super(VGG16_Decoder, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(input_dim, 256, kernel_size=2, stride=2, padding=0)
        self.conv_2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.conv_3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.conv_4 = nn.ConvTranspose2d(128, 32, kernel_size=6, stride=2, padding=2)
        self.conv_5 = nn.ConvTranspose2d(32, image_channel, kernel_size=6, stride=2, padding=2)

    def forward(self, img_feats):
        x = F.relu(self.conv_1(img_feats[-1].unsqueeze(0)))
        x = torch.cat((x, img_feats[-2].unsqueeze(0)), dim=1)
        x = F.relu(self.conv_2(x))
        x = torch.cat((x, img_feats[-3].unsqueeze(0)), dim=1)
        x = F.relu(self.conv_3(x))
        x = torch.cat((x, img_feats[-4].unsqueeze(0)), dim=1)
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))
        return torch.sigmoid(x)


class VGG16_Pixel2Mesh(nn.Module):

    def __init__(self, n_classes_input=3):
        super(VGG16_Pixel2Mesh, self).__init__()
        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

    def forward(self, img):
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))
        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))
        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = torch.squeeze(img)
        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img3 = torch.squeeze(img)
        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img4 = torch.squeeze(img)
        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = torch.squeeze(img)
        return [img2, img3, img4, img5]


class P2M_Model(nn.Module):
    """
    Implement the joint model for Pixel2mesh
    """

    def __init__(self, features_dim, hidden_dim, coord_dim, pool_idx, supports, use_cuda):
        super(P2M_Model, self).__init__()
        self.img_size = 224
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.pool_idx = pool_idx
        self.supports = supports
        self.use_cuda = use_cuda
        self.build()

    def build(self):
        self.nn_encoder = self.build_encoder()
        self.nn_decoder = self.build_decoder()
        self.GCN_0 = GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim, self.supports[0], self.use_cuda)
        self.GCN_1 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim, self.supports[1], self.use_cuda)
        self.GCN_2 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.hidden_dim, self.supports[2], self.use_cuda)
        self.GPL_1 = GraphPooling(self.pool_idx[0])
        self.GPL_2 = GraphPooling(self.pool_idx[1])
        self.GPR_0 = GraphProjection()
        self.GPR_1 = GraphProjection()
        self.GPR_2 = GraphProjection()
        self.GConv = GraphConvolution(in_features=self.hidden_dim, out_features=self.coord_dim, adjs=self.supports[2], use_cuda=self.use_cuda)
        self.GPL_12 = GraphPooling(self.pool_idx[0])
        self.GPL_22 = GraphPooling(self.pool_idx[1])

    def forward(self, img, input):
        img_feats = self.nn_encoder(img)
        x = self.GPR_0(img_feats, input)
        x1, x_cat = self.GCN_0(x)
        x1_2 = self.GPL_12(x1)
        x = self.GPR_1(img_feats, x1)
        x = torch.cat([x, x_cat], 1)
        x = self.GPL_1(x)
        x2, x_cat = self.GCN_1(x)
        x2_2 = self.GPL_22(x2)
        x = self.GPR_2(img_feats, x2)
        x = torch.cat([x, x_cat], 1)
        x = self.GPL_2(x)
        x, _ = self.GCN_2(x)
        x3 = self.GConv(x)
        new_img = self.nn_decoder(img_feats)
        return [x1, x2, x3], [input, x1_2, x2_2], new_img

    def build_encoder(self):
        net = VGG16_Pixel2Mesh(n_classes_input=3)
        return net

    def build_decoder(self):
        net = VGG16_Decoder()
        return net


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (VGG16_Pixel2Mesh,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_Tong_ZHAO_Pixel2Mesh_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

