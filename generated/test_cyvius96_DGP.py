import sys
_module = sys.modules[__name__]
del sys
datasets = _module
image_folder = _module
imagenet = _module
evaluate_awa2 = _module
evaluate_imagenet = _module
glove = _module
make_dense_graph = _module
make_dense_grouped_graph = _module
make_induced_graph = _module
process_resnet = _module
models = _module
gcn = _module
gcn_dense = _module
gcn_dense_att = _module
resnet = _module
train_gcn_basic = _module
train_gcn_dense = _module
train_gcn_dense_att = _module
train_resnet_fit = _module
utils = _module

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


import torch


import torch.nn.functional as F


from torch.utils.data import DataLoader


import numpy as np


import scipy.sparse as sp


import torch.nn as nn


from torch.nn.init import xavier_uniform_


import math


import random


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b
        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long(
        )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


class GCN(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers):
        super().__init__()
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, (0)], edges[:,
            (1)])), shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj
        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False
        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)
            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)
            last_c = c
        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last
            )
        self.add_module('conv-last', conv)
        layers.append(conv)
        self.layers = layers

    def forward(self, x):
        for conv in self.layers:
            x = conv(x, self.adj)
        return F.normalize(x)


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b
        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN_Dense(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers):
        super().__init__()
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, (0)], edges[:,
            (1)])), shape=(n, n), dtype='float32')
        self.adj = spm_to_tensor(normt_spm(adj, method='in'))
        self.r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in'))
        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False
        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)
            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)
            last_c = c
        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last
            )
        self.add_module('conv-last', conv)
        layers.append(conv)
        self.layers = layers

    def forward(self, x):
        graph_side = True
        for conv in self.layers:
            if graph_side:
                x = conv(x, self.adj)
            else:
                x = conv(x, self.r_adj)
            graph_side = not graph_side
        return F.normalize(x)


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj_set, att):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        support = torch.mm(inputs, self.w) + self.b
        outputs = None
        for i, adj in enumerate(adj_set):
            y = torch.mm(adj, support) * att[i]
            if outputs is None:
                outputs = y
            else:
                outputs = outputs + y
        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN_Dense_Att(nn.Module):

    def __init__(self, n, edges_set, in_channels, out_channels, hidden_layers):
        super().__init__()
        self.n = n
        self.d = len(edges_set)
        self.a_adj_set = []
        self.r_adj_set = []
        for edges in edges_set:
            edges = np.array(edges)
            adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, (0)], edges
                [:, (1)])), shape=(n, n), dtype='float32')
            a_adj = spm_to_tensor(normt_spm(adj, method='in'))
            r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in'))
            self.a_adj_set.append(a_adj)
            self.r_adj_set.append(r_adj)
        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False
        self.a_att = nn.Parameter(torch.ones(self.d))
        self.r_att = nn.Parameter(torch.ones(self.d))
        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)
            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)
            last_c = c
        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last
            )
        self.add_module('conv-last', conv)
        layers.append(conv)
        self.layers = layers

    def forward(self, x):
        graph_side = True
        for conv in self.layers:
            if graph_side:
                adj_set = self.a_adj_set
                att = self.a_att
            else:
                adj_set = self.r_adj_set
                att = self.r_att
            att = F.softmax(att, dim=0)
            x = conv(x, adj_set, att)
            graph_side = not graph_side
        return F.normalize(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBase(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.out_channels = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


def make_resnet18_base(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNetBase(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def make_resnet101_base(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetBase(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def make_resnet50_base(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetBase(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def make_resnet34_base(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetBase(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def make_resnet152_base(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNetBase(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def make_resnet_base(version, pretrained=None):
    maker = {'resnet18': make_resnet18_base, 'resnet34': make_resnet34_base,
        'resnet50': make_resnet50_base, 'resnet101': make_resnet101_base,
        'resnet152': make_resnet152_base}
    resnet = maker[version]()
    if pretrained is not None:
        sd = torch.load(pretrained)
        sd.pop('fc.weight')
        sd.pop('fc.bias')
        resnet.load_state_dict(sd)
    return resnet


class ResNet(nn.Module):

    def __init__(self, version, num_classes, pretrained=None):
        super().__init__()
        self.resnet_base = make_resnet_base(version, pretrained=pretrained)
        self.fc = nn.Linear(self.resnet_base.out_channels, num_classes)

    def forward(self, x, need_features=False):
        x = self.resnet_base(x)
        feat = x
        x = self.fc(x)
        if need_features:
            return x, feat
        else:
            return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cyvius96_DGP(_paritybench_base):
    pass

    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
