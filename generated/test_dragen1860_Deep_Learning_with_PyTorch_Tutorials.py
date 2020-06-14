import sys
_module = sys.modules[__name__]
del sys
autograd_demo = _module
gpu_accelerate = _module
main = _module
gd = _module
mnist_train = _module
utils = _module
main = _module
main = _module
main = _module
main = _module
main = _module
main = _module
main = _module
main = _module
resnet = _module
main = _module
main = _module
lenet5 = _module
resnet = _module
rnn = _module
seris = _module
ae = _module
vae = _module
gan = _module
wgan_gp = _module
layers = _module
models = _module
train = _module
pokemon = _module
resnet = _module
train_scratch = _module
train_transfer = _module
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


from torch import nn


from torch.nn import functional as F


from torch import optim


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import DataLoader


import numpy as np


from torch import autograd


import random


import math


import time


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.ReLU(inplace=
            True), nn.Linear(200, 200), nn.ReLU(inplace=True), nn.Linear(
            200, 10), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1,
            padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size
                =1, stride=1), nn.BatchNorm2d(ch_out))

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1,
            padding=1), nn.BatchNorm2d(16))
        self.blk1 = ResBlk(16, 16)
        self.blk2 = ResBlk(16, 32)
        self.outlayer = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


class MyLinear(nn.Module):

    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()
        self.w = nn.Parameter(torch.randn(outp, inp))
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, stride=1, padding=1), nn.
            MaxPool2d(2, 2), Flatten(), nn.Linear(1 * 14 * 14, 10))

    def forward(self, x):
        return self.net(x)


class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(BasicNet(), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(
            inplace=True), nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.model(x)
        return x


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5,
            stride=1, padding=0), nn.MaxPool2d(kernel_size=2, stride=2,
            padding=0), nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=
            0), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.fc_unit = nn.Sequential(nn.Linear(32 * 5 * 5, 32), nn.ReLU(),
            nn.Linear(32, 10))
        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        None

    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, 32 * 5 * 5)
        logits = self.fc_unit(x)
        return logits


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride,
            padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size
                =1, stride=stride), nn.BatchNorm2d(ch_out))

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=3,
            padding=0), nn.BatchNorm2d(64))
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)
        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


hidden_size = 16


input_size = 1


output_size = 1


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=True)
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.
            Linear(256, 64), nn.ReLU(), nn.Linear(64, 20), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.
            Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, x):
        """

        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batchsz, 1, 28, 28)
        return x, None


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.
            Linear(256, 64), nn.ReLU(), nn.Linear(64, 20), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.
            Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())
        self.criteon = nn.MSELoss()

    def forward(self, x):
        """

        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        h_ = self.encoder(x)
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.randn_like(sigma)
        x_hat = self.decoder(h)
        x_hat = x_hat.view(batchsz, 1, 28, 28)
        kld = 0.5 * torch.sum(torch.pow(mu, 2) + torch.pow(sigma, 2) -
            torch.log(1e-08 + torch.pow(sigma, 2)) - 1) / (batchsz * 28 * 28)
        return x_hat, kld


h_dim = 400


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, h_dim), nn.ReLU(True), nn.
            Linear(h_dim, h_dim), nn.ReLU(True), nn.Linear(h_dim, h_dim),
            nn.ReLU(True), nn.Linear(h_dim, 2))

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, h_dim), nn.ReLU(True), nn.
            Linear(h_dim, h_dim), nn.ReLU(True), nn.Linear(h_dim, h_dim),
            nn.ReLU(True), nn.Linear(h_dim, 1), nn.Sigmoid())

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, h_dim), nn.ReLU(True), nn.
            Linear(h_dim, h_dim), nn.ReLU(True), nn.Linear(h_dim, h_dim),
            nn.ReLU(True), nn.Linear(h_dim, 2))

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, h_dim), nn.ReLU(True), nn.
            Linear(h_dim, h_dim), nn.ReLU(True), nn.Linear(h_dim, h_dim),
            nn.ReLU(True), nn.Linear(h_dim, 1), nn.Sigmoid())

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)
            )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """

        :param input:
        :param adj:
        :return:
        """
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """

        :param x: [2708, 1433]
        :param adj: [2708, 2708]
        :return:
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride,
            padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size
                =1, stride=stride), nn.BatchNorm2d(ch_out))

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=3,
            padding=0), nn.BatchNorm2d(16))
        self.blk1 = ResBlk(16, 32, stride=3)
        self.blk2 = ResBlk(32, 64, stride=3)
        self.blk3 = ResBlk(64, 128, stride=2)
        self.blk4 = ResBlk(128, 256, stride=2)
        self.outlayer = nn.Linear(256 * 3 * 3, num_class)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dragen1860_Deep_Learning_with_PyTorch_Tutorials(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicNet(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Discriminator(*[], **{}), [torch.rand([2, 2])], {})

    def test_002(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(GCN(*[], **{'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_004(self):
        self._check(Generator(*[], **{}), [torch.rand([2, 2])], {})

    @_fails_compile()
    def test_005(self):
        self._check(GraphConvolution(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_006(self):
        self._check(MLP(*[], **{}), [torch.rand([784, 784])], {})

    def test_007(self):
        self._check(MyLinear(*[], **{'inp': 4, 'outp': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ResBlk(*[], **{'ch_in': 4, 'ch_out': 4}), [torch.rand([4, 4, 4, 4])], {})

