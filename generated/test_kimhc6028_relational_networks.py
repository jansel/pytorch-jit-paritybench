import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
sort_of_clevr_generator = _module
translator = _module

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


import random


import numpy as np


import torch


from torch.utils.tensorboard import SummaryWriter


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


class ConvInputModel(nn.Module):

    def __init__(self):
        super(ConvInputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


class FCOutputModel(nn.Module):

    def __init__(self):
        super(FCOutputModel, self).__init__()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class BasicModel(nn.Module):

    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100.0 / len(label)
        return accuracy, loss

    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100.0 / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):

    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        self.conv = ConvInputModel()
        self.relation_type = args.relation_type
        if self.relation_type == 'ternary':
            self.g_fc1 = nn.Linear((24 + 2) * 3 + 18, 256)
        else:
            self.g_fc1 = nn.Linear((24 + 2) * 2 + 18, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)
        self.f_fc1 = nn.Linear(256, 256)
        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi
            self.coord_oj = self.coord_oj
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        def cvt_coord(i):
            return [(i / 5 - 2) / 2.0, (i % 5 - 2) / 2.0]
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:, (i), :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        self.fcout = FCOutputModel()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.conv(img)
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1)
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)
        if self.relation_type == 'ternary':
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 1)
            qst = torch.unsqueeze(qst, 1)
            x_i = torch.unsqueeze(x_flat, 1)
            x_i = torch.unsqueeze(x_i, 3)
            x_i = x_i.repeat(1, 25, 1, 25, 1)
            x_j = torch.unsqueeze(x_flat, 2)
            x_j = torch.unsqueeze(x_j, 2)
            x_j = x_j.repeat(1, 1, 25, 25, 1)
            x_k = torch.unsqueeze(x_flat, 1)
            x_k = torch.unsqueeze(x_k, 1)
            x_k = torch.cat([x_k, qst], 4)
            x_k = x_k.repeat(1, 25, 25, 1, 1)
            x_full = torch.cat([x_i, x_j, x_k], 4)
            x_ = x_full.view(mb * (d * d) * (d * d) * (d * d), 96)
        else:
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 2)
            x_i = torch.unsqueeze(x_flat, 1)
            x_i = x_i.repeat(1, 25, 1, 1)
            x_j = torch.unsqueeze(x_flat, 2)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)
            x_full = torch.cat([x_i, x_j], 3)
            x_ = x_full.view(mb * (d * d) * (d * d), 70)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        if self.relation_type == 'ternary':
            x_g = x_.view(mb, d * d * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, d * d * (d * d), 256)
        x_g = x_g.sum(1).squeeze()
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        return self.fcout(x_f)


class CNN_MLP(BasicModel):

    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')
        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(5 * 5 * 24 + 18, 256)
        self.fcout = FCOutputModel()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.conv(img)
        """fully connected layers"""
        x = x.view(x.size(0), -1)
        x_ = torch.cat((x, qst), 1)
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        return self.fcout(x_)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvInputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FCOutputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([256, 256])], {}),
     True),
]

class Test_kimhc6028_relational_networks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

