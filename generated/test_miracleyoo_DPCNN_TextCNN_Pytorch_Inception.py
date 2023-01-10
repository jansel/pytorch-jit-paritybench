import sys
_module = sys.modules[__name__]
del sys
config = _module
main = _module
BasicModule = _module
DPCNN = _module
TextCNNDeep = _module
TextCNNInc = _module
TextCNNIncDeep = _module
models = _module
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


import torch


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.optim as optim


import time


from copy import deepcopy


import torch.autograd


from torch.autograd import Variable


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, path):
        checkpoint = torch.load(path)
        step = checkpoint['step']
        best_acc = checkpoint['best_acc']
        self.load_state_dict(checkpoint['state_dict'])
        return self, step, best_acc

    def save(self, step, test_acc, name=None):
        prefix = './source/trained_net/' + self.model_name + '/'
        if name is None:
            name = 'temp_model.dat'
        path = prefix + name
        torch.save({'step': step + 1, 'state_dict': self.state_dict(), 'best_acc': test_acc}, path)
        return path

    def get_optimizer(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer


class ResnetBlock(nn.Module):

    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(nn.ConstantPad1d(padding=(0, 1), value=0), nn.MaxPool1d(kernel_size=3, stride=2))
        self.conv = nn.Sequential(nn.BatchNorm1d(num_features=self.channel_size), nn.ReLU(), nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=self.channel_size), nn.ReLU(), nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1))

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class DPCNN(BasicModule):
    """
    DPCNN model, 3
    1. region embedding: using TetxCNN to generte
    2. two 3 conv(padding) block
    3. maxpool->3 conv->3 conv with resnet block(padding) feature map: len/2
    """

    def __init__(self, opt):
        super(DPCNN, self).__init__()
        self.model_name = 'DPCNN'
        self.opt = opt
        if opt.USE_CHAR:
            opt.VOCAB_SIZE = opt.CHAR_SIZE
        self.embedding = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        self.region_embedding = nn.Sequential(nn.Conv1d(opt.EMBEDDING_DIM, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP), nn.ReLU(), nn.Dropout(0.2))
        self.conv_block = nn.Sequential(nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP), nn.ReLU(), nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP), nn.ReLU(), nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1))
        self.num_seq = opt.SENT_LEN
        resnet_block_list = []
        while self.num_seq > 2:
            resnet_block_list.append(ResnetBlock(opt.NUM_ID_FEATURE_MAP))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(nn.Linear(opt.NUM_ID_FEATURE_MAP * self.num_seq, opt.NUM_CLASSES), nn.BatchNorm1d(opt.NUM_CLASSES), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(opt.NUM_CLASSES, opt.NUM_CLASSES))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(self.opt.BATCH_SIZE, -1)
        out = self.fc(x)
        return out


class TextCNNDeep(BasicModule):
    """
    Warning! This module is NOT a pure DPCNN! It is a conbination of TextCNN and DPCNN,
    which means it will use TextCNN as its head feature extraction part, and use DPCNN
    to dig its high-level features accordingly.  
    """

    def __init__(self, opt):
        """
        initialize func.
        :param opt: config option class
        """
        super(TextCNNDeep, self).__init__()
        self.model_name = 'TextCNNDeep'
        self.opt = opt
        if opt.USE_CHAR:
            opt.VOCAB_SIZE = opt.CHAR_SIZE
        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        question_convs = [nn.Sequential(nn.Conv1d(in_channels=opt.EMBEDDING_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.TITLE_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.SENT_LEN - kernel_size * 2 + 2)) for kernel_size in opt.KERNEL_SIZE]
        self.change_dim_conv = nn.Conv1d(opt.TITLE_DIM, opt.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)
        self.question_convs = nn.ModuleList(question_convs)
        self.conv = nn.Sequential(nn.BatchNorm1d(num_features=opt.TITLE_DIM), nn.ReLU(), nn.Conv1d(opt.TITLE_DIM, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP), nn.ReLU(), nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1))
        self.num_seq = len(opt.KERNEL_SIZE)
        resnet_block_list = []
        while self.num_seq > 2:
            resnet_block_list.append(ResnetBlock(opt.NUM_ID_FEATURE_MAP))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(nn.Linear(opt.NUM_ID_FEATURE_MAP * self.num_seq, opt.NUM_CLASSES), nn.BatchNorm1d(opt.NUM_CLASSES), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(opt.NUM_CLASSES, opt.NUM_CLASSES))

    def forward(self, question):
        question = self.encoder(question)
        x = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        x = torch.cat(x, dim=2)
        xp = x
        xp = self.change_dim_conv(xp)
        x = self.conv(x)
        x = x + xp
        x = self.resnet_layer(x)
        x = x.view(self.opt.BATCH_SIZE, -1)
        x = self.fc(x)
        return x


class TextCNNInc(BasicModule):

    def __init__(self, opt):
        """
        initialize func.
        :param opt: config option class
        """
        super(TextCNNInc, self).__init__()
        self.model_name = 'TextCNNInc'
        if opt.USE_CHAR:
            opt.VOCAB_SIZE = opt.CHAR_SIZE
        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        question_convs1 = [nn.Sequential(nn.Conv1d(in_channels=opt.EMBEDDING_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.SENT_LEN - kernel_size + 1)) for kernel_size in opt.SIN_KER_SIZE]
        question_convs2 = [nn.Sequential(nn.Conv1d(in_channels=opt.EMBEDDING_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size[0]), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.TITLE_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size[1]), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.SENT_LEN - kernel_size[0] - kernel_size[1] + 2)) for kernel_size in opt.DOU_KER_SIZE]
        question_convs = deepcopy(question_convs1)
        question_convs.extend(question_convs2)
        self.question_convs = nn.ModuleList(question_convs)
        self.num_seq = len(opt.DOU_KER_SIZE) + len(opt.SIN_KER_SIZE)
        self.fc = nn.Sequential(nn.Linear(self.num_seq * opt.TITLE_DIM, opt.LINER_HID_SIZE), nn.BatchNorm1d(opt.LINER_HID_SIZE), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASSES))

    def forward(self, question):
        question = self.encoder(question)
        question_out = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        conv_out = torch.cat(question_out, dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


class TextCNNIncDeep(BasicModule):

    def __init__(self, opt):
        """
        Warning! This module is NOT a pure DPCNN! It is a conbination of TextCNN and DPCNN,
        which means it will use TextCNN as its head feature extraction part, and use DPCNN
        to dig its high-level features accordingly.  
        """
        super(TextCNNIncDeep, self).__init__()
        self.model_name = 'TextCNNIncDeep'
        self.opt = opt
        if opt.USE_CHAR:
            opt.VOCAB_SIZE = opt.CHAR_SIZE
        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        question_convs1 = [nn.Sequential(nn.Conv1d(in_channels=opt.EMBEDDING_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.SENT_LEN - kernel_size + 1)) for kernel_size in opt.SIN_KER_SIZE]
        question_convs2 = [nn.Sequential(nn.Conv1d(in_channels=opt.EMBEDDING_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size[0]), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.Conv1d(in_channels=opt.TITLE_DIM, out_channels=opt.TITLE_DIM, kernel_size=kernel_size[1]), nn.BatchNorm1d(opt.TITLE_DIM), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=opt.SENT_LEN - kernel_size[0] - kernel_size[1] + 2)) for kernel_size in opt.DOU_KER_SIZE]
        question_convs = question_convs1
        question_convs.extend(question_convs2)
        self.num_seq = len(opt.DOU_KER_SIZE) + len(opt.SIN_KER_SIZE)
        self.change_dim_conv = nn.Conv1d(opt.TITLE_DIM, opt.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)
        self.question_convs = nn.ModuleList(question_convs)
        self.conv = nn.Sequential(nn.BatchNorm1d(num_features=opt.TITLE_DIM), nn.ReLU(), nn.Conv1d(opt.TITLE_DIM, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP), nn.ReLU(), nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP, kernel_size=3, padding=1))
        resnet_block_list = []
        while self.num_seq > 2:
            resnet_block_list.append(ResnetBlock(opt.NUM_ID_FEATURE_MAP))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(nn.Linear(opt.NUM_ID_FEATURE_MAP * self.num_seq, opt.NUM_CLASSES), nn.BatchNorm1d(opt.NUM_CLASSES), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(opt.NUM_CLASSES, opt.NUM_CLASSES))

    def forward(self, question):
        question = self.encoder(question)
        x = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        x = torch.cat(x, dim=2)
        xp = x
        xp = self.change_dim_conv(xp)
        x = self.conv(x)
        x = x + xp
        x = self.resnet_layer(x)
        x = x.view(self.opt.BATCH_SIZE, -1)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResnetBlock,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_miracleyoo_DPCNN_TextCNN_Pytorch_Inception(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

