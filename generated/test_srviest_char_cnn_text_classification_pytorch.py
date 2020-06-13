import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
data_loader_txt = _module
metric = _module
model = _module
model_CharCNN2D = _module
model_SentCNN = _module
mydatasets = _module
test = _module
train = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch import nn


from torch.autograd import Variable


from torch import optim


class CharCNN(nn.Module):

    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(args.num_features, 256,
            kernel_size=7, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3,
            stride=3))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7,
            stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            stride=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            stride=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3,
            stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.fc1 = nn.Sequential(nn.Linear(8704, 1024), nn.ReLU(), nn.
            Dropout(p=args.dropout))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.
            Dropout(p=args.dropout))
        self.fc3 = nn.Linear(1024, 4)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


class CharCNN(nn.Module):

    def __init__(self, num_features):
        super(CharCNN, self).__init__()
        self.num_features = num_features
        self.conv1 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(7, self.
            num_features), stride=1), nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(7, 256),
            stride=1), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.conv3 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(3, 256),
            stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(3, 256),
            stride=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(3, 256),
            stride=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(3, 256),
            stride=1), nn.ReLU())
        self.maxpool6 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.fc1 = nn.Sequential(nn.Linear(8704, 1024), nn.ReLU(), nn.
            Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.
            Dropout(p=0.5))
        self.fc3 = nn.Linear(1024, 4)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        debug = False
        x = x.unsqueeze(1)
        if debug:
            None
        x = self.conv1(x)
        if debug:
            None
        x = x.transpose(1, 3)
        if debug:
            None
        x = self.maxpool1(x)
        if debug:
            None
        x = self.conv2(x)
        if debug:
            None
        x = x.transpose(1, 3)
        if debug:
            None
        x = self.maxpool2(x)
        if debug:
            None
        x = self.conv3(x)
        if debug:
            None
        x = x.transpose(1, 3)
        if debug:
            None
        x = self.conv4(x)
        if debug:
            None
        x = x.transpose(1, 3)
        if debug:
            None
        x = self.conv5(x)
        if debug:
            None
        x = x.transpose(1, 3)
        if debug:
            None
        x = self.conv6(x)
        if debug:
            None
        x = x.transpose(1, 3)
        if debug:
            None
        x = self.maxpool6(x)
        if debug:
            None
        x = x.view(x.size(0), -1)
        if debug:
            None
        x = self.fc1(x)
        if debug:
            None
        x = self.fc2(x)
        if debug:
            None
        x = self.fc3(x)
        if debug:
            None
        x = self.softmax(x)
        return x


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)
        if self.args.static:
            x = Variable(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        """
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        """
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_srviest_char_cnn_text_classification_pytorch(_paritybench_base):
    pass
