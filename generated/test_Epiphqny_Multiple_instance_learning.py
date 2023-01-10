import sys
_module = sys.modules[__name__]
del sys
model_attention = _module
train = _module
build_vocab = _module
one_hot = _module
deepmiml = _module
train_visual = _module
visual_concept = _module

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


import torch.nn as nn


import torchvision.models as models


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.functional import avg_pool2d


from torch.autograd import Variable


import numpy as np


from torchvision import transforms


import torch.nn.functional as F


import time


import random


class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg = models.vgg16(pretrained=True)
        modules = list(vgg.features[i] for i in range(29))
        self.vgg = nn.Sequential(*modules)

    def forward(self, images):
        with torch.no_grad():
            features = self.vgg(images)
        return features


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x
    return Variable(x, volatile=volatile)


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=40):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size * 2, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab_size = vocab_size
        self.vis_dim = 512
        self.hidden_dim = 1024
        vis_num = 196
        self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        self.att_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

    def attention(self, features, hiddens):
        att_fea = self.att_vw(features)
        att_h = self.att_hw(hiddens).unsqueeze(1)
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full)
        alpha = nn.Softmax(dim=1)(att_out)
        context = torch.sum(features * alpha, 1)
        return context, alpha

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        feats = torch.mean(features, 1).unsqueeze(1)
        embeddings = torch.cat((feats, embeddings), 1)
        batch_size, time_step = captions.size()
        predicts = to_var(torch.zeros(batch_size, time_step, self.vocab_size))
        hx = to_var(torch.zeros(batch_size, 1024))
        cx = to_var(torch.zeros(batch_size, 1024))
        for i in range(time_step):
            feas, _ = self.attention(features, hx)
            input = torch.cat((feas, embeddings[:, i, :]), -1)
            hx, cx = self.lstm_cell(input, (hx, cx))
            output = self.linear(hx)
            predicts[:, i, :] = output
        return predicts

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        hx = to_var(torch.zeros(1, 1024))
        cx = to_var(torch.zeros(1, 1024))
        inputs = torch.mean(features, 1)
        alphas = []
        for i in range(self.max_seg_length):
            feas, alpha = self.attention(features, hx)
            alphas.append(alpha)
            inputs = torch.cat((feas, inputs), -1)
            hx, cx = self.lstm_cell(inputs, (hx, cx))
            outputs = self.linear(hx.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids, alphas


class DeepMIML(nn.Module):

    def __init__(self, L=1032, K=100):
        super(DeepMIML, self).__init__()
        self.L = L
        self.K = K
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=L * K, kernel_size=1)
        self.pool1 = nn.MaxPool2d((K, 1), stride=(1, 1))
        self.activation = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d((1, 14 * 14), stride=(1, 1))

    def forward(self, features):
        N, C, H, W = features.size()
        n_instances = H * W
        conv1 = self.conv1(features)
        conv1 = conv1.view(N, self.L, self.K, n_instances)
        pool1 = self.pool1(conv1)
        act = self.activation(pool1)
        pool2 = self.pool2(act)
        out = pool2.view(N, self.L)
        None
        return out


class Sample_loss(torch.nn.Module):

    def __init__(self):
        super(Sample_loss, self).__init__()

    def forward(self, x, y, lengths):
        loss = 0
        batch_size = len(lengths) // 8
        for i in range(batch_size):
            label_index = y[i][:lengths[i]]
            values = 1 - x[i][label_index]
            prod = 1
            for value in values:
                prod = prod * value
            None
            loss += 1 - prod
        loss = Variable(loss, requires_grad=True).unsqueeze(0)
        return loss


class bce_loss(torch.nn.Module):

    def __init__(self):
        super(bce_loss, self).__init__()

    def forward(self, x, y):
        loss = F.binary_cross_entropy(x, y)
        loss = Variable(loss, requires_grad=True)
        return loss


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=3)
        self.relu0 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=4096, out_channels=1032, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pool_mil = nn.MaxPool2d(8, stride=0)

    def forward(self, features):
        N, C, H, W = features.size()
        relu0 = self.relu0(features)
        conv1 = self.conv1(relu0)
        relu1 = self.relu1(conv1)
        conv2 = self.conv2(relu1)
        relu2 = self.relu2(conv2)
        conv3 = self.conv3(relu2)
        sigmoid = self.sigmoid(conv3)
        pool = self.pool_mil(sigmoid)
        x = pool.squeeze(2).squeeze(2)
        x1 = torch.add(torch.mul(sigmoid.view(x.size(0), 1032, -1), -1), 1)
        cumprod = torch.prod(x1, 2)
        out = torch.min(x, torch.add(torch.mul(cumprod, -1), 1))
        None
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (bce_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Epiphqny_Multiple_instance_learning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

