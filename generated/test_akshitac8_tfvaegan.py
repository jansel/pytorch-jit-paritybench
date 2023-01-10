import sys
_module = sys.modules[__name__]
del sys
classifier_actions = _module
classifier_entropy = _module
classifier_images = _module
config_actions = _module
config_images = _module
action_util = _module
image_util = _module
TFVAEGAN_model = _module
run_awa_tfvaegan = _module
run_cub_tfvaegan = _module
run_flo_tfvaegan = _module
run_hmdb51_tfvaegan = _module
run_sun_tfvaegan = _module
run_ucf101_tfvaegan = _module
train_actions = _module
train_images = _module

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


from torch.autograd import Variable


import torch.optim as optim


import copy


import torch.nn.functional as F


import numpy as np


from sklearn.preprocessing import MinMaxScaler


import scipy.io as sio


from sklearn import preprocessing


import torch.autograd as autograd


import torch.backends.cudnn as cudnn


import random


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):

    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class ODDetector(nn.Module):

    def __init__(self, input_dim, h_size, num_classes):
        super(ODDetector, self).__init__()
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.classifier = nn.Linear(h_size, num_classes)

    def forward(self, x, center_loss=False):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        pred = self.classifier(h)
        return pred


class HLoss(nn.Module):

    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, neg=True, batch=False):
        b = self.softmax(x) * self.logsoft(x)
        if batch:
            return -1.0 * b.sum(1)
        if neg:
            return -1.0 * b.sum() / x.size(0)
        else:
            return b.sum() / x.size(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Encoder(nn.Module):

    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3 = nn.Linear(layer_sizes[-1], latent_size * 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size * 2, latent_size)
        self.linear_log_var = nn.Linear(latent_size * 2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        latent_size = opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z, c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1 * feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x


class Discriminator_D1(nn.Module):

    def __init__(self, opt):
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h


class Feedback(nn.Module):

    def __init__(self, opt):
        super(Feedback, self).__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        self.x1 = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(self.x1))
        return h


class AttDec(nn.Module):

    def __init__(self, opt, attSize):
        super(AttDec, self).__init__()
        self.embedSz = 0
        self.fc1 = nn.Linear(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        self.out = h
        return h

    def getLayersOutDet(self):
        return self.hidden.detach()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttDec,
     lambda: ([], {'opt': _mock_config(resSize=4, ngh=4), 'attSize': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Discriminator_D1,
     lambda: ([], {'opt': _mock_config(resSize=4, attSize=4, ndh=4)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Feedback,
     lambda: ([], {'opt': _mock_config(ngh=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LINEAR_LOGSOFTMAX_CLASSIFIER,
     lambda: ([], {'input_dim': 4, 'nclass': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ODDetector,
     lambda: ([], {'input_dim': 4, 'h_size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_akshitac8_tfvaegan(_paritybench_base):
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

