import sys
_module = sys.modules[__name__]
del sys
constants = _module
data = _module
evaluation = _module
model = _module
opts = _module
preprocess = _module
test = _module
train = _module
visualize_w2v = _module
vocab = _module

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


import torch.utils.data as data


import torchvision.transforms as transforms


import numpy as np


import numpy


import time


from collections import OrderedDict


import torch.nn as nn


import torch.nn.init


import torchvision.models as models


from torch.autograd import Variable


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.backends.cudnn as cudnn


from torch.nn.utils.clip_grad import clip_grad_norm


import logging


import matplotlib.pyplot as plt


import matplotlib.cm as cm


from sklearn.decomposition import PCA


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    X = torch.div(X, norm.unsqueeze(1).expand_as(X))
    return X


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19', use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.cnn = self.get_cnn(cnn_type, True)
        for param in self.cnn.parameters():
            param.requires_grad = finetune
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features, embed_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()
        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            None
            model = models.__dict__[arch](pretrained=True)
        else:
            None
            model = models.__dict__[arch]()
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model
        else:
            model = nn.DataParallel(model)
        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)
        features = l2norm(features)
        features = self.fc(features)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.use_abs = opt.use_abs
        self.rnn_type = getattr(opt, 'rnn_type', 'GRU')
        self.num_layers = opt.num_layers
        self.embed_size = opt.embed_size
        self.batch_size = opt.batch_size
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        if 'Bi' in self.rnn_type:
            self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, bias=False, batch_first=True, bidirectional=1)
            self.bi_out = nn.Linear(opt.embed_size * 2, opt.embed_size)
        else:
            self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        if 'Bi' in self.rnn_type:
            """Xavier initialization for the fully connected layer
            """
            r = np.sqrt(6.0) / np.sqrt(self.bi_out.in_features + self.bi_out.out_features)
            self.bi_out.weight.data.uniform_(-r, r)
            self.bi_out.bias.data.fill_(0)

    def init_hidden(self, bsz):
        return Variable(torch.zeros(self.num_layers * 2, bsz, self.embed_size))

    def forward_GRU(self, x, lengths):
        """Handles variable size captions
        """
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1)
        out = torch.gather(padded[0], 1, I).squeeze(1)
        out = l2norm(out)
        if self.use_abs:
            out = torch.abs(out)
        return out

    def forward_BiGRU(self, x, lengths):
        outs = []
        x = self.embed(x)
        hiddens = self.init_hidden(x.data.size(0))
        emb = pack_padded_sequence(x, lengths, batch_first=True)
        outputs, hidden_t = self.rnn(emb, hiddens)
        output_unpack = pad_packed_sequence(outputs, batch_first=True)
        hidden_o = output_unpack[0].sum(1)
        output_lengths = Variable(torch.from_numpy(np.asarray(lengths)).float(), requires_grad=False)
        torch.div(hidden_o, output_lengths.view(-1, 1))
        hidden_map = self.bi_out(hidden_o)
        out = l2norm(hidden_map)
        if self.use_abs:
            out = torch.abs(out)
        return out

    def forward(self, x, lengths):
        if 'Bi' in self.rnn_type:
            out = self.forward_BiGRU(x, lengths)
        else:
            out = self.forward_GRU(x, lengths)
        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim
        self.max_violation = max_violation
    """
    For single-modal retrieval, emb1=query, emb2=data
    For cross-modal retrieval, emb1=query in source domain, emb2=data in target domain
    """

    def forward(self, emb1, emb2):
        scores = self.sim(emb1, emb2)
        diagonal = scores.diag().view(emb1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (EncoderImagePrecomp,
     lambda: ([], {'img_dim': 4, 'embed_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_gujiuxiang_Multimodal_Retrieval_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

