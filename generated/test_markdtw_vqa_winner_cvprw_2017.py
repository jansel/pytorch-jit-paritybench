import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
main = _module
model = _module
preproc = _module
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


import random


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


class Model(nn.Module):

    def __init__(self, vocab_size, wemb_init, hid_dim, num_classes):
        super(Model, self).__init__()
        """
            vocab_size (int): how many vocabs
            wemb_init (str): path to pretrained wemb weights
            hid_dim (int): for GRU -> 512
            num_classes (int): output classes -> 3129
        """
        emb_dim = int(wemb_init.split('_')[-1].split('.')[0])
        self.hid_dim = hid_dim
        self.wembed = nn.Embedding(vocab_size + 1, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.att = nn.Linear(hid_dim, 1)
        self.clf = nn.Linear(hid_dim, num_classes)
        self.clf_do = nn.Dropout(0.5, inplace=True)
        pretrained_wemb = np.zeros((vocab_size + 1, emb_dim), dtype=np.float32)
        pretrained_wemb[:vocab_size] = np.load(wemb_init)
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))
        self.gth_iatt = nn.Linear(2048 + hid_dim, hid_dim)
        self.gthp_iatt = nn.Linear(2048 + hid_dim, hid_dim)
        self.gth_q = nn.Linear(hid_dim, hid_dim)
        self.gthp_q = nn.Linear(hid_dim, hid_dim)
        self.gth_i = nn.Linear(2048, hid_dim)
        self.gthp_i = nn.Linear(2048, hid_dim)
        self.gth_clf = nn.Linear(hid_dim, hid_dim)
        self.gthp_clf = nn.Linear(hid_dim, hid_dim)

    def forward(self, image, question):
        """
        question -> shape (batch, 14)
        image -> shape (batch, 36, 2048)
        """
        emb = self.wembed(question)
        enc, hid = self.gru(emb.permute(1, 0, 2))
        qenc = enc[-1]
        qenc_reshape = qenc.repeat(1, 36).view(-1, 36, self.hid_dim)
        image = F.normalize(image, -1)
        concated = torch.cat((image, qenc_reshape), -1)
        concated = torch.mul(torch.tanh(self.gth_iatt(concated)), torch.
            sigmoid(self.gthp_iatt(concated)))
        a = self.att(concated)
        a = F.softmax(a.squeeze(), dim=1)
        v_head = torch.bmm(a.unsqueeze(1), image).squeeze()
        q = torch.mul(torch.tanh(self.gth_q(qenc)), torch.sigmoid(self.
            gthp_q(qenc)))
        v = torch.mul(torch.tanh(self.gth_i(v_head)), torch.sigmoid(self.
            gthp_i(v_head)))
        h = torch.mul(q, v)
        s_head = self.clf(torch.mul(torch.tanh(self.gth_clf(h)), torch.
            sigmoid(self.gthp_clf(h))))
        s_head = self.clf_do(s_head)
        return s_head


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_markdtw_vqa_winner_cvprw_2017(_paritybench_base):
    pass
