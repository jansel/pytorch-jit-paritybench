import sys
_module = sys.modules[__name__]
del sys
base_module = _module
download_CelebA = _module
download_lsun = _module
mmd = _module
mmd_gan = _module
util = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import random


import torch


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.utils.data


import torchvision.utils as vutils


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


class Encoder(nn.Module):

    def __init__(self, isize, nc, k=100, ndf=64):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, 'isize has to be a multiple of 16'
        main = nn.Sequential()
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf), nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1), nn.Conv2d(cndf, k, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


class Decoder(nn.Module):

    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, 'isize has to be a multiple of 16'
        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        main = nn.Sequential()
        main.add_module('initial.{0}-{1}.convt'.format(k, cngf), nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf), nn.ReLU(True))
        csize = 4
        while csize < isize // 2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf // 2), nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid.{0}.relu'.format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2
        main.add_module('final.{0}-{1}.convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc), nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


class NetG(nn.Module):

    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


class NetD(nn.Module):

    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)
        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):

    def __init__(self):
        super(ONE_SIDED, self).__init__()
        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NetD,
     lambda: ([], {'encoder': _mock_layer(), 'decoder': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NetG,
     lambda: ([], {'decoder': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ONE_SIDED,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_OctoberChang_MMD_GAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

