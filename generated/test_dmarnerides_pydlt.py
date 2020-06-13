import sys
_module = sys.modules[__name__]
del sys
dlt = _module
config = _module
dataset = _module
helpers = _module
model = _module
optim = _module
opts = _module
trainer = _module
hdr = _module
io = _module
train = _module
basetrainer = _module
began = _module
fishergan = _module
ganbasetrainer = _module
vanilla = _module
vanillagan = _module
wganct = _module
wgangp = _module
util = _module
barit = _module
checkpointer = _module
data = _module
dispatch = _module
external = _module
accuracy = _module
compose = _module
func = _module
grid = _module
imagesampler = _module
layers = _module
logger = _module
meter = _module
misc = _module
paths = _module
sample = _module
slurm = _module
viz = _module
csvplot = _module
imshow = _module
modules = _module
conf = _module
main = _module
models = _module
setup = _module

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


from itertools import chain


import torch


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import LambdaLR


import math


from torch.autograd import Variable


from torch import autograd


import numpy as np


from torch import nn


def selu_init(model):
    for m in model.modules():
        if any([isinstance(m, x) for x in [nn.Conv2d, nn.ConvTranspose2d,
            nn.Linear]]):
            nn.init.kaiming_normal_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)


class Generator(nn.Module):

    def __init__(self, num_hidden, z_dim, num_chan, num_pix):
        super(Generator, self).__init__()
        self.num_pix = num_pix
        self.num_chan = num_chan
        self.main = nn.Sequential(nn.Linear(z_dim, num_hidden), nn.SELU(),
            nn.Linear(num_hidden, num_hidden), nn.SELU(), nn.Linear(
            num_hidden, num_chan * num_pix * num_pix), nn.Tanh())
        selu_init(self)

    def forward(self, v_input):
        return self.main(v_input).view(v_input.size(0), self.num_chan, self
            .num_pix, self.num_pix)


class Discriminator(nn.Module):

    def __init__(self, num_hidden, num_chan, num_pix):
        super(Discriminator, self).__init__()
        self.num_pix = num_pix
        self.num_chan = num_chan
        self.main = nn.Sequential(nn.Linear(num_chan * num_pix * num_pix,
            num_hidden), nn.SELU(), nn.Linear(num_hidden, num_hidden), nn.
            SELU())
        self.last_layer = nn.Linear(num_hidden, 1)
        selu_init(self)

    def forward(self, v_input, correction_term=False):
        if correction_term:
            main = self.main(v_input.view(v_input.size(0), -1))
            noisy_main = nn.functional.dropout(main, p=0.1)
            return main, self.last_layer(noisy_main)
        else:
            return self.last_layer(self.main(v_input.view(v_input.size(0), -1))
                )


class DiscriminatorBEGAN(nn.Module):

    def __init__(self, num_hidden, num_chan, num_pix):
        super(DiscriminatorBEGAN, self).__init__()
        self.num_pix = num_pix
        self.num_chan = num_chan
        self.main = nn.Sequential(nn.Linear(num_chan * num_pix * num_pix,
            num_hidden), nn.SELU(), nn.Linear(num_hidden, num_hidden), nn.
            SELU(), nn.Linear(num_hidden, num_chan * num_pix * num_pix))
        selu_init(self)

    def forward(self, v_input):
        res = self.main(v_input.view(v_input.size(0), -1))
        return res.view(v_input.size(0), self.num_chan, self.num_pix, self.
            num_pix)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dmarnerides_pydlt(_paritybench_base):
    pass
    def test_000(self):
        self._check(Generator(*[], **{'num_hidden': 4, 'z_dim': 4, 'num_chan': 4, 'num_pix': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Discriminator(*[], **{'num_hidden': 4, 'num_chan': 4, 'num_pix': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(DiscriminatorBEGAN(*[], **{'num_hidden': 4, 'num_chan': 4, 'num_pix': 4}), [torch.rand([4, 4, 4, 4])], {})

