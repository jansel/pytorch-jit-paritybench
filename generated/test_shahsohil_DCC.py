import sys
_module = sys.modules[__name__]
del sys
DCC = _module
DCCComputation = _module
DCCLoss = _module
SDAE = _module
pytorch = _module
config = _module
convSDAE = _module
copyGraph = _module
custom_data = _module
data_params = _module
easy_example = _module
edgeConstruction = _module
extractSDAE = _module
extract_feature = _module
extractconvSDAE = _module
make_data = _module
pretraining = _module

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


import numpy as np


import torch.nn.functional as F


import torch.nn.init as init


import random


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


class DCCWeightedELoss(nn.Module):

    def __init__(self, size_average=True):
        super(DCCWeightedELoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, outputs, weights):
        out = (inputs - outputs).view(len(inputs), -1)
        out = torch.sum(weights * torch.norm(out, p=2, dim=1) ** 2)
        assert np.isfinite(out.data.cpu().numpy()).all(), 'Nan found in data'
        if self.size_average:
            out = out / inputs.nelement()
        return out


class DCCLoss(nn.Module):

    def __init__(self, nsamples, ndim, initU, size_average=True):
        super(DCCLoss, self).__init__()
        self.dim = ndim
        self.nsamples = nsamples
        self.size_average = size_average
        self.U = nn.Parameter(torch.Tensor(self.nsamples, self.dim))
        self.reset_parameters(initU + 1e-06 * np.random.randn(*initU.shape)
            .astype(np.float32))

    def reset_parameters(self, initU):
        assert np.isfinite(initU).all(), 'Nan found in initialization'
        self.U.data = torch.from_numpy(initU)

    def forward(self, enc_out, sampweights, pairweights, pairs, index,
        _sigma1, _sigma2, _lambda):
        centroids = self.U[index]
        out1 = torch.norm((enc_out - centroids).view(len(enc_out), -1), p=2,
            dim=1) ** 2
        out11 = torch.sum(_sigma1 * sampweights * out1 / (_sigma1 + out1))
        out2 = torch.norm((centroids[pairs[:, (0)]] - centroids[pairs[:, (1
            )]]).view(len(pairs), -1), p=2, dim=1) ** 2
        out21 = _lambda * torch.sum(_sigma2 * pairweights * out2 / (_sigma2 +
            out2))
        out = out11 + out21
        if self.size_average:
            out = out / enc_out.nelement()
        return out


class SDAE(nn.Module):

    def __init__(self, dim, dropout=0.2, slope=0.0):
        super(SDAE, self).__init__()
        self.in_dim = dim[0]
        self.nlayers = len(dim) - 1
        self.reluslope = slope
        self.enc, self.dec = [], []
        for i in range(self.nlayers):
            self.enc.append(nn.Linear(dim[i], dim[i + 1]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            self.dec.append(nn.Linear(dim[i + 1], dim[i]))
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
        self.base = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))
        self.dropmodule1 = nn.Dropout(p=dropout)
        self.dropmodule2 = nn.Dropout(p=dropout)
        self.loss = nn.MSELoss(size_average=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.01)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self, x, index):
        inp = x.view(-1, self.in_dim)
        encoded = inp
        for i, encoder in enumerate(self.enc):
            if i < index:
                encoded = encoder(encoded)
                if i < self.nlayers - 1:
                    encoded = F.leaky_relu(encoded, negative_slope=self.
                        reluslope)
            if i == index:
                inp = encoded
                out = encoded
                if index:
                    out = self.dropmodule1(out)
                out = encoder(out)
        if index < self.nlayers - 1:
            out = F.leaky_relu(out, negative_slope=self.reluslope)
            out = self.dropmodule2(out)
        if index >= self.nlayers:
            out = encoded
        for i, decoder in reversed(list(enumerate(self.dec))):
            if index >= self.nlayers:
                out = decoder(out)
                if i:
                    out = F.leaky_relu(out, negative_slope=self.reluslope)
            if i == index:
                out = decoder(out)
                if index:
                    out = F.leaky_relu(out, negative_slope=self.reluslope)
        out = self.loss(out, inp)
        return out


class convSDAE(nn.Module):

    def __init__(self, dim, output_padding, numpen, dropout=0.2, slope=0.0):
        super(convSDAE, self).__init__()
        self.in_dim = dim[0]
        self.nlayers = len(dim) - 1
        self.reluslope = slope
        self.numpen = numpen
        self.enc, self.dec = [], []
        self.benc, self.bdec = [], []
        for i in range(self.nlayers):
            if i == self.nlayers - 1:
                self.enc.append(nn.Linear(dim[i] * numpen * numpen, dim[i + 1])
                    )
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i],
                    kernel_size=numpen, stride=1))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            elif i == 0:
                self.enc.append(nn.Conv2d(dim[i], dim[i + 1], kernel_size=4,
                    stride=2, padding=1))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i],
                    kernel_size=4, stride=2, padding=1, output_padding=
                    output_padding[i]))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            else:
                self.enc.append(nn.Conv2d(dim[i], dim[i + 1], kernel_size=5,
                    stride=2, padding=2))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i],
                    kernel_size=5, stride=2, padding=2, output_padding=
                    output_padding[i]))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            setattr(self, 'benc_{}'.format(i), self.benc[-1])
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
            setattr(self, 'bdec_{}'.format(i), self.bdec[-1])
        self.base = []
        self.bbase = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))
            self.bbase.append(nn.Sequential(*self.benc[:i]))
        self.dropmodule1 = nn.Dropout(p=dropout)
        self.dropmodule2 = nn.Dropout(p=dropout)
        self.loss = nn.MSELoss(size_average=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.01)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self, x, index):
        inp = x
        encoded = x
        for i, (encoder, bencoder) in enumerate(zip(self.enc, self.benc)):
            if i == self.nlayers - 1:
                encoded = encoded.view(encoded.size(0), -1)
            if i < index:
                encoded = encoder(encoded)
                if i < self.nlayers - 1:
                    encoded = bencoder(encoded)
                    encoded = F.leaky_relu(encoded, negative_slope=self.
                        reluslope)
            if i == index:
                inp = encoded
                out = encoded
                if index:
                    out = self.dropmodule1(out)
                out = encoder(out)
                if i < self.nlayers - 1:
                    out = bencoder(out)
        if index < self.nlayers - 1:
            out = F.leaky_relu(out, negative_slope=self.reluslope)
            out = self.dropmodule2(out)
        if index >= self.nlayers:
            out = encoded
        for i, (decoder, bdecoder) in reversed(list(enumerate(zip(self.dec,
            self.bdec)))):
            if index >= self.nlayers - 1 and i == self.nlayers - 1:
                out = out.view(out.size(0), -1, 1, 1)
            if index >= self.nlayers:
                out = decoder(out)
                if i:
                    out = bdecoder(out)
                    out = F.leaky_relu(out, negative_slope=self.reluslope)
            if i == index:
                out = decoder(out)
                if index:
                    out = bdecoder(out)
                    out = F.leaky_relu(out, negative_slope=self.reluslope)
        out = self.loss(out, inp)
        return out


class IdentityNet(nn.Module):
    """Substitute for the autoencoder for visualization and debugging just the clustering part"""

    def __init__(self):
        super(IdentityNet, self).__init__()

    def forward(self, x):
        return x, x


class extractSDAE(nn.Module):

    def __init__(self, dim, slope=0.0):
        super(extractSDAE, self).__init__()
        self.in_dim = dim[0]
        self.nlayers = len(dim) - 1
        self.reluslope = slope
        self.enc, self.dec = [], []
        for i in range(self.nlayers):
            self.enc.append(nn.Linear(dim[i], dim[i + 1]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            self.dec.append(nn.Linear(dim[i + 1], dim[i]))
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
        self.base = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.01)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self, x):
        inp = x.view(-1, self.in_dim)
        encoded = inp
        for i, encoder in enumerate(self.enc):
            encoded = encoder(encoded)
            if i < self.nlayers - 1:
                encoded = F.leaky_relu(encoded, negative_slope=self.reluslope)
        out = encoded
        for i, decoder in reversed(list(enumerate(self.dec))):
            out = decoder(out)
            if i:
                out = F.leaky_relu(out, negative_slope=self.reluslope)
        return encoded, out


class extractconvSDAE(nn.Module):

    def __init__(self, dim, output_padding, numpen, slope=0.0):
        super(extractconvSDAE, self).__init__()
        self.in_dim = dim[0]
        self.nlayers = len(dim) - 1
        self.reluslope = slope
        self.numpen = numpen
        self.enc, self.dec = [], []
        self.benc, self.bdec = [], []
        for i in range(self.nlayers):
            if i == self.nlayers - 1:
                self.enc.append(nn.Linear(dim[i] * numpen * numpen, dim[i + 1])
                    )
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i],
                    kernel_size=numpen, stride=1))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            elif i == 0:
                self.enc.append(nn.Conv2d(dim[i], dim[i + 1], kernel_size=4,
                    stride=2, padding=1))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i],
                    kernel_size=4, stride=2, padding=1, output_padding=
                    output_padding[i]))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            else:
                self.enc.append(nn.Conv2d(dim[i], dim[i + 1], kernel_size=5,
                    stride=2, padding=2))
                self.benc.append(nn.BatchNorm2d(dim[i + 1]))
                self.dec.append(nn.ConvTranspose2d(dim[i + 1], dim[i],
                    kernel_size=5, stride=2, padding=2, output_padding=
                    output_padding[i]))
                self.bdec.append(nn.BatchNorm2d(dim[i]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            setattr(self, 'benc_{}'.format(i), self.benc[-1])
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
            setattr(self, 'bdec_{}'.format(i), self.bdec[-1])
        self.base = []
        self.bbase = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))
            self.bbase.append(nn.Sequential(*self.benc[:i]))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.01)
                if m.bias.data is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias.data is not None:
                    init.constant(m.bias, 0)

    def forward(self, x):
        encoded = x
        for i, (encoder, bencoder) in enumerate(zip(self.enc, self.benc)):
            if i == self.nlayers - 1:
                encoded = encoded.view(encoded.size(0), -1)
            encoded = encoder(encoded)
            if i < self.nlayers - 1:
                encoded = bencoder(encoded)
                encoded = F.leaky_relu(encoded, negative_slope=self.reluslope)
        out = encoded
        for i, (decoder, bdecoder) in reversed(list(enumerate(zip(self.dec,
            self.bdec)))):
            if i == self.nlayers - 1:
                out = out.view(out.size(0), -1, 1, 1)
            out = decoder(out)
            if i:
                out = bdecoder(out)
                out = F.leaky_relu(out, negative_slope=self.reluslope)
        return encoded, out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_shahsohil_DCC(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SDAE(*[], **{'dim': [4, 4]}), [torch.rand([4, 4, 4, 4]), 0], {})

    @_fails_compile()
    def test_001(self):
        self._check(extractSDAE(*[], **{'dim': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(extractconvSDAE(*[], **{'dim': [4, 4], 'output_padding': 4, 'numpen': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DCCWeightedELoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(IdentityNet(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

