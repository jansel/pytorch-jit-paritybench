import sys
_module = sys.modules[__name__]
del sys
cars = _module
cub200 = _module
datasetgetter = _module
dogs = _module
photoimage = _module
testfolder = _module
laboratory = _module
main = _module
discriminator = _module
generator = _module
ops = _module
train = _module
utils = _module
validation = _module

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


import numpy as np


import torch.nn.functional as F


import torch


import warnings


from torch import autograd


from torch.nn import functional as F


import torch.nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


from torch import nn


import math


from scipy import signal


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


from torch.autograd import Variable


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.nf = 32
        self.current_scale = 0
        self.sub_discriminators = nn.ModuleList()
        first_discriminator = nn.ModuleList()
        first_discriminator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1,
            1), nn.LeakyReLU(0.2)))
        for _ in range(3):
            first_discriminator.append(nn.Sequential(nn.Conv2d(self.nf,
                self.nf, 3, 1, 1), nn.BatchNorm2d(self.nf), nn.LeakyReLU(0.2)))
        first_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, 1, 3, 1,
            1)))
        first_discriminator = nn.Sequential(*first_discriminator)
        self.sub_discriminators.append(first_discriminator)

    def forward(self, x):
        out = self.sub_discriminators[self.current_scale](x)
        return out

    def progress(self):
        self.current_scale += 1
        if self.current_scale % 4 == 0:
            self.nf *= 2
        tmp_discriminator = nn.ModuleList()
        tmp_discriminator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1, 
            1), nn.LeakyReLU(0.2)))
        for _ in range(3):
            tmp_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, self.
                nf, 3, 1, 1), nn.BatchNorm2d(self.nf), nn.LeakyReLU(0.2)))
        tmp_discriminator.append(nn.Sequential(nn.Conv2d(self.nf, 1, 3, 1, 1)))
        tmp_discriminator = nn.Sequential(*tmp_discriminator)
        if self.current_scale % 4 != 0:
            prev_discriminator = self.sub_discriminators[-1]
            if self.current_scale >= 1:
                tmp_discriminator.load_state_dict(prev_discriminator.
                    state_dict())
        self.sub_discriminators.append(tmp_discriminator)
        None


class Generator(nn.Module):

    def __init__(self, img_size_min, num_scale, scale_factor=4 / 3):
        super(Generator, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.nf = 32
        self.current_scale = 0
        self.size_list = [int(self.img_size_min * scale_factor ** i) for i in
            range(num_scale + 1)]
        None
        self.sub_generators = nn.ModuleList()
        first_generator = nn.ModuleList()
        first_generator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1),
            nn.BatchNorm2d(self.nf), nn.LeakyReLU(0.2)))
        for _ in range(3):
            first_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf,
                3, 1), nn.BatchNorm2d(self.nf), nn.LeakyReLU(0.2)))
        first_generator.append(nn.Sequential(nn.Conv2d(self.nf, 3, 3, 1),
            nn.Tanh()))
        first_generator = nn.Sequential(*first_generator)
        self.sub_generators.append(first_generator)

    def forward(self, z, img=None):
        x_list = []
        x_first = self.sub_generators[0](z[0])
        x_list.append(x_first)
        if img is not None:
            x_inter = img
        else:
            x_inter = x_first
        for i in range(1, self.current_scale + 1):
            x_inter = F.interpolate(x_inter, (self.size_list[i], self.
                size_list[i]), mode='bilinear', align_corners=True)
            x_prev = x_inter
            x_inter = F.pad(x_inter, [5, 5, 5, 5], value=0)
            x_inter = x_inter + z[i]
            x_inter = self.sub_generators[i](x_inter) + x_prev
            x_list.append(x_inter)
        return x_list

    def progress(self):
        self.current_scale += 1
        if self.current_scale % 4 == 0:
            self.nf *= 2
        tmp_generator = nn.ModuleList()
        tmp_generator.append(nn.Sequential(nn.Conv2d(3, self.nf, 3, 1), nn.
            BatchNorm2d(self.nf), nn.LeakyReLU(0.2)))
        for _ in range(3):
            tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 
                3, 1), nn.BatchNorm2d(self.nf), nn.LeakyReLU(0.2)))
        tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, 3, 3, 1), nn.
            Tanh()))
        tmp_generator = nn.Sequential(*tmp_generator)
        if self.current_scale % 4 != 0:
            prev_generator = self.sub_generators[-1]
            if self.current_scale >= 1:
                tmp_generator.load_state_dict(prev_generator.state_dict())
        self.sub_generators.append(tmp_generator)
        None


class CInstanceNorm(nn.Module):

    def __init__(self, nfilter, nlabels):
        super().__init__()
        self.nlabels = nlabels
        self.nfilter = nfilter
        self.alpha_embedding = nn.Embedding(nlabels, nfilter)
        self.beta_embedding = nn.Embedding(nlabels, nfilter)
        self.bn = nn.InstanceNorm2d(nfilter, affine=False)
        nn.init.uniform(self.alpha_embedding.weight, -1.0, 1.0)
        nn.init.constant_(self.beta_embedding.weight, 0.0)

    def forward(self, x, y):
        dim = len(x.size())
        batch_size = x.size(0)
        assert dim >= 2
        assert x.size(1) == self.nfilter
        s = [batch_size, self.nfilter] + [1] * (dim - 2)
        alpha = self.alpha_embedding(y)
        alpha = alpha.view(s)
        beta = self.beta_embedding(y)
        beta = beta.view(s)
        out = self.bn(x)
        out = alpha * out + beta
        return out


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1,
            self.num_features, 1, 1)
        return out


class MultiConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        self.gamma = nn.Linear(num_classes, num_features)
        self.beta = nn.Linear(num_classes, num_features)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y)
        beta = self.beta(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1,
            self.num_features, 1, 1)
        return out


class SpatialAdaptiveNorm2d(nn.Module):

    def __init__(self, num_features, hid_features=64, momentum=0.1,
        num_classes=0):
        super().__init__()
        self.num_features = num_features
        self.hid_features = hid_features
        self.num_classes = num_classes
        if num_classes > 0:
            self.bn = MultiConditionalBatchNorm2d(num_features, num_classes)
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=
                momentum)
        self.hidden = nn.Sequential(nn.Conv2d(3, hid_features, 3, 1, 1), nn
            .LeakyReLU(0.2))
        self.gamma = nn.Conv2d(hid_features, num_features, 3, 1, 1)
        self.beta = nn.Conv2d(hid_features, num_features, 3, 1, 1)

    def forward(self, feat, img, y=None):
        rimg = F.interpolate(img, size=feat.size()[2:])
        if self.num_classes > 0 and y is not None:
            feat = self.bn(feat, y)
        else:
            feat = self.bn(feat)
        out = self.hidden(rimg)
        gamma = self.gamma(out)
        beta = self.beta(out)
        out = gamma * feat + beta
        return out


class SpatialModulatedNorm2d(nn.Module):

    def __init__(self, num_features, hid_features=64, momentum=0.1,
        num_classes=0):
        super().__init__()
        self.num_features = num_features
        self.hid_features = hid_features
        self.num_classes = num_classes
        if num_classes > 0:
            self.bn = MultiConditionalBatchNorm2d(num_features, num_classes,
                momentum=momentum)
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=
                momentum)
        self.hidden_img = nn.Sequential(nn.Conv2d(3 + 16, hid_features, 3, 
            1, 1), nn.LeakyReLU(0.2))
        self.gamma = nn.Conv2d(hid_features, num_features, 3, 1, 1)
        self.beta = nn.Conv2d(hid_features, num_features, 3, 1, 1)

    def forward(self, feat, img, z, y=None):
        rimg = F.interpolate(img, size=feat.size()[2:])
        rz = F.interpolate(z, size=feat.size()[2:])
        rin = torch.cat((rimg, rz), 1)
        out = self.hidden_img(rin)
        if self.num_classes > 0 and y is not None:
            feat = self.bn(feat, y)
        else:
            feat = self.bn(feat)
        gamma = self.gamma(out)
        beta = self.beta(out)
        out = gamma * feat + beta
        return out


class SelfModulratedBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_latent, num_hidden=0, num_classes=
        0, momentum=0.1):
        super(SelfModulratedBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.num_latent = num_latent
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.bn = nn.BatchNorm2d(num_features, affine=False, momentum=momentum)
        if num_hidden > 0:
            self.fc_z = nn.Sequential(nn.Linear(num_latent, num_hidden), nn
                .ReLU(True))
            num_latent = num_hidden
        self.gamma = nn.Linear(num_latent, num_features)
        self.beta = nn.Linear(num_latent, num_features)
        if num_classes > 0:
            self.fc_y1 = nn.Linear(num_classes, num_latent)
            self.fc_y2 = nn.Linear(num_classes, num_latent)

    def forward(self, h, z, y=None):
        if self.num_hidden > 0:
            z = self.fc_z(z)
        if y is not None and self.num_classes > 0:
            z = z + self.fc_y1(y) + z * self.fc_y2(y)
        out = self.bn(h)
        gamma = self.gamma(z)
        beta = self.beta(z)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1,
            self.num_features, 1, 1)
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def max_singular_value(w_mat, u, power_iterations):
    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))
        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))
    sigma = torch.sum(torch.mm(u, w_mat) * v)
    return u, sigma, v


class Embedding(torch.nn.Embedding):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer('u', torch.randn((1, self.num_embeddings),
                requires_grad=False))
        else:
            self.register_buffer('u', None)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.num_embeddings, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.
                spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight
        return F.embedding(input, w_bar, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class GCRNNCellBase(torch.nn.Module):

    def __init__(self, latent_size, ch, bias, num_hidden=3, sn=False):
        super(GCRNNCellBase, self).__init__()
        self.ch = ch
        self.latent_size = latent_size
        self.concat_size = self.latent_size // 4
        if sn:
            self.layer_zh = torch.nn.utils.spectral_norm(torch.nn.Conv2d(
                latent_size, self.concat_size, 3, 1, 1, bias=bias))
            self.layer_hh = torch.nn.ModuleList([torch.nn.utils.
                spectral_norm(torch.nn.Conv2d(ch, self.concat_size, 3, 1, 1,
                bias=bias))])
            nf = 2 * self.concat_size
            for i in range(num_hidden - 1):
                self.layer_hh.append(torch.nn.Sequential(torch.nn.utils.
                    spectral_norm(torch.nn.Conv2d(nf, 2 * nf, 3, 1, 1, bias
                    =bias)), torch.nn.BatchNorm2d(2 * nf), torch.nn.ReLU(True))
                    )
                nf *= 2
            self.layer_hh.append(torch.nn.Sequential(torch.nn.utils.
                spectral_norm(torch.nn.Conv2d(nf, 3, 3, 1, 1, bias=bias)),
                torch.nn.Tanh()))
        else:
            self.layer_zh = torch.nn.Conv2d(latent_size, self.concat_size, 
                3, 1, 1, bias=bias)
            self.layer_hh = torch.nn.ModuleList([torch.nn.Conv2d(ch, self.
                concat_size, 3, 1, 1, bias=bias)])
            nf = 2 * self.concat_size
            for i in range(num_hidden - 1):
                self.layer_hh.append(torch.nn.Sequential(torch.nn.Conv2d(nf,
                    2 * nf, 3, 1, 1, bias=bias), torch.nn.BatchNorm2d(2 *
                    nf), torch.nn.ReLU(True)))
                nf *= 2
            self.layer_hh.append(torch.nn.Sequential(torch.nn.Conv2d(nf, 3,
                3, 1, 1, bias=bias), torch.nn.Tanh()))

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.latent_size:
            raise RuntimeError(
                'input has inconsistent input_size: got {}, expected {}'.
                format(input.size(1), self.latent_size))


class DCRNNCellBase(torch.nn.Module):

    def __init__(self, nf, ch, bias, num_hidden=3, sn=False):
        super(DCRNNCellBase, self).__init__()
        self.ch = ch
        self.nf = nf
        if sn:
            self.layer_xhh = torch.nn.ModuleList([torch.nn.Sequential(torch
                .nn.utils.spectral_norm(torch.nn.Conv2d(ch + self.nf, self.
                nf, 3, 1, 1, bias=bias)), torch.nn.LeakyReLU(inplace=True))])
            nf_ = self.nf
            for i in range(num_hidden - 1):
                self.layer_xhh.append(torch.nn.Sequential(torch.nn.utils.
                    spectral_norm(torch.nn.Conv2d(nf_, 2 * nf_, 3, 1, 1,
                    bias=bias)), torch.nn.BatchNorm2d(2 * nf_), torch.nn.
                    LeakyReLU(True)))
                nf_ *= 2
            self.layer_xhh.append(torch.nn.Sequential(torch.nn.utils.
                spectral_norm(torch.nn.Conv2d(nf_, self.nf, 3, 1, 1, bias=
                bias)), torch.nn.BatchNorm2d(self.nf), torch.nn.LeakyReLU()))
        else:
            self.layer_xhh = torch.nn.ModuleList([torch.nn.Sequential(torch
                .nn.Conv2d(ch + self.nf, self.nf, 3, 1, 1, bias=bias),
                torch.nn.LeakyReLU(inplace=True))])
            nf_ = self.nf
            for i in range(num_hidden - 1):
                self.layer_xhh.append(torch.nn.Sequential(torch.nn.Conv2d(
                    nf_, 2 * nf_, 3, 1, 1, bias=bias), torch.nn.BatchNorm2d
                    (2 * nf_), torch.nn.LeakyReLU(True)))
                nf_ *= 2
            self.layer_xhh.append(torch.nn.Sequential(torch.nn.Conv2d(nf_,
                self.nf, 3, 1, 1, bias=bias), torch.nn.BatchNorm2d(self.nf),
                torch.nn.LeakyReLU()))

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.latent_size:
            raise RuntimeError(
                'input has inconsistent input_size: got {}, expected {}'.
                format(input.size(1), self.latent_size))


class EMA(torch.nn.Module):

    def __init__(self, mu=0.999):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_FriedRonaldo_SinGAN(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Discriminator(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(CInstanceNorm(*[], **{'nfilter': 4, 'nlabels': 4}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(ConditionalBatchNorm2d(*[], **{'num_features': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_003(self):
        self._check(MultiConditionalBatchNorm2d(*[], **{'num_features': 4, 'num_classes': 4}), [torch.rand([64, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(SelfModulratedBatchNorm2d(*[], **{'num_features': 4, 'num_latent': 4}), [torch.rand([64, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Embedding(*[], **{'num_embeddings': 4, 'embedding_dim': 4}), [torch.zeros([4], dtype=torch.int64)], {})

