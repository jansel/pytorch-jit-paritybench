import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
data_camvid = _module
data_nyu = _module
loss = _module
mse = _module
mse_var = _module
main = _module
model = _module
aleatoric = _module
combined = _module
common = _module
epistemic = _module
normal = _module
op = _module
util = _module

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


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.parallel as P


import math


import numpy as np


from torch.utils.tensorboard import SummaryWriter


from functools import reduce


import torch.optim as optim


import torch.optim.lr_scheduler as lrs


from torch.nn.modules.module import _addindent


class Loss(nn.Module):

    def __init__(self, config):
        super(Loss, self).__init__()
        None
        self.num_gpu = config.num_gpu
        self.losses = []
        self.loss_module = nn.ModuleList()
        if config.uncertainty == 'epistemic' or config.uncertainty == 'normal':
            module = import_module('loss.mse')
            loss_function = getattr(module, 'MSE')()
        else:
            module = import_module('loss.mse_var')
            loss_function = getattr(module, 'MSE_VAR')(var_weight=config.var_weight)
        self.losses.append({'function': loss_function})
        self.loss_module
        if not config.cpu and config.num_gpu > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(self.num_gpu))

    def forward(self, results, label):
        losses = []
        for i, l in enumerate(self.losses):
            if l['function'] is not None:
                loss = l['function'](results, label)
                effective_loss = loss
                losses.append(effective_loss)
        loss_sum = sum(losses)
        if len(self.losses) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, results, label):
        mean = results['mean']
        loss = F.mse_loss(mean, label)
        return loss


class MSE_VAR(nn.Module):

    def __init__(self, var_weight):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, var = results['mean'], results['var']
        var = self.var_weight * var
        loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
        loss2 = var
        loss = 0.5 * (loss1 + loss2)
        return loss.mean()


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        None
        self.is_train = config.is_train
        self.num_gpu = config.num_gpu
        self.uncertainty = config.uncertainty
        self.n_samples = config.n_samples
        module = import_module('model.' + config.uncertainty)
        self.model = module.make_model(config)

    def forward(self, input):
        if self.model.training:
            if self.num_gpu > 1:
                return P.data_parallel(self.model, input, list(range(self.num_gpu)))
            else:
                return self.model.forward(input)
        else:
            forward_func = self.model.forward
            if self.uncertainty == 'normal':
                return forward_func(input)
            if self.uncertainty == 'aleatoric':
                return self.test_aleatoric(input, forward_func)
            elif self.uncertainty == 'epistemic':
                return self.test_epistemic(input, forward_func)
            elif self.uncertainty == 'combined':
                return self.test_combined(input, forward_func)

    def test_aleatoric(self, input, forward_func):
        results = forward_func(input)
        mean1 = results['mean']
        var1 = torch.exp(results['var'])
        var1_norm = var1 / var1.max()
        results = {'mean': mean1, 'var': var1_norm}
        return results

    def test_epistemic(self, input, forward_func):
        mean1s = []
        mean2s = []
        for i_sample in range(self.n_samples):
            results = forward_func(input)
            mean1 = results['mean']
            mean1s.append(mean1 ** 2)
            mean2s.append(mean1)
        mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
        mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
        var1 = mean1s_ - mean2s_ ** 2
        var1_norm = var1 / var1.max()
        results = {'mean': mean2s_, 'var': var1_norm}
        return results

    def test_combined(self, input, forward_func):
        mean1s = []
        mean2s = []
        var1s = []
        for i_sample in range(self.n_samples):
            results = forward_func(input)
            mean1 = results['mean']
            mean1s.append(mean1 ** 2)
            mean2s.append(mean1)
            var1 = results['var']
            var1s.append(torch.exp(var1))
        mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
        mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
        var1s_ = torch.stack(var1s, dim=0).mean(dim=0)
        var2 = mean1s_ - mean2s_ ** 2
        var_ = var1s_ + var2
        var_norm = var_ / var_.max()
        results = {'mean': mean2s_, 'var': var_norm}
        return results

    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, 'model_latest.pt')]
        save_dirs.append(os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Decoder, self).__init__()
        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1), nn.BatchNorm2d(n_in_feat), nn.ReLU(inplace=True)]
        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1), nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)


class _Encoder(nn.Module):

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()
        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1), nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]
        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1), nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class ALEATORIC(nn.Module):

    def __init__(self, config):
        super(ALEATORIC, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        filter_config = 64, 128
        self.encoders = nn.ModuleList()
        self.decoders_mean = nn.ModuleList()
        self.decoders_var = nn.ModuleList()
        encoder_n_layers = 2, 2, 3, 3, 3
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = 3, 3, 3, 2, 1
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        for i in range(0, 2):
            self.encoders.append(_Encoder(encoder_filter_config[i], encoder_filter_config[i + 1], encoder_n_layers[i]))
            self.decoders_mean.append(_Decoder(decoder_filter_config[i], decoder_filter_config[i + 1], decoder_n_layers[i]))
            self.decoders_var.append(_Decoder(decoder_filter_config[i], decoder_filter_config[i + 1], decoder_n_layers[i]))
        self.classifier_mean = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)
        self.classifier_var = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x
        for i in range(0, 2):
            (feat, ind), size = self.encoders[i](feat)
            if i == 1:
                feat = F.dropout(feat, p=self.drop_rate)
            indices.append(ind)
            unpool_sizes.append(size)
        feat_mean = feat
        feat_var = feat
        for i in range(0, 2):
            feat_mean = self.decoders_mean[i](feat_mean, indices[1 - i], unpool_sizes[1 - i])
            feat_var = self.decoders_var[i](feat_var, indices[1 - i], unpool_sizes[1 - i])
            if i == 0:
                feat_mean = F.dropout(feat_mean, p=self.drop_rate)
                feat_var = F.dropout(feat_var, p=self.drop_rate)
        output_mean = self.classifier_mean(feat_mean)
        output_var = self.classifier_var(feat_var)
        results = {'mean': output_mean, 'var': output_var}
        return results


class COMBINED(nn.Module):

    def __init__(self, config):
        super(COMBINED, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        filter_config = 64, 128
        self.encoders = nn.ModuleList()
        self.decoders_mean = nn.ModuleList()
        self.decoders_var = nn.ModuleList()
        encoder_n_layers = 2, 2, 3, 3, 3
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = 3, 3, 3, 2, 1
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        for i in range(0, 2):
            self.encoders.append(_Encoder(encoder_filter_config[i], encoder_filter_config[i + 1], encoder_n_layers[i]))
            self.decoders_mean.append(_Decoder(decoder_filter_config[i], decoder_filter_config[i + 1], decoder_n_layers[i]))
            self.decoders_var.append(_Decoder(decoder_filter_config[i], decoder_filter_config[i + 1], decoder_n_layers[i]))
        self.classifier_mean = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)
        self.classifier_var = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x
        for i in range(0, 2):
            (feat, ind), size = self.encoders[i](feat)
            if i == 1:
                feat = F.dropout(feat, p=self.drop_rate, training=True)
            indices.append(ind)
            unpool_sizes.append(size)
        feat_mean = feat
        feat_var = feat
        for i in range(0, 2):
            feat_mean = self.decoders_mean[i](feat_mean, indices[1 - i], unpool_sizes[1 - i])
            feat_var = self.decoders_var[i](feat_var, indices[1 - i], unpool_sizes[1 - i])
            if i == 0:
                feat_mean = F.dropout(feat_mean, p=self.drop_rate, training=True)
                feat_var = F.dropout(feat_var, p=self.drop_rate, training=True)
        output_mean = self.classifier_mean(feat_mean)
        output_var = self.classifier_var(feat_var)
        results = {'mean': output_mean, 'var': output_var}
        return results


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class EPISTEMIC(nn.Module):

    def __init__(self, config):
        super(EPISTEMIC, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        filter_config = 64, 128
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        encoder_n_layers = 2, 2, 3, 3, 3
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = 3, 3, 3, 2, 1
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        for i in range(0, 2):
            self.encoders.append(_Encoder(encoder_filter_config[i], encoder_filter_config[i + 1], encoder_n_layers[i]))
            self.decoders.append(_Decoder(decoder_filter_config[i], decoder_filter_config[i + 1], decoder_n_layers[i]))
        self.classifier = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x
        for i in range(0, 2):
            (feat, ind), size = self.encoders[i](feat)
            if i == 1:
                feat = F.dropout(feat, p=self.drop_rate, training=True)
            indices.append(ind)
            unpool_sizes.append(size)
        for i in range(0, 2):
            feat = self.decoders[i](feat, indices[1 - i], unpool_sizes[1 - i])
            if i == 0:
                feat = F.dropout(feat, p=self.drop_rate, training=True)
        output = self.classifier(feat)
        results = {'mean': output}
        return results


class NORMAL(nn.Module):

    def __init__(self, config):
        super(NORMAL, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        filter_config = 64, 128
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        encoder_n_layers = 2, 2, 2, 2, 2
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = 2, 2, 2, 2, 1
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        for i in range(0, 2):
            self.encoders.append(_Encoder(encoder_filter_config[i], encoder_filter_config[i + 1], encoder_n_layers[i]))
            self.decoders.append(_Decoder(decoder_filter_config[i], decoder_filter_config[i + 1], decoder_n_layers[i]))
        self.classifier = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x
        for i in range(0, 2):
            (feat, ind), size = self.encoders[i](feat)
            if i == 1:
                feat = F.dropout(feat, p=self.drop_rate)
            indices.append(ind)
            unpool_sizes.append(size)
        for i in range(0, 2):
            feat = self.decoders[i](feat, indices[1 - i], unpool_sizes[1 - i])
            if i == 0:
                feat = F.dropout(feat, p=self.drop_rate)
        output = self.classifier(feat)
        results = {'mean': output}
        return results


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ALEATORIC,
     lambda: ([], {'config': _mock_config(drop_rate=0.5, in_channels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'conv': _mock_layer, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (COMBINED,
     lambda: ([], {'config': _mock_config(drop_rate=0.5, in_channels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EPISTEMIC,
     lambda: ([], {'config': _mock_config(drop_rate=0.5, in_channels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NORMAL,
     lambda: ([], {'config': _mock_config(drop_rate=0.5, in_channels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feats': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Encoder,
     lambda: ([], {'n_in_feat': 4, 'n_out_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hmi88_what(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

