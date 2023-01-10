import sys
_module = sys.modules[__name__]
del sys
data = _module
an4 = _module
common_voice = _module
data_loader = _module
distributed = _module
librispeech = _module
merge_manifests = _module
ted = _module
utils = _module
voxforge = _module
decoder = _module
model = _module
multiproc = _module
noise_inject = _module
test = _module
train = _module
transcribe = _module

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


from torch.distributed import get_rank


from torch.distributed import get_world_size


from torch.utils.data.sampler import Sampler


import numpy as np


import scipy.signal


import torch


from scipy.io.wavfile import read


import math


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import scipy.io.wavfile as wave


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import torch.distributed as dist


from torch.nn.modules import Module


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import torchaudio


import time


from torch.autograd import Variable


import torch.quantization


import random


import torch.utils.data.distributed


import warnings


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = False
        self.module = module
        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)

        def allreduce_params():
            if self.needs_reduction:
                self.needs_reduction = False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if self.warn_on_half:
                    if torch.HalfTensor in buckets:
                        None
                        self.warn_on_half = False
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)
        for param in list(self.module.parameters()):

            def allreduce_hook(*unused):
                param._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)


def _get_stft_kernels(nfft, window):
    nfft = int(nfft)
    assert nfft % 2 == 0

    def kernel_fn(freq, time):
        return np.exp(-1.0j * (2 * np.pi * time * freq) / float(nfft))
    kernels = np.fromfunction(kernel_fn, (nfft // 2 + 1, nfft), dtype=np.float64)
    if window == 'hanning':
        win_cof = scipy.signal.get_window('hanning', nfft)[np.newaxis, :]
    else:
        win_cof = np.ones((1, nfft), dtype=np.float64)
    kernels = kernels[:, np.newaxis, np.newaxis, :] * win_cof
    real_kernels = nn.Parameter(torch.from_numpy(np.real(kernels)).float())
    imag_kernels = nn.Parameter(torch.from_numpy(np.imag(kernels)).float())
    return real_kernels, imag_kernels


class stft(nn.Module):

    def __init__(self, nfft=1024, hop_length=512, window='hanning'):
        super(stft, self).__init__()
        assert nfft % 2 == 0
        self.hop_length = hop_length
        self.n_freq = n_freq = nfft // 2 + 1
        self.real_kernels, self.imag_kernels = _get_stft_kernels(nfft, window)
        self.real_kernels_size = self.real_kernels.size()
        self.conv = nn.Sequential(nn.Conv2d(1, self.real_kernels_size[0], kernel_size=(self.real_kernels_size[2], self.real_kernels_size[3]), stride=self.hop_length), nn.BatchNorm2d(self.real_kernels_size[0]), nn.Hardtanh(0, 20, inplace=True))

    def forward(self, sample):
        sample = sample.unsqueeze(1)
        magn = self.conv(sample)
        magn = magn.permute(0, 2, 1, 3)
        return magn


class PCEN(nn.Module):

    def __init__(self):
        super(PCEN, self).__init__()
        """
        initialising the layer param with the best parametrised values i searched on web (scipy using theese values)
        alpha = 0.98
        delta=2
        r=0.5
        """
        self.log_alpha = Parameter(torch.FloatTensor([0.98]))
        self.log_delta = Parameter(torch.FloatTensor([2]))
        self.log_r = Parameter(torch.FloatTensor([0.5]))
        self.eps = 1e-06

    def forward(self, x, smoother):
        smooth = (self.eps + smoother) ** -self.log_alpha
        pcen = (x * smooth + self.log_delta) ** self.log_r - self.log_delta ** self.log_r
        return pcen


class InferenceBatchSoftmax(nn.Module):

    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class Cov1dBlock(nn.Module):

    def __init__(self, input_size, output_size, kernal_size, stride, drop_out_prob=-1.0, dilation=1, padding='same', bn=True, activationUse=True):
        super(Cov1dBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernal_size = kernal_size
        self.stride = stride
        self.dilation = dilation
        self.drop_out_prob = drop_out_prob
        self.activationUse = activationUse
        self.padding = kernal_size[0]
        """using the below code for the padding calculation"""
        input_rows = input_size
        filter_rows = kernal_size[0]
        effective_filter_size_rows = (filter_rows - 1) * dilation + 1
        out_rows = (input_rows + stride - 1) // stride
        self.rows_odd = False
        if padding == 'same':
            self.padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - input_rows)
            self.padding_rows = max(0, (out_rows - 1) * stride + (filter_rows - 1) * dilation + 1 - input_rows)
            self.rows_odd = self.padding_rows % 2 != 0
            self.addPaddings = self.padding_rows
        elif padding == 'half':
            self.addPaddings = kernal_size[0]
        elif padding == 'invalid':
            self.addPaddings = 0
        self.paddingAdded = nn.ReflectionPad1d(self.addPaddings // 2) if self.addPaddings > 0 else None
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernal_size, stride=stride, padding=0, dilation=dilation))
        self.batchNorm = nn.BatchNorm1d(num_features=output_size, momentum=0.9, eps=0.001) if bn else None
        self.drop_out_layer = nn.Dropout(drop_out_prob) if self.drop_out_prob != -1 else None

    def forward(self, xs, hid=None):
        if self.paddingAdded is not None:
            xs = self.paddingAdded(xs)
        fusedLayer = getattr(self, 'fusedLayer', None)
        if fusedLayer is not None:
            output = fusedLayer(xs)
        else:
            output = self.conv1(xs)
            if self.batchNorm is not None:
                output = self.batchNorm(output)
        if self.activationUse:
            output = torch.clamp(input=output, min=0, max=20)
        if self.training:
            if self.drop_out_layer is not None:
                output = self.drop_out_layer(output)
        return output


class WaveToLetter(nn.Module):

    def __init__(self, sample_rate, window_size, labels='abc', audio_conf=None, mixed_precision=False):
        super(WaveToLetter, self).__init__()
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._sample_rate = sample_rate
        self._window_size = window_size
        self.mixed_precision = mixed_precision
        nfft = self._sample_rate * self._window_size
        input_size = 1 + int(nfft / 2)
        hop_length = sample_rate * self._audio_conf.get('window_stride', 0.01)
        self.frontEnd = stft(hop_length=int(hop_length), nfft=int(nfft))
        conv1 = Cov1dBlock(input_size=input_size, output_size=256, kernal_size=(11,), stride=2, dilation=1, drop_out_prob=0.2, padding='same')
        conv2s = []
        conv2s.append(('conv1d_{}'.format(0), conv1))
        inputSize = 256
        for idx in range(15):
            layergroup = idx // 3
            if layergroup == 0:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=256, kernal_size=(11,), stride=1, dilation=1, drop_out_prob=0.2, padding='same')
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 256
            elif layergroup == 1:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=384, kernal_size=(13,), stride=1, dilation=1, drop_out_prob=0.2)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 384
            elif layergroup == 2:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=512, kernal_size=(17,), stride=1, dilation=1, drop_out_prob=0.2)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 512
            elif layergroup == 3:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=640, kernal_size=(21,), stride=1, dilation=1, drop_out_prob=0.3)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 640
            elif layergroup == 4:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=768, kernal_size=(25,), stride=1, dilation=1, drop_out_prob=0.3)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 768
        conv1 = Cov1dBlock(input_size=inputSize, output_size=896, kernal_size=(29,), stride=1, dilation=2, drop_out_prob=0.4)
        conv2s.append(('conv1d_{}'.format(16), conv1))
        conv1 = Cov1dBlock(input_size=896, output_size=1024, kernal_size=(1,), stride=1, dilation=1, drop_out_prob=0.4)
        conv2s.append(('conv1d_{}'.format(17), conv1))
        conv1 = Cov1dBlock(input_size=1024, output_size=len(self._labels), kernal_size=(1,), stride=1, bn=False, activationUse=False)
        conv2s.append(('conv1d_{}'.format(18), conv1))
        self.conv1ds = nn.Sequential(OrderedDict(conv2s))
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x):
        x = self.frontEnd(x)
        x = x.squeeze(1)
        x = self.conv1ds(x)
        x = x.transpose(1, 2)
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, cuda=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(labels=package['labels'], audio_conf=package['audio_conf'], sample_rate=package['sample_rate'], window_size=package['window_size'], mixed_precision=package.get('mixed_precision', False))
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias', 'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model)
        return model

    @classmethod
    def load_model_package(cls, package, cuda=False):
        model = cls(labels=package['labels'], audio_conf=package['audio_conf'], sample_rate=package.get('sample_rate', 16000), window_size=package.get('window_size', 0.02), mixed_precision=package.get('mixed_precision', False))
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model)
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, cer_results=None, wer_results=None, avg_loss=None, meta=None):
        model_is_cuda = next(model.parameters()).is_cuda
        package = {'version': model._version, 'audio_conf': model._audio_conf, 'labels': model._labels, 'state_dict': model.state_dict(), 'mixed_precision': model.mixed_precision, 'sample_rate': model._sample_rate, 'window_size': model._window_size}
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels

    @staticmethod
    def get_sample_rate(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._sample_rate if model_is_cuda else model._sample_rate

    @staticmethod
    def get_window_size(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._window_size if model_is_cuda else model._window_size

    @staticmethod
    def setAudioConfKey(model, key, value):
        model._audio_conf[key] = value
        return model

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._audio_conf if model_is_cuda else model._audio_conf

    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {'version': m._version}
        return meta

    def fuse_model(self):
        for m in self.modules():
            if type(m) == Cov1dBlock:
                torch.quantization.fuse_modules(m, ['conv1', 'batchNorm'], inplace=True)

    def convertTensorType(self, dtypeToUse=torch.float16):
        module = self._modules
        layermodules = module['conv1ds']
        convLayerPrefix = 'conv1d_{}'
        for i in range(0, 19):
            convLayer = getattr(layermodules, convLayerPrefix.format(i))
            modulesconv = convLayer._modules
            if 'batchNorm' in modulesconv:
                fusedLayer = self.fuse_conv_and_bn(modulesconv['conv1']._modules['0'], modulesconv['batchNorm'])
                setattr(convLayer, 'fusedLayer', fusedLayer)
        return

    def fuse_conv_and_bn(self, conv, bn):
        fusedconv = torch.nn.Conv1d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=0, dilation=conv.dilation, bias=True)
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)
        return fusedconv


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DistributedDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (InferenceBatchSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PCEN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_silversparro_wav2letter_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

