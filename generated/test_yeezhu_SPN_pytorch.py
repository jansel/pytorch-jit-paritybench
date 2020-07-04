import sys
_module = sys.modules[__name__]
del sys
SP_GoogLeNet = _module
evaluation = _module
experiment = _module
demo_voc2007 = _module
engine = _module
models = _module
util = _module
voc = _module
build = _module
setup = _module
spn = _module
SPG = _module
SSO = _module
functions = _module
SoftProposal = _module
SpatialSumOverMap = _module
modules = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Function


from torch.autograd import Variable


from copy import deepcopy


import time


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import numpy as np


import torchvision.models as models


import math


from torch.nn import Module


from scipy.ndimage import label


class CallLegacyModel(Function):

    @staticmethod
    def forward(ctx, model, x):
        if x.is_cuda:
            return model.cuda().forward(x)
        else:
            return model.float().forward(x)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError(
            'The backward call of LegacyModel is not implemented')


class LegacyModel(nn.Module):

    def __init__(self, model):
        super(LegacyModel, self).__init__()
        self.model = model

    def forward(self, x):
        return CallLegacyModel.apply(self.model, x)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.model))


def hook_spn(model):
    if not (hasattr(model, 'sp_hook') and hasattr(model, 'fc_hook')):
        model._training = model.training
        model.train(False)

        def _sp_hook(self, input, output):
            self.parent_modules[0].class_response_maps = output

        def _fc_hook(self, input, output):
            if hasattr(self.parent_modules[0], 'class_response_maps'):
                self.parent_modules[0].class_response_maps = F.conv2d(self.
                    parent_modules[0].class_response_maps, self.weight.
                    unsqueeze(-1).unsqueeze(-1))
            else:
                raise RuntimeError('The SPN is broken, please recreate it.')
        sp_layer = None
        fc_layer = None
        for mod in model.modules():
            if isinstance(mod, SoftProposal):
                sp_layer = mod
            elif isinstance(mod, torch.nn.Linear):
                fc_layer = mod
        if sp_layer is None or fc_layer is None:
            raise RuntimeError('Invalid SPN model')
        else:
            sp_layer.parent_modules = [model]
            fc_layer.parent_modules = [model]
            model.sp_hook = sp_layer.register_forward_hook(_sp_hook)
            model.fc_hook = fc_layer.register_forward_hook(_fc_hook)
    return model


def unhook_spn(model):
    try:
        model.sp_hook.remove()
        model.fc_hook.remove()
        del model.sp_hook
        del model.fc_hook
        model.train(model._training)
        return model
    except:
        raise RuntimeError("The model haven't been hooked!")


class SP_GoogLeNet(nn.Module):

    def __init__(self, state_dict='SP_GoogleNet_ImageNet.pt'):
        super(SP_GoogLeNet, self).__init__()
        state_dict = load_lua(state_dict)
        pretrained_model = state_dict[0]
        pretrained_model.evaluate()
        self.features = LegacyModel(pretrained_model)
        self.pooling = nn.Sequential()
        self.pooling.add_module('adconv', nn.Conv2d(832, 1024, kernel_size=
            3, stride=1, padding=1, groups=2, bias=True))
        self.pooling.add_module('maps', nn.ReLU())
        self.pooling.add_module('sp', SoftProposal(factor=2.1))
        self.pooling.add_module('sum', SpatialSumOverMap())
        self.pooling.adconv.weight.data.copy_(state_dict[1][0])
        self.pooling.adconv.bias.data.copy_(state_dict[1][1])
        self.classifier = nn.Linear(1024, 1000)
        self.classifier.weight.data.copy_(state_dict[2][0])
        self.classifier.bias.data.copy_(state_dict[2][1])
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def inference(self, mode=True):
        hook_spn(self) if mode else unhook_spn(self)
        return self


class SPNetWSL(nn.Module):

    def __init__(self, model, num_classes, num_maps, pooling):
        super(SPNetWSL, self).__init__()
        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.spatial_pooling = pooling
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_maps,
            num_classes))
        self.image_normalization_mean = [103.939, 116.779, 123.68]

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_pooling(x)
        x = x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FunctionBackend(object):

    def __init__(self, lib):
        self.backends = dict()
        self.parse_lib(lib)
        self.current_backend = None

    def __getattr__(self, name):
        func = self.backends[self.current_backend].get(name)
        if func is None:
            raise NotImplementedError(name)
        return func

    def set_type(self, input_type):
        if input_type != self.current_backend:
            if not input_type in self.backends.keys():
                raise NotImplementedError('{} is not supported'.format(
                    input_type))
            self.current_backend = input_type

    def parse_lib(self, lib):
        for func in dir(lib):
            if func.startswith('_'):
                continue
            match_obj = re.match('(\\w+)_(Float|Double)_(.+)', func)
            if match_obj:
                if match_obj.group(1).startswith('cu'):
                    backend = 'torch.cuda.{}Tensor'.format(match_obj.group(2))
                else:
                    backend = 'torch.{}Tensor'.format(match_obj.group(2))
                func_name = match_obj.group(3)
                if backend not in self.backends.keys():
                    self.backends[backend] = dict()
                self.backends[backend][func_name] = getattr(lib, func)


class SPGenerate(Function):

    def __init__(self, distanceMetric, transferMatrix, proposal,
        proposalBuffer, factor, couple):
        super(SPGenerate, self).__init__()
        self.backend = FunctionBackend(libspn)
        self.distanceMetric = distanceMetric
        self.transferMatrix = transferMatrix
        self.proposal = proposal
        self.proposalBuffer = proposalBuffer
        self.factor = factor
        self.couple = couple
        self.tolerance = 0.0001
        self.maxIteration = 20
        self.nBatch = 0
        self.mW = 0
        self.mH = 0

    def lazyInit(self, input):
        self.nBatch = input.size(0)
        self.mW = input.size(2)
        self.mH = input.size(3)
        self.N = self.mW * self.mH
        self.backend.SP_InitDistanceMetric(self.distanceMetric, self.factor,
            self.mW, self.mH, self.N)

    def forward(self, input):
        assert 'cuda' in input.type(
            ), 'CPU version is currently not implemented'
        self.backend.set_type(input.type())
        if self.nBatch != input.size(0) or self.mW != input.size(2
            ) or self.mH != input.size(3):
            self.lazyInit(input)
        output = input.new()
        self.backend.SP_Generate(input, self.distanceMetric, self.
            transferMatrix, self.proposal, self.proposalBuffer, self.
            tolerance, self.maxIteration)
        if self.couple:
            output.resize_(input.size())
            self.backend.SP_Couple(input, self.proposal, output)
        else:
            output.resize_(self.proposal.size())
            output = self.proposal.clone()
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = input.new()
        if self.couple:
            grad_input.resize_(input.size())
            self.backend.SP_Couple(grad_output, self.proposal, grad_input)
        else:
            grad_input.resize_(input.size()).zero_()
        return grad_input


def sp_generate(input, distanceMetric, transferMatrix, proposal,
    proposalBuffer, factor, couple=True):
    return SPGenerate(distanceMetric, transferMatrix, proposal,
        proposalBuffer, factor, couple)(input)


class SoftProposal(nn.Module):

    def __init__(self, couple=True, factor=None):
        super(SoftProposal, self).__init__()
        self.couple = couple
        self.factor = factor
        self.nBatch = 0
        self.mW = 0
        self.mH = 0

    def lazyInit(self, input):
        self.nBatch = input.size(0)
        self.mW = input.size(2)
        self.mH = input.size(3)
        self.N = self.mW * self.mH
        if self.factor is None:
            self.factor = 0.15 * self.N
        input_data = input.data
        self.distanceMetric = input_data.new()
        self.distanceMetric.resize_(self.N, self.N)
        self.transferMatrix = input_data.new()
        self.transferMatrix.resize_(self.N, self.N)
        self.proposal = input_data.new()
        self.proposal.resize_(self.nBatch, self.mW, self.mH)
        self.proposalBuffer = input_data.new()
        self.proposalBuffer.resize_(self.mW, self.mH)

    def forward(self, input):
        if self.nBatch != input.size(0) or self.mW != input.size(2
            ) or self.mH != input.size(3):
            self.lazyInit(input)
        return sp_generate(input, self.distanceMetric, self.transferMatrix,
            self.proposal, self.proposalBuffer, self.factor, self.couple)

    def __repr__(self):
        sp_config = '[couple={},factor={}]'.format(self.couple, self.factor)
        s = '{name}({sp_config})'
        return s.format(name=self.__class__.__name__, sp_config=sp_config,
            **self.__dict__)


class SpatialSumOverMapFunc(Function):

    def __init__(self):
        super(SpatialSumOverMapFunc, self).__init__()

    def forward(self, input):
        batch_size, num_channels, h, w = input.size()
        x = input.view(batch_size, num_channels, h * w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_channels)

    def backward(self, grad_output):
        input, = self.saved_tensors
        batch_size, num_channels, h, w = input.size()
        grad_input = grad_output.view(batch_size, num_channels, 1, 1).expand(
            batch_size, num_channels, h, w).contiguous()
        return grad_input


def spatial_sum_over_map(input):
    return SpatialSumOverMapFunc()(input)


class SpatialSumOverMap(Module):

    def __init__(self):
        super(SpatialSumOverMap, self).__init__()

    def forward(self, input):
        return spatial_sum_over_map(input)

    def __repr__(self):
        return self.__class__.__name__


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yeezhu_SPN_pytorch(_paritybench_base):
    pass
