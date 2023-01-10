import sys
_module = sys.modules[__name__]
del sys
profiler = _module
setup = _module
tests = _module
conftest = _module
exceptions_test = _module
fixtures = _module
genotype = _module
models = _module
tmva_net = _module
gpu_test = _module
half_precision_test = _module
torchinfo_test = _module
torchinfo_xl_test = _module
torchinfo = _module
enums = _module
formatting = _module
layer_info = _module
model_statistics = _module
torchinfo = _module

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


from collections import namedtuple


from torch import nn


import math


from typing import Any


from typing import cast


from torch.nn import functional as F


from torch.nn.utils.rnn import pack_padded_sequence


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils import prune


import torchvision


from typing import Dict


from typing import Iterable


from typing import Sequence


from typing import Union


from torch.jit import ScriptModule


import warnings


from typing import Callable


from typing import Iterator


from typing import List


from typing import Mapping


from typing import Optional


from torch.utils.hooks import RemovableHandle


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=1, stride=1, padding=0, affine=True):
        super().__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, stride, kernel_size=3, padding=1, affine=True):
        super().__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class IdentityModel(nn.Module):
    """Identity Model."""

    def __init__(self) ->None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: Any) ->Any:
        return self.identity(x)


class Cell(nn.Module):

    def __init__(self, C_prev_prev, C_prev, C, reduction=False, reduction_prev=False):
        super().__init__()
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(1), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(1))
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C)
        self.preprocess1 = ReLUConvBN(C_prev, C)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            if name == 'skip_connect':
                op = IdentityModel() if stride == 1 else FactorizedReduce(C, C)
            elif name == 'sep_conv_3x3':
                op = SepConv(C, C, stride)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class GenotypeNetwork(nn.Module):

    def __init__(self, C=16, num_classes=10, layers=1, auxiliary=False):
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.0
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in (layers // 3, 2 * layers // 3):
                C_curr *= 2
            cell = Cell(C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    def forward(self, input_):
        s0 = s1 = self.stem(input_)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)


class LinearModel(nn.Module):
    """Linear Model."""

    def __init__(self) ->None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.layers(x)
        return x


class UninitializedParameterModel(nn.Module):
    """UninitializedParameter test"""

    def __init__(self) ->None:
        super().__init__()
        self.param: nn.Parameter | nn.UninitializedParameter = nn.UninitializedParameter()

    def init_param(self) ->None:
        self.param = nn.Parameter(torch.zeros(128))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        self.init_param()
        return x


class SingleInputNet(nn.Module):
    """Simple CNN model."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    """Model with multiple inputs containing different dtypes."""

    def __init__(self) ->None:
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)
        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) ->torch.Tensor:
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class ScalarNet(nn.Module):
    """Model that takes a scalar as a parameter."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)

    def forward(self, x: torch.Tensor, scalar: float) ->torch.Tensor:
        out = x
        if scalar == 5:
            out = self.conv1(out)
        else:
            out = self.conv2(out)
        return out


class LSTMNet(nn.Module):
    """Batch-first LSTM model."""

    def __init__(self, vocab_size: int=20, embed_dim: int=300, hidden_dim: int=512, num_layers: int=2) ->None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden


class RecursiveNet(nn.Module):
    """Model that uses a layer recursively in computation."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x: torch.Tensor, args1: Any=None, args2: Any=None) ->torch.Tensor:
        del args1, args2
        out = x
        for _ in range(3):
            out = self.conv1(out)
            out = self.conv1(out)
        return out


class CustomParameter(nn.Module):
    """Model that defines a custom parameter."""

    def __init__(self, input_size: int, attention_size: int) ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((attention_size, input_size)), True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        del x
        return self.weight


class ParameterListModel(nn.Module):
    """ParameterList of custom parameters."""

    def __init__(self) ->None:
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(weight) for weight in torch.Tensor(100, 300).split([100, 200], dim=1)])

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        _ = self.weights
        return x


class SiameseNets(nn.Module):
    """Model with MaxPool and ReLU layers."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.pooling = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) ->torch.Tensor:
        x1 = self.pooling(F.relu(self.conv1(x1)))
        x1 = self.pooling(F.relu(self.conv2(x1)))
        x1 = self.pooling(F.relu(self.conv3(x1)))
        x1 = self.pooling(F.relu(self.conv4(x1)))
        x2 = self.pooling(F.relu(self.conv1(x2)))
        x2 = self.pooling(F.relu(self.conv2(x2)))
        x2 = self.pooling(F.relu(self.conv3(x2)))
        x2 = self.pooling(F.relu(self.conv4(x2)))
        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        metric = torch.abs(x1 - x2)
        similarity = torch.sigmoid(self.fc2(self.dropout(metric)))
        return similarity


class FunctionalNet(nn.Module):
    """Model that uses many functional torch layers."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout2d(0.4)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ReturnDictLayer(nn.Module):
    """Model that returns a dict in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) ->dict[str, torch.Tensor]:
        activation_dict = {}
        x = self.conv1(x)
        activation_dict['conv1'] = x
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        activation_dict['conv2'] = x
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        activation_dict['fc1'] = x
        x = self.fc2(x)
        activation_dict['fc2'] = x
        x = F.log_softmax(x, dim=1)
        activation_dict['output'] = x
        return activation_dict


class ReturnDict(nn.Module):
    """Model that uses a ReturnDictLayer."""

    def __init__(self) ->None:
        super().__init__()
        self.return_dict = ReturnDictLayer()

    def forward(self, x: torch.Tensor, y: Any) ->dict[str, torch.Tensor]:
        del y
        activation_dict: dict[str, torch.Tensor] = self.return_dict(x)
        return activation_dict


class DictParameter(nn.Module):
    """Model that takes in a dict in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.constant = 5

    def forward(self, x: dict[int, torch.Tensor], scale_factor: int) ->torch.Tensor:
        return scale_factor * (x[256] + x[512][0]) * self.constant


class ModuleDictModel(nn.Module):
    """Model that uses a ModuleDict."""

    def __init__(self) ->None:
        super().__init__()
        self.choices = nn.ModuleDict({'conv': nn.Conv2d(10, 10, 3), 'pool': nn.MaxPool2d(3)})
        self.activations = nn.ModuleDict({'lrelu': nn.LeakyReLU(), 'prelu': nn.PReLU()})

    def forward(self, x: torch.Tensor, layer_type: str, activation_type: str) ->torch.Tensor:
        x = self.choices[layer_type](x)
        x = self.activations[activation_type](x)
        return x


class NamedTuple(nn.Module):
    """Model that takes in a NamedTuple as input."""
    Point = namedtuple('Point', ['x', 'y'])

    def forward(self, x: Any, y: Any, z: Any) ->Any:
        return self.Point(x, y).x + torch.ones(z.x)


class LayerWithRidiculouslyLongNameAndDoesntDoAnything(nn.Module):
    """Model with a very long name."""

    def __init__(self) ->None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: Any) ->Any:
        return self.identity(x)


class EdgeCaseModel(nn.Module):
    """Model that throws an exception when used."""

    def __init__(self, throw_error: bool=False, return_str: bool=False, return_class: bool=False) ->None:
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.return_class = return_class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = LayerWithRidiculouslyLongNameAndDoesntDoAnything()

    def forward(self, x: torch.Tensor) ->Any:
        x = self.conv1(x)
        x = self.model('string output' if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        if self.return_class:
            x = self.model(EdgeCaseModel)
        return x


class PackPaddedLSTM(nn.Module):
    """LSTM model with pack_padded layers."""

    def __init__(self, vocab_size: int=60, embedding_size: int=128, output_size: int=18, hidden_size: int=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=1)
        self.hidden2out = nn.Linear(self.hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch: torch.Tensor, lengths: torch.Tensor) ->torch.Tensor:
        hidden1 = torch.ones(1, batch.size(-1), self.hidden_size, device=batch.device)
        hidden2 = torch.ones(1, batch.size(-1), self.hidden_size, device=batch.device)
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        _, (ht, _) = self.lstm(packed_input, (hidden1, hidden2))
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = F.log_softmax(output, dim=1)
        return cast(torch.Tensor, output)


class ContainerChildModule(nn.Module):
    """Model using Sequential in different ways."""

    def __init__(self) ->None:
        super().__init__()
        self._sequential = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
        self._between = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self._sequential(x)
        out = self._between(out)
        for layer in self._sequential:
            out = layer(out)
        out = self._sequential(x)
        for layer in self._sequential:
            out = layer(out)
        return cast(torch.Tensor, out)


class ContainerModule(nn.Module):
    """Model using ModuleList."""

    def __init__(self) ->None:
        super().__init__()
        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(5, 5))
        self._layers.append(ContainerChildModule())
        self._layers.append(nn.Linear(5, 5))
        self._layers.append(None)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = x
        for layer in self._layers:
            if layer is not None:
                out = layer(out)
        return out


class EmptyModule(nn.Module):
    """A module that has no layers"""

    def __init__(self) ->None:
        super().__init__()
        self.parameter = torch.rand(3, 3, requires_grad=True)
        self.example_input_array = torch.zeros(1, 2, 3, 4, 5)

    def forward(self) ->dict[str, Any]:
        return {'loss': self.parameter.sum()}


class AutoEncoder(nn.Module):
    """Autoencoder module"""

    def __init__(self) ->None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.decode = nn.Sequential(nn.Conv2d(16, 3, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.encoder(x)
        unpooled_shape = x.size()
        x, indices = self.pool(x)
        x = self.unpool(x, indices=indices, output_size=unpooled_shape)
        x = self.decode(x)
        return x


class PartialJITModel(nn.Module):
    """Partial JIT model."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = torch.jit.script(nn.Linear(320, 50))
        self.fc2 = torch.jit.script(nn.Linear(50, 10))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MixedTrainableParameters(nn.Module):
    """Model with trainable and non-trainable parameters in the same layer."""

    def __init__(self) ->None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(10), requires_grad=True)
        self.b = nn.Parameter(torch.empty(10), requires_grad=False)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.w * x + self.b


class MixedTrainable(nn.Module):
    """Model with fully, partial and non trainable modules."""

    def __init__(self) ->None:
        super().__init__()
        self.fully_trainable = nn.Conv1d(1, 1, 1)
        self.partially_trainable = nn.Conv1d(1, 1, 1, bias=True)
        assert self.partially_trainable.bias is not None
        self.partially_trainable.bias.requires_grad = False
        self.non_trainable = nn.Conv1d(1, 1, 1, 1, bias=True)
        self.non_trainable.weight.requires_grad = False
        assert self.non_trainable.bias is not None
        self.non_trainable.bias.requires_grad = False
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.fully_trainable(x)
        x = self.partially_trainable(x)
        x = self.non_trainable(x)
        x = self.dropout(x)
        return x


class ReuseLinear(nn.Module):
    """Model that uses a reference to the same Linear layer over and over."""

    def __init__(self) ->None:
        super().__init__()
        linear = nn.Linear(10, 10)
        model = []
        for _ in range(4):
            model += [linear, nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class ReuseLinearExtended(nn.Module):
    """Model that uses a reference to the same Linear layer over and over."""

    def __init__(self) ->None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        model = []
        for _ in range(4):
            model += [self.linear, nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class ReuseReLU(nn.Module):
    """Model that uses a reference to the same ReLU layer over and over."""

    def __init__(self) ->None:
        super().__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(4, 1, kernel_size=1, padding=0), nn.BatchNorm2d(1), activation]
        for i in range(3):
            mult = 2 ** i
            model += [nn.Conv2d(mult, mult * 2, kernel_size=1, stride=2, padding=1), nn.BatchNorm2d(mult * 2), activation]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class PrunedLayerNameModel(nn.Module):
    """Model that defines parameters with _orig and _mask as suffixes."""

    def __init__(self, input_size: int, attention_size: int) ->None:
        super().__init__()
        self.weight_orig = nn.Parameter(torch.ones((attention_size, input_size)), True)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        del x
        return self.weight_orig


class FakePrunedLayerModel(nn.Module):
    """Model that defines parameters with _orig and _mask as suffixes."""

    def __init__(self, input_size: int, attention_size: int) ->None:
        super().__init__()
        self.weight_orig = nn.Parameter(torch.ones((attention_size, input_size)), True)
        self.weight_mask = nn.Parameter(torch.zeros((attention_size, input_size)), True)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        del x
        return self.weight_orig


class RegisterParameter(nn.Sequential):
    """A model with one parameter."""
    weights: list[torch.Tensor]

    def __init__(self, *blocks: nn.Module) ->None:
        super().__init__(*blocks)
        self.register_parameter('weights', nn.Parameter(torch.zeros(len(blocks))))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        for k, block in enumerate(self):
            x += self.weights[k] * block(x)
        return x


class InsideModel(nn.Module):
    """Module with a parameter and an inner module with a Parameter."""


    class Inside(nn.Module):
        """Inner module with a Parameter."""

        def __init__(self) ->None:
            super().__init__()
            self.l_1 = nn.Linear(1, 1)
            self.param_1 = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) ->torch.Tensor:
            return cast(torch.Tensor, self.l_1(x) * self.param_1)

    def __init__(self) ->None:
        super().__init__()
        self.l_0 = nn.Linear(2, 1)
        self.param_0 = nn.Parameter(torch.ones(2))
        self.inside = InsideModel.Inside()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self.inside(self.l_0(x)) * self.param_0)


class RecursiveWithMissingLayers(nn.Module):
    """
    Module with more complex recursive layers, which activates add_missing_layers.
    """

    def __init__(self) ->None:
        super().__init__()
        self.out_conv0 = nn.Conv2d(3, 8, 5, padding='same')
        self.out_bn0 = nn.BatchNorm2d(8)
        self.block0 = nn.ModuleDict()
        for i in range(1, 4):
            self.block0.add_module(f'in_conv{i}', nn.Conv2d(8, 8, 3, padding='same', dilation=2 ** i))
            self.block0.add_module(f'in_bn{i}', nn.BatchNorm2d(8))
        self.block1 = nn.ModuleDict()
        for i in range(4, 7):
            self.block1.add_module(f'in_conv{i}', nn.Conv2d(8, 8, 3, padding='same', dilation=2 ** (7 - i)))
            self.block1.add_module(f'in_bn{i}', nn.BatchNorm2d(8))
        self.out_conv7 = nn.Conv2d(8, 1, 1, padding='same')
        self.out_bn7 = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.out_conv0(x)
        x = torch.relu(self.out_bn0(x))
        for i in range(1, 4):
            x = self.block0[f'in_conv{i}'](x)
            x = torch.relu(self.block0[f'in_bn{i}'](x))
        for i in range(4, 7):
            x = self.block1[f'in_conv{i}'](x)
            x = torch.relu(self.block1[f'in_bn{i}'](x))
        x = self.out_conv7(x)
        x = torch.relu(self.out_bn7(x))
        return x


class CNNModuleList(nn.Module):
    """ModuleList with ConvLayers."""

    def __init__(self, conv_layer_cls: type[nn.Module]) ->None:
        super().__init__()
        self.ml = nn.ModuleList([conv_layer_cls() for i in range(5)])

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        for layer in self.ml:
            x = layer(x)
        return x


class ConvLayerA(nn.Module):
    """ConvLayer with the same module instantiation order in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return cast(torch.Tensor, out)


class ConvLayerB(nn.Module):
    """ConvLayer with a different module instantiation order in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(1, 1, 1)
        self.pool = nn.MaxPool1d(1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return cast(torch.Tensor, out)


class SimpleRNN(nn.Module):
    """Simple RNN"""

    def __init__(self, repeat_outside_loop: bool=False) ->None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.repeat_outside_loop = repeat_outside_loop
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.Tanh()
        self.projection = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, token_embedding: torch.Tensor) ->torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)
        if self.repeat_outside_loop:
            hx = self.projection(hx)
            hx = self.activation(hx)
        return hx


class DoubleConvBlock(nn.Module):
    """(2D conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """(3D conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil), nn.BatchNorm3d(out_ch), nn.LeakyReLU(inplace=True), nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil), nn.BatchNorm3d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """(2D conv => BN => LeakyReLU)"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False, mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type):
        super().__init__()
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, k_size=3, pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            x1_pad = F.pad(x1, (0, 1, 0, 0), 'constant', 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)
        x2 = self.double_conv_block2(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            x2_pad = F.pad(x2, (0, 1, 0, 0), 'constant', 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)
        x3 = self.single_conv_block1_1x1(x2_down)
        return x2_down, x3


class TMVANet_Encoder(nn.Module):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.ra_encoding_branch = EncodingBranch('range_angle')
        self.rd_encoding_branch = EncodingBranch('range_doppler')
        self.ad_encoding_branch = EncodingBranch('angle_doppler')
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ad_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x_rd, x_ra, x_ad, printshape=False):
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        return x3, x2_rd, x2_ad, x2_ra


class TMVANet_Decoder(nn.Module):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)
        self.rd_upconv1 = nn.ConvTranspose2d(384, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x3, x2_rd, x2_ad, x2_ra):
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)
        x4_rd = torch.cat((x2_rd, x3_rd, x2_ad), 1)
        x4_ra = torch.cat((x2_ra, x3_ra, x2_ad), 1)
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)
        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)
        return x9_rd, x9_ra


class TMVANet(nn.Module):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.encoder = TMVANet_Encoder(n_classes, n_frames)
        self.decoder = TMVANet_Decoder(n_classes, n_frames)

    def forward(self, x_rd, x_ra, x_ad):
        x3, x2_rd, x2_ad, x2_ra = self.encoder(x_rd, x_ra, x_ad)
        x9_rd, x9_ra = self.decoder(x3, x2_rd, x2_ad, x2_ra)
        return x9_rd, x9_ra


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPPBlock,
     lambda: ([], {'in_ch': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (AutoEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (CNNModuleList,
     lambda: ([], {'conv_layer_cls': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Cell,
     lambda: ([], {'C_prev_prev': 4, 'C_prev': 4, 'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k_size': 4, 'pad': 4, 'dil': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (ConvLayerA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (ConvLayerB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (CustomParameter,
     lambda: ([], {'input_size': 4, 'attention_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DictParameter,
     lambda: ([], {}),
     lambda: ([torch.rand([513, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DoubleConvBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k_size': 4, 'pad': 4, 'dil': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (EdgeCaseModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (EmptyModule,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FakePrunedLayerModel,
     lambda: ([], {'input_size': 4, 'attention_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GenotypeNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (IdentityModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerWithRidiculouslyLongNameAndDoesntDoAnything,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MixedTrainable,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     True),
    (MixedTrainableParameters,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 10])], {}),
     True),
    (ParameterListModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PrunedLayerNameModel,
     lambda: ([], {'input_size': 4, 'attention_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RecursiveNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (RecursiveWithMissingLayers,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RegisterParameter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReuseReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScalarNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64]), 0], {}),
     True),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UninitializedParameterModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_TylerYep_torchinfo(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

