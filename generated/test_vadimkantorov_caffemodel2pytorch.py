import sys
_module = sys.modules[__name__]
del sys
caffemodel2pytorch = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import collections


import torch


import torch.nn as nn


import torch.nn.functional as F


from functools import reduce


class Blob(object):
    AssignmentAdapter = type('', (object,), dict(shape=property(lambda self: self.contents.shape), __setitem__=lambda self, indices, values: setattr(self, 'contents', values)))

    def __init__(self, data=None, diff=None, numpy=False):
        self.data_ = data if data is not None else Blob.AssignmentAdapter()
        self.diff_ = diff if diff is not None else Blob.AssignmentAdapter()
        self.shape_ = None
        self.numpy = numpy

    def reshape(self, *args):
        self.shape_ = args

    def count(self, *axis):
        return reduce(lambda x, y: x * y, self.shape_[slice(*(axis + [-1])[:2])])

    @property
    def data(self):
        if self.numpy and isinstance(self.data_, torch.autograd.Variable):
            self.data_ = self.data_.detach().cpu().numpy()
        return self.data_

    @property
    def diff(self):
        if self.numpy and isinstance(self.diff_, torch.autograd.Variable):
            self.diff_ = self.diff_.detach().cpu().numpy()
        return self.diff_

    @property
    def shape(self):
        return self.shape_ if self.shape_ is not None else self.data_.shape

    @property
    def num(self):
        return self.shape[0]

    @property
    def channels(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[2]

    @property
    def width(self):
        return self.shape[3]


def convert_to_gpu_if_enabled(obj):
    return obj


class Layer(torch.autograd.Function):

    def __init__(self, caffe_python_layer=None, caffe_input_variable_names=None, caffe_output_variable_names=None, caffe_propagate_down=None):
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names
        self.caffe_propagate_down = caffe_propagate_down

    def forward(self, *inputs):
        bottom = [Blob(data=v.cpu().numpy()) for v in inputs]
        top = [Blob() for name in self.caffe_output_variable_names]
        self.caffe_python_layer.setup(bottom, top)
        self.caffe_python_layer.setup = lambda *args: None
        self.caffe_python_layer.forward(bottom, top)
        outputs = tuple(convert_to_gpu_if_enabled(torch.from_numpy(v.data.contents.reshape(*v.shape))) for v in top)
        self.save_for_backward(*(inputs + outputs))
        return outputs

    def backward(self, grad_outputs):
        inputs, outputs = self.saved_tensors[:len(self.caffe_input_variable_names)], self.saved_tensors[len(self.caffe_input_variable_names):]
        bottom = [Blob(data=v.cpu().numpy()) for v in inputs]
        top = [Blob(data=output.cpu().numpy(), diff=grad_output.cpu().numpy()) for grad_output, output in zip(grad_outputs, outputs)]
        self.caffe_python_layer.backward(top, self.caffe_propagate_down, bottom)
        return tuple(convert_to_gpu_if_enabled(torch.from_numpy(blob.diff.contents.reshape(*v.reshape))) if propagate_down else None for v, propagate_down in zip(bottom, self.caffe_propagate_down))


class CaffePythonLayerModule(nn.Module):

    def __init__(self, caffe_python_layer, caffe_input_variable_names, caffe_output_variable_names, param_str):
        super(CaffePythonLayerModule, self).__init__()
        caffe_python_layer.param_str = param_str
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names

    def forward(self, *inputs):
        return Layer(self.caffe_python_layer, self.caffe_input_variable_names, self.caffe_output_variable_names)(*inputs)

    def __getattr__(self, name):
        return nn.Module.__getattr__(self, name) if name in dir(self) else getattr(self.caffe_python_layer, name)


class FunctionModule(nn.Module):

    def __init__(self, forward):
        super(FunctionModule, self).__init__()
        self.forward_func = forward

    def forward(self, *inputs):
        return self.forward_func(*inputs)


TEST = 1


def first_or(param, key, default):
    return param[key] if isinstance(param.get(key), int) else (param.get(key, []) + [default])[0]


def init_weight_bias(self, weight=None, bias=None, requires_grad=[]):
    if weight is not None:
        self.weight = nn.Parameter(weight.type_as(self.weight), requires_grad=self.weight.requires_grad)
    if bias is not None:
        self.bias = nn.Parameter(bias.type_as(self.bias), requires_grad=self.bias.requires_grad)
    for name, requires_grad in zip(['weight', 'bias'], requires_grad):
        param, init = getattr(self, name), getattr(self, name + '_init')
        if init.get('type') == 'gaussian':
            nn.init.normal_(param, std=init['std'])
        elif init.get('type') == 'constant':
            nn.init.constant_(param, val=init['value'])
        param.requires_grad = requires_grad


class Convolution(nn.Conv2d):

    def __init__(self, param):
        super(Convolution, self).__init__(first_or(param, 'group', 1), param['num_output'], kernel_size=first_or(param, 'kernel_size', 1), stride=first_or(param, 'stride', 1), padding=first_or(param, 'pad', 0), dilation=first_or(param, 'dilation', 1), groups=first_or(param, 'group', 1))
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get('weight_filler', {}), param.get('bias_filler', {})

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad]
            super(Convolution, self).__init__(x.size(1), self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(Convolution, self).forward(x)

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight, bias=bias.view(-1) if bias is not None else bias)
        self.in_channels = self.weight.size(1)


class InnerProduct(nn.Linear):

    def __init__(self, param):
        super(InnerProduct, self).__init__(1, param['num_output'])
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get('weight_filler', {}), param.get('bias_filler', {})

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad]
            super(InnerProduct, self).__init__(x.size(1), self.out_features)
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(InnerProduct, self).forward(x if x.size(-1) == self.in_features else x.view(len(x), -1))

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight.view(weight.size(-2), weight.size(-1)) if weight is not None else None, bias=bias.view(-1) if bias is not None else None)
        self.in_features = self.weight.size(1)


modules = dict(Convolution=lambda param: Convolution(param), InnerProduct=lambda param: InnerProduct(param), Pooling=lambda param: [nn.MaxPool2d, nn.AvgPool2d][param['pool']](kernel_size=first_or(param, 'kernel_size', 1), stride=first_or(param, 'stride', 1), padding=first_or(param, 'pad', 0)), Softmax=lambda param: nn.Softmax(dim=param.get('axis', -1)), ReLU=lambda param: nn.ReLU(), Dropout=lambda param: nn.Dropout(p=param['dropout_ratio']), Eltwise=lambda param: [torch.mul, torch.add, torch.max][param.get('operation', 1)], LRN=lambda param: nn.LocalResponseNorm(size=param['local_size'], alpha=param['alpha'], beta=param['beta']))


def to_dict(obj):
    return list(map(to_dict, obj)) if isinstance(obj, collections.Iterable) else {} if obj is None else {f.name: (converter(v) if f.label != FD.LABEL_REPEATED else list(map(converter, v))) for f, v in obj.ListFields() for converter in [{FD.TYPE_DOUBLE: float, FD.TYPE_SFIXED32: float, FD.TYPE_SFIXED64: float, FD.TYPE_SINT32: int, FD.TYPE_SINT64: int, FD.TYPE_FLOAT: float, FD.TYPE_ENUM: int, FD.TYPE_UINT32: int, FD.TYPE_INT64: int, FD.TYPE_UINT64: int, FD.TYPE_INT32: int, FD.TYPE_FIXED64: float, FD.TYPE_FIXED32: float, FD.TYPE_BOOL: bool, FD.TYPE_STRING: str, FD.TYPE_BYTES: lambda x: x.encode('string_escape'), FD.TYPE_MESSAGE: to_dict}[f.type]]}


class Net(nn.Module):

    def __init__(self, prototxt, *args, **kwargs):
        super(Net, self).__init__()
        caffe_proto = kwargs.pop('caffe_proto', None)
        weights = kwargs.pop('weights', None)
        phase = kwargs.pop('phase', None)
        weights = weights or (args + (None, None))[0]
        phase = phase or (args + (None, None))[1]
        self.net_param = initialize(caffe_proto).NetParameter()
        google.protobuf.text_format.Parse(open(prototxt).read(), self.net_param)
        for layer in (list(self.net_param.layer) + list(self.net_param.layers)):
            layer_type = layer.type if layer.type != 'Python' else layer.python_param.layer
            if isinstance(layer_type, int):
                layer_type = layer.LayerType.Name(layer_type)
            module_constructor = ([v for k, v in modules.items() if k.replace('_', '').upper() in [layer_type.replace('_', '').upper(), layer.name.replace('_', '').upper()]] + [None])[0]
            if module_constructor is not None:
                param = to_dict(([v for f, v in layer.ListFields() if f.name.endswith('_param')] + [None])[0])
                caffe_input_variable_names = list(layer.bottom)
                caffe_output_variable_names = list(layer.top)
                caffe_loss_weight = (list(layer.loss_weight) or [1.0 if layer_type.upper().endswith('LOSS') else 0.0]) * len(layer.top)
                caffe_propagate_down = list(getattr(layer, 'propagate_down', [])) or [True] * len(caffe_input_variable_names)
                caffe_optimization_params = to_dict(layer.param)
                param['inplace'] = len(caffe_input_variable_names) == 1 and caffe_input_variable_names == caffe_output_variable_names
                module = module_constructor(param)
                self.add_module(layer.name, module if isinstance(module, nn.Module) else CaffePythonLayerModule(module, caffe_input_variable_names, caffe_output_variable_names, param.get('param_str', '')) if type(module).__name__.endswith('Layer') else FunctionModule(module))
                module = getattr(self, layer.name)
                module.caffe_layer_name = layer.name
                module.caffe_layer_type = layer_type
                module.caffe_input_variable_names = caffe_input_variable_names
                module.caffe_output_variable_names = caffe_output_variable_names
                module.caffe_loss_weight = caffe_loss_weight
                module.caffe_propagate_down = caffe_propagate_down
                module.caffe_optimization_params = caffe_optimization_params
                for optim_param, p in zip(caffe_optimization_params, module.parameters()):
                    p.requires_grad = optim_param.get('lr_mult', 1) != 0
            else:
                None
        if weights is not None:
            self.copy_from(weights)
        self.blobs = collections.defaultdict(Blob)
        self.blob_loss_weights = {name: loss_weight for module in self.children() for name, loss_weight in zip(module.caffe_output_variable_names, module.caffe_loss_weight)}
        self.train(phase != TEST)
        convert_to_gpu_if_enabled(self)

    def forward(self, data=None, **variables):
        if data is not None:
            variables['data'] = data
        numpy = not all(map(torch.is_tensor, variables.values()))
        variables = {k: convert_to_gpu_if_enabled(torch.from_numpy(v.copy()) if numpy else v) for k, v in variables.items()}
        for module in [module for module in self.children() if not all(name in variables for name in module.caffe_output_variable_names)]:
            for name in module.caffe_input_variable_names:
                assert name in variables, 'Variable [{}] does not exist. Pass it as a keyword argument or provide a layer which produces it.'.format(name)
            inputs = [(variables[name] if propagate_down else variables[name].detach()) for name, propagate_down in zip(module.caffe_input_variable_names, module.caffe_propagate_down)]
            outputs = module(*inputs)
            if not isinstance(outputs, tuple):
                outputs = outputs,
            variables.update(dict(zip(module.caffe_output_variable_names, outputs)))
        self.blobs.update({k: Blob(data=v, numpy=numpy) for k, v in variables.items()})
        caffe_output_variable_names = set([name for module in self.children() for name in module.caffe_output_variable_names]) - set([name for module in self.children() for name in module.caffe_input_variable_names if name not in module.caffe_output_variable_names])
        return {k: (v.detach().cpu().numpy() if numpy else v) for k, v in variables.items() if k in caffe_output_variable_names}

    def copy_from(self, weights):
        try:
            import numpy
            state_dict = self.state_dict()
            for k, v in h5py.File(weights, 'r').items():
                if k in state_dict:
                    state_dict[k].resize_(v.shape).copy_(torch.from_numpy(numpy.array(v)))
            None
        except Exception as e:
            None
            bytes_weights = open(weights).read()
            bytes_parsed = self.net_param.ParseFromString(bytes_weights)
            if bytes_parsed != len(bytes_weights):
                None
            for layer in (list(self.net_param.layer) + list(self.net_param.layers)):
                module = getattr(self, layer.name, None)
                if module is None:
                    continue
                parameters = {name: convert_to_gpu_if_enabled(torch.FloatTensor(blob.data)).view(list(blob.shape.dim) if len(blob.shape.dim) > 0 else [blob.num, blob.channels, blob.height, blob.width]) for name, blob in zip(['weight', 'bias'], layer.blobs)}
                if len(parameters) > 0:
                    module.set_parameters(**parameters)
            None

    def save(self, weights):
        with h5py.File(weights, 'w') as h:
            for k, v in self.state_dict().items():
                h[k] = v.cpu().numpy()
        None

    @property
    def layers(self):
        return list(self.children())

