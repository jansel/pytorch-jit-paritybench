import sys
_module = sys.modules[__name__]
del sys
caffemodel2pytorch = _module

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


import collections


import torch


import torch.nn as nn


import torch.nn.functional as F


from functools import reduce


class Blob(object):
    AssignmentAdapter = type('', (object,), dict(shape=property(lambda self:
        self.contents.shape), __setitem__=lambda self, indices, values:
        setattr(self, 'contents', values)))

    def __init__(self, data=None, diff=None, numpy=False):
        self.data_ = data if data is not None else Blob.AssignmentAdapter()
        self.diff_ = diff if diff is not None else Blob.AssignmentAdapter()
        self.shape_ = None
        self.numpy = numpy

    def reshape(self, *args):
        self.shape_ = args

    def count(self, *axis):
        return reduce(lambda x, y: x * y, self.shape_[slice(*(axis + [-1])[
            :2])])

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


TEST = 1


def convert_to_gpu_if_enabled(obj):
    return obj


class FunctionModule(nn.Module):

    def __init__(self, forward):
        super(FunctionModule, self).__init__()
        self.forward_func = forward

    def forward(self, *inputs):
        return self.forward_func(*inputs)


class Layer(torch.autograd.Function):

    def __init__(self, caffe_python_layer=None, caffe_input_variable_names=
        None, caffe_output_variable_names=None, caffe_propagate_down=None):
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
        outputs = tuple(convert_to_gpu_if_enabled(torch.from_numpy(v.data.
            contents.reshape(*v.shape))) for v in top)
        self.save_for_backward(*(inputs + outputs))
        return outputs

    def backward(self, grad_outputs):
        inputs, outputs = self.saved_tensors[:len(self.
            caffe_input_variable_names)], self.saved_tensors[len(self.
            caffe_input_variable_names):]
        bottom = [Blob(data=v.cpu().numpy()) for v in inputs]
        top = [Blob(data=output.cpu().numpy(), diff=grad_output.cpu().numpy
            ()) for grad_output, output in zip(grad_outputs, outputs)]
        self.caffe_python_layer.backward(top, self.caffe_propagate_down, bottom
            )
        return tuple(convert_to_gpu_if_enabled(torch.from_numpy(blob.diff.
            contents.reshape(*v.reshape))) if propagate_down else None for 
            v, propagate_down in zip(bottom, self.caffe_propagate_down))


class CaffePythonLayerModule(nn.Module):

    def __init__(self, caffe_python_layer, caffe_input_variable_names,
        caffe_output_variable_names, param_str):
        super(CaffePythonLayerModule, self).__init__()
        caffe_python_layer.param_str = param_str
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names

    def forward(self, *inputs):
        return Layer(self.caffe_python_layer, self.
            caffe_input_variable_names, self.caffe_output_variable_names)(*
            inputs)

    def __getattr__(self, name):
        return nn.Module.__getattr__(self, name) if name in dir(self
            ) else getattr(self.caffe_python_layer, name)


def first_or(param, key, default):
    return param[key] if isinstance(param.get(key), int) else (param.get(
        key, []) + [default])[0]


def init_weight_bias(self, weight=None, bias=None, requires_grad=[]):
    if weight is not None:
        self.weight = nn.Parameter(weight.type_as(self.weight),
            requires_grad=self.weight.requires_grad)
    if bias is not None:
        self.bias = nn.Parameter(bias.type_as(self.bias), requires_grad=
            self.bias.requires_grad)
    for name, requires_grad in zip(['weight', 'bias'], requires_grad):
        param, init = getattr(self, name), getattr(self, name + '_init')
        if init.get('type') == 'gaussian':
            nn.init.normal_(param, std=init['std'])
        elif init.get('type') == 'constant':
            nn.init.constant_(param, val=init['value'])
        param.requires_grad = requires_grad


class Convolution(nn.Conv2d):

    def __init__(self, param):
        super(Convolution, self).__init__(first_or(param, 'group', 1),
            param['num_output'], kernel_size=first_or(param, 'kernel_size',
            1), stride=first_or(param, 'stride', 1), padding=first_or(param,
            'pad', 0), dilation=first_or(param, 'dilation', 1), groups=
            first_or(param, 'group', 1))
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get('weight_filler', {}
            ), param.get('bias_filler', {})

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad
                ]
            super(Convolution, self).__init__(x.size(1), self.out_channels,
                kernel_size=self.kernel_size, stride=self.stride, padding=
                self.padding, dilation=self.dilation)
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(Convolution, self).forward(x)

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight, bias=bias.view(-1) if bias is not
            None else bias)
        self.in_channels = self.weight.size(1)


class InnerProduct(nn.Linear):

    def __init__(self, param):
        super(InnerProduct, self).__init__(1, param['num_output'])
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get('weight_filler', {}
            ), param.get('bias_filler', {})

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad
                ]
            super(InnerProduct, self).__init__(x.size(1), self.out_features)
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(InnerProduct, self).forward(x if x.size(-1) == self.
            in_features else x.view(len(x), -1))

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(self, weight=weight.view(weight.size(-2), weight.
            size(-1)) if weight is not None else None, bias=bias.view(-1) if
            bias is not None else None)
        self.in_features = self.weight.size(1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_vadimkantorov_caffemodel2pytorch(_paritybench_base):
    pass
