import sys
_module = sys.modules[__name__]
del sys
build = _module
setup = _module
torch2trt = _module
calibration = _module
AdaptiveAvgPool2d = _module
BatchNorm1d = _module
BatchNorm2d = _module
Conv1d = _module
Conv2d = _module
ConvTranspose2d = _module
Identity = _module
Linear = _module
LogSoftmax = _module
ReLU = _module
ReLU6 = _module
converters = _module
activation = _module
adaptive_avg_pool2d = _module
adaptive_max_pool2d = _module
add = _module
avg_pool2d = _module
cat = _module
chunk = _module
clamp = _module
div = _module
dummy_converters = _module
getitem = _module
identity = _module
instance_norm = _module
interpolate = _module
interpolate = _module
max = _module
max_pool2d = _module
mean = _module
min = _module
mul = _module
normalize = _module
pad = _module
permute = _module
pow = _module
prelu = _module
prod = _module
relu = _module
relu6 = _module
sigmoid = _module
softmax = _module
split = _module
sub = _module
sum = _module
tanh = _module
transpose = _module
unary = _module
view = _module
module_test = _module
test = _module
tests = _module
torchvision = _module
classification = _module
save_load = _module
segmentation = _module
torch2trt = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from string import Template


import torch


import torch.nn.functional as F


import torch.nn as nn


import torchvision


import time


import re


from copy import copy


import numpy as np


class Add(torch.nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


class IAdd(torch.nn.Module):

    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


class TorchAdd(torch.nn.Module):

    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


class RAddInt(torch.nn.Module):

    def __init__(self):
        super(RAddInt, self).__init__()

    def forward(self, x):
        return 1 + x


class RAddFloat(torch.nn.Module):

    def __init__(self):
        super(RAddFloat, self).__init__()

    def forward(self, x):
        return 1.0 + x


class TorchChunk(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.chunk(x, *self.args, **self.kwargs)


class TensorChunk(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.chunk(*self.args, **self.kwargs)


class TorchClampMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp_min(x, -0.1)


class TensorClampMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp_min(-0.1)


class TorchClampMax(torch.nn.Module):

    def forward(self, x):
        return torch.clamp_max(x, 0.1)


class TensorClampMax(torch.nn.Module):

    def forward(self, x):
        return x.clamp_max(0.1)


class TorchClamp(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, -0.1, 0.1)


class TensorClamp(torch.nn.Module):

    def forward(self, x):
        return x.clamp(-0.1, 0.1)


class TorchClampOptionMax(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, max=0.1)


class TorchClampOptionMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, min=-0.1)


class TorchClampOptionMaxMin(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, min=-0.1, max=0.1)


class TensorClampOptionMax(torch.nn.Module):

    def forward(self, x):
        return x.clamp(max=0.1)


class TensorClampOptionMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp(min=-0.1)


class TensorClampOptionMaxMin(torch.nn.Module):

    def forward(self, x):
        return x.clamp(min=-0.1, max=0.1)


class Div(torch.nn.Module):

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x, y):
        return x / y


class IDiv(torch.nn.Module):

    def __init__(self):
        super(IDiv, self).__init__()

    def forward(self, x, y):
        x /= y
        return x


class TorchDiv(torch.nn.Module):

    def __init__(self):
        super(TorchDiv, self).__init__()

    def forward(self, x, y):
        return torch.div(x, y)


class RDivInt(torch.nn.Module):

    def __init__(self):
        super(RDivInt, self).__init__()

    def forward(self, x):
        return 100 / x


class RDivFloat(torch.nn.Module):

    def __init__(self):
        super(RDivFloat, self).__init__()

    def forward(self, x):
        return 100.0 / x


class LambdaModule(torch.nn.Module):

    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Interpolate(torch.nn.Module):

    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, mode=self.mode, align_corners=self.align_corners)


class MaxElementwise(torch.nn.Module):

    def forward(self, x, y):
        return torch.max(x, y)


class Mean(torch.nn.Module):

    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(self.dim, self.keepdim)


class MinElementwise(torch.nn.Module):

    def forward(self, x, y):
        return torch.min(x, y)


class Mul(torch.nn.Module):

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y


class IMul(torch.nn.Module):

    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x


class TorchMul(torch.nn.Module):

    def __init__(self):
        super(TorchMul, self).__init__()

    def forward(self, x, y):
        return torch.mul(x, y)


class RMulInt(torch.nn.Module):

    def __init__(self):
        super(RMulInt, self).__init__()

    def forward(self, x):
        return 10 * x


class RMulFloat(torch.nn.Module):

    def __init__(self):
        super(RMulFloat, self).__init__()

    def forward(self, x):
        return 10.0 * x


class Normalize(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Normalize, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.nn.functional.normalize(x, *self.args, **self.kwargs)


class Pad(torch.nn.Module):

    def __init__(self, pad):
        super(Pad, self).__init__()
        self.pad = pad

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad)


class Permute(torch.nn.Module):

    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args).contiguous()


class Pow(torch.nn.Module):

    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, x, y):
        return x ** y


class TorchPow(torch.nn.Module):

    def __init__(self):
        super(TorchPow, self).__init__()

    def forward(self, x, y):
        return torch.pow(x, y)


class RpowInt(torch.nn.Module):

    def __init__(self):
        super(RpowInt, self).__init__()

    def forward(self, x):
        return 2 ** x


class RpowFloat(torch.nn.Module):

    def __init__(self):
        super(RpowFloat, self).__init__()

    def forward(self, x):
        return 2.0 ** x


class TorchSplit(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TorchSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.split(x, *self.args, **self.kwargs)


class TensorSplit(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TensorSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.split(*self.args, **self.kwargs)


class Sub(torch.nn.Module):

    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x, y):
        return x - y


class ISub(torch.nn.Module):

    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x


class TorchSub(torch.nn.Module):

    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


class RSubInt(torch.nn.Module):

    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


class RSubFloat(torch.nn.Module):

    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


class Transpose(torch.nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()


class UnaryModule(torch.nn.Module):

    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class View(torch.nn.Module):

    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


class ModelWrapper(torch.nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


class TRTModule(torch.nn.Module):

    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()
        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].data_ptr()
        self.context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)
        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Add,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Div,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ISub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaModule,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxElementwise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MinElementwise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Pow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RAddFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RAddInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDivFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDivInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMulFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMulInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RSubFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RSubInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RpowFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RpowInt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampOptionMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampOptionMaxMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TensorClampOptionMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampOptionMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampOptionMaxMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchClampOptionMin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchMul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchPow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TorchSub,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (UnaryModule,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVIDIA_AI_IOT_torch2trt(_paritybench_base):
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

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

