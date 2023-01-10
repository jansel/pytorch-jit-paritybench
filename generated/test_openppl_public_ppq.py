import sys
_module = sys.modules[__name__]
del sys
ProgramEntrance = _module
IR = _module
base = _module
command = _module
graph = _module
opdef = _module
deploy = _module
morph = _module
processer = _module
quantize = _module
search = _module
training = _module
ppq = _module
api = _module
fsys = _module
interface = _module
setting = _module
core = _module
common = _module
config = _module
data = _module
defs = _module
ffi = _module
quant = _module
storage = _module
executor = _module
base = _module
op = _module
fp32_backend = _module
ppl_dsp_backend = _module
ppl_gpu_backend = _module
academic = _module
base = _module
cuda = _module
default = _module
dsp = _module
extension = _module
nxp = _module
onnx = _module
lib = _module
quant = _module
log = _module
logger = _module
parser = _module
ascend_export = _module
caffe = _module
caffe_export_utils = _module
caffe_graph_optim = _module
caffe_import_utils = _module
ppl_caffe_pb2 = _module
caffe_exporter = _module
caffe_parser = _module
native = _module
ncnn_exporter = _module
nxp_exporter = _module
onnx_exporter = _module
onnx_parser = _module
onnxruntime_exporter = _module
ppl = _module
qnn_exporter = _module
tengine_exporter = _module
tensorRT = _module
util = _module
core = _module
quantization = _module
algorithm = _module
equalization = _module
exprimental = _module
training = _module
analyse = _module
graphwise = _module
layerwise = _module
util = _module
measure = _module
cosine = _module
norm = _module
statistic = _module
observer = _module
floating = _module
order = _module
range = _module
optim = _module
baking = _module
calibration = _module
equalization = _module
exprimental = _module
legacy = _module
morph = _module
parameters = _module
refine = _module
ssd = _module
training = _module
qfunction = _module
floating = _module
linear = _module
AscendQuantizer = _module
DSPQuantizer = _module
FP8Quantizer = _module
FPGAQuantizer = _module
MetaxQuantizer = _module
MyQuantizer = _module
NCNNQuantizer = _module
NXPQuantizer = _module
ORTQuantizer = _module
OpenvinoQuantizer = _module
PPLQuantizer = _module
RKNNQuantizer = _module
TengineQuantizer = _module
TensorRTQuantizer = _module
quantizer = _module
base = _module
fp8_sample = _module
Imagenet = _module
imagenet_util = _module
Utilities = _module
evaluation_with_imagenet = _module
Example_Benchmark = _module
Example_Fp32 = _module
Example_PTQ = _module
Example_Benchmark = _module
Example_Fp32 = _module
Example_PTQ = _module
Example_QAT = _module
imagenet = _module
myquantizer = _module
trainer = _module
Example_PTQ = _module
Benchmark_with_onnx = _module
Example_Fp32 = _module
Example_PTQ = _module
Example_Profiling = _module
Example_QAT = _module
Example_Torch2trt = _module
create_engine = _module
generate_onnx = _module
lenet_int8 = _module
trt_infer = _module
analyse = _module
bestPractice = _module
calibration = _module
dequantize = _module
dispatch = _module
execute = _module
finetune = _module
fusion = _module
optimization = _module
quantize = _module
targetPlatform = _module
bert_sample = _module
bypass_nms = _module
custimize_quant_func = _module
custimized_quant = _module
dynamic_shape = _module
enable_cuda_kernel = _module
fp8_sample = _module
onnx_converter = _module
quantize_caffe_model = _module
quantize_dsp = _module
quantize_onnx_model = _module
quantize_torch_model = _module
yolo6_sample = _module
scheduler = _module
allin = _module
dispatchers = _module
perseus = _module
TensorRTUtil = _module
utils = _module
attribute = _module
ema = _module
fetch = _module
graph_editor = _module
round = _module
write_qparams_caffe2trt = _module
write_qparams_onnx2trt = _module
write_qparams_to_snpe_dlc = _module
setup = _module
tests = _module
test_activation_fusion = _module
test_block = _module
test_block_split = _module
test_cuda_kernel = _module
test_gemm_fusion = _module
test_gemm_split = _module
test_graph_api = _module
test_layerwise_equalization = _module
test_onnxruntime = _module
test_persus = _module
test_rounding = _module
test_system = _module
tmodel = _module
base = _module
testblocks = _module
blocks = _module
torchmodels = _module
tscheme = _module
placeholder = _module

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


from abc import abstractmethod


from collections import deque


from typing import Any


from typing import Dict


from typing import List


from typing import Text


from typing import Union


import torch


import numpy as np


from typing import Tuple


from typing import Callable


from typing import Iterable


from torch.utils.data import DataLoader


from enum import Enum


from numpy import dtype as np_type


from numpy import ndarray


from torch import Tensor


from torch import dtype as torch_type


from torch.cuda import empty_cache


from torch.cuda import synchronize


from torch.utils.cpp_extension import load


import time


from abc import ABCMeta


from functools import reduce


import torch.nn.functional as F


from torch import _VF


from copy import deepcopy


from typing import Optional


from typing import Type


import torch.nn as nn


from math import sqrt


import random


from random import randint


from torch.autograd import Function


from typing import Iterator


from collections import defaultdict


from functools import partial


import math


from numpy import dot


from numpy.linalg import norm


from math import ceil


from typing import Set


from abc import abstractproperty


from torchvision import models


import pandas as pd


import torch.utils.data


import torchvision.datasets as datasets


import torchvision.transforms as transforms


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Subset


import torchvision


import torchvision.models


import torch.profiler


from torch import nn


from torch.nn import functional as F


import logging


from math import floor


from math import log2


class NetworkFramework(Enum):
    PPL = 1
    ONNX = 2
    CAFFE = 3
    NXP = 4
    NATIVE = 5


COMPUTING_OP = {'Conv', 'Gemm', 'ConvTranspose', 'MatMul'}


class DataType(Enum):
    """
        DataType defines all PPQ internal data type and its enumeration value.
            ATTENTION: PPQ shares same data type enumeration value with Onnx.

        System maintainers and modifier are supposed to keep this corresponding.
        Cause OnnxExporter directly use this value to export PPQ graph towards Onnx.
    """
    INT4 = -1
    UINT4 = -2
    INT8 = 3
    UINT8 = 2
    INT16 = 5
    UINT16 = 4
    INT32 = 6
    UINT32 = 12
    INT64 = 7
    UINT64 = 13
    FP16 = 10
    FP32 = 1
    FP64 = 11
    BOOL = 9
    COMPLEX128 = 15
    COMPLEX64 = 14
    NONETYPE = 0

    @classmethod
    def convert_from_numpy(cls, dtype: np_type):
        numpy_converting_dict = {np_type('bool'): DataType.BOOL, np_type('uint8'): DataType.UINT8, np_type('int8'): DataType.INT8, np_type('int16'): DataType.INT16, np_type('int32'): DataType.INT32, np_type('int64'): DataType.INT64, np_type('float16'): DataType.FP16, np_type('float32'): DataType.FP32, np_type('float64'): DataType.FP64}
        if dtype not in numpy_converting_dict:
            raise TypeError(f'Numpy type {dtype} is not included in ppq now. please contact with system developer.')
        else:
            return numpy_converting_dict[dtype]

    @classmethod
    def convert_from_torch(cls, dtype: torch_type):
        torch_converting_dict = {torch.bool: DataType.BOOL, torch.uint8: DataType.UINT8, torch.int8: DataType.INT8, torch.int16: DataType.INT16, torch.int32: DataType.INT32, torch.int64: DataType.INT64, torch.float16: DataType.FP16, torch.float32: DataType.FP32, torch.float64: DataType.FP64}
        if dtype not in torch_converting_dict:
            raise TypeError(f'Torch dtype {dtype} is not included in ppq now. please contact with system developer.')
        else:
            return torch_converting_dict[dtype]

    @classmethod
    def to_numpy(cls, dtype) ->np_type:
        numpy_converting_dict = {DataType.BOOL: np_type('bool'), DataType.UINT8: np_type('uint8'), DataType.INT8: np_type('int8'), DataType.INT16: np_type('int16'), DataType.INT32: np_type('int32'), DataType.INT64: np_type('int64'), DataType.FP16: np_type('float16'), DataType.FP32: np_type('float32'), DataType.FP64: np_type('float64')}
        assert isinstance(dtype, DataType)
        return numpy_converting_dict[dtype]

    @classmethod
    def to_torch(cls, dtype) ->torch_type:
        torch_converting_dict = {DataType.BOOL: torch.bool, DataType.UINT8: torch.uint8, DataType.INT8: torch.int8, DataType.INT16: torch.int16, DataType.INT32: torch.int32, DataType.INT64: torch.int64, DataType.FP16: torch.float16, DataType.FP32: torch.float32, DataType.FP64: torch.float64}
        assert isinstance(dtype, DataType)
        return torch_converting_dict[dtype]


class TensorMeta:

    def __init__(self, dtype: DataType, shape: List[int], tensor_name: str=None) ->None:
        """TensorMeta structure described metadata of a tensor.

        Which includes tensor's data type and shape.
        TensorMeta is necessary to initialize quantization configuration and hooks,
        and is needed to compute the number of input channels.
        Args:
            dtype (DataType):
                A DataType enumeration described tensor type.
            shape (List[int]):
                A int list contains size of each dimensions.
            tensor_name (str, optional): Not yet used.
        """
        if not isinstance(dtype, DataType):
            raise TypeError(f'Can not create Tensor Meta with dtype {type(dtype)}, only ppq.core.DataType instance is acceptable here.')
        self.dtype = dtype
        self.shape = shape
        self.name = tensor_name

    @classmethod
    def parsing_from_numpy_ndarray(cls, numpy_array: ndarray, name: str=None):
        shape = list(numpy_array.shape)
        dtype = DataType.convert_from_numpy(numpy_array.dtype)
        return TensorMeta(dtype=dtype, shape=shape, tensor_name=name)

    @classmethod
    def parsing_from_torch_tensor(cls, torch_tensor: Tensor, name: str=None):
        if not isinstance(torch_tensor, Tensor):
            raise TypeError(f'Can not parse meta data for {type(torch_tensor)} instance, it should be torch.Tensor object.')
        shape = list(torch_tensor.shape)
        if not shape:
            shape = []
        dtype = DataType.convert_from_torch(torch_tensor.dtype)
        return TensorMeta(dtype=dtype, shape=shape, tensor_name=name)

    def create_tensor(self, device: str, fill_value: Any=0):
        return torch.Tensor(size=self.shape, device='cpu').fill_(fill_value).type(dtype=DataType.to_torch(self.dtype))

    def create_ndarray(self, fill_value: Any=0):
        return ndarray(shape=self.shape, dtype=DataType.to_numpy(self.dtype)).fill(fill_value)

    def __str__(self) ->str:
        return f'Tensor({self.name}) meta: dtype({self.dtype}), shape({self.shape})'

    def copy(self):
        if self.shape is not None:
            return TensorMeta(dtype=self.dtype, shape=self.shape.copy(), tensor_name=self.name)
        else:
            return TensorMeta(dtype=self.dtype, shape=None, tensor_name=self.name)


class OperationMeta:

    def __init__(self, input_metas: List[TensorMeta], output_metas: List[TensorMeta], operation_name: str, operation_type: str, executing_order: int) ->None:
        """OperationMeta structure describes all related tensor metadata of an
        operation.

        It naturally is a collection of TensorMeta.
        Take a look at TensorMeta to get more information.
        Args:
            input_metas (List[TensorMeta]):
                A collection contains all input tensors' metadata.
                ATTENTION: All parameters are considered as input in PPQ.
            output_metas (List[TensorMeta]):
                A collection contains all output tensors' metadata.
            operation_name (str): Not yet used.
            operation_type (str): Not yet used.
            executing_order (int): a int value represents the executing order of this operation.
                (order 0 means this operation is the first operation to be executed)
        """
        assert isinstance(input_metas, list), 'can only accept list object here.'
        assert isinstance(output_metas, list), 'can only accept list object here.'
        self.input_metas = input_metas
        self.output_metas = output_metas
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.executing_order = executing_order

    def __str__(self) ->str:
        return 'Inputs: '.join(str([_ for _ in self.input_metas])) + 'Outputs: '.join(str([_ for _ in self.input_metas]))

    @property
    def num_of_input(self):
        return len(self.input_metas)

    @property
    def num_of_output(self):
        return len(self.output_metas)

    def copy(self):
        return OperationMeta(input_metas=[meta.copy() for meta in self.input_metas], output_metas=[meta.copy() for meta in self.output_metas], operation_name=self.operation_name, operation_type=self.operation_type, executing_order=self.executing_order)


CAFFE_DOMAIN = 'ppq.caffe'


DEFAULT_OPSET_DOMAIN = 'ai.onnx'


DEFAULT_OPSET_VERSION = 11


ONNX_DOMAIN = 'ai.onnx'


class Opset:

    def __init__(self, domain: str=DEFAULT_OPSET_DOMAIN, version: int=DEFAULT_OPSET_VERSION) ->None:
        """Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers 
            to choose the right tools as their project evolves. 
        
        ONNX provides an open source format for AI models, both deep learning and traditional ML. 
        It defines an extensible computation graph model, as well as definitions of
            built-in operators and standard data types. 
        Currently we focus on the capabilities needed for inferencing (scoring).
        
        PPQ IR is built based on ONNX defination.

        Args:
            domain (str, optional): _description_. Defaults to DEFAULT_OPSET_DOMAIN.
            version (int, optional): _description_. Defaults to DEFAULT_OPSET_VERSION.
        """
        self.domain = domain
        self.version = version

    def is_onnx_v13(self):
        return self.domain == ONNX_DOMAIN and self.version == 13

    def is_onnx_v11(self):
        return self.domain == ONNX_DOMAIN and self.version == 11

    def onnx_opset_version(self) ->int:
        if self.domain == ONNX_DOMAIN:
            return self.version
        else:
            return -1

    def is_onnx(self):
        return self.domain == ONNX_DOMAIN

    def is_caffe(self):
        return self.domain == CAFFE_DOMAIN


class TargetPlatform(Enum):
    """TargetPlatform is a core abstraction of PPQ framework, it defines
    "platform" as an attribute of an operation. Platform attribute of an
    operation indicates where this operation is going to be deployed. This
    feature enables PPQ to simulate inter-device computing.

    Platform attribute also tells PPQ how to quantize an operation, and how to execute it.
        ATTENTION: Different platform might bring different behaviour of a same operation.
        ATTENTION: Operation which is assigned to an non-quantizible platform will never be quantized.

    There are several supported platforms for PPQ now,
        however you are supposed to be aware of some particular platforms here:

    SHAPE_OR_INDEX is a virtual platform, however it is an EXTREMELY IMPORTANT components in PPQ.
        Dispatch an operation to platform SHAPE_OR_INDEX means this operation is SOI-related,
        it processes a SOI tensor and gives a processed SOI, all calculation of this operation must be sent to CPU
            (or any platform capable for calculating this.) when deploy.

        An operation with SHAPE_OR_INDEX platform assigned will never be quantized regardless of its type.
        It is a crucial feature for quantizing network that contains SOI-related operation. (Shufflenet etc.)

        By default, PPQ automatically detects all SOI-related operations, and dispatch them to SHAPE_OR_INDEX platform.
        To understand how this feature works, see also: ppq.sche

    UNSPECIFIED is a virtual platform, all operations are sent to this platform once they were created.
        Quantizer then dispatches them towards desired platform through its quantization logic.
    """
    TRT_INT8 = 101
    TRT_FP8 = 105
    NCNN_INT8 = 102
    OPENVINO_INT8 = 103
    TENGINE_INT8 = 104
    ASC_INT8 = 106
    PPL_CUDA_INT8 = 201
    PPL_CUDA_INT4 = 202
    PPL_CUDA_FP16 = 203
    PPL_CUDA_MIX = 204
    PPL_DSP_INT8 = 301
    SNPE_INT8 = 302
    PPL_DSP_TI_INT8 = 303
    QNN_DSP_INT8 = 304
    HOST_INT8 = 401
    NXP_INT8 = 501
    FPGA_INT8 = 502
    RKNN_INT8 = 601
    METAX_INT8_C = 701
    METAX_INT8_T = 702
    HEXAGON_INT8 = 801
    GRAPHCORE_FP8 = 901
    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8 = 3
    INT8 = 4
    SOI = -1
    UNSPECIFIED = -2
    BOUNDARY = -3
    ONNX = -4
    CAFFE = -5
    NATIVE = -6
    ONNXRUNTIME = -7
    EXTENSION = -10086

    @classmethod
    def is_quantized_platform(cls, platform) ->bool:
        return platform in {cls.PPL_DSP_INT8, cls.PPL_DSP_TI_INT8, cls.QNN_DSP_INT8, cls.TRT_INT8, cls.NCNN_INT8, cls.NXP_INT8, cls.SNPE_INT8, cls.PPL_CUDA_INT8, cls.PPL_CUDA_INT4, cls.EXTENSION, cls.PPL_CUDA_MIX, cls.RKNN_INT8, cls.METAX_INT8_C, cls.METAX_INT8_T, cls.OPENVINO_INT8, cls.FPGA_INT8, cls.TENGINE_INT8, cls.FP8, cls.GRAPHCORE_FP8, cls.TRT_FP8, cls.ASC_INT8, cls.UNSPECIFIED, cls.INT8}


class OperationBase(metaclass=ABCMeta):

    def __init__(self, name: str, op_type: str, attributes: Dict[str, Any], opset=None, platform: TargetPlatform=TargetPlatform.UNSPECIFIED) ->None:
        self._name = name
        self._type = op_type
        self._attributes = attributes
        self._platform = platform
        self._meta = None
        self._input_vars = []
        self._output_vars = []
        self._detail = {}
        if opset is None:
            self._opset = Opset()
        else:
            self._opset = opset

    @abstractproperty
    def inputs(self) ->List[Any]:
        pass

    @abstractproperty
    def outputs(self) ->List[Any]:
        pass

    @abstractproperty
    def parameters(self) ->List[Any]:
        pass

    @property
    def name(self) ->str:
        return self._name

    @property
    def type(self) ->str:
        return self._type

    @type.setter
    def type(self, type: str):
        self._type = type

    @property
    def opset(self) ->Opset:
        return self._opset

    @opset.setter
    def opset(self, opset: Opset):
        self._opset = opset

    @property
    def attributes(self) ->Dict[str, Any]:
        return self._attributes

    @property
    def platform(self) ->TargetPlatform:
        return self._platform

    @platform.setter
    def platform(self, platform: TargetPlatform):
        self._platform = platform

    @property
    def num_of_input(self) ->int:
        return len(self._input_vars)

    @property
    def num_of_output(self) ->int:
        return len(self._output_vars)

    @property
    def meta_data(self) ->OperationMeta:
        return self._meta

    @meta_data.setter
    def meta_data(self, meta: OperationMeta) ->OperationMeta:
        self._meta = meta

    @property
    def inputs(self) ->List[object]:
        return self._input_vars

    @property
    def outputs(self) ->List[object]:
        return self._output_vars

    @property
    def is_computing_op(self) ->bool:
        return self.type in COMPUTING_OP

    def set_extension_attrib(self, attrib: str, value: Any):
        self._detail[attrib] = value

    @property
    def extension_attrib(self):
        return self._detail

    def __hash__(self) ->int:
        return self._name.__hash__()


class VLink:

    def __init__(self, in_idx: int, out_idx: int) ->None:
        if not isinstance(in_idx, int):
            raise TypeError(f'Can not create vlink with input_idx {in_idx}, only int value is acceptable here.')
        if not isinstance(out_idx, int):
            raise TypeError(f'Can not create vlink with output_idx {out_idx}, only int value is acceptable here.')
        self.in_idx = in_idx
        self.out_idx = out_idx


class OpSocket:

    def __init__(self, op: OperationBase, in_plat: List[TargetPlatform]=None, out_plat: List[TargetPlatform]=None, links: List[VLink]=None) ->None:
        self.in_plat = in_plat
        if in_plat is None:
            self.in_plat = [TargetPlatform.UNSPECIFIED for _ in range(op.num_of_input)]
        self.out_plat = out_plat
        if out_plat is None:
            self.out_plat = [TargetPlatform.UNSPECIFIED for _ in range(op.num_of_output)]
        self.links = links
        if self.links is None:
            self.links = []
            for i in range(op.num_of_input):
                for j in range(op.num_of_output):
                    self.links.append(VLink(i, j))


def DEFAULT_SOCKET_CREATOR(op: OperationBase) ->OpSocket:
    return OpSocket(op=op)


def CEHCK_TYPE(op: OperationBase, t: str):
    pass


STRICT_OPSET_CHECKING = False


def CHECK_OPSET(min_version_supported: int, max_version_supported: int, op: OperationBase, strict_check: bool=STRICT_OPSET_CHECKING):
    if not strict_check:
        return
    opset = op.opset
    if opset.domain != ONNX_DOMAIN:
        raise TypeError(f'Unrecognizable opset was found, can not generate dispatching scheme with op {op.name}, cause it might not be a standard onnx operation.')
    if opset.version > max_version_supported or opset.version < min_version_supported:
        raise TypeError(f'opset version is not supported, can not generate dispatching scheme with op {op.name}({op.type}), currently we support only [{min_version_supported, max_version_supported}], however {opset.version} was given.')


def Clip_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:
        
    Inputs (1 - 3)
        input (differentiable) : T
            Input tensor whose elements to be clipped
    
        min (optional, non-differentiable) : T
            Minimum value, under which element is replaced by min. 
            It must be a scalar(tensor of empty shape).
    
        max (optional, non-differentiable) : T
            Maximum value, above which element is replaced by max. 
            It must be a scalar(tensor of empty shape).
    
    Outputs
        output (differentiable) : T
            Output tensor with clipped input elements
    """
    CEHCK_TYPE(op=op, t='Clip')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def ConstantOfShape_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 9 - 9:
        
    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor. 
            If empty tensor is given, the output would be a scalar. All values must be >= 0.
    
    Outputs
        output : T2
            Output tensor of shape specified by 'input'.If attribute 'value' is specified, 
            the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, 
            the value in the output defaults to 0, and the datatype defaults to float32.
    """
    CEHCK_TYPE(op=op, t='ConstantOfShape')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[])


def Expand_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 8 - 13:
        
    Inputs
        input (differentiable) : T
    
    Input tensor
        shape (non-differentiable) : tensor(int64)
            A 1-D tensor indicates the shape you want to expand to, following the broadcast rule
    
    Outputs
        output (differentiable) : T
            Output tensor
    """
    CEHCK_TYPE(op=op, t='Expand')
    CHECK_OPSET(op=op, min_version_supported=8, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def GatherElements_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, with the same rank r as the input. 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of the same shape as indices.
    """
    CEHCK_TYPE(op=op, t='GatherElements')
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def GatherND_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.
    """
    CEHCK_TYPE(op=op, t='GatherND')
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Gather_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
        
        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q. 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.
    
    Outputs
        output (differentiable) : T
            Tensor of rank q + (r - 1).
    """
    CEHCK_TYPE(op=op, t='Gather')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    return OpSocket(op=op, in_plat=[TargetPlatform.UNSPECIFIED, TargetPlatform.FP32], links=[VLink(in_idx=0, out_idx=0)])


def GridSampler_Socket(op: OperationBase) ->OpSocket:
    """
    From MMCV
    """
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Logical_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 16:

    Inputs
        A (non-differentiable) : T
            First input operand for the logical operator.
    
        B (non-differentiable) : T
            Second input operand for the logical operator.
    
    Outputs
        C (non-differentiable) : T1
            Result tensor.
    """
    CHECK_OPSET(op=op, min_version_supported=12, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def NonMaxSuppression_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 10 - 13:
        
    Inputs (2 - 5)
        boxes : tensor(float)
            An input tensor with shape [num_batches, spatial_dimension, 4]. 
            The single box data format is indicated by center_point_box.
    
        scores : tensor(float)
            An input tensor with shape [num_batches, num_classes, spatial_dimension]
    
        max_output_boxes_per_class (optional) : tensor(int64)
            Integer representing the maximum number of boxes to be selected per batch per class. 
            It is a scalar. Default to 0, which means no output.
    
        iou_threshold (optional) : tensor(float)
            Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. 
            It is scalar. Value range [0, 1]. Default to 0.
    
        score_threshold (optional) : tensor(float)
            Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
            selected indices from the boxes tensor. [num_selected_indices, 3], 
            the selected index format is [batch_index, class_index, box_index].
    """
    CEHCK_TYPE(op=op, t='NonMaxSuppression')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[])


def NonZero_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 9 - 13:
        
    Inputs
        X (non-differentiable) : T
            input

    Outputs
        Y (non-differentiable) : tensor(int64)
            output
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def Onehot_Socket(op: OperationBase) ->OpSocket:
    """
    Inputs
        indices (non-differentiable) : T1

        depth (non-differentiable) : T2
            
        values (non-differentiable) : T3
    
    Outputs
        output (non-differentiable) : T3

    Args:
        op (Operation): _description_

    Returns:
        OpSocket: _description_
    """
    CHECK_OPSET(op=op, min_version_supported=12, max_version_supported=16)
    in_plat = [TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    out_plat = [TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def Pad_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs (2 - 3)
        data (differentiable) : T
            Input tensor.
            
        pads (non-differentiable) : tensor(int64)
            Tensor of integers indicating the number of padding elements to add or remove 
            (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. 
            `pads` should be a 1D tensor of shape [2 * input_rank]. 
            `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], 
            where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end, 
            the number of pad values added at the end of axis `i`.
            
        constant_value (optional, non-differentiable) : T
            (Optional) A scalar value to be used if the mode chosen is `constant`
            (by default it is 0, empty string or False).
        
    Outputs
        output (differentiable) : T
            Tensor after padding.
    """
    CEHCK_TYPE(op=op, t='Pad')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Range_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        start : T
            Scalar. First entry for the range of output values.

        limit : T
            Scalar. Exclusive upper limit for the range of output values.

        delta : T
            Scalar. Value to step by.

    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=13)
    in_plat = [TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[])


def Reshape_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 5 - 13:
    
    Inputs
        data (differentiable) : T
            An input tensor.
    
        shape (non-differentiable) : tensor(int64)
            Specified shape for output.
    
    Outputs
        reshaped (differentiable) : T
            Reshaped data.
    """
    CEHCK_TYPE(op=op, t='Reshape')
    CHECK_OPSET(op=op, min_version_supported=5, max_version_supported=13)
    return OpSocket(op=op, in_plat=[TargetPlatform.UNSPECIFIED, TargetPlatform.SOI], links=[VLink(in_idx=0, out_idx=0)])


def Resize_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 11 - 13:
    
    Inputs (1 - 4)
        X (differentiable) : T1
            N-D tensor
   
        roi (optional, non-differentiable) : T2
            1-D tensor given as [start1, ..., startN, end1, ..., endN], 
            where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. 
            It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    
        scales (optional, non-differentiable) : tensor(float)
            The scale array along each dimension. It takes value greater than 0. 
            If it's less than 1, it's sampling down, otherwise, it's upsampling. 
            The number of elements of 'scales' should be the same as the rank of input 'X'.
            One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified.
            If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.
    
        sizes (optional, non-differentiable) : tensor(int64)
            The size of the output tensor. The number of elements of 'sizes' should be the same as the rank of input 'X'.
            Only one of 'scales' and 'sizes' can be specified.
    
    Outputs
        Y (differentiable) : T1
            N-D tensor after resizing
    """
    CEHCK_TYPE(op=op, t='Resize')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=13, strict_check=True)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def RoiAlign_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 10 - 16:
        
    Inputs
        X : T1
            Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W), 
            where N is the batch size, C is the number of channels, 
            and H and W are the height and the width of the data.

        rois : T1
            RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4) 
            given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image. 
            Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.

        batch_indices : T2
            1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.

    Outputs
        Y : T1
            RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). 
            The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].
    """
    CEHCK_TYPE(op=op, t='RoiAlign')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def ScatterElements_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 11 - 16:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input). 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.UNSPECIFIED]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)])


def ScatterND_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 11 - 16:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
        
        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1.
        
        updates (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1.
    """
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.UNSPECIFIED]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)])


def Shape_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 16:
        
    Inputs
        data (non-differentiable) : T
            An input tensor.
    
    Outputs
        shape (non-differentiable) : T1
            Shape of the input tensor
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[])


def Slice_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 16:

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Split_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs (1 - 2)
        input (differentiable) : T
            The tensor to split
    
        split (optional, non-differentiable) : tensor(int64)
            Optional length of each output. Values should be >= 0.Sum of the values 
            must be equal to the dim value at 'axis' specified.
    
    Outputs (1 - ∞)
        outputs (variadic, differentiable) : T
            One or more outputs forming list of tensors after splitting
    """
    CEHCK_TYPE(op=op, t='Split')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Squeeze_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:

    Inputs (1 - 2)
        data (differentiable) : T
            Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to squeeze. 
            Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data).
    
    Outputs
        squeezed (differentiable) : T
            Reshaped tensor with same data as input.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Tile_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:
        
    Inputs
        input (differentiable) : T
            Input tensor of any shape.
    
    repeats (non-differentiable) : T1
        1D int64 tensor of the same length as input's dimension number, 
        includes numbers of repeated copies along input's dimensions.

    Outputs
        output (differentiable) : T
            Output tensor of the same dimensions and type as tensor input. 
            output_dim[i] = input_dim[i] * repeats[i]
    """
    CEHCK_TYPE(op=op, t='Tile')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Topk_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 11:
    
    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]
    
        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
    
    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
            containing top K values from the input tensor
    
        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
            containing the corresponding input tensor indices for the top K values.
    """
    CEHCK_TYPE(op=op, t='TopK')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=11)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    out_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], out_plat=out_plat, links=[VLink(in_idx=0, out_idx=0)])


def Unsqueeze_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 1 - 13:

    Inputs
        data (differentiable) : T
            Original tensor

        axes (non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to be inserted. 
            Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(expanded).

    Outputs
        expanded (differentiable) : T
            Reshaped tensor with same data as input.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


def Where_Socket(op: OperationBase) ->OpSocket:
    """
    From Opset 9 - 16:

    Inputs
        condition (non-differentiable) : B
            When True (nonzero), yield X, otherwise yield Y
    
        X (differentiable) : T
            values selected at indices where condition is True
        
        Y (differentiable) : T
            values selected at indices where condition is False
    
    Outputs
        output (differentiable) : T
            Tensor of shape equal to the broadcasted shape of condition, X, and Y.
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.FP32]
    return OpSocket(op=op, in_plat=in_plat[:op.num_of_input], links=[VLink(in_idx=0, out_idx=0)])


DEFAULT_SOCKET_TABLE = {'AdaptiveAvgPool2d': DEFAULT_SOCKET_CREATOR, 'Add': DEFAULT_SOCKET_CREATOR, 'ArgMax': DEFAULT_SOCKET_CREATOR, 'AveragePool': DEFAULT_SOCKET_CREATOR, 'BatchNormalization': DEFAULT_SOCKET_CREATOR, 'Cast': DEFAULT_SOCKET_CREATOR, 'Clip': Clip_Socket, 'Concat': DEFAULT_SOCKET_CREATOR, 'Constant': DEFAULT_SOCKET_CREATOR, 'ConstantOfShape': ConstantOfShape_Socket, 'Conv': DEFAULT_SOCKET_CREATOR, 'ConvTranspose': DEFAULT_SOCKET_CREATOR, 'Div': DEFAULT_SOCKET_CREATOR, 'Equal': Logical_Socket, 'Exp': DEFAULT_SOCKET_CREATOR, 'Expand': Expand_Socket, 'Flatten': DEFAULT_SOCKET_CREATOR, 'Gather': Gather_Socket, 'GatherElements': GatherElements_Socket, 'GatherND': GatherND_Socket, 'Gelu': DEFAULT_SOCKET_CREATOR, 'Gemm': DEFAULT_SOCKET_CREATOR, 'grid_sampler': GridSampler_Socket, 'GlobalAveragePool': DEFAULT_SOCKET_CREATOR, 'GlobalMaxPool': DEFAULT_SOCKET_CREATOR, 'Greater': Logical_Socket, 'LayerNorm': DEFAULT_SOCKET_CREATOR, 'LeakyRelu': DEFAULT_SOCKET_CREATOR, 'Less': Logical_Socket, 'LogSoftmax': DEFAULT_SOCKET_CREATOR, 'MatMul': DEFAULT_SOCKET_CREATOR, 'Max': DEFAULT_SOCKET_CREATOR, 'MaxPool': DEFAULT_SOCKET_CREATOR, 'Min': DEFAULT_SOCKET_CREATOR, 'Mul': DEFAULT_SOCKET_CREATOR, 'MultiHeadAttention': DEFAULT_SOCKET_CREATOR, 'NonMaxSuppression': NonMaxSuppression_Socket, 'NonZero': NonZero_Socket, 'Not': DEFAULT_SOCKET_CREATOR, 'Pad': Pad_Socket, 'PRelu': DEFAULT_SOCKET_CREATOR, 'Range': Range_Socket, 'ReduceL2': DEFAULT_SOCKET_CREATOR, 'ReduceMax': DEFAULT_SOCKET_CREATOR, 'ReduceMean': DEFAULT_SOCKET_CREATOR, 'ReduceSum': DEFAULT_SOCKET_CREATOR, 'Relu': DEFAULT_SOCKET_CREATOR, 'Reshape': Reshape_Socket, 'Resize': Resize_Socket, 'ScatterElements': ScatterElements_Socket, 'ScatterND': ScatterND_Socket, 'Shape': Shape_Socket, 'Sigmoid': DEFAULT_SOCKET_CREATOR, 'Slice': Slice_Socket, 'Softmax': DEFAULT_SOCKET_CREATOR, 'Softplus': DEFAULT_SOCKET_CREATOR, 'Split': Split_Socket, 'Squeeze': Squeeze_Socket, 'Sub': DEFAULT_SOCKET_CREATOR, 'Tile': Tile_Socket, 'TopK': Topk_Socket, 'Transpose': DEFAULT_SOCKET_CREATOR, 'Unsqueeze': Unsqueeze_Socket, 'Where': Where_Socket, 'Sqrt': DEFAULT_SOCKET_CREATOR, 'Log': DEFAULT_SOCKET_CREATOR, 'Floor': DEFAULT_SOCKET_CREATOR, 'RoiAlign': RoiAlign_Socket, 'MMCVRoiAlign': RoiAlign_Socket, 'SpaceToDepth': DEFAULT_SOCKET_CREATOR, 'DepthToSpace': DEFAULT_SOCKET_CREATOR, 'Scale': DEFAULT_SOCKET_CREATOR, 'Tanh': DEFAULT_SOCKET_CREATOR, 'Pow': DEFAULT_SOCKET_CREATOR, 'Crop': DEFAULT_SOCKET_CREATOR, 'ChannelShuffle': DEFAULT_SOCKET_CREATOR, 'InstanceNormalization': DEFAULT_SOCKET_CREATOR, 'Parameter': DEFAULT_SOCKET_CREATOR, 'Interp': DEFAULT_SOCKET_CREATOR, 'CaffeArgMax': DEFAULT_SOCKET_CREATOR, 'HardSigmoid': DEFAULT_SOCKET_CREATOR, 'HardSwish': DEFAULT_SOCKET_CREATOR, 'Neg': DEFAULT_SOCKET_CREATOR, 'GRU': DEFAULT_SOCKET_CREATOR, 'PPQDeviceSwitch': DEFAULT_SOCKET_CREATOR, 'Identity': DEFAULT_SOCKET_CREATOR, 'OneHot': Onehot_Socket, 'Reciprocal': DEFAULT_SOCKET_CREATOR, 'GreaterOrEqual': Logical_Socket, 'LessOrEqual': Logical_Socket, 'Xor': Logical_Socket, 'Or': Logical_Socket, 'And': Logical_Socket, 'Erf': DEFAULT_SOCKET_CREATOR}


LINEAR_ACTIVATIONS = {'Relu', 'Clip'}


class PPQ_GLOBAL_CONFIGURATION:

    def __init__(self) ->None:
        self.USING_CUDA_KERNEL = False
        self.NAME = 'PPL Quantization Tool'
        self.VERSION = '0.6.6'
        self.DUMP_VALUE_WHEN_EXPORT = True
        self.EXPORT_PPQ_INTERNAL_INFO = False
        self.PPQ_DEBUG = False


PPQ_CONFIG = PPQ_GLOBAL_CONFIGURATION()


def convert_any_to_numpy(x: Union[torch.Tensor, np.ndarray, int, float, list, tuple], accept_none: bool=True) ->np.ndarray:
    if x is None and accept_none:
        return None
    if x is None and not accept_none:
        raise ValueError('Trying to convert an empty value.')
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, int) or isinstance(x, float):
        return np.array([x])
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none:
            return None
        if x.numel() == 0 and not accept_none:
            raise ValueError('Trying to convert an empty value.')
        if x.numel() >= 1:
            return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as numpy type.')


def convert_any_to_torch_tensor(x: Union[torch.Tensor, np.ndarray, int, float, list, tuple], accept_none: bool=True, dtype: torch.dtype=None, device='cpu') ->torch.Tensor:
    if x is None and accept_none:
        return None
    if x is None and not accept_none:
        raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple):
        if all([(type(element) == int) for element in x]):
            if dtype is None:
                dtype = torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, int):
        if dtype is None:
            dtype = torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, float):
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        if device is not None:
            x = x
        return x
    elif isinstance(x, np.ndarray):
        if dtype is None:
            dtype = DataType.convert_from_numpy(x.dtype)
            dtype = DataType.to_torch(dtype)
        return torch.tensor(x.copy(), dtype=dtype, device=device)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as torch tensor.')


def ppq_warning(info: str):
    None


class Serializable:
    """An interface which means a class instance is binary serializable,
    nothing funny."""

    def __init__(self) ->None:
        self._export_value = PPQ_CONFIG.DUMP_VALUE_WHEN_EXPORT

    def __setstate__(self, state: dict):
        if not isinstance(state, dict):
            raise TypeError(f'PPQ Data Load Failure. Can not load data from {type(state)}, Your data might get damaged.')
        if '__version__' not in state or state['__version__'] != PPQ_CONFIG.VERSION:
            ppq_warning('You are loading an object created by PPQ with different version, it might cause some problems.')
        for key, value in state.items():
            self.__dict__[key] = value
            if isinstance(value, ValueState):
                self.__dict__[key] = value.unpack()
        return self

    def __getstate__(self) ->dict:
        attribute_dicts = self.__dict__
        attribute_dicts['__version__'] = PPQ_CONFIG.VERSION
        serialized = dict()
        for name, value in attribute_dicts.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                if self._export_value is False:
                    value = None
                serialized[name] = ValueState(value)
            else:
                serialized[name] = value
        return serialized


class Variable(Serializable):

    def __init__(self, name: str, value: Any=None, is_parameter: bool=False, dest_ops: List[OperationBase]=None, source_op: OperationBase=None, shape: List[int]=None, dtype: DataType=DataType.FP32) ->None:
        super().__init__()
        self._name = name
        self._value = None if value is None else value
        self._dest_ops = [] if dest_ops is None else dest_ops
        self._source_op = None if source_op is None else source_op
        self._is_parameter = is_parameter
        self.shape = shape
        self.dtype = dtype

    @property
    def is_parameter(self) ->bool:
        return self._is_parameter

    @is_parameter.setter
    def is_parameter(self, value: bool):
        self._is_parameter = value

    @property
    def name(self) ->str:
        return self._name

    @property
    def value(self) ->torch.Tensor:
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dest_ops(self) ->List[OperationBase]:
        return self._dest_ops

    @property
    def source_op(self) ->OperationBase:
        return self._source_op

    @source_op.setter
    def source_op(self, source_op: OperationBase) ->OperationBase:
        self._source_op = source_op

    @property
    def dest_idx(self) ->List[int]:
        _dest_idx = []
        for op in self.dest_ops:
            _dest_idx.append(op.inputs.index(self))
        return _dest_idx

    @property
    def src_idx(self) ->int:
        if self.source_op is not None:
            return self.source_op.outputs.index(self)
        else:
            return None

    @property
    def meta(self) ->TensorMeta:
        raise Exception('PPQ Variable.meta has been removed since 0.6.6, use Variable.shape, Variable.dtype instead.')

    def __hash__(self) ->int:
        return self._name.__hash__()

    def __str__(self) ->str:
        return f'Variable ({self._name})'

    def __getstate__(self) ->dict:
        state = super().__getstate__()
        state['_dest_ops'] = [op.name for op in self.dest_ops]
        state['_source_op'] = self.source_op.name if self.source_op is not None else None
        return state

    @property
    def shape(self) ->List[int]:
        """ Return tensor shape of this variable
        It is modifiable when current variable is not a paramter.
        """
        if self.value is not None:
            return self.value.shape
        return self._shape

    @shape.setter
    def shape(self, shape: List[Union[Text, int, None]]):
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        if isinstance(shape, torch.Size):
            shape = list(shape)
        if isinstance(shape, list):
            for element in shape:
                if type(element) not in {str, int}:
                    raise TypeError(f'Shape of a variable should only contains int or str. however {type(element)} was given.')
        self._shape = shape

    @property
    def dtype(self) ->DataType:
        """ Return tensor shape of this variable
        It is modifiable when current variable is not a paramter.
        """
        if self.value is not None:
            if isinstance(self.value, torch.Tensor):
                return DataType.convert_from_torch(self.value.dtype)
            if isinstance(self.value, np.ndarray):
                return DataType.convert_from_numpy(self.value.dtype)
        return self._dtype

    @dtype.setter
    def dtype(self, T: DataType):
        if isinstance(T, np.dtype):
            self._dtype = DataType.convert_from_numpy(T)
        elif isinstance(T, torch.dtype):
            self._dtype = DataType.convert_from_torch(T)
        elif isinstance(T, DataType):
            self._dtype = T
        else:
            raise TypeError(f'Invalid Dtype: {T} was given.')

    def copy(self, copy_value: bool=False):
        if not copy_value or self.value is None:
            cloned = Variable(name=self.name, value=self.value, is_parameter=self.is_parameter, shape=self.shape, dtype=self.dtype)
            return cloned
        if not isinstance(self.value, torch.Tensor):
            ppq_warning(f'You are requiring to copy variable {self.name}, however its value is not an instance of torch.Tensor, ppq will automaticall convert it to torch.Tensor now.')
            self.value = convert_any_to_torch_tensor(self.value)
        if isinstance(self.value, torch.Tensor):
            value = self.value.clone()
        return Variable(name=self.name, value=value, is_parameter=self.is_parameter, shape=self.shape, dtype=self.dtype)


class Operation(OperationBase, Serializable):

    def __init__(self, name: str, op_type: str, attributes: Dict[str, Any], platform: TargetPlatform=TargetPlatform.UNSPECIFIED, inputs: List[Variable]=None, outputs: List[Variable]=None, opset: Opset=None) ->None:
        OperationBase.__init__(self, name, op_type, attributes, platform=platform, opset=opset)
        Serializable.__init__(self)
        self._input_vars = [] if inputs is None else inputs
        self._output_vars = [] if outputs is None else outputs

    @property
    def socket(self) ->OpSocket:
        if self.type in DEFAULT_SOCKET_TABLE:
            return DEFAULT_SOCKET_TABLE[self.type](self)
        else:
            return DEFAULT_SOCKET_CREATOR(self)

    @property
    def inputs(self) ->List[Variable]:
        return self._input_vars

    @property
    def outputs(self) ->List[Variable]:
        return self._output_vars

    @property
    def parameters(self) ->List[Variable]:
        return [var for var in self.inputs if var.is_parameter]

    @property
    def num_of_parameters(self) ->int:
        return len(self.parameters)

    @property
    def is_linear_activation(self) ->bool:
        return self.type in LINEAR_ACTIVATIONS

    @property
    def num_of_parameter(self) ->int:
        return len([var for var in self.inputs if var.is_parameter])

    @property
    def is_boundary(self) ->bool:
        up_ops, down_ops = [], []
        for var in self.inputs:
            up_ops.append(var.source_op)
        for var in self.outputs:
            down_ops.extend(var.dest_ops)
        return all([(op is None) for op in up_ops]) or len(down_ops) == 0

    def __hash__(self) ->int:
        return self._name.__hash__()

    def __str__(self) ->str:
        return f'{self._name}({self.platform}) - inputs:{[var.name for var in self.inputs]}, outputs:{[var.name for var in self.outputs]}'

    def __getstate__(self) ->dict:
        state = super().__getstate__()
        state['_input_vars'] = [var.name for var in self.inputs]
        state['_output_vars'] = [var.name for var in self.outputs]
        return state

    def copy(self):
        clone = Operation(name=self.name, op_type=self.type, attributes=self.attributes.copy(), platform=self.platform, opset=self.opset)
        clone._detail = self._detail.copy()
        return clone


class BaseGraph(Serializable):
    """Graph is a PPQ Internal Represtation Data Structure.

    A computational graph is a directed graph where the nodes correspond to operations or variables.
    Variables can feed their value into operations, and operations can feed their output into other operations.
        This way, every node in the graph defines a function of the variables.

    The values that are fed into the nodes and come out of the nodes are called tensors,
        which is just a fancy word for a multi-dimensional array.
    Hence, it subsumes scalars, vectors and matrices as well as tensors of a higher rank.

    The computational graph created by PPQ contains quantization info as well as operations and variables.
        So to say it is a computational graph designed for quantization.

    All quantization related infos are stored within graph and its operations.
        See ppq.IR.quantize for more information.

    Args:
        Serializable ([type]): [description]
    """

    def __init__(self, name: str, built_from: NetworkFramework=NetworkFramework.NATIVE) ->None:
        super().__init__()
        self._operations = {}
        self._variables = {}
        self._graph_inputs = {}
        self._graph_outputs = {}
        self._name = name
        self._built_from = built_from
        self._detail = {}
        self._num_of_generated_var = 0
        self._num_of_generated_op = 0

    @property
    def operations(self) ->Dict[str, Operation]:
        return self._operations

    @property
    def variables(self) ->Dict[str, Variable]:
        return self._variables

    @property
    def inputs(self) ->Dict[str, Variable]:
        return self._graph_inputs

    @property
    def outputs(self) ->Dict[str, Variable]:
        return self._graph_outputs

    def parameters(self) ->List[torch.Tensor]:
        parameters = []
        for var in self.variables.values():
            if var.is_parameter:
                parameters.append(var.value)
        return parameters

    def set_extension_attrib(self, attrib: str, value: Any):
        self._detail[attrib] = value

    @property
    def extension_attrib(self):
        return self._detail

    def append_operation(self, operation: Operation):
        if not isinstance(operation, Operation):
            raise TypeError(f'You can only insert operations via this function, however {type(operation)} was given.')
        if not all([(var.name in self.variables) for var in operation.inputs + operation.outputs]):
            raise KeyError(f'Inserting Operation {operation} has a related variable which are not included in this graph yet, insert such variables before inserting this.')
        if operation.name in self.operations:
            raise KeyError(f'Duplicated Operation({operation}) was found, rename your Operation before inserting.')
        self.operations[operation.name] = operation

    def append_variable(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(f'You can only insert variable via this function, however {type(var)} was given.')
        if not all([(dest_op.name in self.operations) for dest_op in var.dest_ops]):
            raise KeyError(f'Inserting Variable {var} has a related Operation(dest_op) which are not included in this graph yet, insert such Operations before inserting this.')
        if var.name in self.variables:
            raise KeyError(f'Duplicated Variable({var}) was found, rename your Variable before inserting.')
        self.variables[var.name] = var

    def get_downstream_operations(self, operation: Operation) ->List[Operation]:
        if not isinstance(operation, Operation):
            raise TypeError(f'Expect an operation instance, however {type(operation)} is given.')
        if operation.name not in self.operations:
            raise KeyError(f'Operation {operation.name} not in current graph.')
        downstream_ops = []
        for output_var in operation.outputs:
            downstream_ops.extend(output_var.dest_ops)
        return downstream_ops

    def get_upstream_operations(self, operation: Operation) ->List[Operation]:
        if not isinstance(operation, Operation):
            raise TypeError(f'Expect an operation instance, however {type(operation)} is given.')
        if operation.name not in self.operations:
            raise KeyError(f'Operation {operation.name} not in current graph.')
        upstream_ops = []
        for input_var in operation.inputs:
            if input_var.source_op is not None:
                upstream_ops.append(input_var.source_op)
        return upstream_ops

    def topological_sort(self) ->List[Operation]:
        visited = {operation.name: (False) for operation in self.operations.values()}
        sort_ret, pop_list = [], deque()
        num_of_inputs = {operation.name: len(self.get_upstream_operations(operation)) for operation in self.operations.values()}
        for op_name, n_input in num_of_inputs.items():
            if n_input == 0:
                pop_list.append(op_name)
        for _ in range(len(visited)):
            if len(pop_list) == 0:
                break
            op_name = pop_list.popleft()
            op = self.operations[op_name]
            for post_op in self.get_downstream_operations(op):
                num_of_inputs[post_op.name] -= 1
                if num_of_inputs[post_op.name] == 0:
                    pop_list.append(post_op.name)
            visited[op.name] = True
            sort_ret.append(op)
        if all(visited.values()):
            return sort_ret
        else:
            raise RuntimeError('Topological Sort failed. Some operation can not be sorted (might due to circular reference).\n'.join(str(self.operations[op_name]) + '\n' for op_name in visited if visited[op_name] == False))

    def insert_op_on_var(self, inserting_op: Operation, var: str):
        """Insert one operation to current graph. Inserting operation will
        replace var.dest_ops and automatically connect to inserting_op.

        Before insertion:
            op1 -> var -> op2

        After insertion:
            op1 -> var -> inserting_op -> link_var(generated) -> op2

        ATTENTION: Inserting operation must be an empty operation with no input and output variables linked to it.

        Args:
            inserting_op (Operation): [description]
            upstream_op (Operation): [description]
            downstream_op (Operation): [description]
        """
        if not isinstance(var, str):
            raise TypeError(f'Needs a variable name(str) here, however {type(var)} was given')
        if var not in self.variables:
            raise KeyError(f'Can not inserting operation at variable {var}, variable not found.')
        if len(inserting_op.inputs) != 0 or len(inserting_op.outputs) != 0:
            raise PermissionError('Can only insert operation with no input and output variables.')
        variable = self.variables[var]
        if inserting_op.name not in self.operations.keys():
            self.append_operation(inserting_op)
        link_var = self.create_variable(name=None, value=None, is_parameter=False, dest_ops=variable.dest_ops.copy(), source_op=inserting_op)
        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)
        variable.dest_ops.clear()
        variable.dest_ops.append(inserting_op)
        for op in link_var.dest_ops:
            op.inputs[op.inputs.index(variable)] = link_var
        if var in self.outputs:
            self.outputs.pop(var)
            self.outputs[link_var.name] = link_var

    def insert_op_between_ops(self, inserting_op: Operation, up_op: Operation, down_op: Operation):
        """Insert one operation to current graph. Inserting operation will just
        between up_op and down_op.

        Example1(Insert Conv3 between Conv1 and Conv2):
            Before insertion: Conv1 -- Conv2
            After insertion:  Conv1 -- Conv3 -- Conv1

        Example2(Insert Conv3 between Conv1 and Conv2):

            Before insertion: Conv1 ----- Conv2
                                      |
                                      --- Conv4

            After insertion:  Conv1 ----- Conv3 -- Conv2
                                      |
                                      --- Conv4

        ATTENTION: Inserting operation must be an empty operation with no input and output variables linked to it.

        Args:
            inserting_op (Operation): [description]
            up_op (Operation): [description]
            down_op (Operation): [description]
        """
        if up_op.name not in self.operations:
            raise KeyError(f'Can not inserting operation behind {up_op.name}, operation not found.')
        if down_op.name not in self.operations:
            raise KeyError(f'Can not inserting operation behind {down_op.name}, operation not found.')
        if down_op not in self.get_downstream_operations(up_op):
            raise PermissionError(f'operation {up_op.name} and {down_op.name} are not linked, there is no way to insert an op between them.')
        if len(inserting_op.inputs) != 0 or len(inserting_op.outputs) != 0:
            raise PermissionError('Can only insert operation with no input and output variables.')
        variables = []
        for var in down_op.inputs:
            if var.source_op == up_op:
                variables.append(var)
        assert len(variables) == 1, f'Can not insert operation between {up_op.name} and {down_op.name}, graph is too complex.'
        [variable] = variables
        if inserting_op.name not in self.operations.keys():
            self.append_operation(inserting_op)
        link_var = self.create_variable(name=None, value=None, is_parameter=False, dest_ops=[down_op], source_op=inserting_op)
        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)
        assert isinstance(variable, Variable)
        variable.dest_ops[variable.dest_ops.index(down_op)] = inserting_op
        down_op.inputs[down_op.inputs.index(variable)] = link_var

    def insert_op_between_var_and_op(self, inserting_op: Operation, up_var: Variable, down_op: Operation):
        """Insert one operation to current graph. Inserting operation will just
        between up_var and down_op.

        ATTENTION: Inserting operation must be an empty operation with no input and output variables linked to it.

        Args:
            inserting_op (Operation): [description]
            up_op (Operation): [description]
            down_op (Operation): [description]
        """
        if up_var.name not in self.variables:
            raise KeyError(f'Can not inserting operation behind {up_var.name}, variable not found.')
        if down_op.name not in self.operations:
            raise KeyError(f'Can not inserting operation behind {down_op.name}, operation not found.')
        if down_op.name not in [op.name for op in up_var.dest_ops]:
            raise PermissionError(f'variable {up_var.name} and {down_op.name} are not linked, there is no way to insert an op between them.')
        if len(inserting_op.inputs) != 0 or len(inserting_op.outputs) != 0:
            raise PermissionError('Can only insert operation with no input and output variables.')
        variables = []
        for var in down_op.inputs:
            if var == up_var:
                variables.append(var)
        assert len(variables) == 1, f'Can not insert operation between {var.name} and {down_op.name}, graph is too complex.'
        if inserting_op.name not in self.operations.keys():
            self.append_operation(inserting_op)
        link_var = self.create_variable(name=None, value=None, is_parameter=False, dest_ops=[down_op], source_op=inserting_op)
        inserting_op.inputs.append(up_var)
        inserting_op.outputs.append(link_var)
        up_var.dest_ops[up_var.dest_ops.index(down_op)] = inserting_op
        down_op.inputs[down_op.inputs.index(up_var)] = link_var

    def create_link_with_op(self, variable: Variable, upstream_op: Operation, downstream_op: Operation):
        """Create a link with given variable from upstream_op to downstream_op
        variable will be appended to upstream_op's output and downstream_op's
        input given variable must have empty source_op or its source_op ==
        upstream_op.

        Sometime you may want to link a single upstream_op to many downstream_ops with a same variable,
            you are supposed to invoke this function for each downstream_op then.

        You can set upstream_op = None if your variable is a parameter variable.

        Example:
            create_link_with_op(var1, op1, op2)
            create_link_with_op(var1, op1, op3)

        Will makes:
                  --> op2
            op1 --|
                  --> op3

        Args:
            link_variable (Variable): _description_
            upstream_op (Operation): _description_
            downstream_op (Operation): _description_
        """
        if variable.name not in self.variables:
            raise KeyError(f'Can not find your variable {variable.name} in current graph.')
        if upstream_op is not None and upstream_op.name not in self.operations:
            raise KeyError(f'Can not find your operation {upstream_op.name} in current graph.')
        if downstream_op is not None and downstream_op.name not in self.operations:
            raise KeyError(f'Can not find your operation {downstream_op.name} in current graph.')
        if variable.source_op is None:
            variable.source_op = upstream_op
        if variable.source_op != upstream_op:
            raise PermissionError(f'Can not create link with variable {variable}, cause its source operations != {upstream_op}')
        if upstream_op is not None and variable not in upstream_op.outputs:
            upstream_op.outputs.append(variable)
        if downstream_op is None:
            return
        if downstream_op is not None and variable not in downstream_op.inputs:
            variable.dest_ops.append(downstream_op)
            downstream_op.inputs.append(variable)
        else:
            variable.dest_ops.append(downstream_op)
            downstream_op.inputs.append(variable)
            ppq_warning(f'You are trying to link variable with operation, however Variable {variable.name} has already linked with downstream op {downstream_op.name}')

    def create_link_with_var(self, upstream_variable: Variable, downstream_variable: Variable):
        """connect upstream_variable.source_op with
        downstream_variable.dest_ops, downstream variable will be eliminated by
        this function.

        downstream_variable must have None as its source_op.

        Args:
            upstream_variable (_type_): _description_
            downstream_variable (_type_): _description_
        """
        if downstream_variable.source_op is not None:
            raise PermissionError(f'Can not create link with variable {upstream_variable.name} & {downstream_variable.name}, Cause downstream variable has a non-empty source op')
        dest_ops = downstream_variable.dest_ops
        for dest_op in dest_ops:
            dest_op.inputs[dest_op.inputs.index(downstream_variable)] = upstream_variable
            upstream_variable.dest_ops.append(dest_op)
        downstream_variable.dest_ops.clear()
        self.remove_variable(downstream_variable)
        return self

    def remove_operation(self, removing_op: Operation, keep_coherence: bool=False):
        """Remove operation from graph, this function will unlink removing
        operation from current graph, pop it from graph.operations, and remove
        it from all its input and output variables.

        Parameters of this removing operations will be removed from graph by this function, without warning.

        Args:
            removing_op (Operation): [description]
            
            keep_coherence (bool): if keep_coherence = True, 
                PPQ will link downstream operations of removing op to the upstream operation.
        """
        if removing_op.name not in self.operations:
            raise KeyError(f'Can not remove operation {removing_op.name}, operation not found.')
        for parameter in removing_op.inputs.copy():
            if keep_coherence and removing_op.type in {'Constant', 'Identity'}:
                break
            if parameter.is_parameter:
                parameter.dest_ops.clear()
                parameter.value = None
                removing_op.inputs.remove(parameter)
                self.variables.pop(parameter.name)
        if not keep_coherence:
            for output_var in removing_op.outputs:
                output_var.source_op = None
            removing_op.outputs.clear()
            for input_var in removing_op.inputs:
                if removing_op in input_var.dest_ops:
                    input_var.dest_ops.remove(removing_op)
            removing_op.inputs.clear()
        else:
            if removing_op.num_of_input != 1:
                raise PermissionError(f'Can not remove operation {removing_op.name} with keep_coherence = True, operation must has exactly 1 input variable.')
            if removing_op.num_of_output != 1:
                raise PermissionError(f'Can not remove operation {removing_op.name} with keep_coherence = True, operation must has exactly 1 output variable.')
            input_var = removing_op.inputs[0]
            removing_var = removing_op.outputs[0]
            dest_ops = removing_var.dest_ops
            is_graph_output = removing_var.name in self.outputs
            for op in dest_ops:
                op.inputs[op.inputs.index(removing_var)] = input_var
                input_var.dest_ops.append(op)
            removing_var.dest_ops.clear()
            removing_var.source_op = None
            input_var.dest_ops.remove(removing_op)
            self.remove_variable(removing_var)
            if is_graph_output:
                self.mark_variable_as_graph_output(input_var)
        self.operations.pop(removing_op.name)
        return self

    def remove_variable(self, removing_var: Variable):
        """Remove variable from graph, this function will unlink removing
        variable from current graph, pop it from graph.variables, and remove it
        from its source op and dest ops.

        Args:
            removing_var (Variable): [description]
        """
        if removing_var.name not in self.variables:
            raise KeyError(f'Can not remove variable {removing_var.name}, variable not found.')
        source_op = removing_var.source_op
        if source_op is not None:
            assert isinstance(source_op, Operation), f'Can not remove variable {removing_var.name}, it links to a unexpected source operation.'
            if removing_var in source_op.outputs:
                source_op.outputs.remove(removing_var)
            removing_var.source_op = None
        for dest_op in removing_var.dest_ops:
            assert isinstance(dest_op, Operation), f'Can not remove variable {removing_var.name}, it links to a unexpected dest operation.'
            if removing_var in dest_op.inputs:
                dest_op.inputs.remove(removing_var)
        removing_var.dest_ops.clear()
        if removing_var.name in self.outputs:
            self.outputs.pop(removing_var.name)
        if removing_var.name in self.inputs:
            self.inputs.pop(removing_var.name)
        self.variables.pop(removing_var.name)
        return self

    def create_operation(self, op_type: str, name: str=None, attributes: Dict[str, Any]=None, platform: TargetPlatform=TargetPlatform.UNSPECIFIED, inputs: List[Variable]=None, outputs: List[Variable]=None, **kwargs) ->Operation:
        """Create an operation and attach it it current graph. op_type is
        mandatory here, however op_name is not required. PPQ will automatically
        generates a name for your operation:
        PPQ_Operation_{self._num_of_generated_op}.

        Use this function carefully, cause once your network is quantized,
            simply create an operation via this function might cause unexpected error.
        Beawre that operation created by this function has no meta data and quantization info,
            which is needed to export and executing your graph.

        Do not set inputs and outputs via this function,
            to link your operation with others, use graph.create_link_with_var instead.

        Args:
            op_type (str): _description_
            name (str, optional): _description_. Defaults to None.
            attributes (Dict[str, Any], optional): _description_. Defaults to None.
            platform (TargetPlatform, optional): _description_. Defaults to TargetPlatform.UNSPECIFIED.
            inputs (List[Variable], optional): _description_. Defaults to None.
            outputs (List[Variable], optional): _description_. Defaults to None.

        Returns:
            Operation: _description_
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        if name is None:
            name = f'PPQ_Operation_{self._num_of_generated_op}'
            self._num_of_generated_op += 1
        if not isinstance(inputs, list):
            raise TypeError(f'A list of input variable is required for creating operation, however {type(inputs)} was given')
        if attributes is None:
            attributes = {}
        created = Operation(name=name, op_type=op_type, attributes=attributes, platform=platform, inputs=inputs, outputs=outputs)
        self.append_operation(created)
        for item in inputs:
            if not isinstance(item, Variable):
                raise TypeError(f'A list contains variables is required for creating operation, however there is a {type(item)} in your input list.')
            item.dest_ops.append(created)
        if not isinstance(outputs, list):
            raise TypeError(f'A list of output variable is required for creating operation, however {type(inputs)} was given')
        for item in outputs:
            if not isinstance(item, Variable):
                raise TypeError(f'A list contains variables is required for creating operation, however there is a {type(item)} in your output list.')
            item.source_op = created
        return created

    def create_variable(self, name: str=None, value: Any=None, is_parameter: bool=False, dest_ops: List[OperationBase]=None, source_op: OperationBase=None, **kwargs) ->Variable:
        """Create a variable and attach it it current graph. PPQ will
        automatically generates a name for your variable:
        PPQ_Variable_{self._num_of_generated_op}.

        Use this function carefully, cause once your network is quantized,
            simply create an variable via this function might cause unexpected error.
        You'd better invoke this function before running your quantizer.

        Do not set dest_ops and source_op via this function,
            to link this variable with others, use graph.create_link_with_var instead.

        Args:
            name (str, optional): _description_. Defaults to None.
            value (Any, optional): _description_. Defaults to None.
            is_parameter (bool, optional): _description_. Defaults to False.
            dest_ops (List[OperationBase], optional): _description_. Defaults to None.
            source_op (OperationBase, optional): _description_. Defaults to None.

        Returns:
            Variable: _description_
        """
        if name is None:
            name = f'PPQ_Variable_{self._num_of_generated_var}'
            self._num_of_generated_var += 1
        created = Variable(name=name, value=value, is_parameter=is_parameter, dest_ops=dest_ops, source_op=source_op)
        self.append_variable(created)
        return created

    def mark_variable_as_graph_input(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(f'Except a variable here, however {type(var)} was given.')
        var_name = var.name
        if var_name not in self.variables:
            raise KeyError(f'Can not find variable {var_name} within current graph.')
        if var_name in self.inputs:
            return
        if var_name in self.outputs:
            raise KeyError(f'Can not mark variable {var_name} as graph input, cause it is graph output.')
        self.inputs[var_name] = self.variables[var_name]

    def mark_variable_as_graph_output(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(f'Except a variable here, however {type(var)} was given.')
        var_name = var.name
        if var_name not in self.variables:
            raise KeyError(f'Can not find variable {var_name} within current graph.')
        if var_name in self.outputs:
            return
        self.outputs[var_name] = self.variables[var_name]

    def __getstate__(self) ->dict:
        state = super().__getstate__()
        state['_graph_inputs'] = [var for var in self.inputs]
        state['_graph_outputs'] = [var for var in self.outputs]
        return state

    def copy(self, copy_value: bool=False):
        """Clone current graph. 
        Use parameter copy_value to control whether to do a Shallow Copy or Deep Copy.
        
        For copy_value = True, there will be a copy of each parameter in your network.
            ATTENTION: it might cause gpu memory overflow.
        For copy_value = False, cloned network will share the same parameter tensor of current one.
        
        ATTENTION: all quantization config will be cloned, 
            all scales and offsets will be cloned even with copy_valye = False.

        Shallow Copy: Shallow repetition is quicker.
        However, it's “lazy” it handles pointers and references.
        Rather than creating a contemporary copy of the particular knowledge the pointer points to,
            it simply copies over the pointer price.
        So, each the first and therefore the copy can have pointers that reference constant underlying knowledge.

        Deep Copy: Deep repetition truly clones the underlying data.
        It is not shared between the first and therefore the copy.
        """
        cloned = BaseGraph(name=self._name, built_from=self._built_from)
        for op in self.operations.values():
            cloned.append_operation(op.copy())
        for var in self.variables.values():
            cloned.append_variable(var.copy(copy_value=copy_value))
        config_dict = {}
        for op in self.operations.values():
            assert op.name in cloned.operations, f'Graph Copy Error, Operation {op.name} is not correctly cloned'
            c_op = cloned.operations[op.name]
            for i_var in op.inputs:
                assert i_var.name in cloned.variables, f'Graph Copy Error, Variable {i_var.name} is not correctly cloned'
                ci_var = cloned.variables[i_var.name]
                cloned.create_link_with_op(variable=ci_var, upstream_op=ci_var.source_op, downstream_op=c_op)
            for o_var in op.outputs:
                assert o_var.name in cloned.variables, f'Graph Copy Error, Variable {o_var.name} is not correctly cloned'
                co_var = cloned.variables[o_var.name]
                c_op.outputs.append(co_var)
                co_var.source_op = c_op
            if isinstance(op, QuantableOperation):
                for cfg, var in op.config_with_variable:
                    config_dict[cfg._hash] = op, var
        for c_op in cloned.operations.values():
            if isinstance(c_op, QuantableOperation):
                for cfg, var in c_op.config_with_variable:
                    if cfg.dominated_by != cfg:
                        assert cfg.dominated_by._hash in config_dict, 'Graph Copy Error, can not find a corresponding master config.'
                        op, var = config_dict[cfg.dominated_by._hash]
                        op = cloned.operations[op.name]
                        assert isinstance(op, QuantableOperation), 'Graph Copy Error, Unexpected Master Operation Type.'
                        for mcfg, mvar in op.config_with_variable:
                            if mvar.name == var.name:
                                cfg._dominator = mcfg
        for name in self.inputs:
            cloned.inputs[name] = cloned.variables[name]
        for name in self.outputs:
            cloned.outputs[name] = cloned.variables[name]
        for op in self.operations.values():
            if op.name not in cloned.operations:
                raise KeyError(f'Graph Copy Error, Operation {op.name} is Missing')
        for var in self.variables.values():
            if var.name not in cloned.variables:
                raise KeyError(f'Graph Copy Error, Variable {var.name} is Missing')
        for name in self.inputs:
            if name not in cloned.inputs:
                raise KeyError(f'Graph Copy Error, Input {var.name} is Missing')
        for name in self.outputs:
            if name not in cloned.outputs:
                raise KeyError(f'Graph Copy Error, Output {var.name} is Missing')
        cloned._num_of_generated_op = self._num_of_generated_op
        cloned._num_of_generated_var = self._num_of_generated_var
        cloned._detail = self._detail.copy()
        return cloned


class RuntimeHook(metaclass=ABCMeta):
    """RuntimeHook is an abstract class designed for executor customizing.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, operation: Operation, **kwargs) ->None:
        self._hook_to = operation

    def pre_forward_hook(self, inputs: list, **kwargs) ->list:
        """user-customized pre-processing procedure of input data.

        Args:
            inputs (list): a list includes all input data.

        Returns:
            list: a list includes all input data(processed).
        """
        return inputs

    def post_forward_hook(self, outputs: list, **kwargs) ->list:
        """user-customized post-processing procedure of output data.

        Args:
            inputs (list): a list includes all output data.

        Returns:
            list: a list includes all output data(processed).
        """
        return outputs


class BaseGraphExecutor(Callable, metaclass=ABCMeta):
    """PPQ Base Graph Executor.

    Args:
        Callable ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, graph: BaseGraph) ->dict:
        self._graph = None
        self._graph_input_dictionary = None
        self._graph_output_dictionary = None
        self._executing_order = None
        self.load_graph(graph=graph)

    def load_graph(self, graph: BaseGraph) ->dict:
        self._graph = graph
        self._graph_input_dictionary = self._graph.inputs
        self._graph_output_dictionary = self._graph.outputs
        self._executing_order = self._graph.topological_sort()

    def prepare_input(self, inputs: Union[dict, list, torch.Tensor]):
        assert type(inputs) in (dict, list, torch.Tensor), f'Input format misunderstood. Except either dict, list or tensor; while {type(inputs)} was given.'
        inputs_dictionary = self._graph.inputs
        if len(inputs_dictionary) == 0:
            assert inputs is None, 'Graph do not need any inputs. please set your inputs to be None.'
            return None
        if isinstance(inputs, torch.Tensor):
            assert len(inputs_dictionary) == 1, 'Graph needs more than one input, while only one tensor was given.'
            return {list(inputs_dictionary.keys())[0]: inputs}
        elif isinstance(inputs, list):
            assert len(inputs_dictionary) == len(inputs), f'Inputs format misunderstood. Given inputs has {len(inputs)} elements, while graph needs {len(inputs_dictionary)}'
            return {key: inputs[idx] for idx, key in enumerate(inputs_dictionary)}
        elif isinstance(inputs, dict):
            assert len(inputs_dictionary) == len(inputs), f'Inputs format misunderstood. Given inputs has {len(inputs)} elements, while graph needs {len(inputs_dictionary)}'
            return inputs
        else:
            raise Exception('Oops, you can never reach here.')

    @abstractmethod
    def forward(self, inputs: Union[dict, list, torch.Tensor], output_names: List[str]=None, hooks: Dict[str, RuntimeHook]=None) ->List[torch.Tensor]:
        raise NotImplementedError('Please implement this function first.')

    @abstractmethod
    def tracing_operation_meta(self, inputs: Union[dict, list, torch.Tensor], output_names: List[str]=None) ->None:
        raise NotImplementedError('Please implement this function first.')

    def __call__(self, inputs: Union[dict, torch.Tensor], output_names: List[str]=None) ->List[torch.Tensor]:
        return self.forward(inputs=inputs, output_names=output_names)

    def __str__(self) ->str:
        return f'PPQ GraphExecuter Object: {self.__hash__()}'


class GraphCommandType(Enum):
    DEPLOY_TO_CUDA = 1
    DEPLOY_TO_CPU = 2
    DEPLOY_TO_NUMPY = 3
    QUANTIZE_OPERATION = 5
    DISABLE_OPERATION_QUANTIZATION = 6
    RESTORE_OPERATION_QUANTIZATION = 7
    FORMAT_CLIP = 9
    FORMAT_PAD = 10
    FORMAT_GATHER = 11
    FORMAT_CAST = 12
    FORMAT_INT64_CONSTANT = 13
    DELETE_ISOLATED = 14
    REPLACE_SUB = 15
    FORMAT_PARAMETERS = 16
    REPLACE_OP = 17
    REPLACE_VAR = 18
    REMOVE_INPUT = 19
    TRAVERSAL_PATTERN_MATCHING = 20
    TRAVERSAL_OPSET_MATCHING = 21
    ACTIVATION_MATCHING = 22
    CONCAT_MATCHING = 23
    INSERT_SWITCHER = 25
    REMOVE_SWITCHER = 26
    FUSE_BN = 27
    FORMAT_CONSTANT_INPUT = 28
    FORMAT_SLICE = 29
    TRUNCATE_ON_VAR = 30
    FORMAT_RESIZE = 31
    FORMAT_SNG_BN = 32
    REMOVE_IDENTITY = 33


class GraphCommand:

    def __init__(self, command_type: GraphCommandType, **kwargs) ->None:
        assert isinstance(command_type, GraphCommandType), f'Command Type must be a GraphCommandType object, but {type(command_type)} received.'
        self.command_type = command_type
        self.kwargs = kwargs

    def __str__(self) ->str:
        return f'GraphCommand object {self.__hash__()},\t Command type: {self.command_type},\t Args:{self.kwargs}'


class GraphDeployCommand(GraphCommand):

    def __init__(self, device: str) ->None:
        if device.startswith('cuda'):
            super().__init__(GraphCommandType.DEPLOY_TO_CUDA)
        elif device.startswith('cpu'):
            super().__init__(GraphCommandType.DEPLOY_TO_CPU)
        else:
            raise ValueError(f'Device type {device} not understand.')
        self._device = device

    def __str__(self) ->str:
        return super().__str__()


def ASSERT_NUM_OF_INPUT(op: Operation, values: List[torch.Tensor], min_num_of_input: int=-1, max_num_of_input: int=99):
    if min_num_of_input == max_num_of_input:
        if len(values) != min_num_of_input:
            raise ValueError(f'Can not feed value to operation {op.name}, expects exact {min_num_of_input} inputs, however {len(values)} was given')
    elif len(values) > max_num_of_input:
        raise ValueError(f'Too many input value for {op.name}, expects {max_num_of_input} inputs at most, however {len(values)} was given')
    elif len(values) < min_num_of_input:
        raise ValueError(f'Too few input value for {op.name}, expects {min_num_of_input} inputs at least, however {len(values)} was given')


class TorchBackendContext:

    def __init__(self, executing_device: str) ->None:
        self.executing_device = executing_device


def Abs_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Absolute takes one input data (Tensor) and produces one output data (Tensor) 
    where the absolute is, y = abs(x), is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
            Output tensor

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return x.abs()


def AdaptiveAvgPool2d_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    input_value, output_size = values
    output = F.adaptive_avg_pool2d(input_value, output_size)
    return output


def VALUE_TO_EXECUTING_DEVICE(op: Operation, ctx: TorchBackendContext, values: List[torch.Tensor]) ->List[torch.Tensor]:
    if ctx is None:
        device = values[0].device
    else:
        device = ctx.executing_device
    for idx, (plat, value) in enumerate(zip(op.socket.in_plat, values)):
        if value is None:
            continue
        if plat == TargetPlatform.SOI or op.platform == TargetPlatform.SOI:
            values[idx] = value.cpu()
        else:
            values[idx] = value
    return values


def Add_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Performs element-wise binary addition (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting;
        for more details please check the doc.

    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

    Inputs
        A (differentiable) : T
            First operand.
        B (differentiable) : T
            Second operand.

    Outputs
        C (differentiable) : T
            Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        input_values (list): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    a, b = values
    return a + b


def And_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    a, b = values
    return a & b


def ArgMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axis', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.argmax(input_value, dim=dim, keepdim=keepdim)
    return output


def GET_ATTRIBUTE_FROM_OPERATION(op: Operation, attribute: str, compulsive: bool=False, default: Any=None):
    """Try to get an attribute from operation. If an attribute is compulsive,
    then operation must give a value of it, otherwise an error will be thrown.
    If an attribute is not compulsive, a default value will be given if
    operation.attributes do not holds a value of requesting attribute.

    Args:
        op (Operation): Operation instance.
        attribute (str): Attribute name.
        compulsive (bool): Whether is a compulsive attribute.
        default (Any, optional): [description]. default value of attribute.
    """
    if attribute in op.attributes:
        return op.attributes[attribute]
    elif compulsive:
        raise KeyError(f'Operation {op.name} is supposed to have a value of attribute {attribute}. ', 'However this value is missing from currecnt operation.')
    else:
        return default


def convert_onnx_pads_to_torch(onnx_pads: List[int], mode: str=None) ->List[int]:
    if onnx_pads is None:
        return 0
    if isinstance(onnx_pads, int):
        return onnx_pads
    if mode is not None:
        if mode == '1d':
            assert len(onnx_pads) == 2, f'1d Operation needs 2-d padding value, while your padding value is {onnx_pads}'
        elif mode == '2d':
            assert len(onnx_pads) == 4, f'2d Operation needs 4-d padding value, while your padding value is {onnx_pads}'
        elif mode == '3d':
            assert len(onnx_pads) == 6, f'3d Operation needs 6-d padding value, while your padding value is {onnx_pads}'
    middle = len(onnx_pads) // 2
    onnx_pad_begin, onnx_pad_end = onnx_pads[:middle], onnx_pads[middle:]
    onnx_pad_begin, onnx_pad_end = onnx_pad_begin[::-1], onnx_pad_end[::-1]
    torch_pads = []
    for begin, end in zip(onnx_pad_begin, onnx_pad_end):
        torch_pads.extend([begin, end])
    if mode is None:
        return torch_pads
    if len(torch_pads) == 2:
        p1, p2 = torch_pads
        if p1 == p2:
            torch_pads = [p1]
    if len(torch_pads) == 4:
        p1, p2, p3, p4 = torch_pads
        if p1 == p2 and p3 == p4:
            torch_pads = [p1, p3]
    if len(torch_pads) == 6:
        p1, p2, p3, p4, p5, p6 = torch_pads
        if p1 == p2 and p3 == p4 and p5 == p6:
            torch_pads = [p1, p3, p5]
    return torch_pads


COLOR_END = ' \x1b[m'


G_BEGIN = '\x1b[38;5;2m'


class LEVEL(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __le__(self, other):
        return self.value <= other.value

    @staticmethod
    def convert(level: str):
        level = level.lower()
        if level == 'debug':
            return LEVEL.DEBUG
        elif level == 'info':
            return LEVEL.INFO
        elif level == 'warning':
            return LEVEL.WARNING
        elif level == 'error':
            return LEVEL.ERROR
        else:
            raise ValueError(f'the lowercase of param level should be one of debug/info/warning/errorhowever {level} is given')


class Handler:

    def __init__(self, file_name: str=None, level: LEVEL=LEVEL.INFO) ->None:
        self._file_name = file_name
        self._level = level
        self._fd = sys.stdout if file_name is None else open(file_name, 'w+', encoding='utf-8')

    def process(self, msg: str, level: LEVEL) ->None:
        if self._level <= level:
            self._fd.write('\r' + msg + '\n')
            self._fd.flush()

    def set_level(self, level: LEVEL) ->None:
        self._level = level


R_BEGIN = '\x1b[38;5;1m'


Y_BEGIN = '\x1b[38;5;3m'


def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


initialized_loggers = {}


class NaiveLogger(object):
    """A very naive implementation of colored logger, but would be suffice in
    single-process situation here where no race condition would happen."""
    __create_key = object()

    def __init__(self, create_key: object, name: str, level: LEVEL, file_name: str) ->None:
        assert create_key == NaiveLogger.__create_key, 'logger instance must be created using NaiveLogger.get_logger'
        self._name = name
        self._level = level
        self._handlers = {'stdout': Handler(level=level)}
        if file_name is not None:
            self.register_handler(file_name)

    def set_level(self, level: Union[str, LEVEL]):
        if isinstance(level, str):
            level = LEVEL.convert(level)
        assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
        self._level = level
        for handler in self._handlers.values():
            handler.set_level(level)

    @classmethod
    def get_logger(cls, name: str, level: Union[str, LEVEL]=LEVEL.INFO, file_name: str=None):
        if name in initialized_loggers:
            return initialized_loggers[name]
        if isinstance(level, str):
            level = LEVEL.convert(level)
        assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
        logger = NaiveLogger(cls.__create_key, name, level, file_name)
        initialized_loggers[name] = logger
        return logger

    def wrap_header(self, type: str) ->str:
        cur_time = get_current_time()
        return '[{}][{}][{}]: '.format(type, self._name, cur_time)

    def info(self, msg: str):
        header = self.wrap_header('INFO')
        print_msg = G_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.INFO)
            else:
                handler.process(print_msg, LEVEL.INFO)

    def warning(self, msg: str):
        header = self.wrap_header('WARNING')
        print_msg = Y_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.WARNING)
            else:
                handler.process(print_msg, LEVEL.WARNING)

    def error(self, msg: str):
        header = self.wrap_header('ERROR')
        print_msg = R_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.ERROR)
            else:
                handler.process(print_msg, LEVEL.ERROR)

    def debug(self, msg: str):
        header = self.wrap_header('DEBUG')
        print_msg = G_BEGIN + header + COLOR_END + msg
        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.DEBUG)
            else:
                handler.process(print_msg, LEVEL.DEBUG)

    def register_handler(self, file_name: str, level: Union[str, LEVEL]=LEVEL.INFO):
        if file_name not in self._handlers:
            if isinstance(level, str):
                level = LEVEL.convert(level)
            assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
            self._handlers[file_name] = Handler(file_name, level)

    def remove_handler(self, file_name: str):
        if file_name in self._handlers:
            handler = self._handlers[file_name]
            if handler._file_name is not None:
                handler._fd.close()
            self._handlers.pop(file_name)


logger = NaiveLogger.get_logger('PPQ')


def process_attribute(attr, input_shape, kernel_shape=None, op_type=None):
    auto_pad = attr.get('auto_pad', 'NOTSET')
    strides = attr.get('strides', [1, 1])
    dilations = attr.get('dilations', [1, 1])
    kernels = attr.get('kernel_shape', kernel_shape)
    pad_needed = None
    if op_type == 'ConvTranspose' and 'output_shape' in attr:
        output_shape = attr['output_shape']
        out_pad = [0, 1] if output_shape % 2 != 0 else [0, 0]
        pad_needed = [((input_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 + out_pad[i] - output_shape[i]) for i in range(len(input_shape))]
    if auto_pad != 'NOTSET':
        if 'pads' in attr:
            logger.warning('auto_pad is conflict with pads attribute. Use pads here.')
        elif auto_pad == 'VALID':
            attr['pads'] = [0, 0, 0, 0]
        elif auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            if op_type == 'ConvTranspose':
                out_pad = attr.get('output_padding', [0, 0])
                output_shape = [(input_shape[i] * strides[i]) for i in range(len(input_shape))]
                pad_needed = [((input_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 + out_pad[i] - output_shape[i]) for i in range(len(input_shape))]
            else:
                output_shape = [((input_shape[i] + strides[i] - 1) // strides[i]) for i in range(len(input_shape))]
                pad_needed = [((output_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 - input_shape[i]) for i in range(len(input_shape))]
        else:
            raise ValueError(f'Invalid auto_pad value {auto_pad}')
    if pad_needed is not None:
        pads = []
        for item in pad_needed:
            pads.append((item if auto_pad == 'SAME_UPPER' else item + 1) // 2)
        pads = pads + [(pad_needed[i] - p) for i, p in enumerate(pads)]
        attr['pads'] = pads
        attr.pop('auto_pad')


def AveragePool_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])
    [x] = values
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    if op.type == 'GlobalAveragePool':
        kernel_size = x.size()[2:]
    else:
        kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    ndim = x.ndim
    if ndim == 3:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool1d(x, kernel_size=kernel_size, padding=torch_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    if ndim == 4:
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0
        output = F.avg_pool2d(x, kernel_size=kernel_size, padding=onnx_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool3d(x, kernel_size=kernel_size, padding=torch_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def BatchNormalization_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=5, max_num_of_input=5)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data, weight, bias, running_mean, running_var = values
    op_attr = {'eps': op.attributes.get('epsilon', 1e-05), 'momentum': 1 - op.attributes.get('momentum', 0.9)}
    output = F.batch_norm(input_data, running_mean, running_var, weight=weight, bias=bias, **op_attr)
    return output


def CaffeArgMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    assert len(values) == 1
    input_data = values[0]
    dim = op.attributes.get('axis', None)
    output = input_data.topk(op.attributes.get('top_k', 1), dim=dim)
    return output
    """
    # There are some gaps between ppl-argmax and standard argmax
    # If out_max_val is true, produce pairs (argmax, maxval)
    output = (output[1], output[0])
    if op.attributes.get('out_max_val', False):
        _update_output(op, output, 1)
    else:
        _update_output(op, output[0], 1)
    """


def Cast_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    new_data_type = DataType.to_torch(op.attributes['to'])
    output = input_value
    return output


def ChannelShuffle_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    group = op.attributes.get('group', 1)
    assert input_data.shape[1] % group == 0
    n, c, h, w = input_data.shape
    input_data = input_data.view(n, group, c // group, h, w)
    input_data = input_data.permute(0, 2, 1, 3, 4)
    output = input_data.contiguous().view(n, c, h, w)
    return output


def Clip_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    if len(values) == 1:
        values.append(op.attributes.get('min', float('-inf')))
        values.append(op.attributes.get('max', float('+inf')))
    output = torch.clamp(*values)
    return output


def Concat_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Concatenate a list of tensors into a single tensor. All input tensors
    must have the same shape, except for the dimension size of the axis to
    concatenate on.

    Attributes
        axis : int (required)
            Which axis to concat on.
            A negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(inputs)..

    Inputs (1 - ∞)
        inputs (variadic, differentiable) : T
            List of tensors for concatenation

    Outputs
        concat_result (differentiable) : T
            Concatenated tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', compulsive=True)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    concat_view = []
    for value in values:
        if value.ndim == 0:
            value = value.unsqueeze(0)
        concat_view.append(value)
    return torch.cat(concat_view, axis=axis)


def convert_any_to_python_primary_type(x: Union[torch.Tensor, np.ndarray, int, float, list, str], accept_none: bool=True) ->Union[int, float, list, str]:
    if x is None and accept_none:
        return None
    if x is None and not accept_none:
        raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple):
        return list(x)
    elif isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none:
            return None
        if x.numel() == 0 and not accept_none:
            raise ValueError('Trying to convert an empty value.')
        if str(x.device) != 'cpu':
            x = x.cpu()
        if x.numel() == 1:
            return x.item()
        if x.numel() > 1:
            return x.tolist()
    elif isinstance(x, np.ndarray):
        if x.size == 0 and accept_none:
            return None
        if x.size == 0 and not accept_none:
            raise ValueError('Trying to convert an empty value.')
        if x.size == 1:
            return x.reshape((1,)).tolist()[0]
        if x.size > 1:
            return x.tolist()
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as python primary type.')


def ConstantOfShape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Generate a tensor with given value and shape.

    Attributes
    value : tensor
    (Optional) The value of the output elements.Should be a one-element tensor.
        If not specified, it defaults to a tensor of value 0 and datatype float32

    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor.
            If empty tensor is given, the output would be a scalar. All values must be >= 0.

    Outputs
        output : T2
        Output tensor of shape specified by 'input'.If attribute 'value' is specified,
            the value and datatype of the output tensor is taken from 'value'. If attribute 'value' is not specified,
            the value in the output defaults to 0, and the datatype defaults to float32.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    value = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='value', compulsive=False, default=0.0)
    [shape], fill_value = values, convert_any_to_python_primary_type(value)
    output = torch.Tensor().new_full(size=shape.tolist(), fill_value=fill_value)
    if isinstance(fill_value, int):
        output = output.long()
    elif isinstance(fill_value, float):
        output = output.float()
    else:
        raise TypeError(f'Can not parse value type{type(value)}.')
    return output


def Constant_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """A constant tensor. Exactly one of the two attributes, either value or
    sparse_value, must be specified.

    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Attributes
        sparse_value : sparse_tensor
            The value for the elements of the output tensor in sparse format.

        value : tensor
            The value for the elements of the output tensor.
    Inputs

    Outputs
        output : T
            Output tensor containing the same value of the provided tensor.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=0, max_num_of_input=0)
    return op.attributes['value']


def ConvTranspose_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """The convolution transpose operator consumes an input tensor and a
    filter, and computes the output.

    If the pads parameter is provided the shape of the output is calculated via the following equation:

        output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
        output_shape can also be explicitly specified in which case pads values are
            auto generated using these equations:

    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]

    If (auto_pads == SAME_UPPER):
        pads[start_i] = total_padding[i]/2;
        pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else:
        pads[start_i] = total_padding[i] - (total_padding[i]/2);
        pads[end_i] = (total_padding[i]/2).

    Attributes
        auto_pad : string (default is NOTSET)
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET,
            which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that
            `output_shape[i] = input_shape[i] * strides[i]` for each axis `i`.
        The padding is split between the two sides equally or almost equally
            (depending on whether it is even or odd).
        In case the padding is an odd number,
            the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

        dilations : list of ints
        dilation value along each spatial axis of the filter.
            If not present, the dilation defaults to 1 along each spatial axis.

        group : int (default is 1)
        number of groups input channels and output channels are divided into.

        kernel_shape : list of ints
        The shape of the convolution kernel. If not present, should be inferred from input W.

        output_padding : list of ints
        Additional elements added to the side with higher coordinate indices in the output.
            Each padding value in "output_padding" must be less than the corresponding stride/dilation dimension.
            By default, this attribute is a zero vector.
        Note that this attribute doesn't directly affect the computed output values.
            It only controls the selection of the computed values,
            so changing this attribute only adds or removes output elements.
        If "output_shape" is explicitly provided,
            "output_padding" does not contribute additional size to "output_shape"
            but participates in the computation of the needed padding amount.
            This is also called adjs or adjustment in some frameworks.

        output_shape : list of ints
        The shape of the output can be explicitly set which will cause pads values to be auto generated.
        If output_shape is specified pads values are ignored. See doc for details for equations to generate pads

        pads : list of ints
        Padding for the beginning and ending along each spatial axis,
            it can take any value greater than or equal to 0.
        The value represent the number of pixels added to the beginning and end part of the corresponding axis.
        `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
            where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
            the number of pixels added at the end of axis `i`.
        This attribute cannot be used simultaneously with auto_pad attribute.
        If not present, the padding defaults to 0 along start and end of each spatial axis.

        strides : list of ints
        Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

    Inputs (2 - 3)
        X (differentiable) : T
        Input data tensor from previous layer; has size (N x C x H x W),
            where N is the batch size, C is the number of channels, and H and W are the height and width.
            Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)

        W (differentiable) : T
        The weight tensor that will be used in the convolutions; has size (C x M/group x kH x kW),
        where C is the number of channels, and kH and kW are the height and width of the kernel,
            and M is the number of feature maps.
        For more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn),
            where (k1 x k2 x ... x kn) is the dimension of the kernel.
        The number of channels in the output should be equal to
            W.shape[1] * group (assuming zero based indices of the shape array)

        B (optional, differentiable) : T
        Optional 1D bias to be added to the convolution, has size of M.

    Outputs
        Y (differentiable) : T
        Output data tensor that contains the result of the convolution.
        The output dimensions are functions of the kernel size, stride size, pad lengths and group count.
        The number of channels in the output should be equal to
            W.shape[1] * group (assuming zero based indices of the shape array)

    Type Constraints
        T : tensor(float16), tensor(float), tensor(double)
        Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:], 'ConvTranspose')
    groups = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    output_padding = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_padding', default=0)
    x, w = values[:2]
    b = values[2] if len(values) > 2 else None
    ndim = x.ndim
    if ndim == 4:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='2d')
        if isinstance(torch_pads, list) and len(torch_pads) == 2:
            output = F.conv_transpose2d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose2d(input=x, weight=w, bias=b, groups=groups, padding=0, dilation=dilation, stride=stride, output_padding=output_padding)
            p1, p2, p3, p4 = torch_pads
            _, _, h, w = output.shape
            output = output[:, :, 0 + p1:h - p2, 0 + p3:w - p4]
    elif ndim in {2, 3}:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) == 1:
            output = F.conv_transpose1d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose1d(input=x, weight=w, bias=b, groups=groups, padding=0, dilation=dilation, stride=stride, output_padding=output_padding)
            p1, p2 = torch_pads
            _, _, h = output.shape
            output = output[:, :, 0 + p1:h - p2]
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) == 3:
            output = F.conv_transpose3d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose3d(input=x, weight=w, bias=b, groups=groups, padding=0, dilation=dilation, stride=stride, output_padding=output_padding)
            p1, p2, p3, p4, p5, p6 = torch_pads
            _, _, d, h, w = output.shape
            output = output[:, :, 0 + p1:d - p2, 0 + p3:h - p4, 0 + p5:w - p6]
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def Conv_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """The convolution operator consumes an input tensor and a filter, and
    computes the output.

    Attributes
        auto_pad : string (default is NOTSET)
            auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET,
            which means explicit padding is used.

            SAME_UPPER or SAME_LOWER mean pad the input so that
                `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
            The padding is split between the two sides equally or almost equally
                (depending on whether it is even or odd).
            In case the padding is an odd number,
                the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

        dilations : list of ints
            dilation value along each spatial axis of the filter.
            If not present, the dilation defaults is 1 along each spatial axis.

        group : int (default is 1)
            number of groups input channels and output channels are divided into.

        kernel_shape : list of ints
            The shape of the convolution kernel. If not present, should be inferred from input W.

        pads : list of ints
            Padding for the beginning and ending along each spatial axis,
            it can take any value greater than or equal to 0.

            The value represent the number of pixels added to the beginning and end part of the corresponding axis.
            `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
            where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
            the number of pixels added at the end of axis `i`.

            This attribute cannot be used simultaneously with auto_pad attribute.
            If not present, the padding defaults to 0 along start and end of each spatial axis.

        strides : list of ints
            Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    Inputs (2 - 3)
        X (differentiable) : T
            Input data tensor from previous layer;
            has size (N x C x H x W), where N is the batch size,
            C is the number of channels, and H and W are the height and width.
            Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn).

            Optionally, if dimension denotation is in effect,
            the operation expects input data tensor to arrive
                with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

        W (differentiable) : T
            The weight tensor that will be used in the convolutions;
            has size (M x C/group x kH x kW), where C is the number of channels,
            and kH and kW are the height and width of the kernel,
            and M is the number of feature maps. For more than 2 dimensions,
            the kernel shape will be (M x C/group x k1 x k2 x ... x kn),
                where (k1 x k2 x ... kn) is the dimension of the kernel.
            Optionally, if dimension denotation is in effect,
                the operation expects the weight tensor to arrive with the dimension
                denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...].

            Assuming zero based indices for the shape array,
            X.shape[1] == (W.shape[1] * group) == C and W.shape[0] mod G == 0.

            Or in other words FILTER_IN_CHANNEL multiplied by the number of groups should be
            equal to DATA_CHANNEL and the number of feature maps M should be a multiple of the number of groups G.

        B (optional, differentiable) : T
            Optional 1D bias to be added to the convolution, has size of M.

    Outputs
        Y (differentiable) : T
            Output data tensor that contains the result of the convolution.
            The output dimensions are functions of the kernel size, stride size, and pad lengths.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    groups = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    auto_pad = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='auto_pad', default='NOTSET')
    x, w = values[:2]
    b = values[2] if len(values) > 2 else None
    ndim = w.ndim
    if ndim in {2, 3}:
        if auto_pad != 'NOTSET':
            raise NotImplementedError(f'auto_pad must be "NOTSET" with 1-d conv {op.name}')
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) == 2:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.conv1d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride)
    elif ndim == 4:
        process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:])
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0
        output = F.conv2d(input=x, weight=w, bias=b, groups=groups, padding=onnx_pads, dilation=dilation, stride=stride)
    elif ndim == 5:
        if auto_pad != 'NOTSET':
            raise NotImplementedError(f'auto_pad must be "NOTSET" with 3-d conv {op.name}')
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) == 6:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.conv3d(input=x, weight=w, bias=b, groups=groups, padding=torch_pads, dilation=dilation, stride=stride)
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def Cos_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Calculates the cosine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The cosine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.cos(x)


def DepthToSpace_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data = values[0]
    upsample = op.attributes.get('blocksize', 1)
    mode = op.attributes.get('mode', 'DCR')
    if mode == 'DCR':
        output = F.pixel_shuffle(input_data, upsample)
    else:
        output = F.pixel_shuffle(input_data, upsample)
    return output


def Eltwise_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    if op.type == 'Add':
        assert len(values) == 2
        output = torch.add(*values).float()
    elif op.type == 'Sub':
        assert len(values) == 2
        output = torch.sub(*values).float()
    elif op.type == 'Mul':
        assert len(values) == 2
        output = torch.mul(*values).float()
    elif op.type == 'Div':
        assert len(values) == 2
        version = torch.__version__
        if version < '1.5.0' or version >= '1.7.0':
            output = torch.div(*values)
        elif values[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            output = torch.floor_divide(*values)
        else:
            output = torch.div(*values)
    elif op.type == 'Max':
        output = torch.max(*values)
    elif op.type == 'Min':
        output = torch.min(*values)
    else:
        logger.warning('Not Eltwise op, return input as output')
        output = values
    return output


def Elu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    Elu takes one input data (Tensor) and produces one output data (Tensor) 
    where the function f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0., 
    is applied to the tensor elementwise.

    Version
    This version of the operator has been available since version 6 of the default ONNX operator set.

    Other versions of this operator: 1

    Attributes
        alpha : float (default is 1.0)
            Coefficient of ELU.
    
    Inputs
        X (differentiable) : T
            1D input tensor
    
    Outputs
        Y (differentiable) : T
            1D output tensor
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='alpha', default=1.0)
    return F.elu(x, alpha=alpha)


def Equal_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_a, input_b = values
    output = torch.eq(input_a, input_b)
    return output


def Erf_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    Elu takes one input data (Tensor) and produces one output data (Tensor) 
    where the function f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0., 
    is applied to the tensor elementwise.

    Version
    This version of the operator has been available since version 6 of the default ONNX operator set.

    Other versions of this operator: 1

    Attributes
        alpha : float (default is 1.0)
            Coefficient of ELU.
    
    Inputs
        X (differentiable) : T
            1D input tensor
    
    Outputs
        Y (differentiable) : T
            1D output tensor
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.erf(x)


def Expand_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    """Broadcast the input tensor following the given shape and the broadcast
    rule.

    The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
        Dimensions are right alignment;
        Two corresponding dimension must have the same value, or one of them is equal to 1.

    Also, this operator is similar to numpy.broadcast_to(input, shape),
    but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
    It is possible that the output.shape is not equal to shape,
    when some dimensions in shape is equal to 1, or the shape.ndim < input.shape.ndim.

    Inputs
        input (differentiable) : T
        Input tensor

        shape (non-differentiable) : tensor(int64)
        A 1-D tensor indicates the shape you want to expand to, following the broadcast rule

    Outputs
        output (differentiable) : T
        Output tensor

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    tensor, shape = values
    output = tensor * torch.ones(convert_any_to_python_primary_type(shape), dtype=tensor.dtype, device=tensor.device)
    return output


def Flatten_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axis', 1)
    shape = list(input_value.shape)
    new_shape = [1, -1] if dim == 0 else [reduce(operator.mul, shape[:dim], 1), -1]
    output = input_value.reshape(new_shape)
    return output


def Floor_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    output = torch.floor(input_data)
    return output


def GET_VALUE_FROM_INPUTS(values: list, idx: int) ->torch.Tensor:
    assert isinstance(idx, int)
    assert idx > 0
    if len(values) > idx:
        return values[idx]
    else:
        return None


GRU_FLATTEN_WEIGHT_ATTRIB = 'GRU_FLATTEN_WEIGHT_ATTRIB'


def GRU_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.
    只支持 pytorch 导出来的 GRU 啊亲; 必须要 6 个输入 Variable
    Notations:
        X - input tensor
        z - update gate
        r - reset gate
        h - hidden gate
        t - time step (t-1 means previous time step)
        W[zrh] - W parameter weight matrix for update, reset, and hidden gates
        R[zrh] - R recurrence weight matrix for update, reset, and hidden gates
        Wb[zrh] - W bias vectors for update, reset, and hidden gates
        Rb[zrh] - R bias vectors for update, reset, and hidden gates
        WB[zrh] - W parameter weight matrix for backward update, reset, and hidden gates
        RB[zrh] - R recurrence weight matrix for backward update, reset, and hidden gates
        WBb[zrh] - W bias vectors for backward update, reset, and hidden gates
        RBb[zrh] - R bias vectors for backward update, reset, and hidden gates
        H - Hidden state
        num_directions - 2 if direction == bidirectional else 1
    Activation functions:
        Relu(x)                - max(0, x)
        Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
        Sigmoid(x)             - 1/(1 + e^{-x})
    (NOTE: Below are optional)
        Affine(x)              - alpha*x + beta
        LeakyRelu(x)           - x if x >= 0 else alpha * x
        ThresholdedRelu(x)     - x if x >= alpha else 0
        ScaledTanh(x)          - alpha*Tanh(beta*x)
        HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
        Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
        Softsign(x)            - x/(1 + |x|)
        Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh):
        - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    This operator has optional inputs/outputs. See the doc for more details about the representation of optional arguments.
    An empty string may be used in the place of an actual argument's name to indicate a missing argument.
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    Version
    This version of the operator has been available since version 14 of the default ONNX operator set.
    Other versions of this operator: 1, 3, 7
    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions.
            The values are consumed in the order of activation functions,
            for example (f, g, h) in LSTM.
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu,
            the default alpha is 0.01.
        activation_beta : list of floats
            Optional scaling values used by some activation functions.
            The values are consumed in the order of activation functions,
            for example (f, g, h) in LSTM.
            Default values are the same as of corresponding ONNX operators.
        activations : list of strings
            A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates.
            The activation functions must be one of the activation functions specified above.
            Optional: See the equations for default if not specified.
        clip : float
            Cell clip threshold.
            Clipping bounds the elements of a tensor in the range of [-threshold, +threshold]
            and is applied to the input of activations. No clip if not specified.
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional.
            Must be one of forward (default), reverse, or bidirectional.
        hidden_size : int
            Number of neurons in the hidden layer
        layout : int (default is 0)
            The shape format of inputs X, initial_h and outputs Y, Y_h.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
        linear_before_reset : int (default is 0)
            When computing the output of the hidden gate,
            apply the linear transformation before multiplying by the output of the reset gate.
    Inputs (3 - 6)
        X (differentiable) : T
            The input sequences packed (and potentially padded) into one 3-D tensor with the shape of
            `[seq_length, batch_size, input_size]`.
        W (differentiable) : T
            The weight tensor for the gates.
            Concatenation of `W[zrh]` and `WB[zrh]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 3*hidden_size, input_size]`.
        R (differentiable) : T
            The recurrence weight tensor.
            Concatenation of `R[zrh]` and `RB[zrh]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 3*hidden_size, hidden_size]`.
        B (optional, differentiable) : T
            The bias tensor for the gates.
            Concatenation of `[Wb[zrh], Rb[zrh]]` and `[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 6*hidden_size]`. Optional: If not specified - assumed to be 0
        sequence_lens (optional, non-differentiable) : T1
            Optional tensor specifying lengths of the sequences in a batch.
            If not specified - assumed all sequences in the batch to have length `seq_length`.
            It has shape `[batch_size]`.
        initial_h (optional, non-differentiable) : T
            Optional initial value of the hidden.
            If not specified - assumed to be 0.
            It has shape `[num_directions, batch_size, hidden_size]`.
    Outputs (0 - 2)
        Y (optional, differentiable) : T
            A tensor that concats all the intermediate output values of the hidden.
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
        Y_h (optional, differentiable) : T
            The last output value of the hidden.
            It has shape `[num_directions, batch_size, hidden_size]`.
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
        T1 : tensor(int32)
    Constrain seq_lens to integer tensor.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    linear_before_reset = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if linear_before_reset != 0:
        raise NotImplementedError('PPQ do not support LSTM with linear_before_reset != 1.')
    if activation_alpha is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    bidirectional = direction == 'bidirectional'
    has_bias = b is not None
    if GRU_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([w[0][hidden_size * 1:hidden_size * 2], w[0][hidden_size * 0:hidden_size * 1], w[0][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
        forward_r = torch.cat([r[0][hidden_size * 1:hidden_size * 2], r[0][hidden_size * 0:hidden_size * 1], r[0][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([b[0, hidden_size * 1:hidden_size * 2], b[0, hidden_size * 0:hidden_size * 1], b[0, hidden_size * 2:hidden_size * 3]]).contiguous()
            forward_bias_2 = torch.cat([b[0, hidden_size * 4:hidden_size * 5], b[0, hidden_size * 3:hidden_size * 4], b[0, hidden_size * 5:hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([w[1][hidden_size * 1:hidden_size * 2], w[1][hidden_size * 0:hidden_size * 1], w[1][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
            reverse_r = torch.cat([r[1][hidden_size * 1:hidden_size * 2], r[1][hidden_size * 0:hidden_size * 1], r[1][hidden_size * 2:hidden_size * 3]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([b[1, hidden_size * 1:hidden_size * 2], b[1, hidden_size * 0:hidden_size * 1], b[1, hidden_size * 2:hidden_size * 3]]).contiguous()
                reverse_bias_2 = torch.cat([b[1, hidden_size * 4:hidden_size * 5], b[1, hidden_size * 3:hidden_size * 4], b[1, hidden_size * 5:hidden_size * 6]]).contiguous()
        flatten_weight = [forward_w, forward_r]
        if has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:
            flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(GRU_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(size=[s, x.shape[1], x.shape[2]], device=x.device, dtype=torch.float32)
    result = _VF.gru(x, initial_h, op._detail[GRU_FLATTEN_WEIGHT_ATTRIB], has_bias, 1, 0.0, False, bidirectional, False)
    hidden_vector, last_state = result
    return hidden_vector.unsqueeze(1), last_state


def GatherND_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data, indices = values
    batch_dims = op.attributes.get('batch_dims', 0)
    data_rank = len(input_data.shape)
    assert indices.shape[-1] <= data_rank
    num_i = batch_dims
    num_k = len(input_data.shape) - num_i - indices.shape[-1]
    num_idx = indices.shape[-1]
    shape_i = indices.shape[:num_i]
    shape_j = indices.shape[num_i:-1]
    shape_k = input_data.shape[num_i + num_idx:]
    shape_idx = input_data.shape[num_i:num_i + num_idx]
    reshaped_indices = indices.reshape(*shape_i, -1, num_idx)
    strides = torch.tensor([reduce(operator.mul, shape_idx[i + 1:], 1) for i in range(num_idx)], device=input_data.device, dtype=torch.float)
    merged_indices = torch.tensordot(reshaped_indices.float(), strides, 1)
    expanded_indices = merged_indices.reshape(*merged_indices.shape, *([1] * num_k)).expand(*merged_indices.shape, *shape_k).long()
    reshaped_input = input_data.reshape(*shape_i, -1, *shape_k)
    output = reshaped_input.gather(batch_dims, expanded_indices)
    reshaped_output = output.reshape(*shape_i, *shape_j, *shape_k)
    return reshaped_output


def Gather_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Given data tensor of rank r >= 1, and indices tensor of rank q, 
    gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, 
    and concatenates them in an output tensor of rank q + (r - 1).

    axis = 0 :

    Let k = indices[i_{0}, ..., i_{q-1}] 
        Then output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

        data = [
            [1.0, 1.2],
            [2.3, 3.4],
            [4.5, 5.7],
        ]
        indices = [
            [0, 1],
            [1, 2],
        ]
        output = [
            [
                [1.0, 1.2],
                [2.3, 3.4],
            ],
            [
                [2.3, 3.4],
                [4.5, 5.7],
            ],
        ]
    
    axis = 1 :

    Let k = indices[i_{0}, ..., i_{q-1}] 
        Then output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

        data = [
            [1.0, 1.2, 1.9],
            [2.3, 3.4, 3.9],
            [4.5, 5.7, 5.9],
        ]
        indices = [
            [0, 2],
        ]
        axis = 1,
        output = [
                [[1.0, 1.9]],
                [[2.3, 3.9]],
                [[4.5, 5.9]],
        ]
    
    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

        Other versions of this operator: 1, 11

    Attributes
        axis : int (default is 0)
            Which axis to gather on. Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data).
    
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
    
        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q. All index values are expected to be 
            within bounds [-s, s-1] along axis of size s. 
            
            It is an error if any of the index values are out of bounds.
    
    Outputs
        output (differentiable) : T
            Tensor of rank q + (r - 1).
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data, indices = values
    indices = indices.long()
    axis = op.attributes.get('axis', 0)
    if op.type == 'Gather':
        array_idx = [(indices if axis == i else slice(dim)) for i, dim in enumerate(input_data.shape)]
        output = input_data[array_idx]
    elif op.type == 'GatherElements':
        output = torch.gather(input_data, axis, indices)
    else:
        logger.warning('Not Gather op, return input as output')
        output = values
    return output


def Gelu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    return F.gelu(input_value)


def Gemm_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    A, B = values[:2]
    if op.opset.is_caffe() and A.ndim > 2:
        axis = op.attributes.get('axis', 1)
        A = A.flatten(start_dim=axis)
    C = values[2] if len(values) > 2 else 0
    alpha = op.attributes.get('alpha', 1.0)
    beta = op.attributes.get('beta', 1.0)
    transA = op.attributes.get('transA', 0)
    transB = op.attributes.get('transB', 0)
    A = A.transpose(0, 1) if transA else A
    B = B.transpose(0, 1) if transB else B
    output = alpha * torch.matmul(A, B) + beta * C
    return output


def Greater_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_a, input_b = values
    if input_a.dim() >= input_b.dim() or input_a.shape > input_b.shape:
        output = torch.gt(input_a, input_b)
    else:
        output = torch.lt(input_b, input_a)
    return output


def GridSampler_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    Given an input and a flow-field grid, computes the output using input values and pixel locations from grid. 
    Currently, only spatial (4-D) inputs are supported. 
    For input with shape (N, C, H, W) and grid with shape (N, H_out, W_out, 2), 
    the output will have shape (N, C, H_out, W_out). 
    For each output location output[N, C, H_out, W_out], 
    the size-2 vector grid[N, H_out, W_out] specifies input pixel locations x and y, 
    which are used to interpolate the output value output[N, C, H_out, W_out].

    The GridSample operator is often used in doing grid generator and sampler in the Spatial Transformer Networks. 
    See also in torch.nn.functional.grid_sample.

    Attributes
        align_corners : int (default is 0)
            If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. 
            If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels,
            making the sampling more resolution agnostic.

        mode : string (default is bilinear)
            Three interpolation modes: bilinear (default), nearest and bicubic.
    
        padding_mode : string (default is zeros)
            Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. 
            zeros: use 0 for out-of-bound grid locations, border: use border values for out-of-bound grid locations, 
            reflection: use values at locations reflected by the border for out-of-bound grid locations.
            If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. 
            For location far away from the border, it will keep being reflected until becoming in bound. 
            If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' = 0.5.
    
    Inputs
        X (differentiable) : T1
            4-D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, 
            H and W are the height and width of the input data.
        
        grid (non-differentiable) : T1
            Input offset, 4-D tensor of shape (N, H_out, W_out, 2), 
            where H_out and W_out are the height and width of grid and output, 
            Grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore,
            it should have most values in the range of [-1, 1]. 
            If grid has values outside the range of [-1, 1], 
            the corresponding outputs will be handled as defined by padding_mode.
    
    Outputs
        Y (differentiable) : T2
            4-D tensor of shape (N, C, H_out, W_out).
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, grid = values
    output = F.grid_sample(value, grid, align_corners=False)
    return output


def HardSigmoid_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    HardSigmoid takes one input data (Tensor) and produces one output data (Tensor) where the HardSigmoid function, 
    y = max(0, min(1, alpha * x + beta)), is applied to the tensor elementwise.

    Attributes
        alpha : float (default is 0.2)
            Value of alpha.
    
        beta : float (default is 0.5)
            Value of beta.
    
    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
            Output tensor
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    alpha = GET_ATTRIBUTE_FROM_OPERATION(op, 'alpha', default=0.2)
    beta = GET_ATTRIBUTE_FROM_OPERATION(op, 'beta', default=0.5)
    [value] = values
    value = alpha * value + beta
    return torch.clip(value, 0, 1)


def HardSwish_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    HardSwish takes one input data (Tensor) and produces one output data (Tensor) where the HardSwish function, 
        y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x), 
        where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
        Output tensor
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return F.hardswish(value)


def Identity_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    return values[0]


def InstanceNormalization_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    num_features = op.attributes.get('num_features', 1)
    eps = op.attributes.get('eps', 1e-05)
    affine = op.attributes.get('affine', False)
    if len(values) == 3:
        input_data, weight, bias = values
        running_mean, running_var = None, None
    elif len(values) == 1:
        input_data = values[0]
        running_mean, running_var, weight, bias = None, None, None, None
    else:
        raise ValueError(f'The number of input data in InstanceNom is {len(values)}')
    if affine:
        assert num_features == input_data.shape[1]
    output = F.instance_norm(input_data, running_mean, running_var, weight, bias, eps=eps)
    return output


def Interp_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    input_data = values[0]
    mode = op.attributes.get('mode', 'nearest')
    linear_mode_map = {(1): 'linear', (2): 'bilinear', (3): 'trilinear'}
    input_shape = input_data.shape
    align_corners = False if not op.attributes.get('align_corners') else True
    zoom_factor = op.attributes.get('zoom_factor', 1)
    shrink_factor = op.attributes.get('shrink_factor', 1)
    pad_beg = op.attributes.get('pad_beg', 0)
    pad_end = op.attributes.get('pad_end', 0)
    height_in_eff = input_shape[-2] + pad_beg + pad_end
    width_in_eff = input_shape[-1] + pad_beg + pad_end
    height_out = height_in_eff
    width_out = width_in_eff
    if zoom_factor != 1:
        height_out = height_in_eff + (height_in_eff - 1) * (zoom_factor - 1)
        width_out = width_in_eff + (width_in_eff - 1) * (zoom_factor - 1)
    if shrink_factor != 1:
        height_out = (height_in_eff - 1) // shrink_factor + 1
        width_out = (width_in_eff - 1) // shrink_factor + 1
    if bool(op.attributes.get('height', None)):
        height_out = op.attributes.get('height')
        width_out = op.attributes.get('width')
    if len(values) == 2:
        height_out, width_out = values[1].shape[-2:]
    sizes = list(input_shape[:2]) + [height_out, width_out]
    scales = None
    assert sizes[:2] == list(input_data.shape[:2])
    sizes = sizes[2:]
    mode = linear_mode_map[len(sizes)] if mode == 'linear' else mode
    output = F.interpolate(input_data, sizes, scales, mode, align_corners)
    return output


LSTM_FLATTEN_WEIGHT_ATTRIB = 'LSTM_FLATTEN_WEIGHT_ATTRIB'


def LSTM_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    只支持 pytorch 导出来的 LSTM 啊亲; 必须要 7 个输入 Variable

    Computes an one-layer LSTM. This operator is usually supported via some custom implementation such as CuDNN.

    Notations:

    X - input tensor

    i - input gate

    o - output gate

    f - forget gate

    c - cell gate

    t - time step (t-1 means previous time step)

    W[iofc] - W parameter weight matrix for input, output, forget, and cell gates

    R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates

    Wb[iofc] - W bias vectors for input, output, forget, and cell gates

    Rb[iofc] - R bias vectors for input, output, forget, and cell gates

    P[iof] - P peephole weight vector for input, output, and forget gates

    WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates

    RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates

    WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates

    RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates

    PB[iof] - P peephole weight vector for backward input, output, and forget gates

    H - Hidden state

    num_directions - 2 if direction == bidirectional else 1

    Activation functions:

    Relu(x)                - max(0, x)

    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

    Sigmoid(x)             - 1/(1 + e^{-x})

    (NOTE: Below are optional)

    Affine(x)              - alpha*x + beta

    LeakyRelu(x)           - x if x >= 0 else alpha * x

    ThresholdedRelu(x)     - x if x >= alpha else 0

    ScaledTanh(x)          - alpha*Tanh(beta*x)

    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

    Softsign(x)            - x/(1 + |x|)

    Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

    - Ct = ft (.) Ct-1 + it (.) ct

    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

    - Ht = ot (.) h(Ct)
    This operator has optional inputs/outputs. 
    See the doc for more details about the representation of optional arguments. 
    An empty string may be used in the place of an actual argument's name to indicate a missing argument. 
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

    Version
    This version of the operator has been available since version 7 of the default ONNX operator set.

    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
                for example (f, g, h) in LSTM. 
    
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01.
    
        activation_beta : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
            for example (f, g, h) in LSTM. 
            
            Default values are the same as of corresponding ONNX operators.
    
        activations : list of strings
            A list of 3 (or 6 if bidirectional) activation functions for input, 
            output, forget, cell, and hidden. 
            
            The activation functions must be one of the activation functions specified above. 
            Optional: See the equations for default if not specified.
    
        clip : float
            Cell clip threshold. Clipping bounds the elements of a tensor in the range of 
            [-threshold, +threshold] and is applied to the input of activations.
            No clip if not specified.
        
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional. 
            Must be one of forward (default), reverse, or bidirectional.

        hidden_size : int
            Number of neurons in the hidden layer
    
        input_forget : int (default is 0)
            Couple the input and forget gates if 1.
    
    Inputs (3 - 8)
        X : T
            The input sequences packed (and potentially padded) into one 3-D tensor 
                with the shape of `[seq_length, batch_size, input_size]`.
   
        W : T
            The weight tensor for the gates. Concatenation of `W[iofc]` and `WB[iofc]` 
            (if bidirectional) along dimension 0. The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
    
        R : T
            The recurrence weight tensor. Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0. 
            This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
    
        B (optional) : T
            The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, 
            and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. 
            
            This tensor has shape `[num_directions, 8*hidden_size]`. 
            Optional: If not specified - assumed to be 0.
    
        sequence_lens (optional) : T1
            Optional tensor specifying lengths of the sequences in a batch. 
            If not specified - assumed all sequences in the batch to have length `seq_length`. 
            It has shape `[batch_size]`.
        
        initial_h (optional) : T
            Optional initial value of the hidden. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        initial_c (optional) : T
            Optional initial value of the cell. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        P (optional) : T
            The weight tensor for peepholes.
            Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. 
            It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
    
    Outputs (0 - 3)
        Y (optional) : T
            A tensor that concats all the intermediate output values of the hidden. 
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
    
        Y_h (optional) : T
            The last output value of the hidden. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        Y_c (optional) : T
            The last output value of the cell. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=8)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, w, r = values[:3]
    b = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    initial_c = GET_VALUE_FROM_INPUTS(values, 6)
    p = GET_VALUE_FROM_INPUTS(values, 7)
    if p is not None:
        raise NotImplementedError('PPQ do not support LSTM with peepholes.')
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    input_forget = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='input_forget', default=0)
    layout = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if layout != 0:
        raise NotImplementedError('PPQ do not support LSTM with layout != 1.')
    if activation_alpha is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None:
        raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    bidirectional = direction == 'bidirectional'
    has_bias = b is not None
    if direction == 'reverse':
        raise NotImplementedError('GRU do not support reverse mode now.')
    if LSTM_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([w[0][hidden_size * 0:hidden_size * 1], w[0][hidden_size * 2:hidden_size * 3], w[0][hidden_size * 3:hidden_size * 4], w[0][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
        forward_r = torch.cat([r[0][hidden_size * 0:hidden_size * 1], r[0][hidden_size * 2:hidden_size * 3], r[0][hidden_size * 3:hidden_size * 4], r[0][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([b[0, hidden_size * 0:hidden_size * 1], b[0, hidden_size * 2:hidden_size * 3], b[0, hidden_size * 3:hidden_size * 4], b[0, hidden_size * 1:hidden_size * 2]]).contiguous()
            forward_bias_2 = torch.cat([b[0, hidden_size * 4:hidden_size * 5], b[0, hidden_size * 6:hidden_size * 7], b[0, hidden_size * 7:hidden_size * 8], b[0, hidden_size * 5:hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([w[1][hidden_size * 0:hidden_size * 1], w[1][hidden_size * 2:hidden_size * 3], w[1][hidden_size * 3:hidden_size * 4], w[1][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
            reverse_r = torch.cat([r[1][hidden_size * 0:hidden_size * 1], r[1][hidden_size * 2:hidden_size * 3], r[1][hidden_size * 3:hidden_size * 4], r[1][hidden_size * 1:hidden_size * 2]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([b[1, hidden_size * 0:hidden_size * 1], b[1, hidden_size * 2:hidden_size * 3], b[1, hidden_size * 3:hidden_size * 4], b[1, hidden_size * 1:hidden_size * 2]]).contiguous()
                reverse_bias_2 = torch.cat([b[1, hidden_size * 4:hidden_size * 5], b[1, hidden_size * 6:hidden_size * 7], b[1, hidden_size * 7:hidden_size * 8], b[1, hidden_size * 5:hidden_size * 6]]).contiguous()
        flatten_weight = [forward_w, forward_r]
        if has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:
            flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias:
            flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(LSTM_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(size=[s, x.shape[1], hidden_size], device=x.device, dtype=torch.float32)
    if initial_c is None:
        initial_c = torch.zeros(size=[s, x.shape[1], hidden_size], device=x.device, dtype=torch.float32)
    result = _VF.lstm(x, (initial_h, initial_c), op._detail[LSTM_FLATTEN_WEIGHT_ATTRIB], has_bias, 1, 0.0, False, bidirectional, False)
    hs, h, c = result
    if bidirectional:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 2, hs.shape[-1] // 2))
        hs = hs.permute((0, 2, 1, 3))
    else:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 1, hs.shape[-1]))
        hs = hs.permute((0, 2, 1, 3))
    return hs, h, c


def LayerNorm_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    """
    This is layer normalization defined in ONNX as function. 
    The overall computation can be split into two stages. 
    The first stage is standardization, which makes the normalized elements have zero mean and unit variances. 
    The computation required by standardization can be described by the following equations. 
    Mean = ReduceMean<axes=normalized_axes>(X) D = Sub(X, Mean) DD = Mul(Diff, Diff) 
    Var = ReduceMean<axes=normalized_axes>(DD) VarEps = Add(Var, epsilon) 
    StdDev = Sqrt(VarEps) InvStdDev = Reciprocal(StdDev) 
    Normalized = Mul(D, InvStdDev) where normalized_axes is [axis, ..., rank of X - 1]. 
    
    The variables Var and StdDev stand for variance and standard deviation, respectively. 
    The second output is Mean and the last one is InvStdDev. Depending on stash_type attribute, 
    the actual computation must happen in different floating-point precision. 
    
    For example, if stash_type is 1, this operator casts all input variables to 32-bit float, perform the computation, 
    and finally cast Normalized back to the original type of X. 
    
    The second stage then scales and shifts the outcome of the first stage using
        NormalizedScaled = Mul(Normalized, Scale) 
        Y = Add(NormalizedScaled, B) 
    The second stage doesn't depends on stash_type. All equations are in this syntax. 
    
    The same variable (i.e., input, output, and attribute) uses the same name in the equations above and this operator's definition. 
    Let d[i] indicate the i-th dimension of X. If X's shape is [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]], 
    the shape of Mean and InvStdDev is [d[0], ..., d[axis-1], 1, ..., 1]. Y and X have the same shape.

    Version
    This version of the operator has been available since version 17 of the default ONNX operator set.

    Attributes
        axis : int (default is -1)
            The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r]. 
            Negative value means counting dimensions from the back.

        epsilon : float (default is 1e-05)
            The epsilon value to use to avoid division by zero.

        stash_type : int (default is 1)
            Type of Mean and InvStdDev. 
            This also specifies stage one's computation precision.

    Inputs (2 - 3)
        X : T
            Tensor to be normalized.
        
        Scale : T
            Scale tensor.
        
        B (optional) : T
            Bias tensor.
    
    Outputs (1 - 3)
        Y : T
            Normalized tensor.
    
        Mean (optional) : U
            Saved mean used during training to speed up gradient computation
        
        InvStdDev (optional) : U
            Saved inverse standard deviation used during training to speed up gradient computation.

    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    x, weight = values[0], values[1]
    if len(values) == 3:
        bias = values[-1]
    else:
        bias = None
    eps = op.attributes.get('epsilon', 1e-05)
    axis = op.attributes.get('axis', -1)
    if axis != -1 and axis != x.ndim - 1:
        raise ValueError('Unsupported Layernorm axis. We will implement it soon.')
    normalized_shape = weight.shape
    output = F.layer_norm(x, normalized_shape, weight, bias, eps)
    return output


def LeakyRelu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    [input_data] = values
    alpha = op.attributes.get('alpha', 0.01)
    output = F.leaky_relu(input_data, alpha)
    return output


def Less_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_a, input_b = values
    if input_a.dim() >= input_b.dim() or input_a.shape > input_b.shape:
        output = torch.lt(input_a, input_b)
    else:
        output = torch.gt(input_b, input_a)
    return output


def Log_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    output = torch.log(input_data)
    return output


def Softmax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """The operator computes the normalized exponential values for the given
    input:

    Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

    The "axis" attribute indicates the dimension along which Softmax will be performed.
    The output tensor has the same shape and contains the Softmax values of the corresponding input.

    Attributes
        axis : int (default is -1)
            Describes the dimension Softmax will be performed on.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(input).

    Inputs
        input (differentiable) : T
        The input tensor of rank >= axis.

    Outputs
        output (differentiable) : T
        The output values with the same shape as the input tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    output = F.softmax(input, axis)
    return output


def LogSoftmax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    x = Softmax_forward(op=op, values=values, ctx=ctx, kwargs=kwargs)
    x = Log_forward(op=op, values=[x], ctx=ctx, kwargs=kwargs)
    return x


def FORCE_CONVERT_DEVICE(value: torch.Tensor, device: str) ->torch.Tensor:
    return value


def MMCVRoiAlign_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    data, rois = values
    rois = FORCE_CONVERT_DEVICE(rois, device=data.device)
    mode = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='mode', default='avg')
    aligned = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='aligned', default=True)
    output_height = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_height', default=1)
    output_width = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_width', default=1)
    sampling_ratio = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sampling_ratio', default=0)
    spatial_scale = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='spatial_scale', default=1.0)
    output_size = output_height, output_width
    if rois.shape[0] == 0:
        output = torch.empty([0, data.shape[1], 14, 14])
    else:
        output = mmcv_roi_align(data, rois, output_size, spatial_scale, sampling_ratio, mode, aligned)
    return output


def MatMul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    output = torch.matmul(values[0], values[1])
    return output


def MaxPool2d_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])
    [x] = values
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    if op.type == 'GlobalMaxPool':
        kernel_size = x.size()[2:]
    else:
        kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    ndim = x.ndim
    if ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.max_pool3d(x, kernel_size=kernel_size, padding=torch_pads, dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    elif ndim == 4:
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=convert_onnx_pads_to_torch(onnx_pads), value=float('-inf'))
                onnx_pads = 0
        output = F.max_pool2d(x, kernel_size=kernel_size, padding=onnx_pads, dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    elif ndim in {2, 3}:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.max_pool1d(x, kernel_size=kernel_size, padding=torch_pads, dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def Mul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Performs element-wise binary multiplication (with Numpy-style
    broadcasting support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

    Inputs
        A (differentiable) : T
            First operand.

        B (differentiable) : T
            Second operand.

    Outputs
        C (differentiable) : T
            Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    multiplicand, multiplier = values
    return multiplicand * multiplier


def MultiHeadAttention_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Perform MultiHeadAttetion opr forward.

    Args:
        op (Operation): MultiHeadAttention
        values (List[torch.Tensor]): opr inputs
        ctx (TorchBackendContext, optional): Context. Defaults to None.

    Raises:
        NotImplementedError: In [Vit Paper](https://arxiv.org/abs/2010.11929), MultiHeadAttention inputs are actually the same tensor, we suppose that this would **not** be simplified.
        ValueError: MultiHeadAttention contains `embed_dim` and `num_heads`.

    Returns:
        list: opr output and internal result for quantization.
    """
    if len(values) != 11:
        raise NotImplementedError('Not implement simplified MultiHeadAttention')
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    q_in, k_in, v_in, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b = values
    embed_dim = op.attributes.get('embed_dim')
    num_heads = op.attributes.get('num_heads')
    if embed_dim is None or num_heads is None:
        raise ValueError('Cannot fetch embed_dim or num_heads')
    batch_size = q_in.shape[0]
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    xq = F.linear(q_in, q_w, q_b)
    xk = F.linear(k_in, k_w, k_b)
    xv = F.linear(v_in, v_w, v_b)
    B, N, _ = xq.shape
    q = xq.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k = xk.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    v = xv.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    energy = q @ k.transpose(-2, -1) * scale
    attn = energy.softmax(dim=-1)
    feat = (attn @ v).transpose(1, 2).reshape(batch_size, -1, embed_dim)
    out = F.linear(feat, o_w, o_b)
    return out


def Neg_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    Neg takes one input data (Tensor) and produces one output data (Tensor)
    where each element flipped sign, y = -x, is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
        Input tensor

    Outputs
        Y (differentiable) : T
        Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return -x


def NonZero_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    output = torch.nonzero(input_value, as_tuple=True)
    output = torch.stack(output)
    return output


def Not_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return ~value


def Onehot_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Produces a one-hot tensor based on inputs. The locations represented by
    the index values in the 'indices' input tensor will have 'on_value' and the
    other locations will have 'off_value' in the output tensor,

    where 'on_value' and 'off_value' are specified as part of required input argument 'values',
    which is a two-element tensor of format [off_value, on_value].

    The rank of the output tensor will be one greater than the rank of the input tensor.
    The additional dimension is for one-hot representation. The additional dimension will be inserted at the position specified by 'axis'.
    If 'axis' is not specified then then additional dimension will be inserted as the innermost dimension,
    i.e. axis=-1. The size of the additional dimension is specified by required scalar input 'depth'.

    The type of the output tensor is the same as the type of the 'values' input. Any entries in the 'indices'
    input tensor with values outside the range [-depth, depth-1] will result in one-hot representation
    with all 'off_value' values in the output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Attributes
    axis : int (default is -1)
    (Optional) Axis along which one-hot representation in added. Default: axis=-1. axis=-1 means that
        the additional dimension will be inserted as the innermost/last dimension in the output tensor.
    Negative value means counting dimensions from the back. Accepted range is [-r-1, r] where r = rank(indices).

    Inputs
    indices (non-differentiable) : T1
        Input tensor containing indices. Any entries in the 'indices' input tensor with values outside the range [-depth, depth-1]
            will result in one-hot representation with all 'off_value' values in the output tensor.In case 'indices' is of non-integer type,
            the values will be casted to int64 before use.

    depth (non-differentiable) : T2
        Scalar specifying the number of classes in one-hot tensor.
        This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor.
            The values in the 'indices' input tensor are expected to be in the range [-depth, depth-1].
            In case 'depth' is of non-integer type, it will be casted to int64 before use.

    values (non-differentiable) : T3
        Rank 1 tensor containing exactly two elements,
        in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor,
        and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor.

    Outputs
    output (non-differentiable) : T3
        Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1.
        The data type for the elements of the output tensor is the same as the type of input 'values' is used.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    indices, depth, values = values
    off_value, on_value = values
    out = F.one_hot(indices, depth.item())
    out = out * (on_value - off_value) + off_value
    rank = len(indices.shape)
    if axis < 0:
        axis += rank + 1
    if not rank == axis:
        order = list(range(len(indices.shape)))
        order.insert(axis, -1)
        out = out.permute(order)
    return out


def PPQDeviceSwitch_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    [value] = values
    return value


def PRelu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data, weight = values
    output = F.prelu(input_data, weight)
    return output


def Pad_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    """
    Given a tensor containing the data to be padded (data), 
        a tensor containing the number of start and end pad values for axis (pads), 
        (optionally) a mode, and (optionally) constant_value, a padded tensor (output) is generated.

    The three supported modes are (similar to corresponding modes supported by numpy.pad):

        constant(default) - pads with a given constant value as specified by constant_value 
            (which defaults to 0, empty string, or False)

        reflect - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

        edge - pads with the edge values of array

    Version
        This version of the operator has been available since version 18 of the default ONNX operator set.

        Other versions of this operator: 1, 2, 11, 13

    Attributes
        mode : string (default is constant)
        Supported modes: `constant`(default), `reflect`, `edge`
    
    Inputs (2 - 4)
        data (differentiable) : T
            Input tensor.
    
        pads (non-differentiable) : tensor(int64)
            Tensor of integers indicating the number of padding elements to add or remove 
            (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. 
            `pads` should be a 1D tensor of shape [2 * num_axes] where `num_axes` refers to the number of elements 
            in the `axes` input or the input rank if `axes` are not provided explicitly. 
            
            `pads` format should be: [x1_begin, x2_begin, ..., x1_end, x2_end,...], 
            where xi_begin is the number of pad values added at the beginning of 
            axis `axes[i]` and xi_end, the number of pad values added at the end of axis `axes[i]`.
    
        constant_value (optional, non-differentiable) : T
            (Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0, empty string or False).
        
        axes (optional, non-differentiable) : Tind
            1-D tensor of axes that `pads` apply to. Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated. 
            If not provided, all axes are assumed (`[0, 1, ..., input_rank-1]`).
    
    Outputs
        output (differentiable) : T
            Tensor after padding.

    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    mode = op.attributes.get('mode', 'constant')
    value, pads = values[0], values[1]
    if len(values) > 3:
        raise ValueError('Unsupported pad version, except at most 2 input variable.')
    pads = pads.tolist()
    pads = convert_onnx_pads_to_torch(pads)
    if mode == 'constant':
        if len(values) == 3 and values[-1] is None:
            constant_value = 0
        else:
            constant_value = values[-1].item() if len(values) == 3 else 0
        output = F.pad(value, pads, mode, constant_value)
    elif mode == 'reflect':
        output = value
        while len(pads) > 4:
            output = F.pad(value, pads[-4:], mode)
            pads = pads[:-4]
        output = F.pad(value, pads, mode)
    elif mode == 'edge':
        output = F.pad(value, pads, 'replicate')
    else:
        raise TypeError(f'Unsupported mode {mode} in Pad op')
    return output


def Parameter_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    m = op.attributes.get('m', -1)
    n = op.attributes.get('n', -1)
    output = input_data.reshape(m, n)
    return output


def Pow_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data = values[0]
    power = op.attributes.get('power', 1)
    scale = op.attributes.get('scale', 1)
    shift = op.attributes.get('shift', 0)
    if len(values) == 2:
        power = values[1]
        scale, shift = 1.0, 0.0
    output = torch.pow(input_data * scale + shift, power)
    return output


def Range_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    """
    Generate a tensor containing a sequence of numbers that 
        begin at start and extends by increments of delta up to limit (exclusive).

    The number of elements in the output of range is computed as below-

    number_of_elements = max( ceil( (limit - start) / delta ) , 0 )

    The pseudocode determining the contents of the output is shown below-

    for(int i=0; i<number_of_elements; ++i)

    {

        output[i] = start + (i * delta);

    }

    Example 1 Inputs: start = 3, limit = 9, delta = 3 Output: [3, 6]

    Example 2 Inputs: start = 10, limit = 4, delta = -2 Output: [10, 8, 6]

    Inputs
        start : T
            Scalar. First entry for the range of output values.
    
        limit : T
            Scalar. Exclusive upper limit for the range of output values.
    
        delta : T
            Scalar. Value to step by.
    
    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.
    """
    start, limit, delta = values
    start = start.item()
    limit = limit.item()
    delta = delta.item()
    output = torch.arange(start, limit, delta, device=ctx.executing_device)
    return output


def Reciprocal_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    Reciprocal takes one input data (Tensor) and produces one output data (Tensor) where the reciprocal is,
        y = 1/x, is applied to the tensor elementwise.

    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

    Inputs
        X (differentiable) : T Input tensor
    Outputs
        Y (differentiable) : T Output tensor

    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return 1 / x


def ReduceL2_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    [input_value] = values
    axis = op.attributes['axes']
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.norm(input_value, dim=axis, keepdim=keepdim)
    if axis is None and keepdim:
        output = output.reshape([1] * input_value.dim())
    return output


def ReduceMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    elif dim is None:
        output = torch.max(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output, _ = torch.max(input_value, dim=dim[0], keepdim=keepdim)
    return output


def ReduceMean_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    elif dim is None:
        output = torch.mean(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output = torch.mean(input_value, dim=dim, keepdim=keepdim)
    return output


def ReduceSum_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
        input_value, dim = values[0], None
        if len(values) > 1:
            dim = values[1]
        keepdim, noop_with_empty_axes = bool(op.attributes.get('keepdims', 1)), op.attributes.get('noop_with_empty_axes', 0)
        if dim is None:
            if noop_with_empty_axes:
                return input_value
            else:
                output = torch.sum(input_value)
                if keepdim:
                    output = output.reshape([1] * input_value.dim())
                return output
        else:
            dim = dim.tolist()
            if isinstance(dim, int):
                dim = [dim]
            output = torch.sum(input_value, dim=dim, keepdim=keepdim)
            return output
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if dim is None:
        output = torch.sum(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output = torch.sum(input_value, dim=dim, keepdim=keepdim)
    return output


def Reshape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Reshape the input tensor similar to numpy.reshape. First input is the
    data tensor, second input is a shape tensor which specifies the output
    shape. It outputs the reshaped tensor.

    At most one dimension of the new shape can be -1.
    In this case, the value is inferred from the size of the tensor and the remaining dimensions.
    A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
    If 'allowzero' is set, and the new shape includes 0,
        the dimension will be set explicitly to zero (i.e. not taken from input tensor)

    Attributes
        allowzero : int (default is 0)
        (Optional) By default, when any value in the 'shape' input is equal to zero
            the corresponding dimension value is copied from the input tensor dynamically.

        allowzero=1 indicates that if any value in the 'shape' input is set to zero,
            the zero value is honored, similar to NumPy.
    Inputs
        data (differentiable) : T
            An input tensor.

        shape (non-differentiable) : tensor(int64)
            Specified shape for output.

    Outputs
        reshaped (differentiable) : T
            Reshaped data.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    data, shape = values
    shape = shape.cpu()
    shape = [(shape[i] if shape[i] != 0 else data.shape[i]) for i in range(len(shape))]
    shape = [(shape[i].item() if hasattr(shape[i], 'item') else shape[i]) for i in range(len(shape))]
    return data.reshape(shape)


def Resize_forward(op, input_value, device=None):
    """NXP Platform has a custimized resize operation implementation, which
    gives different result from torch.nn.Resize. To correctly simulate hardware
    beviour and have the same result with NXP, it is necessary to force resize
    to run with nearest mode. Any other mode of resize will be ignored by this
    function.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    input_data = input_value[0]
    scales = input_value[2] if len(input_value) > 2 else None
    sizes = input_value[-1].tolist() if len(input_value) == 4 else None
    mode = 'nearest'
    if sizes is None or len(sizes) == 0:
        sizes = None
        if scales.numel() == 1:
            scales = scales.item()
        else:
            assert scales.numel() % 2 == 0
            scales = scales[-2].cpu().numpy().tolist()
    else:
        scales = None
        assert sizes[:2] == list(input_data.shape[:2])
        sizes = sizes[2:]
    trans_mode = op.attributes.get('coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        output = F.interpolate(input_data, sizes, scales, mode, align_corners=True)
    else:
        output = F.interpolate(input_data, sizes, scales, mode)
    return output


def RoiAlign_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    from torchvision.ops import roi_align as torch_roi_align
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value = values[0]
    rois = values[1]
    rois = rois
    output_height = op.attributes.get('output_height', 1)
    output_width = op.attributes.get('output_width', 1)
    sampling_ratio = op.attributes.get('sampling_ratio', 0)
    spatial_scale = op.attributes.get('spatial_scale', 1.0)
    if isinstance(rois, torch.Tensor):
        if rois.shape[1] == 5:
            boxes = rois
        elif rois.shape[1] == 4:
            boxes = [rois]
        else:
            raise ValueError(f'Unsupported rois shape {rois.shape}')
    else:
        raise TypeError('Unsupported rois type')
    output_size = output_height, output_width
    output = torch_roi_align(value, boxes, output_size, spatial_scale, sampling_ratio)
    return output


def Scale_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    assert len(values) >= 2
    input_data = values[0]
    scale = values[1]
    bias_term = op.attributes.get('bias_term', False)
    axis = op.attributes.get('axis', 1)
    scale_shape = list(scale.shape)
    for i in range(axis):
        scale_shape.insert(0, 1)
    for i in range(input_data.dim() - scale.dim() - axis):
        scale_shape.append(1)
    scale = scale.reshape(scale_shape)
    scale = scale.expand_as(input_data)
    if bias_term:
        bias = values[2]
        bias = bias.reshape(scale_shape).expand_as(input_data)
        output = input_data * scale + bias
    else:
        output = input_data * scale
    return output


def ScatterElements_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    """
    ScatterElements takes three inputs data, updates, 
    and indices of the same rank r >= 1 and an optional attribute axis that identifies an axis of data 
    (by default, the outer-most axis, that is axis 0). 
    
    The output of the operation is produced by creating a copy of the input data, 
    and then updating its value to values specified by updates at specific index positions specified by indices. 
    Its output shape is the same as the shape of data.

    For each entry in updates, 
    the target index in data is obtained by combining the corresponding entry in indices with the index of the entry itself: 
        the index-value for dimension = axis is obtained from the value of the corresponding entry in indices and the index-value 
    for dimension != axis is obtained from the index of the entry itself.

    reduction allows specification of an optional reduction operation, 
        which is applied to all values in updates tensor into output at the specified indices.
    In cases where reduction is set to "none", indices should not have duplicate entries: that is, 
        if idx1 != idx2, then indices[idx1] != indices[idx2]. 
    
    For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] = updates[i][j] if axis = 0,
    output[i][indices[i][j]] = updates[i][j] if axis = 1,
    When reduction is set to "add", the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] += updates[i][j] if axis = 0,
    output[i][indices[i][j]] += updates[i][j] if axis = 1,
    When reduction is set to "mul", the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] *= updates[i][j] if axis = 0,
    output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

    Attributes
        axis : int (default is 0)
            Which axis to scatter on. Negative value means counting dimensions from the back.
                Accepted range is [-r, r-1] where r = rank(data).
            reduction : string (default is none)
            Type of reduction to apply: none (default), add, mul. 

            'none': no reduction applied. 
            'add': reduction using the addition operation. 
            'mul': reduction using the multiplication operation.

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input). 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, indices, updates = values
    dim = op.attributes.get('axis', 0)
    indices[indices < 0] += value.shape[dim]
    output = value.scatter(dim, indices, updates)
    return output


def ScatterND_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """OPSET 11: ScatterND takes three inputs data tensor of rank r >= 1,
    indices tensor of rank q >= 1,

        and updates tensor of rank q + r - indices.shape[-1] - 1.

    The output of the operation is produced by creating a copy of the input data,
    and then updating its value to values specified by updates at specific index positions specified by indices.
    Its output shape is the same as the shape of data. Note that indices should not have duplicate entries.
    That is, two or more updates for the same index-location is not supported.

    indices is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of indices.
    indices is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into data.
    Hence, k can be a value at most the rank of data. When k equals rank(data),
    each update entry specifies an update to a single element of the tensor.
    When k is less than rank(data) each update entry specifies an update to a slice of the tensor.

    updates is treated as a (q-1)-dimensional tensor of replacement-slice-values.
    Thus, the first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
    The remaining dimensions of updates correspond to the dimensions of the replacement-slice-values.
    Each replacement-slice-value is a (r-k) dimensional tensor, corresponding to the trailing (r-k) dimensions of data.
    Thus, the shape of updates must equal indices.shape[0:q-1] ++ data.shape[k:r-1],
    where ++ denotes the concatenation of shapes.

    Inputs

        data : T
            Tensor of rank r >= 1.

        indices : tensor(int64)
            Tensor of rank q >= 1.

        updates : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs

        output : T
            Tensor of rank r >= 1.

    Args:
        op ([type]): [description]
        values ([type]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, indices, updates = values
    output = value.clone()
    ind_dim = indices.dim()
    indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
    updates = updates.reshape((-1, *updates.shape[ind_dim - 1:]))
    output[indices] = updates
    return output


def Shape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Takes a tensor as input and outputs an 1D int64 tensor containing the
    shape of the input tensor.

    Version
        This version of the operator has been available since version 1 of the default ONNX operator set.

    Inputs
        data : T
        An input tensor.

    Outputs
        shape : T1
        Shape of the input tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    shape_tensor = torch.Tensor([k for k in value.shape]).long()
    return shape_tensor


def Sin_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Calculates the sine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The sine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.sin(x)


def Slice_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Produces a slice of the input tensor along multiple axes. Similar to
    numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slices uses starts, ends, axes and steps inputs to specify the start and
    end dimension and step for each axis in the list of axes,

    it uses this information to slice the input data tensor.
    If a negative value is passed for any of the start or end indices,
        it represents number of elements before the end of that dimension.

    If the value passed to start or end is larger than the n (the number of elements in this dimension),
        it represents n. For slicing to the end of a dimension with unknown size,
        it is recommended to pass in INT_MAX when sclicing forward and 'INT_MIN' when slicing backward.

    If a negative value is passed for step, it represents slicing backward.
    However step value cannot be 0. If axes are omitted, they are set to [0, ..., ndim-1].

    If steps are omitted, they are set to [1, ..., 1] of length len(starts)

    Example 1: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] axes = [0, 1] starts = [1, 0] ends = [2, 3] steps = [1, 2] result = [ [5, 7], ]
    Example 2: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] starts = [0, 1] ends = [-1, 1000] result = [ [2, 3, 4], ]

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=5)
    data, starts, ends = values[:3]
    axes = values[3] if len(values) > 3 else torch.tensor([int(_) for idx, _ in enumerate(starts.tolist())])
    steps = values[4] if len(values) > 4 else torch.ones_like(starts)
    if axes is not None:
        axes = axes.tolist()
    starts, ends, steps = starts.tolist(), ends.tolist(), steps.tolist()
    slices, flip_dims = {}, []
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step
        slices[axis] = slice(int(start), int(end), int(step))
    pos_axes_slices = list(slices.get(a, slice(None, None)) for a in range(max(axes) + 1))
    neg_axes_slices = list(slices.get(a, slice(None, None)) for a in range(min(axes), 0))
    if neg_axes_slices:
        neg_axes_slices = [Ellipsis] + neg_axes_slices
    if flip_dims:
        data = torch.flip(data, dims=flip_dims)
    if pos_axes_slices:
        data = data[pos_axes_slices]
    if neg_axes_slices:
        data = data[neg_axes_slices]
    return data


def Softplus_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    input_data = values[0]
    output = torch.log(torch.exp(input_data) + 1)
    return output


def SpaceToDepth_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input_data = values[0]
    downsample = op.attributes.get('blocksize', 1)
    output = F.pixel_unshuffle(input_data, downsample)
    return output


def Split_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
        value = values[0]
        axis = op.attributes.get('axis', 0)
        split = value.shape[axis] // len(op.outputs)
        if len(values) > 1:
            split = values[1]
            split = split.tolist()
    else:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        axis = op.attributes.get('axis', 0)
        split = op.attributes.get('split', 0)
        [value] = values
        if 'split' not in op.attributes:
            split = value.shape[axis] // len(op.outputs)
    outputs = torch.split(value, split, axis)
    return outputs


def Sqrt_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    output = torch.sqrt(input_data)
    return output


def Squeeze_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Remove single-dimensional entries from the shape of a tensor. Takes an
    input axes with a list of axes to squeeze. If axes is not provided, all the
    single dimensions will be removed from the shape. If an axis is selected
    with shape entry not equal to one, an error is raised.

    Inputs (1 - 2)
        data (differentiable) : T
        Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
        List of integers indicating the dimensions to squeeze.
        Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

    Outputs
        squeezed (differentiable) : T
        Reshaped tensor with same data as input.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        squeezing_tensor = values[0]
        axes = [axis for axis in range(squeezing_tensor.ndim) if squeezing_tensor.shape[axis] == 1]
        if len(values) > 1:
            axes = values[1]
            axes = axes.tolist()
    else:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        [squeezing_tensor], axes = values, GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=False, default=None)
    if axes is None:
        axes = []
        shape = squeezing_tensor.shape
        for dim, s in enumerate(shape):
            if s == 1:
                axes.append(dim)
    if isinstance(axes, list):
        for squeezing_dim in sorted(axes, reverse=True):
            squeezing_tensor = torch.squeeze(squeezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        squeezing_tensor = torch.squeeze(squeezing_tensor, axes)
    else:
        raise TypeError(f'Parameter axes of operation {op.name} misunderstood, expect int value of list of int, while {type(axes)} was given.')
    return squeezing_tensor


def Sum_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """
    Element-wise sum of each of the input tensors (with Numpy-style broadcasting support). 
    All inputs and outputs must have the same data type. 
    This operator supports multidirectional (i.e., Numpy-style) broadcasting; 
    for more details please check the doc.

    Version
    This version of the operator has been available since version 13 of the default ONNX operator set.

    Other versions of this operator: 1, 6, 8

    Inputs (1 - ∞)
        data_0 (variadic, differentiable) : T
            List of tensors for sum.
    Outputs
        sum (differentiable) : Tq
            Output tensor.
    
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    if op.platform != TargetPlatform.SOI:
        target_device = ctx.executing_device
    else:
        target_device = 'cpu'
    output = torch.zeros_like(values[0])
    for value in values:
        output += value
    return output


def Tan_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    output = torch.tan(input_data)
    return output


def Tanh_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    input_data = values[0]
    output = torch.tanh(input_data)
    return output


def Tile_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Constructs a tensor by tiling a given tensor. This is the same as
    function tile in Numpy, but no broadcast.

    For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

    Version
        This version of the operator has been available since version 6 of the default ONNX operator set.

    Inputs
        input : T
            Input tensor of any shape.

        repeats : T1
            1D int64 tensor of the same length as input's dimension number,
            includes numbers of repeated copies along input's dimensions.

    Outputs
        output : T
            Output tensor of the same dimension and type as tensor input.
            output_dim[i] = input_dim[i] * repeats[i]

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    input, repeats = values
    axis = op.attributes.get('axis', None)
    tiles = op.attributes.get('tiles', None)
    if axis is not None:
        repeats = [(1) for _ in range(input.ndim)]
        repeats[axis] = tiles
    else:
        repeats = convert_any_to_python_primary_type(values[-1])
        if not isinstance(repeats, list):
            repeats = [repeats]
    assert input.ndim == len(repeats)
    output = input.repeat(tuple(repeats))
    return output


def TopK_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Retrieve the top-K largest or smallest elements along a specified axis.
    Given an input tensor of shape [a_1, a_2, ..., a_n, r] and integer argument
    k, return two outputs: -Value tensor of shape [a_1, a_2, ..., a_{axis-1},
    k, a_{axis+1}, ... a_n] which contains the values of the top k elements.

    along the specified axis - Index tensor of shape [a_1, a_2, ...,
    a_{axis-1}, k, a_{axis+1}, ... a_n] which contains the indices of the top k
    elements (original indices from the input tensor).

    If "largest" is 1 (the default value) then the k largest elements are returned.
    If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
    If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

    Given two equivalent values, this operator uses the indices along the axis as a tiebreaker.
    That is, the element with the lower index will appear first.

    Attributes
        axis : int (default is -1)
            Dimension on which to do the sort. Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(input).

        largest : int (default is 1)
            Whether to return the top-K largest or smallest elements.

        sorted : int (default is 1)
            Whether to return the elements in sorted order.

    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]

        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve

    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing top K values from the input tensor

        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing the corresponding input tensor indices for the top K values.

    Args:
        op (Operation): [description]
        input_value ([type]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    largest = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='largest', default=1)
    sorted = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sorted', default=1)
    largest, sorted = bool(largest), bool(sorted)
    x, k = values
    k = convert_any_to_python_primary_type(k)
    values, indices = torch.topk(input=x, k=k, dim=axis, largest=largest, sorted=sorted)
    return values, indices


def Transpose_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Transpose the input tensor similar to numpy.transpose. For example, when
    perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
    will be (2, 1, 3).

    Attributes
        perm : list of ints
            A list of integers. By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

    Inputs
        data (differentiable) : T
            An input tensor.

    Outputs
        transposed (differentiable) : T
            Transposed output.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    perm = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='perm', compulsive=True)
    [data] = values
    output = data.permute(perm)
    return output


def UnaryEltwise_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    [input_value] = values
    if op.type == 'Exp':
        output = torch.exp(input_value)
    elif op.type == 'Sigmoid':
        output = torch.sigmoid(input_value)
    elif op.type == 'Relu':
        output = F.relu(input_value)
    else:
        logger.warning('Not UnaryEltwise op, return input as output')
        output = input_value
    return output


def Unsqueeze_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Insert single-dimensional entries to the shape of an input tensor
    (data).

    Takes one required argument axes - which contains a list of dimension indices and
        this operator will insert a dimension of value 1 into the corresponding
        index of the output tensor (expanded).

    For example: Given an input tensor (data) of shape [3, 4, 5],
        then Unsqueeze(data, axes=[0, 4]) outputs a tensor (expanded)
        containing same data as data but with shape [1, 3, 4, 5, 1].

    The attribute axes should not contain any duplicate entries.
    It is an error if it contains duplicates.

    The rank of the output tensor (output_rank) is the rank of the input tensor (data)
        plus the number of values in axes.

    Each value in axes should be within the (inclusive) range [-output_rank , output_rank - 1].
    The order of values in axes does not matter and can come in any order.

    Attributes
        axes : list of ints (required)
            List of integers indicating the dimensions to be inserted.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(expanded).

    Inputs
        data : T
            Original tensor

    Outputs
        expanded : T
            Reshaped tensor with same data as input.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
        unsqueezing_tensor, axes = values
        axes = axes.tolist()
    else:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        [unsqueezing_tensor] = values
        axes = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=True)
    if isinstance(axes, list):
        for squeezing_dim in sorted(axes, reverse=True):
            unsqueezing_tensor = torch.unsqueeze(unsqueezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        unsqueezing_tensor = torch.unsqueeze(unsqueezing_tensor, axes)
    else:
        raise TypeError(f'Parameter axes of operation {op.name} misunderstood, expect int value of list of int, while {type(axes)} was given.')
    return unsqueezing_tensor


def Where_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    condition, x, y = values
    output = torch.where(condition, x, y)
    return output


def _NMS_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext=None, **kwargs) ->torch.Tensor:
    """Filter out boxes that have high intersection-over-union (IOU) overlap
    with previously selected boxes. Bounding boxes with score less than
    score_threshold are removed. Bounding box format is indicated by attribute
    center_point_box.

    Note that this algorithm is agnostic to where the origin is in the coordinate system and
    more generally is invariant to orthogonal transformations and translations of the coordinate system;

    thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm.
    The selected_indices output is a set of integers indexing into the input collection of
        bounding boxes representing the selected boxes.
    The bounding box coordinates corresponding to the selected indices
        can then be obtained using the Gather or GatherND operation.

    Attributes
        center_point_box : int (default is 0)
        Integer indicate the format of the box data.
        The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2)
            are the coordinates of any diagonal pair of box corners and the coordinates can be provided as normalized
            (i.e., lying in the interval [0, 1]) or absolute.
            Mostly used for TF models. 1 - the box data is supplied as
                [x_center, y_center, width, height]. Mostly used for Pytorch models.

    Inputs (2 - 5)
        boxes : tensor(float)
        An input tensor with shape [num_batches, spatial_dimension, 4].
        The single box data format is indicated by center_point_box.

        scores : tensor(float)
        An input tensor with shape [num_batches, num_classes, spatial_dimension]

        max_output_boxes_per_class (optional) : tensor(int64)
        Integer representing the maximum number of boxes to be selected per batch per class.
        It is a scalar. Default to 0, which means no output.

        iou_threshold (optional) : tensor(float)
        Float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
            It is scalar. Value range [0, 1]. Default to 0.

        score_threshold (optional) : tensor(float)
        Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
        selected indices from the boxes tensor. [num_selected_indices, 3],
            the selected index format is [batch_index, class_index, box_index].

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    boxes, scores = values[:2]
    max_output_boxes_per_class = values[2].item() if len(values) > 2 else 0
    iou_threshold = values[3].item() if len(values) > 3 else 0
    score_threshold = values[4].item() if len(values) > 4 else 0
    center_point_box = op.attributes.get('center_point_box', 0)
    batch, num_classes = boxes.shape[0], scores.shape[1]
    output = []
    for i in range(batch):
        sub_boxes = boxes[i]
        sub_scores = scores[i]
        if center_point_box:
            sub_boxes = torch.stack((sub_boxes[:, 0] - sub_boxes[:, 2] / 2, sub_boxes[:, 1] - sub_boxes[:, 3] / 2, sub_boxes[:, 0] + sub_boxes[:, 2] / 2, sub_boxes[:, 1] + sub_boxes[:, 3] / 2), dim=1)
        for j in range(num_classes):
            """# Strange speed, Revert back to original implementation.

            # May lead to error if boxes are not in the order given above

            sorted_boxes = torch.tensor([[min(item[0], item[2]), min(item[1], item[3]), max(item[0], item[2]),
                                          max(item[1], item[3])] for item in sub_boxes], device=device)
            keep = mmcv.ops.nms(sorted_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            """
            keep = mmcv.ops.nms(sub_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            keep = keep[sub_scores[j][keep] > score_threshold]
            keep = keep[:max_output_boxes_per_class]
            keep = torch.stack((torch.full_like(keep, i), torch.full_like(keep, j), keep), dim=1)
            output.append(keep)
    output = torch.cat(output)
    return output


DEFAULT_BACKEND_TABLE = {'Abs': Abs_forward, 'AdaptiveAvgPool2d': AdaptiveAvgPool2d_forward, 'And': And_forward, 'Add': Add_forward, 'ArgMax': ArgMax_forward, 'AveragePool': AveragePool_forward, 'BatchNormalization': BatchNormalization_forward, 'Cast': Cast_forward, 'Clip': Clip_forward, 'Concat': Concat_forward, 'Constant': Constant_forward, 'ConstantOfShape': ConstantOfShape_forward, 'Conv': Conv_forward, 'ConvTranspose': ConvTranspose_forward, 'Cos': Cos_forward, 'Div': Eltwise_forward, 'Equal': Equal_forward, 'Exp': UnaryEltwise_forward, 'Expand': Expand_forward, 'Flatten': Flatten_forward, 'Gather': Gather_forward, 'GatherElements': Gather_forward, 'GatherND': GatherND_forward, 'Gelu': Gelu_forward, 'Gemm': Gemm_forward, 'grid_sampler': GridSampler_forward, 'GlobalAveragePool': AveragePool_forward, 'GlobalMaxPool': MaxPool2d_forward, 'Greater': Greater_forward, 'LayerNorm': LayerNorm_forward, 'LayerNormalization': LayerNorm_forward, 'LeakyRelu': LeakyRelu_forward, 'Less': Less_forward, 'LogSoftmax': LogSoftmax_forward, 'MatMul': MatMul_forward, 'Max': Eltwise_forward, 'MaxPool': MaxPool2d_forward, 'Min': Eltwise_forward, 'Mul': Mul_forward, 'MultiHeadAttention': MultiHeadAttention_forward, 'NonMaxSuppression': _NMS_forward, 'NonZero': NonZero_forward, 'Not': Not_forward, 'Pad': Pad_forward, 'PRelu': PRelu_forward, 'Range': Range_forward, 'ReduceL2': ReduceL2_forward, 'ReduceMax': ReduceMax_forward, 'ReduceMean': ReduceMean_forward, 'ReduceSum': ReduceSum_forward, 'Relu': UnaryEltwise_forward, 'Reshape': Reshape_forward, 'Resize': Resize_forward, 'ScatterElements': ScatterElements_forward, 'ScatterND': ScatterND_forward, 'Shape': Shape_forward, 'Sigmoid': UnaryEltwise_forward, 'Sin': Sin_forward, 'Slice': Slice_forward, 'Softmax': Softmax_forward, 'Softplus': Softplus_forward, 'Split': Split_forward, 'Squeeze': Squeeze_forward, 'Sub': Eltwise_forward, 'Tile': Tile_forward, 'TopK': TopK_forward, 'Transpose': Transpose_forward, 'Unsqueeze': Unsqueeze_forward, 'Where': Where_forward, 'Sqrt': Sqrt_forward, 'Log': Log_forward, 'Floor': Floor_forward, 'RoiAlign': RoiAlign_forward, 'MMCVRoiAlign': MMCVRoiAlign_forward, 'SpaceToDepth': SpaceToDepth_forward, 'DepthToSpace': DepthToSpace_forward, 'Scale': Scale_forward, 'Tanh': Tanh_forward, 'Tan': Tan_forward, 'Pow': Pow_forward, 'ChannelShuffle': ChannelShuffle_forward, 'InstanceNormalization': InstanceNormalization_forward, 'Parameter': Parameter_forward, 'Interp': Interp_forward, 'CaffeArgMax': CaffeArgMax_forward, 'HardSigmoid': HardSigmoid_forward, 'HardSwish': HardSwish_forward, 'Neg': Neg_forward, 'GRU': GRU_forward, 'PPQDeviceSwitch': PPQDeviceSwitch_forward, 'Identity': Identity_forward, 'OneHot': Onehot_forward, 'Reciprocal': Reciprocal_forward, 'LSTM': LSTM_forward, 'Sum': Sum_forward, 'Elu': Elu_forward, 'Erf': Erf_forward}


OPERATION_FORWARD_TABLE = {platform: DEFAULT_BACKEND_TABLE.copy() for platform in TargetPlatform}


EXPORT_OVERLAPPED_CONFIG = False


class QuantizationProperty(Enum):
    """QuantizationProperty is a core abstraction for PPQ quantization
    calculation. QuantizationProperty and QuantizationPolicy together build a
    bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 8 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Linear quantization, follow formula: quant(x) = clip(round(x / scale))

        FLOATING: Low precision float quantization, FP8, BF16, FP16.

        SYMMETRICAL: Symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Power-of-2 quantization, scale must be pow(2, k) in this mode.

        DYNAMIC: Dynamic Activation Quantization, scale is computed on the fly.

    ATTENTION: Not all combinations of all 8 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    PER_TENSOR = 1
    PER_CHANNEL = 2
    LINEAR = 4
    FLOATING = 8
    SYMMETRICAL = 16
    ASYMMETRICAL = 32
    POWER_OF_2 = 64
    DYNAMIC = 128

    def __or__(self, other: int) ->int:
        return self.value + other

    def __ror__(self, other: int) ->int:
        return self.value + other

    def __and__(self, other: int) ->int:
        return self.value & other

    def __rand__(self, other: int) ->int:
        return self.value & other

    def __radd__(self, other: int) ->int:
        return self.value + other

    def __add__(self, other: int) ->int:
        return self.value + other

    def __sub__(self, other: int) ->int:
        return self - (self.value & other)

    def __rsub__(self, other: int) ->int:
        return other - (self.value & other)


class QuantizationPolicy:
    """QuantizationPolicy is a core abstraction for PPQ quantization
    calculation. QuantizationProperty and QuantizationPolicy together build a
    bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 8 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Linear quantization, follow formula: quant(x) = clip(round(x / scale))

        EXPONENTIAL: Exponential quantization, not yet used.

        SYMMETRICAL: Symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Power-of-2 quantization, scale must be pow(2, k) in this mode.

        DYNAMIC: Dynamic Activation Quantization, scale is computed on the fly.

    ATTENTION: Not all combinations of all 8 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """

    def __init__(self, policy: int) ->None:
        if not QuantizationPolicy.__check_valid(policy):
            raise ValueError('invalid quantization pattern, valid partterns are listed in ppq.core.OperationQuantizationPolicy.__check_valid')
        self._policy = policy

    def has_property(self, property: QuantizationProperty) ->bool:
        return self._policy & property.value != 0

    def __eq__(self, o: object) ->bool:
        if not isinstance(o, QuantizationPolicy):
            raise TypeError('Can only compare QuantizationPolicy object with another QuantizationPolicy object.')
        return self._policy == o._policy

    @classmethod
    def __check_valid(cls, policy):
        return policy in {QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2}

    def to_dict(self) ->dict:
        """return a dictionary to describe this policy.

        nothing funny.
        """
        return {property.name: self.has_property(property) for property in QuantizationProperty}


class QuantizationStates(Enum):
    """QuantizationStates is a core data structure for PPQ quantization.
    QuantizationStates tells whether a quantization configuration is activated.

    ATTENTION: Changes of QuantizationState will greatly affect execution result.

    For a TensorQuantizationConfig instance, there are 9 available quantization states now.
    Only when state is ACTIVATED or NEGATIVE, corresponding tensor will be quantized during the execution.

    Here we give a brief description of each quantization state:

        INITIAL: given when TensorQuantizationConfig is created, is an initial state of all quantization configuration.

        PASSIVE_INIT: for particular parameter like bias of GEMM(Convolution) and padding value of Pad. Usually it
        does not have an independent quantization scale and offset, while gets quantized with other tensor's configuration.
            For GEMM and Convolution, there bias will be quantized with input scale * weight scale.
            For padding value and clip value, it shares the same scale with its input.
        Those parameters will have a PASSIVE_INIT state when created.

        OVERLAPPED: state OVERLAPPED means there is someone else takes control of current tensor,
        and overlapped tensor quantization configuration will be ignored by optimization algorithms and executor.

        Graph fusion always generate overlapped quantization, for a typical conv - relu fusion,
        the output quantization of convolution will be overlapped by the output tensor of relu.
        State OVERLAPPED cares only about quantization behaviour that cross layers.

        ACTIVATE: means corresponding tensor is ready to be quantized with its configuration.

        PASSIVE: means corresponding tensor is ready to be quantized with its configuration.
            (however its configuration is not stand alone, its scale and offset depends on someone else.)

        BAKED: means corresponding tensor has been pre-quantized, its value can directly
            go forward without quantization.
    """
    INITIAL = 1
    ACTIVATED = 4
    BAKED = 2
    OVERLAPPED = 3
    PASSIVE_INIT = 6
    PASSIVE = 5
    PASSIVE_BAKED = 7
    FP32 = 8
    SOI = -1
    DEQUANTIZED = -2
    DEACTIVED = -3

    @classmethod
    def is_activated(cls, state) ->bool:
        return state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}

    @classmethod
    def can_export(cls, state) ->bool:
        return state not in {QuantizationStates.INITIAL, QuantizationStates.PASSIVE_INIT, QuantizationStates.DEQUANTIZED, QuantizationStates.DEACTIVED}


class QuantizationVisibility(Enum):
    FORCE_EXPORT = 1
    EXPORT_WHEN_ACTIVE = 2
    INTERNAL = 3


class RoundingPolicy(Enum):
    """RoundingPolicy is a core setting for PPQ quantization calculation. It
    defines rounding behaviour inside quantization calculation.

    Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

    PPQ Supports 7 different rounding policies now.
    Take a look at https://en.wikipedia.org/wiki/Rounding

    ATTENTION: RoundingPolicy greatly affects PPQ executor behaviour in some cases,
        to get a correct result from PPQ executor,
        make sure your RoundingPolicy is the same as your hardware.
    """
    ROUND_HALF_EVEN = 0
    ROUND_HALF_UP = 1
    ROUND_HALF_DOWN = 2
    ROUND_HALF_TOWARDS_ZERO = 3
    ROUND_HALF_FAR_FORM_ZERO = 4
    ROUND_TO_NEAR_INT = 5
    ROUND_UP = 6


class TensorQuantizationConfig(Serializable):
    """
    ## TensorQuantizationConfig(Tensor 量化控制结构体)

    PPQ 使用量化控制结构体描述量化行为，该结构体被定义在 ppq.core.quant 中。
    截止 PPQ 0.6.6 版本，该结构体由 15 项不同的属性组成。本文将向你介绍这一核心数据结构体的设计构想。

    ### QuantizationPolicy 量化策略
    
    在 TensorQuantizationConfig 当中，首当其冲地内容是 TQC.policy，这是一个 QuantizationPolicy 对象。
    policy 属性用于描述量化的规则，一个完整的量化策略是由多个量化属性(QuantizationProperty)组合完成的；
    在 PPQ 中目前我们支持 8 种不同的量化属性，你可以使用以下属性来组合形成自定义的量化规则:
        - PER_TENSOR: 以 Tensor 为单位完成量化，每个 Tensor 使用一个 scale 和 offset 信息。
        - PER_CHANNEL: 以 Channel 为单位完成量化，每个 Channel 使用一个 scale 和 offset 信息。
        - LINEAR: 线性量化，通常的 INT8, INT16 皆属于线性量化，在线性量化的表示中不存在指数位。
        - FLOATING: 浮点量化，包括 FP8 E4M3, FP8 E5M2, FP16, BF16 等格式，在浮点量化中数据由底数和指数两部分组成。
        - SYMMETRICAL: 对称量化，在量化计算中不启用 offset。
        - ASYMMETRICAL: 非对称量化，在量化计算中启用 offset 完成量化偏移。
        - POWER_OF_2: 限制 scale 取值必须为 2 的整数次幂，这一量化行为多见于端侧以及浮点量化。
        - DYNAMIC: 启用动态量化策略，对于每一批次的数据，scale 与 offset 都将被动态地计算更新。

    下图解释了浮点量化与线性量化的区别：

    ![image](https://user-images.githubusercontent.com/43309460/199235366-1e83ed97-0731-4e1d-abeb-b7121e3d2a94.png)

    ### 线性量化与相关属性

    线性量化允许与下列属性进行组合：

        QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
        QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
        QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
        QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
        QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
        QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
        QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
        QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,

    在线性量化中，量化函数的计算方法为：

    - Unscaled FP32 = (FP32 / scale) - offset
    - INT8 = Clip(Round(Unscale FP32), quant_min, quant_max)
    - Dequantized FP32 = (INT8 + offset) * scale

    其中 Round 函数行为由 TQC.rounding(RoundingPolicy) 属性确定，PPQ 支持 7 种不同的取整策略，其中 ROUND_HALF_EVEN 是最常见的取整策略，
    关于取整策略的详细讨论可以参考 https://en.wikipedia.org/wiki/Rounding

    quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于线性量化而言他们是整数，通常为[-128, 127]

    ### 浮点量化与相关属性

    浮点量化允许与下列属性进行组合：

        QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
        QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,

    在浮点量化中，量化函数的计算方法为：

    - Unscaled FP32 = (FP32 / scale)
    - FP8 = Convert(Unscale FP32, quant_min, quant_max)
    - Dequantized FP32 = FP8 * scale

    其中 Convert 函数行为复杂，其转换过程分为三种不同的情况：
    - 当 Unscaled FP32 大于 quant_max，或者小于 quant_min，则直接进行截断
    - 当 Unscaled FP32 幅值大于 FP8 能够表达的最小值，此时需要移去多余的底数位，并对底数进行四舍五入
    - 当 Unscaled FP32 数据小于规范化 FP8 能够表达的最小值，此时浮点下溢出，此时我们计算 FP8 = Round(Unscaled FP32 / FP8_min) * FP8_min

    其中 FP8_min 是非规格化 FP8 能够表达的最小值。

    quant_min, quant_max 分别由 TQC.quant_min, TQC.quant_max 属性确定，对于浮点量化而言他们是浮点数，通常为[-448.0, +448.0]，
    对于 FLOATING 量化，我们引入一个新的属性 TQC.exponent_bits(int)。使用这个属性来指定总位宽中有多少数位用于表示指数(相应地，底数位为总位宽-指数位-1)。

    关于浮点量化的具体细节可以参考 [本文](https://zhuanlan.zhihu.com/p/574825662)

    ### 其他属性
        - TQC.num_of_bits(int)：量化位宽，对于 INT8, FP8 量化，量化位宽为 8。对于 INT16, FP16 量化，量化位宽为16。
        - TQC.state(QuantizationStates): 量化状态，在 PPQ 中目前有共计 8 种不同的量化状态，该属性极大地丰富了 PPQ 量化信息的语义，
                                         使得我们能够更加灵活地控制量化行为。该属性可以被用于切换 量化 / 非量化 状态；执行量化联合定点；执行参数烘焙。
        - TQC.channel_axis(int): 量化轴，对于 PER_CHANNEL 量化，使用这个属性来指定沿着那一维度展开量化
        - TQC.observer_algorithm(str): observer 算法，其中 observer 是用于确定 scale 和 offset 的对象，使用这个属性指明要使用何种类型的 observer 确定 scale 和 offset
        - TQC.dominator(TensorQuantizationConfig): 一个指向父量化信息的指针。在 PPQ 中 TQC 与 TQC 之间并不是独立的，他们之间可以存在父子关系。所有子量化信息与父量化信息共享 scale 和 offset
        - TQC.visiblity(QuantizationVisibility): 导出可见性，使用这个属性来告知 ppq 的导出器是否需要导出当前的 TQC。

    ### 量化控制结构体的初始化

    TensorQuantizationConfig 是 PPQ 中的核心数据结构，它总是由 Quantizer 对象完成创建的：

        # 下面这段代码为一个指定的算子创建了相应的 Tensor Quantization Config
        quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # 取得 TRT_FP8 所对应的量化器
        quantizer.quantize_operation(op_name = op.name, platform = dispatching[op.name])

    在 PPQ 当中，Quantizer 的职责即是为算子初始化他们的量化控制结构体。不同的量化器将按照不同的规则创建控制结构体，
    如 TRT_FP8 所对应的量化器 只会为了 Conv, Gemm 算子创建量化信息，要求他们的输入按照对称-浮点-Per Channel的方式完成量化。
    而 DSP_INT8 所对应的量化器为几乎所有算子创建量化信息，要求他们按照非对称-线性-Per Tensor的方式完成量化。

    ### 量化控制结构体的校准

    绝大部分的 TensorQuantizationConfig 在完成初始化之后都无法使用-他们的 scale 与 offset 均为空值，
    且 Quantizer 在初始化他们时会将其状态(TQC.state)置为 INITIAL，处于这个状态的量化信息在计算过程中不会被启用。

    我们必须送入一定量数据，进行必要 Calibration 操作后才能为网络中的量化信息确定合理的 scale 与 offset 值，这一过程是由种类繁多的 Observer 完成的：

        # PPQ 目前支持 7 种不同的 Observer
        OBSERVER_TABLE = {
            'minmax': TorchMinMaxObserver,
            'kl': TorchHistObserver,
            'percentile': TorchPercentileObserver,
            'mse': TorchMSEObserver,
            'isotone': TorchIsotoneObserver,
            'constant': ConstantObserver,
            'floating': DirectMSEObserver
        }

    这些 Observer 会负责在网络计算过程中收集必要的统计信息，并为 TQC 的 scale 与 offset 赋予有效的值。在完成一切之后，
    Observer 还会负责将 TQC 的状态(TQC.state)修改为 ACTIVED。此时量化信息将被正式启用，从而在网络前向传播模拟量化计算。

    关于 Observer 的讨论，可以参考 [本视频](https://www.bilibili.com/video/BV1QF41157aM)

    ### 量化控制结构体的父子链接

    在我们讨论量化时，对于那些存在着多个输入的算子，例如 add, concat，它们的所有输入总是被要求有着相同的 scale。为了表述这种语义，
    我们为 TQC 添加了 TQC.dominator 属性，这一属性可以指向另一个量化控制结构体。

    假设我们存在两个不同的量化控制结构体 A, B：

    - 语句 A.dominator = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。于此同时 A.state 将被修改为 OVERLAPPED(A 将不再启用)
    - 语句 A.master = B 表示 A 将共享 B 的 scale 与 offset(A.policy, A.num_of_bits等属性仍将使用自己的)。于此同时 A.state 将被修改为 PASSIVE(A 将仍然启用，但不具有独立的量化参数)

    如果 A 已经是其他量化结构体 C 的父节点，则上述过程将级联地使得 B 成为 A, C 共同的父节点，A, C 都将共享 B 的 scale 与 offset。

    下图展示了在量化控制结构体的生命周期中，量化状态是如何变迁的：

    ![Quantization State](https://user-images.githubusercontent.com/43309460/199236632-ec69ca29-9900-4875-8299-a196546d0dde.png)
    """

    def __init__(self, policy: QuantizationPolicy, rounding: RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN, num_of_bits: int=8, quant_min: int=-127, quant_max: int=128, exponent_bits: int=0, scale: Any=None, offset: Any=None, observer_algorithm: str=None, detail: Any=None, channel_axis: int=None, visibility: QuantizationVisibility=QuantizationVisibility.EXPORT_WHEN_ACTIVE, state: QuantizationStates=QuantizationStates.INITIAL):
        """Create a PPQ Tensor Quantization Configuration Instance.

        Args:
            policy (QuantizationPolicy):
                Quantization policy instance which defines the quantization behaviour from marco view.

            rounding (RoundingPolicy): Rounding policy used in quantization.

            num_of_bits (int): Quantization fraction bits. (2 < num_of_bits < 32)
            
            exponent_bits (int): Quantization exponent bits. (0 < num_of_bits < 8)
                For Int8 Quantization, num_of_bits = 8 and exponent_bits = 0
                For FP8 Quantization, num_of_bits = 4 and exponent_bits = 4

            quant_min (int): An integer value represents the upper bound(inclusive) of quantized value.

            quant_max (int): An integer value represents the lower bound(inclusive) of quantized value.

            scale (Any):
                Scale of quantized value, for per-tensor quantization policy, we use a single float as its scale,
                while for per-channel quantization policy, it will be an array that contains scales for each channel.

            offset (Any): Quantization offset for ASYMMETRICAL quantization policy,
                it will be set as 0 in SYMMETRICAL quantization schema.

            observer_algorithm (str): A string represents an observing algorithm for this tensor.
                PPQ support 'kl', 'minmax' observer now.

            detail (Any, optional): Only used by PPQ internal logic, detail is used to store some internal data,
                you are not supposed to use it.

            channel_axis (int, optional): Only used in PER_CHANNEL quantization, channel index.
        
            visiblity (Visiblity): visiblity is the attribute that controls export logic.

            Currently, there are 3 Visiblity level in PPQ:
            if Visiblity == FORCE_EXPORT, ppq exporter will export this TQC 
                ignoring state check(even if current TQC has been overrlapped).
            if Visiblity == EXPORT_WHEN_ACTIVD, ppq exporter will export this TQC only when it has been actived.
            if Visiblity == INTERNAL, This TQC will not be exported.

            state (QuantizationStates, optional):
                Defaults to QuantizationStates.INITIAL, see QuantizationStates for more detail.
        """
        assert num_of_bits <= 32, 'Cannot quantize a tensor with more than 32 bits.'
        assert num_of_bits >= 2, 'Cannot quantize a tensor with less than 2 bits.'
        assert exponent_bits <= 8, 'Cannot quantize a tensor with more than 8 bits exponent(fp32 overflow).'
        assert exponent_bits >= 0, 'Cannot quantize a tensor with less than 0 bits exponent.'
        self._policy = policy
        self._exponent_bits = exponent_bits
        self._num_of_bits = num_of_bits
        self._scale = scale
        self._offset = offset
        self.state = state
        self._rounding = rounding
        self._quant_min = quant_min
        self._quant_max = quant_max
        self._channel_axis = channel_axis
        self.observer_algorithm = observer_algorithm
        self.detail = {} if detail is None else detail
        self._dominator = self
        self._hash = self.__create_hash()
        self._visibility = visibility
        super().__init__()

    def can_export(self, export_overlapped: bool=EXPORT_OVERLAPPED_CONFIG) ->bool:
        if self.visibility == QuantizationVisibility.INTERNAL:
            return False
        type_check = isinstance(self.scale, torch.Tensor) and isinstance(self.offset, torch.Tensor)
        valid_states = {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}
        if export_overlapped:
            valid_states.add(QuantizationStates.OVERLAPPED)
        state_check = QuantizationStates.is_activated(self.state) or self.state in valid_states
        if state_check or self.visibility == QuantizationVisibility.FORCE_EXPORT:
            if type_check:
                return True
        return False

    def __eq__(self, o: object) ->bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError('Can only compare TensorQuantizationConfig object with another TensorQuantizationConfig object.')
        return self._hash == o._hash

    def __str__(self) ->str:
        return f'PPQ TensorQuantizationConfig({self.__hash__()})'
    _hash_seed = int(time.time())

    @staticmethod
    def __create_hash():
        TensorQuantizationConfig._hash_seed = (214013 * TensorQuantizationConfig._hash_seed + 2531011) % (2 << 31)
        return TensorQuantizationConfig._hash_seed

    def __hash__(self) ->int:
        return self._hash

    def is_same_scheme(self, o: object) ->bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError('Can only compare TensorQuantizationConfig object with another TensorQuantizationConfig object.')
        return self.quant_max == o.quant_max and self.quant_min == o.quant_min and self.policy == o.policy and self.num_of_bits == o.num_of_bits and self.exponent_bits == o.exponent_bits and self.channel_axis == o.channel_axis and self.rounding == o.rounding

    @property
    def dominated_by(self):
        """dominated_by is a crucial feature for tensor quantization
        configuration in PPQ. This property is actually maintained by union-
        find set data structure.

        Every tensor quantization configuration(A) is created with dominated_by = self, and only when
            it is overlapped by other configuration(B), it shall set A.dominated_by = B.
            Setting A.dominated_by = B also makes A, B as a quantization group.
            (quantization state of A is always set as OVERLAPPED here)

        So to say every tensor quantization configuration with dominated_by != self is overrlaped by
            other quantization configuration. When a tensor quantization configuration is overlapped,
            it means this tensor is already been quantized with another quantization configuration,
            and there is no need to be quantized with this configuration anymore.

        PPQ use this property to find root configuration for each configuration group,

        Returns:
            [TensorQuantizationConfig]: root configuration of this quantization group.

        ATTENTION: This configuration is invalid when self.dominated_by != self.
        """
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @dominated_by.setter
    def dominated_by(self, o):
        assert isinstance(o, TensorQuantizationConfig), 'Can only set this attribute with another tensor config.'
        if o._hash == self._hash:
            raise ValueError('Error with TQC.dominated_by = o: o must not equal to TQC its self.')
        root, dominator = self.dominated_by, o.dominated_by
        assert isinstance(root, TensorQuantizationConfig)
        if dominator != root:
            root._dominator = dominator
            self._dominator = dominator
            root.state = QuantizationStates.OVERLAPPED
            self.state = QuantizationStates.OVERLAPPED

    @property
    def master_by(self):
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @master_by.setter
    def master_by(self, master):
        if not isinstance(master, TensorQuantizationConfig):
            raise TypeError(f'Error with TQC.master_by(o): o must be another Tensor Quantization Config, however {type(master)} was given.')
        if master._hash == self._hash:
            raise ValueError('Error with TQC.dominated_by = o: o must not equal to TQC its self.')
        self._dominator = master
        if master.scale is not None and master.offset is not None:
            self.state = QuantizationStates.PASSIVE
        else:
            self.state = QuantizationStates.PASSIVE_INIT

    def is_revisable(self):
        return self.dominated_by == self and self.state in {QuantizationStates.ACTIVATED, QuantizationStates.FP32, QuantizationStates.FP32, QuantizationStates.INITIAL, QuantizationStates.FP32, QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_INIT}

    @property
    def visibility(self) ->QuantizationVisibility:
        return self._visibility

    @visibility.setter
    def visibility(self, visiblity: QuantizationVisibility):
        self._visibility = visiblity

    @property
    def scale(self) ->torch.Tensor:
        if self.dominated_by == self:
            return self._scale
        else:
            return self.dominated_by.scale

    @property
    def offset(self) ->torch.Tensor:
        if self.dominated_by == self:
            return self._offset
        else:
            return self.dominated_by.offset

    @property
    def policy(self) ->QuantizationPolicy:
        return self._policy

    @property
    def num_of_bits(self) ->int:
        return self._num_of_bits

    @property
    def rounding(self) ->RoundingPolicy:
        return self._rounding

    @property
    def quant_min(self) ->int:
        return self._quant_min

    @property
    def quant_max(self) ->int:
        return self._quant_max

    @property
    def exponent_bits(self) ->int:
        return self._exponent_bits

    @property
    def mantissa_bits(self) ->int:
        return self.num_of_bits - self._exponent_bits - 1

    @property
    def channel_axis(self) ->int:
        return self._channel_axis

    @scale.setter
    def scale(self, value: Any):
        if not self.is_revisable():
            raise PermissionError('Can not change scale of this tensor quantization configuration now. It has been overlapped or has an inactive state. Due to it is not a active config, any change of this configuration is not allowed.')
        else:
            self._scale = value

    @offset.setter
    def offset(self, value: Any):
        if not self.is_revisable():
            raise PermissionError('Can not change offset of this tensor quantization configuration now. It has been overlapped or has an inactive state. Due to it is not a active config, any change of this configuration is not allowed.')
        else:
            self._offset = value

    @policy.setter
    def policy(self, policy: QuantizationPolicy):
        self._policy = policy

    @num_of_bits.setter
    def num_of_bits(self, bits: int):
        self._num_of_bits = bits

    @rounding.setter
    def rounding(self, policy: RoundingPolicy):
        self._rounding = policy

    @quant_min.setter
    def quant_min(self, min: int):
        self._quant_min = min

    @quant_max.setter
    def quant_max(self, max: int):
        self._quant_max = max

    @exponent_bits.setter
    def exponent_bits(self, bits: int):
        if not self.policy.has_property(QuantizationProperty.FLOATING):
            raise PermissionError('Can not change property: exponent bits for this TQC. self.policy.has_property(QuantizationProperty.FLOATING) == False.')
        self._exponent_bits = bits

    @channel_axis.setter
    def channel_axis(self, channel_axis: int):
        if not self.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError('Can not change property: quantization channel axis for this TQC. self.policy.has_property(QuantizationProperty.PER_CHANNEL) == False.')
        self._channel_axis = channel_axis

    def copy(self):
        """Create a tensor config from this one, keep policy and state
        unchanged.

        if there is an non-empty scale and offset, they will be cloned too.
        """
        scale, offset = None, None
        if self.scale is not None:
            if isinstance(self.scale, torch.Tensor):
                scale = self.scale.clone()
            else:
                scale = self.scale
        if self.offset is not None:
            if isinstance(self.offset, torch.Tensor):
                offset = self.offset.clone()
            else:
                offset = self.offset
        config = TensorQuantizationConfig(policy=self.policy, rounding=self.rounding, num_of_bits=self.num_of_bits, quant_min=self.quant_min, quant_max=self.quant_max, scale=scale, offset=offset, observer_algorithm=self.observer_algorithm, detail=self.detail.copy(), state=self.state, exponent_bits=self.exponent_bits, channel_axis=self.channel_axis, visibility=self.visibility)
        if self.state == QuantizationStates.OVERLAPPED:
            config._dominator = self._dominator
        return config


class PPQTensorRoundImpl(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, policy: RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) ->torch.Tensor:
        """
            reference: https://en.wikipedia.org/wiki/Rounding

        Args:
            value (torch.Tensor): [description]
            policy (RoundingPolicy, optional): [description]. Defaults to RoundingPolicy.ROUND_HALF_EVEN.

        Raises:
            ValueError: [description]

        Returns:
            torch.Tensor: [description]
        """
        assert isinstance(value, torch.Tensor), 'tensor round only takes effect on torch tensor.'
        if policy == RoundingPolicy.ROUND_HALF_EVEN:
            return value.round()
        elif policy == RoundingPolicy.ROUND_UP:
            return value.ceil()
        elif policy == RoundingPolicy.ROUND_HALF_TOWARDS_ZERO:
            return torch.sign(value) * torch.ceil(value.abs() - 0.5)
        elif policy == RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO:
            return torch.sign(value) * torch.floor(value.abs() + 0.5)
        elif policy == RoundingPolicy.ROUND_HALF_DOWN:
            return torch.ceil(value - 0.5)
        elif policy == RoundingPolicy.ROUND_HALF_UP:
            return torch.floor(value + 0.5)
        elif policy == RoundingPolicy.ROUND_TO_NEAR_INT:
            raise NotImplementedError(f'Torch Tensor can not use this rounding policy({policy}) try ROUND_HALF_EVEN instead.')
        else:
            raise ValueError('Unexpected rounding policy found.')

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None


def ppq_tensor_round(value: torch.Tensor, policy: RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) ->torch.Tensor:
    """
        reference: https://en.wikipedia.org/wiki/Rounding

    Args:
        value (torch.Tensor): [description]
        policy (RoundingPolicy, optional): [description]. Defaults to RoundingPolicy.ROUND_HALF_EVEN.

    Raises:
        ValueError: [description]

    Returns:
        torch.Tensor: [description]
    """
    return PPQTensorRoundImpl.apply(value, policy)


class ChannelwiseDynamicLinearQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
        channelwise_view = tensor.transpose(dim0=0, dim1=config.channel_axis).unsqueeze(-1)
        channelwise_view = torch.flatten(channelwise_view, start_dim=1)
        scales, offsets = [], []
        for _min, _max in zip(channelwise_view.min(dim=1)[0].tolist(), channelwise_view.max(dim=1)[0].tolist()):
            s, o = minmax_to_scale_offset(_min, _max, config)
            scales.append(s)
            offsets.append(o)
        scales = torch.tensor(scales, dtype=torch.float32, device=tensor.device)
        offsets = torch.tensor(offsets, dtype=torch.float32, device=tensor.device)
        shape = [(1 if axis != config.channel_axis else -1) for axis in range(tensor.ndim)]
        scales, offsets = scales.view(shape), offsets.view(shape)
        tensor = ppq_tensor_round(tensor / scales, config.rounding) + offsets
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        tensor = (tensor - offsets) * scales
        return tensor

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None


class TensorwiseDynamicLinearQuantImpl(Function):
    """Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
        scales, offsets = minmax_to_scale_offset(tensor.min().item(), tensor.max().item(), config=config)
        None
        tensor = ppq_tensor_round(tensor / scales, config.rounding) + offsets
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        tensor = (tensor - offsets) * scales
        return tensor

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None


def PPQDyamicLinearQuantFunction(tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
    """
    Dynamic Linear Quantization Function(PPQ 动态量化函数).
    
    When calling this method, we firstly solve a scale & offset setting by min-max observer.
    
    Then we applys ordinary Linear Quantization Function with solved setting.
    
    If there is a pre-defined scale & offset within given config, they will be dropped without warning.
    
    动态量化函数将在执行量化之前统计出 tensor 的 min - max, 而后计算出 scale & offset 并完成量化
    
    此时 TQC 中的 scale 与 offset 将被忽略
    """
    if not QuantizationStates.is_activated(config.state):
        return tensor
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if not config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Quantization Policy Do Not Have Dynamic Flag!')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseDynamicLinearQuantImpl.apply(tensor, config)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseDynamicLinearQuantImpl.apply(tensor, config)


class ChannelwiseFloatingQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, channel_axis: int, exponet_bits: int, mantissa_bits: int, quant_min: float, quant_max: float, rounding: RoundingPolicy) ->torch.Tensor:
        scales, offsets = scales, offsets
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
        else:
            quantized = CUDA.FloatingQuantize_C(tensor=tensor, scales=scales, offsets=offsets, channel_axis=channel_axis, exponent=exponet_bits, mantissa=mantissa_bits, minimum=quant_min, maximum=quant_max, rounding=rounding.value)
            return quantized

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None, None


class TensorwiseFloatingQuantImpl(Function):
    """Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, exponet_bits: int, mantissa_bits: int, quant_min: float, quant_max: float, rounding: RoundingPolicy) ->torch.Tensor:
        scales, offsets = scales, offsets
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
        else:
            quantized = CUDA.FloatingQuantize_T(tensor=tensor, scales=scales, offsets=offsets, exponent=exponet_bits, mantissa=mantissa_bits, minimum=quant_min, maximum=quant_max, rounding=rounding.value)
            return quantized

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None


def PPQFloatingQuantFunction(tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
    if not PPQ_CONFIG.USING_CUDA_KERNEL:
        raise PermissionError('PPQ Floating Quant Function require PPQ_CONFIG.USING_CUDA_KERNEL = True')
    if not tensor.is_cuda:
        raise PermissionError('PPQ Floating Quant Function requires tensor device to be cuda, CPU floating quantization is not implemented yet.')
    """PPQ 核心量化函数，没啥好说的了吧，这个玩意既做 quant 也做 dequant"""
    if not QuantizationStates.is_activated(config.state):
        return tensor
    if not config.policy.has_property(QuantizationProperty.FLOATING):
        raise ValueError('Critical Quantization Error! Unexpected policy detected. PPQFloatingQuantFunction except a Floating Quantization Config.')
    if config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Unexpected Dynamic Flag in Quantization Policy.')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseFloatingQuantImpl.apply(tensor, config.scale, config.offset, config.channel_axis, config.exponent_bits, config.mantissa_bits, config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseFloatingQuantImpl.apply(tensor, config.scale, config.offset, config.exponent_bits, config.mantissa_bits, config.quant_min, config.quant_max, config.rounding)


class ChannelwiseLinearQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, channel_axis: int, quant_min: int, quant_max: int, rounding: RoundingPolicy) ->torch.Tensor:
        scales, offsets = scales, offsets
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            shape = [(1 if axis != channel_axis else -1) for axis in range(tensor.ndim)]
            scale, offset = scales.view(shape), offsets.view(shape)
            tensor = ppq_tensor_round(tensor / scale, rounding) + offset
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offset) * scale
            return tensor
        else:
            quantized = CUDA.LinearQuantize_C(tensor=tensor, scales=scales, offsets=offsets, channel_axis=channel_axis, minimum=quant_min, maximum=quant_max, rounding=rounding.value)
            return quantized

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None, None


class TensorwiseLinearQuantImpl(Function):
    """Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, quant_min: int, quant_max: int, rounding: RoundingPolicy) ->torch.Tensor:
        scales, offsets = scales, offsets
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            tensor = ppq_tensor_round(tensor / scales, rounding) + offsets
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offsets) * scales
            return tensor
        else:
            quantized = CUDA.LinearQuantize_T(tensor=tensor, scales=scales, offsets=offsets, minimum=quant_min, maximum=quant_max, rounding=rounding.value)
            return quantized

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None


def PPQLinearQuantFunction(tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
    """PPQ 核心量化函数，没啥好说的了吧，这个玩意既做 quant 也做 dequant"""
    if not QuantizationStates.is_activated(config.state):
        return tensor
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Unexpected Dynamic Flag in Quantization Policy. Use PPQDyamicQuantFunction Instead.')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseLinearQuantImpl.apply(tensor, config.scale, config.offset, config.channel_axis, config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseLinearQuantImpl.apply(tensor, config.scale, config.offset, config.quant_min, config.quant_max, config.rounding)


def PPQuantFunction(tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
    """
    ## PPQ 核心量化函数

    根据 config 中描述的策略，量化给定的 tensor.
    
    请注意 config.state 必须处在激活状态，该函数起作用。如果 config.state 处于
        INITIAL, FP32, PASSIVE_INIT 等未激活状态，该函数不进行任何处理，直接返回 tensor.

    ### 线性量化(QuantizationProperty.LINEAR):

        INT8 = Clip(Round((FP32 / scale)))

    ### 浮点量化(QuantizationProperty.FLOATING):

        FP8 = Clip(FP32_TO_FP8((FP32 / scale)))

    ### 动态线性量化(QuantizationProperty.DYNMAIC)

        scale = max(FP32) / 255

        INT8  = Clip(Round((FP32 / scale)))

    """
    if tensor is None:
        raise ValueError('Tensor is empty.')
    if config.policy.has_property(QuantizationProperty.LINEAR):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return PPQLinearQuantFunction(tensor, config)
        else:
            return PPQDyamicLinearQuantFunction(tensor, config)
    if config.policy.has_property(QuantizationProperty.FLOATING):
        return PPQFloatingQuantFunction(tensor, config)
    raise ValueError('Unexpected Quantization Property Found in PPQuantFunction. Do not konw how to quantize your config yet.')


class BaseQuantFunction(Callable, metaclass=ABCMeta):

    def __init__(self) ->None:
        pass

    @abstractmethod
    def __call__(self, input_tensor: Any, quantization_config: TensorQuantizationConfig, **kwargs) ->Any:
        raise NotImplemented('Implement this first.')


class OperationQuantizationConfig(Iterable):
    """OperationQuantizationConfig serves as a collection of tensor
    quantization configuration.

    See TensorQuantizationConfig for more information.
    """

    def __init__(self, input_quantization_configs: List[TensorQuantizationConfig]=None, output_quantization_configs: List[TensorQuantizationConfig]=None, is_positive_quant_op: bool=True):
        """Create an operation quantization configuration.

        Args:
            input_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all input variables.

            output_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all output variables.

            ATTENTION: whether a variable is gonna to be quantized or not, it must have a quantization configuration.

            is_positive_quant_op (bool, optional): [description]. Defaults to True.
                some operations are passively quantized, such as Maxpooling, Padding.
                For those operations, set this property as False, PPQ will use this property to optimize your graph.
        """
        self.input_quantization_config = self.__check_famliy_config(input_quantization_configs)
        self.output_quantization_config = self.__check_famliy_config(output_quantization_configs)
        self.is_active_quant_op = is_positive_quant_op

    def export(self) ->str:
        raise Exception('Implement this first')

    def __check_famliy_config(self, famliy_configs):
        for famliy_config in famliy_configs:
            if not isinstance(famliy_config, TensorQuantizationConfig):
                raise TypeError(f'You are trying to set famliy quantization config of {str(self)}, However your input is invalid, except one TensorQuantizationConfig object, while a {type(famliy_config)} was given.')
        return famliy_configs

    def __str__(self) ->str:
        return f'Inputs config: {self.input_quantization_config}, Outputs config {self.output_quantization_config}'

    def __iter__(self) ->TensorQuantizationConfig:
        return (self.input_quantization_config + self.output_quantization_config).__iter__()

    def copy(self):
        """Create an operation config from this one, keep policy and state
        unchanged.

        if this one has an non-empty scale or offset, they will be cloned too.
        """
        return OperationQuantizationConfig(input_quantization_configs=[_.copy() for _ in self.input_quantization_config], output_quantization_configs=[_.copy() for _ in self.output_quantization_config], is_positive_quant_op=self.is_active_quant_op)


class QuantableVariable(Variable):

    def __init__(self, convert_from: Variable) ->None:
        super().__init__(name=convert_from.name, dest_ops=convert_from.dest_ops.copy(), source_op=convert_from.source_op, value=convert_from.value, is_parameter=convert_from.is_parameter, shape=convert_from.shape, dtype=convert_from.dtype)
        self._fp32_value = None
        if convert_from.value is not None:
            self._fp32_value = convert_any_to_torch_tensor(convert_from.value, device='cpu')

    @property
    def stored_value(self) ->Any:
        return self._fp32_value

    @stored_value.setter
    def stored_value(self, value: Any):
        self._fp32_value = value

    @property
    def dest_op_configs(self) ->List[TensorQuantizationConfig]:
        _dest_op_configs, _dest_idx = [], self.dest_idx
        for idx, op in enumerate(self.dest_ops):
            if isinstance(op, QuantableOperation):
                _dest_op_configs.append(op.config.input_quantization_config[_dest_idx[idx]])
            else:
                _dest_op_configs.append(None)
        return _dest_op_configs

    @property
    def dest_op_platforms(self) ->List[TargetPlatform]:
        _dest_op_platforms = []
        for op in self.dest_ops:
            if op is not None:
                _dest_op_platforms.append(op.platform)
            else:
                _dest_op_platforms.append(TargetPlatform.FP32)
        return _dest_op_platforms

    @property
    def source_op_config(self) ->TensorQuantizationConfig:
        if self.source_op is not None:
            if isinstance(self.source_op, QuantableOperation):
                return self.source_op.config.output_quantization_config[self.src_idx]
            else:
                return None
        return None

    @property
    def source_op_platform(self) ->TargetPlatform:
        if self.source_op is None:
            return TargetPlatform.FP32
        else:
            return self.source_op.platform

    def copy(self, copy_value: bool=False):
        clone = QuantableVariable(super().copy(copy_value))
        if copy_value:
            clone._fp32_value = self._fp32_value.clone()
        else:
            clone._fp32_value = self._fp32_value
        return clone


class QuantableOperation(Operation):
    """ Quantable Operation (量化算子) 用来表示一个已经被量化了的算子
    相比于普通算子，一个量化算子具有以下额外的功能
    
    1. 每一个量化算子都将具有一个 config(OperationQuantizationConfig) 属性
       PPQ 使用这个东西描述量化细节，在整个网络中，有且只有这一个量化表示
       executor, dispatcher, optimization pass, exporter都是围绕这一属性工作的
    
    2. 每一个量化算子都将有一个 dequantize 方法和 restore_quantize_state 方法，
       一旦一个量化算子被 dequantize() 方法解除量化，该算子的 OperationQuantizationConfig 将被修改状态
       从而使得该算子的输入输出量化被暂时停用
       被解除量化的算子可以随时通过 restore_quantize_state 方法恢复量化状态
       对一个算子多次重复执行 dequantize 是可以的
    
    3. 每一个量化算子都将有一个 baking parameter 方法
       当算子具有有效的量化参数时，baking_parameters() 方法将对该算子的参数执行静态量化
       一旦静态量化完成，算子参数将被量化后的值替换；同时 config 的状态将被设置为: baked
    
    4. 每一个量化算子都将有一个 store_parameter_value 方法
       该方法将算子目前的参数保存入缓存；PPQ 将在创建 QuantableOperation 时执行此函数
       从而保存算子的原始参数，以备后续取用。
       一个显而易见的例子是，一旦算子执行了 baking_parameters 方法，它的参数值将被修改，
       此时若要完全还原算子状态，需要从缓存中取出算子的原始参数，并替换当前的值
       当你调用 restore_quantize_state 时，该方法会从缓存中取回保存的参数值并执行替换
       你不应当手动调用该方法，该方法将影响到 PPQ 的核心逻辑正确性

    5. 一个量化算子是可拷贝的，该拷贝只会拷贝算子的基本信息以及绑定的 OperationQuantizationConfig

    Args:
        Operation (_type_): _description_
    """

    def __init__(self, convert_from: Operation, quantize_config: OperationQuantizationConfig, platform: TargetPlatform):
        super().__init__(op_type=convert_from.type, inputs=convert_from.inputs.copy(), outputs=convert_from.outputs.copy(), attributes=convert_from.attributes, name=convert_from.name, platform=platform, opset=convert_from.opset)
        self._config = quantize_config
        self._meta = convert_from.meta_data
        self._dequantized = False

    @property
    def config(self) ->OperationQuantizationConfig:
        return self._config

    @config.setter
    def config(self, config):
        """we will update variable's during this function.

        Args:
            config ([type]): [description]

        Raises:
            TypeError: [description]
            ExecError: [description]

        Returns:
            [type]: [description]
        """
        if not isinstance(config, OperationQuantizationConfig):
            raise TypeError(f'object {str(config)}({type(config)}) is not a acceptable config for operation {self.name}')
        self._config = config

    @property
    def input_quant_config(self) ->List[TensorQuantizationConfig]:
        return self.config.input_quantization_config

    @property
    def output_quant_config(self) ->List[TensorQuantizationConfig]:
        return self.config.output_quantization_config

    def baking_parameters(self, quant_func: BaseQuantFunction):
        for config, var in self.config_with_variable:
            if var.is_parameter and config.state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}:
                assert isinstance(var, QuantableVariable)
                assert len(var.dest_ops) == 1, f', Parameter {var.name} has {len(var.dest_ops)} destinations, Baking parameter that has more than 1 destinations will incur unexpected problems, PPQ does not support parameters with more than 1 related operation, reform your graph first.'
                var.value = quant_func(var.value, config)
                if config.state == QuantizationStates.ACTIVATED:
                    config.state = QuantizationStates.BAKED
                if config.state == QuantizationStates.PASSIVE:
                    config.state = QuantizationStates.PASSIVE_BAKED
        return self

    def store_parameter_value(self):
        for var, _ in zip(self.inputs + self.outputs, self.config.input_quantization_config + self.config.output_quantization_config):
            if var.is_parameter:
                assert isinstance(var, QuantableVariable), 'Unexpected error.'
                var.stored_value = convert_any_to_torch_tensor(var.value, device='cpu').clone()
        return self

    def dequantize(self, parameter_only: bool=False, expire_device: str='cpu'):
        if self._dequantized:
            return self
        for var, quant_config in zip(self.inputs + self.outputs, self.config.input_quantization_config + self.config.output_quantization_config):
            if parameter_only and not var.is_parameter:
                continue
            quant_config.detail['Stored State'] = quant_config.state
            assert isinstance(var, QuantableVariable), f'Unexpected error with variable {var.name}.'
            if var.is_parameter:
                stored_value = convert_any_to_torch_tensor(var.value, device=expire_device)
                var.value = convert_any_to_torch_tensor(var.value, device=None)
                var.value = convert_any_to_torch_tensor(var.stored_value, device=var.value.device if var.value is not None else None)
                var.stored_value = stored_value
            quant_config.state = QuantizationStates.FP32
        self._dequantized = True
        return self

    def restore_quantize_state(self, expire_device: str='cpu'):
        if not self._dequantized:
            return self
        for var, quant_config in zip(self.inputs + self.outputs, self.config.input_quantization_config + self.config.output_quantization_config):
            if 'Stored State' in quant_config.detail:
                quant_config.state = quant_config.detail['Stored State']
                quant_config.detail.pop('Stored State')
                if var.is_parameter:
                    assert isinstance(var, QuantableVariable), 'Unexpected error.'
                    stored_value = convert_any_to_torch_tensor(var.value, device=expire_device)
                    var.value = convert_any_to_torch_tensor(var.value, device=None)
                    var.value = convert_any_to_torch_tensor(var.stored_value, device=var.value.device if var.value is not None else None)
                    var.stored_value = stored_value
        self._dequantized = False
        return self

    @property
    def config_with_variable(self) ->List[Tuple[TensorQuantizationConfig, Variable]]:
        """Just a helper function, This function will list all related config
        and variable with current operation.

        Returns:
            List[Tuple[TensorQuantizationConfig, Variable]]: [description]
        """
        ret = []
        for cfg, var in zip(self.config.input_quantization_config, self.inputs):
            ret.append((cfg, var))
        for cfg, var in zip(self.config.output_quantization_config, self.outputs):
            ret.append((cfg, var))
        return ret

    def copy(self):
        return QuantableOperation(convert_from=super().copy(), quantize_config=self.config.copy(), platform=self.platform)


class QuantOPRuntimeHook(RuntimeHook, metaclass=ABCMeta):
    """QuantOPRuntimeHook is an abstract class designed for executor
    customizing.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, operation: QuantableOperation, **kwargs) ->None:
        if not isinstance(operation, QuantableOperation):
            raise TypeError(f'You are trying to bind a QuantRuntimeHook to a non-quantized operation {operation}.')
        super().__init__(operation)

    def pre_forward_hook(self, inputs: list, quant_inputs: list, quant_configs: List[TensorQuantizationConfig]) ->list:
        return quant_inputs

    def post_forward_hook(self, outputs: list, quant_outputs: list, quant_configs: List[TensorQuantizationConfig]) ->list:
        return quant_outputs


class GraphCommandProcessor(Callable, metaclass=ABCMeta):

    def __init__(self, graph_or_processor: Union[BaseGraph, Callable]) ->None:
        """GraphCommandProcessor 是用于处理图上相关操作的抽象基类.

            我们使用指令-责任链模式处理 PPQ 计算图的相关操作，具体来说：

                所有图上相关操作都由一个 GraphCommand 对象进行封装，
                这一对象封装了操作的类型和必要参数

                同时我们设计了 GraphCommandProcessor 类用于接收并处理对应的 GraphCommand

                GraphCommandProcessor 被设计为责任链模式，当接收到无法处理的 GraphCommand 时
                将把无法识别 GraphCommand 传递给责任链上的下一任 GraphCommandProcessor
                直到 GraphCommand 被处理，或抛出无法处理的异常

                当你实现一个新的 GraphCommandProcessor 时，需要实现其中的方法 _acceptable_command_types，
                该方法返回了所有可以被识别的 GraphCommand 类型，同时在 _process 的逻辑中对 GraphCommand 的请求进行处理

                这两个方法被设计为私有的，这意味着你不能单独访问责任链中的独立 GraphCommandProcessor，
                只能够通过责任链的方式发起请求

                如果在责任链中有多个可以处理同一类请求的 GraphCommandProcessor，
                只有最先接触到 GraphCommand 的 GraphCommandProcessor将会被调用

                GraphCommandProcessor 将按照自定义逻辑解析 GraphCommand，
                在 BaseGraph 做出相应处理并给出结果，实现方法请参考 RunnableGraph

            GraphCommandProcessor is an abstract class for manipulating the graph

            We use Command-Responsibility Chain to process operations on the computational graph:

                all graph-related operations are encapsulated by GraphCommand objects, which contain
                the operation type and necessary parameters

                we design GraphCommandProcessor to process corresponding GraphCommand

                GraphCommandProcessor follows responsibility chain convention, a processor will pass
                the unrecogonized GraphCommand to the next processor in the chain, the GraphCommand will
                finally be executed when meeting the corresponding processor. An exception will be thrown
                if no processor in the current chain is able to deal with the given command

                When you implement a new GraphCommandProcessor, you should implement its inner method
                _acceptable_command_types, it returns all available GraphCommand types for execution
                and you should implement corresponding execution details in _process

                You are not supposed to visit single GraphCommandProcessor in the chain, and if there are
                multiple processors which could deal with same Command type in the chain, the first processor
                receiving the command will execute

                GraphCommandProcessor follows its predefined logic to parse GraphCommand, execute on the graph
                correspondingly and return the final result

        Args:
            graph_or_processor (BaseGraph, Callable): 被处理的图对象，可以是 BaseGraph 或者 GraphCommandProcessor
                如果是 GraphCommandProcessor 对象，则自动将 self 与 graph 链接成链
                the graph being executed or previous GraphCommandProcessor in the chain
        """
        if isinstance(graph_or_processor, GraphCommandProcessor):
            self._next_command_processor = graph_or_processor
            self._graph = graph_or_processor._graph
        else:
            self._next_command_processor = None
            self._graph = graph_or_processor

    @property
    @abstractproperty
    def _acceptable_command_types(self) ->List[GraphCommandType]:
        """Subclass of GraphCommandProcessor must give an implementation of
        this function.

            Return all acceptable GraphCommandTypes in a list as result.
            something like:
                return [
                    GraphCommandType.DEPLOY_TO_CPU,
                    GraphCommandType.DEPLOY_TO_CUDA,
                    GraphCommandType.DEPLOY_TO_NUMPY
                ]

        Returns:
            List[GraphCommandType]: all acceptable GraphCommandTypes
        """
        raise NotImplementedError('Oh, seems you forgot to implement GraphCommandProcessor._acceptable_command_types function')

    def __call__(self, command: GraphCommand) ->Any:
        """Invoking interface of GraphCommandProcessor responsibility chain.
        All processors within the chain shall be invoked by this function one
        be one, until there is a processor claim to accept input command
        object, the entire processing of responsibility chain ends then.

            invoke a GraphCommandProcessor chain like that:

                _ = GraphCommandProcessor(graph, next_command_processor=None)
                _ = GraphCommandProcessor(graph, next_command_processor=_)
                command_processor = GraphCommandProcessor(graph, next_command_processor=_)

                command = GraphCommand(GraphCommandType.DEPLOY_TO_CUDA)
                command_processor(command)

            All three GraphCommandProcessor will then be called one by one

            Never attempt to use function like _(command) in above case,
            all responsibility chains should only be called by its head.

        Args:
            command (GraphCommand): An acceptable GraphCommand object
            if an improper GraphCommand is given, it will incurs ValueError at end.

        Raises:
            ValueError: raise when there is no suitable processor for your command.

        Returns:
            Any: processor will decide what is it result.
        """
        if not isinstance(command, GraphCommand):
            raise ValueError(f'command should be an instance of GraphCommand, {type(command)} received yet.')
        if command.command_type in self._acceptable_command_types():
            return self.process(command)
        elif self._next_command_processor is not None:
            self._next_command_processor(command)
        else:
            raise ValueError(f'Command Type {command.command_type} is not acceptable in this graph, please make sure you have added proper command processor into processing chain.\nFor more information, you may refer to ppq.IR.graph.GraphCommandType')

    def acceptable_command_types(self) ->List[GraphCommandType]:
        """Return all acceptable command types of current chain. Notice there
        might be duplicated types.

        Returns:
            List[GraphCommandType]: all acceptable command types
        """
        my_types = self._acceptable_command_types()
        assert isinstance(my_types, list), 'GraphCommandProcessor.__acceptable_command_types must return a list of GraphCommandType'
        if self._next_command_processor is not None:
            other_types = self._next_command_processor.acceptable_command_types()
        return my_types.extend(other_types)

    @abstractmethod
    def process(self, command: GraphCommand) ->Any:
        """Subclass of GraphCommandProcessor must give an implementation of
        this function.

            Process received GraphCommand instance and give result(if there is any)

        Args:
            command (GraphCommand): input command object.

        Returns:
            Any: any result is fine.
        """
        raise NotImplementedError('Oh, seems you forgot to implement GraphCommandProcessor.process function(or may be forgot to return?)')

    @property
    def graph(self) ->BaseGraph:
        return self._graph

    def __str__(self) ->str:
        return f'GraphCommandProcessor {self.__hash__}'


class RunnableGraph(GraphCommandProcessor):

    def __init__(self, graph: BaseGraph, device: str=None):
        """RunnableGraph object aims at dealing things related with graph
        executing.

        Literally it helps you move values of your graph towards device and vice versa.
            And give an executable order of all operations in your graph which actual executor will follow.
        Args:
            graph (BaseGraph): BaseGraph instance.
            device (str, optional): This attribute is only used by with RunnableGraph(graph, device) syntactic.
            next_command_processor (Callable, optional): next processor in processing chain.
        """
        super().__init__(graph_or_processor=graph)
        self._device = device

    def process(self, command: GraphCommand) ->Any:
        if command.command_type == GraphCommandType.DEPLOY_TO_CPU:
            return self.deploy('cpu')
        elif command.command_type == GraphCommandType.DEPLOY_TO_CUDA:
            if isinstance(command, GraphDeployCommand):
                device = command._device
                return self.deploy(device)
            else:
                return self.deploy('cuda')
        elif command.command_type == GraphCommandType.DEPLOY_TO_NUMPY:
            return self.retrieve()

    def __enter__(self):
        self.deploy(self._device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.retrieve()

    def _acceptable_command_types(self) ->List[GraphCommandType]:
        return [GraphCommandType.DEPLOY_TO_CPU, GraphCommandType.DEPLOY_TO_CUDA, GraphCommandType.DEPLOY_TO_NUMPY]

    def retrieve(self):
        for _, operator in self._graph.operations.items():
            assert isinstance(operator, Operation), f'Failed to retrieve graph to numpy, incorrect operator {operator} found.'
            if operator.type == 'Constant':
                operator.attributes['value'] = convert_any_to_numpy(operator.attributes['value'])
        for _, variable in self._graph.variables.items():
            assert isinstance(variable, Variable), f'Failed to send graph to device, incorrect variable {variable} found.'
            variable.value = convert_any_to_numpy(variable.value, accept_none=True)
        return self

    def deploy(self, device: str):
        for _, operator in self._graph.operations.items():
            assert isinstance(operator, Operation), f'Failed to send graph to device, incorrect operator {operator} found.'
            if operator.type == 'Constant' and operator.platform != TargetPlatform.SOI:
                operator.attributes['value'] = convert_any_to_torch_tensor(operator.attributes['value'], accept_none=False)
            if operator.type == 'Constant' and operator.platform == TargetPlatform.SOI:
                value = operator.attributes['value']
                operator.attributes['value'] = convert_any_to_torch_tensor(value, accept_none=False, device='cpu')
            if isinstance(operator, QuantableOperation):
                for cfg, var in operator.config_with_variable:
                    if isinstance(cfg._scale, torch.Tensor):
                        cfg._scale = cfg._scale
                    if isinstance(cfg._offset, torch.Tensor):
                        cfg._offset = cfg._offset
        for _, variable in self._graph.variables.items():
            assert isinstance(variable, Variable), f'Failed to send graph to device, incorrect variable {variable} found.'
            if len(variable.dest_ops) == 0:
                continue
            if variable.value is None:
                continue
            platforms = [op.platform for op in variable.dest_ops]
            if all([(_ == platforms[0]) for _ in platforms]) and platforms[0] == TargetPlatform.SOI:
                platform = TargetPlatform.SOI
            else:
                platform = TargetPlatform.UNSPECIFIED
            if platform == TargetPlatform.SOI:
                variable.value = convert_any_to_torch_tensor(variable.value, accept_none=True)
            else:
                variable.value = convert_any_to_torch_tensor(variable.value, accept_none=True)
            if variable.is_parameter:
                if len(variable.dest_ops) > 1:
                    raise PermissionError(f'PPQ can not process parameter variable({variable.name}) with multiple destinations({[op.name for op in variable.dest_ops]}), split it first.')
                dest_op = variable.dest_ops[0]
                dest_idx = dest_op.inputs.index(variable)
                assert isinstance(dest_op, Operation)
                socket = dest_op.socket
                if socket.in_plat[dest_idx] == TargetPlatform.SOI:
                    variable.value = convert_any_to_torch_tensor(variable.value, accept_none=True)
        return self


class TorchMetaDataTracingHook(RuntimeHook):

    def __init__(self, operation: Operation) ->None:
        super().__init__(operation)

    def pre_forward_hook(self, inputs: List[torch.Tensor], **kwargs) ->list:
        for tensor, var in zip(inputs, self._hook_to.inputs):
            if tensor is None:
                ppq_warning(f'Unexpected input value of operation {self._hook_to.name}, recieving "None" at its input {self._hook_to.inputs.index(var)}')
            else:
                var.shape = tensor.shape
                var.dtype = tensor.dtype
        return inputs

    def post_forward_hook(self, outputs: List[torch.Tensor], **kwargs) ->list:
        for tensor, var in zip(outputs, self._hook_to.outputs):
            var.shape = tensor.shape
            var.dtype = tensor.dtype
        return outputs


class TorchQuantizeDelegator(Callable):
    """Since PPQ 0.6.2, Interface TorchQuantizeDelegate is introduced to
    customize quantization logic: To be specific, you are suppose to inherit
    this class, and define your own computation logic within function __call__.

        Pass your Delegate to TorchExecutor by TorchExecutor.register_quantize_delegate(c, d)
            Where c is the target quantization config, d is your delegator class.
            Once you invoke this function, PPQ execution system will hand the quantization
            computation of config c over to your delegate. PPQ execution system will no
            longer quantize variable related with config c anymore.

    Notice that a delegate replaces quantization computation only, it still under the control of PPQ quantization
    System, so to say if your config has an invalid state like DEQUANTIZED, PPQ execution system will never been
    required to quantize related tensor and so your delegate class will take no effects on config c.

        Remove delegate function by TorchExecutor.remove_quantize_delegate(c)

    If you have some customized parameter of your delegator logic, set them as class attributes.
    Like: self.param1 = ..., self.param2 = ...

    Do not edit config structure directly.

    Args:
        Callable (_type_): _description_
    """

    def __init__(self) ->None:
        super().__init__()

    def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) ->torch.Tensor:
        raise NotImplementedError('Implement this function first.')


def empty_ppq_cache(func: Callable):
    """Using empty_ppq_cache decorator to clear ppq memory cache, both gpu
    memory and cpu memory will be clear via this function.

    Function which get decorated by this will clear all ppq system cache BEFORE its running.
    Args:
        func (Callable): decorated function
    """

    def _wrapper(*args, **kwargs):
        empty_cache()
        gc.collect()
        return func(*args, **kwargs)
    return _wrapper


class TorchExecutor(BaseGraphExecutor, torch.nn.Module):

    def __init__(self, graph: BaseGraph, fp16_mode: bool=True, device: str='cuda') ->None:
        """
        ## PPQ Graph Executor(PPQ 执行引擎)

        为了量化并优化神经网络模型，PPQ 实现了基于 Pytorch 的执行引擎，该执行引擎能够执行 Onnx 与 Caffe 的模型文件，目前支持 90 余种常见 Onnx 算子，涵盖 1d, 2d, 3d 视觉、语音、文本模型。
        PPQ 的执行引擎位于 ppq.executor 目录下，由两个主要部分组成： ppq.executor.torch.py 文件中包含了执行引擎自身； ppq.executor.op 文件夹中则包含了不同后端的算子库。

        在开始阅理解执行引擎之前，我们先介绍算子库的相关内容

        ### PPQ Backend Functions(PPQ 算子库)

        核心算子库位于 ppq.executor.op.torch.default 文件中，该文件中包含了所有算子默认的执行逻辑。

        我们知道，对于一个量化算子而言，由于硬件的不同其执行逻辑也可能发生变化。例如 LeakyRelu 算子的负数部分在 GPU 上会采用 x * alpha 的方式完成计算，
        而在 FPGA 则会采用 x = x >> 3 完成计算。正因为这种差异的存在， PPQ 允许相同的算子在不同平台(TargetPlatform)上拥有不同的执行逻辑。
        这也意味着针对每一个平台，我们都将实现一个平台独特的算子库文件，这些算子库都继承于 ppq.executor.op.torch.default。

            def Mul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
                ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
                values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
                multiplicand, multiplier = values
                return multiplicand * multiplier

        上文中的内容即 ppq.executor.op.torch.default 中 Mul 算子的执行逻辑，在 PPQ 中，所有算子在执行时都将接受一系列 torch.Tensor 作为输入，而后我们调用 pytorch 完成算子的计算逻辑。
        你可以打开 PPQ 的算子库文件查看其他算子的执行逻辑，并且 PPQ 也提供了 register_operation_handler 函数，借助该函数你可以注册自定义算子的执行逻辑；或是覆盖现有算子的执行逻辑。

            def register_operation_handler(handler: Callable, operation_type: str, platform: TargetPlatform):
                if platform not in GLOBAL_DISPATCHING_TABLE:
                    raise ValueError('Unknown Platform detected, Please check your platform setting.')
                GLOBAL_DISPATCHING_TABLE[platform][operation_type] = handler
                
        该函数位于 ppq.api, 你可以使用语句 from ppq.api import register_operation_handler 来引入它。
        
        ### PPQ Executor(PPQ 执行引擎)
        接下来我们向你介绍 PPQ 执行引擎 TorchExecutor，你可以使用语句 from ppq import TorchExecutor 导入执行引擎。初始化执行引擎则需要传入一个 PPQ 计算图实例对象，
        在这里我们假设已经获取到了一个量化后的计算图对象 ppq_quant_ir，并使用下面的语句初始化计算引擎

            
            executor = TorchExecutor(graph=ppq_quant_ir)
            executor.forward(inputs=..., output_names=..., hooks=...)

        我们使用 executor.forward 接口获取图的执行结果，它将可以传入三个参数:

        * inputs: inputs (Union[dict, list, torch.Tensor]): [input tensors or somewhat]
        * output_names (List[str], optional): output variable names. default is None.
        * hooks (Dict[str, RuntimeHook], optional): A hook table for customizing operation behaviour and collate data during executing.

        当执行引擎获取到推理请求时，它将按拓扑顺序依次执行图中的算子：下图展示了一个简单的示例

        ![图1. 执行图示例](https://user-images.githubusercontent.com/43309460/190056256-8b664993-3af4-4151-8451-f60d42f3145c.png)

        在这里，我们的图中包含三个算子 Conv, Relu, Softmax，他们将按照拓扑次序被依次执行。PPQ 的执行引擎会在执行完 Conv 算子后，将 Conv 算子的结果暂存于 Var 1 中，供 Relu 算子取用。
        而在执行完 Relu 算子后，PPQ 执行引擎则会及时地释放 Var 1 中暂存的数据，因为他们不会被其他算子取用，而且也不是网络的输出 Variable。在每一次推理过后，PPQ 还会清空网络中所有的暂存变量以释放显存。
        下面的代码段展示了一个非量化算子的执行逻辑：

            for operation in executing_order:
                outputs = operation_forward_func(operation, inputs, self._executing_context)
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                fp_outputs = outputs

                for output_idx, output_var in enumerate(operation.outputs):
                    output_var       = operation.outputs[output_idx]
                    output_var.value = outputs[output_idx]

            for var in self._graph.variables.values():
                if not var.is_parameter:
                    var.value = None

        PPQ 的执行引擎是专为量化计算图的执行而设计的————接下来让我们深入到量化算子的执行过程中去。
        对于一个量化算子而言，其每一个输入和输出变量都会有一个 Tensor Quantization Config (TQC) 控制结构体对量化过程进行描述。

        对于一个量化 Conv 算子而言，PPQ 将为它创建 2-3 个 Input TQC，以及一个 Output TQC。分别对其输入变量以及输出变量的量化行为进行描述。
        下面的代码展示了如何为量化算子创建特定的 TQC 描述量化逻辑。

            if operation.type == 'Conv':
                config = self.create_default_quant_config(
                    op                 = operation, 
                    num_of_bits        = 8,
                    quant_max          = 127, 
                    quant_min          = -128,
                    observer_algorithm = 'percentile', 
                    policy             = QuantizationPolicy(
                        QuantizationProperty.PER_TENSOR +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.SYMMETRICAL),
                    rounding           = RoundingPolicy.ROUND_HALF_EVEN)

                for tensor_quant_config in config.input_quantization_config:
                    tensor_quant_config.state = QuantizationStates.FP32

                operation.config = config

        在 图2 中，我们展示了 PPQ 执行引擎对于量化算子的执行逻辑：

        ![图2. 算子执行流程](https://user-images.githubusercontent.com/43309460/190065996-02bf7fb4-7421-4417-883d-77967dd1863a.png)

        在 PPQ 中，算子的执行被分为四个过程：
        * 首先 PPQ 将根据算子上的 TQC 信息量化算子的输入。量化过程并非是原地的，量化后的数据将会是一个新的 torch.Tensor。
        * 随后 PPQ 在算子库中寻找算子的执行逻辑，我们已经提到对于每一个平台，他们都可以拥有自己的一套独立的算子库。
        PPQ 将按照算子的平台找到特定的计算逻辑，并调用他们完成计算得到结果。
        * PPQ 将根据算子上的 TQC 信息量化算子的输出。同样地，输出的量化也不是原地的。
        * 最后我们将量化好的结果写入到计算图的 Variable 上，从而供后续的算子取用。

        对于一个非量化算子而言，上述步骤中的 1，3 是可以省略的。

        下 图3 展示了一个量化卷积算子 与 TQC 之间的关系：

        ![图3. 量化 Conv 的执行过程](https://user-images.githubusercontent.com/43309460/190067258-4a4daec7-f898-4734-85a3-19f449c6a963.png)

        ### Quantize Delegate (量化代理函数)

        PPQ 允许你为网络中特定的 TQC 注册量化代理函数。这样你就可以注册自定义的量化处理逻辑，而非使用 PPQLinearQuantFunction 完成量化。

            def register_quantize_delegate(
                self, config: TensorQuantizationConfig,
                delegator: TorchQuantizeDelegator):

        使用 executor.register_quantize_delegate(config, function) 完成函数注册，被注册的函数必须满足 TorchQuantizeDelegator 所定义的接口。
        下面我们给出一个简单的量化代理函数例子：
            
            class MyQuantDelegator(TorchQuantizeDelegator):
                def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
                    if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                        raise ValueError('Sorry, this delegator handles only Symmetrical Quantizations.')
                    print('You are invoking cusitmized quant function now.')
                    return torch.round(tensor / config.scale) * config.scale

        在执行器遇到 TQC 时，将会调用 executor.quantize_function 执行量化，其逻辑为：

            def quantize_function(self, tensor: torch.Tensor, config: TensorQuantizationConfig = None) -> torch.Tensor:
                if config is None or not QuantizationStates.is_activated(config.state): return tensor
                elif config in self._delegates: return self._delegates[config](tensor, config)
                else: 
                    if config.policy.has_property(QuantizationProperty.DYNAMIC):
                        return self._dynamic_quant_fn(tensor, config)
                    else:
                        return self._default_quant_fn(tensor, config)

        ### Usage (用法示例)

        PPQ 的执行器初始化需要一个计算图实例作为参数：

            executor = TorchExecutor(graph=ppq_quant_ir)
            executor.forward(inputs=..., output_names=..., hooks=...)

        这一计算图可以是量化过后的，也可以是没有量化的。但 PPQ 希望传入的计算图经过正确调度，传入没有调度的计算图将会触发警报：

            if not graph.extension_attrib.get(IS_DISPATCHED_GRAPH, False):
                ppq_warning('Can not create executor with your graph, graph is not correctly dispatched, '
                            'use dispatch_graph(graph=ir, platform=platfrom, setting=setting) first.')

        executor.forward 需要三个参数，下面举例对其进行说明：

            # 传入三个变量 a, b, c 作为输入
            executor.forward(inputs=[a, b, c], output_names=..., hooks=...)
            
            # 分别对图中 input, var 1 两个变量传入 a, b 作为输入
            executor.forward(inputs={'input': a, 'var 1': b}, output_names=..., hooks=...)
            
            # 传入一个完整的 tensor 作为输入
            executor.forward(inputs=torch.zeros(shape=[1,3,224,224]), output_names=..., hooks=...)
            
            # 要求网络输出 output, Var 1 的值
            executor.forward(inputs=..., output_names=['output 1', 'Var 1'], hooks=...)
            
        executor.forward 函数默认不需要梯度，如果希望执行带有梯度的网络，需要使用 executor.forward_with_gradient 函数。 forward 函数的返回值永远是一个 torch.Tensor 数组，其中元素的顺序由 output_names 参数决定。


        ### Hook (执行钩子函数)
        在调用 executor.forward 函数时可以传入 hooks 参数。钩子函数是注册在 op 上的，你可以传入一个字典用来说明需要调用的钩子函数：
        字典 {'Conv 1': myhook} 说明了希望在算子 Conv 1 的执行器件调用钩子函数 myhook。

        钩子函数必须继承于 RuntimeHook 类，必须实现成员函数 pre_forward_hook, post_forward_hook。在这两个函数中，你可以制定特定的逻辑修改算子输入输出的值。

            class RuntimeHook(metaclass=ABCMeta):
                def __init__(self, operation: Operation) -> None:
                    self._hook_to = operation

                def pre_forward_hook(self, inputs: list, **kwargs) -> list:
                    return inputs

                def post_forward_hook(self, outputs: list, **kwargs) -> list:
                    return outputs
        
            TorchExecutor - executor object which use torch as its backend.
                torch backend is used to graph simulating & training(QAT)

                all operation forward functions are written with pytorch,
                so that they will have gradient recorded by torch engine.

                which means you can directly access to tensor.grad after using output.backward()
        Args:
            graph (BaseGraph):
                executing graph object,
                TorchExecutor will automatically send all graph parameters towards executing device.

            fp16_mode (bool, optional): [whether the simulator is running in fp16 mode(unimplemented).]. Defaults to True.

            device (str, optional): [
                executing device, as same as torch.device,
                you can not select gpu to executing yet,
                graph will always be send to the very first visible cuda device.
            ]. Defaults to 'cuda'.
        """
        self._default_quant_fn = PPQuantFunction
        self._deployed = False
        self._device = device
        self._executing_context = TorchBackendContext(executing_device=self._device)
        super().__init__(graph)
        self._runnable_graph = RunnableGraph(self._graph)
        self._delegates = {}
        self.fp16_mode = fp16_mode
        self.deploy()

    def register_quantize_delegate(self, config: TensorQuantizationConfig, delegator: TorchQuantizeDelegator):
        """Since PPQ 0.6.2, Interface TorchQuantizeDelegate is introduced to
        customize quantization logic: To be specific, you are suppose to
        inherit this class, and define your own computation logic within
        function __call__.

            Pass your Delegate to TorchExecutor by TorchExecutor.register_quantize_delegate(c, d)
                Where c is the target quantization config, d is your delegator class.
                Once you invoke this function, PPQ execution system will hand the quantization
                computation of config c over to your delegate. PPQ execution system will no
                longer quantize variable related with config c anymore.

        Notice that a delegate replaces quantization computation only, it still under the control of PPQ quantization
        System, so to say if your config has an invalid state like DEQUANTIZED, PPQ execution system will never been
        required to quantize related tensor and so your delegate class will take no effects on config c.

        Remove delegate function by TorchExecutor.remove_quantize_delegate(c)
        """
        if not isinstance(delegator, TorchQuantizeDelegator):
            raise TypeError(f'You can only register a TorchQuantizeDelegate as quantization delegator function, however a/an {type(delegator)} was given')
        if not isinstance(config, TensorQuantizationConfig):
            raise TypeError(f'Except a TensorQuantizationConfig instance, however {type(config)} was passed.')
        self._delegates[config] = delegator

    def remove_quantize_delegate(self, config: TensorQuantizationConfig):
        """Since PPQ 0.6.2, Interface TorchQuantizeDelegate is introduced to
        customize quantization logic: To be specific, you are suppose to
        inherit this class, and define your own computation logic within
        function __call__.

            Pass your Delegate to TorchExecutor by TorchExecutor.register_quantize_delegate(c, d)
                Where c is the target quantization config, d is your delegator class.
                Once you invoke this function, PPQ execution system will hand the quantization
                computation of config c over to your delegate. PPQ execution system will no
                longer quantize variable related with config c anymore.

        Notice that a delegate replaces quantization computation only, it still under the control of PPQ quantization
        System, so to say if your config has an invalid state like DEQUANTIZED, PPQ execution system will never been
        required to quantize related tensor and so your delegate class will take no effects on config c.

        Remove delegate function by TorchExecutor.remove_quantize_delegate(c)
        """
        if not isinstance(config, TensorQuantizationConfig):
            raise TypeError(f'Except a TensorQuantizationConfig instance, however {type(config)} was passed.')
        if config in self._delegates:
            self._delegates.pop(config)

    def deploy(self):
        """Deploy graph parameters towards target device.

        Raises:
            ValueError: [when target device is unacceptable]
        """
        self._deployed = True
        self._runnable_graph(GraphDeployCommand(device=self._device))

    def to(self, device: str):
        self._device = torch.device(device)
        self.deploy()
        return self

    @torch.no_grad()
    def forward(self, inputs: Union[dict, list, torch.Tensor], output_names: List[str]=None, hooks: Dict[str, RuntimeHook]=None) ->List[torch.Tensor]:
        """Forward function of this executor.

        Notice this forward function will never store and compute gradients.

        Args:
            inputs (Union[dict, list, torch.Tensor]): [input tensor or somewhat]

            output_names (List[str], optional):
                onnx output node names, which used to confirm a output order.

                Defaults to None.

            hooks (Dict[str, RuntimeHook], optional):
                A hook table for customizing operation behaviour and collate data during executing.
                All hooks should inherit from class RuntimeHook, with all necessary methods implemented.
                    See also: ppq.executor.base.RuntimeHook

                Executor calls hook.pre_forward_hook(operation, input_data) before dispatching operation,
                by using this feature, you can dynamically dispatch operation during executing,
                or processing input data as you want.(remember to return processed input data)

                Executor calls hook.post_forward_hook(operation, output_data) after the execution,
                you are supposed to  gather all necessary data from execution via this feature.

                For Quantable Operation, a much more powerful class:
                    ppq.executor.base.QuantOpRuntimeHook is provided.
                see also: ppq.executor.base.QuantOpRuntimeHook

                Defaults to None.

        Returns:
            List[torch.Tensor]: [executing result, list of tensor objects.]
        """
        return self.__forward(inputs=inputs, output_names=output_names, executing_order=self._executing_order, hooks=hooks)

    def forward_with_gradient(self, inputs: Union[dict, list, torch.Tensor], output_names: List[str]=None, hooks: Dict[str, RuntimeHook]=None) ->List[torch.Tensor]:
        """forward function of this executor.

            Notice this one will store and compute gradient.

        Args:
            inputs (Union[dict, list, torch.Tensor]): [input tensor or somewhat]
            output_names (List[str], optional):
                onnx output node names, which used to confirm a output order.

                Defaults to None.

            hooks (Dict[str, RuntimeHook], optional):
                A hook table for customizing operation behaviour and collate data during executing.
                All hooks should inherit from class RuntimeHook, with all necessary methods implemented.
                    See also: ppq.executor.base.RuntimeHook

                Executor calls hook.pre_forward_hook(operation, input_data) before dispatching operation,
                by using this feature, you can dynamically dispatch operation during executing,
                or processing input data as you want.(remember to return processed input data)

                Executor calls hook.post_forward_hook(operation, output_data) after the execution,
                you are supposed to  gather all necessary data from execution via this feature.

                For Quantable Operation, a much more powerful class:
                    ppq.executor.base.QuantOpRuntimeHook is provided.
                see also: ppq.executor.base.QuantOpRuntimeHook

                Defaults to None.

        Returns:
            List[torch.Tensor]: [executing result, list of tensor objects.]
        """
        return self.__forward(inputs=inputs, output_names=output_names, executing_order=self._executing_order, hooks=hooks)

    def __forward(self, inputs: Union[dict, list, torch.Tensor], executing_order: List[Operation], output_names: List[str]=None, hooks: Dict[str, RuntimeHook]=None) ->List[torch.Tensor]:
        if isinstance(inputs, dict):
            for name, value in inputs.items():
                if name in self._graph.variables:
                    var = self._graph.variables[name]
                    var.value = value
                else:
                    None
        else:
            inputs = self.prepare_input(inputs=inputs)
            for key, value in inputs.items():
                assert isinstance(value, torch.Tensor), f'TorchExecutor can only accept tensor as its input, while {type(value)} was given'
                self._graph_input_dictionary[key].value = value
        last_idx = 0
        if output_names is None:
            output_names = [name for name in self._graph.outputs]
        for name in output_names:
            if name not in self._graph.variables:
                raise KeyError(f'You are requiring output value of variable {name}(is not a variable name), however it is not a valid variable of current graph.')
            source_op = self._graph.variables[name].source_op
            if source_op is not None:
                last_idx = max(last_idx, executing_order.index(source_op) + 1)
        visited_op, result_collector = [], [None for _ in output_names]
        for name in output_names:
            if name in inputs:
                result_collector[output_names.index(name)] = inputs[name]
        for operation in executing_order[:last_idx]:
            try:
                assert isinstance(operation, Operation), 'Oops, seems you got something weird in your graph'
                assert isinstance(operation.platform, TargetPlatform), f'Operation {operation.name} has an invalid platform setting, only PPQ.core.TargetPlatform is expected here, while {type(operation.platform)} was given'
                platform_dispatching_table = OPERATION_FORWARD_TABLE[operation.platform]
                if operation.type not in platform_dispatching_table:
                    raise NotImplementedError(f'Graph op: {operation.name}({operation.type}) has no backend implementation on target platform {operation.platform}.Register this op to ppq.executor.base.py and ppq.executor.op first')
                operation_forward_func = platform_dispatching_table[operation.type]
                operation_runtime_hook = hooks[operation.name] if hooks is not None and operation.name in hooks else None
                inputs = [var.value for var in operation.inputs]
                if isinstance(operation, QuantableOperation):
                    input_configs = [_ for _ in operation.config.input_quantization_config]
                    inputs = [self.quantize_function(input, config) for input, config in zip(inputs, input_configs)]
                for idx, var in enumerate(operation.inputs):
                    if var.name in output_names:
                        result_collector[output_names.index(var.name)] = inputs[idx]
                if operation_runtime_hook is not None:
                    if isinstance(operation_runtime_hook, QuantOPRuntimeHook):
                        inputs = operation_runtime_hook.pre_forward_hook(inputs=[var.value for var in operation.inputs], quant_inputs=inputs, quant_configs=input_configs)
                    elif isinstance(operation_runtime_hook, RuntimeHook):
                        inputs = operation_runtime_hook.pre_forward_hook(inputs=inputs)
                    else:
                        raise TypeError(f'invalid hook instance was given with operation: {operation}')
                outputs = operation_forward_func(operation, inputs, self._executing_context)
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                fp_outputs = outputs
                if isinstance(operation, QuantableOperation):
                    output_configs = [_ for _ in operation.config.output_quantization_config]
                    outputs = [self.quantize_function(output, config) for output, config in zip(outputs, output_configs)]
                if operation_runtime_hook is not None:
                    if isinstance(operation_runtime_hook, QuantOPRuntimeHook):
                        outputs = operation_runtime_hook.post_forward_hook(outputs=fp_outputs, quant_outputs=outputs, quant_configs=output_configs)
                    elif isinstance(operation_runtime_hook, RuntimeHook):
                        outputs = operation_runtime_hook.post_forward_hook(outputs=outputs)
                    else:
                        raise TypeError(f'invalid hook instance was given with operation: {operation}')
                for output_idx, output_var in enumerate(operation.outputs):
                    output_var = operation.outputs[output_idx]
                    output_var.value = outputs[output_idx]
                    if output_var.name in output_names:
                        result_collector[output_names.index(output_var.name)] = outputs[output_idx]
            except Exception as _:
                raise RuntimeError(f'Error happens when dealing with operation {str(operation)}') from _
            visited_op.append(operation)
            for var in operation.inputs:
                if var.is_parameter:
                    continue
                if all(op in visited_op for op in var.dest_ops):
                    var.value = None
        for var in self._graph.variables.values():
            if not var.is_parameter:
                var.value = None
        return result_collector

    @torch.no_grad()
    @empty_ppq_cache
    def tracing_operation_meta(self, inputs: Union[dict, list, torch.Tensor], output_names: List[str]=None) ->None:
        """Tracing meta data for each operation, if there are some already
        created meta data with your operation, They will be override without
        warrning.

        Args:
            inputs (Union[dict, list, torch.Tensor]): [description]
            output_names (List[str], optional): [description]. Defaults to None.
        """
        hooks = {}
        for op_name, operation in self._graph.operations.items():
            hooks[op_name] = TorchMetaDataTracingHook(operation=operation)
        self.__forward(inputs=inputs, output_names=output_names, executing_order=self._executing_order, hooks=hooks)

    def load_graph(self, graph: BaseGraph) ->dict:
        super().load_graph(graph)
        self._deployed = False
        self._runnable_graph = RunnableGraph(self._graph)
        self._runnable_graph(GraphDeployCommand(device=self._device))

    def quantize_function(self, tensor: torch.Tensor, config: TensorQuantizationConfig=None) ->torch.Tensor:
        if config is None or not QuantizationStates.is_activated(config.state):
            return tensor
        elif config in self._delegates:
            return self._delegates[config](tensor, config)
        else:
            return self._default_quant_fn(tensor, config)

    def dummy_forward(self, hooks: Dict[str, RuntimeHook]=None) ->None:
        """This function allows you to execute entire graph without feeding any
        data. This feature is required for operation parameter quantization.
        See also: ppq.quantization.optim.ParameterQuantizePass.

        This function fakes some input tensors via operation metadata.
            ATTENTION: operation must have metadata before invoking this function.

        Args:
            hooks (Dict[str, RuntimeHook], optional):
                A hook table for customizing operation behaviour and collate data during executing.
                All hooks should inherit from class RuntimeHook, with all necessary methods implemented.
                    See also: ppq.executor.base.RuntimeHook

                Executor calls hook.pre_forward_hook(operation, input_data) before dispatching operation,
                by using this feature, you can dynamically dispatch operation during executing,
                or processing input data as you want.(remember to return processed input data)

                Executor calls hook.post_forward_hook(operation, output_data) after the execution,
                you are supposed to  gather all necessary data from execution via this feature.

                For Quantable Operation, a much more powerful class:
                    ppq.executor.base.QuantOpRuntimeHook is provided.
                see also: ppq.executor.base.QuantOpRuntimeHook

                Defaults to None.
        """
        feed_dict = {}
        for var_name, input_var in self._graph.inputs.items():
            if len(input_var.dest_ops) == 0:
                continue
            assert input_var.shape is not None, f'Can not generate dummy input for input variable {input_var.name}, input shape is not specified.'
            feed_dict[var_name] = torch.Tensor(size=input_var.shape, device='cpu').fill_(0).type(dtype=DataType.to_torch(input_var.dtype))
        self.forward(inputs=feed_dict, hooks=hooks)

    def partial_graph_forward(self, operations: List[Operation], feed_dict: Dict[str, torch.Tensor], output_names: List[str]) ->List[torch.Tensor]:
        """This forward function allows you to execute a series operations in
        your graph. (only operations list in your params will be executed with
        this function) Which serves as a great feature for quantization aware
        training.

        Args:
            operations (List[Operation]):
                operations that you want to execute,
                notice that executing will strictly follow your operation order.

            feed_dict (Dict[str, torch.Tensor]):
                an dictionary contains {variable name: value}, as an input to this execution.

            output_names (List[str]):
                output variable names.

        Returns:
            List[torch.Tensor]: [description]
        """
        return self.__forward(inputs=feed_dict, output_names=output_names, executing_order=operations)


def ppq_debug_function(func: Callable):
    """mark a function to be a debug function.

    Args:
        func (Callable): decorated function
    """

    def _wrapper(*args, **kwargs):
        if PPQ_CONFIG.PPQ_DEBUG:
            debug_str = func(*args, **kwargs)
            if debug_str is None:
                return None
            assert isinstance(debug_str, str), f'ppq_debug_function should only return string instance, while {str(func)} returns {type(debug_str)}'
            None
        else:
            return None
    return _wrapper


class BaseTensorObserver(metaclass=ABCMeta):

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        self._watch_on = watch_on
        self._quant_cfg = quant_cfg

    @abstractmethod
    def observe(self, value: Any):
        raise NotImplementedError('Implement this function first.')

    @abstractmethod
    def render_quantization_config(self):
        raise NotImplementedError('Implement this function first.')

    def __str__(self) ->str:
        return 'PPQ Tensor Observer (' + self.__class__.__name__ + ') mount on variable ' + self._watch_on.name + ' observing algorithm: ' + self._quant_cfg.observer_algorithm

    @ppq_debug_function
    def report(self) ->str:
        if self._quant_cfg.state == QuantizationStates.ACTIVATED:
            return f'Observer on Variable {self._watch_on.name}, computed scale: {self._quant_cfg.scale}, computed offset: {self._quant_cfg.offset}\n'


class ConstantObserver(BaseTensorObserver):
    """
    This observer will directly return scale = 1, offset = 0
    """

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._value_shape = None
        self._value_device = None

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        self._value_shape = value.shape
        self._value_device = value.device

    def render_quantization_config(self):
        device = self._value_device
        if self._quant_cfg.policy.has_property(QuantizationProperty.FLOATING):
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                self._quant_cfg.scale = torch.tensor([1.0], dtype=torch.float32, device=device).squeeze(0)
                self._quant_cfg.offset = torch.tensor([0.0], dtype=torch.float32, device=device).squeeze(0)
                self._quant_cfg.state = QuantizationStates.ACTIVATED
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                num_of_elements = self._value_shape[self._quant_cfg.channel_axis]
                scales = [(1.0) for _ in range(num_of_elements)]
                offsets = [(0.0) for _ in range(num_of_elements)]
                self._quant_cfg.scale = torch.tensor(scales, dtype=torch.float32, device=device)
                self._quant_cfg.offset = torch.tensor(offsets, dtype=torch.float32, device=device)
                self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError('This Observer is designed for floating quantization.')


OBSERVER_FLOATING_MSE_FETCHES = 4096


def generate_indexer(num_of_fetches: int, num_of_elements: int, seed: int=539038256) ->torch.Tensor:
    """Sample with a given seed. This function will generates a indexer based
    on your seed.

    Args:
        num_of_fetches (int): [description]
        num_of_elements (int): [description]
        seed (int, optional): [description]. Defaults to 0x20211230.

    Returns:
        torch.Tensor: [description]
    """
    indexer = []
    for i in range(num_of_fetches):
        indexer.append(seed % num_of_elements)
        seed = (214013 * seed + 2531011) % (2 << 31)
    return torch.tensor(indexer, dtype=torch.int32)


def generate_torch_indexer(num_of_fetches: int, num_of_elements: int) ->torch.Tensor:
    return torch.randint(low=0, high=num_of_elements, size=[num_of_fetches])


def channel_random_fetch(tensor: torch.Tensor, fetchs_per_channel: int=1024, seed: int=None, channel_axis: int=0) ->torch.Tensor:
    """Fetch some elements from tensor randomly by each channel. if a valid
    seed is given, elements will be sampled based on your seed, otherwise a
    random seed will be generated.

    result: [num_of_channel, fetchs_per_channel]

    Args:
        tensor (torch.Tensor): [description]
        fetchs_per_channel (int, optional): [description]. Defaults to 1024.
        channel_axis (int, optional): [description]. Defaults to 0.

    Returns:
        torch.Tensor: [description]
    """
    tensor = tensor.transpose(0, channel_axis)
    tensor = tensor.flatten(start_dim=1)
    num_of_elements = tensor.shape[-1]
    assert num_of_elements > 0, 'Can not fetch data from tensor with less than 1 elements.'
    if seed is None:
        indexer = generate_torch_indexer(num_of_fetches=fetchs_per_channel, num_of_elements=num_of_elements)
    else:
        indexer = generate_indexer(num_of_fetches=fetchs_per_channel, num_of_elements=num_of_elements, seed=seed)
    return tensor.index_select(dim=-1, index=indexer.long())


def tensor_random_fetch(tensor: torch.Tensor, seed: int=None, num_of_fetches: int=1024) ->torch.Tensor:
    """Fetch some elements from tensor randomly. if a valid seed is given,
    elements will be sampled based on your seed, otherwise a random seed will
    be generated.

    Args:
        tensor (torch.Tensor): [description]
        num_of_fetches (int, optional): [description]. Defaults to 1024.
    """
    tensor = tensor.flatten()
    num_of_elements = tensor.numel()
    assert num_of_elements > 0, 'Can not fetch data from tensor with less than 1 elements.'
    if seed is None:
        indexer = generate_torch_indexer(num_of_fetches=num_of_fetches, num_of_elements=num_of_elements)
    else:
        indexer = generate_indexer(num_of_fetches=num_of_fetches, num_of_elements=num_of_elements, seed=seed)
    return tensor.index_select(dim=0, index=indexer.long())


class DirectMSEObserver(BaseTensorObserver):
    """
    Direct MSE Observer, this observer compute MSE Loss directly.
    In PPQ there is another version of MSE Observer called TorchMSEObserver,
        which uses histogram for accelerating the computation of MSE Loss.

    We prefer to implements a direct MSE Observer for floating quantization,
        cause floating quantization can have only a few candidates of scale, which
        greatly reduce the computation complexity of solving MSE Loss.
    """

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        if not quant_cfg.policy.has_property(QuantizationProperty.FLOATING):
            raise TypeError('MSE Floating Observer is designed for floating quantization.')
        if not quant_cfg.policy.has_property(QuantizationProperty.POWER_OF_2):
            raise TypeError('MSE Floating Observer is designed for power-of-2 quantization.')
        self._value_shape = None
        self._value_device = None
        self._collector = []
        self._fetches = OBSERVER_FLOATING_MSE_FETCHES

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            if self._watch_on.is_parameter:
                value = torch.transpose(value, dim0=0, dim1=self._quant_cfg.channel_axis)
                self._collector.append(torch.flatten(value, start_dim=1))
            else:
                self._collector.append(channel_random_fetch(value, fetchs_per_channel=self._fetches))
                self._value_device = value.device
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            self._collector.append(tensor_random_fetch(value, num_of_fetches=self._fetches))

    def render_quantization_config(self):
        device = self._value_device
        scale_candidates = [0.0078125, 0.03125, 0.125, 1.0, 4.0, 16.0, 64.0]
        if not self._collector:
            raise PermissionError('Observer collector is empty, you should invoke observe function before render quantization config.')
        self._quant_cfg.state = QuantizationStates.ACTIVATED
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            self._collector = torch.cat(self._collector, dim=0)
            num_of_channel = self._collector.shape[0]
            scales = torch.ones(size=[num_of_channel], dtype=torch.float32, device=device)
            offsets = torch.zeros(size=[num_of_channel], dtype=torch.float32, device=device)
            losses = []
            for scale in scale_candidates:
                self._quant_cfg._scale = scales * scale
                self._quant_cfg._offset = offsets
                qt = PPQuantFunction(self._collector, self._quant_cfg)
                fp = self._collector
                loss = torch.mean(torch.square(qt - fp), dim=-1, keepdim=True)
                losses.append(loss)
            scale_index = torch.argmin(torch.cat(losses, dim=-1), dim=-1)
            best_scales = []
            for index in scale_index.cpu():
                best_scales.append(scale_candidates[index])
            self._quant_cfg._scale = torch.tensor(best_scales)
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            self._collector = torch.cat(self._collector, dim=0)
            scales = torch.ones(size=[1], dtype=torch.float32, device=device)
            offsets = torch.zeros(size=[1], dtype=torch.float32, device=device)
            losses = []
            for scale in scale_candidates:
                self._quant_cfg._scale = scales * scale
                self._quant_cfg._offset = offsets
                qt = PPQuantFunction(self._collector, self._quant_cfg)
                fp = self._collector
                loss = torch.mean(torch.square(qt - fp))
                losses.append((loss.item(), scale))
            best_scale = sorted(losses)[0][1]
            self._quant_cfg._scale = scales * best_scale


class SingletonMeta(type):
    """The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.

    see also: https://refactoring.guru/design-patterns/singleton/python/example
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Possible changes to the value of the `__init__` argument do not
        affect the returned instance."""
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ComplieHelper(metaclass=SingletonMeta):

    def __init__(self) ->None:
        self.__CUDA_EXTENTION__ = None

    def complie(self):
        ppq_warning('Compling Kernels... Please wait (It will take a few minutes).')
        lock_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/build/lock')
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception as e:
                raise PermissionError(f'Can not delete lock file at {lock_file}, delete it first!')
        self.__CUDA_EXTENTION__ = load(name='PPQ_Cuda_Impls', sources=[os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/export.cc'), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/linear.cu'), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/sort.cu'), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/train.cu'), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/floating.cu'), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cpu/hist_mse.cc')], build_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/build/'), with_cuda=True, extra_cflags=['-O3'])

    @property
    def CUDA_EXTENSION(self):
        if self.__CUDA_EXTENTION__ is None:
            raise Exception('Cuda Extension has not been compiled, invoke ppq.core.ffi.ComplieHelper.complie() First.')
        return self.__CUDA_EXTENTION__


CUDA_COMPLIER = ComplieHelper()


class CUDA:
    """CUDA is a helper class for invoking highly-effcient custimized cuda
    kernel. PPQ developer team has implemented a series of quantization related
    cuda kernel, They are 5-100x faster than torch kernels, with less gpu
    memory cost.

    You can easily extend your cuda kernel via this class:
        Firstly, implement your kernel within ppq/csrc/cuda, write your own .cu file and .h file.
        Secondly, add your functions to ppq/csrc/cuda/export.cc, add them to export table.
        Finally, add a interface with this python class(ppq.core.ffi.CUDA),
        following the signature as same as others.

    PPQ CUDA EXTENSION 命名规则:
        我们使用函数名+后缀名的形式命名 CUDA Extension 函数:

        后缀名 _T 表示 Tensorwise 函数
        后缀名 _C 表示 Channelwise 函数
        后缀名 _B 表示 导函数

    例如函数 LinearQuantize_T_B 表示线性量化函数的 Tensorwise 版本，并且是导函数。
    """

    @staticmethod
    def LinearQuantize_T(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, minimum: int=-128, maximum: int=127, rounding: int=0) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LT(tensor, scales, offsets, minimum, maximum, rounding)

    @staticmethod
    def LinearQuantize_C(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, channel_axis: int, minimum: int=-128, maximum: int=127, rounding: int=0) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LC(tensor, scales, offsets, minimum, maximum, channel_axis, rounding)

    @staticmethod
    def LinearQuantize_T_B(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, dy: torch.Tensor, minimum: int, maximum: int, rounding: int) ->List[torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LT_B(tensor, scales, offsets, dy, minimum, maximum, rounding)

    @staticmethod
    def LinearQuantize_C_B(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, dy: torch.Tensor, minimum: int, maximum: int, channel_axis: int, rounding: int) ->List[torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LC_B(tensor, scales, offsets, dy, minimum, maximum, rounding, channel_axis)

    @staticmethod
    def Histogram_T(tensor: torch.Tensor, histogram: torch.Tensor, scale: float, clip_outliers: bool=True) ->torch.Tensor:
        CUDA_COMPLIER.CUDA_EXTENSION.Histogram_T(tensor, scale, clip_outliers, histogram)
        return histogram

    @staticmethod
    def Histogram_Asymmetric_T(min_value: float, max_value: float, tensor: torch.Tensor, histogram: torch.Tensor, clip_outliers: bool=True) ->torch.Tensor:
        CUDA_COMPLIER.CUDA_EXTENSION.Histogram_Asymmetric_T(min_value, max_value, tensor, clip_outliers, histogram)
        return histogram

    @staticmethod
    def Histogram_C(tensor: torch.Tensor, channel_axis: int, histogram: torch.Tensor, scale: float, clip_outliers: bool=True) ->torch.Tensor:
        CUDA_COMPLIER.CUDA_EXTENSION.Histogram_C(tensor, channel_axis, scale, clip_outliers, histogram)
        return histogram

    @staticmethod
    def Quantile(tensor: torch.Tensor, q: float) ->torch.Tensor:
        return CUDA_COMPLIER.CUDA_EXTENSION.Quantile_T(tensor, q)

    @staticmethod
    def TensorClip_T(tensor: torch.Tensor, reference: torch.Tensor, limit: torch.Tensor) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not reference.is_contiguous():
            tensor = reference.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.TensorClip_T(tensor, reference, limit)

    @staticmethod
    def TensorClip_C(tensor: torch.Tensor, reference: torch.Tensor, limit: torch.Tensor, channel_axis: int) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not reference.is_contiguous():
            tensor = reference.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.TensorClip_C(tensor, reference, limit, channel_axis)

    @staticmethod
    def RoundingLoss_LT(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, minimum: int=-128, maximum: int=127, rounding: int=0) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LT(tensor, scales, offsets, minimum, maximum, rounding)

    @staticmethod
    def RoundingLoss_LT_B(tensor: torch.Tensor, dy: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, minimum: int=-128, maximum: int=127, rounding: int=0) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LT_B(tensor, dy, scales, offsets, minimum, maximum, rounding)

    @staticmethod
    def RoundingLoss_LC(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, channel_axis: int, minimum: int=-128, maximum: int=127, rounding: int=0) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LC(tensor, scales, offsets, minimum, maximum, channel_axis, rounding)

    @staticmethod
    def RoundingLoss_LC_B(tensor: torch.Tensor, dy: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, channel_axis: int, minimum: int=-128, maximum: int=127, rounding: int=0) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LC_B(tensor, dy, scales, offsets, minimum, maximum, channel_axis, rounding)

    @staticmethod
    def OrderPreservingObserve(tensor: torch.Tensor) ->torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LC_B(tensor)

    @staticmethod
    def compute_mse_loss(histogram: list, start: int, step: int, end: int) ->float:
        return CUDA_COMPLIER.CUDA_EXTENSION.compute_mse_loss(histogram, start, step, end)

    @staticmethod
    def FloatingQuantize_T(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, exponent: int=4, mantissa: int=3, minimum: float=-448, maximum: float=+448, rounding: int=0) ->torch.Tensor:
        if exponent <= 0:
            raise ValueError('Floating Quantization requires exponent > 0')
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FT(tensor, scales, offsets, exponent, mantissa, minimum, maximum, rounding)

    @staticmethod
    def FloatingQuantize_C(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, channel_axis: int, exponent: int=4, mantissa: int=3, minimum: float=-448, maximum: float=+448, rounding: int=0) ->torch.Tensor:
        if exponent <= 0:
            raise ValueError('Floating Quantization requires exponent > 0')
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FC(tensor, scales, offsets, exponent, mantissa, minimum, maximum, channel_axis, rounding)

    @staticmethod
    def FloatingQuantize_T_B(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, dy: torch.Tensor, exponent: int, mantissa: int, minimum: float, maximum: float, rounding: int) ->List[torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FT_B(tensor, scales, offsets, dy, exponent, mantissa, minimum, maximum, rounding)

    @staticmethod
    def FloatingQuantize_C_B(tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, dy: torch.Tensor, exponent: int, mantissa: int, minimum: float, maximum: float, channel_axis: int, rounding: int) ->List[torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FC_B(tensor, scales, offsets, dy, exponent, mantissa, minimum, maximum, rounding, channel_axis)

    @staticmethod
    def Sync():
        """Synchronize device."""
        synchronize()


OBSERVER_KL_COMPUTING_DEVICE = 'cpu'


OBSERVER_KL_HIST_BINS = 4096


OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE = 'OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE'


OBSERVER_MIN_SCALE = 1e-08


OBSERVER_MIN_SCALE_MANUL_OVERRIDE = 'OBSERVER_MIN_SCALE_MANUL_OVERRIDE'


OBSERVER_WARNING = False


def ppq_numerical_round(value: float, policy: RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) ->int:
    """
        reference: https://en.wikipedia.org/wiki/Rounding

        decimal definition:
            - decimal.ROUND_CEILING (towards Infinity)
            - decimal.ROUND_DOWN (towards zero)
            - decimal.ROUND_FLOOR (towards -Infinity)
            - decimal.ROUND_HALF_DOWN (to nearest with ties going towards zero)
            - decimal.ROUND_HALF_EVEN (to nearest with ties going to nearest even integer)
            - decimal.ROUND_HALF_UP (to nearest with ties going away from zero)
            - decimal.ROUND_UP (away from zero)
            - decimal.ROUND_05UP (away from zero if last digit after rounding towards zero would have been 0 or 5; otherwise towards zero)

    Args:
        value (float): [description]
        policy (RoundingPolicy, optional): [description]. Defaults to RoundingPolicy.ROUND_HALF_EVEN.

    Raises:
        ValueError: [description]

    Returns:
        int: [description]
    """
    assert isinstance(value, float), 'numerical round only takes effect on float number.'
    if policy == RoundingPolicy.ROUND_HALF_EVEN:
        return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_EVEN))
    elif policy == RoundingPolicy.ROUND_HALF_UP:
        if value > 0:
            return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_UP))
        else:
            return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_DOWN))
    elif policy == RoundingPolicy.ROUND_HALF_DOWN:
        if value > 0:
            return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_DOWN))
        else:
            return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_UP))
    elif policy == RoundingPolicy.ROUND_HALF_TOWARDS_ZERO:
        return ppq_numerical_round(value, RoundingPolicy.ROUND_HALF_DOWN)
    elif policy == RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO:
        return ppq_numerical_round(value, RoundingPolicy.ROUND_HALF_UP)
    elif policy == RoundingPolicy.ROUND_TO_NEAR_INT:
        if value > 0:
            return floor(value + 0.5)
        else:
            return ceil(value - 0.5)
    elif policy == RoundingPolicy.ROUND_UP:
        return ceil(value)
    else:
        raise ValueError('Unexpected rounding policy found.')


def ppq_quant_param_computing_function(func: Callable):
    """mark a function to be a scale-computing function.

    Args:
        func (Callable): decorated function
    """

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return _wrapper


def ppq_round_to_power_of_2(value: Union[float, int], policy: RoundingPolicy=RoundingPolicy.ROUND_UP) ->float:
    """
    Round a given value to Integer, under Power-of-2 restrction.

    ### Formula:
    ppq_round_to_power_of_2(x) = sign * float(pow(2, round(log2(sign * value), policy=policy)))
        where sign is +1 or -1

    Args:
        value (Union[float, int]): _description_
        policy (RoundingPolicy, optional): _description_. Defaults to RoundingPolicy.ROUND_UP.

    Returns:
        float: _description_
    """
    if value == 0:
        return 0
    sign = 1 if value >= 0 else -1
    assert isinstance(value, float) or isinstance(value, int), 'power-of-2 round only takes effect on float or int.'
    return sign * float(pow(2, ppq_numerical_round(log2(sign * value), policy=policy)))


@ppq_quant_param_computing_function
def minmax_to_scale_offset(min_val: float, max_val: float, config: TensorQuantizationConfig, scale_threshold: float=OBSERVER_MIN_SCALE) ->Tuple[float, float]:
    if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
        scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
    scale, offset = 1, 0
    if min_val > 0:
        min_val = 0
    if max_val < 0:
        max_val = 0
    if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
        range = float(max_val - min_val)
        scale = range / (config.quant_max - config.quant_min)
        if scale < scale_threshold and OBSERVER_WARNING:
            ppq_warning('Numeric instability detected: ppq find there is a scale value < 1e-7, which probably cause numeric underflow in further computation.')
        scale = max(scale, scale_threshold)
        offset = ppq_numerical_round(-min_val / scale)
    elif config.policy.has_property(QuantizationProperty.SYMMETRICAL):
        range = 2 * float(max(abs(max_val), abs(min_val)))
        scale = range / (config.quant_max - config.quant_min)
        if scale < scale_threshold and OBSERVER_WARNING:
            ppq_warning('Numeric instability detected: ppq find there is a scale value < 1e-7, which probably cause numeric underflow in further computation.')
        scale = max(scale, scale_threshold)
        offset = 0
    else:
        raise TypeError('Tensor Min Max Observer Excepts either ASYMMETRICAL or SYMMETRICAL quantization config.')
    if config.policy.has_property(QuantizationProperty.POWER_OF_2):
        scale = ppq_round_to_power_of_2(scale, policy=RoundingPolicy.ROUND_UP)
    return scale, offset


class TorchMinMaxObserver(BaseTensorObserver):

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._min_val_collector = []
        self._max_val_collector = []

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), 'TorchMinMaxObserver can only deal with torch Tensor values'
        assert value.numel() > 0, f'You are observing an empty tensor({self._watch_on.name}).'
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                self._min_val_collector.append(value.min().reshape(shape=[1]))
                self._max_val_collector.append(value.max().reshape(shape=[1]))
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                channel_axis = self._quant_cfg.channel_axis
                channelwise_view = value.transpose(dim0=0, dim1=channel_axis).unsqueeze(-1)
                channelwise_view = torch.flatten(channelwise_view, start_dim=1)
                self._min_val_collector.append(torch.min(channelwise_view, dim=1, keepdim=True)[0])
                self._max_val_collector.append(torch.max(channelwise_view, dim=1, keepdim=True)[0])
            else:
                raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')

    def render_quantization_config(self):
        if len(self._max_val_collector) == 0:
            raise ValueError('Can not render quantization config yet, Observer data collator is empty. Invoke observe() function before render config.')
        device = self._max_val_collector[-1].device
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            scale, offset = minmax_to_scale_offset(min_val=torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item(), max_val=torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item(), config=self._quant_cfg)
            self._quant_cfg.scale = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            min_vals = torch.min(torch.cat(self._min_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
            max_vals = torch.max(torch.cat(self._max_val_collector, dim=-1), dim=-1, keepdim=False)[0].cpu().numpy()
            assert len(min_vals) == len(max_vals), 'Min values and max values should at same length.'
            scales, offsets = [], []
            for min_val, max_val in zip(min_vals, max_vals):
                scale, offset = minmax_to_scale_offset(min_val=min_val, max_val=max_val, config=self._quant_cfg)
                scales.append(scale)
                offsets.append(offset)
            self._quant_cfg.scale = torch.tensor(scales, dtype=torch.float32, device=device)
            self._quant_cfg.offset = torch.tensor(offsets, dtype=torch.float32, device=device)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')


def torch_KL_divergence(hist: torch.Tensor, ref_hist: torch.Tensor, eps=1e-30) ->float:
    if hist.ndim != 1 or ref_hist.ndim != 1:
        raise ValueError(f'Only 1 dimension tensor can compute KL divergence with another tensor. While your input hist has dimension {hist.ndim} and ref_hist has dimension {ref_hist.ndim}')
    if len(hist) != len(ref_hist):
        raise ValueError('Can not compute KL divergence, len(hist) != len(ref_hist')
    return torch.dot(hist.double(), torch.log10(hist.double() + eps) - torch.log10(ref_hist.double() + eps)).item()


class TorchHistObserver(TorchMinMaxObserver):

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig, hist_bins: int=OBSERVER_KL_HIST_BINS):
        self._phase = 'Detecting Minmax'
        self._hist = None
        self._hist_scale = None
        self._min = None
        self._max = None
        if OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE in quant_cfg.detail:
            hist_bins = quant_cfg.detail[OBSERVER_KL_HIST_BINS_MANUL_OVERRIDE]
        self._hist_bins = hist_bins
        super().__init__(watch_on, quant_cfg)

    def observe(self, value: torch.Tensor):
        assert value.numel() > 0, f'You are observing an empty tensor({self._watch_on.name}).'
        if self._phase == 'Detecting Minmax':
            return super().observe(value)
        elif self._phase == 'Collating Hist':
            if self._hist is None:
                self._hist = torch.zeros(size=(self._hist_bins,), dtype=torch.int32, device=value.device)
            if self._quant_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                if PPQ_CONFIG.USING_CUDA_KERNEL and value.is_cuda:
                    CUDA.Histogram_Asymmetric_T(self._min, self._max, tensor=value, histogram=self._hist)
                else:
                    hist = torch.histc(value, self._hist_bins, min=self._min, max=self._max)
                    self._hist += hist.int()
            elif self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                if PPQ_CONFIG.USING_CUDA_KERNEL and value.is_cuda:
                    CUDA.Histogram_T(tensor=value, histogram=self._hist, scale=self._hist_scale)
                else:
                    hist = torch.histc(torch.abs(value), self._hist_bins, min=0, max=self._hist_scale * self._hist_bins)
                    self._hist += hist.int()
            else:
                raise TypeError('Quantization Property is invalid, expect either ASYMMETRICAL or SYMMETRICAL config here.')

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(self, histogram: torch.Tensor, hist_bins: int, hist_scale: float, config: TensorQuantizationConfig, computing_device: str=OBSERVER_KL_COMPUTING_DEVICE, scale_threshold: float=OBSERVER_MIN_SCALE) ->Tuple[float, int]:
        """
        PPQ core quant parameter computing method - Histogram to scale & offset

        With a pre-defined histogram,
        this function will automatically search best clip value
        to minimize KL divergence between quantized result and fp32 input.

        only work for per-tensor symmetrical quantization policy for now.
        see also https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

        Args:
            histogram (torch.Tensor): histogram records activation's statistics.
            hist_bins (int): how many bins are included in histogram(also known as histogram length)
            hist_scale (float): histogram step size. it can be solved by histogram.max_val / histogram.bins
            config (TensorQuantizationConfig): quantization config.
            computing_device (str, optional): computing device. Defaults to 'cpu'.

        Raises:
            ValueError: given quantization config is invalid.

        Returns:
            Tuple[float, int]: scale(fp32) and offset(int).
        """
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            raise PermissionError('KL observer is not designed for ASYMMETRICAL quantization')
        if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
            scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
        histogram = histogram.float()
        losses, quant_bins = [], 2 ** (config.num_of_bits - 1)
        histogram[:int(hist_bins * 0.002)] = 0
        histogram[int(hist_bins * 0.002)] = 1
        hist_sum = torch.sum(histogram)
        for bin_range in range(quant_bins, hist_bins + quant_bins - 1, quant_bins):
            p_hist = torch.zeros(size=(bin_range,), dtype=torch.float, device=computing_device)
            p_hist[:bin_range].copy_(histogram[:bin_range])
            p_hist[bin_range - 1] += torch.sum(histogram[bin_range:])
            p_hist = p_hist / hist_sum
            expand_ratio = int(bin_range / quant_bins)
            q_hist = histogram[:bin_range].clone()
            q_hist = q_hist.reshape((quant_bins, expand_ratio))
            positive_map = q_hist > 0
            positive_cnt = positive_map.sum(axis=1, keepdim=True)
            positive_cnt[positive_cnt == 0] = 1
            q_hist = torch.div(q_hist.sum(axis=1, keepdim=True), positive_cnt)
            q_hist = q_hist.repeat([1, expand_ratio])
            q_hist = q_hist * positive_map
            q_hist = q_hist / torch.sum(q_hist)
            q_hist = q_hist.flatten()
            losses.append({'kl': torch_KL_divergence(p_hist, q_hist), 'bin_range': bin_range})
        best_bin_range = sorted(losses, key=lambda x: x['kl'])[0]['bin_range']
        scale, offset = best_bin_range / self._hist_bins * hist_scale * (self._hist_bins / quant_bins), 0
        if scale < scale_threshold and OBSERVER_WARNING:
            ppq_warning('Numeric instability detected: ppq find there is a scale value < 1e-7, which probably cause numeric underflow in further computation.')
        scale = max(scale, scale_threshold)
        if config.policy.has_property(QuantizationProperty.POWER_OF_2):
            scale = ppq_round_to_power_of_2(scale, policy=RoundingPolicy.ROUND_HALF_UP)
        return scale, offset

    def render_quantization_config(self):
        if not self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            raise ValueError('Hist observer can only apply with per-tensor quantization config.')
        if self._phase == 'Detecting Minmax':
            min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item()
            max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item()
            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                hist_range = float(max(abs(max_val), abs(min_val)))
            else:
                hist_range = max_val - min_val
            self._min = min_val
            self._max = max_val
            self._hist_scale = hist_range / self._hist_bins
            self._phase = 'Collating Hist'
        elif self._phase == 'Collating Hist':
            scale, offset = self.hist_to_scale_offset(histogram=self._hist, hist_bins=self._hist_bins, hist_scale=self._hist_scale, config=self._quant_cfg)
            device = self._hist.device
            self._quant_cfg.scale = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED


class TorchIsotoneObserver(BaseTensorObserver):

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._cache = []
    """For softmax or sigmoid activations, usually we just need
    argmax(softmax(x)) == argmax(softmax(quant(x))) which is argmax(x) ==
    argmax(quant(x))

    Inspired by this Property, we designed an order-preserving calibration method,
        which cares only about max(x) [or min(x)]

    To keep argmax(x) == argmax(quant(x)), we only need to
        distinguish the largest element and the second largert element with quantization

        let L1 represents the largest element of x,
        while L2 represents the second largest.

        For symmetric quantization:
        We want
            quant(L1, scale) > quant(L2, scale)
            clip(round(L1 / scale)) > clip(round(L2 / scale))
        Which means:
            1. L1 - L2 > 0.5 * scale
            2. round(L2 / scale) < clip_max - 1
    Args:
        BaseTensorObserver ([type]): [description]
    """

    def observe(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), 'IsotoneObserver can only deal with torch Tensor values'
        assert value.numel() > 0, f'You are observing an empty tensor({self._watch_on.name}).'
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                value = value.flatten(start_dim=1)
                value, _ = torch.topk(value, k=2, dim=-1, largest=True, sorted=True)
                self._cache.append(value)
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                raise TypeError('IsotoneObserver is not designed for channelwise quantization.')
            else:
                raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')

    def render_quantization_config(self):
        device = self._cache[-1].device
        collected = torch.cat(self._cache, dim=0)
        collected = collected.cpu().numpy()
        s_maxs, s_mins = [], []
        for l1, l2 in collected:
            scale_min = l2 / (self._quant_cfg.quant_max - 1)
            scale_max = 2 * (l1 - l2)
            s_maxs.append(scale_max)
            s_mins.append(scale_min)
        best_satisfied, best_scales = 0, []
        for s_candidate in (s_maxs + s_mins):
            satisfied = 0
            for s_max, s_min in zip(s_maxs, s_mins):
                if s_candidate <= s_max:
                    satisfied += 1
                if s_candidate >= s_min:
                    satisfied += 1
            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_scales = [s_candidate]
            if satisfied == best_satisfied:
                best_scales.append(s_candidate)
        best_scale = sum(best_scales) / len(best_scales)
        self._quant_cfg.scale = torch.tensor([best_scale], dtype=torch.float32, device=device).squeeze(0)
        self._quant_cfg.offset = torch.tensor([0], dtype=torch.float32, device=device).squeeze(0)
        self._quant_cfg.state = QuantizationStates.ACTIVATED


OBSERVER_MSE_COMPUTE_INTERVAL = 8


OBSERVER_MSE_HIST_BINS = 2048


class TorchMSEObserver(TorchHistObserver):
    """Histogram accelerated MSE Observer, inspired by LightGBM This observer
    will collect data in histogram firstly, all mse computing will directly use
    histogram rather than data itself.

        Time complexity: O(Iteration * Num_of_Batch * Length(Data)) -> O(Iteration * Length(histogram))
        Space complexity: O(Num_of_Batch * Length(Data)) -> O(Iteration * Length(histogram))

    Args:
        TorchHistObserver ([type]): [description]
    """

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig, bins: int=OBSERVER_MSE_HIST_BINS):
        super().__init__(watch_on, quant_cfg)
        self._hist_bins = bins

    def compute_mse_loss(self, histogram: list, start: int, step: int, end: int):
        if PPQ_CONFIG.USING_CUDA_KERNEL:
            return CUDA.compute_mse_loss(histogram=histogram, start=start, step=step, end=end)
        else:
            num_of_elements = sum(histogram)
            loss = 0
            for idx, bin in enumerate(histogram):
                if idx < start:
                    error = start - idx - 1 + 0.5
                elif idx > end:
                    error = idx - end + 0.5
                else:
                    l_idx = (idx - start) % step
                    r_idx = step - l_idx - 1
                    if l_idx == r_idx:
                        error = l_idx + 0.25
                    else:
                        l_err = l_idx + 0.5
                        r_err = r_idx + 0.5
                        error = min(l_err, r_err)
                loss += bin * error * error / num_of_elements
            return loss

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(self, histogram: torch.Tensor, hist_bins: int, hist_scale: float, config: TensorQuantizationConfig, scale_threshold: float=OBSERVER_MIN_SCALE) ->Tuple[float, int]:
        if OBSERVER_MIN_SCALE_MANUL_OVERRIDE in config.detail:
            scale_threshold = config.detail[OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
        histogram = convert_any_to_numpy(histogram).tolist()
        num_of_quant_levels = self._quant_cfg.quant_max - self._quant_cfg.quant_min + 1
        losses = []
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and config.policy.has_property(QuantizationProperty.PER_TENSOR):
            step = hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(histogram=histogram, start=0, step=step, end=num_of_quant_levels * step)
            losses.append({'mse': loss, 'start': 0, 'end': num_of_quant_levels * step})
            for start in range(0, hist_bins, OBSERVER_MSE_COMPUTE_INTERVAL):
                if start * hist_scale + self._min > 0:
                    break
                for step in range(1, hist_bins // num_of_quant_levels + 1):
                    end = start + num_of_quant_levels * step
                    if end > hist_bins + num_of_quant_levels:
                        break
                    loss = self.compute_mse_loss(histogram=histogram, start=start, step=step, end=end)
                    losses.append({'mse': loss, 'start': start, 'end': end})
            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_start = best_policy['start']
            best_end = best_policy['end']
            range_min, range_max = best_start * hist_scale + self._min, best_end * hist_scale + self._min
            scale, offset = minmax_to_scale_offset(range_min, range_max, config, scale_threshold)
            return scale, offset
        elif config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError('Torch Mse observer do not support PER_CHANNEL policy now, please wait.')
        elif config.policy.has_property(QuantizationProperty.SYMMETRICAL) and config.policy.has_property(QuantizationProperty.PER_TENSOR):
            step = hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(histogram=histogram, start=0, step=step, end=num_of_quant_levels * step)
            losses.append({'mse': loss, 'end': num_of_quant_levels * step})
            for step in range(1, hist_bins // num_of_quant_levels + 1):
                end = num_of_quant_levels * step
                if end > hist_bins + num_of_quant_levels:
                    break
                loss = self.compute_mse_loss(histogram=histogram, start=0, step=step, end=end)
                losses.append({'mse': loss, 'end': end})
            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_end = best_policy['end']
            range_min, range_max = -(best_end * hist_scale), best_end * hist_scale
            scale, offset = minmax_to_scale_offset(range_min, range_max, config, scale_threshold)
            return scale, offset
        raise Exception('Oops, there might be some mistakes.')


OBSERVER_PERCENTILE = 0.9999


OBSERVER_PERCENTILE_MANUL_OVERRIDE = 'OBSERVER_PERCENTILE_MANUL_OVERRIDE'


class TorchPercentileObserver(BaseTensorObserver):

    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        if not OBSERVER_PERCENTILE_MANUL_OVERRIDE in quant_cfg.detail:
            self._percentile = OBSERVER_PERCENTILE
        else:
            self._percentile = quant_cfg.detail[OBSERVER_PERCENTILE_MANUL_OVERRIDE]
        self._percentile_collector = []
        self._percentile_maxs = []
        self._percentile_mins = []

    @torch.no_grad()
    def observe(self, value: torch.Tensor):
        assert value.numel() > 0, f'You are observing an empty tensor({self._watch_on.name}).'
        assert isinstance(value, torch.Tensor), 'TorchMinMaxObserver can only deal with torch Tensor values'
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                if not PPQ_CONFIG.USING_CUDA_KERNEL or not value.is_cuda:
                    numel = value.numel()
                    min_idx, max_idx = int(numel * (1 - self._percentile)), int(numel * self._percentile)
                    min_idx = max(0, min_idx) + 1
                    max_idx = min(max_idx, numel - 1) + 1
                    _min = torch.kthvalue(value.flatten(), k=min_idx, dim=0)[0].view(1, -1)
                    _max = torch.kthvalue(value.flatten(), k=max_idx, dim=0)[0].view(1, -1)
                    self._percentile_collector.append(torch.cat([_max, _min], dim=-1))
                else:
                    self._percentile_collector.append(CUDA.Quantile(value, self._percentile).view(1, -1))
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                raise PermissionError('Percentile observer can not deal with per channel quantization.')
                channel_axis = self._quant_cfg.channel_axis
                channelwise_view = value.transpose(dim0=0, dim1=channel_axis)
                channelwise_view = torch.flatten(channelwise_view, start_dim=1)
                self._percentile_mins.append(-torch.quantile(-channelwise_view, q=self._percentile, dim=1, keepdim=True)[0])
                self._percentile_maxs.append(torch.quantile(channelwise_view, q=self._percentile, dim=1, keepdim=True)[0])
            else:
                raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')

    def render_quantization_config(self):
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            if len(self._percentile_collector) == 0:
                raise ValueError('Can not render quantization config yet, Observer data collator is empty. Invoke observe() function before render config.')
            device = self._percentile_collector[-1].device
            self._percentile_collector = torch.cat(self._percentile_collector, dim=0).mean(dim=0).cpu()
            scale, offset = minmax_to_scale_offset(min_val=self._percentile_collector[1].item(), max_val=self._percentile_collector[0].item(), config=self._quant_cfg)
            self._quant_cfg.scale = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError('Percentile observer can not deal with per channel quantization.')
            if len(self._percentile_maxs) == 0:
                raise ValueError('Can not render quantization config yet, Observer data collator is empty. Invoke observe() function before render config.')
            device = self._percentile_maxs[-1].device
            min_vals = torch.mean(torch.cat(self._percentile_mins, dim=-1), dim=-1, keepdim=False)
            max_vals = torch.mean(torch.cat(self._percentile_maxs, dim=-1), dim=-1, keepdim=False)
            min_vals = min_vals.cpu()
            max_vals = max_vals.cpu()
            assert len(min_vals) == len(max_vals), 'Min values and max values should at same length.'
            scales, offsets = [], []
            for min_val, max_val in zip(min_vals, max_vals):
                scale, offset = minmax_to_scale_offset(min_val=min_val, max_val=max_val, config=self._quant_cfg)
                scales.append(scale)
                offsets.append(offset)
            self._quant_cfg.scale = torch.tensor(scales, dtype=torch.float32, device=device)
            self._quant_cfg.offset = torch.tensor(offsets, dtype=torch.int32, device=device)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')


OBSERVER_TABLE = {'minmax': TorchMinMaxObserver, 'kl': TorchHistObserver, 'percentile': TorchPercentileObserver, 'mse': TorchMSEObserver, 'isotone': TorchIsotoneObserver, 'constant': ConstantObserver, 'floating': DirectMSEObserver}


class TensorObserverFactroy:

    def __init__(self) ->None:
        raise NotImplementedError('Observer Factory can not be initialized, use TensorObserverFactroy.build_observer instead.')

    @classmethod
    def build_observer(cls, variable: Variable, config: TensorQuantizationConfig) ->BaseTensorObserver:
        algorithm = str(config.observer_algorithm.lower())
        if algorithm not in OBSERVER_TABLE:
            raise ValueError(f'Observer type not understand, Except one of {OBSERVER_TABLE.keys()}, while {str(algorithm)} was given.')
        return OBSERVER_TABLE[algorithm](watch_on=variable, quant_cfg=config)


def Observer(quant_config: TensorQuantizationConfig, variable: Variable=None) ->BaseTensorObserver:
    """
    Get a Tensor Observer.

    Args:
        quant_config (TensorQuantizationConfig): _description_
        variable (Variable, optional): _description_. Defaults to None.

    Returns:
        BaseTensorObserver: _description_
    """
    return TensorObserverFactroy.build_observer(variable=variable, config=quant_config)


class TensorQuant(torch.nn.Module):

    def __init__(self, quant_config: TensorQuantizationConfig) ->None:
        """
        PPQ Tensor Quant

        Args:
            quant_config (TensorQuantizationConfig): _description_
            name (str, optional): _description_. Defaults to 'PPQ Quant Stub'.
        """
        self._quant_config = quant_config
        self._delegator = None
        self._batch_observed = 0
        self._observer = Observer(quant_config=quant_config)

    @property
    def delegator(self) ->Callable:
        return self._delegator

    @delegator.setter
    def delegator(self, func: Callable):
        self._delegator = func

    def forward(self, value: torch.Tensor) ->torch.Tensor:
        if self._delegator is not None:
            return self._delegator(value, self._quant_config)
        return QuantFunction(tensor=value, config=self._quant_config)

    def observe(self, value: torch.Tensor):
        self._batch_observed += 1
        self._observer.observe(value)

    def render(self):
        if self._batch_observed == 0:
            raise PermissionError('You have not provide any data to this QuantStub, PPQ can not render its quant config yet.')
        self._observer.render_quantization_config()


class ParameterQuant(TensorQuant):

    def __init__(self, quant_config: TensorQuantizationConfig, parameter: torch.Tensor) ->None:
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f'Expect a torch.Tensor here. However {type(parameter)} was given.')
        super().__init__(quant_config)
        self.observe(parameter)
        self.render()


def LinearQuantizationConfig(symmetrical: bool=True, dynamic: bool=False, power_of_2: bool=False, channel_axis: int=None, quant_min: int=-128, quant_max: int=127, num_of_bits=8, calibration: str='minmax', rounding: RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) ->TensorQuantizationConfig:
    sym = QuantizationProperty.SYMMETRICAL if symmetrical else QuantizationProperty.ASYMMETRICAL
    dyn = QuantizationProperty.DYNAMIC if dynamic else 0
    pw2 = QuantizationProperty.POWER_OF_2 if power_of_2 else 0
    chn = QuantizationProperty.PER_TENSOR if channel_axis is None else QuantizationProperty.PER_CHANNEL
    return TensorQuantizationConfig(policy=QuantizationPolicy(sym + dyn + pw2 + chn + QuantizationProperty.LINEAR), rounding=rounding, num_of_bits=num_of_bits, quant_min=quant_min, quant_max=quant_max, observer_algorithm=calibration)


class QATController:

    def __init__(self) ->None:
        self.export_mode = 'Plain Onnx'


class QuantLayer:

    def __init__(self) ->None:
        self._quantize_controler = None


class QConv1d(torch.nn.Conv1d, QuantLayer):

    def __init__(self, controller: QATController, **kwargs) ->None:
        super(torch.nn.Conv1d).__init__(**kwargs)
        self._quantize_controler = controller
        self.input_quant = TensorQuant(quant_config=LinearQuantizationConfig(symmetrical=True, power_of_2=False, channel_axis=None, calibration='percentile'))
        self.weight_quant = ParameterQuant(quant_config=LinearQuantizationConfig(symmetrical=True, power_of_2=False, channel_axis=0, calibration='minmax'))

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        x = self.input_quant(input)
        w = self.weight_quant(self.weight)
        return self._conv_forward(x, w, self.bias)


class QConv2d(torch.nn.Conv2d, QuantLayer):

    def __init__(self, controller: QATController, **kwargs) ->None:
        super(torch.nn.Conv2d).__init__(**kwargs)
        self._quantize_controler = controller
        self.input_quant = TensorQuant(quant_config=LinearQuantizationConfig(symmetrical=True, power_of_2=False, channel_axis=None, calibration='percentile'))
        self.weight_quant = ParameterQuant(quant_config=LinearQuantizationConfig(symmetrical=True, power_of_2=False, channel_axis=0, calibration='minmax'))

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        x = self.input_quant(input)
        w = self.weight_quant(self.weight)
        return self._conv_forward(x, w, self.bias)


class QConv3d(torch.nn.Conv3d, QuantLayer):

    def __init__(self, controller: QATController, **kwargs) ->None:
        super(torch.nn.Conv3d).__init__(**kwargs)
        self._quantize_controler = controller
        self.input_quant = TensorQuant(quant_config=LinearQuantizationConfig(symmetrical=True, power_of_2=False, channel_axis=None, calibration='percentile'))
        self.weight_quant = ParameterQuant(quant_config=LinearQuantizationConfig(symmetrical=True, power_of_2=False, channel_axis=1, calibration='minmax'))

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        x = self.input_quant(input)
        w = self.weight_quant(self.weight)
        return self._conv_forward(x, w, self.bias)


class TimeDecay:
    """A helper class computing time decay."""

    def __init__(self, t_max: int, decay: float=0.2, beta_start: float=20, beta_end: float=2):
        self.t_max = t_max
        self.start_decay = decay * t_max
        self.start_b = beta_start
        self.end_b = beta_end

    def __call__(self, t):
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class AdaroundRegTerm(torch.nn.Module):
    """Adaround Reg Term is a part of Adaround optimization algorithm.
    This term represents the difference between a fp32 value and its quantized counter-part.
        We use a same implementation as proposed in Adaround paper.
    Args:
        torch ([type]): [description]
    """

    def __init__(self, max_iter: int=20000, zeta: float=1.1, gamma: float=-0.1, alpha: float=0.01, beta: float=20, warm_ratio: float=0.2):
        self.max_iter = max_iter
        self.zeta = zeta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.warm_ratio = warm_ratio
        self.temp_anneal = TimeDecay(self.max_iter, self.warm_ratio)
        super().__init__()

    def rectified_sigmoid(self, r: torch.Tensor) ->torch.Tensor:
        return ((self.zeta - self.gamma) * torch.sigmoid(r) + self.gamma).clamp(0, 1)

    def forward(self, r: torch.Tensor, iter: int) ->torch.Tensor:
        if iter < self.max_iter * self.warm_ratio:
            round_loss = 0
        else:
            self.beta = self.temp_anneal(iter)
            round_loss = self.alpha * (1 - torch.pow((self.rectified_sigmoid(r) - 0.5).abs() * 2, self.beta)).sum()
        return round_loss


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        None
        x = F.relu(self.conv1(x))
        None
        x = self.pool1(x)
        None
        x = F.relu(self.conv2(x))
        None
        x = self.pool1(x)
        None
        x = x.view(x.size(0), -1)
        None
        x = F.relu(self.fc1(x))
        None
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class MyModel(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        with torch.no_grad():
            self.conv1 = torch.nn.ConvTranspose3d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.ConvTranspose3d(in_channels=32, out_channels=32, groups=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.ConvTranspose3d(in_channels=32, out_channels=8, groups=8, kernel_size=5, stride=1, padding=2)
            self.conv4 = torch.nn.ConvTranspose3d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv1.bias.copy_(torch.rand_like(self.conv1.bias))
            self.conv3.bias.copy_(torch.rand_like(self.conv3.bias))
            self.conv4.bias.copy_(torch.rand_like(self.conv4.bias))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class TestBlock1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x + x


class TestBlock2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.conv(x)


class TestBlock3(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.relu1(self.conv1(x))


class TestBlock4(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.relu1(self.conv1(x)) + self.relu2(self.conv2(x))


class TestBlock5(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU6()
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu2 = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.relu1(self.conv1(x)) + self.relu2(self.conv2(x))


class TestBlock6(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv0(x)
        x = self.relu1(self.conv1(x)) + self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class TestBlock7(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(num_features=16)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv0(x)
        x = self.relu1(self.bn1(self.conv1(x))) + self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class TestBlock8(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv0(x)
        x = x.mean(dim=1)
        return x


class TestBlock9(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv0(x)
        x = x.sum(dim=1)
        return x


class TestBlock10(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gemm = torch.nn.Linear(in_features=1000, out_features=1000)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.gemm(x)
        return x


class TestBlock11(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.gemm1(x)
        x = self.gemm2(x)
        x = self.gemm3(x)
        return x


class TestBlock12(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.relu2 = torch.nn.ReLU()
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.relu1(self.gemm1(x))
        x = self.relu2(self.gemm2(x))
        x = self.relu3(self.gemm3(x))
        return x


class TestBlock13(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn1 = torch.nn.BatchNorm1d(num_features=1000)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn2 = torch.nn.BatchNorm1d(num_features=1000)
        self.relu2 = torch.nn.ReLU()
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn3 = torch.nn.BatchNorm1d(num_features=1000)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.relu1(self.bn1(self.gemm1(x)))
        x = self.relu2(self.bn2(self.gemm2(x)))
        x = self.relu3(self.bn3(self.gemm3(x)))
        return x


class TestBlock14(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn1 = torch.nn.BatchNorm1d(num_features=1000)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn2 = torch.nn.BatchNorm1d(num_features=1000)
        self.relu2 = torch.nn.ReLU()
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn3 = torch.nn.BatchNorm1d(num_features=1000)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x1 = self.relu1(self.bn1(self.gemm1(x)))
        x2 = self.relu2(self.bn2(self.gemm2(x)))
        x3 = self.relu3(self.bn3(self.gemm3(x)))
        return torch.cat([x1, x2, x3], dim=-1)


class TestBlock15(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
        self.conv5 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.conv6 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=6)
        self.conv7 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class TestBlock17(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=4)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=8)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=16)
        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class TestBlock18(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=8, padding=[1, 2])
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=16, padding=[2, 0])
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=[0, 2])

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class TestBlock19(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2, padding=1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class TestBlock20(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2, padding=1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = x[0].unsqueeze(0)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class TestBlock21(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, dilation=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, dilation=1, padding=1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x1 = self.conv2(x[:, :6])
        x2 = self.conv3(x[:, 6:])
        return x1 + x2


class TestBlock22(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=16)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.relu6 = torch.nn.ReLU6()
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=16)
        self.sigmoid = torch.nn.Sigmoid()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.bn1(self.relu(self.conv1(x)))
        y = self.relu6(self.bn2(self.conv2(x)))
        z = self.sigmoid(self.bn3(self.conv3(x)))
        x = torch.cat([x, y, z], dim=1)
        x = self.maxpool(self.avgpool(x))
        return x


class TestBlock23(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm1d(num_features=16)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(num_features=16)
        self.relu6 = torch.nn.ReLU6()
        self.conv3 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(num_features=16)
        self.sigmoid = torch.nn.Sigmoid()
        self.avgpool = torch.nn.AvgPool1d(kernel_size=3)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.bn1(self.relu(self.conv1(x)))
        y = self.relu6(self.bn2(self.conv2(x)))
        z = self.sigmoid(self.bn3(self.conv3(x)))
        x = torch.cat([x, y, z], dim=1)
        x = self.maxpool(self.avgpool(x))
        return x


class TestBlock24(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(num_features=16)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(num_features=16)
        self.relu6 = torch.nn.ReLU6()
        self.conv3 = torch.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(num_features=16)
        self.sigmoid = torch.nn.Sigmoid()
        self.avgpool = torch.nn.AvgPool3d(kernel_size=3)
        self.maxpool = torch.nn.MaxPool3d(kernel_size=3)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.bn1(self.relu(self.conv1(x)))
        y = self.relu6(self.bn2(self.conv2(x)))
        z = self.sigmoid(self.bn3(self.conv3(x)))
        x = torch.cat([x, y, z], dim=1)
        x = self.maxpool(self.avgpool(x))
        return x


class TestBlock25(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(num_features=3)
        self.relu1 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=3)
        self.relu2 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm2d(num_features=3)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.bn1(x)
        x = self.relu1(x)
        y = self.bn2(x)
        z = self.relu2(x)
        u = self.bn3(z)
        v = self.relu3(y)
        return x + y + z + u + v


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaroundRegTerm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (TestBlock1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestBlock15,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock20,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock21,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock23,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
    (TestBlock24,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     True),
    (TestBlock3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock6,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock7,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TestBlock9,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_openppl_public_ppq(_paritybench_base):
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

