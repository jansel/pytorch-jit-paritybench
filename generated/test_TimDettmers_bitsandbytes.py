import sys
_module = sys.modules[__name__]
del sys
bitsandbytes = _module
autograd = _module
_functions = _module
cextension = _module
cuda_setup = _module
env_vars = _module
main = _module
paths = _module
debug_cli = _module
functional = _module
nn = _module
modules = _module
optim = _module
adagrad = _module
adam = _module
adamw = _module
lamb = _module
lars = _module
optimizer = _module
rmsprop = _module
sgd = _module
utils = _module
setup = _module
test_autograd = _module
test_cuda_setup_evaluator = _module
test_functional = _module
test_modules = _module
test_optim = _module

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


from warnings import warn


import torch


import warnings


from functools import reduce


import random


import itertools


import math


from typing import Tuple


from torch import Tensor


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterator


from typing import Mapping


from typing import Optional


from typing import Set


from typing import TypeVar


from typing import Union


from typing import overload


import torch.nn.functional as F


from torch import device


from torch import dtype


from torch import nn


from torch.nn.parameter import Parameter


import torch.distributed as dist


from torch.optim import Optimizer


from collections import abc as container_abcs


from collections import defaultdict


from copy import deepcopy


from itertools import chain


from itertools import product


from itertools import permutations


import time


import numpy as np


from scipy.stats import norm


import uuid


class GlobalOptimManager(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group['params']):
                if id(p) in self.pid2config:
                    self.index2config[group_index, p_index] = self.pid2config[id(p)]

    def override_config(self, parameters, key=None, value=None, key_value_dict=None):
        """
        Overrides initial optimizer config for specific parameters.

        The key-values of the optimizer config for the input parameters are overidden
        This can be both, optimizer parameters like "betas", or "lr" or it can be
        8-bit specific paramters like "optim_bits", "percentile_clipping".

        Parameters
        ----------
        parameters : torch.Tensor or list(torch.Tensors)
            The input parameters.
        key : str
            The hyperparamter to override.
        value : object
            The value for the hyperparamters.
        key_value_dict : dict
            A dictionary with multiple key-values to override.
        """
        self.uses_config_override = True
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}
        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:
                    self.pid2config[id(p)].update(key_value_dict)
                else:
                    self.pid2config[id(p)] = key_value_dict

    def register_module_override(self, module, param_name, config):
        self.module_weight_config_triple.append((module, param_name, config))


class StableEmbedding(torch.nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, _weight: Optional[Tensor]=None) ->None:
        super(StableEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        self.norm = torch.nn.LayerNorm(embedding_dim)
        GlobalOptimManager.get_instance().register_module_override(self, 'weight', {'optim_bits': 32})

    def reset_parameters(self) ->None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()
    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) ->None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) ->Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return self.norm(emb)


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, _weight: Optional[Tensor]=None) ->None:
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        GlobalOptimManager.get_instance().register_module_override(self, 'weight', {'optim_bits': 32})

    def reset_parameters(self) ->None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()
    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) ->None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) ->Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return emb


T = TypeVar('T', bound='torch.nn.Module')


class Int8Params(torch.nn.Parameter):

    def __new__(cls, data=None, requires_grad=True, has_fp16_weights=False, CB=None, SCB=None):
        cls.has_fp16_weights = has_fp16_weights
        cls.CB = None
        cls.SCB = None
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def cuda(self, device):
        if self.has_fp16_weights:
            return super()
        else:
            B = self.data.contiguous().half()
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            self.data = CB
            setattr(self, 'CB', CB)
            setattr(self, 'SCB', SCB)
        return self

    @overload
    def to(self: T, device: Optional[Union[int, device]]=..., dtype: Optional[Union[dtype, str]]=..., non_blocking: bool=...) ->T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool=...) ->T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool=...) ->T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == 'cuda' and self.data.device.type == 'cpu':
            return self
        else:
            new_param = Int8Params(super(), requires_grad=self.requires_grad, has_fp16_weights=self.has_fp16_weights)
            new_param.CB = self.CB
            new_param.SCB = self.SCB
            return new_param


class Linear8bitLt(nn.Linear):

    def __init__(self, input_features, output_features, bias=True, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0, index=None):
        super(Linear8bitLt, self).__init__(input_features, output_features, bias)
        self.state = bnb.MatmulLtState()
        self.index = index
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        if not self.state.has_fp16_weights:
            if not self.state.memory_efficient_backward and self.state.CB is not None:
                del self.state.CB
                self.weight.data = self.state.CxB
            elif self.state.memory_efficient_backward and self.state.CxB is not None:
                del self.state.CxB
        return out


class FFN(torch.nn.Module):

    def __init__(self, input_features, hidden_size, bias=True):
        super(FFN, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, hidden_size, bias=bias)
        self.fc2 = torch.nn.Linear(hidden_size, input_features, bias=bias)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP8bit(torch.nn.Module):

    def __init__(self, dim1, dim2, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0):
        super(MLP8bit, self).__init__()
        self.fc1 = bnb.nn.Linear8bitLt(dim1, dim2, has_fp16_weights=has_fp16_weights, memory_efficient_backward=memory_efficient_backward, threshold=threshold)
        self.fc2 = bnb.nn.Linear8bitLt(dim2, dim1, has_fp16_weights=has_fp16_weights, memory_efficient_backward=memory_efficient_backward, threshold=threshold)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LinearFunction(torch.autograd.Function):

    @staticmethod
    def get_8bit_linear_trimmed(x, stochastic=False, trim_value=3.0):
        round_func = LinearFunction.round_stoachastic if stochastic else torch.round
        norm = math.sqrt(math.pi) / math.sqrt(2.0)
        std = torch.std(x)
        max1 = std * trim_value
        x = x / max1 * 127
        x = round_func(x)
        x[x > 127] = 127
        x[x < -127] = -127
        x = x / 127 * max1
        return x

    def quant(x, quant_type, dim=1):
        if quant_type == 'linear':
            max1 = torch.abs(x).max().float()
            xq = torch.round(x / max1 * 127)
            return xq, max1
        elif quant_type == 'vector':
            max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
            xq = torch.round(x / max1 * 127)
            return xq, max1
        elif quant_type == 'min-max':
            maxA = torch.amax(x, dim=dim, keepdim=True).float()
            minA = torch.amin(x, dim=dim, keepdim=True).float()
            scale = (maxA - minA) / 2.0
            xq = torch.round(127 * (x - minA - scale) / scale)
            return xq, (minA.float(), scale.float())
        else:
            return None

    def dequant(xq, S1, S2, dtype, quant_type):
        if quant_type == 'linear':
            norm = S1 * S2 / (127 * 127)
            return xq.float() * norm
        elif quant_type == 'vector':
            x = xq.float()
            if len(xq.shape) == 2 and len(S1.shape) == 3:
                S1 = S1.squeeze(0)
            if len(xq.shape) == 2 and len(S2.shape) == 3:
                S2 = S2.squeeze(0)
            if len(S1.shape) == 2:
                x *= S1.t() / 127
            else:
                x *= S1 / 127
            x *= S2 / 127
            return x
        else:
            return None

    def dequant_min_max(xq, A, B, SA, SB, dtype):
        offset = B.float().t().sum(0) * (SA[0] + SA[1])
        x = xq.float()
        if len(xq.shape) == 2 and len(SB.shape) == 3:
            SB = SB.squeeze(0)
        if len(xq.shape) == 2 and len(SA.shape) == 3:
            SA = SA.squeeze(0)
        if len(SB.shape) == 2:
            x *= SB.t() / 127
        else:
            x *= SB / 127
        x *= SA[1] / 127
        x += offset
        return x

    def get_8bit_linear(x, stochastic=False):
        round_func = LinearFunction.round_stoachastic if stochastic else torch.round
        max1 = torch.abs(x).max()
        x = x / max1 * 127
        x = round_func(x) / 127 * max1
        return x

    @staticmethod
    def get_8bit_vector_wise(x, dim, stochastic=False):
        round_func = LinearFunction.round_stoachastic if stochastic else torch.round
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        max1[max1 == 0] = 1.0
        x = x * 127 / max1
        x = round_func(x) / 127 * max1
        return x

    @staticmethod
    def round_stoachastic(x):
        sign = torch.sign(x)
        absx = torch.abs(x)
        decimal = absx - torch.floor(absx)
        rdm = torch.rand_like(decimal)
        return sign * (torch.floor(absx) + (rdm < decimal))

    @staticmethod
    def fake_8bit_storage(w, exponent_bits):
        code = bnb.functional.create_dynamic_map(n=exponent_bits)
        absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
        out = bnb.functional.dequantize_blockwise(absmax, C, code)
        out = out.half()
        w.copy_(out)
        return out

    @staticmethod
    def fake_8bit_storage_quantile(w, args):
        code = bnb.functional.estimate_quantiles(w.data, offset=args.offset)
        code /= torch.max(torch.abs(code))
        absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
        out = bnb.functional.dequantize_blockwise(absmax, C, code)
        out = out.half()
        w.copy_(out)
        return out

    @staticmethod
    def fake_8bit_storage_stoachstic(w):
        rand = torch.rand(1024, device=w.device)
        absmax, C = bnb.functional.quantize_blockwise(w.data, rand=rand)
        out = bnb.functional.dequantize_blockwise(absmax, C)
        out = out.half()
        w.copy_(out)
        return out

    @staticmethod
    def fake_8bit_storage_with_max(w, topk=8):
        blocked_w = einops.rearrange(w.flatten(), '(h b) -> h b', b=256)
        max_val, idx = torch.sort(torch.abs(blocked_w), dim=1, descending=True)
        idx = idx[:, :topk]
        max_val = max_val[:, :topk]
        mask = torch.zeros_like(blocked_w)
        mask.scatter_(dim=1, index=idx, src=torch.ones_like(max_val))
        mask = mask.bool()
        values = blocked_w[mask]
        blocked_w[mask] = 0
        code = bnb.functional.create_dynamic_map()
        code = code
        absmax, C = bnb.functional.quantize_blockwise(blocked_w.data)
        bnb.functional.dequantize_blockwise(absmax, C, out=blocked_w)
        blocked_w[mask] = values
        unblocked_w = blocked_w.flatten().view(w.shape)
        w.copy_(unblocked_w)
        return unblocked_w

    @staticmethod
    def forward(ctx, x, weight, bias=None, args=None):
        if args.use_8bit_training != 'off':
            weight8, S1 = LinearFunction.quant(weight, args.quant_type, dim=1)
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=2)
            outputq = bnb.functional.igemm(x8, weight8.t())
            output = LinearFunction.dequant(outputq, S1, S2, x.dtype, args.quant_type)
        else:
            output = torch.einsum('bsi,oi->bso', x, weight)
        ctx.save_for_backward(x, weight, bias)
        ctx.args = args
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        args = ctx.args
        stochastic = False
        grad_input = grad_weight = grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        if args.use_8bit_training == 'forward+wgrad':
            grad_output8, S1 = LinearFunction.quant(grad_output, args.quant_type, dim=[0, 1])
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=[0, 1])
            grad_weight8 = bnb.functional.igemm(grad_output8, x8)
            grad_weight = LinearFunction.dequant(grad_weight8, S1, S2, grad_output.dtype, args.quant_type)
            grad_input = grad_output.matmul(weight)
        elif args.use_8bit_training == 'full':
            grad_output8, S1 = LinearFunction.quant(grad_output, args.quant_type, dim=[0, 1])
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=[0, 1])
            grad_weight8 = torch.zeros_like(weight, dtype=torch.int32)
            bnb.functional.igemm(grad_output8, x8, out=grad_weight8)
            grad_weight = LinearFunction.dequant(grad_weight8, S1, S2, grad_output.dtype, args.quant_type)
            grad_output8, S1 = LinearFunction.quant(grad_output, args.quant_type, dim=2)
            weight8, S3 = LinearFunction.quant(weight, args.quant_type, dim=0)
            grad_input8 = bnb.functional.igemm(grad_output8, weight8)
            grad_input = LinearFunction.dequant(grad_input8, S1, S3, grad_output.dtype, args.quant_type)
        else:
            grad_input = grad_output.matmul(weight)
            grad_weight = torch.einsum('bsi,bso->oi', x, grad_output)
        return grad_input, grad_weight, grad_bias, None


class Linear8bit(nn.Module):

    def __init__(self, input_features, output_features, bias=True, args=None):
        super(Linear8bit, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.args = args
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        self.args.training = self.training
        return LinearFunction.apply(x, self.weight, self.bias, self.args)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FFN,
     lambda: ([], {'input_features': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_TimDettmers_bitsandbytes(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

