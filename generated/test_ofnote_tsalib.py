import sys
_module = sys.modules[__name__]
del sys
allennlp_multi_head_similarity = _module
modeling = _module
modeling_test = _module
openai_transformer = _module
resnet = _module
snippets_pytorch = _module
setup = _module
test = _module
tsalib = _module
backend = _module
tensor_ops = _module
transforms = _module
ts = _module
ts_lite = _module
tsn = _module
utils = _module

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


import torch


from torch.nn.parameter import Parameter


from typing import NamedTuple


from typing import List


import copy


import logging


import math


import re


import numpy as np


from torch.nn import Parameter


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


class LayerNorm(torch.nn.Module):
    """Construct a layernorm module in the OpenAI style (epsilon inside the square root)."""

    def __init__(self, n_state, e=1e-05):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(n_state))
        self.b = torch.nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class TransformerConfig(NamedTuple):
    """
    The transformer has to pass a bunch of params to its submodules,
    this bundles them together to make things easier.
    """
    embedding_dim: int = 768
    num_heads: int = 12
    embedding_dropout_probability: float = 0.1
    attention_dropout_probability: float = 0.1
    residual_dropout_probability: float = 0.1
    activation_function: str = 'gelu'


class DimVar:
    decls = {}
    parse_regexp = '(\\w+)(?:\\((\\w+)\\))?(?::(\\d+))?'

    def __init__(self, decl, exists_ok, cache):
        """
        :decl: declaration string of variable ('Batch(b):20')
        :exists_ok: if declared earlier, nop
        :cache: store in `decls` cache
        """
        assert isinstance(decl, str)
        decl = decl.strip()
        m = re.search(DimVar.parse_regexp, decl)
        name, sname, val = m.groups()
        self._name = name
        self._sname = sname if sname is not None else name
        self._val = int(val) if val is not None else nan
        self._e = Symbol(self._sname)
        if self._e in DimVar.decls:
            prevd = DimVar.decls[self._e]
            if not exists_ok:
                raise ValueError(
                    f'DimVar {self._sname} already declared as {prevd._name}({self._e}). Use exists_ok=True to skip check.'
                    )
        elif cache:
            DimVar.decls[self._e] = self

    @property
    def exp(self):
        return self._e

    @property
    def size(self):
        return self._val

    @property
    def shortname(self):
        return self._sname

    @property
    def name(self):
        ret = f'{self._name}'
        if self._name != self._sname:
            ret += f'({self._sname})'
        return ret

    def update_len(self, new_val):
        assert isinstance(new_val, int)
        self._val = new_val

    @staticmethod
    def check_decl(sname):
        return Symbol(sname) in DimVar.decls

    @staticmethod
    def lookup(sname):
        sn = Symbol(sname)
        if len(DimVar.decls) == 0:
            assert False
        assert sn in DimVar.decls, f'DimVar short name {sn} not declared.'
        return DimVar.decls[sn]

    @staticmethod
    def lookup2(name):
        for k, decl in DimVar.decls.items():
            if decl._name == name:
                return decl
        assert False, f'DimVar full name {name} not declared.'

    @staticmethod
    def eval(e):
        sub_map = [(e, dv._val) for e, dv in DimVar.decls.items()]
        ret = e.subs(sub_map)
        return ret

    @staticmethod
    def eval_name(e):
        sub_map = [(e, dv.shortname) for e, dv in DimVar.decls.items()]
        return str(e.subs(sub_map))


def arith_op(op, s1, s2):
    assert isinstance(s1, DimExpr)
    s2 = DimExpr(s2)
    s1e = s1.exp
    s2e = s2.exp
    if op == 'add':
        se = s1e + s2e
    elif op == 'mul':
        se = s1e * s2e
    elif op == 'truediv':
        se = s1e / s2e
    elif op == 'floordiv':
        se = s1e // s2e
    else:
        raise NotImplementedError(f'{op}')
    return DimExpr(se)


class DimExpr:
    """
    Encapsulates the expression for a particular axis/dimension
    """

    def __init__(self, t, is_dvar=False):
        self._e = None
        self.dim_var = None
        self._val = None
        if isinstance(t, int):
            self._e = Integer(t)
            self._val = t
        elif isinstance(t, DimVar):
            self._e, self._val, self.dim_var = t.exp, t.size, t
        elif isinstance(t, DimExpr):
            self._e, self._val, self.dim_var = t._e, t._val, t.dim_var
        else:
            self._e = t
            self._val = DimVar.eval(t)

    @property
    def exp(self):
        return self._e

    @property
    def len(self):
        return self._val if self._val != nan else None

    def update_len(self, new_len):
        if self.dim_var is None:
            raise ValueError(
                'Cannot update length of arbitrary dim expression.')
        else:
            self.dim_var.update_len(new_len)
            self._val = new_len

    def __int__(self):
        if self._val != nan:
            return int(self._val)
        else:
            raise ValueError(
                f'Cannot cast to integer: Default value of {self._e} not provided'
                )

    def __index__(self):
        return self.__int__()

    def __add__(self, n):
        return arith_op('add', self, n)

    def __radd__(self, n):
        return self.__add__(n)

    def __mul__(self, n):
        return arith_op('mul', self, n)

    def __rmul__(self, n):
        return self.__mul__(n)

    def __floordiv__(self, n):
        return arith_op('floordiv', self, n)

    def __rfloordiv__(self, n):
        return self.__floordiv__(n)

    def __truediv__(self, n):
        return arith_op('truediv', self, n)

    def __rtruediv__(self, n):
        return self.__truediv__(n)

    def __eq__(self, d):
        if isinstance(d, int):
            if self._val == nan:
                return True
            else:
                return self._val == d
        elif isinstance(d, DimExpr):
            res = self._e == d._e
            return res
        else:
            return False

    def __hash__(self):
        return hash(self._e)

    def __repr__(self):
        s = DimVar.eval_name(self._e)
        if self._val != nan:
            s += f':{self._val}'
        return s


def dim_var(name, exists_ok=False, cache=True):
    """
    Declare a single dimension variable
    """
    d = DimVar(name, exists_ok=exists_ok, cache=cache)
    return DimExpr(d)


def dim_vars(names, exists_ok=False, cache=True):
    """
    Declare multiple dimension variables in one go
    """
    names = names.split()
    tss = [dim_var(name, exists_ok=exists_ok, cache=cache) for name in names]
    if len(names) == 1:
        return tss[0]
    else:
        return tss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ofnote_tsalib(_paritybench_base):
    pass
    def test_000(self):
        self._check(LayerNorm(*[], **{'n_state': 4}), [torch.rand([4, 4, 4, 4])], {})

