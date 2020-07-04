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


class TupleSeq:

    def __init__(self, s):
        self.s = s

    def item(self):
        return self.s


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


def dummy_dvar(pos):
    """
    Declare a dummy dimension variable at a particular dim position. Do not cache.
    """
    assert pos >= 0
    name = f'_dm_{pos}'
    d = dim_var(name, exists_ok=True, cache=False)
    return d


def _sexpr_to_ts(e, dummy_idx=0, strict=False, num_to_sym=False):
    """
    A single string expression (sexpr) to Tensor Shape expressions (ts)
    Converts shorthand dummy/empty placeholders to dummy TSs
    """
    if isinstance(e, DimExpr):
        t = e
    else:
        assert isinstance(e, str)
        if e.isdigit() and num_to_sym:
            e = '_'
        if e == '' or e == '_':
            t = dummy_dvar(dummy_idx)
            dummy_idx += 1
        elif e == '^':
            t = DimExpr(Symbol(e))
        else:
            t = DimExpr(sympify(e))
    return t, dummy_idx


def _sexprs_to_ts(exprs, strict=False, num_to_sym=False):
    """
    String expressions (sexprs) to Tensor Shape expressions (ts)
    Converts shorthand dummy/empty placeholders to dummy TSs
    Returns a tuple of TSs
    """
    dummy_idx = 0
    res = []
    for e in exprs:
        t, dummy_idx = _sexpr_to_ts(e, dummy_idx=dummy_idx, strict=strict,
            num_to_sym=num_to_sym)
        res.append(t)
    return tuple(res)


seq_re = '\\((.+)\\)\\*'


def tsn_to_str_list(ss: str):
    ss = re.sub('\\s+', '', ss)
    is_seq = False
    m = re.search(seq_re, ss)
    if m is not None:
        ss = m.groups()[0]
        is_seq = True
    if ',' in ss:
        exprs = ss.strip().split(',')
    else:
        exprs = list(ss)
    return exprs, is_seq


def tsn_to_tuple(ss, num_to_sym=False):
    """
    :ss is shape string, e.g., 'btd' or 'b,t,d*2' or '(btd)*'
    : num_to_sym : converts numeric values in tsn to anonymous symbols ('_')
    :returns the shape representation in tuple/TupleSeq form
    """
    if isinstance(ss, (list, tuple)):
        for s in ss:
            assert isinstance(s, (DimExpr, int))
        return tuple(ss)
    elif isinstance(ss, TupleSeq):
        return ss
    elif isinstance(ss, str):
        exprs, is_seq = tsn_to_str_list(ss)
        exprs = _sexprs_to_ts(exprs, num_to_sym=num_to_sym)
        for e in exprs:
            assert isinstance(e, DimExpr)
        exprs = tuple(exprs)
        if is_seq:
            exprs = TupleSeq(exprs)
        return exprs
    else:
        raise ValueError('Unknown type of ss')


def _join_transform(tlist, src, to):
    """
    src: Union[str, TupleSeq]
    to: Union[str, tuple]
    src: (b,c,d)* , to: '^, b, c, d'    OR
    src: (b,c,d)* , to: 'b, 3*c, d'

    returns the dims shorthand for joining (backend independent)

    """
    lhs = tsn_to_tuple(src)
    rhs = tsn_to_tuple(to)
    assert isinstance(lhs, TupleSeq)
    assert isinstance(rhs, tuple)
    lhs = lhs.item()
    int1 = Integer(1)
    sub_map = [(d.exp, int1) for d in lhs]
    dims = tuple([t.exp.subs(sub_map) for t in rhs])
    if len(rhs) == len(lhs):
        dims = ','.join(map(lambda x: '' if x == int1 else '*', dims))
    elif len(rhs) == len(lhs) + 1:
        dims = ','.join(map(lambda x: '' if x == int1 else '^', dims))
    else:
        raise ValueError(f'Unable to join from {src} to {to}')
    return dims


def _permute_transform(src, to):
    """
    Permute Transform
    src, to: Union[str, Tuple]

    :src is the current dimension arrangement, list of named dim variables
    :to is the target dimension arragement, list of named dim variables
    :returns the index tuple for the permutation (backend independent)
    """
    lhs = tsn_to_tuple(src, num_to_sym=True)
    rhs = tsn_to_tuple(to, num_to_sym=True)
    assert isinstance(lhs, tuple)
    assert isinstance(rhs, tuple)
    assert len(lhs) == len(rhs
        ), 'Source and Target shapes for permutation are not same'
    sub_map = [(d.exp, f'{i}') for i, d in enumerate(lhs)]
    perm_indices = tuple([t.exp.subs(sub_map) for t in rhs])
    perm_indices = tuple([int(str(s)) for s in perm_indices])
    return perm_indices


def check_int_tuple(s):
    for d in s:
        try:
            d = int(d)
        except:
            raise ValueError(f'Unable to resolve expression {d}')


def resolve_to_int_tuple(s):
    """
    resolve non-int elements by casting to int or looking up their DimVar values
    """
    res = []
    for d in s:
        try:
            d = int(d)
            res.append(d)
        except:
            if isinstance(d, DimExpr):
                e = d.exp
            else:
                e = d
            r = DimVar.eval(e)
            try:
                r = int(r)
                res.append(r)
            except:
                raise ValueError(f'Unable to resolve {d}')
    return tuple(res)


def _view_transform(src, to, in_shape, checkin=False):
    """
    View Transform
    src, to: Union[str, Tuple]
    :src is the current view of the tensor
    :to is the target view, may contain expressions over named dim variables
    :in_shape is the shape (list/tuple) of the source tensor 
    :returns the new size of the tensor after view transformation (backend independent)
    """
    if checkin:
        check_int_tuple(in_shape)
    src = tsn_to_tuple(src)
    if len(src) != len(in_shape):
        print(f'{src}, {in_shape}')
        raise ValueError("Source DimExpr does not match input tensor's shape")
    to = tsn_to_tuple(to)
    assert isinstance(in_shape, (list, tuple))
    assert isinstance(src, tuple)
    assert isinstance(to, tuple)
    sub_map = [(d.exp, in_shape[i]) for i, d in enumerate(src)]
    out_shape = tuple([(t.exp.subs(sub_map) if isinstance(t, DimExpr) else
        int(t)) for t in to])
    out_shape = resolve_to_int_tuple(out_shape)
    return out_shape


def align_transform(src, to, tile=False):
    """
    src: tsn 'd,d'
    to: tsn '6, d, t, d'
    tile: duplicate src values along new dimensions
    return: '^,,^,' [expansion shorthand for src]
        if tile is True: also return (6,1,int(T),1)
    """
    lhs = tsn_to_tuple(src)
    rhs = tsn_to_tuple(to)
    assert isinstance(lhs, tuple)
    assert isinstance(rhs, tuple)
    lhs_pos, rhs_pos = 0, 0
    expand_dims = []
    expand_ratio = []
    for rhs_pos, S in enumerate(rhs):
        if lhs_pos < len(lhs) and lhs[lhs_pos] == S:
            expand_dims.append('')
            if tile:
                expand_ratio.append(1)
            lhs_pos += 1
        else:
            expand_dims.append('^')
            if tile:
                try:
                    expand_ratio.append(int(S))
                except:
                    expand_ratio.append(S)
    if lhs_pos != len(lhs):
        print(f'Unable to align {src} to {to}: {src} not a subsequence of {to}'
            )
        raise ValueError
    return ','.join(expand_dims), expand_ratio


def expand_dims_transform(x, tfm):
    """
    x: (backend) tensor, e.g., shape 'd,d'
    tfm: expand tsn: '^,,^,'
    returns tensor of shape '1,d,1,d'
    """
    colon = slice(None)
    expand_tup = tuple(None if c == '^' else colon for c in tfm.split(','))
    res = x[expand_tup]
    return res


def alignto(x, ys, tile=False):
    """
    Align tensor x's shape to y's shape.
    Assume x's shape is a subsequence of y's shape
    Assume tensors x and y support numpy's "None, :"" indexing notation
    x: (tensor_var, x_shape)
    ys: y_shape (tsn of target tensor)
    """
    if tile:
        raise NotImplementedError('tiling to be implemented.')
    assert isinstance(x, tuple), 'First argument is of form (tensor_var, tsn)'
    assert isinstance(ys, (str, tuple))
    xt, xs = x
    expand_tfm, expand_ratio = align_transform(xs, ys, tile)
    exp_1 = expand_dims_transform(xt, expand_tfm)
    return exp_1


class ABackend:
    name = None

    def shape(self, x):
        raise NotImplementedError

    def contiguous(self, x):
        raise NotImplementedError

    def view(self, x, shape):
        raise NotImplementedError

    def transpose(self, x, dims):
        raise NotImplementedError

    def expand(self, x, mul_shape):
        raise NotImplementedError

    def stack(self, x, mul_shape):
        raise NotImplementedError

    def concat(self, x, mul_shape):
        raise NotImplementedError

    def einsum(self, eqn, args):
        raise NotImplementedError


class Numpy(ABackend):
    name = 'numpy'

    def __init__(self):
        import numpy
        self.np = numpy

    def shape(self, x):
        return x.shape

    def contiguous(self, x):
        self.np.ascontiguousarray(x)

    def view(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, dims):
        return x.transpose(dims)

    def expand(self, x, mul_shape):
        return x.tile(mul_shape)

    def stack(self, xlist, axis):
        return self.np.stack(xlist, axis=axis)

    def concat(self, xlist, axis):
        return self.np.concatenate(xlist, axis=axis)

    def einsum(self, eqn, args):
        return self.np.einsum(eqn, *args)


class PyTorch(ABackend):
    name = 'pytorch'

    def __init__(self):
        import torch
        self.torch = torch

    def shape(self, x):
        return x.size()

    def contiguous(self, x):
        return x.contiguous()

    def view(self, x, shape):
        return x.view(shape)

    def transpose(self, x, dims):
        return x.permute(dims)

    def expand(self, x, mul_shape):
        return x.expand(mul_shape)

    def stack(self, xlist, axis):
        return self.torch.stack(xlist, dim=axis)

    def concat(self, xlist, axis):
        return self.torch.cat(xlist, dim=axis)

    def einsum(self, eqn, args):
        return self.torch.einsum(eqn, args)


def get_tf_shape(tf, tensor):
    """Returns a list of the shape of tensor, preferring static dimensions.
  (inspired by get_shape_list in BERT code)

  Args:
    tensor: A tf.Tensor object to find the shape of.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    shape = tuple(tensor.shape.as_list())
    non_static_indexes = []
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def int_shape(*s):
    if len(s) == 1:
        assert isinstance(s, (tuple, list))
        s = s[0]
    else:
        s = tuple(s)
    return tuple([int(d) for d in s])


class TF(ABackend):
    name = 'tensorflow'

    def __init__(self):
        import tensorflow
        self.tf = tensorflow

    def shape(self, x):
        if self.tf.executing_eagerly():
            return int_shape(x.shape)
        else:
            return get_tf_shape(self.tf, x)

    def contiguous(self, x):
        return x

    def view(self, x, shape):
        return self.tf.reshape(x, shape)

    def transpose(self, x, dims):
        return self.tf.transpose(x, dims)

    def expand(self, x, mul_shape):
        return self.tf.tile(x, mul_shape)

    def stack(self, xlist, axis):
        return self.tf.stack(xlist, axis=axis)

    def concat(self, xlist, axis):
        return self.tf.concat(xlist, axis=axis)

    def einsum(self, eqn, args):
        return self.tf.einsum(eqn, args)


becache = {}


def from_cache(C):
    s = str(C)
    if s not in becache:
        becache[s] = C()
    return becache[s]


def get_backend_by_name(b):
    if isinstance(b, (Numpy, TF, PyTorch)):
        return b
    assert isinstance(b, str)
    bemap = {'numpy': Numpy, 'np': Numpy, 'torch': PyTorch, 'pytorch':
        PyTorch, 'tensorflow': TF, 'tf': TF}
    if b in bemap:
        return from_cache(bemap[b])
    else:
        raise NotImplementedError(
            f'Unsupported backend {b}. Contributions welcome.')


def get_str_type(x):
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            return ''
        x = x[0]
    t = str(type(x))
    return t


def get_tensor_lib(x):
    t = get_str_type(x)
    if 'numpy.' in t:
        ret = Numpy
    elif 'torch.' in t:
        ret = PyTorch
    elif 'tensorflow.' in t:
        ret = TF
    else:
        ret = None
    return ret


def get_backend_for_tensor(x):
    """
    get backend for tensor x
    """
    tlib = get_tensor_lib(x)
    if tlib is None:
        raise NotImplementedError(
            f'Unsupported tensor type {type(x)}. Contributions welcome.')
    ret = from_cache(tlib)
    return ret


def get_backend(backend, x):
    if backend is not None:
        be = get_backend_by_name(backend)
    else:
        be = get_backend_for_tensor(x)
    return be


def join(tlist, dims, backend=None):
    """
    tlist: List[tensor], list of tensors
    dims: str = '..,^,...' or '..,*,...' 
    """
    assert isinstance(tlist, list), 'Can only group a list of tensors.'
    assert len(tlist) > 1, 'Can only group more than one tensors.'
    be = get_backend(backend, tlist[0])
    if len(dims) > 1:
        assert ',' in dims, 'Separate dimensions by "," here.'
    out_shape = dims.strip().split(',')
    if '^' in out_shape:
        pos = out_shape.index('^')
        return be.stack(tlist, axis=pos)
    else:
        if '*' not in out_shape:
            assert pos != '-1', 'Group specification does not contain "^" or "*".'
        pos = out_shape.index('*')
        return be.concat(tlist, axis=pos)


def norm_tfm_names(tfm_names):
    """
    tfm_names: 'abc' or ['a', 'bc'] 
    returns: ['a', 'b', 'c']
    """
    res = []
    if isinstance(tfm_names, str):
        res = list(tfm_names)
    elif isinstance(tfm_names, (list, tuple)):
        for n in tfm_names:
            assert isinstance(n, str)
            res.extend(list(n))
    return res


def tsnseq2shape_pairs(tfms):
    res = [tsn_to_tuple(t.strip()) for t in tfms.split('->')]
    assert len(res) >= 2, 'At least 2 tfms required: {res}'
    shape_pairs = list(zip(res[:-1], res[1:]))
    return shape_pairs


def norm_tfms_to_shape_pairs(tfms):
    """
    tfms  'x -> y -> z -> u' or ['x -> y', 'y -> z', 'z -> u'] or ['x -> y -> z', 'z -> u']
    'x', 'y', 'z' are tsns
    returns: [(X, Y), (Y, Z), (Z, U)], where X is the tuple rep for tsn 'x', ...
    """
    res = []
    if isinstance(tfms, str):
        shape_pairs = tsnseq2shape_pairs(tfms)
        res.extend(shape_pairs)
    elif isinstance(tfms, (list, tuple)):
        for pos, tfm in enumerate(tfms):
            assert isinstance(tfm, str)
            shape_pairs = tsnseq2shape_pairs(tfm)
            res.extend(shape_pairs)
    else:
        raise ValueError(f'unknown format: {tfms}')
    return res


def tfm_seq_decompose(tfms, tfm_names):
    """
    Decompose a multi-step transform into basic (view, permute, expand) transforms
    tfms  'btd -> b,t,2,d//2 -> b,2,t,d//2'
    tfm_names 'vp' , i.e., view, then permute transform
    """
    tfm_symbols = norm_tfm_names(tfm_names)
    tfm_symbols_no_c = [n for n in tfm_symbols if n != 'c']
    shape_pairs = norm_tfms_to_shape_pairs(tfms)
    assert len(tfm_symbols_no_c) == len(shape_pairs
        ), f'Num of transform steps {len(shape_pairs)} and names {len(tfm_symbols_no_c)} do not match'
    tfm_list = []
    curr_pos = 0
    for sym in tfm_symbols:
        if sym == 'c':
            tfm_list.append((sym, None, None))
        else:
            l, r = shape_pairs[curr_pos]
            tfm_list.append((sym, l, r))
            curr_pos += 1
    return tfm_list


def warp(x, tfms, tfm_names, backend=None, debug=False):
    """
    Perform a multi-step transform on the tensor x
    x: tensor
    tfms:  'btd -> b,t,2,d//2 -> b,2,t,d//2 -> b,2,t,^n,d//2'
    tfm_names: 'vp' [first (v)iew, then (p)ermute transform]
    backend:  either a string('numpy', 'tf', 'torch') or the corresponding backend.<class>
    debug: prints per-step debugging information
    """
    be = get_backend(backend, x)
    tfm_list = tfm_seq_decompose(tfms, tfm_names)
    ret = x
    for sym, l, r in tfm_list:
        if debug:
            print(f'*** processing transform.. {sym}\n {l} -> {r}')
        if sym == 'v' or sym == 'r':
            new_shape = _view_transform(l, r, be.shape(ret))
            ret = be.view(ret, new_shape)
        elif sym == 'p' or sym == 't':
            perm_indices = _permute_transform(l, r)
            ret = be.transpose(ret, perm_indices)
        elif sym == 'a':
            ret = alignto((ret, l), r)
        elif sym == 'c':
            ret = be.contiguous(ret)
        elif sym == 'j':
            dims = _join_transform(ret, l, r)
            ret = join(ret, dims, backend=be)
        else:
            assert False, f'Invalid transform symbol {sym}'
        if debug:
            print(f'after transform, shape is: {be.shape(ret)}')
    return ret


def gelu(x: torch.Tensor) ->torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
        torch.pow(x, 3))))


def swish(x: torch.Tensor) ->torch.Tensor:
    return x * torch.sigmoid(x)


_ACTIVATION_FUNCTIONS = {'relu': torch.nn.ReLU, 'swish': swish, 'gelu': gelu}


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ofnote_tsalib(_paritybench_base):
    pass
    def test_000(self):
        self._check(LayerNorm(*[], **{'n_state': 4}), [torch.rand([4, 4, 4, 4])], {})

