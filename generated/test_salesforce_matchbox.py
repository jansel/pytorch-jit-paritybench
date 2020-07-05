import sys
_module = sys.modules[__name__]
del sys
test_transformer = _module
transformer = _module
matchbox = _module
compat = _module
data = _module
functional = _module
constructors = _module
elementwise = _module
indexing = _module
nnet = _module
reduction = _module
special = _module
tensor_math = _module
tensor_shape = _module
macro = _module
recompile = _module
code_to_ast = _module
compile_function = _module
test_utils = _module
setup = _module
test_control_flow = _module
test_functions = _module
test_rnns = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


from torchtext import data


from torchtext import datasets


import math


import random


import time


import torch.nn.modules.sparse


import torch.nn.modules.linear


import torch.nn.modules.dropout


from torch.nn import functional as F


from torch.autograd import Variable


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, drop_ratio):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.norm = LayerNorm(d_model)

    def forward(self, *x):
        return x[0] + self.dropout(self.norm(self.layer(*x)))


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = query @ key.transpose(1, 2)
        if self.causal and query.dim() == 3:
            dot_products = dot_products.causal_mask(in_dim=2, out_dim=1)
        return self.dropout((dot_products / self.scale).softmax()) @ value


class MultiHead(nn.Module):

    def __init__(self, attention, d_key, d_value, n_heads):
        super().__init__()
        self.attention = attention
        self.wq = nn.Linear(d_key, d_key)
        self.wk = nn.Linear(d_key, d_key)
        self.wv = nn.Linear(d_value, d_value)
        self.wo = nn.Linear(d_value, d_key)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.split_dim(-1, self.n_heads).join_dims(0, -1) for x in (query, key, value))
        outputs = self.attention(query, key, value)
        outputs = outputs.split_dim(0, self.n_heads).join_dims(-1, 1)
        return self.wo(outputs)


class EncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.selfattn = ResidualBlock(MultiHead(Attention(args.d_model, args.drop_ratio), args.d_model, args.d_model, args.n_heads), args.d_model, args.drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(args.d_model, args.d_hidden, args.drop_ratio), args.d_model, args.drop_ratio)

    def forward(self, x):
        x = self.selfattn(x, x, x)
        return self.feedforward(x)


class DecoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.selfattn = ResidualBlock(MultiHead(Attention(args.d_model, args.drop_ratio, True), args.d_model, args.d_model, args.n_heads), args.d_model, args.drop_ratio)
        self.attention = ResidualBlock(MultiHead(Attention(args.d_model, args.drop_ratio), args.d_model, args.d_model, args.n_heads), args.d_model, args.drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(args.d_model, args.d_hidden, args.drop_ratio), args.d_model, args.drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)
        x = self.attention(x, encoding, encoding)
        return self.feedforward(x)


def positional_encodings_like(x, t=None):
    T, D = x.maxsize(-2), x.maxsize(-1)
    positions = torch.arange(0, T, out=x.new(T)) if t is None else t
    channels = torch.arange(0, D, 2, out=x.new(D)) / D
    channels = 1 / 10000 ** channels
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)
    encodings = torch.stack((encodings.sin(), encodings.cos()), -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)
    if encodings.dim() == 2:
        encodings = encodings.unsqueeze(0).expand(x.maxsize())
    return encodings


class Encoder(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.out = field.out
        self.layers = nn.ModuleList([EncoderLayer(args) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        self.field = field
        self.d_model = args.d_model

    def forward(self, x):
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        x += positional_encodings_like(x)
        x = self.dropout(x)
        encoding = []
        for layer in self.layers:
            x = layer(x)
            encoding.append(x)
        return encoding


class Decoder(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.out = field.out
        self.layers = nn.ModuleList([DecoderLayer(args) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio

    def forward(self, x, encoding):
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        x += positional_encodings_like(x)
        x = self.dropout(x)
        for l, (layer, enc) in enumerate(zip(self.layers, encoding)):
            x = layer(x, enc)
        return self.out(x)


class Transformer(nn.Module):

    def __init__(self, src, trg, args):
        super().__init__()
        for field in set((src, trg)):
            field.out = nn.Linear(args.d_model, len(field.vocab))
        self.encoder = Encoder(src, args)
        self.decoder = Decoder(trg, args)

    def forward(self, encoder_inputs, decoder_inputs, decoding=False, beam=1, alpha=0.6, return_probs=False):
        encoding = self.encoder(encoder_inputs)
        if return_probs and decoding or not decoding:
            out = self.decoder(decoder_inputs, encoding)
        if decoding:
            if beam == 1:
                output = self.decoder.greedy(encoding)
            else:
                output = self.decoder.beam_search(encoding, beam, alpha)
            if return_probs:
                return output, out
            return output
        return out

    def loss(self, batch, reduce=True, unbatch=False):
        if unbatch:
            loss = 0
            for src, trg in zip(batch.src.examples(), batch.trg.examples()):
                logits = self(src, trg[:, :-1])
                loss += F.cross_entropy(logits, trg[:, 1:], reduce=reduce)
            return loss
        logits = self(batch.src, batch.trg[:, :-1])
        return F.cross_entropy(logits, batch.trg[:, 1:], reduce=reduce)


def any_active(mask_expr):
    return gast.Call(gast.Attribute(mask_expr, gast.Name('any', gast.Load(), None), gast.Load()), [], [])


def pushmask(mask_expr):
    return gast.Expr(gast.Call(gast.Attribute(gast.Name('matchbox', gast.Load(), None), gast.Name('push_execution_mask', gast.Load(), None), gast.Load()), [mask_expr], []))


class CodeToAst(object):
    """Given a module, or a function that was compiled as part
    of a module, re-compile the module into an AST and extract
    the sub-AST for the function.  Allow caching to reduce
    number of compiles.
    Also contains static helper utility functions to
    look for python files, to parse python files, and to extract
    the file/line information from a code object.
    """

    @staticmethod
    def parse_file(fname):
        """Parse a python file into an AST.
        This is a very thin wrapper around ast.parse
            TODO: Handle encodings other than the default (issue #26)
        """
        try:
            with open(fname, 'r') as f:
                fstr = f.read()
        except IOError:
            if fname != 'stdin':
                raise
            sys.stdout.write('\nReading from stdin:\n\n')
            fstr = sys.stdin.read()
        fstr = fstr.replace('\r\n', '\n').replace('\r', '\n')
        if not fstr.endswith('\n'):
            fstr += '\n'
        return ast.parse(fstr, filename=fname)

    @staticmethod
    def get_file_info(codeobj):
        """Returns the file and line number of a code object.
            If the code object has a __file__ attribute (e.g. if
            it is a module), then the returned line number will
            be 0
        """
        fname = getattr(codeobj, '__file__', None)
        linenum = 0
        if fname is None:
            func_code = codeobj.__code__
            fname = func_code.co_filename
            linenum = func_code.co_firstlineno
        fname = fname.replace('.pyc', '.py')
        return fname, linenum

    def __init__(self, cache=None):
        self.cache = cache or {}

    def __call__(self, codeobj):
        cache = self.cache
        key = self.get_file_info(codeobj)
        result = cache.get(key)
        if result is not None:
            return result
        fname = key[0]
        cache[fname, 0] = mod_ast = gast.ast_to_gast(self.parse_file(fname))
        for obj in gast.walk(mod_ast):
            if isinstance(obj, gast.FunctionDef):
                cache[fname, obj.lineno] = obj
        return cache[key]


code_to_ast = CodeToAst()


def compile_file(source, globals_=None):
    """Compile by saving to file and importing that.
    Compiling the AST/source code this way ensures that the source code is
    readable by e.g. `pdb` or `inspect`.
    Args:
    source: The code to compile, either as a string or as an AST.
    globals_: A dictionary of variables that should be available as globals in
        the compiled module. They will be monkey patched after importing the
        module.
    Returns:
    A module object containing the compiled source code.
    """
    if isinstance(source, gast.AST):
        source = astor.to_source(gast.gast_to_ast(source))
    tempdir = tempfile.mkdtemp()
    uuid = str(uuid4().hex[:4])
    tmpname = os.path.join(tempdir, 'matchbox_%s.py' % uuid)
    with open(tmpname, 'w') as f:
        f.write(source)
    module_name = 'matchbox_%s' % uuid
    if six.PY3:
        spec = util.spec_from_file_location(module_name, tmpname)
        m = util.module_from_spec(spec)
        spec.loader.exec_module(m)
    else:
        m = imp.load_source(module_name, tmpname)
    if globals_:
        m.__dict__.update(globals_)
    return m


def compile_function(node, globals_=None):
    """Convert an AST into a function with inspectable source.
    This function uses `compile_file` internally, but instead of returning the
    entire module it will return the function only.
    Args:
    node: A `FunctionDef` node or a `Module` node which contains at least one
        `FunctionDef` node. If a module contains multiple functions, a handle
        to the first one will be returned.
    globals_: See `compile_file`
    Returns:
    A handle to the compiled function.
    Raises:
    TypeError: If the input is not a string or AST.
    ValueError: If no function can be found.
    """
    module = compile_file(node, globals_)
    return getattr(module, node.name)


def batch(fn):
    node = code_to_ast(fn)
    node = ExecutionMasking().visit(node)
    return compile_function(node, fn.__globals__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForward,
     lambda: ([], {'d_model': 4, 'd_hidden': 4, 'drop_ratio': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_salesforce_matchbox(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

