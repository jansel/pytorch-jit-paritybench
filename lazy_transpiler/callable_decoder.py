import ast
import inspect
import logging
import math
import os
import textwrap
from collections import OrderedDict
from functools import wraps

import torch
import numpy

from lazy_transpiler.dynamic_analysis import TrackingState, Flag, FlagSet
from lazy_transpiler.transformations import Flatten, Rename, OffsetLineno
from paritybench.module_extractor import to_source
from paritybench.static_analysis import ExtractReadsWrites, copy_locations_recursive

log = logging.getLogger(__name__)

ALLOWED_BUILTIN_COLLECTIONS = {list, set, dict, OrderedDict}
ALLOWED_BUILTIN_MODULES = {math, numpy, logging}
ALLOWED_BUILTIN_FUNCTIONS = {str, repr, int, float, range}


def is_user_file(filename):
    # TODO(jansel): need to make this check better
    return filename.startswith("./")


def is_callable_allowlist(fn):
    """
    We only let a small number of pure functions to be called.

    :param fn: a callable
    :return: True if it is ok to call this without inlining
    """
    module = inspect.getmodule(fn)
    self_ptr = getattr(fn, "__self__", None)

    if fn in ALLOWED_BUILTIN_FUNCTIONS:
        return True

    if module in ALLOWED_BUILTIN_MODULES:
        return True

    if module is None:
        module_name = str(getattr(fn, "__module__", None))

        # usually C or builtin stuff
        if module_name.startswith("torch._C."):
            return True

        if module_name in {"torch", "numpy"}:
            return True

        if type(self_ptr) in ALLOWED_BUILTIN_COLLECTIONS:
            return True

        if getattr(torch, fn.__name__, None) is fn:
            return True

        if isinstance(fn, numpy.ufunc):
            return True

        if isinstance(self_ptr, (torch.Tensor, str, numpy.ndarray)):
            return True

        if inspect.isclass(self_ptr) and issubclass(self_ptr, torch.autograd.Function) and fn.__name__ == "apply":
            # TODO(jansel): for now, allow custom autograd functions -- later we should inline them
            return True

        assert module is not None, f"No module? {fn} {self_ptr}"

    try:
        filename = inspect.getfile(fn)
    except TypeError:
        filename = module.__file__

    if filename.startswith(os.path.dirname(torch.__file__)):
        return True

    if is_user_file(filename):
        return False

    assert False, f"Unsupported function {fn} {filename}"
    return True


def is_self_mutating(fn):
    return type(getattr(fn, "__self__", None)) in ALLOWED_BUILTIN_COLLECTIONS


class CallableDecoder(object):
    """
    Wrapper around a callable the parses it into python AST.

    We also flatten nested expressions such that every statement contains
    at most one expression.
    """

    @staticmethod
    def parse(callable):
        if isinstance(callable, torch.nn.Module):
            return NNModuleDecoder(callable)
        elif hasattr(callable, "__func__"):
            return MethodDecoder(callable)
        else:
            assert not hasattr(callable, "__self__")
            return FunctionDecoder(callable)

        assert False, f"Can't decode {callable}"

    def __str__(self):
        return f"{self.name}{self.signature}"

    def is_bound_method(self):
        return self.self_ptr is not None

    def is_simple_args(self):
        return all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                   p.default == inspect.Parameter.empty
                   for p in self.signature.parameters.values())

    def globals(self):
        return self.reads - self.writes - set(self.signature.parameters.keys()) - set(TrackingState.builtins)

    def bind(self, args, kwargs):
        bound = self.signature.bind(self.self_ptr, *args, **kwargs)
        bound.apply_defaults()
        return bound

    def statements(self):
        return wrap_statements(self, self.tree.body)

    def initial_tracking(self):
        tracking = TrackingState()
        args = list(self.signature.parameters.keys())
        tracking.add_flags(self.writes, set())  # locals
        tracking.add_flags(["__ltvm__"], Flag.special)
        tracking.add_globals(self.globals())
        tracking.set_flags(args, FlagSet({Flag.from_args}))
        if self.is_bound_method():
            tracking.set_flags(args[:1], FlagSet({Flag.from_self}))
        return tracking

    def inline_static_binding(self, prefix: str, args: list, kwargs: dict, locations_from):
        """
        Return te statements needed to inline this method inside another

        :param prefix: rename all variables to start with this
        :param args: to bind in inlined code
        :param kwargs: to bind in inlined code
        :return: [LTVMStatement(...), ...]
        """
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bindings = []
        for name, value in bound.arguments.items():
            assert isinstance(value, (ast.AST, dict, list, str, int, float, bool, type(None))), str(value)
            if isinstance(value, list):  # *args
                assert all(isinstance(x, ast.AST) for x in value)
                value = ast.List(value, ast.Load())
            if isinstance(value, dict):  # **kwargs
                assert all(isinstance(x, ast.AST) for x in value.values())
                items = list(value.items())
                value = ast.Dict([ast.Constant(k, None) for k, v in items],
                                 [v for k, v in items])
            if not isinstance(value, ast.AST):
                value = ast.Constant(value, None)
            bindings.append(copy_locations_recursive(
                ast.Assign(
                    [ast.Name(prefix + name, ast.Store())],
                    value
                ),
                locations_from))
        return self.inline_statements(prefix, bindings)

    def inline_runtime_binding(self, index: int, prefix: str, args: list, keywords: dict, locations_from):
        """ Inline more complex calls that do things with **kwargs """
        args = [ast.Constant(index, None)] + args
        unpack = ast.Assign(
            [ast.Tuple([ast.Name(prefix + name, ast.Store())
                        for name in self.signature.parameters.keys()],
                       ctx=ast.Store())],
            ast.Call(ast.Attribute(ast.Name('__ltvm__', ast.Load()), 'unpack_', ast.Load()),
                     args=args,
                     keywords=keywords)
        )
        copy_locations_recursive(unpack, locations_from)
        return self.inline_statements(prefix, [unpack])

    def inline_statements(self, prefix, bindings):
        renamer = Rename({
            name: prefix + name for name in self.variables()
        })
        return wrap_statements(self, bindings) + wrap_statements(
            self, renamer.visit(self.tree).body)

    def variables(self):
        args = set(self.signature.parameters.keys())
        args.update(self.writes)
        args.update(self.reads - set(TrackingState.builtins))
        return args

    def debug_call(self, args: list, kwargs: dict):
        """
        Used to check the correctness of our static transforms by running
        the loaded code directly.

        :param args: to user function
        :param kwargs: to user function
        :return: result of user function
        """

        def _staticmethod(fn):
            @wraps(fn)
            def _fn(self, *args, **kwargs):
                return fn(*args, **kwargs)

            return _fn

        def _classmethod(fn):
            @wraps(fn)
            def _fn(self, *args, **kwargs):
                return fn(self.__class__, *args, **kwargs)

            return _fn

        scope = {
            'staticmethod': _staticmethod,
            'classmethod': _classmethod,
        }
        exec(compile(ast.Interactive([self.tree]), self.filename, "single"), self.module.__dict__, scope)
        fn = scope[self.tree.name]
        return fn(self.self_ptr, *args, **kwargs)

    def read_source_from(self, fn):
        self.signature = inspect.signature(getattr(fn, "__func__", fn))
        self.filename = inspect.getfile(fn)
        self.module = inspect.getmodule(fn)

        source1 = textwrap.dedent(inspect.getsource(fn)).lstrip()
        log.info(f"{fn.__class__.__name__}:\n{source1}")

        tree = ast.parse(source1)
        lineno = getattr(fn, "__func__", fn).__code__.co_firstlineno
        tree = OffsetLineno(lineno - 1).visit(tree)
        tree = Flatten.run(tree)

        assert len(tree.body) == 1, to_source(tree)
        assert isinstance(tree.body[0], ast.FunctionDef)

        tree.body[0].name = self.name = f"_v_{tree.body[0].name}"

        source2 = to_source(tree)
        log.info(f"{fn.__class__.__name__}:\n{source2}\n")

        self.reads, self.writes = ExtractReadsWrites.run(tree.body[0].body)
        self.tree = tree.body[0]

        if not isinstance(self.tree.body[-1], ast.Return):
            # add an explict return at the end
            self.tree.body = self.tree.body + [ast.copy_location(ast.Return(None), self.tree)]

        assert not self.tree.decorator_list, (
            f"function decorators not yet supported {to_source(self.tree.decorator_list[0]).strip()}")


class MethodDecoder(CallableDecoder):
    def __init__(self, fn):
        super().__init__()
        self.self_ptr = fn.__self__
        self.read_source_from(fn.__func__)


class FunctionDecoder(CallableDecoder):
    def __init__(self, fn):
        super().__init__()
        self.self_ptr = None
        self.read_source_from(fn)


class NNModuleDecoder(CallableDecoder):
    def __init__(self, nn):
        super().__init__()

        self.self_ptr = nn

        forward = nn.forward
        if nn.forward.__qualname__ == "Module._forward_unimplemented":
            forward = nn.__call__

        self.read_source_from(forward)

        self.name = nn.__class__.__name__ + ".forward"

        # rw = set.intersection(self.reads, self.writes)
        # log.debug("\nREAD+WRITE: %s \nREADS: %s, \nWRITES %s\n\n",
        #          ','.join(rw),
        #          ','.join(self.reads - rw),
        #          ','.join(self.writes - rw))


def wrap_statements(self, statements):
    # TODO(jansel): fix circular import
    from lazy_transpiler.ltvm import LTVMStatement
    return [LTVMStatement(s, self) for s in statements]
