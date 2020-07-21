import ast
import inspect
import logging
import textwrap
from functools import wraps

import torch

from lazy_transpiler.dynamic_analysis import TrackingState, Flag
from lazy_transpiler.transformations import Flatten
from paritybench.module_extractor import to_source
from paritybench.static_analysis import ExtractReadsWrites

log = logging.getLogger(__name__)


def is_callable_whitelist(fn):
    """
    :param fn: a callable function
    :return: True if it is ok to call this
    """
    return True


class CallableDecoder(object):
    """
    Wrapper around a callable the parses it into python AST.

    We also flatten nested expressions such that every statement contains
    at most one expression.
    """

    @staticmethod
    def parse(callable):
        return CallableDecoder(callable)

    def __init__(self, callable):
        super().__init__()
        assert isinstance(callable, torch.nn.Module)
        self.decode_nn_module(callable)

    def bind(self, args, kwargs):
        bound = self.signature.bind(self.self_ptr, *args, **kwargs)
        bound.apply_defaults()
        return bound

    def statements(self):
        # TODO(jansel): fix circular import
        from lazy_transpiler.ltvm import LTVMStatement
        return [LTVMStatement(s, self) for s in self.tree.body[0].body]

    def initial_tracking(self):
        tracking = TrackingState()
        args = set(self.signature.parameters.keys())
        tracking.add_flags(next(iter(self.signature.parameters.keys())), Flag.from_self)
        tracking.add_flags(self.writes, set())  # locals
        tracking.add_flags(["__ltvm__"], Flag.special)
        tracking.add_globals(self.reads - self.writes - args)
        return tracking

    def decode_nn_module(self, nn):
        # This is hacky and needs some cleanup

        self.self_ptr = nn

        forward = nn.forward
        if nn.forward.__qualname__ == "Module._forward_unimplemented":
            forward = nn.__call__

        self.signature = inspect.signature(forward.__func__)
        self.filename = inspect.getfile(forward)
        self.module = inspect.getmodule(forward)

        source1 = textwrap.dedent(inspect.getsource(forward)).lstrip()
        log.info(f"{nn.__class__.__name__}:\n{source1}")

        tree = ast.parse(source1)
        tree = Flatten.run(tree)

        assert len(tree.body) == 1, ast.dump(tree)
        assert isinstance(tree.body[0], ast.FunctionDef), ast.dump(tree)

        tree.body[0].name = self.name = f"_v_{tree.body[0].name}"

        source2 = to_source(tree)
        log.info(f"{nn.__class__.__name__}:\n{source2}\n")

        # TODO(jansel): remove this hack needed for `super()` support
        self.module.__t_class = nn.__class__
        self.module.__t_self = nn
        self.tree = tree
        self.reads, self.writes = ExtractReadsWrites.run(tree.body[0].body)

        rw = set.intersection(self.reads, self.writes)
        log.debug("\nREAD+WRITE: %s \nREADS: %s, \nWRITES %s\n\n",
                  ','.join(rw),
                  ','.join(self.reads - rw),
                  ','.join(self.writes - rw))

    def debug_call(self, args, kwargs):
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
        exec(compile(ast.Interactive(self.tree.body), self.filename, "single"), self.module.__dict__, scope)
        fn = scope[self.name]
        return fn(self.self_ptr, *args, **kwargs)
