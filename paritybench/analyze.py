import ast
import copy
import inspect
import logging
import textwrap
from functools import partial
from functools import wraps
from types import ModuleType

import torch

from paritybench import evaluate
from paritybench.evaluate import JitFailed, init_module, run_eager, check_output
from paritybench.module_extractor import to_source
from paritybench.static_analysis import Flatten
from paritybench.utils import subproc_wrapper
from collections import deque

log = logging.getLogger(__name__)


class LazyTranspilerVirtualMachine(object):
    """
    Virtual machine that executes python and translates it to python+graphs just-in-time.

    We process python one expression at a time.  If that statement is
    "python-stuff" we run it, if that statement is `torch.*` operations we
    defer them until the results are needed and we have built up a large graph.
    This outputs runnable python, so on the second call we don't need to
    use this VM.
    """

    def __init__(self, root_callable):
        super().__init__()
        self.root_callable = LTVMCallable.parse(root_callable)
        self.root_block = LTVMBlock(self.root_callable.statements())

        # current local scope
        self.scope = {}

        # global __ltvm object in all generated code
        self.ltvm_state = ModuleType('__ltvm')

    def run(self, args, kwargs):
        self.scope.update(self.root_callable.bind(args, kwargs).arguments)
        try:
            self.root_block.run(self)
        except LTVMReturnValue as rv:
            return rv.value

    def get_value(self, key):
        if isinstance(key, str):
            return self.scope[key]
        if isinstance(key, ast.Name):
            return self.get_value(key.id)
        assert False, f"don't know how to get the value of {key}"


class LTVMCallable(object):
    """
    Wrapper around a callable the parses it into python AST.

    We also flatten nested expressions such that every statement contains
    at most one expression.
    """

    @staticmethod
    def parse(callable):
        return LTVMCallable(callable)

    def __init__(self, callable):
        super().__init__()
        assert isinstance(callable, torch.nn.Module)
        self.decode_nn_module(callable)

    def bind(self, args, kwargs):
        bound = self.signature.bind(self.self_ptr, *args, **kwargs)
        bound.apply_defaults()
        return bound

    def decode_nn_module(self, nn):
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
        log.info(f"{nn.__class__.__name__}:\n{source2}\n\n")

        # TODO(jansel): remove this hack needed for `super()` support
        self.module.__t_class = nn.__class__
        self.module.__t_self = nn

        self.tree = tree

    def statements(self):
        return [LTVMStatement(s, self) for s in self.tree.body[0].body]

    def debug_call(self, *args, **kwargs):
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
        exec(compile(self.tree, self.filename, "exec"), self.module.__dict__, scope)
        fn = scope[self.name]
        return fn(self.self_ptr, *args, **kwargs)


class LTVMStatement(object):
    def __init__(self, statement: ast.AST, func: LTVMCallable):
        super().__init__()
        self.statement : ast.AST = statement
        self.filename : str = func.filename
        self.module : ModuleType = func.module
        self.func = func

    def run(self, ltvm: LazyTranspilerVirtualMachine):
        """
        Run one statement and log the python code

        :param ltvm: virtual machine state
        """
        # log.info(f"RUN:{to_source(self.statement).strip()} # {ltvm.scope}")
        return getattr(self, f"run_{self.statement.__class__.__name__}")(ltvm)

    def run_generic(self, ltvm: LazyTranspilerVirtualMachine):
        exec(compile(ast.Interactive([self.statement]),
                     self.filename,
                     "single"),
             self.module.__dict__,
             ltvm.scope)

    run_Delete = run_generic
    run_Assign = run_generic
    run_AugAssign = run_generic
    run_AnnAssign = run_generic
    run_Assert = run_generic
    run_Expr = run_generic
    run_Pass = run_generic

    def run_Return(self, ltvm: LazyTranspilerVirtualMachine):
        raise LTVMReturnValue(ltvm.get_value(self.statement.value))

    def run_If(self, ltvm: LazyTranspilerVirtualMachine):
        test = ltvm.get_value(self.statement.test)
        body = LTVMBlock([LTVMStatement(s, self.func) for s in self.statement.body])
        orelse = LTVMBlock([LTVMStatement(s, self.func) for s in self.statement.orelse])
        if test:
            body.run(ltvm)
        else:
            orelse.run(ltvm)

    def _unimplemented(self, _):
        raise NotImplementedError(self.statement.__class__.__name__)

    run_FunctionDef = _unimplemented
    run_AsyncFunctionDef = _unimplemented
    run_ClassDef = _unimplemented
    run_AsyncWith = _unimplemented
    run_Import = _unimplemented
    run_ImportFrom = _unimplemented
    run_Global = _unimplemented
    run_Nonlocal = _unimplemented
    run_For = _unimplemented
    run_AsyncFor = _unimplemented
    run_While = _unimplemented
    run_With = _unimplemented
    run_Try = _unimplemented
    run_Break = _unimplemented
    run_Continue = _unimplemented
    run_Raise = _unimplemented



class LTVMBlock(object):
    def __init__(self, statements):
        super(LTVMBlock, self).__init__()
        # hopper is the queue of statements to run
        self.hopper = deque(statements)

    def run(self, ltvm):
        while self.hopper:
            self.hopper.popleft().run(ltvm)


class LTVMReturnValue(Exception):
    """
    Exception to end execution and return
    """

    def __init__(self, value):
        super().__init__()
        self.value = value


def analyze_nn_module(nn_cls, get_init_args, get_forward_args, record_error):
    nn = init_module(record_error, nn_cls, get_init_args)
    args, kwargs, result1, result2 = run_eager(record_error, nn, get_forward_args)

    """"
    try:
        cd = LTCallable.parse(nn)
        cdd = cd.debug_callable()
    except Exception as e:
        record_error('decode', e)
        raise JitFailed()

    try:
        result3 = cdd(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    except Exception as e:
        record_error('run_decoded', e)
        raise JitFailed()

    check_output(record_error, result1, result2, result3, 'decoded_output')
    del result3
    """

    try:
        ltvm = LazyTranspilerVirtualMachine(nn)
        result3 = ltvm.run(args, kwargs)
    except Exception as e:
        record_error('ltvm', e)
        raise JitFailed()

    check_output(record_error, result1, result2, result3, 'ltvm_output')

    return True


analyze_pyfile_subproc = partial(evaluate.evaluate_pyfile_subproc, check_module=analyze_nn_module)
analyze_pyfile = partial(subproc_wrapper, fn=analyze_pyfile_subproc)
analyze_all = partial(evaluate.evaluate_all, fn=analyze_pyfile)
