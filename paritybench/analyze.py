import ast
import inspect
import logging
import textwrap
from collections import defaultdict
from collections import deque
from copy import deepcopy
from functools import partial
from functools import wraps
from types import ModuleType
from typing import List

import torch

from paritybench import evaluate
from paritybench.evaluate import JitFailed, init_module, run_eager, check_output
from paritybench.module_extractor import to_source
from paritybench.static_analysis import Flatten
from paritybench.utils import subproc_wrapper

log = logging.getLogger(__name__)


def type_specialization_key(value):
    """ Convert a value to a hashable id we can use to decide to reuse or regenerate a trace. """
    if isinstance(value, torch.nn.Module):
        # each instance of a module gets a different trace
        return id(value)
    return type(value)


class LazyTranspilerVirtualMachine(ModuleType):
    """
    Virtual machine that executes python and translates it to python+graphs just-in-time.

    We process python one expression at a time.  If that statement is
    "python-stuff" we run it, if that statement is `torch.*` operations we
    defer them until the results are needed and we have built up a large graph.
    This outputs runnable python, so on the second call we can reuse the
    past output.

    In generated code this object can be reference by the __ltvm__ variable.
    """

    def __init__(self, root_callable):
        super().__init__("__ltvm__")
        self.root_callable = LTVMCallable.parse(root_callable)
        self.root_block = LTVMBlock(self.root_callable.statements())
        # contains a copy of the root block specific to input types
        self.specializations = defaultdict(self.root_block.clone)
        # generated blocks the root block can jump to
        self.blocks: List[LTVMBlock] = []
        # active local variable used in generated code
        self.scope = {}
        self.break_ex = LTVMBreak
        self.continue_ex = LTVMContinue

    def __str__(self):
        """ Debug printout """
        # TODO(jansel): print all the specializations
        root_block = textwrap.indent(str(next(iter(self.specializations.values()))), '    ')
        return (
                f"LazyTranspilerVirtualMachine:\n__ltvm__.root_block:\n{root_block}\n"
                +
                "\n".join(
                    f"__ltvm__.blocks[{i}]:\n{textwrap.indent(str(v), '    ')}"
                    for i, v in enumerate(self.blocks)
                )
        )

    def init_scope(self, args):
        self.scope = {"__ltvm__": self}
        self.scope.update(args)

    def run(self, *args, **kwargs):
        """
        Transpile and run the wrapped module with the given args

        :param args: for user function we are transpiling
        :param kwargs: for user function we are transpiling
        :return: same as user function
        """
        try:
            args = self.root_callable.bind(args, kwargs).arguments
            specialize_key = tuple(map(type_specialization_key, args.values()))
            self.init_scope(args)
            self.specializations[specialize_key].run(self)
        except LTVMReturnValue as rv:
            return rv.value
        finally:
            self.scope.clear()  # free memory

    @staticmethod
    def nameof(key):
        if isinstance(key, str):
            return key
        if isinstance(key, ast.Name):
            return key.id
        assert False, f"don't know how to get the value of {key}"

    def get_value(self, key):
        return self.scope[self.nameof(key)]

    def set_value(self, key, value, derived_from):
        self.scope[self.nameof(key)] = value

    def block_(self, index):
        """ Called from generated user code to implement switching blocks of code """
        self.blocks[index].run(self)

    def return_(self, value):
        """ Called from generated user code to implement `return value` """
        raise LTVMReturnValue(value)

    def break_(self):
        """ Called from generated user code to implement `break` """
        raise LTVMBreak()

    def continue_(self):
        """ Called from generated user code to implement `continue` """
        raise LTVMContinue()


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
        # This is a bit hacky and needs some cleanup

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


class LTVMStatement(object):
    """
    Wrapper around a ast.AST that tracks where it came from and can cache
    some analysis.
    """

    def __init__(self, statement: ast.AST, func: LTVMCallable):
        super().__init__()
        self.node: ast.AST = statement
        self.filename: str = func.filename
        self.module: ModuleType = func.module
        self.func = func

    @property
    def node_name(self):
        return self.node.__class__.__name__

    def block(self, body, suffix=None):
        return LTVMBlock([LTVMStatement(s, self.func) for s in body] + (suffix or []))


class LTVMBlock(object):
    """
    Holds a block of code.  The first time we run this we use
    LTVMBlockTranspiler, then subsequent calls just run the generated code
    directly.
    """

    def __init__(self, statements):
        super(LTVMBlock, self).__init__()
        self.statements = statements
        self.specializations = []

    def clone(self):
        return self.__class__(self.statements)

    def __str__(self):
        if self.specializations:
            return str(self.specializations[0])
        return "<not yet compiled>"

    def run(self, ltvm: LazyTranspilerVirtualMachine):
        if self.specializations:
            # TODO(jansel): turn this into a lookup table and inline the checks in the generated code
            # for now we assume exactly one specialization
            return self.specializations[0].run(ltvm, self.statements[0].module)

        transpiler = LTVMBlockTranspiler(self, ltvm)
        try:
            transpiler.run_all()
        finally:
            self.specializations.append(transpiler.finalize())


class LTVMBlockTranspiler(object):
    """
    Run a block of code statement-by-statement and produce a
    LTVMSpecializedBlock which we can run on subsequent invocations.
    """

    def __init__(self, block: LTVMBlock, ltvm: LazyTranspilerVirtualMachine):
        super(LTVMBlockTranspiler, self).__init__()
        self.output_statements = []
        self.debug_locations = []
        # hopper contains statements we still need to run
        self.hopper = deque(block.statements)
        self.ltvm = ltvm

    def finalize(self):
        """
        End the transpilation and produce compiled specialized code
        """
        return LTVMSpecializedBlock(self.output_statements)

    def exec_and_record(self, node: ast.AST, stmt: LTVMStatement):
        """ exec() a statement now, track its impact, and add it to the generated code """
        node.filename = stmt.filename
        # node.lineno already exists in ast.AST
        self.output_statements.append(node)

        exec(compile(ast.Interactive([node]),
                     stmt.filename,
                     "single"),
             stmt.module.__dict__,
             self.ltvm.scope)

    def make_jump(self, block: LTVMBlock, locations_from: ast.AST):
        if not block.statements:
            return []
        index = len(self.ltvm.blocks)
        self.ltvm.blocks.append(block)
        return [self.make_ltvm_call("block_", [ast.Constant(index, None)], locations_from)]

    def make_ltvm_call(self, name: str, args: List[ast.AST], locations_from: ast.AST):
        """
        Build an AST node to call a __ltvm__.* function from generated code.

        :param name: of the method to call
        :param args:  AST node list of args
        :param locations_from: copy locations from this node
        :return: ast.Expr(...)
        """
        node = ast.Expr(
            ast.Call(
                ast.Attribute(
                    ast.Name("__ltvm__", ast.Load()),
                    name,
                    ast.Load()
                ),
                args,
                [],
            )
        )
        ast.fix_missing_locations(ast.copy_location(node, locations_from))
        return node

    def run_all(self):
        while self.hopper:
            stmt = self.hopper.popleft()
            getattr(self, f"run_{stmt.node_name}")(stmt)

    def run_Return(self, stmt):
        self.exec_and_record(
            self.make_ltvm_call("return_", [stmt.node.value], stmt.node),
            stmt
        )

    def run_Break(self, stmt):
        self.exec_and_record(
            self.make_ltvm_call("break_", [], stmt.node),
            stmt
        )

    def run_Continue(self, stmt):
        self.exec_and_record(
            self.make_ltvm_call("continue_", [], stmt.node),
            stmt
        )

    def run_If(self, stmt):
        node = deepcopy(stmt.node)
        remaining_statements = list(self.hopper)
        self.hopper.clear()
        node.body = self.make_jump(stmt.block(node.body, suffix=remaining_statements), node)
        node.orelse = self.make_jump(stmt.block(node.orelse, suffix=remaining_statements), node)
        self.exec_and_record(node, stmt)

    def run_For(self, stmt):
        node = deepcopy(stmt.node)
        # TODO(jansel): add support for break/continue
        node.body = self.make_jump(stmt.block(node.body), node)
        node.orelse = self.make_jump(stmt.block(node.orelse), node)
        self.exec_and_record(node, stmt)

    def run_generic(self, stmt: LTVMStatement):
        self.exec_and_record(stmt.node, stmt)

    run_Import = run_generic
    run_ImportFrom = run_generic
    run_Delete = run_generic
    run_Assign = run_generic
    run_AugAssign = run_generic
    run_AnnAssign = run_generic
    run_Assert = run_generic
    run_Expr = run_generic
    run_Pass = run_generic

    def _unimplemented(self, stmt):
        raise NotImplementedError(stmt.node_name)

    run_FunctionDef = _unimplemented
    run_AsyncFunctionDef = _unimplemented
    run_ClassDef = _unimplemented
    run_AsyncWith = _unimplemented
    run_Global = _unimplemented
    run_Nonlocal = _unimplemented
    run_AsyncFor = _unimplemented
    run_While = _unimplemented
    run_With = _unimplemented
    run_Try = _unimplemented
    run_Raise = _unimplemented


class LTVMSpecializedBlock(object):
    """
    Contains a block that has been specialized based on input types
    """

    def __init__(self, statements):
        super().__init__()
        self.statements = statements
        self.bytecode = compile(ast.Module(statements, []), "<ltvm>", "exec")

    def run(self, ltvm, module):
        exec(self.bytecode, module.__dict__, ltvm.scope)

    def __str__(self):
        return "\n".join(map(str.rstrip, map(to_source, self.statements)))


class LTVMException(Exception):
    pass


class LTVMReturnValue(LTVMException):
    """
    End execution and return
    """

    def __init__(self, value):
        super().__init__()
        self.value = value


class LTVMBreak(LTVMException):
    pass


class LTVMContinue(LTVMException):
    pass


def analyze_nn_module(nn_cls, get_init_args, get_forward_args, record_error):
    nn = init_module(record_error, nn_cls, get_init_args)
    args, kwargs, result1, result2 = run_eager(record_error, nn, get_forward_args)

    try:
        result3 = LTVMCallable(nn).debug_call(args, kwargs)
    except Exception as e:
        record_error('flatten', e)
        raise JitFailed()

    check_output(record_error, result1, result2, result3, 'flatten_output')

    try:
        ltvm = LazyTranspilerVirtualMachine(nn)
        result3 = ltvm.run(*deepcopy(args), **deepcopy(kwargs))
        result4 = ltvm.run(*args, **kwargs)
    except Exception as e:
        record_error('ltvm', e)
        raise JitFailed()

    log.info(str(ltvm))

    check_output(record_error, result1, result2, result3, 'ltvm_output')
    check_output(record_error, result1, result2, result4, 'ltvm_rerun')

    return True


analyze_pyfile_subproc = partial(evaluate.evaluate_pyfile_subproc, check_module=analyze_nn_module)
analyze_pyfile = partial(subproc_wrapper, fn=analyze_pyfile_subproc)
analyze_all = partial(evaluate.evaluate_all, fn=analyze_pyfile)
