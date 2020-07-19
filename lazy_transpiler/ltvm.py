import ast
import inspect
import logging
import textwrap
from collections import defaultdict
from collections import deque
from copy import deepcopy
from types import ModuleType
from typing import List

import torch

from lazy_transpiler.callable_decoder import CallableDecoder
from lazy_transpiler.dynamic_analysis import TrackingState, Flag, DeferredGraph
from paritybench.module_extractor import to_source
from paritybench.static_analysis import ExtractReadsWrites

log = logging.getLogger(__name__)


def type_specialization_key(value):
    """ Convert a value to a hashable id we can use to decide to reuse or regenerate a trace. """
    if isinstance(value, torch.nn.Module):
        # each instance of a module gets a different trace
        return id(value)
    return type(value)


class LazyTranspilerVirtualMachine(ModuleType):
    """
    Virtual machine that executes python and translates it to python+graphs.

    We process python one expression/statement at a time.  If that statement is
    "python-stuff" we run it, if that statement is `torch.*` operations we
    defer them until the results are needed and we have built up a large graph.
    This outputs runnable python, so on the second call we can reuse the
    past output.

    In generated code this object can be reference by the __ltvm__ variable.
    """
    # Setting to control where to insert type checks
    guard_sources = {
        # Flag.from_global,
        # Flag.from_self,
        Flag.from_args,
    }

    def __init__(self, root_callable):
        super().__init__("__ltvm__")
        self.root_callable = CallableDecoder.parse(root_callable)
        self.root_block = LTVMBlock(self.root_callable.statements(),
                                    self.root_callable.initial_tracking())
        # contains a copy of the root block specific to input types
        self.specializations = defaultdict(self.root_block.clone)
        # generated blocks the root block can jump to
        self.blocks: List[LTVMBlock] = []
        self.graphs: List[LTVMGraph] = []
        # active local variable used in generated code
        self.local_vars = set(self.root_callable.writes)
        self.scope = {}
        self.modules = {self.root_callable.module.__name__: self.root_callable.module}
        self.break_ex = LTVMBreak
        self.continue_ex = LTVMContinue
        self.unique_name_counter = 0

    def unique_name(self, code='d'):
        c = self.unique_name_counter
        self.unique_name_counter += 1
        return f"_{code}_{c:02d}"

    def __str__(self):
        """ Debug printout """
        # TODO(jansel): cleanup debug printouts
        root_block = textwrap.indent(str(next(iter(self.specializations.values()))), '    ')
        blocks = "\n".join(f"__ltvm__.blocks[{i}]:\n{textwrap.indent(str(v), '    ')}"
                           for i, v in enumerate(self.blocks))
        graphs = "\n".join(f"__ltvm__.graphs[{i}]:\n{textwrap.indent(str(v), '    ')}"
                           for i, v in enumerate(self.graphs))
        return f"LazyTranspilerVirtualMachine:\n__ltvm__.root_block:\n{root_block}\n{blocks}\n{graphs}"

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
            self.specializations[specialize_key].tracking.init_args(args.keys())
            self.specializations[specialize_key].run(self, is_root=True)
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

    def graph_(self, index):
        """ Called from generated user code to implement switching blocks of code """
        self.graphs[index].run(self)

    def return_(self, value=None):
        """ Called from generated user code to implement `return value` """
        raise LTVMReturnValue(value)

    def break_(self):
        """ Called from generated user code to implement `break` """
        raise LTVMBreak()

    def continue_(self):
        """ Called from generated user code to implement `continue` """
        raise LTVMContinue()


class LTVMStatement(object):
    """
    Wrapper around a ast.AST that tracks where it came from and can cache
    some analysis.
    """

    def __init__(self, node: ast.AST, func: CallableDecoder):
        super().__init__()
        self.node: ast.AST = node
        self.filename: str = func.filename
        self.module: ModuleType = func.module
        self.func = func

        node.filename = self.filename
        # node.lineno already exists in ast.AST

        self.reads, self.writes = ExtractReadsWrites.run(node)

    @property
    def node_name(self):
        return self.node.__class__.__name__

    def block(self, body, tracking, suffix=None):
        return LTVMBlock(
            [LTVMStatement(s, self.func) for s in body] + (suffix or []),
            tracking
        )

    def derived(self, node):
        """ another statement derived from this one """
        if isinstance(node, str):
            node = ast.parse(node).body[0]
        ast.fix_missing_locations(ast.copy_location(node, self.node))
        return LTVMStatement(node, self.func)

    def before_execute(self, tracking: TrackingState, ltvm: LazyTranspilerVirtualMachine):
        """ Some cleanup right before running self.node """
        # module_name = self.module.__name__
        # replacements = {}
        # for var in self.reads:
        #     if var in tracking.global_vars:
        #         replacements[var] = ast.parse(f"__ltvm__.modules['{module_name}'].{var}").body[0].value

        # if replacements:
        #     ltvm.modules[module_name] = self.module
        #     return Replace(replacements).visit(deepcopy(self.node))

        return self.node


class LTVMBlock(object):
    """
    Holds a block of code.  The first time we run this we use
    LTVMBlockTranspiler, then subsequent calls just run the generated code
    directly.
    """

    def __init__(self, statements, tracking: TrackingState):
        super(LTVMBlock, self).__init__()
        self.statements = statements
        self.specializations = []
        self.tracking = tracking.clone()

    def clone(self):
        return self.__class__(self.statements, self.tracking)

    def __str__(self):
        if self.specializations:
            return str(self.specializations[0])
        return "<not yet compiled>"

    def run(self, ltvm: LazyTranspilerVirtualMachine, is_root=False):
        if self.specializations:
            # TODO(jansel): turn this into a lookup table and inline the checks in the generated code
            # for now we assume exactly one specialization
            return self.specializations[0].run(ltvm)

        transpiler = LTVMBlockTranspiler(self, ltvm)
        try:
            if is_root:
                transpiler.load_globals(ltvm.root_callable.module.__name__)
            transpiler.run_all()
        finally:
            self.specializations.append(transpiler.finalize())


class LTVMGraph(LTVMBlock):
    def __init__(self, statements: DeferredGraph, tracking: TrackingState, ltvm: LazyTranspilerVirtualMachine):
        self.specialize_on_vars = []
        for var in statements.inputs:
            if tracking.var_flags[var] & ltvm.guard_sources:
                self.specialize_on_vars.append(var)

        self.nodes = []
        for stmt in statements:
            self.nodes.append(stmt.before_execute(tracking, ltvm))

        super().__init__(statements, tracking)
        self.specializations = dict()

    def __str__(self):
        if not self.specializations:
            return "<not yet compiled>"
        code = str(next(iter(self.specializations.values())))
        return f"specialized_on: {', '.join(self.specialize_on_vars)}\n{code}"

    def run(self, ltvm: LazyTranspilerVirtualMachine):
        key = tuple(type_specialization_key(ltvm.get_value(var)) for var in self.specialize_on_vars)
        specialized = self.specializations.get(key)
        if not specialized:
            self.specializations[key] = specialized = LTVMSpecializedGraph(self.nodes, self.tracking)
        return specialized.run(ltvm)


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
        self.tracking = block.tracking.clone()

    def finalize(self):
        """
        End the transpilation and produce compiled specialized code
        """
        self.hopper.clear()
        self.unwind_deferred()
        return LTVMSpecializedBlock(self.output_statements, self.tracking)

    def is_pytorch(self, var: str):
        """ Check if we should defer a statement consuming the given variable name """
        if self.tracking.has_flags([var], Flag.deferred) or var in TrackingState.builtins:
            return False  # can't tell yet
        value = self.ltvm.get_value(var)
        try:
            module = inspect.getmodule(value)
        except AttributeError:
            module = inspect.getmodule(type(value))
        if module is None:
            return False
        # log.info(f"{var} {type(var)} {module}")
        module_name = module.__name__
        return module_name == "torch" or module_name.startswith("torch.")

    def should_defer(self, stmt: LTVMStatement):
        """ True if we should add a statement to the deferred graph """
        for var in stmt.reads:
            if self.is_pytorch(var):
                self.tracking.add_flags([var], Flag.pytorch)
        input_flags = self.tracking.combined_flags(stmt.reads)
        write_clash = bool(stmt.writes.intersection(self.tracking.deferred_graph.vars))
        if Flag.special in input_flags:
            self.unwind_deferred()
            return False

        if stmt.node_name in {"Assign", "AugAssign", "AnnAssign", "Expr"}:
            # defer pytorch stuff, run non-pytorch stuff
            return write_clash or Flag.deferred in input_flags or Flag.pytorch in input_flags

        if write_clash:
            # TODO(jansel): introduce a temporary variable to avoid the clash
            self.unwind_deferred()

        if stmt.node_name in {"Delete", "Import", "ImportFrom", "Global", "Nonlocal", "Pass", "If"}:
            # These don't trigger an unwind of our deferrals
            return False

        # TODO(jansel): support defer past other statements
        # Default behavior is to unwind the deferred graph
        self.unwind_deferred()
        return False

    def execute_or_defer(self, stmt: LTVMStatement):
        """ execute/defer a statement, track its impact, and add it to the generated code """
        if self.should_defer(stmt):
            self.defer_statement(stmt)
        else:
            self.execute_statement(stmt)

    def unwind_deferred_call(self):
        graph = LTVMGraph(self.tracking.pop_deferred(), self.tracking, self.ltvm)
        index = len(self.ltvm.graphs)
        self.ltvm.graphs.append(graph)
        call = ast.fix_missing_locations(ast.copy_location(
            self.make_ltvm_call("graph_", [ast.Constant(index, None)]),
            graph.statements[0].node))
        call.filename = graph.statements[0].filename
        return call

    def unwind_deferred(self):
        """ Run all deferred statements """
        if not self.tracking.deferred_graph:
            return
        call = self.unwind_deferred_call()
        self.output_statements.append(call)
        exec(compile(ast.Interactive([call]),
                     call.filename,
                     "single"),
             self.ltvm.scope,
             self.ltvm.scope)

    def defer_statement(self, stmt: LTVMStatement):
        self.tracking.defer(stmt)

    def execute_statement(self, stmt: LTVMStatement):
        self.tracking.execute(stmt)
        node = stmt.before_execute(self.tracking, self.ltvm)
        self._execute_node(node)

    def _execute_node(self, node: ast.AST):
        # log.info("RUN: %s", to_source(node).strip())
        self.output_statements.append(node)
        exec(compile(ast.Interactive([node]),
                     node.filename,
                     "single"),
             self.ltvm.scope,
             self.ltvm.scope)

    def make_jump(self, block: LTVMBlock, locations_from):
        if not block.statements and not block.tracking.deferred_graph:
            return []
        index = len(self.ltvm.blocks)
        self.ltvm.blocks.append(block)
        return [ast.fix_missing_locations(ast.copy_location(
            self.make_ltvm_call("block_", [ast.Constant(index, None)]),
            locations_from))]

    def make_ltvm_call(self, name: str, args: List[ast.AST]):
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
        return node

    def load_globals(self, module_name):
        """ copy globals into local variables """
        for var in self.tracking.global_vars:
            node = ast.parse(f"{var} = __ltvm__.modules['{module_name}'].{var}").body[0]
            node.filename = "<load_globals>"
            self._execute_node(node)

    def run_all(self):
        while self.hopper:
            stmt = self.hopper.popleft()
            getattr(self, f"run_{stmt.node_name}")(stmt)

    def run_Return(self, stmt):
        self.unwind_deferred()
        self.execute_or_defer(stmt.derived(self.make_ltvm_call("return_", [stmt.node.value])))

    def run_Break(self, stmt):
        self.unwind_deferred()
        self.execute_or_defer(stmt.derived(self.make_ltvm_call("break_", [])))

    def run_Continue(self, stmt):
        self.unwind_deferred()
        self.execute_or_defer(stmt.derived(self.make_ltvm_call("continue_", [])))

    def run_If(self, stmt):
        node = deepcopy(stmt.node)
        suffix = list(self.hopper)

        reads, _ = ExtractReadsWrites.run(node.test)
        if self.tracking.has_flags(reads, Flag.deferred):
            self.unwind_deferred()

        # value = self.ltvm.get_value(node.test)
        # if value:
        #    # force if to be false this run to generate cleaner code
        #    node.body, node.orelse = node.orelse, node.body
        #    node.test = ast.copy_location(ast.UnaryOp(ast.Not(), node.test), node)

        node.body = self.make_jump(stmt.block(node.body, self.tracking, suffix=suffix), node)
        node.orelse = self.make_jump(stmt.block(node.orelse, self.tracking, suffix=suffix), node)

        # pending work is moved inside the if
        self.hopper.clear()
        self.tracking.deferred_graph.clear()

        self.execute_or_defer(stmt.derived(node))

    def run_For(self, stmt):
        self.unwind_deferred()

        node = deepcopy(stmt.node)
        # TODO(jansel): double check dynamic analyis handling
        # TODO(jansel): add support for break/continue
        # TODO(jansel): run tracking to a fixed point
        node.body = self.make_jump(stmt.block(node.body, self.tracking), node)
        node.orelse = self.make_jump(stmt.block(node.orelse, self.tracking), node)
        self.execute_or_defer(stmt.derived(node))

    run_Import = execute_or_defer
    run_ImportFrom = execute_or_defer
    run_Delete = execute_or_defer
    run_Assign = execute_or_defer
    run_AugAssign = execute_or_defer
    run_AnnAssign = execute_or_defer
    run_Assert = execute_or_defer
    run_Expr = execute_or_defer
    run_Pass = execute_or_defer

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

    def __init__(self, statements, final_tracking_debug):
        super().__init__()
        self.statements = statements
        self.bytecode = compile(ast.Module(statements, []), "<ltvm>", "exec")
        self.final_tracking_debug = final_tracking_debug

    def run(self, ltvm):
        exec(self.bytecode, ltvm.scope, ltvm.scope)

    def __str__(self):
        result = "\n".join(map(str.rstrip, map(to_source, self.statements)))
        # f"{result}\n{self.final_tracking_debug}"
        return result


class LTVMSpecializedGraph(LTVMSpecializedBlock):
    pass


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
