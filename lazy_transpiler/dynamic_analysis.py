import ast
import builtins
import enum
from collections import namedtuple
from functools import reduce
import logging

log = logging.getLogger(__name__)


class DeferredGraph(list):
    def __init__(self):
        super().__init__()
        self.reads = set()
        self.writes = set()
        self.vars = set()
        self.inputs = set()

    def append(self, val):
        self.reads.update(val.reads)
        self.writes.update(val.writes)
        self.vars.update(val.reads)
        self.vars.update(val.writes)
        self.inputs.update(val.reads - self.writes)
        super().append(val)

    def clone(self):
        copy = DeferredGraph()
        copy.extend(self)
        copy.reads.update(self.reads)
        copy.writes.update(self.writes)
        copy.vars.update(self.vars)
        copy.inputs.update(self.inputs)
        return copy


AttributeSource = namedtuple("AttributeSource", ["src", "attr"])


class TrackingState(object):
    """
    Dynamically computed metadata about variables in an LTVM function
    being transpiled.
    """
    builtins = set(dir(builtins))

    def __init__(self):
        super().__init__()
        self.var_flags = {k: FlagSet() for k in self.builtins}
        self.var_source = dict()
        self.deferred_graph = DeferredGraph()
        self.return_stack = []
        self.global_vars = set()

    def clone(self):
        copy = TrackingState()
        copy.var_flags = dict(self.var_flags)
        copy.var_source = dict(self.var_source)
        copy.global_vars = set(self.global_vars)
        copy.deferred_graph = self.deferred_graph.clone()
        copy.return_stack = list(self.return_stack)
        return copy

    def pop_deferred(self):
        graph = self.deferred_graph
        self.deferred_graph = DeferredGraph()
        for stmt in graph:
            self.propogate_flags(stmt.reads, stmt.writes)
        self.remove_flags(graph.writes, Flag.deferred)
        assert not self.has_flags(graph.vars, Flag.deferred), f"{self}"
        return graph

    def add_globals(self, vars):
        vars = set(vars) - self.builtins
        self.global_vars.update(vars)
        self.add_flags(vars, Flag.from_global)

    def defer(self, stmt):
        self.deferred_graph.append(stmt)
        self.add_flags(stmt.writes, Flag.deferred)

    def execute(self, stmt, ltvm, allow_missing=False):
        assert not self.has_flags(stmt.reads, Flag.deferred)
        flags = self.propogate_flags(stmt.reads, stmt.writes, allow_missing=allow_missing)
        log.debug(f"propogate_flags %s  # %s=%s", stmt, stmt.writes, flags)
        if stmt.is_getattr():
            dst, src, attr = stmt.split_getattr()
            self.var_source[dst] = AttributeSource(src, attr)
        if stmt.is_call():
            func_var = stmt.get_call_var()
            source = self.var_source.get(func_var)
            if (isinstance(source, AttributeSource) and (allow_missing or
                    getattr(ltvm.get_value(func_var), "__self__", None) is not None)):
                self.add_flags([source.src], flags)

    def init_args(self, vars):
        assert not isinstance(vars, str)
        for var in vars:
            old = self.var_flags.get(var, FlagSet())
            if Flag.from_self not in old:
                self.var_flags[var] = old | {Flag.from_args}

    def set_flags(self, vars, flags):
        assert not isinstance(vars, str)
        assert isinstance(flags, FlagSet)
        for var in vars:
            self.var_flags[var] = flags

    def add_flags(self, vars, flags):
        assert not isinstance(vars, str)
        for var in vars:
            self.var_flags[var] = self.var_flags.get(var, FlagSet()) | flags

    def remove_flags(self, vars, flags):
        assert not isinstance(vars, str)
        for var in vars:
            self.var_flags[var] = self.var_flags[var] - flags

    def has_flags(self, vars, flags):
        assert not isinstance(vars, str)
        for var in vars:
            if self.var_flags[var] & flags:
                return True
        return False

    def propogate_flags(self, reads, writes, allow_missing=False):
        if allow_missing:
            flags = FlagSet.combine([self.var_flags.get(x, FlagSet()) for x in reads])
        else:
            flags = FlagSet.combine(map(self.var_flags.__getitem__, reads))
        self.set_flags(writes, flags)
        self.add_flags(writes, Flag.computed)
        for var in writes:
            self.var_source.pop(var, None)
        return flags

    def fixed_point(self, stmt, ltvm):
        """ Don't know how many times we will go around a loop, need to find a fixed point """
        subblocks = list(stmt.subblocks())
        if subblocks:
            bare = stmt.without_subblocks()
            self.execute(bare, ltvm, allow_missing=True)
            if stmt.is_loop():
                self.add_flags(bare.writes, Flag.from_iter)
            for statements, category in subblocks:
                getattr(self, f"fixed_point_{category.name}")(statements, ltvm)
        else:
            self.execute(stmt, ltvm, allow_missing=True)

    def fixed_point_looping(self, statements, ltvm):
        """ for code that could run many times, like loops """
        for _ in range(16):
            if self.fixed_point_maybe(statements, ltvm):
                return

    def fixed_point_maybe(self, statements, ltvm):
        """ for code that might run, like conditionals """
        before = dict(self.var_flags)
        for stmt in statements:
            self.fixed_point(stmt, ltvm)
        after = {k: v | before.get(k, FlagSet()) for k, v in self.var_flags.items()}
        self.var_flags = after
        return before == after

    def fixed_point_once(self, statements, ltvm):
        """ for code that might run, like conditionals """
        for stmt in statements:
            self.fixed_point(stmt, ltvm)

    def fixed_point_define(self, statements, ltvm):
        pass

    def __str__(self):
        return "TrackingState({})".format(", ".join(
            f"{k}: {v}"
            for k, v in self.var_flags.items()
            if k not in self.builtins
        ))

    def combined_flags(self, vars):
        return FlagSet.combine(map(self.var_flags.__getitem__, vars))

    def is_constants(self, vars):
        return not self.has_flags(vars, {Flag.from_args, Flag.from_global, Flag.from_self,
                                         Flag.from_iter, Flag.deferred})

    def is_builtin(self, var):
        # TODO(jansel): handle shadowing builtins with locals
        return var in self.builtins


@enum.unique
class Flag(enum.IntEnum):
    # Where did a variable come from?
    from_global = 1
    from_self = 2
    from_args = 4
    from_iter = 8

    # Other things to track:
    computed = 16
    pytorch = 32  # points to torch.*
    deferred = 64  # not yet executed
    special = 128  # __ltvm__


# could optimize this to a bitmask
class FlagSet(frozenset):
    def __or__(self, other):
        if isinstance(other, Flag):
            other = {other}
        else:
            other = other or set()
            assert isinstance(other, (frozenset, set))
        return self.__class__(self.union(other))

    def __and__(self, other):
        if isinstance(other, Flag):
            other = {other}
        else:
            other = other or set()
            assert isinstance(other, (frozenset, set))
        return self.__class__(self.intersection(other))

    def __sub__(self, other):
        if isinstance(other, Flag):
            other = {other}
        else:
            assert isinstance(other, (frozenset, set))
        return self.__class__(super().__sub__(other))

    @classmethod
    def combine(cls, inputs):
        inputs = list(inputs)
        if len(inputs) == 0:
            return cls()
        if len(inputs) == 1:
            return cls(inputs[0])
        return reduce(cls.__or__, inputs)

    def __str__(self):
        return "+".join(f.name for f in sorted(self))


@enum.unique
class Subblock(enum.Enum):
    maybe = 0  # runs 0 or 1 times
    once = 1  # runs 1 time
    looping = 2  # runs 0 or more times
    defined = 3  # runs 0 times
    _handlers = 4  # ast.Try(...).handlers
