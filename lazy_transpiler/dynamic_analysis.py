import builtins
import enum
from copy import deepcopy
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


class TrackingState(object):
    """
    Dynamically computed metadata about variables in an LTVM function
    being transpiled.
    """
    builtins = set(dir(builtins))

    def __init__(self):
        super().__init__()
        self.var_flags = {k: FlagSet() for k in self.builtins}
        self.deferred_graph = DeferredGraph()
        self.return_stack = []
        self.global_vars = set()

    def clone(self):
        copy = TrackingState()
        copy.var_flags = dict(self.var_flags)
        copy.global_vars = set(self.global_vars)
        copy.deferred_graph = self.deferred_graph.clone()
        copy.return_stack = list(self.return_stack)
        return copy

    def pop_deferred(self):
        graph = self.deferred_graph
        self.deferred_graph = DeferredGraph()
        for stmt in graph:
            self.propogate_flags(stmt)
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

    def execute(self, stmt, mutates=None):
        assert not self.has_flags(stmt.reads, Flag.deferred)
        self.propogate_flags(stmt, mutates)

    def init_args(self, vars):
        for var in vars:
            old = self.var_flags.get(var, FlagSet())
            if Flag.from_self not in old:
                self.var_flags[var] = old | {Flag.from_args}

    def set_flags(self, vars, flags):
        for var in vars:
            self.var_flags[var] = flags

    def add_flags(self, vars, flags):
        for var in vars:
            self.var_flags[var] = self.var_flags.get(var, FlagSet()) | flags

    def remove_flags(self, vars, flags):
        for var in vars:
            self.var_flags[var] = self.var_flags[var] - flags

    def has_flags(self, vars, flags):
        for var in vars:
            if self.var_flags[var] & flags:
                return True
        return False

    def propogate_flags(self, stmt, mutates=None):
        mutates = mutates or []
        reads = list(stmt.reads) + mutates
        writes = list(stmt.writes) + mutates
        flags = self.propogate_flags_(reads, writes)
        log.debug(f"propogate_flags %s  # %s=%s", stmt, stmt.writes, flags)

    def run_to_fixed_point(self, statements):
        for _ in range(64):
            before = dict(self.var_flags)
            for stmt in statements:
                self.propogate_flags(stmt)
            after = {k: v | before.get(k, {}) for k, v in self.var_flags.items()}
            if before == after:
                return
            self.var_flags = after

    def propogate_flags_(self, reads, writes):
        flags = FlagSet.combine(map(self.var_flags.__getitem__, reads))
        self.set_flags(writes, flags)
        self.add_flags(writes, Flag.computed)
        return flags

    def __str__(self):
        return "TrackingState({})".format(", ".join(
            f"{k}: {v}"
            for k, v in self.var_flags.items()
            if k not in self.builtins
        ))

    def combined_flags(self, vars):
        return FlagSet.combine(map(self.var_flags.__getitem__, vars))

    def is_constants(self, vars):
        log.debug("is_constants %s", vars)
        return not self.has_flags(vars, {
            Flag.from_args, Flag.from_global, Flag.from_self, Flag.deferred})

    def is_builtin(self, var):
        # TODO(jansel): handle shadowing builtins with locals
        return var in self.builtins


@enum.unique
class Flag(enum.IntEnum):
    # Where did a variable come from?
    from_global = 1
    from_self = 2
    from_args = 4

    # Other things to track:
    computed = 8
    pytorch = 16  # points to torch.*
    deferred = 32  # not yet executed
    special = 64  # __ltvm__


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
