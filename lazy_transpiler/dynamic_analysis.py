import copy
import enum
from functools import reduce


class TrackingState(object):
    """
    Dynamically computed metadata about variables in an LTVM function
    being transpiled.
    """
    builtins = {
        # TODO(jansel): this list is incomplete
        "int", "float", "long", "complex", "list", "dict", "set", "range",
        "isinstance", "issubclass", "id", "str", "bytes", "iter", "next"}

    def __init__(self):
        super().__init__()
        self.var_flags = {k: FlagSet() for k in self.builtins}
        self.deferred_graph = []

    def defer(self, stmt):
        self.deferred_graph.append(stmt)
        self.propogate_flags(stmt)
        self.add_flags(stmt.writes, Flag.deferred)

    def execute(self, stmt):
        assert not self.has_flags(stmt.reads, Flag.deferred)
        self.propogate_flags(stmt)

    def clone(self):
        return copy.deepcopy(self)

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

    def propogate_flags(self, stmt):
        # TODO(jansel): double check this handles ast.AugAssign correctly
        self.set_flags(stmt.writes,
                       FlagSet.combine(map(self.var_flags.__getitem__, stmt.reads)))
        self.add_flags(stmt.writes, Flag.computed)

    def __str__(self):
        return "TrackingState({})".format(", ".join(
            f"{k}: {v}"
            for k, v in self.var_flags.items()
            if k not in self.builtins
        ))


@enum.unique
class Flag(enum.IntEnum):
    # Where did a variable come from? """
    from_global = 1
    from_self = 2
    from_args = 4

    # Other things to track:
    computed = 16
    pytorch = 32  # points to a pytorch object
    deferred = 64  # not yet executed
    special = 128


# could optimize this to a bitmask
class FlagSet(frozenset):
    def __or__(self, other):
        if isinstance(other, Flag):
            other = {other}
        else:
            assert isinstance(other, (frozenset, set))
        return self.__class__(self.union(other))

    def __and__(self, other):
        if isinstance(other, Flag):
            other = {other}
        else:
            assert isinstance(other, (frozenset, set))
        return self.__class__(self.intersection(other))

    def __sub__(self, other):
        if isinstance(other, Flag):
            other = {other}
        else:
            assert isinstance(other, (frozenset, set))
        return self.__class__(self - other)

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
