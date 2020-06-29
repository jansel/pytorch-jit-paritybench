import ast
import re
import logging

log = logging.getLogger(__name__)

CONFIG_NAMES = {"argv", "args", "config", "cfg", "params", "_global_config"}


class ASTCleanup(ast.NodeTransformer):
    """
    Remove prints, imports, and cudas from a AST.
    """

    def visit_Import(self, node):
        return ast.Constant(value=None, kind=None)

    visit_ImportFrom = visit_Import

    def visit_Call(self, node: ast.Call):
        if getattr(node.func, 'id', '') == 'print':
            # Strip print() calls
            return ast.Constant(value=None, kind=None)
        if getattr(node.func, 'attr', '') in ('cuda', 'to'):
            # foo.cuda() => foo
            return node.func.value
        return self.generic_visit(node)


class ExtractReadsWrites(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, tree):
        visitor = cls()
        visitor.visit(tree)
        assert len(visitor.context) == 1
        return visitor.context[0]

    def __init__(self):
        super().__init__()
        self.context = [(set(), set())]  # Read/Writes

    def visit_Global(self, node):
        global_reads, global_writes = self.context[0]
        global_reads.update(node.names)
        global_writes.update(node.names)

    def visit_Name(self, node):
        reads, writes = self.context[-1]
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            writes.add(node.id)
        else:
            assert isinstance(node.ctx, ast.Load)
            reads.add(node.id)

    def visit_Import(self, node):
        reads, writes = self.context[-1]
        for alias in node.names:
            if alias.asname:
                writes.add(alias.asname)
            else:
                writes.add(re.findall(r"[^.]+$", alias.name)[0])

    visit_ImportFrom = visit_Import

    def visit_FunctionDef(self, node):
        _, parent_writes = self.context[-1]
        try:
            parent_writes.add(node.name)
        except AttributeError:
            pass  # Lambda
        self.context.append((set(), set()))
        self.generic_visit(node)
        reads, writes = self.context.pop()
        self.context[-1][0].update(reads - writes)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_arg(self, node):
        reads, writes = self.context[-1]
        writes.add(node.arg)


class ExtractConfigUsage(ast.NodeVisitor):
    """
    Find items like `config.hidden_size` and return {"hidden_size"}
    """

    @classmethod
    def run(cls, tree):
        visitor = cls()
        visitor.visit(tree)
        return visitor.needed_keys

    def __init__(self):
        super().__init__()
        self.needed_keys = set()

    def visit_Attribute(self, node):
        lhs = getattr(node.value, "id", "")
        if lhs in CONFIG_NAMES:
            self.needed_keys.add(node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        lhs = getattr(node.value, "id", "")
        rhs = getattr(getattr(node.slice, "value", ""), "value", "")
        if lhs in CONFIG_NAMES and rhs and isinstance(rhs, (str, int)):
            self.needed_keys.add(rhs)
        self.generic_visit(node)


class CheckCallableMembers(ast.NodeVisitor):
    """
    Find `self.foo()` in the AST then check to make sure `obj.foo` is
    callable on the constructed module.  Used to find cases where __init__
    runs, but produces invalid modules.
    """

    @classmethod
    def run(cls, tree):
        visitor = cls()
        if tree:
            visitor.visit(tree)
        return visitor

    def __init__(self):
        super().__init__()
        self.callable_members = set()

    def check(self, obj):
        for name in self.callable_members:
            member = getattr(obj, name, None)
            if member is not None and not callable(member):
                raise ValueError(f"member {repr(name)} should be callable")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            attr = node.func
            if getattr(attr.value, 'id', '') == 'self':
                self.callable_members.add(attr.attr)
        return self.generic_visit(node)
