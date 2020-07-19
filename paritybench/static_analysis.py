import ast
import logging
import re
import torch

log = logging.getLogger(__name__)

CONFIG_NAMES = {"argv", "args", "config", "cfg", "params", "_global_config"}
IMPORT_WHITELIST = {
    "abc",
    "collections",
    "copy",
    "enum",
    "functools",
    "inspect",
    "itertools",
    "logging",
    "math",
    "matplotlib",
    "numbers",
    "numpy",
    "pandas",
    "queue",
    "random",
    "re",
    "scipy",
    "sklearn",
    "string",
    "tensorflow",
    "time",
    "torch",
    "torchaudio",
    "torchtext",
    "torchvision",
    "types",
    "typing",
    "uuid",
    "warnings",
}


class ASTCleanup(ast.NodeTransformer):
    """
    Remove prints, imports, and cudas from a AST.
    """

    def visit_Import(self, node):
        result = []
        for module_name, new_node in split_import(node):
            if module_name in IMPORT_WHITELIST:
                result.append(new_node)
        return result

    visit_ImportFrom = visit_Import

    def visit_Call(self, node: ast.Call):
        if getattr(node.func, 'id', '') == 'print':
            # Strip print() calls
            return ast.Expr(ast.Constant(value=None, kind=None))
        if getattr(node.func, 'attr', '') in ('cuda', 'to'):
            # foo.cuda() => foo
            return node.func.value
        if getattr(node.func, 'id', '') == 'cuda_' and len(node.args) == 1:
            return node.args[0]
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Attribute):
            node2 = node.value
            if getattr(node2.value, 'id', '') == "torch" and node2.attr == "cuda":
                if hasattr(torch, node.attr):
                    # torch.cuda.FloatTensor => torch.FloatTensor
                    new_node = ast.Attribute(value=node2.value,
                                             attr=node.attr,
                                             ctx=node.ctx)
                    ast.copy_location(new_node, node)
                    return new_node
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        if not node.decorator_list:
            new_node = node
        else:
            # Strip some decorators
            new_node = ast.ClassDef(
                name=node.name,
                bases=node.bases,
                keywords=node.keywords,
                body=node.body,
                decorator_list=filter_decorators(node.decorator_list))
            ast.copy_location(new_node, old_node=node)

        return self.generic_visit(new_node)

    def visit_Assert(self, node: ast.Assert):
        if 'is_cuda' in ast.dump(node):
            return None
        return self.generic_visit(node)


def filter_decorators(decorator_list):
    return [
        node for node in decorator_list
        if 'regist' not in ast.dump(node)
    ]


class ExtractReadsWrites(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, tree):
        visitor = cls()
        if isinstance(tree, (list, tuple)):
            for node in tree:
                visitor.visit(node)
        else:
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
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)

        reads, _ = self.context[-1]
        _, target_writes = ExtractReadsWrites.run(node.target)
        reads.update(target_writes)


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


def split_import(node):
    """
    Replace `import a,b` with `import a; import b`
    """
    if isinstance(node, ast.Import):
        for name in node.names:
            tmp = ast.Import([name])
            ast.copy_location(tmp, node)
            module_name = re.sub(r"[.].*$", "", name.name)
            yield module_name, tmp
    else:
        assert isinstance(node, ast.ImportFrom)
        if node.level != 0:
            return  # not supported
        module_name = re.sub(r"[.].*$", "", node.module)
        for name in node.names:
            tmp = ast.ImportFrom(re.sub(r"^torch.legacy\b", "torch", node.module),
                                 [name],
                                 level=0)
            ast.copy_location(tmp, node)
            yield module_name, tmp


def copy_locations_recursive(new_node, old_old):
    ast.copy_location(new_node, old_old)
    ast.fix_missing_locations(new_node)
    return new_node
