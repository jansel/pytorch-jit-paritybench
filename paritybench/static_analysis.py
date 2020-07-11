import ast
import logging
import re
import torch

import paritybench.evaluate

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


class Flatten(ast.NodeTransformer):
    """
    Simplify AST to remove nested expressions.

    a = b + c + foo()

    becomes
    _t_0 = foo()
    _t_1 = c + _t_0
    a = b + _t_1
    """

    def __init__(self):
        super().__init__()
        self._cnt = 0

    def unique_name(self):
        v = self._cnt
        self._cnt += 1
        return f"_t_{v}"

    def visit_Import(self, node):
        result = []
        for module_name, new_node in split_import(node):
            result.append(new_node)
        return result

    visit_ImportFrom = visit_Import

    def visit_Return(self, node: ast.Return):
        statements = []
        node.value = self.flatten(node.value, statements)
        return statements + [node]

    def visit_Delete(self, node: ast.Delete):
        statements = []
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                target.value = self.flatten(target.value, statements)
                target.slice = self.flatten(target.slice, statements)
                statements.append(ast.Delete([target]))
                ast.copy_location(statements[-1], node)
            else:
                statements.append(ast.Delete([self.flatten(target, statements)]))
                ast.copy_location(statements[-1], node)
        return statements

    def visit_Assign(self, node: (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        statements = []
        suffix = []
        node.value = self.flatten(node.value, statements)
        if len(node.targets) > 1:
            node.targets = [self.flatten_target(t, suffix) for t in node.targets]
        return statements + [node] + suffix

    visit_AugAssign = visit_Assign
    visit_AnnAssign = visit_Assign

    def visit_For(self, node: (ast.For, ast.AsyncFor)):
        statements = []
        node.iter = self.flatten(node.iter, statements)
        # TODO(jansel): handle targets
        node = self.generic_visit(node)
        return statements + [node]

    visit_AsyncFor = visit_For

    def visit_If(self, node: ast.If):
        statements = []
        node.test = self.flatten(node.test, statements)
        node = self.generic_visit(node)
        return statements + [node]

    def visit_Raise(self, node: ast.Raise):
        statements = []
        node.exc = self.flatten(node.exc, statements)
        node.cause = self.flatten(node.cause, statements)
        return statements + [node]

    def visit_Assert(self, node: ast.Assert):
        statements = []
        node.test = self.flatten(node.test, statements)
        # TODO(jansel): handlle node.msg
        return statements + [node]

    def visit_With(self, node: (ast.With, ast.AsyncWith)):
        statements = []
        for item in node.items:
            item.context_expr = self.flatten(item.context_expr, statements)
            # TODO(jansel): handle item.optional_vars
        node = self.generic_visit(node)
        return statements + [node]

    visit_AsyncWith = visit_With

    def to_tmp(self, node, output):
        ident = self.unique_name()
        store = ast.Name(ident, ast.Store())
        load = ast.Name(ident, ast.Load())
        assign = ast.Assign(
            targets=[store],
            value=node
        )
        ast.copy_location(assign, node)
        ast.copy_location(store, node)
        ast.copy_location(load, node)
        output.append(assign)
        return load

    def to_tmp_target(self, node, output):
        ident = self.unique_name()
        store = ast.Name(ident, ast.Store())
        load = ast.Name(ident, ast.Load())
        assign = ast.Assign(
            targets=[node],
            value=load
        )
        ast.copy_location(assign, node)
        ast.copy_location(store, node)
        ast.copy_location(load, node)
        output.append(assign)
        return store

    def visit_Expr(self, node: ast.Expr):
        statements = []
        node.value = self.flatten(node.value, statements)
        return statements + [node]

    #########################################################
    # simplify the left hand side of an assignment
    #     self.foo, self.bar = blah
    # becomes:
    #     _t_0, _t_1 = blah
    #     self.foo = _t_0
    #     self.bar = _t_1

    def flatten_target(self, node, suffix):
        method = 'flatten_target_' + node.__class__.__name__
        return getattr(self, method)(node, suffix)

    def flatten_target_Name(self, node, _):
        # (identifier id, expr_context ctx)
        return node

    def flatten_target_Starred(self, node, suffix):
        # (expr value, expr_context ctx)
        node.value = self.flatten_target(node.value, suffix)
        return node

    def flatten_target_Attribute(self, node, suffix):
        # (expr value, identifier attr, expr_context ctx)
        node.value = self.flatten(node.value, suffix)
        return self.to_tmp_target(node, suffix)

    def flatten_target_Subscript(self, node, suffix):
        # (expr value, slice slice, expr_context ctx)
        node.value = self.flatten(node.value, suffix)
        node.slice = self.flatten_slice(node.slice, suffix)
        return self.to_tmp_target(node, suffix)

    def flatten_target_List(self, node, suffix):
        # (expr* elts, expr_context ctx)
        node.elts = [self.flatten_target(n, suffix) for n in node.elts]
        return node

    flatten_target_Tuple = flatten_target_List

    #########################################################

    def flatten(self, node, output_statements):
        method = 'flatten_' + node.__class__.__name__
        return getattr(self, method)(node, output_statements)

    def flatten_None(self, node, _):
        assert node is None
        return None

    def flatten_Name(self, node, _):
        return node

    def flatten_Constant(self, node, output):
        return node

    def flatten_BoolOp(self, node, output):
        node.values[0] = self.flatten(node.values[0], output)
        # TODO(jansel): handle [1:] values
        return self.to_tmp(node, output)

    def flatten_BinOp(self, node, output):
        node.left = self.flatten(node.left, output)
        node.right = self.flatten(node.right, output)
        return self.to_tmp(node, output)

    def flatten_UnaryOp(self, node, output):
        node.operand = self.flatten(node.operand, output)
        return self.to_tmp(node, output)

    def flatten_NamedExpr(self, node, output):
        value = self.flatten(node.value, output)
        assert isinstance(value, (ast.Name, ast.Constant))
        assign = ast.Assign(
            targets=[node.target],
            value=value
        )
        ast.copy_location(assign, node)
        output.append(assign)
        return value

    def flatten_IfExp(self, node, output):
        paritybench.evaluate.evaluate = self.flatten(paritybench.evaluate.evaluate, output)
        # TODO(jansel): handle node.{body, orelse}
        return self.to_tmp(node, output)

    def flatten_Compare(self, node, output):
        node.left = self.flatten(node.left, output)
        node.comparators = [self.flatten(n, output) for n in node.comparators]
        return self.to_tmp(node, output)

    def flatten_Call(self, node, output):
        node.func = self.flatten(node.func, output)
        node.args = [self.flatten(a, output) for a in node.args]
        node.keywords = [ast.keyword(arg=k.arg, value=self.flatten(k.value, output)) for k in node.keywords]
        return self.to_tmp(node, output)

    def flatten_Attribute(self, node, output):
        node.value = self.flatten(node.value, output)
        return self.to_tmp(node, output)

    def flatten_Subscript(self, node, output):
        node.value = self.flatten(node.value, output)
        node.slice = self.flatten(node.slice, output)
        return self.to_tmp(node, output)

    def flatten_slice(self, node: ast.slice, output):
        node.lower = self.flatten(node.lower, output)
        node.upper = self.flatten(node.upper, output)
        node.step = self.flatten(node.step, output)
        return node

    def flatten_Starred(self, node, output):
        node.value = self.flatten(node.value, output)
        return node

    def flatten_Attribute(self, node, output):
        # (expr value, identifier attr, expr_context ctx)
        node.value = self.flatten(node.value, output)
        return self.to_tmp(node, output)

    def flatten_Subscript(self, node, output):
        # (expr value, slice slice, expr_context ctx)
        node.value = self.flatten(node.value, output)
        node.slice = self.flatten_slice(node.slice, output)
        return self.to_tmp(node, output)

    def flatten_List(self, node, output):
        # (expr* elts, expr_context ctx)
        node.elts = [self.flatten_target(n, output) for n in node.elts]
        return self.to_tmp(node)

    flatten_Tuple = flatten_List

    def flatten_Yield(self, node, output):  #
        node.value = self.flatten(node.value, output)
        return node

    flatten_YeildFrom = flatten_Yield
    flatten_Await = flatten_Yield

    def flatten_Dict(self, node, output):
        # (expr* keys, expr* values)
        node.keys = [self.flatten(n) for n in node.keys]
        node.values = [self.flatten(n) for n in node.values]
        return self.to_tmp(node, output)

    def flatten_Set(self, node, output):
        # (expr* elts)
        node.elts = [self.flatten(n) for n in node.elts]
        return self.to_tmp(node, output)

    def flatten_JoinedStr(self, node, output):
        # (expr* values)
        node.values = [self.flatten(n, output) for n in node.values]
        return self.to_tmp(node, output)

    def flatten_FormattedValue(self, node, output):  # (expr value, int? conversion, expr? format_spec)
        node.value = self.flatten(node.value, output)
        return node

    def flatten_Lambda(self, node, output):
        """ Converts Lambda to FunctionDef """
        # (arguments args, expr body)
        name = self.unique_name()
        rv = ast.Return(node.body)
        ast.copy_location(rv, node)

        fn = ast.FunctionDef(
            name,
            node.args,  # TODO(jansel): flatten these?
            body=[rv]
        )
        ast.copy_location(fn, node)
        output.append(self.generic_visit(fn))

        load = ast.Name(name, ctx=ast.Load())
        ast.copy_location(load, node)
        return load

    '''
    def flatten_ListComp(self, node, output):  # (expr elt, comprehension* generators)
        # TODO(jansel): need to implement this
        return self.to_tmp(node, output)

    def flatten_SetComp(self, node, output):  # (expr elt, comprehension* generators)
        # TODO(jansel): need to implement this
        return self.to_tmp(node, output)

    def flatten_DictComp(self, node, output):  # (expr key, expr value, comprehension* generators)
        # TODO(jansel): need to implement this
        return self.to_tmp(node, output)

    def flatten_GeneratorExp(self, node, output):  # (expr elt, comprehension* generators)
       # TODO(jansel): need to implement this
       return self.to_tmp(node, output)
   '''
