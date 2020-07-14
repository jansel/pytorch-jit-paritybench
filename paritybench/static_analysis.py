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


class FlattenStatement(ast.NodeTransformer):
    """
    Simplify AST to remove nested expressions.

    a = b + c + foo()

    becomes
    _t_0 = b + c
    _t_1 = foo()
    a = _t_0 + _t_1
    """

    def __init__(self, flattener):
        super().__init__()
        self.unique_name = flattener.unique_name
        self.prefix = []
        self.suffix = []

    def __call__(self, node):
        node = self.visit(node)
        if node is None:
            return self.prefix + self.suffix
        elif isinstance(node, list):
            return self.prefix + node + self.suffix
        else:
            return self.prefix + [node] + self.suffix

    def to_tmp(self, node):
        if isinstance(node, (
                ast.Name, ast.Constant, ast.NamedExpr, ast.expr_context, ast.keyword, ast.arguments,
                ast.withitem, ast.excepthandler,
                ast.operator, ast.boolop, ast.unaryop, ast.cmpop, type(None))):
            return node

        ctx = getattr(node, "ctx", ast.Load())
        if isinstance(ctx, ast.Load):
            return self.to_tmp_Load(node)
        if isinstance(ctx, ast.Store):
            return self.to_tmp_Store(node)
        if isinstance(ctx, ast.Del):
            return None
        assert False, f"Unknown ctx: {ast.dump(node)}"

    def to_tmp_Load(self, node):
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
        self.prefix.append(assign)
        return load

    def to_tmp_Store(self, node):
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
        self.suffix.append(assign)
        return store

    def to_tmp_Del(self, node):
        return node

    def to_tmp_visit(self, node):
        return self.to_tmp(self.visit(node))

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            new_node = ast.Delete([self.visit(target)])
            ast.copy_location(new_node, node)
            self.prefix.append(new_node)
        return None

    def visit_Subscript(self, node):
        node.value = self.to_tmp_visit(node.value)
        node.slice = self.visit(node.slice)
        return node

    visit_ExtSlice = ast.NodeTransformer.generic_visit
    visit_Assign = ast.NodeTransformer.generic_visit

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.to_tmp_visit(value)
                    if value is not None:
                        new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.to_tmp_visit(old_value)
                setattr(node, field, new_node)
        return node

    def visit_BoolOp(self, node):
        node.values[0] = self.to_tmp_visit(node.values[0])
        # TODO(jansel): convert values[0] to if
        return self.to_tmp(node)

    def visit_IfExp(self, node):
        ident = self.unique_name()
        assign_if = ast.If(
            test=self.to_tmp_visit(node.test),
            body=FlattenStatement(self)(
                ast.Assign(
                    targets=[ast.Name(ident, ast.Store())],
                    value=node.body)
            ),
            orelse=FlattenStatement(self)(
                ast.Assign(
                    targets=[ast.Name(ident, ast.Store())],
                    value=node.orelse)
            )
        )
        ast.copy_location(assign_if, node)
        ast.fix_missing_locations(assign_if)

        load = ast.Name(ident, ast.Load())
        ast.copy_location(load, node)
        return load

    def visit_Lambda(self, node):
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
        self.prefix.append(self.generic_visit(fn))

        load = ast.Name(name, ctx=ast.Load())
        ast.copy_location(load, node)
        return load

    def visit_Expr(self, node):
        node.value = self.visit(node.value)
        return node

    def _comprehension(self, node, add_name, init, add_args):
        assert len(node.generators) == 1, "expected 1 generator " + ast.dump(node)
        data = self.unique_name()
        add = f"{data}_{add_name}"
        statements = [
            ast.Assign(
                targets=[ast.Name(data, ast.Store())],
                value=init
            ),
            ast.Assign(
                targets=[ast.Name(add, ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(data, ast.Load()),
                    attr=add_name,
                    ctx=ast.Load())
            ),
            ast.For(
                target=node.generators[0].target,  # TODO(jansel): flatten target
                iter=self.to_tmp_visit(node.generators[0].iter),
                body=FlattenStatement(self)(self._comprehension_if(
                    node.generators[0].ifs,
                    ast.Expr(value=ast.Call(
                        func=ast.Name(add, ast.Load()),
                        args=add_args,
                        keywords=[],
                    )))),
                orelse=[]
            ),
        ]
        for stmt in statements:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
            self.prefix.append(stmt)

        load = ast.Name(data, ast.Load())
        ast.copy_location(load, node)
        return load

    def _comprehension_if(self, conds, inner):
        for cond in reversed(conds):
            inner = ast.If(cond, [inner], [])
        return inner

    def visit_ListComp(self, node):
        return self._comprehension(
            node,
            "append",
            ast.List([], ctx=ast.Load()),
            [node.elt])

    '''
    def visit_SetComp(self, node, output):  # (expr elt, comprehension* generators)
        # TODO(jansel): need to implement this
        return self.to_tmp(node, output)

    def visit_DictComp(self, node, output):  # (expr key, expr value, comprehension* generators)
        # TODO(jansel): need to implement this
        return self.to_tmp(node, output)

    def visit_GeneratorExp(self, node, output):  # (expr elt, comprehension* generators)
       # TODO(jansel): need to implement this
       return self.to_tmp(node, output)
   '''


class Flatten(ast.NodeTransformer):
    """
    Simplify AST to remove nested expressions.

    a = b + c + foo()

    becomes
    _t_0 = foo()
    _t_1 = c + _t_0
    a = b + _t_1
    """

    @classmethod
    def run(cls, tree):
        return cls().visit(tree)

    def __init__(self):
        super().__init__()
        self._cnt = 0
        self.prefix = None
        self.suffix = None

    def unique_name(self):
        """ Create a name for a new local variable """
        v = self._cnt
        self._cnt += 1
        return f"_t_{v:02d}"

    def visit_Import(self, node):
        result = []
        for module_name, new_node in split_import(node):
            result.append(new_node)
        return result

    visit_ImportFrom = visit_Import

    def flatten_statement(self, node):
        return FlattenStatement(self)(node)

    visit_Return = flatten_statement
    visit_Delete = flatten_statement
    visit_Assign = flatten_statement
    visit_AugAssign = flatten_statement
    visit_AnnAssign = flatten_statement
    visit_Raise = flatten_statement
    visit_Expr = flatten_statement

    def visit_For(self, node: (ast.For, ast.AsyncFor)):
        log.info(ast.dump(node))

        fs1 = FlattenStatement(self)
        node.iter = fs1.to_tmp_visit(node.iter)
        assert not fs1.suffix, "for.iter should not add to suffix"

        fs2 = FlattenStatement(self)
        node.target = fs2.to_tmp_visit(node.target)
        assert not fs2.prefix, f"For.targets should not add to prefix"

        if node.body:
            node.body = fs2.suffix + node.body

        if node.orelse:
            node.orelse = fs2.suffix + node.orelse

        return fs1.prefix + [self.generic_visit(node)]

    visit_AsyncFor = visit_For

    # TODO(jansel): visit_While
    # TODO(jansel): visit_Try

    def visit_If(self, node: (ast.If, ast.Assert)):
        fs = FlattenStatement(self)
        node.test = fs.to_tmp_visit(node.test)
        assert not fs.suffix, f"{node.__class__} should not cause a suffix"
        return fs.prefix + [self.generic_visit(node)]

    visit_Assert = visit_If

    def visit_With(self, node: (ast.With, ast.AsyncWith)):
        fs = FlattenStatement(self)
        for item in node.items:
            item.context_expr = fs.visit(item.context_expr)
            # TODO(jansel): handle item.optional_vars
        assert not fs.suffix, f"{node.__class__} should not cause a suffix"
        return fs.prefix + [self.generic_visit(node)]

    visit_AsyncWith = visit_With
