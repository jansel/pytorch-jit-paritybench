import ast
import copy
import warnings
import logging

from paritybench.static_analysis import copy_locations_recursive, split_import, ExtractReadsWrites

log = logging.getLogger(__name__)


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

    def to_tmp(self, node, aug_assign=False):
        if isinstance(node, (
                ast.Name, ast.Constant, ast.NamedExpr, ast.expr_context, ast.keyword, ast.arguments,
                ast.withitem, ast.excepthandler, ast.Starred, ast.FormattedValue, ast.arg,
                ast.operator, ast.boolop, ast.unaryop, ast.cmpop, type(None))):
            return node

        ctx = getattr(node, "ctx", ast.Load())
        if isinstance(ctx, ast.Load):
            return self.to_tmp_Load(node)
        if isinstance(ctx, ast.Store):
            return self.to_tmp_Store(node, aug_assign=aug_assign)
        if isinstance(ctx, ast.Del):
            return None
        assert False, f"Unknown ctx: {ast.dump(node)}"

    def visit_Call(self, node):
        if getattr(node.func, "id", "") == "super" and not node.args and not node.keywords:
            node.args = [ast.Name("__t_class", ast.Load()),
                         ast.Name("__t_self", ast.Load())]
            ast.fix_missing_locations(node)
        return self.generic_visit(node)

    def unique_name_vars(self, node=None):
        ident = self.unique_name()
        store = ast.Name(ident, ast.Store())
        load = ast.Name(ident, ast.Load())
        if node:
            ast.copy_location(store, node)
            ast.copy_location(load, node)
        return load, store

    def to_tmp_Load(self, node):
        load, store = self.unique_name_vars(node)
        assign = ast.copy_location(ast.Assign(
            targets=[store],
            value=node
        ), node)
        self.prefix.append(assign)
        return load

    def to_tmp_Store(self, node, aug_assign=False):
        if aug_assign:
            # load the old value, then store the new one
            assert isinstance(node.ctx, ast.Store)
            node2 = copy.deepcopy(node)
            node2.ctx = ast.Load()
            load = self.to_tmp_Load(node2)
            store = ast.copy_location(ast.Name(load.id, ast.Store()), node)
        else:
            load, store = self.unique_name_vars(node)
        assign = ast.copy_location(ast.Assign(
            targets=[node],
            value=load
        ), node)
        self.suffix = [assign] + self.suffix
        return store

    def to_tmp_Del(self, node):
        return node

    def to_tmp_visit(self, node):
        if isinstance(node, list):
            return list(map(self.to_tmp_visit, node))
        else:
            return self.to_tmp(self.visit(node))

    def visit_AugAssign(self, node):
        # TODO(jansel): should we convert this to regular Assign?
        node.value = self.to_tmp_visit(node.value)
        node.target = self.to_tmp(self.visit(node.target), aug_assign=True)
        return node

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

    # TODO(jansel): handle: part[i] = _t_28()
    # TODO(jansel): handle: a = b = c

    def visit_Assign(self, node):
        if (isinstance(node.value, ast.Lambda) and
                len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name)):
            # unwrap x=lambda:...
            return copy_locations_recursive(ast.FunctionDef(
                node.targets[0].id,
                node.value.args,
                body=FlattenStatement(self)(ast.Return(node.value.body)),
                decorator_list=[],
            ), node)

        node.targets = self.to_tmp_visit(node.targets)
        node.value = self.visit(node.value)
        return node

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
        self.prefix.append(assign_if)

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
            body=FlattenStatement(self)(rv),
            decorator_list=[],
        )
        ast.copy_location(fn, node)
        self.prefix.append(self.visit(fn))

        load = ast.Name(name, ctx=ast.Load())
        ast.copy_location(load, node)
        return load

    def visit_FunctionDef(self, node):
        # TODO(jansel): handle node.args
        # TODO(jansel): handle node.returns
        # TODO(jansel): handle node.type_comment
        node.decorator_list = list(map(self.to_tmp_visit, node.decorator_list))
        node.body = self._body(node.body)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _body(self, nodes):
        new_body = []
        for node in nodes:
            new_body.extend(FlattenStatement(self)(node))
        return new_body

    def visit_Expr(self, node):
        node.value = self.visit(node.value)
        return node

    def _comprehension(self, node, add_name, init, add_args):
        assert len(node.generators) == 1, "expected 1 generator " + ast.dump(node)

        data = self.unique_name()

        # Name mangling:
        _, target_writes = ExtractReadsWrites.run(node.generators[0].target)
        node = Rename({var: f"{data}_{var}" for var in target_writes}).visit(node)

        statements = [
            copy_locations_recursive(ast.Assign(
                targets=[ast.Name(data, ast.Store())],
                value=init
            ), node)
        ]

        statements.extend(
            FlattenStatement(self)(copy_locations_recursive(
                new_node=ast.For(
                    target=node.generators[0].target,
                    iter=node.generators[0].iter,
                    body=self._comprehension_if(
                        node.generators[0].ifs,
                        FlattenStatement(self)(ast.Expr(value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(data, ast.Load()),
                                attr=add_name,
                                ctx=ast.Load()),
                            args=add_args,
                            keywords=[],
                        )))),
                    orelse=[]
                ),
                old_old=node)
            )
        )
        self.prefix.extend(statements)

        load = ast.Name(data, ast.Load())
        ast.copy_location(load, node)
        return load

    def _comprehension_if(self, conds, inner):
        for cond in reversed(conds):
            inner = [ast.If(cond, inner, [])]
        return inner

    def visit_If(self, node):
        node.test = self.to_tmp_visit(node.test)
        node.body = self._body(node.body)
        node.orelse = self._body(node.orelse)
        return node

    def visit_Assert(self, node):
        node.test = self.to_tmp_visit(node.test)
        # TODO(jansel): convert this to an if
        return node

    def visit_ListComp(self, node):
        return self._comprehension(
            node,
            "append",
            ast.List([], ctx=ast.Load()),
            [node.elt])

    def visit_SetComp(self, node):
        return self._comprehension(
            node,
            "add",
            # ast.Call(ast.Name("set", ast.Load()), [], []),  # "set" might be shadowed?
            ast.Set([], ctx=ast.Load()),
            [node.elt])

    def visit_DictComp(self, node):
        return self._comprehension(
            node,
            "__setitem__",
            ast.Dict([], [], ctx=ast.Load()),
            [node.key, node.value])

    def visit_GeneratorExp(self, node):
        warnings.warn("GeneratorExp not yet supported, converting to ListComp")
        return self.visit_ListComp(node)

    """
    # This works, but creates a FunctionDef+Yield, which we don't support yet
    def visit_GeneratorExp(self, node):
        assert len(node.generators) == 1, "expected 1 generator " + ast.dump(node)
        fn_name = self.unique_name()
        iter_name = self.unique_name()
        statements = [
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments([], [], None, [], [], None, []),
                decorator_list=[],
                body=FlattenStatement(self)(
                    ast.For(
                        target=node.generators[0].target,
                        iter=node.generators[0].iter,
                        body=FlattenStatement(self)(self._comprehension_if(
                            node.generators[0].ifs,
                            ast.Expr(value=ast.Yield(node.elt)))),
                        orelse=[]
                    ),
                ),
            ),
            ast.Assign(
                targets=[ast.Name(iter_name, ast.Store())],
                value=ast.Call(
                    func=ast.Name(fn_name, ast.Load()),
                    args=[],
                    keywords=[],
                )
            ),
        ]
        for stmt in statements:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
            self.prefix.append(stmt)

        load = ast.Name(iter_name, ast.Load())
        ast.copy_location(load, node)
        return load
    """

    def visit_For(self, node):
        node.iter = self.to_tmp_visit(node.iter)
        fs = FlattenStatement(self)
        node.target = fs.to_tmp_visit(node.target)
        assert not fs.prefix
        # TODO(jansel): handle node.targets?
        node.body = fs.suffix + self._body(node.body)
        node.orelse = self._body(node.orelse)
        return node

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        node.test = node.test  # TODO(jansel): need handle the test
        node.body = self._body(node.body)
        node.orelse = self._body(node.orelse)
        return node

    def visit_With(self, node: (ast.With, ast.AsyncWith)):
        for item in node.items:
            item.context_expr = self.to_tmp_visit(item.context_expr)
        node.body = self._body(node.body)
        return node

    visit_AsyncWith = visit_With

    def visit_Try(self, node):
        node.body = self._body(node.body)
        for handler in node.handlers:
            handler.body = self._body(handler.body)
        node.orelse = self._body(node.orelse)
        node.finalbody = self._body(node.finalbody)
        return node

    # def visit_YieldFrom(self, node):
    # TODO(jansel): convert yieldfrom into yield


class Flatten(ast.NodeTransformer):
    """
    Simplify AST to remove nested expressions.

    a = b + c + foo()

    becomes
    _t_0 = b + c
    _t_1 = foo()
    a = _t_0 + _t_1
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
    visit_For = flatten_statement
    visit_AsyncFor = flatten_statement
    visit_While = flatten_statement
    visit_If = flatten_statement
    visit_Assert = flatten_statement
    visit_With = flatten_statement
    visit_AsyncWith = flatten_statement
    visit_Try = flatten_statement


class Replace(ast.NodeTransformer):
    def __init__(self, replacements: dict):
        super().__init__()
        self.replacements = replacements

    def visit_Name(self, node: ast.Name):
        if node.id in self.replacements:
            assert isinstance(node.ctx, ast.Load), f"{self.replacements} {ast.dump(node)}"
            return copy_locations_recursive(self.replacements[node.id], node)
        return node


class Rename(ast.NodeTransformer):
    def __init__(self, renames: dict):
        super().__init__()
        self.renames = renames

    def visit_Name(self, node: ast.Name):
        node.id = self.renames.get(node.id, node.id)
        return node


class OffsetLineno(ast.NodeTransformer):
    def __init__(self, offset):
        super(OffsetLineno, self).__init__()
        self.offset = offset

    def generic_visit(self, node: ast.AST):
        if hasattr(node, "lineno"):
            node.lineno = node.lineno + self.offset
            node.end_lineno = node.lineno + self.offset
        return super().generic_visit(node)
