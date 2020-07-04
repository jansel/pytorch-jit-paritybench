#!/usr/bin/env python3
import ast
import inspect
import logging
import os
import re
import subprocess
import tempfile
import types
import zipfile
from typing import TextIO, List

import torch
from astor import to_source
from torch.nn.parallel import DistributedDataParallel

from .deduce_parameters import DeduceParameters, DeduceParameter
from .reporting import Stats, ErrorAggregatorDict
from .static_analysis import ASTCleanup, ExtractReadsWrites, ExtractConfigUsage, CONFIG_NAMES, CheckCallableMembers, \
    split_import
from .static_analysis import IMPORT_WHITELIST
from .utils import call_with_timeout

log = logging.getLogger(__name__)

RUN_SCRIPT = False  # some scripts hang when run, so this causes many timeouts
NN_MODULE_RE = re.compile(r"(\btorch[.]nn\b)|(\bnn[.]Module\b)", re.MULTILINE)
PREFIX = f'''
from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import {', '.join(IMPORT_WHITELIST)}
import numpy as np

patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
{" = ".join(sorted(CONFIG_NAMES))} = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = "1.0.0"
'''
SUFFIX = '''
import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_{basename}(_paritybench_base):
    pass
'''

TESTCASE_TEMPLATE = '''    def test_{index:03}(self):
        self._check({script}, {args}, {kwargs})

'''


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(self, tempdir: str, errors: ErrorAggregatorDict, stats: Stats, output_py: TextIO, name_filter=None):
        super(PyTorchModuleExtractor, self).__init__()
        self.errors = errors
        self.stats = stats

        self.output = IncrementalModule(tempdir, output_py)

        self.imports = dict()
        self.constants = []
        self.module_statements = []

        self.available_symbols = dict()
        self.global_config = None

        self.testcases = []

        self.name_filter = name_filter

    def search_file(self, filename: str, open_fn=open):
        if not filename.endswith(".py") or '.#' in filename:
            return

        with open_fn(filename, 'r') as fp:
            source = fp.read()
            if isinstance(source, bytes):
                source = source.decode('utf-8')

        has_match = bool(NN_MODULE_RE.search(source))

        try:
            tree = self.ast_parse(source, filename)
        except Exception as e:
            return self.errors.record("parse", e)

        m = re.search(r"([a-z0-9_]+)/__init__.py$", filename, re.I)
        if m:
            self.output.add_module_alias(m.group(1), has_match)
        else:
            self.output.add_module_alias(os.path.splitext(os.path.basename(filename))[0], has_match)

        self.search_ast(tree, has_match)

    @staticmethod
    def ast_parse(source, filename):
        try:
            return ast.parse(source, filename)
        except SyntaxError:
            # perhaps python2?
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".py") as tmp:
                tmp.write(re.sub(r"\basync *=", "non_blocking=", source)
                            .replace("\t", "    ")
                            .encode('utf-8'))
                tmp.flush()
                with open("/dev/null", "w") as null:
                    subprocess.check_call(["2to3", "-w", tmp.name], stderr=null, stdout=null)
                return ast.parse(open(tmp.name).read(), filename)

    def search_ast(self, tree: ast.AST, overwrite: bool):
        scope = types.ModuleType("_scope")
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = [to_source(x).strip() for x in node.bases]
                if overwrite and any(self.is_torch_nn_module(scope, x) for x in bases):
                    self.module_statements.append(node)
                else:
                    self.add_available_symbol(node, overwrite)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if overwrite:
                    for module_name, import_node in split_import(node):
                        if module_name == "torch":
                            # Run torch imports so we can run issubclass(.., torch.nn.Module)
                            try:
                                exec(compile(ast.Module([import_node], []), "<string>", "exec"),
                                     scope.__dict__,
                                     scope.__dict__)
                            except Exception:
                                log.exception('Bad torch import')
                                continue
                        if module_name in IMPORT_WHITELIST:
                            self.imports[to_source(import_node)] = import_node

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign)):
                self.add_available_symbol(node, overwrite)

    @staticmethod
    def is_torch_nn_module(scope: types.ModuleType, base: str):
        if base in ('torch.nn.Module', 'nn.Module', 'Module'):
            return True
        try:
            for part in base.split('.'):
                scope = getattr(scope, part, object)
            return issubclass(scope, torch.nn.Module)
        except Exception:
            log.exception("Error in is_torch_nn_module()")

    def search_directory(self, filename: str):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def search_zipfile(self, filename: str):
        with zipfile.ZipFile(filename) as archive:
            for name in sorted(archive.namelist()):
                self.search_file(name, archive.open)

    def add_available_symbol(self, node, overwrite=False):
        try:
            if overwrite:
                self.available_symbols[node.name] = node
            else:
                self.available_symbols.setdefault(node.name, node)
        except AttributeError:  # node.name is missing
            reads, writes = ExtractReadsWrites.run(node)
            for name in writes:
                if overwrite:
                    self.available_symbols[name] = node
                else:
                    self.available_symbols.setdefault(name, node)

    def construct_module(self):
        self.output.run_statement(self.ast_parse(PREFIX, "<string>"), source_required=True)
        self.global_config = self.output.output_module.__dict__["_global_config"]
        self.name_to_ast = dict()

        for statement in self.imports.values():
            try:
                self.output.run_statement(statement)
            except Exception as e:
                self.errors.record("import", e, "")
        for statement in self.constants:
            try:
                self.output.run_statement(statement)
            except Exception as e:
                self.errors.record("constant", e, getattr(statement, "name", ""))
        for statement in self.module_statements:
            try:
                self.add_requirements(statement)
                statement = ast.fix_missing_locations(ASTCleanup().visit(statement))
                self.output.run_statement(statement, source_required=True)
                self.name_to_ast[statement.name] = statement
            except Exception as e:
                self.errors.record("define", e, getattr(statement, "name", ""))

    def add_requirements(self, statement):
        """
        Recursively add symbols to the output module needed by statement.

        :param statement: ast.Node to add to the module
        """
        reads, writes = ExtractReadsWrites.run(statement)
        need_config = False
        for name in sorted(reads - writes):
            if name in self.available_symbols and name not in self.output:
                requirement = self.available_symbols.pop(name)
                self.add_requirements(requirement)
                try:
                    self.output.run_statement(requirement, source_required=True)
                except:
                    log.warning("Error adding requirement", exc_info=True)
            elif name in CONFIG_NAMES:
                need_config = True

        if need_config:
            try:
                for key in ExtractConfigUsage.run(statement):
                    if key not in sorted(self.global_config):
                        value = repr(DeduceParameter.initial_arg_init(key, None))
                        self.output.run_statement(
                            self.ast_parse(f"_global_config['{key}'] = {value}\n", "<string>"),
                            source_required=True)
            except Exception:
                log.exception("global_config error")

    def test_modules(self):
        for name, value in list(sorted(self.output.items())):
            if self.should_test_cls(value):
                self.test_nn_module(name, value)

    def should_test_cls(self, cls):
        if not isinstance(cls, type):
            return False
        if not issubclass(cls, torch.nn.Module):
            return False
        if issubclass(cls, DistributedDataParallel):
            return False
        return self.output.same_module(cls)

    def test_nn_module(self, name: str, nn_cls: type):
        if self.name_filter and self.name_filter not in name:
            return

        self.stats["total"] += 1
        checker = CheckCallableMembers.run(self.name_to_ast.get(name))
        try:
            stats, errors, testcases = call_with_timeout(
                extract_nn_module,
                args=(name, nn_cls, checker, self.errors.context),
                timeout=300)
            self.errors.update(errors)
            self.stats.update(stats)
            self.testcases.extend(testcases)
        except OSError:
            self.stats["module_crash"] += 1
        except TimeoutError:
            self.stats["module_timeout"] += 1

    def main(self, filename: str):
        basename = re.sub(r"[.]zip$", "", os.path.basename(filename))

        self.output.writelines([
            "import sys\n",
            "_module = sys.modules[__name__]\n",
            "del sys\n"])

        if os.path.isdir(filename):
            self.search_directory(filename)
        else:
            self.search_zipfile(filename)

        self.construct_module()
        self.test_modules()
        self.write_testcases(basename)

        log.info(f"{basename}: {self.stats}")

    def write_testcases(self, basename):
        self.output.write(SUFFIX.format(basename=basename))
        index = 0
        for name, init_args, forward_args, compiles in self.testcases:
            script = f"{name}(*{init_args[0]}, **{init_args[1]})"
            args, kwargs = forward_args
            if kwargs:
                if not compiles:
                    self.output.write("    @_fails_compile()\n")
                self.output.write(TESTCASE_TEMPLATE.format(
                    index=index,
                    script=script,
                    args=args,
                    kwargs=kwargs,
                ))

            index += 1


def extract_nn_module(name: str, nn_cls: type, checker, context):
    errors = ErrorAggregatorDict(context)
    stats = Stats()
    testcases = []
    extract_nn_module_inner(name, nn_cls, checker, stats, errors, testcases)
    return stats, errors, testcases


def extract_nn_module_inner(name: str, nn_cls: type, checker, stats, errors, testcases):
    init_signature = inspect.signature(nn_cls)
    try:
        init_deducer = DeduceParameters(
            nn_cls,
            *DeduceParameters.initial_args_init(init_signature),
            checker=checker.check)
        init_deducer.search()
        nn_module = init_deducer.last_result
    except Exception as e:
        return errors.record('init', e, nn_cls)

    try:
        nn_module.eval()
    except:
        pass

    stats["init_ok"] += 1

    forward_signature = inspect.signature(nn_module.forward)
    try:
        forward_deducer = DeduceParameters(
            nn_module,
            *DeduceParameters.initial_args_forward(forward_signature))
        forward_deducer.search()
    except Exception as e:
        return errors.record('deduce', e, nn_cls)

    stats["deduced_args_ok"] += 1

    try:
        torch.jit.script(nn_module)
    except Exception as e:
        testcases.append((
            name,
            init_deducer.testcase_args(),
            forward_deducer.testcase_args(),
            False
        ))

        return errors.record('compile', e, nn_cls)

    stats["jit_compiles"] += 1

    testcases.append((
        name,
        init_deducer.testcase_args(),
        forward_deducer.testcase_args(),
        True
    ))


class IncrementalModule(object):
    """
    Construct a python module statement by statement, recording the result
    to a generated python file.
    """

    def __init__(self, tempdir: str, output_py: TextIO):
        super().__init__()
        self.tempdir = tempdir
        self.output_module = types.ModuleType(f"{__name__}.output")
        self.output_py = output_py

    def __contains__(self, name):
        """
        :param name: symbol to check for
        :return: True if output module contains name (and it is not an alias)
        """
        return getattr(self.output_module, name, self.output_module) is not self.output_module

    def items(self):
        return self.output_module.__dict__.items()

    def same_module(self, obj):
        """
        :param obj: a python object
        :return: True if obj is defined in this module
        """
        return obj.__module__ == self.output_module.__name__

    def write(self, data: str):
        self.output_py.write(data)

    def writelines(self, data: List[str]):
        self.output_py.writelines(data)

    def run_statement(self, statement, source_required=False):
        source = to_source(statement)
        if not source_required:
            code = compile(ast.Module([statement], []), "<string>", "exec")
        else:
            # TorchScript requires source code to exist on disk
            assert self.tempdir
            fn, filename = tempfile.mkstemp(suffix='.py', dir=self.tempdir, prefix="pb")
            with os.fdopen(fn, "w") as fd:
                fd.write(source)
                fd.flush()
            code = compile(source, filename, "exec")
        exec(code, self.output_module.__dict__, self.output_module.__dict__)
        self.output_py.writelines(["\n", source, "\n"])

    def add_module_alias(self, name: str, overwrite: bool):
        """
        We flatten everything we extract into a single module, this adds
        a symbol to that unified module that points to the same module
        so that internal a.b.c references work.

        :param name: alternate name for self.output_module
        :param overwrite: if true, replace an existing symbol
        """
        if name in {'global', 'try', 'except', 'if', 'in', 'else', 'for', 'return', 'def'}:
            return
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return
        if name in self.output_module.__dict__ and not overwrite:
            return
        self.output_module.__dict__[name] = self.output_module
        self.output_py.write(f"{name} = _module\n")
