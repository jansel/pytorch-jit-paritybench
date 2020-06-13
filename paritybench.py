#!/usr/bin/env python3
import argparse
import ast
import csv
import inspect
import itertools
import json
import logging
import multiprocessing
import os
import random
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
import zipfile
from collections import Counter, defaultdict
from functools import reduce
from multiprocessing.pool import ThreadPool
from typing import List, Callable, TextIO
from unittest.mock import patch

import requests
from astor import to_source

import torch

log = logging.getLogger(__name__)

RUN_SCRIPT = False  # some scripts hang when run, so this causes many timeouts
NN_MODULE_RE = re.compile(r"(\btorch[.]nn\b)|(\bnn[.]Module\b)", re.MULTILINE)
IMPORT_WHITELIST = {
    # TODO: torchvision/torchaudio/etc is used by many
    "abc",
    "collections",
    "copy",
    "enum",
    "functools",
    "inspect",
    "itertools",
    "logging",
    "math",
    "numpy",
    "random",
    "re",
    "scipy",
    "string",
    "torch",
    "types",
    "typing",
    "uuid",
    "warnings",
}
CONFIG_NAMES = {"argv", "args", "config", "cfg", "params", "_global_config"}
PREFIX = f'''
from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module

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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_{basename}(_paritybench_base):
    pass
'''

PARITYBENCH_HELPERS = '''
import torch, unittest, copy, os
from torch.testing._internal.jit_utils import JitTestCase


def _mock_layer(in_features=None, out_features=None, bias=True):
    if in_features and out_features:
        return torch.nn.Linear(in_features, out_features, bias)
    return torch.nn.ReLU()


class _mock_config(dict):
    __getattr__ = dict.__getitem__


def _fails_compile():
    if os.environ.get('TEST_ALL'):
        return lambda x: x
    return unittest.skip("jit compile fails")


class _paritybench_base(JitTestCase):
    def _check(self, script, args, kwargs):
        try:
            script.eval()
        except:
            pass
        result1 = script(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        result2 = script(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        if os.environ.get('TEST_PY_ONLY'):
            return
        jit_script = torch.jit.script(script)
        if os.environ.get('TEST_COMPILE_ONLY'):
            return
        result3 = jit_script(*args, **kwargs)
        if os.environ.get('TEST_RUN_ONLY'):
            return
        try:
            self.assertEqual(result1, result2)
        except AssertionError:
            return  # output is not deterministic
        self.assertEqual(result2, result3)
'''

TESTCASE_TEMPLATE = '''    def test_{index:03}(self):
        self._check({script}, {args}, {kwargs})

'''


class Stats(Counter):
    """
    Collect and group error messages for a debug report at the end
    """

    def __str__(self):
        """
        Reorder key print order by stage in the process
        """
        stats_keys = [
            "total",
            "init_ok",
            "deduced_args_ok",
            "jit_compiles",
        ]
        stats_keys = stats_keys + list(set(self.keys()) - set(stats_keys))
        return str([(k, self[k]) for k in stats_keys])


class ErrorAggregator(object):
    """
    Collect and group error messages for report at the end
    """

    def __init__(self, context=None, log=None):
        super(ErrorAggregator, self).__init__()
        if context:
            self.context = re.sub(r"\.zip$", ".py", context)
        else:
            self.context = ""
        self.error_groups = []
        self.bigram_to_group_ids = defaultdict(list)
        self.log = log or logging.getLogger(__name__)

    def record(self, e: Exception, module):
        ex_msg = str(e).strip().split('\n')[0]
        error_msg = f"{e.__class__.__name__}: {ex_msg}"
        full_msg = f"{e.__class__.__name__}: {str(e)}"
        return self._add(error_msg, [(error_msg, f"{self.context}:{module}", full_msg)])

    def update(self, other):
        for errors in other.error_groups:
            self._add(errors[0][0], errors)

    def _add(self, error_msg: str, errors: List):
        msg_words = list(re.findall(r"[a-zA-Z]+", error_msg))
        if "NameError" in error_msg:
            msg_bigrams = [error_msg]  # need exact match
        else:
            msg_bigrams = [f"{a}_{b}" for a, b in zip(msg_words, msg_words[1:])] or msg_words

        shared_bigrams = Counter()
        for bigram in msg_bigrams:
            shared_bigrams.update(self.bigram_to_group_ids[bigram])

        if shared_bigrams:
            best_match, count = shared_bigrams.most_common(1)[0]
            if count > len(msg_bigrams) // 2:
                self.error_groups[best_match].extend(errors)
                return False

        # No match, create a new error group
        group_id = len(self.error_groups)
        self.error_groups.append(errors)
        for bigram in msg_bigrams:
            self.bigram_to_group_ids[bigram].append(group_id)

        return True

    @staticmethod
    def format_error_group(errors):
        context, context_count = random.choice(list(Counter(context for msg, context, _ in errors).items()))
        return f"  - {len(errors)} errors like: {errors[0][0]} (example {context})"

    def __str__(self):
        errors = sorted(self.error_groups, key=len, reverse=True)
        return '\n'.join(map(self.format_error_group, errors[:20]))

    def __len__(self):
        return sum(map(len, self.error_groups))

    csv_headers = ["phase", "count", "example_short", "example_long", "example_from"]

    def write_csv(self, phase, out: csv.writer):
        for errors in sorted(self.error_groups, key=len, reverse=True)[:20]:
            short, context, long = random.choice(errors)
            out.writerow([phase, len(errors), short, long, context])


class ErrorAggregatorDict(object):
    """
    Collect and group error messages for a debug report at the end
    """

    @classmethod
    def single(cls, name: str, e: Exception, context=None):
        errors = cls(context)
        errors.record(name, e, 'global')
        return errors

    def __init__(self, context=None):
        super(ErrorAggregatorDict, self).__init__()
        self.aggregator = dict()
        self.context = context
        if context:
            self.name = re.sub(r"[.]zip$", "", os.path.basename(context))
        else:
            self.name = __name__

    def __getitem__(self, item):
        if item not in self.aggregator:
            self.aggregator[item] = ErrorAggregator(self.context, logging.getLogger(f"{item}.{self.name}"))
        return self.aggregator[item]

    def update(self, other):
        for key, value in other.aggregator.items():
            self[key].update(other=value)

    def print_report(self):
        for name in sorted(list(self.aggregator.keys())):
            self[name].log.info(f"\nTop errors in {name} ({len(self[name])} total):\n{self[name]}\n")

        with open('errors.csv', "w") as fd:
            out = csv.writer(fd)
            out.writerow(ErrorAggregator.csv_headers)
            for name in sorted(list(self.aggregator.keys())):
                self[name].write_csv(name, out)

    def record(self, error_type, error, module=None):
        module = str(getattr(module, "__name__", module))
        if self[error_type].record(error, module):
            log.exception(f"{error_type} error from {self.context}:{module}")


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(self, tempdir: str, errors: ErrorAggregatorDict, stats: Stats, output_py: TextIO):
        super(PyTorchModuleExtractor, self).__init__()
        self.tempdir = tempdir
        self.errors = errors
        self.stats = stats

        self.output_module = types.ModuleType(f"{__name__}.output")
        self.output_py = output_py

        self.imports = dict()
        self.constants = []
        self.module_statements = []

        self.available_symbols = dict()
        self.testcases = []
        self.global_config = None

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
            self.add_module_alias(m.group(1), has_match)
        else:
            self.add_module_alias(os.path.splitext(os.path.basename(filename))[0], has_match)

        self.search_ast(tree, has_match)

    @staticmethod
    def ast_parse(source, filename):
        try:
            return ast.parse(source, filename)
        except SyntaxError:
            # perhaps python2?
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".py") as tmp:
                tmp.write(re.sub(r"\basync *=", "non_blocking=", source).encode('utf-8'))
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
                    for module_name, import_node in self.split_import(node):
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

    @staticmethod
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

    def construct_module(self):
        self.run_statement(self.ast_parse(PREFIX, "<string>"), source_required=True)
        self.global_config = self.output_module.__dict__["_global_config"]

        for statement in self.imports.values():
            try:
                self.run_statement(statement)
            except Exception as e:
                self.errors.record("import", e, "")
        for statement in self.constants:
            try:
                self.run_statement(statement)
            except Exception as e:
                self.errors.record("constant", e, getattr(statement, "name", ""))
        for statement in self.module_statements:
            try:
                self.add_requirements(statement)
                statement = ast.fix_missing_locations(ASTCleanup().visit(statement))
                self.run_statement(statement, source_required=True)
            except Exception as e:
                self.errors.record("define", e, getattr(statement, "name", ""))

    def add_requirements(self, statement):
        """
        Recursively add symbols to the output module needed by statement.

        :param statement: ast.Node to add to the module
        """
        reads, writes = ExtractReadsWrites.run(statement)
        need_config = False
        for name in reads - writes:
            if (name in self.available_symbols and
                    getattr(self.output_module, name, self.output_module) is self.output_module):
                requirement = self.available_symbols.pop(name)
                self.add_requirements(requirement)
                self.run_statement(requirement, source_required=True)
            elif name in CONFIG_NAMES:
                need_config = True

        if need_config:
            try:
                for key in ExtractConfigUsage.run(statement):
                    if key not in self.global_config:
                        value = repr(DeduceParameter.initial_arg_init(key, None))
                        self.run_statement(self.ast_parse(f"_global_config['{key}'] = {value}\n", "<string>"),
                                           source_required=True)
            except Exception:
                log.exception("global_config error")

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

    def test_modules(self):
        for name, value in list(sorted(self.output_module.__dict__.items())):
            if (isinstance(value, type) and
                    issubclass(value, torch.nn.Module) and
                    value.__module__ == self.output_module.__name__):
                self.test_nn_module(name, value)

    def test_nn_module(self, name: str, nn_cls: type):
        self.stats["total"] += 1

        init_signature = inspect.signature(nn_cls)
        try:
            init_deducer = DeduceParameters(
                nn_cls,
                *DeduceParameters.initial_args_init(init_signature))
            init_deducer.search()
            nn_module = init_deducer.last_result
        except Exception as e:
            return self.errors.record('init', e, nn_cls)

        try:
            nn_module.eval()
        except:
            pass

        self.stats["init_ok"] += 1

        forward_signature = inspect.signature(nn_module.forward)
        try:
            forward_deducer = DeduceParameters(
                nn_module,
                *DeduceParameters.initial_args_forward(forward_signature))
            forward_deducer.search()
            args = forward_deducer.last_args
            kwargs = forward_deducer.last_kwargs
            python_output = forward_deducer.last_result
        except Exception as e:
            return self.errors.record('deduce', e, nn_cls)

        self.stats["deduced_args_ok"] += 1

        try:
            script = torch.jit.script(nn_module)
        except Exception as e:
            self.testcases.append((
                name,
                init_deducer.testcase_args(),
                forward_deducer.testcase_args(),
                False
            ))

            return self.errors.record('compile', e, nn_cls)

        self.stats["jit_compiles"] += 1

        self.testcases.append((
            name,
            init_deducer.testcase_args(),
            forward_deducer.testcase_args(),
            True
        ))

        if not RUN_SCRIPT:
            return

        try:
            script_output = script(*args, **kwargs)
        except Exception as e:
            return self.errors.record('run', e, nn_cls)

        try:
            # JitTestCase().checkScript(nn_module, args)  doesn't work
            self.assertEqual(script_output, python_output)
        except Exception as e:
            return self.errors.record('output', e, nn_cls)

        self.stats["jit_correct"] += 1

    def assertEqual(self, a, b):
        # TODO(jansel): find/reuse an existing version of this
        tc = unittest.TestCase()
        if isinstance(a, torch.Tensor):
            tc.assertTrue(torch.allclose(a, b))
        elif isinstance(a, (list, tuple)):
            tc.assertEqual(len(a), len(b))
            for a_, b_ in zip(a, b):
                self.assertEqual(a_, b_)
        elif isinstance(a, dict):
            tc.assertEqual(set(a.keys()), set(b.keys()))
            for key in a.keys():
                self.assertEqual(a[key], b[key])
        else:
            tc.assertEqual(a, b)

    def main(self, filename: str):
        basename = re.sub(r"[.]zip$", "", os.path.basename(filename))

        self.output_py.writelines([
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
        self.output_py.write(SUFFIX.format(basename=basename))
        index = 0
        for name, init_args, forward_args, compiles in self.testcases:
            script = f"{name}(*{init_args[0]}, **{init_args[1]})"
            args, kwargs = forward_args
            if kwargs:
                if not compiles:
                    self.output_py.write("    @_fails_compile()\n")
                self.output_py.write(TESTCASE_TEMPLATE.format(
                    index=index,
                    script=script,
                    args=args,
                    kwargs=kwargs,
                ))

            index += 1


class DeductionFailed(RuntimeError):
    def __init__(self, attempt_log, name='', traceback=''):
        attempt_lines = "\n".join(f" - {attempt}" for attempt in attempt_log)
        error_msg = f"{attempt_log[-1][1]}\n{name}:\n{attempt_lines}\n----\n{traceback}\n----\n"
        super().__init__(error_msg)


class DeduceParameters(object):
    """
    Try to figure out a valid input for an NN module by repeated
    guessing based on error messages.
    """
    default_size = 4

    @staticmethod
    def needed_args(signature: inspect.Signature):
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue  # ignore args/kwargs
            if param.default is inspect.Parameter.empty:
                yield param

    @classmethod
    def initial_args_init(cls, signature: inspect.Signature = None):
        return [], {param.name: DeduceParameter.initial_arg_init(param, position)
                    for position, param in enumerate(cls.needed_args(signature))}

    @classmethod
    def initial_args_forward(cls, signature: inspect.Signature):
        return [DeduceParameter.initial_arg_forward(param, position)
                for position, param in enumerate(cls.needed_args(signature))], {}

    def __init__(self, nn_module: Callable, args: list, kwargs: dict):
        super(DeduceParameters, self).__init__()
        self.nn_module = nn_module
        self.args = args
        self.kwargs = kwargs
        self.tried = set()

        self.attempt_log = []
        self.last_args = None
        self.last_kwargs = None
        self.last_result = None
        self.last_traceback = None

    def __str__(self):
        return ", ".join(itertools.chain(
            map(str, self.args),
            [f"{name}={arg}" for name, arg in self.kwargs.items()]))

    def testcase_args(self):
        args = repr(self.args)
        kwargs = repr(self.kwargs)
        return args, kwargs

    @classmethod
    def run(cls, nn_module: Callable, needed_args: List[inspect.Parameter]):
        return DeduceParameters(nn_module, needed_args).search()

    def search_once(self):
        self.last_args = [arg.guess() for arg in self.args]
        self.last_kwargs = {name: arg.guess() for name, arg in self.kwargs.items()}
        guess_str = str(self)
        self.tried.add(guess_str)

        try:
            self.last_result = self.nn_module(*self.last_args, **self.last_kwargs)
            return True
        except Exception:
            error_type, error_value, tb = sys.exc_info()
            error_msg = f"{error_type.__name__}: {error_value}"
            sorted_args = self.sorted_args(tb)
            self.last_traceback = traceback.format_exc(-2)

        self.attempt_log.append((guess_str, error_msg))

        if Guess.apply_fixors(self.get_fixors(), error_msg):
            if str(self) not in self.tried:
                return False

        for pass_number in (0, 1, 2):
            for arg in sorted_args:
                if arg.try_to_fix(error_msg, pass_number):
                    if str(self) not in self.tried:
                        return False
                    arg.rollback()

        raise DeductionFailed(self.attempt_log, str(type(self.nn_module)), self.last_traceback)

    def all_args(self):
        return list(self.args) + list(self.kwargs.values())

    def sorted_args(self, trackback) -> List:
        """
        Order args by when they are seen in the traceback so we can fix
        relevant to the error args first.

        :param trackback: from sys.exc_info()
        :return: parameters ordered by where they are seen in the traceback
        """
        this_file = os.path.basename(__file__)
        args = self.all_args()
        # args.sort(lambda x: x.num_guesses())
        for frame in reversed(traceback.extract_tb(trackback, limit=-10)):
            if this_file in frame.filename:
                break
            line = frame.line
            args.sort(key=lambda x: x.contained_in_line(line), reverse=True)
        return args

    def search(self, limit: int = 10):
        try:
            for attempt in range(limit):
                if self.search_once():
                    return self.last_result
        except DeductionFailed:
            pass  # ignore error as we will try again

        for arg in self.all_args():
            arg.alt_guess()

        if str(self) not in self.tried:
            for attempt in range(limit):
                if self.search_once():
                    return self.last_result

        raise DeductionFailed(self.attempt_log, str(type(self.nn_module)), self.last_traceback)

    def get_fixors(self):
        return [
            (r"missing.*arguments?: (?P<name>['][a-zA-Z0-9_]+['])",
             self.fix_missing_arg),
            (r"unexpected keyword argument (?P<name>['][a-zA-Z0-9_]+['])",
             self.fix_extra_arg),
            (r"size mismatch, m1: (?P<a>\[.*\]), m2: (?P<b>\[.*\])",
             self.fix_size_mismatch),
            (r"shape (?P<a>\[[\d, ]+\]) doesn't match the broadcast shape (?P<b>\[[\d, ]+\])]",
             self.fix_size_mismatch),
        ]

    def fix_size_mismatch(self, a, b):
        matches_a = [arg for arg in self.all_args() if arg.is_shape_match(a)]
        matches_b = [arg for arg in self.all_args() if arg.is_shape_match(b)]
        if not matches_a and not matches_b:
            matches_a = [arg for arg in self.all_args() if arg.is_element_count_match(a)]
            matches_b = [arg for arg in self.all_args() if arg.is_element_count_match(b)]

        if matches_a and matches_b:
            if max(x.created for x in matches_a) > max(x.created for x in matches_b):
                # prefer changing the old one
                matches_a, matches_b = matches_b, matches_a
            guess_a = min(matches_a, key=lambda x: x.created)
            guess_b = max(matches_b, key=lambda x: x.created)
            guess_a.change_guess(guess_b.clone_guess())
            return True

        if matches_b:
            matches_a, matches_b = matches_b, matches_a
            a, b = b, a

        if matches_a:
            guess = min(matches_a, key=lambda x: x.created)
            guess.change_guess(TensorGuess(shape=b))
            return True

    def fix_missing_arg(self, name):
        if any(arg.name == name for arg in self.args):
            return
        if name in self.kwargs:
            # try moving it to args?
            self.args.append(self.kwargs.pop(name))
            self.args.sort(key=lambda x: x.position)
        else:
            self.kwargs[name] = DeduceParameter.initial_arg_init(name, float('inf'))
        return True

    def fix_extra_arg(self, name):
        if name in self.kwargs:
            del self.kwargs[name]
            return True


class DeduceParameter(object):
    """
    Contains and tracks the guesses of the value of a single parameter.
    """

    @classmethod
    def initial_arg_init(cls, name, position):
        name = getattr(name, 'name', name)
        if 'dataset' in name:
            # TODO: this likely wont work...
            return cls.initial_arg_forward(name, position)

        common_args = {
            "stride": 1,
            "scale": 1.0,
            "layer": 1,
            "dilation": 1,
            "groups": 1,
            "block": 1,
            "depth": 1,
            "gpu": False,
            "train": False,
            "loss": torch.nn.MSELoss(),
            "dropout": 0.5,
            "drop_rate": 0.5,
        }
        for search_name, placeholder_value in common_args.items():
            if search_name in name:
                return cls(name, position, LiteralGuess(placeholder_value))

        for search_name in ('cfg', 'config', 'options', 'args'):
            if search_name in name:
                return cls(name, position, ConfigGuess())

        return cls(name, position, LiteralGuess(DeduceParameters.default_size))

    @classmethod
    def initial_arg_forward(cls, name, position=None, dims=4):
        name = getattr(name, 'name', name)
        return cls(name, position, TensorGuess([TensorGuess.default_size] * dims))

    def __init__(self, name: str, position, initial_guess):
        super(DeduceParameter, self).__init__()
        self.name = name
        self.position = position
        self._guesses = [initial_guess]

    def __str__(self):
        val = str(self._guesses[-1])
        if val.startswith('<function _mock_layer'):
            # TODO: workaround, fix this better
            return '_mock_layer'
        return val

    __repr__ = __str__

    @property
    def created(self):
        return self._guesses[-1].created

    def guess(self):
        return self._guesses[-1].guess()

    def clone_guess(self):
        return self._guesses[-1].clone()

    def num_guesses(self):
        return len(self._guesses)

    def try_to_fix(self, error_message: str, pass_number: int) -> bool:
        new_guess = self._guesses[-1].get_fix(error_message, pass_number, self.name)
        if new_guess is not None:
            self.change_guess(new_guess)
            return True
        return False

    def change_guess(self, guess):
        self._guesses.append(guess)

    def contained_in_line(self, line: str):
        pattern = r"\b{}\b".format(re.escape(self.name))
        return bool(re.search(pattern, line))

    def rollback(self):
        self._guesses[-1].rollback()
        self._guesses.pop()

    def is_shape_match(self, shape):
        if isinstance(self._guesses[-1], TensorGuess):
            return self._guesses[-1].shape == shape

    def is_element_count_match(self, shape):
        if isinstance(self._guesses[-1], TensorGuess):
            count = reduce(lambda a, b: a * b, shape)
            this_count = reduce(lambda a, b: a * b, self._guesses[-1].shape)
            return count == this_count

    def alt_guess(self):
        if isinstance(self._guesses[-1], TensorGuess):
            # try starting with a smaller size
            self.change_guess(TensorGuess([TensorGuess.default_size] * 2))


class Guess(object):
    """
    Base class of a single guess of the value of parameter.
    """

    def __init__(self, value=None):
        super(Guess, self).__init__()
        self.value = value
        self.created = time.time()

    def __str__(self):
        val = repr(self.value)
        if '_mock_layer' in val:
            return '_mock_layer'
        return val

    __repr__ = __str__

    @staticmethod
    def apply_fixors(fixors, error_msg):
        for pattern, fixor in fixors:
            match = re.search(pattern, error_msg, flags=re.I)
            if match:
                fix = fixor(**{k: Guess.literal(v) for k, v in match.groupdict().items()})
                if fix is not None:
                    log.debug(f"FIX: {fixor.__name__} {error_msg}")
                    return fix

    @staticmethod
    def literal(value):
        # [1 x 1] => [1, 1]
        return ast.literal_eval(value.replace(" x ", ","))

    def guess(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def get_fix(self, error_message: str, pass_number: int, name: str):
        pass

    def rollback(self):
        pass


def _mock_layer(in_features=None, out_features=None, bias=True):
    if in_features and out_features:
        return torch.nn.Linear(in_features, out_features, bias)
    return torch.nn.ReLU()


class LiteralGuess(Guess):
    def get_fix(self, error_message: str, pass_number: int, name: str):
        fix = super(LiteralGuess, self).get_fix(error_message, pass_number, name)
        if fix:
            return fix

        if pass_number > 0:
            return

        fixors = []

        if isinstance(self.value, int):
            def fix_too_small():
                if self.value < 64:
                    return 64

            fixors.extend([
                (r"TypeError: cannot unpack non-iterable int object",
                 lambda: [TensorGuess.default_size] * 2),
                (r"TypeError: 'int' object is not subscriptable",
                 lambda: [TensorGuess.default_size] * 2),
                (r"TypeError: 'int' object is not iterable",
                 lambda: [TensorGuess.default_size] * 2),
                (r"TypeError: object of type 'int' has no len()",
                 lambda: [TensorGuess.default_size] * 2),
                (r"AttributeError: 'int' object has no attribute '(size|shape|dtype|device|ndim)'",
                 lambda: TensorGuess([TensorGuess.default_size] * 2)),
                (r"TypeError: int is not a Module subclass",
                 lambda: torch.nn.ReLU()),
                (r"DeductionFailed: TypeError: 'int' object is not callable",
                 lambda: _mock_layer),
                (r"ModuleList.extend should be called with an iterable, but got int",
                 lambda: [torch.nn.ReLU()]),
                (r"ModuleDict.update should be called.*but got int",
                 lambda: {'relu': torch.nn.ReLU()}),
                (r"AttributeError: 'int' object has no attribute 'split'",
                 lambda: "2,2"),
                (r"IndexError: index 0 is out of bounds for dimension 0 with size 0",
                 fix_too_small),
                (r"ValueError: .* must be divisible by groups",
                 fix_too_small),
                (r"KeyError: [1-9]",
                 lambda: self.value // 2),
                (r"multiple of (?P<m>\d{1,3})",
                 lambda m: m),
            ])

        if isinstance(self.value, list):
            def fix_too_many(want):
                if len(self.value) > want:
                    return [TensorGuess.default_size] * want

            def fix_too_few(want, got):
                if len(self.value) == got:
                    return [TensorGuess.default_size] * want

            fixors.extend([
                (r"ValueError: too many values to unpack \(expected (?P<want>\d+)\)",
                 fix_too_many),
                (r"ValueError: not enough values to unpack \(expected (?P<want>\d+), got (?P<got>\d+)\)",
                 fix_too_few),
                (r"not supported between instances of '(list|int)' and '(list|int)'",
                 lambda: TensorGuess.default_size),
                (r"TypeError: 'list' object is not callable",
                 lambda: _mock_layer),
                (r"IndexError: list index out of range",
                 lambda: self.value + [TensorGuess.default_size]),
            ])

        new_value = self.apply_fixors(fixors, error_message)
        if new_value is not None:
            if isinstance(new_value, Guess):
                return new_value
            return LiteralGuess(new_value)

    def fix_not_subscriptable(self, typename="int"):
        if typename == "int" and isinstance(self.value, int):
            return [TensorGuess.default_size] * 2


class TensorGuess(Guess):
    default_size = DeduceParameters.default_size

    def __init__(self, shape, dtype=torch.float32):
        super(TensorGuess, self).__init__()
        assert isinstance(shape, list)
        assert all(isinstance(x, int) for x in shape)
        self.shape = shape
        self.dtype = dtype
        if self.dtype == torch.int64:
            # used for embedding lookups often
            self.value = torch.zeros(self.shape, dtype=self.dtype)
        else:
            self.value = torch.rand(self.shape, dtype=self.dtype)

    def clone(self):
        return self.__class__(self.shape, self.dtype)

    def __str__(self):
        if self.dtype == torch.float32:
            return f"torch.rand({self.shape})"
        elif self.dtype == torch.int64:
            return f"torch.zeros({self.shape}, dtype={self.dtype})"
        else:
            return f"torch.rand({self.shape}, dtype={self.dtype})"

    __repr__ = __str__

    def get_fix(self, error_message: str, pass_number: int, name: str):
        fix = super(TensorGuess, self).get_fix(error_message, pass_number, name)
        if fix:
            return fix

        new_shape = self.apply_fixors(self.shape_fixors(pass_number), error_message)
        if new_shape:
            return self.__class__(new_shape, self.dtype)

        def tried_to_call():
            keywords = ("layer", "activation", "dropout", "normalization")
            for keyword in keywords:
                if keyword in name:
                    return LiteralGuess(torch.nn.ReLU())

        other_fixors = [
            (r"scalar type Long; but got torch.FloatTensor",
             lambda: self.__class__([self.default_size], torch.int64)),
            (r"Expected dtype int64 for index",
             lambda: self.__class__([self.default_size], torch.int64)),
            (r"TypeError: [']Tensor['] object is not callable",
             tried_to_call),
            (r"TypeError: only integer tensors of a single element can be converted to an index",
             lambda: LiteralGuess(0)),
            (r"'lengths' argument should be a 1D CPU int64 tensor",
             lambda: (self.__class__([self.default_size], torch.int64) if "len" in name else None)),
            (r"Boolean value of Tensor with more than one value is ambiguous",
             lambda: LiteralGuess(0)),
        ]
        return self.apply_fixors(other_fixors, error_message)

    def shape_fixors(self, pass_number: int):
        if pass_number == 0:
            return [
                (r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_if_matching),
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_if_matching),
                (r"(?P<want>\d+) channels, but got (?P<got>\d+) channels",
                 self.fix_num_channels),
                (r"same number of dimensions: got (?P<want>\d+) and (?P<got>\d+)",
                 self.fix_dimensions),
                (r"Got (?P<got>\d+)D .*needs (?P<want>\d+)D",
                 self.fix_dimensions),
                (r"input must have (?P<want>\d+) dimensions, got (?P<got>\d+)",
                 self.fix_dimensions),
                (r"Expected (?P<want>\d+)-dimensional tensor, but got (?P<got>\d+)-dimensional tensor",
                 self.fix_dimensions),
                (r"The size.*[(](?P<want>\d+)[)] must match.*[(](?P<got>\d+)[)] at.*dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"must match except in dimension \d+. Got (?P<want>\d+) and (?P<got>\d+) in dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"matrices expected, got (?P<got>\d+)D, (?P<want>\d+)D ",
                 self.fix_dimensions),
                (r"expected \d+D or (?P<want>\d+)D input \(got (?P<got>\d+)D input\)",
                 self.fix_dimensions),
                (r"Expected.*size (?P<want>[\d, ()]+), got (?P<got>[\d, ()]+)",
                 self.fix_shape),
                (r"Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
                 lambda: self.shape[:-1]),
                (r"Expected tensor to have size (?P<want>\d+) at dimension (?P<dim>\d+), but got size (?P<got>\d+)",
                 self.fix_dimensions_at),
                (r"RuntimeError: number of dims don't match in permute",
                 self.fix_dimensions_unknown),
                (r"ValueError: too many values to unpack \(expected (?P<want>\d+)\)",
                 self.fix_dimensions),
                (r"ValueError: not enough values to unpack \(expected (?P<want>\d+), got (?P<got>\d+)\)",
                 self.fix_dimensions),
                (r"expected to be in range of \[-\d+, (?P<got>\d+)\], but got (?P<want>\d+)",
                 self.fix_dimension_out_of_range),
                (r"sizes provided \((?P<want>\d+)\) must be greater or equal.* \((?P<got>\d+)\)",
                 self.fix_dimensions),
                (r"size mismatch, m1: (?P<a>\[.*\]), m2: (?P<b>\[.*\])",
                 self.fix_size_mismatch),
            ]
        if pass_number == 1:
            return [
                (r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input\[[\d, ]+\]",
                 self.fix_convolution),
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*\[[\d, ]+\]",
                 self.fix_convolution),
                (r"same number of dimensions: got (?P<got>\d+) and (?P<want>\d+)",
                 self.fix_dimensions),
                (r"Got \d+D .*needs (?P<want>\d+)D",
                 self.fix_dimensions),
                (r"The size.*[(](?P<got>\d+)[)] must match.*[(](?P<want>\d+)[)] at.*dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"must match except in dimension \d+. Got (?P<got>\d+) and (?P<want>\d+) in dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"expected (?P<want>\d+)D or \d+D input \(got (?P<got>\d+)D input\)",
                 self.fix_dimensions),
                (r"Expected tensor to have size (?P<got>\d+) at dimension (?P<dim>\d+), but got size (?P<want>\d+)",
                 self.fix_dimensions_at),
            ]
        if pass_number == 2:
            return [
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_offset),
            ]

    def fix_size_mismatch(self, a, b):
        # b may be hidden part of a nn.Linear()
        if self.shape[-1] == a[-1]:
            return self.shape[:-1] + [b[0]]

    def fix_dimension_out_of_range(self, got, want):
        if 0 <= got < want:
            return self.fix_dimensions(want + 1, got + 1)

    def fix_shape(self, want, got):
        if self.shape == list(got):
            return list(want)

    def fix_convolution(self, weight: List[int], groups: int = 1):
        return [self.default_size, weight[1] * groups] + [64 for _ in weight[2:]]

    def fix_convolution_if_matching(self, weight, got, groups=1):
        if got == self.shape:
            return self.fix_convolution(weight, groups)

    def fix_convolution_offset(self, weight: List[int], got: List[int]):
        if len(got) == len(self.shape) - 1:
            return [self.default_size, self.default_size, weight[1]] + [64 for _ in weight[2:]]

    def fix_num_channels(self, want, got):
        guess = list(self.shape)
        for idx in (1, 2, 3):
            if len(guess) > idx and guess[idx] == got:
                guess[idx] = want
                return guess
        if len(guess) > 1:
            guess[1] = guess[1] * want // got
            if guess[1] > 0:
                return guess

    def fix_dimensions(self, want, got=None):
        shape = list(self.shape)
        if got is None or len(shape) == got:
            shape.extend([self.default_size] * want)
            return shape[:want]

    def fix_dimensions_at(self, want, got, dim):
        shape = list(self.shape)
        if dim < len(shape) and shape[dim] == got:
            shape[dim] = want
            return shape

    def fix_dimensions_unknown(self):
        shape = list(self.shape)
        if len(shape) > 2:
            shape.pop()
            return


class ConfigGuess(Guess):
    def __init__(self):
        super(ConfigGuess, self).__init__(value=MockConfig())
        self._rollback = []

    def get_fix(self, error_message: str, pass_number: int, name: str):
        guesses = sorted(self.value._guesses.values(),
                         key=lambda x: x.created, reverse=True)
        for guess in guesses:
            if guess.try_to_fix(error_message, pass_number):
                self._rollback.append(guess)
                return self

    def rollback(self):
        self._rollback.pop().rollback()


class MockConfig(object):
    def __init__(self):
        super(MockConfig, self).__init__()
        self._guesses = dict()

    def __str__(self):
        return "_mock_config({})".format(
            ", ".join(f"{key}={repr(value)}" for key, value in self._guesses.items())
        )

    def __getitem__(self, item):
        if item not in self._guesses:
            self._guesses[item] = DeduceParameter.initial_arg_init(name=item, position=None)
        return self._guesses[item].guess()

    __getattr__ = __getitem__

    def __iter__(self):
        return iter([])


class ASTCleanup(ast.NodeTransformer):
    def visit_Import(self, node):
        return ast.Constant(value=None, kind=None)

    visit_ImportFrom = visit_Import

    def visit_Call(self, node: ast.Call):
        if getattr(node.func, 'id', '') == 'print':
            # Strip print() calls
            return ast.Constant(value=None, kind=None)
        if getattr(node.func, 'attr', '') == 'cuda':
            # foo.cuda() => foo
            return node.func.value
        return self.generic_visit(node)


class ExtractReadsWrites(ast.NodeVisitor):
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


class CrawlGitHub(object):
    """
    Download projects from github with 100+ stars and the word "pytorch"
    """

    def __init__(self, download_dir):
        super(CrawlGitHub, self).__init__()
        self.download_dir = download_dir

    def github_search(self):
        base = "https://api.github.com/search/repositories?per_page=100&sort=stars"
        query = "pytorch+language:Python+stars:>100+size:<100000"
        seen = set()
        # both orders gets us 20 pages (past 10 limit), need 12 for current query
        for order in ("desc", "asc"):
            page = 1
            while True:
                time.sleep(6)  # https://developer.github.com/v3/search/#rate-limit
                rs = requests.get(f"{base}&page={page}&order={order}&q={query}")
                rs.raise_for_status()
                result = rs.json()
                assert not result['incomplete_results']
                for project in result["items"]:
                    name = project["full_name"]
                    if name not in seen:
                        seen.add(name)
                        yield project
                total_count = result['total_count']
                log.info(f"total_count={total_count} seen={len(seen)} page={page} {order}")
                page += 1
                if len(result["items"]) == 0 or len(seen) >= total_count:
                    return
                if page == 11:
                    break  # not allowed by API

    def download_project(self, project: dict):
        name = project["full_name"]
        url = project["html_url"]
        default_branch = project["default_branch"]
        output_filename = re.sub(r"[^a-zA-Z0-9]+", "_", name) + ".zip"
        output_path = os.path.join(self.download_dir, output_filename)
        if os.path.exists(output_path):
            return output_filename
        time.sleep(60)
        rs = requests.get(f"{url}/archive/{default_branch}.zip", stream=True)
        rs.raise_for_status()
        with open(output_path, "wb") as fd:
            for chunk in rs.iter_content(chunk_size=8192):
                fd.write(chunk)
        return output_filename

    def download(self):
        metadata_path = os.path.join(self.download_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            return

        os.path.exists(self.download_dir) or os.mkdir(self.download_dir)
        projects = list(self.github_search())
        metadata = dict()
        for i, project in enumerate(projects):
            log.info(f"Downloading {project['full_name']} ({i + 1} of {len(projects)})")
            metadata[self.download_project(project)] = project
        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)


def test_all(download_dir, limit=None):
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    zipfiles = [os.path.join(download_dir, f)
                for f in os.listdir(download_dir)
                if f.endswith(".zip")]

    if limit:
        zipfiles = zipfiles[:limit]
    pool = ThreadPool(8)
    for errors_part, stats_part in pool.imap_unordered(test_zipfile, zipfiles):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()
    errors.print_report()
    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds")


def test_zipfile(path):
    log.info(f"Running {path}")
    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        try:
            return call_with_timeout(test_zipfile_subproc, (tempdir, path), {}, timeout=120)
        except TimeoutError:
            return ErrorAggregatorDict.single(
                "meta",
                TimeoutError("Timeout testing module"),
                path
            ), Stats({"timeout": 1})
        except OSError:
            return ErrorAggregatorDict.single(
                "meta",
                OSError("Crash testing module"),
                path
            ), Stats({"crash": 1})


def call_with_timeout(fn, args, kwargs, timeout=10):
    parent_conn, child_conn = multiprocessing.Pipe()
    start = time.time()
    proc = multiprocessing.Process(target=call_with_timeout_subproc, args=(fn, args, kwargs, child_conn))
    proc.start()
    while proc.is_alive():
        if parent_conn.poll(1):
            result = parent_conn.recv()
            proc.join()
            return result
        if time.time() - start > timeout:
            os.kill(proc.pid, signal.SIGINT)
            time.sleep(1)
            proc.terminate()
            proc.join(10)
            raise TimeoutError(f"took longer than {timeout} seconds")

    proc.join()
    if proc.exitcode == 0:
        return parent_conn.recv()
    else:
        raise OSError(f"exitcode should be 0, got {proc.exitcode}")


def call_with_timeout_subproc(fn, args, kwargs, return_pipe):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (10 * 1024 ** 3, hard))
    try:
        result = fn(*args, *kwargs)
    except Exception:
        log.exception("Error from subprocess")
    return_pipe.send(result)


def test_zipfile_subproc(tempdir: str, path: str):
    altpath = re.sub(r"\.[a-z]{1,3}$", ".zip", path)
    if os.path.exists(altpath):
        path = altpath

    errors = ErrorAggregatorDict(path)
    stats = Stats()
    with open("generated/test_{}.py".format(re.sub(r"([.]zip|/)$", "", os.path.basename(path))), "w") as output_py:
        extractor = PyTorchModuleExtractor(tempdir, errors, stats, output_py=output_py)

        with patch.object(torch.Tensor, "cuda", lambda x: x):
            extractor.main(path)

    return errors, stats


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--run", help="Process a .zip file from a github download")
    group.add_argument("--run-direct")
    parser.add_argument("--download-dir", "-d", default="../paritybench_download")
    parser.add_argument("--limit", "-l", type=int)
    args = parser.parse_args()

    if args.download:
        CrawlGitHub(args.download_dir).download()
        return

    with open("generated/_paritybench_helpers.py", "w") as fd, patch('sys.argv', sys.argv[:1]):
        fd.write(PARITYBENCH_HELPERS)
        fd.flush()
        helpers = types.ModuleType("_paritybench_helpers")
        exec(compile(PARITYBENCH_HELPERS, "./generated/_paritybench_helpers.py", "exec"),
             helpers.__dict__, helpers.__dict__)
        sys.modules["_paritybench_helpers"] = helpers

    if args.run:
        print("Y")
        assert os.path.isfile(args.run)
        errors, stats = test_zipfile(args.run)
        errors.print_report()
        log.info(f"Stats: {stats}")
        return

    if args.run_direct:
        assert os.path.isfile(args.run_direct)
        with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
            test_zipfile_subproc(tempdir, args.run_direct)
        return

    test_all(args.download_dir, args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
