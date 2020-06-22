import argparse
import logging
import multiprocessing
import os
import re
import resource
import signal
import sys
import tempfile
import time
import types
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

import torch

from paritybench.reporting import Stats, ErrorAggregatorDict
from paritybench.module_extractor import PyTorchModuleExtractor
from paritybench.crawler import CrawlGitHub

log = logging.getLogger(__name__)

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


def write_helpers():
    with open("generated/_paritybench_helpers.py", "w") as fd, patch('sys.argv', sys.argv[:1]):
        fd.write(PARITYBENCH_HELPERS)
        fd.flush()
        helpers = types.ModuleType("_paritybench_helpers")
        exec(compile(PARITYBENCH_HELPERS, "generated/_paritybench_helpers.py", "exec"),
             helpers.__dict__, helpers.__dict__)
        sys.modules["_paritybench_helpers"] = helpers


def test_all(download_dir, limit=None):
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    zipfiles = [os.path.join(download_dir, f)
                for f in os.listdir(download_dir)
                if f.endswith(".zip")]
    zipfiles.sort()

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
            os.kill(proc.pid, signal.SIGINT)  # maybe generate a stack trace for debugging
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
        return_pipe.send(result)
    except Exception:
        log.exception("Error from subprocess")
        sys.exit(1)


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
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--run", help="Process a .zip file from a github download")
    group.add_argument("--run-direct")
    parser.add_argument("--download-dir", "-d", default="./paritybench_download")
    parser.add_argument("--limit", "-l", type=int)
    args = parser.parse_args()

    if args.download:
        CrawlGitHub(args.download_dir).download()
        return

    write_helpers()

    if args.run:
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
