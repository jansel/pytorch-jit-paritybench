import argparse
import logging
import os
import re
import sys
import tempfile
import time
import types
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

import torch

from paritybench.crawler import CrawlGitHub
from paritybench.module_extractor import PyTorchModuleExtractor
from paritybench.reporting import Stats, ErrorAggregatorDict
from paritybench.utils import call_with_timeout

log = logging.getLogger(__name__)


def write_helpers():
    src = "paritybench/_paritybench_helpers.py"
    dst = "generated/_paritybench_helpers.py"
    os.unlink(dst)
    os.symlink(os.path.join("..", src), dst)
    helpers_code = open(dst).read()
    with patch('sys.argv', sys.argv[:1]):  # testcase import does annoying stuff
        helpers = types.ModuleType("_paritybench_helpers")
        exec(compile(helpers_code, "generated/_paritybench_helpers.py", "exec"),
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
    pool = ThreadPool(4)
    for errors_part, stats_part in pool.imap_unordered(test_zipfile, zipfiles):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()
    errors.print_report()
    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds")


def test_zipfile(path, name_filter=None):
    log.info(f"Running {path}")
    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        try:
            return call_with_timeout(test_zipfile_subproc, (tempdir, path, name_filter), {}, timeout=900)
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


def test_zipfile_subproc(tempdir: str, path: str, name_filter=None):
    altpath = re.sub(r"\.[a-z]{1,3}$", ".zip", path)
    if os.path.exists(altpath):
        path = altpath

    errors = ErrorAggregatorDict(path)
    stats = Stats()
    with open("generated/test_{}.py".format(re.sub(r"([.]zip|/)$", "", os.path.basename(path))), "w") as output_py:
        extractor = PyTorchModuleExtractor(tempdir, errors, stats, output_py=output_py, name_filter=name_filter)

        with patch.object(torch.Tensor, "cuda", lambda x: x):
            extractor.main(path)

    return errors, stats


def main():
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--run", "-r", help="Process a .zip file from a github download")
    group.add_argument("--run-direct")
    parser.add_argument("--download-dir", "-d", default="./paritybench_download")
    parser.add_argument("--limit", "-l", type=int)
    parser.add_argument("--filter", "-f")
    args = parser.parse_args()

    if args.download:
        CrawlGitHub(args.download_dir).download()
        return

    write_helpers()

    if args.run:
        if ':' in args.run and not args.filter:
            args.run, args.filter = args.run.split(':', 2)
        assert os.path.isfile(args.run)
        errors, stats = test_zipfile(args.run, args.filter)
        errors.print_report()
        log.info(f"Stats: {stats}")
        return

    if args.run_direct:
        assert os.path.isfile(args.run_direct)
        with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
            test_zipfile_subproc(tempdir, args.run_direct, name_filter=args.filter)
        return

    test_all(args.download_dir, args.limit)
