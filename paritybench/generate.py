import logging
import os
import re
import sys
import time
import types
from functools import partial
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

from paritybench.module_extractor import PyTorchModuleExtractor
from paritybench.reporting import Stats, ErrorAggregatorDict
from paritybench.utils import subproc_wrapper

log = logging.getLogger(__name__)


def write_helpers():
    src = "paritybench/_paritybench_helpers.py"
    dst = "generated/_paritybench_helpers.py"
    os.path.exists(dst) and os.unlink(dst)
    os.symlink(os.path.join("..", src), dst)
    helpers_code = open(dst).read()
    with patch('sys.argv', sys.argv[:1]):  # testcase import does annoying stuff
        helpers = types.ModuleType("_paritybench_helpers")
        exec(compile(helpers_code, "generated/_paritybench_helpers.py", "exec"),
             helpers.__dict__, helpers.__dict__)
        sys.modules["_paritybench_helpers"] = helpers


def generate_zipfile_subproc(tempdir: str, path: str, args):
    '''
    Args:
        tempdir: temporary dir
        path: input path process a .zip file from a github download
    '''
    errors = ErrorAggregatorDict(path)
    stats = Stats()
    test_path = "{}/test_{}.py".format(args.tests_dir, re.sub(r"([.]zip|/)$", "", os.path.basename(path)))
    with open(test_path, "w") as output_py:
        extractor = PyTorchModuleExtractor(tempdir, errors, stats, output_py=output_py, args=args)
        extractor.main(path)
    return errors, stats




def generate_all(args, download_dir, limit=None, jobs=4):
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    zipfiles = [os.path.join(download_dir, f)
                for f in os.listdir(download_dir)
                if f.endswith(".zip")]
    zipfiles.sort()

    fgen = partial(generate_zipfile_subproc, args=args)
    generate_zipfile = partial(subproc_wrapper, fn=fgen)

    if limit:
        zipfiles = zipfiles[:limit]
    pool = ThreadPool(jobs)
    for errors_part, stats_part in pool.imap_unordered(generate_zipfile, zipfiles):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()
    errors.print_report()
    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds")
