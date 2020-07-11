import copy
import logging
import os
import re
import time
from functools import partial
from multiprocessing.pool import ThreadPool

import pandas as pd
import torch
from torch.testing._internal.jit_utils import JitTestCase

from paritybench.reporting import ErrorAggregatorDict, Stats
from paritybench.utils import import_file, subproc_wrapper

log = logging.getLogger(__name__)


def test_nn_module(nn_cls, get_init_args, get_forward_args, record_error):
    """
    Run an nn.Module with torch.jit.script and see if it works the same
    as eager.

    :param nn_cls: a subclass of nn.Module to be tested
    :param get_init_args: function that returns (args, kwargs)
    :param get_forward_args: function that returns (args, kwargs)
    :param record_error: function to record an exception for debugging/reporting
    :return: True if the test passes
    """
    try:
        args, kwargs = get_init_args()
        nn = nn_cls(*args, **kwargs)
    except Exception as e:
        record_error('init', e)
        return False

    try:
        nn.eval()
    except Exception:
        pass

    try:
        nn_script = torch.jit.script(nn)
    except Exception as e:
        record_error('compile', e)
        return False

    try:
        args, kwargs = get_forward_args()
        result1 = nn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        result2 = nn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    except Exception as e:
        record_error('run_eager', e)
        return False

    try:
        result3 = nn_script(*args, **kwargs)
    except Exception as e:
        record_error('run_jit', e)
        return False

    try:
        JitTestCase().assertEqual(result1, result2)
        try:
            JitTestCase().assertEqual(result2, result3)
        except Exception as e:
            record_error('check_output', e)
            return False
    except AssertionError:
        pass  # output is not deterministic, cant check it

    return True


def test_pyfile_subproc(tempdir: str, path: str, name_filter=None):
    """
    Evaluate/test all the TESTCASES in path.

    :param path: *.py file to test
    :return: errors, stats
    """
    errors = ErrorAggregatorDict(path)
    stats = Stats()
    module = import_file(path)

    if not module.TESTCASES:
        return errors, stats

    stats["projects"] += 1
    stats["tests"] += len(module.TESTCASES)

    index = -1
    for nn_cls, get_init_args, get_forward_args, compiles in module.TESTCASES:
        index += 1
        repro = f"{nn_cls.__name__} # pytest {path} -f test_{index:03d}"
        if test_nn_module(nn_cls,
                          get_init_args,
                          get_forward_args,
                          partial(errors.record, module=repro)):
            stats["tests_passed"] += 1

    stats["tests_failed"] = stats["tests"] - stats["tests_passed"]

    if stats["tests_failed"]:
        stats["projects_failed"] += 1
    else:
        stats["projects_passed"] += 1

    return errors, stats


test_pyfile = partial(subproc_wrapper, fn=test_pyfile_subproc)


def evaluate(tests_dir: str = './generated', limit: int = None, fn: callable = test_pyfile, name_filter: str = None):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    """
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    testfiles = [os.path.join(tests_dir, f)
                 for f in os.listdir(tests_dir)
                 if re.search(r"test_.*[.]py$", f)]
    testfiles.sort()

    if name_filter:
        testfiles = [f for f in testfiles if name_filter in f]

    if limit:
        testfiles = testfiles[:limit]

    pool = ThreadPool(4)
    for errors_part, stats_part in pool.imap_unordered(fn, testfiles):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()
    errors.print_report()
    index = ("projects", "tests")
    report = pd.DataFrame(
        [[stats[f"{k}"], stats[f"{k}_passed"], "{:.1%}".format(stats[f"{k}_passed"] / (stats[f"{k}"] or 1))]
         for k in index],
        index=index,
        columns=["total", "passing", "score"],
    )

    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds\n\nTorchScript ParityBench:\n{report}")
