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


class EagerFailed(RuntimeError):
    pass


class JitFailed(RuntimeError):
    pass


def evaluate_nn_module(nn_cls, get_init_args, get_forward_args, record_error, jit_script=torch.jit.script):
    """
    Run an nn.Module with torch.jit.script and see if it works the same
    as eager.

    :param nn_cls: a subclass of nn.Module to be tested
    :param get_init_args: function that returns (args, kwargs)
    :param get_forward_args: function that returns (args, kwargs)
    :param record_error: function to record an exception for debugging/reporting
    :return: True if the test passes
    """
    nn = init_module(record_error, nn_cls, get_init_args)

    try:
        nn_script = jit_script(nn)
    except Exception as e:
        record_error('compile', e)
        raise JitFailed()

    args, kwargs, result1, result2 = run_eager(record_error, nn, get_forward_args)

    try:
        result3 = nn_script(*args, **kwargs)
    except Exception as e:
        record_error('run_jit', e)
        raise JitFailed()

    check_output(record_error, result1, result2, result3)

    return True


def run_eager(record_error, nn, get_forward_args):
    try:
        args, kwargs = get_forward_args()
        result1 = nn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        result2 = nn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    except Exception as e:
        record_error('run_eager', e)
        raise EagerFailed()
    return args, kwargs, result1, result2


def init_module(record_error, nn_cls, get_init_args):
    try:
        args, kwargs = get_init_args()
        nn = nn_cls(*args, **kwargs)
    except Exception as e:
        record_error('init', e)
        raise EagerFailed()
    try:
        nn.eval()
    except Exception:
        pass
    return nn


def check_output(record_error, result1, result2, result3, category="check_output"):
    try:
        JitTestCase().assertEqual(result1, result2)
        try:
            JitTestCase().assertEqual(result2, result3)
        except Exception as e:
            record_error(category, e)
            raise JitFailed()
    except AssertionError:
        pass  # output is not deterministic, cant check it -- assuming correct


def evaluate_pyfile_subproc(tempdir: str, path: str, name_filter=None, check_module=evaluate_nn_module):
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

    index = -1
    for nn_cls, get_init_args, get_forward_args, compiles in module.TESTCASES:
        index += 1

        if name_filter and name_filter not in nn_cls.__name__:
            continue

        stats["tests"] += 1
        repro = f"{nn_cls.__name__} # pytest {path} -k test_{index:03d}"
        try:
            rv = check_module(
                nn_cls,
                get_init_args,
                get_forward_args,
                partial(errors.record, module=repro))
            stats["tests_passed"] += int(rv)
        except JitFailed:
            pass
        except EagerFailed:
            stats["eager_failed"] += 1

    stats["tests"] = stats["tests"] - stats["eager_failed"]
    stats["tests_failed"] = stats["tests"] - stats["tests_passed"]

    if not stats["tests"]:
        # eager failed not the jit, remove from totals
        stats["projects"] -= 1
    elif stats["tests_failed"]:
        stats["projects_failed"] += 1
    else:
        stats["projects_passed"] += 1

    return errors, stats


evaluate_pyfile = partial(subproc_wrapper, fn=evaluate_pyfile_subproc)


def evaluate_all(tests_dir: str = './generated', limit: int = None, fn: callable = evaluate_pyfile,
                 jobs=4):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    :param jobs: how many processes to run at once
    """
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    testfiles = [os.path.join(tests_dir, f)
                 for f in os.listdir(tests_dir)
                 if re.search(r"test_.*[.]py$", f)]
    testfiles.sort()

    if limit:
        testfiles = testfiles[:limit]

    pool = ThreadPool(jobs)
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
