import argparse
import logging
import os
import sys
from copy import deepcopy
from functools import partial

from paritybench import evaluate
from lazy_transpiler.ltvm import LazyTranspilerVirtualMachine, log
from lazy_transpiler.callable_decoder import NNModuleDecoder
from paritybench.evaluate import init_module, run_eager, JitFailed, check_output
from paritybench.generate import write_helpers
from paritybench.main import main_one_file, add_options
from paritybench.utils import subproc_wrapper


def analyze_nn_module(nn_cls, get_init_args, get_forward_args, record_error):
    nn = init_module(record_error, nn_cls, get_init_args)
    args, kwargs, result1, result2 = run_eager(record_error, nn, get_forward_args)

    try:
        result3 = NNModuleDecoder(nn).debug_call(args, kwargs)
    except Exception as e:
        record_error('flatten', e)
        raise JitFailed()

    check_output(record_error, result1, result2, result3, 'flatten_output')

    try:
        ltvm = LazyTranspilerVirtualMachine(nn)
        result3 = ltvm.run(*deepcopy(args), **deepcopy(kwargs))
        result4 = ltvm.run(*args, **kwargs)
    except Exception as e:
        record_error('ltvm', e)
        raise JitFailed()

    log.info(str(ltvm))

    check_output(record_error, result1, result2, result3, 'ltvm_output')
    check_output(record_error, result1, result2, result4, 'ltvm_rerun')

    return True


analyze_pyfile_subproc = partial(evaluate.evaluate_pyfile_subproc,
                                 check_module=analyze_nn_module,
                                 repro_fmt="{nn_cls.__name__}")
analyze_pyfile = partial(subproc_wrapper, fn=analyze_pyfile_subproc)
analyze_all = partial(evaluate.evaluate_all, fn=analyze_pyfile)


def main():
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", "-e", "-a",
                        help="run just a single test file")
    parser.add_argument("--debug", "-d", action="store_true")
    add_options(parser)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    os.environ["RLIMIT_AS_GB"] = str(args.memory_limit_gb)

    write_helpers()

    if args.analyze:
        return main_one_file(analyze_pyfile_subproc, args.analyze, args)

    return analyze_all(tests_dir=args.tests_dir, limit=args.limit, jobs=args.jobs)


