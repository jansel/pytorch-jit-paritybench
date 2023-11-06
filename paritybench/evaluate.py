import dataclasses
import logging
import os
import re
import time
from functools import partial
from multiprocessing.pool import ThreadPool
import threading

import pandas as pd
import torch
import torch._dynamo
import torch._inductor
from torch.testing._internal.jit_utils import JitTestCase
from torch._decomp import core_aten_decompositions
from torch._dynamo.testing import same
from torch._export import ExportDynamoConfig

from paritybench.reporting import ErrorAggregatorDict, Stats
from paritybench.utils import import_file, get_skiplist, get_cosine_and_fp64_outputs, get_tol, \
    patch_torch_manual_seed, reset_rng_state, subproc_wrapper, wrap_args, wrap_kwargs, export_aot_inductor


log = logging.getLogger(__name__)

# Remove inductor randomness
torch._inductor.config.fallback_random = True
# Remove randomeness when torch manual seed is called
patch_torch_manual_seed()

lock = threading.Lock()


class EagerFailed(RuntimeError):
    pass

class OnnxFailed(RuntimeError):
    pass

class JitFailed(RuntimeError):
    pass


def evaluate_nn_module(nn_cls, get_init_args, get_forward_args, record_error, main_args, path):
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
        raise EagerFailed()

    device = torch.device(main_args.device)

    try:
        nn.eval()
        nn.to(device)
    except Exception:
        pass

    nn_script = None
    if main_args.compile_mode == 'torchscript':
        try:
            nn_script = torch.jit.script(nn)
        except Exception as e:
            record_error('compile {}'.format(main_args.compile_mode), e)
            raise JitFailed()

    is_inductor_test = main_args.compile_mode in ('dynamo', 'aot_inductor') and main_args.backend == 'inductor'
    cosine = False
    fp64_outputs = None

    try:
        args, kwargs = get_forward_args()
        args = wrap_args(args, device)
        kwargs = wrap_kwargs(kwargs, device)

        if is_inductor_test:
            cosine, fp64_outputs = get_cosine_and_fp64_outputs(nn, args, kwargs)

        if main_args.metric_path:
            torch.cuda.synchronize()
            eager_start_ts = time.perf_counter()
        # The first eager run
        reset_rng_state()
        result1 = nn(*args, **kwargs)
        if main_args.metric_path:
            torch.cuda.synchronize()
            eager_elapse = time.perf_counter() - eager_start_ts

        # The second eager run
        reset_rng_state()
        result2 = nn(*args, **kwargs)
    except Exception as e:
        record_error('run_eager', e)
        raise EagerFailed()

    if main_args.onnxdir:
        try:
            onnx_path = "{}/{}.onnx".format(main_args.onnxdir, nn_cls.__name__)
            torch.onnx.export(nn, *args, onnx_path)
        except Exception as e:
            record_error('export_onnx', e)
            raise OnnxFailed()

    if main_args.metric_path:
        torch.cuda.synchronize()
        dynamo_start_ts = time.perf_counter()

    try:
        if nn_script:
            result3 = nn_script(*args, **kwargs)
        else:
            # Dynamo/Inductor/Export run
            reset_rng_state()
            torch._dynamo.reset()
            if main_args.compile_mode == 'dynamo':
                compiled_model = torch._dynamo.optimize(
                    main_args.backend, nopython=main_args.fullgraph
                )(nn)
                result3 = compiled_model(*args, **kwargs)
            elif main_args.compile_mode == 'export':
                DECOMP_TABLE = core_aten_decompositions()

                with torch._dynamo.config.patch(dataclasses.asdict(ExportDynamoConfig())):
                    exported_model, _ = torch._dynamo.export(
                        nn,
                        *args,
                        aten_graph=True,
                        tracing_mode="symbolic",
                        decomposition_table=DECOMP_TABLE,
                        constraints=None,
                        assume_static_by_default=True,
                        **kwargs
                    )
                    result3 = exported_model(*args, **kwargs)
            elif main_args.compile_mode == 'aot_inductor':
                compiled_model = export_aot_inductor(nn, args, kwargs, device.type)
                result3 = compiled_model(compiled_model, args, kwargs)
            else:
                raise AssertionError("Invalid compile_mode")

    except Exception as e:
        record_error('run_jit {} '.format(main_args.compile_mode), e)
        raise JitFailed()

    if main_args.metric_path:
        torch.cuda.synchronize()
        dynamo_elapse = time.perf_counter() - dynamo_start_ts

    tol = get_tol(main_args)
    try:
        JitTestCase().assertEqual(result1, result2)
        try:
            # Dynamo/Inductor/Export accuracy check against eager mode
            if is_inductor_test:
                JitTestCase().assertTrue(
                    same(
                        result2,
                        result3,
                        fp64_ref=fp64_outputs,
                        cos_similarity=cosine,
                        tol=tol,
                    )
                )
            else:
                JitTestCase().assertEqual(result2, result3, atol=tol, rtol=tol)
        except Exception as e:
            record_error('check_output', e)
            raise JitFailed()
    except AssertionError:
        pass  # output is not deterministic, cant check it -- assuming correct

    # Record compilation metrics
    if main_args.metric_path:
        from torch._dynamo.utils import compilation_metrics
        model_id = f"{nn_cls.__module__}.{nn_cls.__name__}"
        compilation_metrics = {
            "model_id": model_id,
            "dynamo_wall_time": dynamo_elapse,
            "eager_wall_time": eager_elapse,
            "wall_time_diff": dynamo_elapse - eager_elapse,
            "_compile": compilation_metrics.get("_compile", [0.0])[0]
        }

        with lock, open(main_args.metric_path, "a") as f:
            logline = []
            for _, v in compilation_metrics.items():
                if isinstance(v, float):
                    logline.append(f"{v:.3f}")
                else:
                    logline.append(str(v))
            f.write(' '.join(logline))
            f.write('\n')

    return True


def evaluate_pyfile_subproc(tempdir: str, path: str, args):
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

        if args.filter and args.filter not in nn_cls.__name__:
            continue

        if f"{path}:{nn_cls.__name__}" in get_skiplist(args):
            continue

        # nn.module doesn't have `forward` function(e.g, has __call__ instead).
        # dynamo doesn't plan to support it yet.
        if nn_cls.forward.__name__ == "_forward_unimplemented":
            continue

        stats["tests"] += 1
        repro = f"{nn_cls.__name__} # pytest {path} -k test_{index:03d}"
        try:
            rv = evaluate_nn_module(
                nn_cls,
                get_init_args,
                get_forward_args,
                partial(errors.record, module=repro),
                main_args=args,
                path=path)
            stats["tests_passed"] += int(rv)
        except JitFailed:
            pass
        except EagerFailed:
            stats["eager_failed"] += 1
        except OnnxFailed:
            pass

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


def evaluate_all(args, tests_dir: str = './generated', offset: int = 0, limit: int = None,
                 jobs=4):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    :param jobs: how many processes to run at once
    """
    feval = partial(evaluate_pyfile_subproc, args=args)
    fn = partial(subproc_wrapper, fn=feval, fresh_cache_dir=args.fresh_cache_dir)
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    testfiles = [os.path.join(tests_dir, f)
                 for f in os.listdir(tests_dir)
                 if re.search(r"test_.*[.]py$", f)]
    testfiles.sort()

    if limit:
        testfiles = testfiles[offset: offset+limit]

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

    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds\n\n{args.compile_mode} {args.backend} ParityBench:\n{report}")
