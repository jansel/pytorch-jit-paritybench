import copy
import logging
import os
import re
import resource
import signal
import sys
import tempfile
import time
import types
import torch

from torch import multiprocessing

from paritybench.reporting import ErrorAggregatorDict, Stats

log = logging.getLogger(__name__)


def call_with_timeout(fn, args, kwargs=None, timeout=10):
    kwargs = kwargs or {}
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
    resource.setrlimit(resource.RLIMIT_AS, (int(os.environ.get("RLIMIT_AS_GB", 10)) * 1024 ** 3, hard))
    try:
        result = fn(*args, *kwargs)
        return_pipe.send(result)
    except Exception:
        log.exception("Error from subprocess")
        sys.exit(1)


def import_file(path):
    """
    :param path: to a *.py file
    :return: a python module
    """
    module = types.ModuleType(re.findall(r"test_[^.]+", path)[0])
    sys.modules[module.__name__] = module
    exec(compile(open(path).read(), filename=path, mode='exec'),
         module.__dict__, module.__dict__)
    if not hasattr(module, "TESTCASES"):
        module.TESTCASES = []
    return module


def subproc_wrapper(path: str, fn: callable, timeout: int = 900):
    """
    A wrapper around call_with_timeout() adding a temp dir and error handling.

    :param path: path to code to test
    :param fn: function to run in subprocess
    :param timeout: seconds to wait
    :return: errors, stats
    """
    log.info(f"Running {path}")
    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        try:
            return call_with_timeout(fn, (tempdir, path), {}, timeout=timeout)
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


def tempdir_wrapper(path: str, fn: callable):
    """ Non-forking version of subproc_wrapper """
    log.info(f"Running {path}")
    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        return fn(tempdir, path)


def wrap_args(args, device="cuda"):
    device = torch.device(device)
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in copy.deepcopy(args)]


def wrap_kwargs(kwargs, device="cuda"):
    device = torch.device(device)
    wrapped_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            wrapped_kwargs.update({k: v.clone().to(device)})
        else:
            wrapped_kwargs.update({k: copy.deepcopy(v)})
    return wrapped_kwargs


def get_skiplist(main_args):
    if main_args.compile_mode == 'export':
        return SKIP.get("export")
    else:
        return SKIP.get(main_args.backend)


SKIP_DYNAMO_EAGER = [
    "./generated/test_deepinsight_insightface.py:deeplab_xception_transfer_basemodel",  # try ... catch ...
    "./generated/test_BlinkDL_RWKV_LM.py:RWKV_ChannelMix",  # Subclasses torch.jit.ScriptModule
]
SKIP_INDUCTOR = []
SKIP = {
    "eager": SKIP_DYNAMO_EAGER,
    "inductor": SKIP_DYNAMO_EAGER + SKIP_INDUCTOR,
    "export": SKIP_DYNAMO_EAGER,
}