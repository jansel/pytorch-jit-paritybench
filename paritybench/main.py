import argparse
import logging
import os
import sys
import torch
import torch._dynamo
from functools import partial

from paritybench.crawler import CrawlGitHub
from paritybench.evaluate import evaluate_all, evaluate_pyfile_subproc
from paritybench.generate import generate_all, generate_zipfile_subproc
from paritybench.generate import write_helpers
from paritybench.utils import subproc_wrapper, tempdir_wrapper

log = logging.getLogger(__name__)

def main_one_file(fn, path, args):
    if ':' in path and not args.filter:
        path, args.filter = path.split(':', 2)
    assert os.path.isfile(path) or os.path.isdir(path)

    fn = partial(fn, args=args)

    if not args.no_fork:
        wrapper = subproc_wrapper
    else:
        wrapper = tempdir_wrapper

    errors, stats = wrapper(path, fn=fn)

    errors.print_report()
    log.info(f"Stats: {stats}")
    return

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="[SLOW:days] crawl and download top github projects")
    group.add_argument("--generate-all", action="store_true",
                       help="Turn crawled github projects into generated testcases")
    group.add_argument("--generate-one", "-g", help="Process a .zip file from a github download")
    group.add_argument("--evaluate-one", "-e", help="Check torch.jit.script on a given test_*.py file")
    group.add_argument("--evaluate-all", action="store_true", help="Check torch.jit.script parity")

    parser.add_argument("--jobs", "-j", type=int, default=4)
    parser.add_argument("--limit", "-l", type=int, help="only run the first N files")
    parser.add_argument("--filter", "-f", "-k", help="only run module containing given name")
    parser.add_argument("--no-fork", action="store_true", help="don't run *-one test in a subprocess")
    parser.add_argument("--memory-limit-gb", type=int, default=10)

    parser.add_argument("--onnxdir", type=str, help="dir where to export modules to onnx during evaluate")
    parser.add_argument("--exportdir", type=str, help="dir where to export modules by torch.export during evaluate")
    parser.add_argument("--compile_mode", default="dynamo", type=str, help="choose a mode of compilation: dynamo or torchscript")
    parser.add_argument("--backend", default="inductor", type=str, help="dynamo backends: {}".format(torch._dynamo.list_backends()))
    parser.add_argument("--device", default="cuda", type=str, help="evaluate modules using cuda or cpu")
    parser.add_argument("--download-dir", default="./paritybench_download", help="dir where to download project default: ./paritybench_download")
    parser.add_argument("--tests-dir", default="./generated", help="dir where to generate test scripts default: ./generated")
    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    logging.basicConfig(level=logging.INFO)
    args = get_args(raw_args)

    os.environ["RLIMIT_AS_GB"] = str(args.memory_limit_gb)

    if args.download:
        return CrawlGitHub(args.download_dir, max_count=args.limit).download()

    write_helpers()
    torch.multiprocessing.set_start_method('spawn')

    if args.evaluate_one:
        return main_one_file(evaluate_pyfile_subproc, args.evaluate_one, args)

    if args.generate_one:
        return main_one_file(generate_zipfile_subproc, args.generate_one, args)

    if args.generate_all:
        return generate_all(args, download_dir=args.download_dir, limit=args.limit, jobs=args.jobs)

    # args.evaluate_all is the default:
    return evaluate_all(args, tests_dir=args.tests_dir, limit=args.limit, jobs=args.jobs)
