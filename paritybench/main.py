import argparse
import logging
import os
import sys
from functools import partial

from paritybench.analyze import analyze_all, analyze_pyfile_subproc
from paritybench.crawler import CrawlGitHub
from paritybench.evaluate import evaluate_all, evaluate_pyfile_subproc
from paritybench.generate import generate_all, generate_zipfile_subproc
from paritybench.generate import write_helpers
from paritybench.utils import subproc_wrapper, tempdir_wrapper

log = logging.getLogger(__name__)


def main_one_file(fn, path, args):
    if ':' in path and not args.filter:
        path, args.filter = path.split(':', 2)
    assert os.path.isfile(path)

    if args.filter:
        fn = partial(fn, name_filter=args.filter)

    if not args.no_fork:
        wrapper = subproc_wrapper
    else:
        wrapper = tempdir_wrapper

    errors, stats = wrapper(path, fn=fn)

    errors.print_report()
    log.info(f"Stats: {stats}")
    return


def main():
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="[SLOW:days] crawl and download top github projects")

    group.add_argument("--generate-all", action="store_true",
                       help="Turn crawled github projects into generated testcases")
    group.add_argument("--generate-one", "-g", help="Process a .zip file from a github download")

    group.add_argument("--evaluate-one", "-e", help="Check torch.jit.script on a given test_*.py file")
    group.add_argument("--evaluate-all", action="store_true", help="Check torch.jit.script parity")

    group.add_argument("--analyze-one", "-a")
    group.add_argument("--analyze-all", "-z", action="store_true")

    parser.add_argument("--jobs", "-j", type=int, default=4)
    parser.add_argument("--limit", "-l", type=int, help="only run the first N files")
    parser.add_argument("--filter", "-f", "-k", help="only run module containing given name")
    parser.add_argument("--no-fork", action="store_true", help="don't run *-one test in a subprocess")
    parser.add_argument("--memory-limit-gb", type=int, default=10)

    parser.add_argument("--download-dir", default="./paritybench_download", help="./paritybench_download")
    parser.add_argument("--tests-dir", default="./generated", help="./generated")
    args = parser.parse_args()

    os.environ["RLIMIT_AS_GB"] = str(args.memory_limit_gb)

    if args.download:
        return CrawlGitHub(args.download_dir).download()

    write_helpers()

    if args.generate_one:
        return main_one_file(generate_zipfile_subproc, args.generate_one, args)

    if args.generate_all:
        return generate_all(download_dir=args.download_dir, limit=args.limit, jobs=args.jobs)

    if args.analyze_one:
        return main_one_file(analyze_pyfile_subproc, args.analyze_one, args)

    if args.analyze_all:
        return analyze_all(tests_dir=args.tests_dir, limit=args.limit, jobs=args.jobs)

    if args.evaluate_one:
        return main_one_file(evaluate_pyfile_subproc, args.evaluate_one, args)

    # args.evaluate_all is the default:
    return evaluate_all(tests_dir=args.tests_dir, limit=args.limit, jobs=args.jobs)
