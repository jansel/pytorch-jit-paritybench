import argparse
import logging
import os
import sys
import tempfile

from paritybench.crawler import CrawlGitHub
from paritybench.evaluate import evaluate
from paritybench.generate import generate_all
from paritybench.generate import generate_zipfile
from paritybench.generate import generate_zipfile_subproc
from paritybench.generate import write_helpers

log = logging.getLogger(__name__)


def main():
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true")
    group.add_argument("--generate-all", action="store_true")
    group.add_argument("--generate-one", "-r", help="Process a .zip file from a github download")
    group.add_argument("--evaluate", action="store_true", help="Check torch.jit.script parity")
    parser.add_argument("--download-dir", "-d", default="./paritybench_download")
    parser.add_argument("--limit", "-l", type=int)
    parser.add_argument("--filter", "-f")
    parser.add_argument("--no-fork", action="store_true")
    args = parser.parse_args()

    if args.download:
        CrawlGitHub(args.download_dir).download()
        return

    write_helpers()

    if args.generate_one and args.no_fork:
        assert os.path.isfile(args.run_direct)
        with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
            generate_zipfile_subproc(tempdir, args.run_direct, name_filter=args.filter)
        return

    if args.generate_one:
        if ':' in args.run and not args.filter:
            args.run, args.filter = args.run.split(':', 2)
        assert os.path.isfile(args.run)
        errors, stats = generate_zipfile(args.run, args.filter)
        errors.print_report()
        log.info(f"Stats: {stats}")
        return

    if args.generate_all:
        return generate_all(args.download_dir, args.limit)

    return evaluate(limit=args.limit, name_filter=args.filter)
