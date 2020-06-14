import csv
import logging
import os
import random
import re
from collections import Counter, defaultdict
from typing import List

log = logging.getLogger(__name__)


class Stats(Counter):
    """
    Collect and group error messages for a debug report at the end
    """

    def __str__(self):
        """
        Reorder key print order by stage in the process
        """
        stats_keys = [
            "total",
            "init_ok",
            "deduced_args_ok",
            "jit_compiles",
        ]
        stats_keys = stats_keys + list(set(self.keys()) - set(stats_keys))
        return str([(k, self[k]) for k in stats_keys])


class ErrorAggregator(object):
    """
    Collect and group error messages for report at the end
    """

    def __init__(self, context=None, log=None):
        super(ErrorAggregator, self).__init__()
        if context:
            self.context = re.sub(r"\.zip$", ".py", context)
        else:
            self.context = ""
        self.error_groups = []
        self.bigram_to_group_ids = defaultdict(list)
        self.log = log or logging.getLogger(__name__)

    def record(self, e: Exception, module):
        ex_msg = str(e).strip().split('\n')[0]
        error_msg = f"{e.__class__.__name__}: {ex_msg}"
        full_msg = f"{e.__class__.__name__}: {str(e)}"
        return self._add(error_msg, [(error_msg, f"{self.context}:{module}", full_msg)])

    def update(self, other):
        for errors in other.error_groups:
            self._add(errors[0][0], errors)

    def _add(self, error_msg: str, errors: List):
        msg_words = list(re.findall(r"[a-zA-Z]+", error_msg))
        if "NameError" in error_msg:
            msg_bigrams = [error_msg]  # need exact match
        else:
            msg_bigrams = [f"{a}_{b}" for a, b in zip(msg_words, msg_words[1:])] or msg_words

        shared_bigrams = Counter()
        for bigram in msg_bigrams:
            shared_bigrams.update(self.bigram_to_group_ids[bigram])

        if shared_bigrams:
            best_match, count = shared_bigrams.most_common(1)[0]
            if count > len(msg_bigrams) // 2:
                self.error_groups[best_match].extend(errors)
                return False

        # No match, create a new error group
        group_id = len(self.error_groups)
        self.error_groups.append(errors)
        for bigram in msg_bigrams:
            self.bigram_to_group_ids[bigram].append(group_id)

        return True

    @staticmethod
    def format_error_group(errors):
        context, context_count = random.choice(list(Counter(context for msg, context, _ in errors).items()))
        return f"  - {len(errors)} errors like: {errors[0][0]} (example {context})"

    def __str__(self):
        errors = sorted(self.error_groups, key=len, reverse=True)
        return '\n'.join(map(self.format_error_group, errors[:20]))

    def __len__(self):
        return sum(map(len, self.error_groups))

    csv_headers = ["phase", "count", "example_short", "example_long", "example_from"]

    def write_csv(self, phase, out: csv.writer):
        for errors in sorted(self.error_groups, key=len, reverse=True)[:20]:
            short, context, long = random.choice(errors)
            out.writerow([phase, len(errors), short, long, context])


class ErrorAggregatorDict(object):
    """
    Collect and group error messages for a debug report at the end
    """

    @classmethod
    def single(cls, name: str, e: Exception, context=None):
        errors = cls(context)
        errors.record(name, e, 'global')
        return errors

    def __init__(self, context=None):
        super(ErrorAggregatorDict, self).__init__()
        self.aggregator = dict()
        self.context = context
        if context:
            self.name = re.sub(r"[.]zip$", "", os.path.basename(context))
        else:
            self.name = __name__

    def __getitem__(self, item):
        if item not in self.aggregator:
            self.aggregator[item] = ErrorAggregator(self.context, logging.getLogger(f"{item}.{self.name}"))
        return self.aggregator[item]

    def update(self, other):
        for key, value in other.aggregator.items():
            self[key].update(other=value)

    def print_report(self):
        for name in sorted(list(self.aggregator.keys())):
            self[name].log.info(f"\nTop errors in {name} ({len(self[name])} total):\n{self[name]}\n")

        with open('errors.csv', "w") as fd:
            out = csv.writer(fd)
            out.writerow(ErrorAggregator.csv_headers)
            for name in sorted(list(self.aggregator.keys())):
                self[name].write_csv(name, out)

    def record(self, error_type, error, module=None):
        module = str(getattr(module, "__name__", module))
        if self[error_type].record(error, module):
            log.exception(f"{error_type} error from {self.context}:{module}")
