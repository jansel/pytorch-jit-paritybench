import ast
import inspect
import itertools
import logging
import os
import re
import sys
import time
import traceback
from functools import reduce
from typing import Callable, List

import torch

log = logging.getLogger(__name__)


def _mock_layer(in_features=None, out_features=None, bias=True):
    if in_features and out_features:
        return torch.nn.Linear(in_features, out_features, bias)
    return torch.nn.ReLU()


class DeductionFailed(RuntimeError):
    def __init__(self, attempt_log, name='', traceback=''):
        attempt_lines = "\n".join(f" - {attempt}" for attempt in attempt_log)
        error_msg = f"{attempt_log[-1][1]}\n{name}:\n{attempt_lines}\n----\n{traceback}\n----\n"
        super().__init__(error_msg)


class DeduceParameters(object):
    """
    Try to figure out a valid input for an NN module by repeated
    guessing based on error messages.
    """
    default_size = 4

    @staticmethod
    def needed_args(signature: inspect.Signature):
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue  # ignore args/kwargs
            if param.default is inspect.Parameter.empty:
                yield param

    @classmethod
    def initial_args_init(cls, signature: inspect.Signature = None):
        return [], {param.name: DeduceParameter.initial_arg_init(param, position)
                    for position, param in enumerate(cls.needed_args(signature))}

    @classmethod
    def initial_args_forward(cls, signature: inspect.Signature):
        return [DeduceParameter.initial_arg_forward(param, position)
                for position, param in enumerate(cls.needed_args(signature))], {}

    def __init__(self, nn_module: Callable, args: list, kwargs: dict):
        super(DeduceParameters, self).__init__()
        self.nn_module = nn_module
        self.args = args
        self.kwargs = kwargs
        self.tried = set()

        self.attempt_log = []
        self.last_args = None
        self.last_kwargs = None
        self.last_result = None
        self.last_traceback = None

    def __str__(self):
        return ", ".join(itertools.chain(
            map(str, self.args),
            [f"{name}={arg}" for name, arg in self.kwargs.items()]))

    def testcase_args(self):
        args = repr(self.args)
        kwargs = repr(self.kwargs)
        return args, kwargs

    @classmethod
    def run(cls, nn_module: Callable, needed_args: List[inspect.Parameter]):
        return DeduceParameters(nn_module, needed_args).search()

    def search_once(self):
        self.last_args = [arg.guess() for arg in self.args]
        self.last_kwargs = {name: arg.guess() for name, arg in self.kwargs.items()}
        guess_str = str(self)
        self.tried.add(guess_str)

        try:
            self.last_result = self.nn_module(*self.last_args, **self.last_kwargs)
            return True
        except Exception:
            error_type, error_value, tb = sys.exc_info()
            error_msg = f"{error_type.__name__}: {error_value}"
            sorted_args = self.sorted_args(tb)
            self.last_traceback = traceback.format_exc(-2)

        self.attempt_log.append((guess_str, error_msg))

        if Guess.apply_fixors(self.get_fixors(), error_msg):
            if str(self) not in self.tried:
                return False

        for pass_number in (0, 1, 2):
            for arg in sorted_args:
                if arg.try_to_fix(error_msg, pass_number):
                    if str(self) not in self.tried:
                        return False
                    arg.rollback()

        raise DeductionFailed(self.attempt_log, str(type(self.nn_module)), self.last_traceback)

    def all_args(self):
        return list(self.args) + list(self.kwargs.values())

    def sorted_args(self, trackback) -> List:
        """
        Order args by when they are seen in the traceback so we can fix
        relevant to the error args first.

        :param trackback: from sys.exc_info()
        :return: parameters ordered by where they are seen in the traceback
        """
        this_file = os.path.basename(__file__)
        args = self.all_args()
        # args.sort(lambda x: x.num_guesses())
        for frame in reversed(traceback.extract_tb(trackback, limit=-10)):
            if this_file in frame.filename:
                break
            line = frame.line
            args.sort(key=lambda x: x.contained_in_line(line), reverse=True)
        return args

    def search(self, limit: int = 10):
        try:
            for attempt in range(limit):
                if self.search_once():
                    return self.last_result
        except DeductionFailed:
            pass  # ignore error as we will try again

        for arg in self.all_args():
            arg.alt_guess()

        if str(self) not in self.tried:
            for attempt in range(limit):
                if self.search_once():
                    return self.last_result

        raise DeductionFailed(self.attempt_log, str(type(self.nn_module)), self.last_traceback)

    def get_fixors(self):
        return [
            (r"missing.*arguments?: (?P<name>['][a-zA-Z0-9_]+['])",
             self.fix_missing_arg),
            (r"unexpected keyword argument (?P<name>['][a-zA-Z0-9_]+['])",
             self.fix_extra_arg),
            (r"size mismatch, m1: (?P<a>\[.*\]), m2: (?P<b>\[.*\])",
             self.fix_size_mismatch),
            (r"shape (?P<a>\[[\d, ]+\]) doesn't match the broadcast shape (?P<b>\[[\d, ]+\])]",
             self.fix_size_mismatch),
        ]

    def fix_size_mismatch(self, a, b):
        matches_a = [arg for arg in self.all_args() if arg.is_shape_match(a)]
        matches_b = [arg for arg in self.all_args() if arg.is_shape_match(b)]
        if not matches_a and not matches_b:
            matches_a = [arg for arg in self.all_args() if arg.is_element_count_match(a)]
            matches_b = [arg for arg in self.all_args() if arg.is_element_count_match(b)]

        if matches_a and matches_b:
            if max(x.created for x in matches_a) > max(x.created for x in matches_b):
                # prefer changing the old one
                matches_a, matches_b = matches_b, matches_a
            guess_a = min(matches_a, key=lambda x: x.created)
            guess_b = max(matches_b, key=lambda x: x.created)
            guess_a.change_guess(guess_b.clone_guess())
            return True

        if matches_b:
            matches_a, matches_b = matches_b, matches_a
            a, b = b, a

        if matches_a:
            guess = min(matches_a, key=lambda x: x.created)
            guess.change_guess(TensorGuess(shape=b))
            return True

    def fix_missing_arg(self, name):
        if any(arg.name == name for arg in self.args):
            return
        if name in self.kwargs:
            # try moving it to args?
            self.args.append(self.kwargs.pop(name))
            self.args.sort(key=lambda x: x.position)
        else:
            self.kwargs[name] = DeduceParameter.initial_arg_init(name, float('inf'))
        return True

    def fix_extra_arg(self, name):
        if name in self.kwargs:
            del self.kwargs[name]
            return True


class DeduceParameter(object):
    """
    Contains and tracks the guesses of the value of a single parameter.
    """

    @classmethod
    def initial_arg_init(cls, name, position):
        name = getattr(name, 'name', name)
        if 'dataset' in name:
            # TODO: this likely wont work...
            return cls.initial_arg_forward(name, position)

        common_args = {
            "stride": 1,
            "scale": 1.0,
            "layer": 1,
            "dilation": 1,
            "groups": 1,
            "block": 1,
            "depth": 1,
            "gpu": False,
            "train": False,
            "loss": torch.nn.MSELoss(),
            "dropout": 0.5,
            "drop_rate": 0.5,
        }
        for search_name, placeholder_value in common_args.items():
            if search_name in name:
                return cls(name, position, LiteralGuess(placeholder_value))

        for search_name in ('cfg', 'config', 'options', 'args'):
            if search_name in name:
                return cls(name, position, ConfigGuess())

        return cls(name, position, LiteralGuess(DeduceParameters.default_size))

    @classmethod
    def initial_arg_forward(cls, name, position=None, dims=4):
        name = getattr(name, 'name', name)
        return cls(name, position, TensorGuess([TensorGuess.default_size] * dims))

    def __init__(self, name: str, position, initial_guess):
        super(DeduceParameter, self).__init__()
        self.name = name
        self.position = position
        self._guesses = [initial_guess]

    def __str__(self):
        val = str(self._guesses[-1])
        if val.startswith('<function _mock_layer'):
            # TODO: workaround, fix this better
            return '_mock_layer'
        return val

    __repr__ = __str__

    @property
    def created(self):
        return self._guesses[-1].created

    def guess(self):
        return self._guesses[-1].guess()

    def clone_guess(self):
        return self._guesses[-1].clone()

    def num_guesses(self):
        return len(self._guesses)

    def try_to_fix(self, error_message: str, pass_number: int) -> bool:
        new_guess = self._guesses[-1].get_fix(error_message, pass_number, self.name)
        if new_guess is not None:
            self.change_guess(new_guess)
            return True
        return False

    def change_guess(self, guess):
        self._guesses.append(guess)

    def contained_in_line(self, line: str):
        pattern = r"\b{}\b".format(re.escape(self.name))
        return bool(re.search(pattern, line))

    def rollback(self):
        self._guesses[-1].rollback()
        self._guesses.pop()

    def is_shape_match(self, shape):
        if isinstance(self._guesses[-1], TensorGuess):
            return self._guesses[-1].shape == shape

    def is_element_count_match(self, shape):
        if isinstance(self._guesses[-1], TensorGuess):
            count = reduce(lambda a, b: a * b, shape)
            this_count = reduce(lambda a, b: a * b, self._guesses[-1].shape)
            return count == this_count

    def alt_guess(self):
        if isinstance(self._guesses[-1], TensorGuess):
            # try starting with a smaller size
            self.change_guess(TensorGuess([TensorGuess.default_size] * 2))


class Guess(object):
    """
    Base class of a single guess of the value of parameter.
    """

    def __init__(self, value=None):
        super(Guess, self).__init__()
        self.value = value
        self.created = time.time()

    def __str__(self):
        val = repr(self.value)
        if '_mock_layer' in val:
            return '_mock_layer'
        return val

    __repr__ = __str__

    @staticmethod
    def apply_fixors(fixors, error_msg):
        for pattern, fixor in fixors:
            match = re.search(pattern, error_msg, flags=re.I)
            if match:
                fix = fixor(**{k: Guess.literal(v) for k, v in match.groupdict().items()})
                if fix is not None:
                    log.debug(f"FIX: {fixor.__name__} {error_msg}")
                    return fix

    @staticmethod
    def literal(value):
        # [1 x 1] => [1, 1]
        return ast.literal_eval(value.replace(" x ", ","))

    def guess(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def get_fix(self, error_message: str, pass_number: int, name: str):
        pass

    def rollback(self):
        pass


class LiteralGuess(Guess):
    def get_fix(self, error_message: str, pass_number: int, name: str):
        fix = super(LiteralGuess, self).get_fix(error_message, pass_number, name)
        if fix:
            return fix

        if pass_number > 0:
            return

        fixors = []

        if isinstance(self.value, int):
            def fix_too_small():
                if self.value < 64:
                    return 64

            fixors.extend([
                (r"TypeError: cannot unpack non-iterable int object",
                 lambda: [TensorGuess.default_size] * 2),
                (r"TypeError: 'int' object is not subscriptable",
                 lambda: [TensorGuess.default_size] * 2),
                (r"TypeError: 'int' object is not iterable",
                 lambda: [TensorGuess.default_size] * 2),
                (r"TypeError: object of type 'int' has no len()",
                 lambda: [TensorGuess.default_size] * 2),
                (r"AttributeError: 'int' object has no attribute '(size|shape|dtype|device|ndim)'",
                 lambda: TensorGuess([TensorGuess.default_size] * 2)),
                (r"TypeError: int is not a Module subclass",
                 lambda: torch.nn.ReLU()),
                (r"DeductionFailed: TypeError: 'int' object is not callable",
                 lambda: _mock_layer),
                (r"ModuleList.extend should be called with an iterable, but got int",
                 lambda: [torch.nn.ReLU()]),
                (r"ModuleDict.update should be called.*but got int",
                 lambda: {'relu': torch.nn.ReLU()}),
                (r"AttributeError: 'int' object has no attribute 'split'",
                 lambda: "2,2"),
                (r"IndexError: index 0 is out of bounds for dimension 0 with size 0",
                 fix_too_small),
                (r"ValueError: .* must be divisible by groups",
                 fix_too_small),
                (r"KeyError: [1-9]",
                 lambda: self.value // 2),
                (r"multiple of (?P<m>\d{1,3})",
                 lambda m: m),
                (r"dropout probability has to be between 0 and 1, but got (?P<v>\d+)",
                 lambda v: 0.5 if v == self.value else None),
            ])

        if isinstance(self.value, list):
            def fix_too_many(want):
                if len(self.value) > want:
                    return [TensorGuess.default_size] * want

            def fix_too_few(want, got):
                if len(self.value) == got:
                    return [TensorGuess.default_size] * want

            fixors.extend([
                (r"ValueError: too many values to unpack \(expected (?P<want>\d+)\)",
                 fix_too_many),
                (r"ValueError: not enough values to unpack \(expected (?P<want>\d+), got (?P<got>\d+)\)",
                 fix_too_few),
                (r"not supported between instances of '(list|int)' and '(list|int)'",
                 lambda: TensorGuess.default_size),
                (r"TypeError: 'list' object is not callable",
                 lambda: _mock_layer),
                (r"IndexError: list index out of range",
                 lambda: self.value + [TensorGuess.default_size]),
            ])

        new_value = self.apply_fixors(fixors, error_message)
        if new_value is not None:
            if isinstance(new_value, Guess):
                return new_value
            return LiteralGuess(new_value)

    def fix_not_subscriptable(self, typename="int"):
        if typename == "int" and isinstance(self.value, int):
            return [TensorGuess.default_size] * 2


class TensorGuess(Guess):
    default_size = DeduceParameters.default_size

    def __init__(self, shape, dtype=torch.float32):
        super(TensorGuess, self).__init__()
        assert isinstance(shape, list)
        assert all(isinstance(x, int) for x in shape)
        self.shape = shape
        self.dtype = dtype
        if self.dtype == torch.int64:
            # used for embedding lookups often
            self.value = torch.zeros(self.shape, dtype=self.dtype)
        else:
            self.value = torch.rand(self.shape, dtype=self.dtype)

    def clone(self):
        return self.__class__(self.shape, self.dtype)

    def __str__(self):
        if self.dtype == torch.float32:
            return f"torch.rand({self.shape})"
        elif self.dtype == torch.int64:
            return f"torch.zeros({self.shape}, dtype={self.dtype})"
        else:
            return f"torch.rand({self.shape}, dtype={self.dtype})"

    __repr__ = __str__

    def get_fix(self, error_message: str, pass_number: int, name: str):
        fix = super(TensorGuess, self).get_fix(error_message, pass_number, name)
        if fix:
            return fix

        new_shape = self.apply_fixors(self.shape_fixors(pass_number), error_message)
        if new_shape:
            return self.__class__(new_shape, self.dtype)

        def tried_to_call():
            keywords = ("layer", "activation", "dropout", "normalization")
            for keyword in keywords:
                if keyword in name:
                    return LiteralGuess(torch.nn.ReLU())

        other_fixors = [
            (r"scalar type Long; but got torch.FloatTensor",
             lambda: self.__class__([self.default_size], torch.int64)),
            (r"Expected dtype int64 for index",
             lambda: self.__class__([self.default_size], torch.int64)),
            (r"TypeError: [']Tensor['] object is not callable",
             tried_to_call),
            (r"TypeError: only integer tensors of a single element can be converted to an index",
             lambda: LiteralGuess(0)),
            (r"'lengths' argument should be a 1D CPU int64 tensor",
             lambda: (self.__class__([self.default_size], torch.int64) if "len" in name else None)),
            (r"Boolean value of Tensor with more than one value is ambiguous",
             lambda: LiteralGuess(0)),
        ]
        return self.apply_fixors(other_fixors, error_message)

    def shape_fixors(self, pass_number: int):
        if pass_number == 0:
            return [
                (r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_if_matching),
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_if_matching),
                (r"(?P<want>\d+) channels, but got (?P<got>\d+) channels",
                 self.fix_num_channels),
                (r"same number of dimensions: got (?P<want>\d+) and (?P<got>\d+)",
                 self.fix_dimensions),
                (r"Got (?P<got>\d+)D .*needs (?P<want>\d+)D",
                 self.fix_dimensions),
                (r"input must have (?P<want>\d+) dimensions, got (?P<got>\d+)",
                 self.fix_dimensions),
                (r"Expected (?P<want>\d+)-dimensional tensor, but got (?P<got>\d+)-dimensional tensor",
                 self.fix_dimensions),
                (r"The size.*[(](?P<want>\d+)[)] must match.*[(](?P<got>\d+)[)] at.*dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"must match except in dimension \d+. Got (?P<want>\d+) and (?P<got>\d+) in dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"matrices expected, got (?P<got>\d+)D, (?P<want>\d+)D ",
                 self.fix_dimensions),
                (r"expected \d+D or (?P<want>\d+)D input \(got (?P<got>\d+)D input\)",
                 self.fix_dimensions),
                (r"Expected.*size (?P<want>[\d, ()]+), got (?P<got>[\d, ()]+)",
                 self.fix_shape),
                (r"Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
                 lambda: self.shape[:-1]),
                (r"Expected tensor to have size (?P<want>\d+) at dimension (?P<dim>\d+), but got size (?P<got>\d+)",
                 self.fix_dimensions_at),
                (r"RuntimeError: number of dims don't match in permute",
                 self.fix_dimensions_unknown),
                (r"ValueError: too many values to unpack \(expected (?P<want>\d+)\)",
                 self.fix_dimensions),
                (r"ValueError: not enough values to unpack \(expected (?P<want>\d+), got (?P<got>\d+)\)",
                 self.fix_dimensions),
                (r"expected to be in range of \[-\d+, (?P<got>\d+)\], but got (?P<want>\d+)",
                 self.fix_dimension_out_of_range),
                (r"sizes provided \((?P<want>\d+)\) must be greater or equal.* \((?P<got>\d+)\)",
                 self.fix_dimensions),
                (r"size mismatch, m1: (?P<a>\[.*\]), m2: (?P<b>\[.*\])",
                 self.fix_size_mismatch),
            ]
        if pass_number == 1:
            return [
                (r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input\[[\d, ]+\]",
                 self.fix_convolution),
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*\[[\d, ]+\]",
                 self.fix_convolution),
                (r"same number of dimensions: got (?P<got>\d+) and (?P<want>\d+)",
                 self.fix_dimensions),
                (r"Got \d+D .*needs (?P<want>\d+)D",
                 self.fix_dimensions),
                (r"The size.*[(](?P<got>\d+)[)] must match.*[(](?P<want>\d+)[)] at.*dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"must match except in dimension \d+. Got (?P<got>\d+) and (?P<want>\d+) in dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"expected (?P<want>\d+)D or \d+D input \(got (?P<got>\d+)D input\)",
                 self.fix_dimensions),
                (r"Expected tensor to have size (?P<got>\d+) at dimension (?P<dim>\d+), but got size (?P<want>\d+)",
                 self.fix_dimensions_at),
            ]
        if pass_number == 2:
            return [
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_offset),
            ]

    def fix_size_mismatch(self, a, b):
        # b may be hidden part of a nn.Linear()
        if self.shape[-1] == a[-1]:
            return self.shape[:-1] + [b[0]]

    def fix_dimension_out_of_range(self, got, want):
        if 0 <= got < want:
            return self.fix_dimensions(want + 1, got + 1)

    def fix_shape(self, want, got):
        if self.shape == list(got):
            return list(want)

    def fix_convolution(self, weight: List[int], groups: int = 1):
        return [self.default_size, weight[1] * groups] + [64 for _ in weight[2:]]

    def fix_convolution_if_matching(self, weight, got, groups=1):
        if got == self.shape:
            return self.fix_convolution(weight, groups)

    def fix_convolution_offset(self, weight: List[int], got: List[int]):
        if len(got) == len(self.shape) - 1:
            return [self.default_size, self.default_size, weight[1]] + [64 for _ in weight[2:]]

    def fix_num_channels(self, want, got):
        guess = list(self.shape)
        for idx in (1, 2, 3):
            if len(guess) > idx and guess[idx] == got:
                guess[idx] = want
                return guess
        if len(guess) > 1:
            guess[1] = guess[1] * want // got
            if guess[1] > 0:
                return guess

    def fix_dimensions(self, want, got=None):
        shape = list(self.shape)
        if got is None or len(shape) == got:
            shape.extend([self.default_size] * want)
            return shape[:want]

    def fix_dimensions_at(self, want, got, dim):
        shape = list(self.shape)
        if dim < len(shape) and shape[dim] == got:
            shape[dim] = want
            return shape

    def fix_dimensions_unknown(self):
        shape = list(self.shape)
        if len(shape) > 2:
            shape.pop()
            return


class ConfigGuess(Guess):
    def __init__(self):
        super(ConfigGuess, self).__init__(value=MockConfig())
        self._rollback = []

    def get_fix(self, error_message: str, pass_number: int, name: str):
        guesses = sorted(self.value._guesses.values(),
                         key=lambda x: x.created, reverse=True)
        for guess in guesses:
            if guess.try_to_fix(error_message, pass_number):
                self._rollback.append(guess)
                return self

    def rollback(self):
        self._rollback.pop().rollback()


class MockConfig(object):
    def __init__(self):
        super(MockConfig, self).__init__()
        self._guesses = dict()

    def __str__(self):
        return "_mock_config({})".format(
            ", ".join(f"{key}={repr(value)}" for key, value in self._guesses.items())
        )

    def __getitem__(self, item):
        if item not in self._guesses:
            self._guesses[item] = DeduceParameter.initial_arg_init(name=item, position=None)
        return self._guesses[item].guess()

    __getattr__ = __getitem__

    def __iter__(self):
        return iter([])
