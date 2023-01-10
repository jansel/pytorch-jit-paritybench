import sys
_module = sys.modules[__name__]
del sys
conf = _module
add_header = _module
make_rst_table = _module
test_headers = _module
hydra_zen = _module
_compatibility = _module
_hydra_overloads = _module
_launch = _module
_utils = _module
coerce = _module
errors = _module
funcs = _module
structured_configs = _module
_globals = _module
_implementations = _module
_just = _module
_make_config = _module
_make_custom_builds = _module
_type_guards = _module
_utils = _module
_value_conversion = _module
third_party = _module
beartype = _module
pydantic = _module
typing = _module
_builds_overloads = _module
wrapper = _module
tests = _module
behaviors = _module
declarations = _module
mypy_checks = _module
conftest = _module
custom_strategies = _module
dummy_zen_main = _module
pyright_utils = _module
test_hydra_supports_partial = _module
test_old_partial_implementation = _module
test_omegaconf_830 = _module
test_primitive_support = _module
test_config_value_validation = _module
test_custom_strategies = _module
test_dataclass_conversion = _module
test_dataclass_semantics = _module
test_defaults_list = _module
test_docs_typecheck = _module
test_documented_examples = _module
test_hydra_behaviors = _module
test_hydra_overloads = _module
test_hydrated_dataclass = _module
test_just = _module
test_callbacks = _module
test_implementations = _module
test_logging = _module
test_validation = _module
test_make_config = _module
test_make_custom_builds_fn = _module
test_project = _module
test_protocols = _module
test_py310 = _module
test_py36 = _module
test_py39 = _module
test_pyright_utils = _module
test_roundtrips = _module
test_signature_parsing = _module
test_store = _module
test_against_jax = _module
test_against_numpy = _module
test_against_pytorch_lightning = _module
test_against_torch = _module
test_type_validators = _module
test_using_beartype = _module
test_using_pydantic = _module
test_utils = _module
test_sequence_coercion = _module
test_validation_py38 = _module
test_value_conversion = _module
test_with_hydra_submitit = _module
test_zen_decorator = _module
test_inheritance = _module
test_meta_fields = _module
test_zen_wrappers = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import inspect


import warnings


from enum import Enum


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import List


from typing import Mapping


from typing import Optional


from typing import Sequence


from typing import Set


from typing import Tuple


from typing import Type


from typing import TypeVar


from typing import Union


from typing import cast


from typing import overload


from collections import Counter


from collections import deque


from functools import partial


import logging


from copy import deepcopy


from abc import ABC


from inspect import Parameter


from typing import get_type_hints


import torch as tr


from torch.optim import Adam


from torch.optim import AdamW


from torch.utils.data import DataLoader


from torch.utils.data import Dataset

