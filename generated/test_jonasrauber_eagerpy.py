import sys
_module = sys.modules[__name__]
del sys
eagerpy = _module
astensor = _module
framework = _module
lib = _module
modules = _module
norms = _module
tensor = _module
base = _module
extensions = _module
jax = _module
numpy = _module
pytorch = _module
tensorflow = _module
types = _module
utils = _module
setup = _module
conftest = _module
test_lib = _module
test_main = _module
test_norms = _module

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


from typing import TypeVar


from typing import TYPE_CHECKING


from typing import Union


from typing import overload


from typing import Tuple


from typing import Generic


from typing import Any


import inspect


from types import ModuleType


from typing import Callable


from typing import Iterable


import functools


from typing import cast


from typing import Optional


import numpy as np

