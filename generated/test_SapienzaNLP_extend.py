import sys
_module = sys.modules[__name__]
del sys
extend = _module
data = _module
data_drivers = _module
esc_ed_dataset = _module
demo = _module
serve = _module
ui = _module
esc_ed_module = _module
evaluation = _module
save_transformer_weights = _module
spacy_test = _module
tsv_to_sqlite = _module
spacy_component = _module
utils = _module
commons = _module
sqlite3_mentions_inventory = _module
setup = _module

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


from typing import Dict


from typing import Any


from typing import Tuple


from typing import Callable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Iterable


import numpy as np


import torch


import logging


from typing import Union


from typing import Set

