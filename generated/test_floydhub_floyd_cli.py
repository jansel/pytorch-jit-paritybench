import sys
_module = sys.modules[__name__]
del sys
floyd = _module
cli = _module
auth = _module
data = _module
data_upload_utils = _module
experiment = _module
run = _module
utils = _module
version = _module
client = _module
base = _module
common = _module
dataset = _module
env = _module
files = _module
module = _module
project = _module
resource = _module
task_instance = _module
tus_data = _module
constants = _module
date_utils = _module
development = _module
dev = _module
local = _module
exceptions = _module
log = _module
main = _module
manager = _module
auth_config = _module
data_config = _module
experiment_config = _module
floyd_ignore = _module
login = _module
model = _module
access_token = _module
credentials = _module
user = _module
setup = _module
tests = _module
auth_test = _module
delete_test = _module
mocks = _module
test_run = _module
utils_test = _module
version_test = _module
get_unignored_file_paths_test = _module
matches_glob_list_test = _module
auth_config_test = _module
get_lists_test = _module
access_token_test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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

