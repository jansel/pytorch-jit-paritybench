import sys
_module = sys.modules[__name__]
del sys
contrib = _module
conftest = _module
sentence_transformers = _module
test_sentence_transformers = _module
timm = _module
test_timm = _module
utils = _module
setup = _module
huggingface_hub = _module
_commit_api = _module
_login = _module
_snapshot_download = _module
_space_api = _module
commands = _module
_cli_utils = _module
delete_cache = _module
env = _module
huggingface_cli = _module
lfs = _module
scan_cache = _module
user = _module
community = _module
constants = _module
fastai_utils = _module
file_download = _module
hf_api = _module
hub_mixin = _module
inference_api = _module
keras_mixin = _module
repocard = _module
repocard_data = _module
repository = _module
_cache_assets = _module
_cache_manager = _module
_chunk_utils = _module
_datetime = _module
_deprecation = _module
_errors = _module
_fixes = _module
_git_credential = _module
_headers = _module
_hf_folder = _module
_http = _module
_pagination = _module
_paths = _module
_runtime = _module
_subprocess = _module
_typing = _module
_validators = _module
endpoint_helpers = _module
logging = _module
sha = _module
tqdm = _module
tests = _module
test = _module
test_cache_layout = _module
test_cache_no_symlinks = _module
test_cli = _module
test_command_delete_cache = _module
test_commit_api = _module
test_endpoint_helpers = _module
test_fastai_integration = _module
test_file_download = _module
test_hf_api = _module
test_hubmixin = _module
test_inference_api = _module
test_init_lazy_loading = _module
test_keras_integration = _module
test_lfs = _module
test_login_utils = _module
test_offline_utils = _module
test_repocard = _module
test_repocard_data = _module
test_repository = _module
test_snapshot_download = _module
test_tf_import = _module
test_utils_assets = _module
test_utils_cache = _module
test_utils_chunks = _module
test_utils_cli = _module
test_utils_datetime = _module
test_utils_deprecation = _module
test_utils_errors = _module
test_utils_fixes = _module
test_utils_git_credentials = _module
test_utils_headers = _module
test_utils_hf_folder = _module
test_utils_http = _module
test_utils_pagination = _module
test_utils_paths = _module
test_utils_runtime = _module
test_utils_sha = _module
test_utils_tqdm = _module
test_utils_validators = _module
testing_constants = _module
testing_utils = _module
check_contrib_list = _module
check_static_imports = _module
push_repocard_examples = _module

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


from typing import List


from typing import Optional


from typing import Union


import re


import time


from typing import Callable


from typing import Iterator


from typing import Tuple


from typing import Any

