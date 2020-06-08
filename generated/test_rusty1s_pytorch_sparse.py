import sys
_module = sys.modules[__name__]
del sys
main = _module
rename_wheel = _module
setup = _module
test = _module
test_cat = _module
test_coalesce = _module
test_convert = _module
test_diag = _module
test_eye = _module
test_matmul = _module
test_metis = _module
test_overload = _module
test_padding = _module
test_permute = _module
test_saint = _module
test_spmm = _module
test_spspmm = _module
test_storage = _module
test_transpose = _module
utils = _module
torch_sparse = _module
add = _module
bandwidth = _module
cat = _module
coalesce = _module
convert = _module
diag = _module
eye = _module
index_select = _module
masked_select = _module
matmul = _module
metis = _module
mul = _module
narrow = _module
padding = _module
permute = _module
reduce = _module
rw = _module
saint = _module
sample = _module
select = _module
spmm = _module
spspmm = _module
storage = _module
tensor = _module
transpose = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rusty1s_pytorch_sparse(_paritybench_base):
    pass
