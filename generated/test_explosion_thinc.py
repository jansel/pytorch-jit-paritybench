import sys
_module = sys.modules[__name__]
del sys
lstm_tagger = _module
mnist = _module
transformers_tagger = _module
type_checking = _module
setup = _module
thinc = _module
about = _module
api = _module
backends = _module
_cupy_allocators = _module
_custom_kernels = _module
_param_server = _module
cupy_ops = _module
jax_ops = _module
ops = _module
config = _module
extra = _module
tests = _module
initializers = _module
layers = _module
add = _module
array_getitem = _module
bidirectional = _module
cauchysimilarity = _module
chain = _module
clone = _module
concatenate = _module
dropout = _module
embed = _module
expand_window = _module
featureextractor = _module
hashembed = _module
layernorm = _module
linear = _module
list2array = _module
list2padded = _module
list2ragged = _module
logistic = _module
lstm = _module
maxout = _module
mish = _module
multisoftmax = _module
mxnetwrapper = _module
noop = _module
padded2list = _module
parametricattention = _module
pytorchwrapper = _module
ragged2list = _module
reduce_max = _module
reduce_mean = _module
reduce_sum = _module
relu = _module
remap_ids = _module
residual = _module
siamese = _module
softmax = _module
softmax_activation = _module
staticvectors = _module
strings2arrays = _module
tensorflowwrapper = _module
uniqued = _module
with_array = _module
with_cpu = _module
with_debug = _module
with_flatten = _module
with_getitem = _module
with_list = _module
with_padded = _module
with_ragged = _module
with_reshape = _module
loss = _module
model = _module
mypy = _module
optimizers = _module
schedules = _module
shims = _module
mxnet = _module
pytorch = _module
shim = _module
tensorflow = _module
test_lstm = _module
test_mem = _module
test_ops = _module
conftest = _module
test_beam_search = _module
test_basic_tagger = _module
test_combinators = _module
test_feature_extractor = _module
test_feed_forward = _module
test_hash_embed = _module
test_layers_api = _module
test_linear = _module
test_mnist = _module
test_mxnet_wrapper = _module
test_pytorch_wrapper = _module
test_sparse_linear = _module
test_tensorflow_wrapper = _module
test_transforms = _module
test_uniqued = _module
test_with_debug = _module
test_with_transforms = _module
test_jax_model = _module
test_model = _module
test_validation = _module
modules = _module
fail_no_plugin = _module
fail_plugin = _module
success_no_plugin = _module
success_plugin = _module
test_mypy = _module
regression = _module
test_issue208 = _module
strategies = _module
test_config = _module
test_examples = _module
test_indexing = _module
test_initializers = _module
test_loss = _module
test_optimizers = _module
test_schedules = _module
test_serialize = _module
test_types = _module
test_util = _module
util = _module
types = _module

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


from typing import Optional


from typing import Tuple


from typing import Callable


from typing import cast


from functools import partial


import numpy


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_explosion_thinc(_paritybench_base):
    pass
