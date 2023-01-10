import sys
_module = sys.modules[__name__]
del sys
conf = _module
minify = _module
comment_eater = _module
compiler_unparse = _module
docscrape = _module
docscrape_sphinx = _module
linkcode = _module
numpydoc = _module
phantom_import = _module
plot_directive = _module
sphinx_gallery = _module
backreferences = _module
docs_resolv = _module
downloads = _module
gen_gallery = _module
gen_rst = _module
notebook = _module
py_source_parser = _module
sorting = _module
sphinx_compatibility = _module
tests = _module
conftest = _module
test_backreferences = _module
test_docs_resolv = _module
test_gen_gallery = _module
test_gen_rst = _module
test_notebook = _module
test_sorting = _module
test_sphinx_compatibility = _module
plot_IL2 = _module
plot_covid = _module
plot_image_compression = _module
plot_cp_line_search = _module
plot_guide_for_constrained_cp = _module
plot_nn_cp_hals = _module
plot_nn_tucker = _module
plot_parafac2 = _module
plot_permute_factors = _module
plot_tensor = _module
plot_cp_regression = _module
plot_tucker_regression = _module
setup = _module
tensorly = _module
_factorized_tensor = _module
backend = _module
core = _module
cupy_backend = _module
jax_backend = _module
mxnet_backend = _module
numpy_backend = _module
pytorch_backend = _module
tensorflow_backend = _module
base = _module
contrib = _module
decomposition = _module
_tt_cross = _module
test_mps_decomposition_cross = _module
sparse = _module
cp_tensor = _module
tenalg = _module
test_decomposition = _module
test_tenalg = _module
datasets = _module
data_imports = _module
synthetic = _module
test_imports = _module
test_synthetic = _module
_base_decomposition = _module
_cmtf_als = _module
_constrained_cp = _module
_cp = _module
_cp_power = _module
_nn_cp = _module
_parafac2 = _module
_symmetric_cp = _module
_tr = _module
_tt = _module
_tucker = _module
robust_decomposition = _module
test_cmtf_als = _module
test_constrained_parafac = _module
test_cp = _module
test_cp_power = _module
test_parafac2 = _module
test_robust_decomposition = _module
test_symmetric_cp = _module
test_tr_decomposition = _module
test_tt_decomposition = _module
test_tucker = _module
metrics = _module
entropy = _module
factors = _module
regression = _module
similarity = _module
test_entropy = _module
test_factors = _module
test_regression = _module
test_similarity = _module
parafac2_tensor = _module
random = _module
test_base = _module
cp_plsr = _module
cp_regression = _module
test_cp_plsr = _module
test_cp_regression = _module
test_tucker_regression = _module
tucker_regression = _module
base_tenalg = _module
core_tenalg = _module
_batched_tensordot = _module
_khatri_rao = _module
_kronecker = _module
_tt_matrix = _module
generalised_inner_product = _module
moments = _module
mttkrp = _module
n_mode_product = _module
outer_product = _module
einsum_tenalg = _module
proximal = _module
svd = _module
tenalg_utils = _module
test_batched_tensordot = _module
test_generalised_inner_product = _module
test_khatri_rao = _module
test_kronecker = _module
test_n_mode_product = _module
test_outer_product = _module
test_proximal = _module
test_tenalg_utils = _module
test_unfolding_dot_khatri_rao = _module
testing = _module
test_backend = _module
test_core = _module
test_cp_tensor = _module
test_parafac2_tensor = _module
test_testing = _module
test_tr_tensor = _module
test_tt_matrix = _module
test_tt_tensor = _module
test_tucker_tensor = _module
tr_tensor = _module
tt_matrix = _module
tt_tensor = _module
tucker_tensor = _module
utils = _module
_prod = _module
deprecation = _module
test_deprecation = _module
test_prod = _module

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


import warnings


import numpy as np


from time import time


from scipy.linalg import svd

