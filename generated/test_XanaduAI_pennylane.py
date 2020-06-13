import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
benchmark_revisions = _module
benchmark_utils = _module
bm_entangling_layers = _module
bm_iqp_circuit = _module
bm_mutable_complicated_params = _module
bm_mutable_rotations = _module
edit_on_github = _module
conf = _module
directives = _module
pennylane = _module
_device = _module
_qubit_device = _module
_queuing_context = _module
_version = _module
about = _module
beta = _module
plugins = _module
default_tensor = _module
default_tensor_tf = _module
numpy_ops = _module
circuit_drawer = _module
charsets = _module
grid = _module
representation_resolver = _module
circuit_graph = _module
collections = _module
apply = _module
dot = _module
map = _module
qnode_collection = _module
sum = _module
configuration = _module
init = _module
interfaces = _module
autograd = _module
tf = _module
torch = _module
io = _module
measure = _module
numpy = _module
fft = _module
linalg = _module
random = _module
tensor = _module
wrapper = _module
operation = _module
ops = _module
cv = _module
qubit = _module
optimize = _module
adagrad = _module
adam = _module
gradient_descent = _module
momentum = _module
nesterov_momentum = _module
qng = _module
rms_prop = _module
rotoselect = _module
rotosolve = _module
default_gaussian = _module
default_qubit = _module
default_qubit_tf = _module
tf_ops = _module
qnn = _module
cost = _module
keras = _module
torch = _module
qnodes = _module
base = _module
decorator = _module
device_jacobian = _module
jacobian = _module
passthru = _module
rev = _module
templates = _module
broadcast = _module
embeddings = _module
amplitude = _module
angle = _module
basis = _module
displacement = _module
iqp = _module
qaoa = _module
squeezing = _module
layers = _module
basic_entangler = _module
cv_neural_net = _module
simplified_two_design = _module
strongly_entangling = _module
state_preparations = _module
arbitrary_state_preparation = _module
mottonen = _module
subroutines = _module
arbitrary_unitary = _module
double_excitation_unitary = _module
interferometer = _module
single_excitation_unitary = _module
uccsd = _module
utils = _module
variable = _module
vqe = _module
wires = _module
pennylane_qchem = _module
qchem = _module
setup = _module
tests = _module
conftest = _module
test_active_space = _module
test_convert_hamiltonian = _module
test_decompose_hamiltonian = _module
test_generate_hamiltonian = _module
test_hf_state = _module
test_meanfield_data = _module
test_read_structure = _module
test_sd_excitations = _module
test_default_tensor = _module
test_default_tensor_tf = _module
test_circuit_drawer = _module
test_grid = _module
test_representation_resolver = _module
test_circuit_graph = _module
test_circuit_graph_hash = _module
test_qasm = _module
test_collections = _module
test_qnode_collection = _module
gate_data = _module
test_autograd = _module
test_tf = _module
test_torch = _module
test_cv_ops = _module
test_qubit_ops = _module
test_default_gaussian = _module
test_default_qubit = _module
test_default_qubit_tf = _module
test_cost = _module
test_keras = _module
test_qnn_torch = _module
test_qnode_base = _module
test_qnode_cv = _module
test_qnode_decorator = _module
test_qnode_jacobian = _module
test_qnode_metric_tensor = _module
test_qnode_passthru = _module
test_qnode_qubit = _module
test_qnode_reversible = _module
test_broadcast = _module
test_decorator = _module
test_embeddings = _module
test_integration = _module
test_layers = _module
test_state_preparations = _module
test_subroutines = _module
test_templ_utils = _module
test_about = _module
test_classical_gradients = _module
test_configuration = _module
test_device = _module
test_hermitian_edge_cases = _module
test_init = _module
test_io = _module
test_measure = _module
test_numpy_wrapper = _module
test_observable = _module
test_operation = _module
test_optimize = _module
test_optimize_qng = _module
test_prob = _module
test_quantum_gradients = _module
test_qubit_device = _module
test_queuing_context = _module
test_tensor_measurements = _module
test_utils = _module
test_variable = _module
test_vqe = _module
test_wires = _module

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


import functools


import inspect


import math


from collections.abc import Iterable


from typing import Callable


from typing import Optional


import numpy as np


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_XanaduAI_pennylane(_paritybench_base):
    pass
