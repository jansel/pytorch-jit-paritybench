import sys
_module = sys.modules[__name__]
del sys
coremltools = _module
_deps = _module
converters = _module
_converters_entry = _module
_profile_utils = _module
libsvm = _module
_libsvm_converter = _module
_libsvm_util = _module
mil = _module
_deployment_compatibility = _module
backend = _module
backend_helper = _module
helper = _module
load = _module
passes = _module
adjust_io_to_supported_types = _module
fuse_activation_silu = _module
insert_image_preprocessing_op = _module
mil_passes = _module
sanitize_name_strings = _module
test_passes = _module
test_helper = _module
test_model_input_params = _module
nn = _module
mil_to_nn_mapping_registry = _module
op_mapping = _module
alert_return_type_cast = _module
commingle_loop_vars = _module
handle_return_inputs_as_outputs = _module
handle_return_unused_inputs = _module
handle_unused_inputs = _module
mlmodel_passes = _module
nn_passes = _module
test_mlmodel_passes = _module
conftest = _module
converter = _module
experimental = _module
generic_conv_batchnorm_fusion = _module
generic_conv_bias_fusion = _module
generic_conv_scale_fusion = _module
generic_layernorm_instancenorm_pattern_fusion = _module
generic_linear_bias_fusion = _module
generic_pass_infrastructure = _module
frontend = _module
_utils = _module
milproto = _module
test_load = _module
tensorflow = _module
basic_graph_ops = _module
convert_utils = _module
dialect_ops = _module
dot_visitor = _module
naming_utils = _module
ops = _module
parse = _module
parsed_tf_node = _module
ssa_passes = _module
backfill_make_list_elem_type = _module
expand_tf_lstm = _module
tf_lstm_to_core_lstm = _module
tf_passes = _module
test = _module
test_composite_ops = _module
test_custom_ops = _module
test_graphs = _module
test_ops = _module
test_parse = _module
test_parsed_tf_node = _module
test_tf_conversion_api = _module
testing_utils = _module
tf_graph_pass = _module
cond_to_where = _module
constant_propagation = _module
delete_asserts = _module
delete_constant = _module
delete_disconnected_nodes = _module
functionalize_loops = _module
fuse_dilation_conv = _module
insert_get_tuple = _module
quantization_pass = _module
tensor_array_transform = _module
variable_node_transform = _module
visitors = _module
tf_op_registry = _module
tfssa = _module
tensorflow2 = _module
remove_vacuous_cond = _module
test_v2_passes = _module
test_tf2_conversion_api = _module
test_v2_load = _module
test_v2_ops = _module
test_v2_ops_tf_keras = _module
rewrite_control_flow_functions = _module
converter = _module
dialect_ops = _module
internal_graph = _module
load = _module
ops = _module
torch_passes = _module
torch_tensor_assign_to_core = _module
torch_upsample_to_core_upsample = _module
test_api = _module
test_custom_ops = _module
test_examples = _module
test_internal_graph = _module
test_passes = _module
test_torch_conversion_api = _module
test_torch_ops = _module
testing_utils = _module
torch_op_registry = _module
torchir_passes = _module
input_types = _module
block = _module
builder = _module
input_type = _module
operation = _module
defs = _module
_op_reqs = _module
iOS15 = _module
activation = _module
classify = _module
control_flow = _module
conv = _module
elementwise_binary = _module
elementwise_unary = _module
image_resizing = _module
linear = _module
normalization = _module
pool = _module
random = _module
recurrent = _module
reduction = _module
scatter_gather = _module
tensor_operation = _module
tensor_transformation = _module
iOS16 = _module
constexpr_ops = _module
tensor_transformation = _module
iOS17 = _module
registry = _module
tests = _module
test_activation = _module
test_const = _module
test_constexpr_ops = _module
test_control_flow = _module
test_conv = _module
test_elementwise_binary = _module
test_elementwise_unary = _module
test_image_resizing = _module
test_linear = _module
test_normalization = _module
test_pool = _module
test_random = _module
test_recurrent = _module
test_reduction = _module
test_scatter_gather = _module
test_slice = _module
test_tensor_operation = _module
test_tensor_transformation = _module
test_utils = _module
add_conv_transpose_output_shape = _module
apply_common_pass_pipeline = _module
cast_optimization = _module
compression_passes = _module
concat_to_pixel_shuffle = _module
const_elimination = _module
conv_batchnorm_fusion = _module
conv_bias_fusion = _module
conv_scale_fusion = _module
dead_code_elimination = _module
dedup_op_and_var_names = _module
detect_concat_interleave = _module
divide_to_multiply = _module
elementwise_batchnorm_fusion = _module
gelu_exact_fusion = _module
gelu_tanh_approximation_fusion = _module
graph_pass = _module
image_input_preprocessing = _module
layernorm_instancenorm_pattern_fusion = _module
leaky_relu_fusion = _module
linear_bias_fusion = _module
loop_invariant_elimination = _module
matmul_weight_bias_fusion = _module
merge_consecutive_paddings = _module
merge_consecutive_relus = _module
name_sanitization_utils = _module
noop_elimination = _module
onehot_matmul_to_gather = _module
pad_conv_connect = _module
pass_registry = _module
prelu_fusion = _module
prelu_to_lrelu = _module
quantization_passes = _module
rank0_expand_dims_swap = _module
reduce_mean_fusion = _module
reduce_transposes = _module
remove_redundant_ops = _module
remove_symbolic_reshape = _module
replace_stack_reshape = _module
sanitize_input_output_names = _module
test_cast_optimization = _module
test_compression_passes = _module
test_concat_to_pixel_shuffle = _module
test_conv_batchnorm_fusion_pass = _module
test_conv_scale_fusion = _module
test_dedup_op_and_var_names = _module
test_elementwise_fusions = _module
test_fp16_compute_precision = _module
test_image_preprocessing = _module
test_layernorm_instancenorm_fusion = _module
test_linear_bias_fusion = _module
test_merge_consecutive_paddings = _module
test_merge_consecutive_relus = _module
test_noop_elimination = _module
test_pad_conv_pass = _module
test_reduce_transposes_pass = _module
test_replace_stack_reshape = _module
test_use_reflection_padding = _module
topological_reorder = _module
update_output_dtypes = _module
use_reflection_padding = _module
program = _module
test_block = _module
test_programs = _module
types = _module
annotate = _module
get_type_info = _module
global_methods = _module
symbolic = _module
type_bool = _module
type_dict = _module
type_double = _module
type_globals_pseudo_type = _module
type_int = _module
type_list = _module
type_mapping = _module
type_spec = _module
type_str = _module
type_tensor = _module
type_tuple = _module
type_unknown = _module
type_void = _module
var = _module
test_flexible_shape_inputs = _module
testing_reqs = _module
_LinearSVC = _module
_LinearSVR = _module
_NuSVC = _module
_NuSVR = _module
_SVC = _module
_SVR = _module
sklearn = _module
_converter = _module
_converter_internal = _module
_decision_tree_classifier = _module
_decision_tree_regressor = _module
_dict_vectorizer = _module
_gradient_boosting_classifier = _module
_gradient_boosting_regressor = _module
_imputer = _module
_k_neighbors_classifier = _module
_linear_regression = _module
_logistic_regression = _module
_normalizer = _module
_one_hot_encoder = _module
_random_forest_classifier = _module
_random_forest_regressor = _module
_ridge_regression = _module
_sklearn_util = _module
_standard_scaler = _module
_svm_common = _module
_tree_ensemble = _module
xgboost = _module
_tree = _module
models = _module
_deprecation = _module
_feature_management = _module
_interface_management = _module
array_feature_extractor = _module
datatypes = _module
feature_vectorizer = _module
ml_program = _module
compression_utils = _module
model = _module
nearest_neighbors = _module
neural_network = _module
flexible_shape_utils = _module
optimization_utils = _module
printer = _module
quantization_utils = _module
spec_inspection_utils = _module
update_optimizer_utils = _module
utils = _module
pipeline = _module
tree_ensemble = _module
ArrayFeatureExtractor_pb2 = _module
AudioFeaturePrint_pb2 = _module
BayesianProbitRegressor_pb2 = _module
CategoricalMapping_pb2 = _module
CustomModel_pb2 = _module
DataStructures_pb2 = _module
DictVectorizer_pb2 = _module
FeatureTypes_pb2 = _module
FeatureVectorizer_pb2 = _module
GLMClassifier_pb2 = _module
GLMRegressor_pb2 = _module
Gazetteer_pb2 = _module
Identity_pb2 = _module
Imputer_pb2 = _module
ItemSimilarityRecommender_pb2 = _module
LinkedModel_pb2 = _module
MIL_pb2 = _module
Model_pb2 = _module
NamedParameters_pb2 = _module
NearestNeighbors_pb2 = _module
NeuralNetwork_pb2 = _module
NonMaximumSuppression_pb2 = _module
Normalizer_pb2 = _module
OneHotEncoder_pb2 = _module
Parameters_pb2 = _module
SVM_pb2 = _module
Scaler_pb2 = _module
SoundAnalysisPreprocessing_pb2 = _module
TextClassifier_pb2 = _module
TreeEnsemble_pb2 = _module
VisionFeaturePrint_pb2 = _module
WordEmbedding_pb2 = _module
WordTagger_pb2 = _module
proto = _module
api = _module
test_api_examples = _module
test_api_visibilities = _module
blob = _module
test_weights = _module
test_compression = _module
modelpackage = _module
test_mlmodel = _module
test_modelpackage = _module
test_custom_neural_nets = _module
test_model = _module
test_neural_networks = _module
test_nn_builder = _module
test_numpy_nn_layers = _module
test_quantization = _module
test_simple_nn_inference = _module
test_tf_numeric = _module
test_model_updatable = _module
test_pipeline = _module
sklearn_tests = _module
test_NuSVC = _module
test_NuSVR = _module
test_SVC = _module
test_SVR = _module
test_categorical_imputer = _module
test_composite_pipelines = _module
test_dict_vectorizer = _module
test_feature_names = _module
test_glm_classifier = _module
test_imputer = _module
test_io_types = _module
test_k_neighbors_classifier = _module
test_linear_regression = _module
test_nearest_neighbors_builder = _module
test_normalizer = _module
test_one_hot_encoder = _module
test_random_forest_classifier = _module
test_random_forest_classifier_numeric = _module
test_random_forest_regression = _module
test_random_forest_regression_numeric = _module
test_ridge_regression = _module
test_standard_scalar = _module
xgboost_tests = _module
test_boosted_trees_classifier = _module
test_boosted_trees_classifier_numeric = _module
test_boosted_trees_regression = _module
test_boosted_trees_regression_numeric = _module
test_decision_tree_classifier = _module
test_decision_tree_classifier_numeric = _module
test_decision_tree_regression = _module
test_decision_tree_regression_numeric = _module
version = _module
conformance_python = _module
update_failure_list = _module
add_person = _module
list_people = _module
generate_changelog = _module
make_test_output = _module
pddm = _module
pddm_tests = _module
setup = _module
google = _module
protobuf = _module
internal = _module
descriptor_test = _module
generator_test = _module
message_test = _module
service_reflection_test = _module
test_util = _module
text_format_test = _module
wire_format_test = _module
descriptor = _module
descriptor_database = _module
descriptor_pool = _module
_parameterized = _module
api_implementation = _module
containers = _module
decoder = _module
descriptor_database_test = _module
descriptor_pool_test = _module
encoder = _module
enum_type_wrapper = _module
import_test_package = _module
json_format_test = _module
message_factory_test = _module
message_listener = _module
proto_builder_test = _module
python_message = _module
reflection_test = _module
symbol_database_test = _module
testing_refleaks = _module
text_encoding_test = _module
type_checkers = _module
unknown_fields_test = _module
well_known_types = _module
well_known_types_test = _module
wire_format = _module
json_format = _module
message = _module
message_factory = _module
proto_builder = _module
pyext = _module
cpp_message = _module
reflection = _module
service = _module
service_reflection = _module
symbol_database = _module
text_encoding = _module
text_format = _module
mox = _module
stubout = _module
benchmark = _module
conf = _module
pybind11 = _module
_version = _module
test_alias_initialization = _module
test_buffers = _module
test_callbacks = _module
test_chrono = _module
test_class_args = _module
test_constants_and_functions = _module
test_copy_move_policies = _module
test_docstring_options = _module
test_eigen = _module
test_enum = _module
test_eval = _module
test_eval_call = _module
test_exceptions = _module
test_inheritance = _module
test_issues = _module
test_keep_alive = _module
test_kwargs_and_defaults = _module
test_methods_and_attributes = _module
test_modules = _module
test_multiple_inheritance = _module
test_numpy_array = _module
test_numpy_dtypes = _module
test_numpy_vectorize = _module
test_opaque_types = _module
test_operator_overloading = _module
test_pickling = _module
test_python_types = _module
test_sequences_and_iterators = _module
test_smart_ptr = _module
test_stl_binders = _module
test_virtual_functions = _module
libsize = _module
mkdoc = _module
preprocess = _module
readme_session = _module
upload_docs = _module
build_tag = _module

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


import re as _re


import collections


import warnings as _warnings


import numpy as np


import torch


from collections import OrderedDict


import numpy as _np


import torch as _torch


from itertools import islice


import math as _math


import numbers


from collections.abc import Iterable


import torchvision


import torch.nn as nn


import itertools


import torch.nn.functional as F


from typing import Tuple


from collections import defaultdict


import scipy


import functools


from copy import deepcopy as _deepcopy


import numpy as _numpy


import math


import random


import uuid


class _invalid_placeholder_type:
    pass


def annotate(return_type=_invalid_placeholder_type, **kwargs):
    """
    A decorator that informs the compyler about the return type of a function
    and a collection of hint for other variable names. These can include
     - captured variables
     - function arguments
     - other variables within the function

    Ex:

        @annotate(compyler.double, a=compyler.double, b=compyler.double)
        def add(a, b):

    In certain cases when the class members are annotated this does not work.
    For instance this fails because the annotate decorator is called before
    the class double is fully defined.

        class double:
            @annotate(double, other=double)
            def __add__(self, other):

     So it is necessary to add one level of laziness and delay the type

        @class_annotate()
        class double:
            @annotate(delay_type.double, other=delay_type.double)
            def __add__(self, other):

    After which apply_delayed_types() must be called to fill in the delayed
    type.
    """
    global annotated_function_list

    def decorator(func):
        global annotated_function_list
        func.type_annotations = kwargs
        if return_type is not _invalid_placeholder_type:
            func.return_type = return_type
        annotated_function_list += [func]
        return func
    return decorator


class FunctionType:
    """
    - FunctionType.inputs : A list of Type objects defining the types of the input
    - FunctionType.output: A Type object defining the type of the output
    - FunctionType.python_function : The original python function implementing
                                     this type. Two FunctionType objects compare
                                     equal only on inputs and output and not
                                     python_function
    """
    __slots__ = ['inputs', 'output', 'python_function']

    def __init__(self, inputs, output, python_function=None):
        assert isinstance(inputs, list)
        assert isinstance(output, (FunctionType, Type))
        self.inputs = inputs
        self.output = output
        self.python_function = python_function

    def __hash__(self):
        return hash((tuple(self.inputs), self.output))

    def __eq__(self, other):
        return self.inputs == other.inputs and self.output == other.output

    def __repr__(self):
        return '(' + ','.join(repr(x) for x in self.inputs) + ')->' + repr(self.output)

    def __str__(self):
        return self.__repr__()

    def return_sexp(self):
        return self.output.sexp()

    def inputs_sexp(self):
        return [i.sexp() for i in self.inputs]


def class_annotate():
    """
    Registers a class to be used by delay_type. See annotate()
    """
    global annotated_class_list

    def decorator(cls):
        global annotated_class_list
        annotated_class_list[cls.__name__] = cls
        return cls
    return decorator


class delay_type_cls:

    def __getattr__(self, t):
        return t


delay_type = delay_type_cls()


class void:

    @classmethod
    def __type_info__(cls):
        return Type('void', python_class=cls)


def get_python_method_type(py_function):
    function_inputs = []
    function_output = get_type_info(void)
    annotations = {}
    if hasattr(py_function, 'type_annotations'):
        annotations = {k: get_type_info(v) for k, v in py_function.type_annotations.items()}
    if hasattr(py_function, 'return_type'):
        function_output = get_type_info(py_function.return_type)
    try:
        if hasattr(py_function, '__func__'):
            argcount = py_function.__func__.__code__.co_argcount
            argnames = py_function.__func__.__code__.co_varnames[:argcount]
        else:
            argcount = py_function.__code__.co_argcount
            argnames = py_function.__code__.co_varnames[:argcount]
    except:
        raise TypeError('Unable to derive type information from method %s. You might have a misspecified type. Ex: use compyler.int and not int' % py_function)
    for arg in argnames:
        if arg in annotations:
            function_inputs.append(annotations[arg])
        elif arg != 'self':
            raise TypeError('Function ' + str(py_function) + ' insufficient annotations. ' + arg + ' needs a type')
    typeinfo = FunctionType(function_inputs, function_output, py_function)
    return typeinfo


def get_type_info(t):
    if hasattr(t, '__type_info__'):
        ret = t.__type_info__()
        assert ret.python_class is not None
        return ret
    elif isinstance(t, type):
        return Type(t.__name__, python_class=t)
    elif hasattr(t, '__call__'):
        return get_python_method_type(t)
    raise TypeError('Unsupported type %s' % t)


def memoize(f):
    memo = {}

    def helper(x, y):
        if (x, y) not in memo:
            memo[x, y] = f(x, y)
        return memo[x, y]
    return helper


@memoize
def list(arg, init_length=None, dynamic_length=True):


    class list:
        T = [arg, init_length, dynamic_length]

        def __init__(self):
            self.val = []

        @classmethod
        def __type_info__(cls):
            return Type('list', [get_type_info(arg)], python_class=cls)

        @annotate(void, other=T[0])
        def append(self, other):
            assert isinstance(other, self.T[0])
            self.val.append(other)

        @annotate(T[0], index=type_int.int64)
        def __getitem__(self, index):
            assert isinstance(index, type_int.int64)
            return self.val[index.val]

        @annotate(void, index=type_int.int64, newval=T[0])
        def __setitem__(self, index, newval):
            assert isinstance(index, type_int.int64)
            assert isinstance(newval, self.T[0])
            self.val[index.val] = newval

        @annotate(type_int.int64)
        def __len__(self):
            return type_int.int64(len(self.val)) if self.T[1] is None else self.T[1]
    list.__template_name__ = 'list[' + arg.__name__ + ']'
    return list


class Type:
    """
     - Type.name : A string with the name of the object
     - Type.tparam : For classes with template parameters, (list, dict), this
         contains a list of Type objects of the template parameters
     - Type.python_class : The original python class implementing this type.
                         Two Type objects compare equal
                         only on name and tparam and not python_class
    """
    __slots__ = ['name', 'tparam', 'python_class']

    def __init__(self, name, tparam=None, python_class=None):
        if tparam is None:
            tparam = []
        assert isinstance(name, str)
        assert isinstance(tparam, list)
        self.name = name
        self.tparam = tparam
        self.python_class = python_class

    def __hash__(self):
        return hash((self.name, tuple(self.tparam)))

    def __eq__(self, other):
        return self.name == other.name and self.tparam == other.tparam

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        ret = self.name
        if len(self.tparam) > 0:
            ret += '[' + ','.join(repr(x) for x in self.tparam) + ']'
        return ret

    def __str__(self):
        return self.__repr__()

    def sexp(self):
        if len(self.tparam) == 0:
            return self.name
        else:
            ret = [self.name]
            ret.append([(a.sexp() if hasattr(a, 'sexp') else a) for a in self.tparam])
            return ret


_global_tuple = tuple


@memoize
def tuple(args):
    args = _global_tuple(i if i is not None else type_unknown.unknown for i in args)


    class tuple:
        T = args

        def __init__(self):
            self.val = [arg() for arg in args]

        @classmethod
        def __type_info__(cls):
            return Type('tuple', [get_type_info(arg) for arg in args], python_class=cls)

        @annotate(type_int.int64)
        def __len__(self):
            return len(args)
    tuple.__template_name__ = 'tuple[' + ','.join([get_type_info(arg).name for arg in args]) + ']'
    return tuple


class StripCellAndHidden(nn.Module):

    def __init__(self, flagReturnTuple_):
        super(StripCellAndHidden, self).__init__()
        self.flagReturnTuple = flagReturnTuple_

    def forward(self, x):
        return tuple(x[0]) if self.flagReturnTuple else x[0]


class ModuleWrapper(nn.Module):
    """
    Helper class to transform torch function into torch nn module.
    This helps to keep the testing interface same for torch functional api.
    """

    def __init__(self, function, kwargs=None):
        super(ModuleWrapper, self).__init__()
        self.function = function
        self.kwargs = kwargs if kwargs else {}

    def forward(self, *args):
        return self.function(*args, **self.kwargs)

