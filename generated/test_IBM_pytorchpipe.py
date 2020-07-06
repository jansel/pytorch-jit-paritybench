import sys
_module = sys.modules[__name__]
del sys
ptp = _module
application = _module
component_factory = _module
pipeline_manager = _module
sampler_factory = _module
task_manager = _module
component = _module
language = _module
bow_encoder = _module
label_indexer = _module
sentence_indexer = _module
sentence_one_hot_encoder = _module
sentence_tokenizer = _module
word_decoder = _module
losses = _module
loss = _module
nll_loss = _module
masking = _module
join_masked_predictions = _module
string_to_mask = _module
embeddings = _module
io = _module
word_mappings = _module
models = _module
attention_decoder = _module
feed_forward_network = _module
recurrent_neural_network = _module
seq2seq = _module
index_embeddings = _module
sentence_embeddings = _module
model = _module
compact_bilinear_pooling = _module
factorized_bilinear_pooling = _module
low_rank_bilinear_pooling = _module
question_driven_attention = _module
relational_network = _module
self_attention = _module
convnet_encoder = _module
generic_image_encoder = _module
lenet5 = _module
publishers = _module
global_variable_publisher = _module
stream_file_exporter = _module
statistics = _module
accuracy_statistics = _module
batch_size_statistics = _module
bleu_statistics = _module
precision_recall_statistics = _module
image_text_to_class = _module
clevr = _module
gqa = _module
vqa_med_2019 = _module
image_to_class = _module
cifar_100 = _module
mnist = _module
simple_molecules = _module
task = _module
text_to_class = _module
dummy_language_identification = _module
language_identification = _module
wily_language_identification = _module
wily_ngram_language_modeling = _module
text_to_text = _module
translation_pairs = _module
wikitext_language_modeling = _module
transforms = _module
concatenate_tensor = _module
list_to_tensor = _module
reduce_tensor = _module
reshape_tensor = _module
viewers = _module
image_viewer = _module
stream_viewer = _module
configuration = _module
config_interface = _module
config_parsing = _module
config_registry = _module
configuration_error = _module
data_types = _module
data_definition = _module
data_streams = _module
utils = _module
app_state = _module
data_streams_parallel = _module
globals_facade = _module
key_mappings_facade = _module
logger = _module
samplers = _module
singleton = _module
statistics_aggregator = _module
statistics_collector = _module
termination_condition = _module
workers = _module
offline_trainer = _module
online_trainer = _module
processor = _module
test_data_dict_parallel = _module
trainer = _module
worker = _module
setup = _module
tests = _module
pipeline_tests = _module
sampler_factory_tests = _module
samplers_tests = _module
component_tests = _module
clevr_tests = _module
gqa_tests = _module
task_tests = _module
config_interface_tests = _module
config_registry_tests = _module
handshaking_tests = _module
data_definition_tests = _module
data_streams_tests = _module
app_state_tests = _module
statistics_tests = _module

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


import torch


from numpy import inf


from numpy import average


import numpy as np


import torch.utils.data.sampler as pt_samplers


import logging


from torch.utils.data import DataLoader


import torch.nn as nn


from torch.nn import Module


import torch.nn.functional as F


import torchvision.models as models


import math


from torchvision import transforms


import string


from torchvision import datasets


from torch.utils.data import Dataset


import collections


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.parallel_apply import parallel_apply


from math import ceil


from torch._six import int_classes as _int_classes


from torch.utils.data.sampler import Sampler


from time import sleep


import time


from random import randrange


from abc import abstractmethod


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AppState(metaclass=SingletonMetaClass):
    """
    Represents the application state. A singleton that can be accessed by calling:

        >>> app_state = AppState()

    Contains global variables that can be accessed with standard setted/getter methods:

        >>> app_state["test1"] = 1 
        >>> app_state["test2"] = 2
        >>> print(app_state["test1"])

    .. warning::
        It is assumed that global variables are immutable, i.e. once a variable is set, it cannot be changed        

            >>> app_state["test1"] = 3 # Raises AtributeError

    Additionally, it stores all properly parsed commandline arguments.
    """

    def __init__(self):
        """
        Constructor. Initializes dictionary with global variables, sets CPU types as default.

        """
        self.args = None
        self.__globals = dict()
        ptp_path = path.expanduser('~/.ptp/')
        with open(path.join(ptp_path, 'config.txt')) as file:
            self.absolute_config_path = file.readline()
        self.log_file = None
        self.logger = None
        self.log_dir = path.expanduser('.')
        self.set_cpu_types()
        self.use_gpu = False
        self.use_dataparallel = False
        self.device = torch.device('cpu')
        self.epoch = None
        self.episode = 0

    def set_types(self):
        """
        Enables computations on CUDA if GPU is available.
        Sets the default data types.
        """
        if torch.cuda.is_available() and self.args.use_gpu:
            self.logger.info('Running computations on GPU using CUDA')
            self.set_gpu_types()
            self.use_gpu = True
            self.device = torch.device('cuda')
            if torch.cuda.device_count() > 1:
                self.use_dataparallel = True
        elif self.args.use_gpu:
            self.logger.warning('GPU utilization is demanded but there are no available GPU devices! Using CPUs instead')
        else:
            self.logger.info('GPU utilization is disabled, performing all computations on CPUs')

    def set_cpu_types(self):
        """
        Sets all tensor types to CPU data types.
        """
        self.FloatTensor = torch.FloatTensor
        self.DoubleTensor = torch.DoubleTensor
        self.HalfTensor = torch.HalfTensor
        self.ByteTensor = torch.ByteTensor
        self.CharTensor = torch.CharTensor
        self.ShortTensor = torch.ShortTensor
        self.IntTensor = torch.IntTensor
        self.LongTensor = torch.LongTensor

    def set_gpu_types(self):
        """
        Sets all tensor types to GPU/CUDA data types.
        """
        self.FloatTensor = torch.FloatTensor
        self.DoubleTensor = torch.DoubleTensor
        self.HalfTensor = torch.HalfTensor
        self.ByteTensor = torch.ByteTensor
        self.CharTensor = torch.CharTensor
        self.ShortTensor = torch.ShortTensor
        self.IntTensor = torch.IntTensor
        self.LongTensor = torch.LongTensor

    def globalkeys(self):
        """
        Yields global keys.
        """
        for key in self.__globals.keys():
            yield key

    def globalitems(self):
        """
        Yields global keys.
        """
        for key, value in self.__globals.items():
            yield key, value

    def __setitem__(self, key, value, override=False):
        """
        Adds global variable. 

        :param key: Dict Key.

        :param value: Associated value.

        :param override: Indicate whether or not it is authorized to override the existing key.        Default: ``False``.
        :type override: bool

        .. warning::
            Once global variable is set, its value cannot be changed (it becomes immutable).
        """
        if not override and key in self.__globals.keys():
            if self.__globals[key] != value:
                raise KeyError("Global key '{}' already exists and has different value (existing {} vs received {})!".format(key, self.__globals[key], value))
        else:
            self.__globals[key] = value

    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Dict Key.

        :return: Associated Value.
        """
        if key not in self.__globals.keys():
            msg = "Key '{}' not present in global variables".format(key)
            raise KeyError(msg)
        else:
            return self.__globals[key]


class GlobalsFacade(object):
    """
    Simple facility for accessing global variables using provided mappings using list-like read-write access.
    """

    def __init__(self, key_mappings):
        """
        Constructor. Initializes app state and stores key mappings.

        :param key_mappings: Dictionary of global key mappings of the parent object.
        """
        self.key_mappings = key_mappings
        self.app_state = AppState()

    def __setitem__(self, key, value):
        """
        Sets global value using parent object key mapping.

        :param key: Global key name (that will be mapped).

        :param value: Value that will be set.
        """
        mapped_key = self.key_mappings.get(key, key)
        self.app_state[mapped_key] = value

    def __getitem__(self, key):
        """
        Global value getter function.
        Uses parent object key mapping for accesing the value.

        :param key: Global key name (that will be mapped).

        :return: Associated Value.
        """
        mapped_key = self.key_mappings.get(key, key)
        return self.app_state[mapped_key]


class KeyMappingsFacade(object):
    """
    Simple facility for accessing key names using provided mappings using list-like (read-only) access.
    """

    def __init__(self, key_mappings):
        """
        Constructor. Stores key mappings.

        :param key_mappings: Dictionary of key mappings of the parent object.
        """
        self.keys_mappings = key_mappings

    def __getitem__(self, key):
        """
        Global value getter function.
        Uses parent object key mapping for accesing the value.

        :param key: Global key name (that will be mapped).

        :return: Associated Value.
        """
        return self.keys_mappings.get(key, key)


def load_class_default_config_file(class_type):
    """
    Function loads default configuration from the default config file associated with the given class type and adds it to parameter registry.

    :param class_type: Class type of a given object.

    :raturn: Loaded default configuration.
    """
    module = class_type.__module__.replace('.', '/')
    rel_path = module[module.find('ptp') + 4:]
    abs_default_config = os.path.join(AppState().absolute_config_path, 'default', rel_path) + '.yml'
    if not os.path.isfile(abs_default_config):
        None
        exit(-1)
    try:
        with open(abs_default_config, 'r') as stream:
            param_dict = yaml.safe_load(stream)
        if param_dict is None:
            None
            return {}
        else:
            return param_dict
    except yaml.YAMLError as e:
        None
        exit(-2)


class Component(abc.ABC):

    def __init__(self, name, class_type, config):
        """
        Initializes the component. This constructor:

            - sets the access to ``AppState`` (for dtypes, settings, globals etc.)
            - stores the component name and type
            - stores reference to the passed configuration registry section
            - loads default component parameters
            - initializes the logger
            - initializes mapping facilities and facades

        :param name: Name of the component.

        :param class_type: Class type of the component.

        :param config: Dictionary of parameters (read from configuration ``.yaml`` file).
        :type config: :py:class:`ptp.configuration.ConfigInterface`

        """
        self.name = name
        self.config = config
        self.app_state = AppState()
        self.logger = logging.initialize_logger(self.name)
        if class_type is not None:
            self.config.add_default_params(load_class_default_config_file(class_type))
        if 'streams' not in config or config['streams'] is None:
            self.__stream_keys = {}
        else:
            self.__stream_keys = config['streams']
        self.stream_keys = KeyMappingsFacade(self.__stream_keys)
        if 'globals' not in config or config['globals'] is None:
            self.__global_keys = {}
        else:
            self.__global_keys = config['globals']
        self.global_keys = KeyMappingsFacade(self.__global_keys)
        if 'statistics' not in config or config['statistics'] is None:
            self.__statistics_keys = {}
        else:
            self.__statistics_keys = config['statistics']
        self.statistics_keys = KeyMappingsFacade(self.__statistics_keys)
        self.globals = GlobalsFacade(self.__global_keys)

    def summarize_io(self, priority=-1):
        """
        Summarizes the component by showing its name, type and input/output definitions.

        :param priority: Component priority (DEFAULT: -1)

        :return: Summary as a str.

        """
        summary_str = '  + {} ({}) [{}] \n'.format(self.name, type(self).__name__, priority)
        summary_str += '      Inputs:\n'
        for key, value in self.input_data_definitions().items():
            summary_str += '        {}: {}, {}, {}\n'.format(key, value.dimensions, value.types, value.description)
        summary_str += '      Outputs:\n'
        for key, value in self.output_data_definitions().items():
            summary_str += '        {}: {}, {}, {}\n'.format(key, value.dimensions, value.types, value.description)
        return summary_str

    @abc.abstractmethod
    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.
        Abstract, must be implemented by all derived classes.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).
        """
        pass

    @abc.abstractmethod
    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.
        Abstract, must be implemented by all derived classes.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).
        """
        pass

    def handshake_input_definitions(self, all_definitions, log_errors=True):
        """ 
        Checks whether all_definitions contain fields required by the given component.

        :param all_definitions: dictionary containing output data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).

        :param log_errors: Logs the detected errors (DEFAULT: TRUE)

        :return: number of detected errors.
        """
        errors = 0
        for key, id in self.input_data_definitions().items():
            if key not in all_definitions.keys():
                if log_errors:
                    self.logger.error("Input definition: expected field '{}' not found in DataStreams keys ({})".format(key, all_definitions.keys()))
                errors += 1
                continue
            dd = all_definitions[key]
            if len(id.dimensions) != len(dd.dimensions):
                if log_errors:
                    self.logger.error("Input definition: field '{}' in DataStreams has different dimensions from expected (expected {} while received {})".format(key, id.dimensions, dd.dimensions))
                errors += 1
            else:
                for index, (did, ddd) in enumerate(zip(id.dimensions, dd.dimensions)):
                    if did != -1 and did != ddd:
                        if log_errors:
                            self.logger.error("Input definition: field '{}' in DataStreams has dimension {} different from expected (expected {} while received {})".format(key, index, id.dimensions, dd.dimensions))
                        errors += 1
            if len(id.types) != len(dd.types):
                if log_errors:
                    self.logger.error("Input definition: field '{}' in DataStreams has number of types different from expected (expected {} while received {})".format(key, id.types, dd.types))
                errors += 1
            else:
                for index, (tid, tdd) in enumerate(zip(id.types, dd.types)):
                    if tid != tdd:
                        if log_errors:
                            self.logger.error("Input definition: field '{}' in DataStreams has type {} different from expected (expected {} while received {})".format(key, index, id.types, dd.types))
                        errors += 1
        return errors

    def export_output_definitions(self, all_definitions, log_errors=True):
        """ 
        Exports output definitions to all_definitions, checking errors (e.g. if output field is already present in all_definitions).

        :param all_definitions: dictionary containing output data definitions (each of type :py:class:`ptp.configuration.DataDefinition`).

        :param log_errors: Logs the detected errors (DEFAULT: TRUE)

        :return: number of detected errors.
        """
        errors = 0
        for key, od in self.output_data_definitions().items():
            if key in all_definitions.keys():
                if log_errors:
                    self.logger.error("Output definition error: field '{}' cannot be added to DataStreams, as it is already present in its keys ({})".format(key, all_definitions.keys()))
                errors += 1
            else:
                all_definitions[key] = od
        return errors

    @abc.abstractmethod
    def __call__(self, data_streams):
        """
        Method responsible for processing the data dict.
        Abstract, must be implemented by all derived classes.

        :param data_streams: :py:class:`ptp.data_types.DataStreams` object containing both input data to be processed and that will be extended by the results.
        """
        pass

    def add_statistics(self, stat_col):
        """
        Adds statistics to :py:class:`ptp.configuration.StatisticsCollector`.

        .. note::

            Empty - To be redefined in inheriting classes.

        :param stat_col: :py:class:`ptp.configuration.StatisticsCollector`.

        """
        pass

    def collect_statistics(self, stat_col, data_streams):
        """
        Base statistics collection.

         .. note::

            Empty - To be redefined in inheriting classes. The user has to ensure that the corresponding entry             in the :py:class:`ptp.configuration.StatisticsCollector` has been created with             :py:func:`add_statistics` beforehand.

        :param stat_col: :py:class:`ptp.configuration.StatisticsCollector`.

        :param data_streams: ``DataStreams`` containing inputs, targets etc.
        :type data_streams: :py:class:`ptp.data_types.DataStreams`

        """
        pass

    def add_aggregators(self, stat_agg):
        """
        Adds statistical aggregators to :py:class:`ptp.configuration.StatisticsAggregator`.

        .. note::

            Empty - To be redefined in inheriting classes.

        :param stat_agg: :py:class:`ptp.configuration.StatisticsAggregator`.

        """
        pass

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates the statistics collected by :py:class:`ptp.configuration.StatisticsCollector` and adds the         results to :py:class:`ptp.configuration.StatisticsAggregator`.

         .. note::

            Empty - To be redefined in inheriting classes.
            The user can override this function in subclasses but should call             :py:func:`aggregate_statistics` to collect basic statistical aggregators (if set).

        :param stat_col: :py:class:`ptp.configuration.StatisticsCollector`.

        :param stat_agg: :py:class:`ptp.configuration.StatisticsAggregator`.

        """
        pass


class Model(Module, Component):
    """
    Class representing base class for all Models.

    Inherits from :py:class:`torch.nn.Module` as all subclasses will represent a trainable model.

    Hence, all subclasses should override the ``forward`` function.

    Implements features & attributes used by all subclasses.

    """

    def __init__(self, name, class_type, config):
        """
        Initializes a Model object.

        :param name: Model name.
        :type name: str

        :param class_type: Class type of the component.

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        This constructor:

        - calls base class constructors (save config, name, logger, app_state etc.)

        - initializes the best model loss (used to select which model to save) to ``np.inf``:

            >>> self.best_loss = np.inf

        """
        Component.__init__(self, name, class_type, config)
        Module.__init__(self)
        self.frozen = False

    def save_to_checkpoint(self, chkpt):
        """
        Adds model's state dictionary to checkpoint, which will be next stored to file.

        :param: Checkpoint (dictionary) that will be saved to file.
        """
        chkpt[self.name] = self.state_dict()

    def load_from_checkpoint(self, chkpt, section=None):
        """
        Loads state dictionary from checkpoint.

        :param chkpt: Checkpoint (dictionary) loaded from file.
        
        :param section: Name of the section containing params (DEFAULT: None, means that model name from current configuration will be used)        """
        if section is None:
            section = self.name
        self.load_state_dict(chkpt[section])

    def freeze(self):
        """
        Freezes the trainable weigths of the model.
        """
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def summarize(self):
        """
        Summarizes the model by showing the trainable/non-trainable parameters and weights         per layer ( ``nn.Module`` ).

        Uses ``recursive_summarize`` to iterate through the nested structure of the model (e.g. for RNNs).

        :return: Summary as a str.

        """
        summary_str = self.recursive_summarize(self, 0, self.name)
        num_total_params = sum([np.prod(p.size()) for p in self.parameters()])
        mod_trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        num_trainable_params = sum([np.prod(p.size()) for p in mod_trainable_params])
        summary_str += 'Total Trainable Params: {}\n'.format(num_trainable_params)
        summary_str += 'Total Non-trainable Params: {}\n'.format(num_total_params - num_trainable_params)
        summary_str += '=' * 80 + '\n'
        return summary_str

    def recursive_summarize(self, module_, indent_, module_name_):
        """
        Function that recursively inspects the (sub)modules and records their statistics          (like names, types, parameters, their numbers etc.)

        :param module_: Module to be inspected.
        :type module_: ``nn.Module`` or subclass

        :param indent_: Current indentation level.
        :type indent_: int

        :param module_name_: Name of the module that will be displayed before its type.
        :type module_name_: str

        :return: Str summarizing the module.
        """
        child_lines = []
        for key, module in module_._modules.items():
            child_lines.append(self.recursive_summarize(module, indent_ + 1, key))
        mod_str = ''
        if indent_ > 0:
            mod_str += ' ' + '| ' * (indent_ - 1) + '+ '
        mod_str += module_name_ + ' (' + module_._get_name() + ')'
        if indent_ == 0:
            if self.frozen:
                mod_str += '\t\t[FROZEN]'
            else:
                mod_str += '\t\t[TRAINABLE]'
        mod_str += '\n'
        mod_str += ''.join(child_lines)
        mod_direct_params = list(filter(lambda np: np[0].find('.') == -1, module_.named_parameters()))
        if len(mod_direct_params) != 0:
            mod_weights = [(n, tuple(p.size())) for n, p in mod_direct_params]
            mod_str += ' ' + '| ' * indent_ + '+ ' + 'Matrices: {}\n'.format(mod_weights)
            num_total_params = sum([np.prod(p.size()) for _, p in mod_direct_params])
            mod_trainable_params = filter(lambda np: np[1].requires_grad, mod_direct_params)
            num_trainable_params = sum([np.prod(p.size()) for _, p in mod_trainable_params])
            mod_str += ' ' + '| ' * indent_ + '  Trainable Params: {}\n'.format(num_trainable_params)
            mod_str += ' ' + '| ' * indent_ + '  Non-trainable Params: {}\n'.format(num_total_params - num_trainable_params)
        mod_str += ' ' + '| ' * indent_ + '\n'
        return mod_str


class CompactBilinearPooling(Model):
    """
    Element of one of classical baselines for Visual Question Answering.

    The model inputs (question and image encodings) are combined with Compact Bilinear Pooling mechanism.

    Fukui, A., Park, D. H., Yang, D., Rohrbach, A., Darrell, T., & Rohrbach, M. (2016). Multimodal compact bilinear pooling for visual question answering and visual grounding. arXiv preprint arXiv:1606.01847.

    Gao, Y., Beijbom, O., Zhang, N., & Darrell, T. (2016). Compact bilinear pooling. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 317-326).

    Inspired by implementation from:
    https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch/blob/master/CompactBilinearPooling.py
    """

    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(CompactBilinearPooling, self).__init__(name, CompactBilinearPooling, config)
        self.key_image_encodings = self.stream_keys['image_encodings']
        self.key_question_encodings = self.stream_keys['question_encodings']
        self.key_outputs = self.stream_keys['outputs']
        self.image_encoding_size = self.globals['image_encoding_size']
        self.question_encoding_size = self.globals['question_encoding_size']
        self.output_size = self.globals['output_size']
        image_sketch_projection_matrix = self.generate_count_sketch_projection_matrix(self.image_encoding_size, self.output_size)
        question_sketch_projection_matrix = self.generate_count_sketch_projection_matrix(self.question_encoding_size, self.output_size)
        trainable_projections = self.config['trainable_projections']
        self.image_sketch_projection_matrix = torch.nn.Parameter(image_sketch_projection_matrix, requires_grad=trainable_projections)
        self.question_sketch_projection_matrix = torch.nn.Parameter(question_sketch_projection_matrix, requires_grad=trainable_projections)

    def generate_count_sketch_projection_matrix(self, input_size, output_size):
        """ 
        Initializes Count Sketch projection matrix for given input (size).
        Its role will be to project vector v∈Rn to y∈Rd.
        We initialize two vectors s∈{−1,1}n and h∈{1,...,d}n:
            * s contains either 1 or −1 for each index
            * h maps each index i in the input v to an index j in the output y.
        Both s and h are initialized randomly from a uniform distribution and remain constant.
        """
        s = 2 * np.random.randint(2, size=input_size) - 1
        s = torch.from_numpy(s)
        h = np.random.randint(output_size, size=input_size)
        indices = np.concatenate((np.arange(input_size)[..., np.newaxis], h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        sparse_sketch_matrix = torch.sparse.FloatTensor(indices.t(), s, torch.Size([input_size, output_size]))
        dense_ssm = sparse_sketch_matrix.to_dense().type(self.app_state.FloatTensor)
        return dense_ssm

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_image_encodings: DataDefinition([-1, self.image_encoding_size], [torch.Tensor], 'Batch of encoded images [BATCH_SIZE x IMAGE_ENCODING_SIZE]'), self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], 'Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]')}

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        enc_img = data_streams[self.key_image_encodings]
        enc_q = data_streams[self.key_question_encodings]
        sketch_pm_img = self.image_sketch_projection_matrix
        sketch_pm_q = self.question_sketch_projection_matrix
        sketch_img = enc_img.mm(sketch_pm_img)
        sketch_q = enc_q.mm(sketch_pm_q)
        sketch_img_reim = torch.stack([sketch_img, torch.zeros(sketch_img.shape).type(self.app_state.FloatTensor)], dim=2)
        sketch_q_reim = torch.stack([sketch_q, torch.zeros(sketch_q.shape).type(self.app_state.FloatTensor)], dim=2)
        fft_img = torch.fft(sketch_img_reim, signal_ndim=1)
        fft_q = torch.fft(sketch_q_reim, signal_ndim=1)
        real1 = fft_img[:, :, (0)]
        imag1 = fft_img[:, :, (1)]
        real2 = fft_q[:, :, (0)]
        imag2 = fft_q[:, :, (1)]
        fft_product = torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)
        cbp = torch.ifft(fft_product, signal_ndim=1)[:, :, (0)]
        data_streams.publish({self.key_outputs: cbp})


class FactorizedBilinearPooling(Model):
    """
    Element of one of the classical baselines for Visual Question Answering.
    The multi-modal data is fused via sum-pooling of the element-wise multiplied high-dimensional representations and returned (for subsequent classification, done in a separate component e.g. ffn).

    On the basis of: Zhou Yu, Jun Yu. "Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering" (2015).
    Code: https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
    """

    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(FactorizedBilinearPooling, self).__init__(name, FactorizedBilinearPooling, config)
        self.key_image_encodings = self.stream_keys['image_encodings']
        self.key_question_encodings = self.stream_keys['question_encodings']
        self.key_outputs = self.stream_keys['outputs']
        self.image_encoding_size = self.globals['image_encoding_size']
        self.question_encoding_size = self.globals['question_encoding_size']
        self.latent_size = self.config['latent_size']
        self.factor = self.config['pool_factor']
        self.output_size = self.latent_size
        self.globals['output_size'] = self.output_size
        self.image_encodings_ff = torch.nn.Linear(self.image_encoding_size, self.latent_size * self.factor)
        self.question_encodings_ff = torch.nn.Linear(self.question_encoding_size, self.latent_size * self.factor)
        self.activation = torch.nn.ReLU()
        dropout_rate = self.config['dropout_rate']
        self.dropout = torch.nn.Dropout(dropout_rate)

    def input_data_definitions(self):
        """
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_image_encodings: DataDefinition([-1, self.image_encoding_size], [torch.Tensor], 'Batch of encoded images [BATCH_SIZE x IMAGE_ENCODING_SIZE]'), self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], 'Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]')}

    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        enc_img = data_streams[self.key_image_encodings]
        enc_q = data_streams[self.key_question_encodings]
        latent_img = self.dropout(self.image_encodings_ff(enc_img))
        latent_q = self.dropout(self.question_encodings_ff(enc_q))
        enc_z = latent_img * latent_q
        enc_z = self.dropout(enc_z)
        enc_z = enc_z.view(enc_z.size(0), self.latent_size, self.factor)
        enc_z = enc_z.sum(2)
        enc_z = torch.sqrt(self.activation(enc_z)) - torch.sqrt(self.activation(-enc_z))
        outputs = F.normalize(enc_z, p=2, dim=1)
        data_streams.publish({self.key_outputs: outputs})


class LowRankBilinearPooling(Model):
    """
    Element of one of classical baselines for Visual Question Answering.
    The model inputs (question and image encodings) are fused via element-wise multiplication and returned (for subsequent classification, done in a separate component e.g. ffn).

    On the basis of: Jiasen Lu and Xiao Lin and Dhruv Batra and Devi Parikh. "Deeper LSTM and normalized CNN visual question answering model" (2015).
    """

    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(LowRankBilinearPooling, self).__init__(name, LowRankBilinearPooling, config)
        self.key_image_encodings = self.stream_keys['image_encodings']
        self.key_question_encodings = self.stream_keys['question_encodings']
        self.key_outputs = self.stream_keys['outputs']
        self.image_encoding_size = self.globals['image_encoding_size']
        self.question_encoding_size = self.globals['question_encoding_size']
        self.output_size = self.globals['output_size']
        self.image_encodings_ff = torch.nn.Linear(self.image_encoding_size, self.output_size)
        self.question_encodings_ff = torch.nn.Linear(self.question_encoding_size, self.output_size)
        self.activation = torch.nn.ReLU()
        dropout_rate = self.config['dropout_rate']
        self.dropout = torch.nn.Dropout(dropout_rate)

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_image_encodings: DataDefinition([-1, self.image_encoding_size], [torch.Tensor], 'Batch of encoded images [BATCH_SIZE x IMAGE_ENCODING_SIZE]'), self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], 'Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]')}

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        enc_img = data_streams[self.key_image_encodings]
        enc_q = data_streams[self.key_question_encodings]
        enc_img = self.activation(enc_img)
        enc_img = self.dropout(enc_img)
        enc_q = self.activation(enc_q)
        enc_q = self.dropout(enc_q)
        latent_img = self.image_encodings_ff(enc_img)
        latent_q = self.question_encodings_ff(enc_q)
        outputs = latent_img * latent_q
        data_streams.publish({self.key_outputs: outputs})


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)
    input = input.view(n, 1, c, -1)
    attention = attention.view(n, glimpses, -1)
    attention = torch.nn.functional.softmax(attention, dim=-1).unsqueeze(2)
    weighted = attention * input
    weighted_mean = weighted.sum(dim=-1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


class QuestionDrivenAttention(Model):
    """
    Element of one of the classical baselines for Visual Question Answering.
    Attention-weighted image maps are computed based on the question.
    The multi-modal data (question and attention-weighted image maps) are fused via concatenation and returned (for subsequent classification, done in a separate component e.g. ffn).

    On the basis of: Vahid Kazemi Ali Elqursh. "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering" (2017).
    Code: https://github.com/Cyanogenoid/pytorch-vqa/blob/master/model.py
    """

    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(QuestionDrivenAttention, self).__init__(name, QuestionDrivenAttention, config)
        self.key_feature_maps = self.stream_keys['feature_maps']
        self.key_question_encodings = self.stream_keys['question_encodings']
        self.key_outputs = self.stream_keys['outputs']
        self.feature_maps_height = self.globals['feature_maps_height']
        self.feature_maps_width = self.globals['feature_maps_width']
        self.feature_maps_depth = self.globals['feature_maps_depth']
        self.question_encoding_size = self.globals['question_encoding_size']
        self.latent_size = self.config['latent_size']
        self.num_attention_heads = self.config['num_attention_heads']
        self.output_mode = self.config['output_mode']
        if self.output_mode == 'Image':
            self.output_size = self.feature_maps_depth * self.num_attention_heads
        elif self.output_mode == 'Fusion':
            self.output_size = self.feature_maps_depth * self.num_attention_heads + self.question_encoding_size
        else:
            None
        self.globals['output_size'] = self.output_size
        self.image_encodings_conv = torch.nn.Conv2d(self.feature_maps_depth, self.latent_size, 1, bias=False)
        self.question_encodings_ff = torch.nn.Linear(self.question_encoding_size, self.latent_size)
        self.attention_conv = torch.nn.Conv2d(self.latent_size, self.num_attention_heads, 1)
        self.activation = torch.nn.ReLU()
        dropout_rate = self.config['dropout_rate']
        self.dropout = torch.nn.Dropout(dropout_rate)

    def input_data_definitions(self):
        """
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_feature_maps: DataDefinition([-1, self.feature_maps_depth, self.feature_maps_height, self.feature_maps_width], [torch.Tensor], 'Batch of feature maps [BATCH_SIZE x FEAT_DEPTH x FEAT_HEIGHT x FEAT_WIDTH]'), self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], 'Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]')}

    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        enc_img = data_streams[self.key_feature_maps]
        enc_q = data_streams[self.key_question_encodings]
        enc_img = enc_img / (enc_img.norm(p=2, dim=1, keepdim=True).expand_as(enc_img) + 1e-08)
        latent_img = self.image_encodings_conv(self.dropout(enc_img))
        latent_q = self.question_encodings_ff(self.dropout(enc_q))
        latent_q_tile = tile_2d_over_nd(latent_q, latent_img)
        attention = self.activation(latent_img + latent_q_tile)
        attention = self.attention_conv(self.dropout(attention))
        attention_enc_img = apply_attention(enc_img, attention)
        if self.output_mode == 'Image':
            outputs = attention_enc_img
        elif self.output_mode == 'Fusion':
            outputs = torch.cat([attention_enc_img, enc_q], dim=1)
        data_streams.publish({self.key_outputs: outputs})


class ConfigurationError(Exception):
    """ Error thrown when encountered a configuration issue. """

    def __init__(self, msg):
        """ Stores message """
        self.msg = msg

    def __str__(self):
        """ Prints the message """
        return repr(self.msg)


class RelationalNetwork(Model):
    """
    Model implements relational network.
    Model expects image (CNN) features and encoded question.

    
    Santoro, A., Raposo, D., Barrett, D. G., Malinowski, M., Pascanu, R., Battaglia, P., & Lillicrap, T. (2017). A simple neural network module for relational reasoning. In Advances in neural information processing systems (pp. 4967-4976).
    Reference paper: https://arxiv.org/abs/1706.01427.
    """

    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(RelationalNetwork, self).__init__(name, RelationalNetwork, config)
        self.key_feature_maps = self.stream_keys['feature_maps']
        self.key_question_encodings = self.stream_keys['question_encodings']
        self.key_outputs = self.stream_keys['outputs']
        self.feature_maps_height = self.globals['feature_maps_height']
        self.feature_maps_width = self.globals['feature_maps_width']
        self.feature_maps_depth = self.globals['feature_maps_depth']
        self.question_encoding_size = self.globals['question_encoding_size']
        self.obj_coords = []
        for h in range(self.feature_maps_height):
            for w in range(self.feature_maps_width):
                self.obj_coords.append((h, w))
        input_size = 2 * self.feature_maps_depth + self.question_encoding_size
        modules = []
        dropout_rate = self.config['dropout_rate']
        g_theta_sizes = self.config['g_theta_sizes']
        if type(g_theta_sizes) == list and len(g_theta_sizes) > 1:
            input_dim = input_size
            for hidden_dim in g_theta_sizes:
                modules.append(torch.nn.Linear(input_dim, hidden_dim))
                modules.append(torch.nn.ReLU())
                if dropout_rate > 0:
                    modules.append(torch.nn.Dropout(dropout_rate))
                input_dim = hidden_dim
            modules.append(torch.nn.Linear(input_dim, hidden_dim))
            self.logger.info('Created g_theta network with {} layers'.format(len(g_theta_sizes) + 1))
        else:
            raise ConfigurationError("'g_theta_sizes' must contain a list with numbers of neurons in g_theta layers (currently {})".format(self.hidden_sizes))
        self.output_size = g_theta_sizes[-1]
        self.globals['output_size'] = self.output_size
        self.g_theta = torch.nn.Sequential(*modules)

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_feature_maps: DataDefinition([-1, self.feature_maps_depth, self.feature_maps_height, self.feature_maps_width], [torch.Tensor], 'Batch of feature maps [BATCH_SIZE x FEAT_DEPTH x FEAT_HEIGHT x FEAT_WIDTH]'), self.key_question_encodings: DataDefinition([-1, self.question_encoding_size], [torch.Tensor], 'Batch of encoded questions [BATCH_SIZE x QUESTION_ENCODING_SIZE]')}

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        feat_m = data_streams[self.key_feature_maps]
        enc_q = data_streams[self.key_question_encodings]
        relational_inputs = []
        for h1, w1 in self.obj_coords:
            for h2, w2 in self.obj_coords:
                fm1 = feat_m[:, :, (h1), (w1)].view(-1, self.feature_maps_depth)
                fm2 = feat_m[:, :, (h2), (w2)].view(-1, self.feature_maps_depth)
                concat = torch.cat([fm1, fm2, enc_q], dim=1)
                relational_inputs.append(concat)
        stacked_inputs = torch.stack(relational_inputs, dim=1)
        shape = stacked_inputs.shape
        stacked_inputs = stacked_inputs.contiguous().view(-1, shape[-1])
        stacked_relations = self.g_theta(stacked_inputs)
        stacked_relations = stacked_relations.view(*shape[0:-1], self.output_size)
        summed_relations = torch.sum(stacked_relations, dim=1)
        data_streams.publish({self.key_outputs: summed_relations})


class SelfAttention(Model):
    """
    Element of one of the classical baselines for Visual Question Answering.
    Attention within an image or text is computed.
    The attention weighted data (question or image) is returned (for subsequent classification, done in a separate component e.g. ffn).
    Currently only supports self-attention on text data

    On the basis of: Vaswani et. al Attention is all you need (2017)

    """

    def __init__(self, name, config):
        """
        Initializes the model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(SelfAttention, self).__init__(name, SelfAttention, config)
        self.key_question_encodings = self.stream_keys['question_encodings']
        self.key_outputs = self.stream_keys['outputs']
        self.question_encoding_size = self.globals['question_encoding_size']
        self.latent_size = self.config['latent_size']
        self.num_attention_heads = self.config['num_attention_heads']
        self.output_size = self.question_encoding_size * self.num_attention_heads
        self.activation = torch.nn.ReLU()
        self.W1 = torch.nn.Linear(self.question_encoding_size, self.latent_size)
        self.W2 = torch.nn.Linear(self.latent_size, self.num_attention_heads)

    def input_data_definitions(self):
        """
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_question_encodings: DataDefinition([-1, -1, self.question_encoding_size], [torch.Tensor], 'Batch of encoded questions [BATCH_SIZE x SEQ_LEN x QUESTION_ENCODING_SIZE]')}

    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs [BATCH_SIZE x OUTPUT_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'images',**})
        :type data_streams: ``ptp.dadatypes.DataStreams``
        """
        input_enc = data_streams[self.key_question_encodings]
        batch_size = input_enc.size()[0]
        self.Attention = torch.softmax(self.W2(self.activation(self.W1(input_enc))), dim=1)
        input_enc_weighted = torch.matmul(self.Attention.transpose(1, 2), input_enc)
        outputs = input_enc_weighted.view(batch_size, -1)
        data_streams.publish({self.key_outputs: outputs})


class ConvNetEncoder(Model):
    """
    A simple image encoder consisting of 3 consecutive convolutional layers.     The parameters of input image (width, height and depth) are not hardcoded so the encoder can be adjusted for given application.
    """

    def __init__(self, name, config):
        """
        Constructor of the ``SimpleConvNet``. 
        The overall structure of this CNN is as follows:

            (Conv1 -> MaxPool1 -> ReLu) -> (Conv2 -> MaxPool2 -> ReLu) -> (Conv3 -> MaxPool3 -> ReLu)

        The parameters that the user can change are:

         - For Conv1, Conv2 & Conv3: number of output channels, kernel size, stride and padding.
         - For MaxPool1, MaxPool2 & MaxPool3: Kernel size


        .. note::

            We are using the default values of ``dilatation``, ``groups`` & ``bias`` for ``nn.Conv2D``.

            Similarly for the ``stride``, ``padding``, ``dilatation``, ``return_indices`` & ``ceil_mode`` of             ``nn.MaxPool2D``.


        :param name: Name of the model (tken from the configuration file).

        :param config: dict of parameters (read from configuration ``.yaml`` file).
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(ConvNetEncoder, self).__init__(name, ConvNetEncoder, config)
        self.key_inputs = self.stream_keys['inputs']
        self.key_feature_maps = self.stream_keys['feature_maps']
        self.input_width = self.globals['input_width']
        self.input_height = self.globals['input_height']
        self.input_depth = self.globals['input_depth']
        self.out_channels_conv1 = config['conv1']['out_channels']
        self.kernel_size_conv1 = config['conv1']['kernel_size']
        self.stride_conv1 = config['conv1']['stride']
        self.padding_conv1 = config['conv1']['padding']
        self.kernel_size_maxpool1 = config['maxpool1']['kernel_size']
        self.out_channels_conv2 = config['conv2']['out_channels']
        self.kernel_size_conv2 = config['conv2']['kernel_size']
        self.stride_conv2 = config['conv2']['stride']
        self.padding_conv2 = config['conv2']['padding']
        self.kernel_size_maxpool2 = config['maxpool2']['kernel_size']
        self.out_channels_conv3 = config['conv3']['out_channels']
        self.kernel_size_conv3 = config['conv3']['kernel_size']
        self.stride_conv3 = config['conv3']['stride']
        self.padding_conv3 = config['conv3']['padding']
        self.kernel_size_maxpool3 = config['maxpool3']['kernel_size']
        self.conv1 = nn.Conv2d(in_channels=self.input_depth, out_channels=self.out_channels_conv1, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, dilation=1, groups=1, bias=True)
        self.width_features_conv1 = np.floor((self.input_width - self.kernel_size_conv1 + 2 * self.padding_conv1) / self.stride_conv1 + 1)
        self.height_features_conv1 = np.floor((self.input_height - self.kernel_size_conv1 + 2 * self.padding_conv1) / self.stride_conv1 + 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool1)
        self.width_features_maxpool1 = np.floor((self.width_features_conv1 - self.maxpool1.kernel_size + 2 * self.maxpool1.padding) / self.maxpool1.stride + 1)
        self.height_features_maxpool1 = np.floor((self.height_features_conv1 - self.maxpool1.kernel_size + 2 * self.maxpool1.padding) / self.maxpool1.stride + 1)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels_conv1, out_channels=self.out_channels_conv2, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, dilation=1, groups=1, bias=True)
        self.width_features_conv2 = np.floor((self.width_features_maxpool1 - self.kernel_size_conv2 + 2 * self.padding_conv2) / self.stride_conv2 + 1)
        self.height_features_conv2 = np.floor((self.height_features_maxpool1 - self.kernel_size_conv2 + 2 * self.padding_conv2) / self.stride_conv2 + 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool2)
        self.width_features_maxpool2 = np.floor((self.width_features_conv2 - self.maxpool2.kernel_size + 2 * self.maxpool2.padding) / self.maxpool2.stride + 1)
        self.height_features_maxpool2 = np.floor((self.height_features_conv2 - self.maxpool2.kernel_size + 2 * self.maxpool2.padding) / self.maxpool2.stride + 1)
        self.conv3 = nn.Conv2d(in_channels=self.out_channels_conv2, out_channels=self.out_channels_conv3, kernel_size=self.kernel_size_conv3, stride=self.stride_conv3, padding=self.padding_conv3, dilation=1, groups=1, bias=True)
        self.width_features_conv3 = np.floor((self.width_features_maxpool2 - self.kernel_size_conv3 + 2 * self.padding_conv3) / self.stride_conv3 + 1)
        self.height_features_conv3 = np.floor((self.height_features_maxpool2 - self.kernel_size_conv3 + 2 * self.padding_conv3) / self.stride_conv3 + 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=self.kernel_size_maxpool3)
        self.width_features_maxpool3 = np.floor((self.width_features_conv3 - self.maxpool3.kernel_size + 2 * self.maxpool3.padding) / self.maxpool3.stride + 1)
        self.height_features_maxpool3 = np.floor((self.height_features_conv3 - self.maxpool1.kernel_size + 2 * self.maxpool3.padding) / self.maxpool3.stride + 1)
        self.globals['feature_map_height'] = self.height_features_maxpool3
        self.globals['feature_map_width'] = self.width_features_maxpool3
        self.globals['feature_map_depth'] = self.out_channels_conv3
        self.logger.info('Input: [-1, {}, {}, {}]'.format(self.input_depth, self.input_width, self.input_height))
        self.logger.info('Computed output shape of each layer:')
        self.logger.info('Conv1: [-1, {}, {}, {}]'.format(self.out_channels_conv1, self.width_features_conv1, self.height_features_conv1))
        self.logger.info('MaxPool1: [-1, {}, {}, {}]'.format(self.out_channels_conv1, self.width_features_maxpool1, self.height_features_maxpool1))
        self.logger.info('Conv2: [-1, {}, {}, {}]'.format(self.out_channels_conv2, self.width_features_conv2, self.height_features_conv2))
        self.logger.info('MaxPool2: [-1, {}, {}, {}]'.format(self.out_channels_conv2, self.width_features_maxpool2, self.height_features_maxpool2))
        self.logger.info('Conv3: [-1, {}, {}, {}]'.format(self.out_channels_conv3, self.width_features_conv3, self.height_features_conv3))
        self.logger.info('MaxPool3: [-1, {}, {}, {}]'.format(self.out_channels_conv3, self.width_features_maxpool3, self.height_features_maxpool3))

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_inputs: DataDefinition([-1, self.input_depth, self.input_height, self.input_width], [torch.Tensor], 'Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]')}

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_feature_maps: DataDefinition([-1, self.out_channels_conv3, self.height_features_maxpool3, self.width_features_maxpool3], [torch.Tensor], 'Batch of filter maps [BATCH_SIZE x FEAT_DEPTH x FEAT_HEIGHT x FEAT_WIDTH]')}

    def forward(self, data_streams):
        """
        forward pass of the ``SimpleConvNet`` model.

        :param data_streams: DataStreams({'inputs','outputs'}), where:

            - inputs: [batch_size, in_depth, in_height, in_width],
            - feature_maps: batch of feature maps [batch_size, out_depth, out_height, out_width]

        """
        images = data_streams[self.key_inputs]
        out_conv1 = self.conv1(images)
        out_maxpool1 = torch.nn.functional.relu(self.maxpool1(out_conv1))
        out_conv2 = self.conv2(out_maxpool1)
        out_maxpool2 = torch.nn.functional.relu(self.maxpool2(out_conv2))
        out_conv3 = self.conv3(out_maxpool2)
        out_maxpool3 = torch.nn.functional.relu(self.maxpool3(out_conv3))
        data_streams.publish({self.key_feature_maps: out_maxpool3})


def get_value_from_dictionary(key, parameter_dict, accepted_values=[]):
    """
    Parses value of the parameter retrieved from a given parameter dictionary using key.
    Optionally, checks is the values is one of the accepted values.

    :param key: Key of the parameter.
    :param parameter_dict: Dictionary containing given key (e.g. config or globals)
    :param accepted_values: List of accepted values (DEFAULT: [])

    :return: List of parsed values
    """
    value = parameter_dict[key]
    assert type(value) == str, 'Parameter value must be a string'
    if value == '':
        return None
    if len(accepted_values) > 0:
        if value not in accepted_values:
            raise ConfigurationError("One of the values in '{}' is invalid (current: '{}', accepted: {})".format(key, value, accepted_values))
    return value


class GenericImageEncoder(Model):
    """
    Class
    """

    def __init__(self, name, config):
        """
        Initializes the ``LeNet5`` model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(GenericImageEncoder, self).__init__(name, GenericImageEncoder, config)
        self.key_inputs = self.stream_keys['inputs']
        self.key_outputs = self.stream_keys['outputs']
        self.return_feature_maps = self.config['return_feature_maps']
        pretrained = self.config['pretrained']
        self.model_type = get_value_from_dictionary('model_type', self.config, 'vgg16 | densenet121 | resnet152 | resnet50'.split(' | '))
        if self.model_type == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            if self.return_feature_maps:
                self.model = self.model.features
                self.feature_maps_height = 7
                self.globals['feature_maps_height'] = self.feature_maps_height
                self.feature_maps_width = 7
                self.globals['feature_maps_width'] = self.feature_maps_width
                self.feature_maps_depth = 512
                self.globals['feature_maps_depth'] = self.feature_maps_depth
            else:
                self.output_size = self.globals['output_size']
                self.model.classifier._modules['6'] = torch.nn.Linear(4096, self.output_size)
        elif self.model_type == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            if self.return_feature_maps:
                raise ConfigurationError("'densenet121' doesn't support 'return_feature_maps' mode (yet)")
            self.output_size = self.globals['output_size']
            self.model.classifier = torch.nn.Linear(1024, self.output_size)
        elif self.model_type == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
            if self.return_feature_maps:
                modules = list(self.model.children())[:-2]
                self.model = torch.nn.Sequential(*modules)
                self.feature_maps_height = 7
                self.globals['feature_maps_height'] = self.feature_maps_height
                self.feature_maps_width = 7
                self.globals['feature_maps_width'] = self.feature_maps_width
                self.feature_maps_depth = 2048
                self.globals['feature_maps_depth'] = self.feature_maps_depth
            else:
                self.output_size = self.globals['output_size']
                self.model.fc = torch.nn.Linear(2048, self.output_size)
        elif self.model_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            if self.return_feature_maps:
                modules = list(self.model.children())[:-2]
                self.model = torch.nn.Sequential(*modules)
                self.feature_maps_height = 7
                self.globals['feature_maps_height'] = self.feature_maps_height
                self.feature_maps_width = 7
                self.globals['feature_maps_width'] = self.feature_maps_width
                self.feature_maps_depth = 2048
                self.globals['feature_maps_depth'] = self.feature_maps_depth
            else:
                self.output_size = self.globals['output_size']
                self.model.fc = torch.nn.Linear(2048, self.output_size)

    def input_data_definitions(self):
        """
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_inputs: DataDefinition([-1, 3, 224, 224], [torch.Tensor], 'Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]')}

    def output_data_definitions(self):
        """
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        if self.return_feature_maps:
            return {self.key_outputs: DataDefinition([-1, self.feature_maps_depth, self.feature_maps_height, self.feature_maps_width], [torch.Tensor], 'Batch of feature maps [BATCH_SIZE x FEAT_DEPTH x FEAT_HEIGHT x FEAT_WIDTH]')}
        else:
            return {self.key_outputs: DataDefinition([-1, self.output_size], [torch.Tensor], 'Batch of outputs, each represented as probability distribution over classes [BATCH_SIZE x PREDICTION_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the model.

        :param data_streams: DataStreams({'inputs', ....}), where:

            - inputs: expected stream containing images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]
            - outpus: added stream containing outputs [BATCH_SIZE x PREDICTION_SIZE]

        :type data_streams: ``ptp.data_types.DataStreams``

        """
        img = data_streams[self.key_inputs]
        outputs = self.model(img)
        data_streams.publish({self.key_outputs: outputs})


class LeNet5(Model):
    """
    A classical LeNet-5 model for MNIST digits classification. 
    """

    def __init__(self, name, config):
        """
        Initializes the ``LeNet5`` model, creates the required layers.

        :param name: Name of the model (taken from the configuration file).

        :param config: Parameters read from configuration file.
        :type config: ``ptp.configuration.ConfigInterface``

        """
        super(LeNet5, self).__init__(name, LeNet5, config)
        self.key_inputs = self.stream_keys['inputs']
        self.key_predictions = self.stream_keys['predictions']
        self.prediction_size = self.globals['prediction_size']
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.linear1 = torch.nn.Linear(120, 84)
        self.linear2 = torch.nn.Linear(84, self.prediction_size)

    def input_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of input data that are required by the component.

        :return: dictionary containing input data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_inputs: DataDefinition([-1, 1, 32, 32], [torch.Tensor], 'Batch of images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]')}

    def output_data_definitions(self):
        """ 
        Function returns a dictionary with definitions of output data produced the component.

        :return: dictionary containing output data definitions (each of type :py:class:`ptp.utils.DataDefinition`).
        """
        return {self.key_predictions: DataDefinition([-1, self.prediction_size], [torch.Tensor], 'Batch of predictions, each represented as probability distribution over classes [BATCH_SIZE x PREDICTION_SIZE]')}

    def forward(self, data_streams):
        """
        Main forward pass of the ``LeNet5`` model.

        :param data_streams: DataStreams({'images',**}), where:

            - images: [batch_size, num_channels, width, height]

        :type data_streams: ``miprometheus.utils.DataStreams``

        :return: Predictions [batch_size, num_classes]

        """
        img = data_streams[self.key_inputs]
        x = self.conv1(img)
        x = torch.nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 120)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        predictions = torch.nn.functional.log_softmax(x, dim=1)
        data_streams.publish({self.key_predictions: predictions})


class DataStreams(collections.abc.MutableMapping):
    """
    - Mapping: A container object that supports arbitrary key lookups and implements the methods ``__getitem__``,     ``__iter__`` and ``__len__``.

    - Mutable objects can change their value but keep their id() -> ease modifying existing keys' value.

    DataStreams: Dict used for storing batches of data by tasks.

    **This is the main object class used to share data between all components through a worker, starting from task to loss and visualization.**
    """

    def __init__(self, *args, **kwargs):
        """
        DataStreams constructor. Can be initialized in different ways:

            >>> data_streams = DataStreams()
            >>> data_streams = DataStreams({'inputs': torch.tensor(), 'targets': numpy.ndarray()})
            >>> # etc.

        :param args: Used to pass a non-keyworded, variable-length argument list.

        :param kwargs: Used to pass a keyworded, variable-length argument list.
        """
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value, addkey=False):
        """
        key:value setter function.

        :param key: Dict Key.

        :param value: Associated value.

        :param addkey: Indicate whether or not it is authorized to add a new key `on-the-fly`.        Default: ``False``.
        :type addkey: bool

        .. warning::

            `addkey` is set to ``False`` by default as setting it to ``True`` removes the constraints of the            ``DataStreams`` and enables it to become mutable.
        """
        if not addkey and key not in self.keys():
            msg = 'Cannot modify a non-existing key "{}" in DataStreams'.format(key)
            raise KeyError(msg)
        else:
            self.__dict__[key] = value

    def publish(self, dict_to_add):
        """
        Publishes a new data streams - extends data stream object by adding (keys,values) from data_definitions.

        .. warning::
            This is in-place operation, i.e. extends existing object, does not return a new one.

        :param data_streams: :py:class:`ptp.utils.DataStreams` object to be extended.

        :param data_definitions: key-value pairs.

        """
        for key, value in dict_to_add.items():
            if key in self.keys():
                msg = 'Cannot extend DataStreams, as {} already present in its keys'.format(key)
                raise KeyError(msg)
            self.__setitem__(key, value, addkey=True)

    def reinitialize(self, streams_to_leave):
        """
        Removes all streams (keys and associated values) from DataStreams EXCEPT the ones passed in ``streams_to_leave``.
        """
        rem_keys = [key for key in self.keys() if key not in streams_to_leave.keys()]
        if 'index' in rem_keys:
            rem_keys.remove('index')
        for key in rem_keys:
            self.__delitem__(key, delkey=True)

    def __getitem__(self, key):
        """
        Value getter function.

        :param key: Dict Key.

        :return: Associated Value.

        """
        return self.__dict__[key]

    def __delitem__(self, key, delkey=False):
        """
        Delete a key:value pair.

        :param delkey: Indicate whether or not it is authorized to add a delete the key `on-the-fly`.        Default: ``False``.
        :type delkey: bool

        .. warning::

            By default, it is not authorized to delete an existing key. Set `delkey` to ``True`` to ignore this            restriction and 

        :param key: Dict Key.

        """
        if not delkey:
            msg = 'Cannot delete key "{}" from DataStreams'.format(key)
            raise KeyError(msg)
        else:
            del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        """
        :return: A simple Dict representation of ``DataStreams``.

        """
        return str(self.__dict__)

    def __repr__(self):
        """
        :return: Echoes class, id, & reproducible representation in the Read–Eval–Print Loop.

        """
        return '{}, DataStreams({})'.format(super(DataStreams, self).__repr__(), self.__dict__)

    def to(self, device=None, keys_to_move=None, non_blocking=False):
        """
        Moves object(s) to device

        .. note::

            Wraps call to ``torch.Tensor.to()``: If this object is already in CUDA memory and on the correct device,             then no copy is performed and the original object is returned.
            If an element of `self` is not a ``torch.tensor``, it is returned as is,             i.e. We only move the ``torch.tensor`` (s) contained in `self`. 

        :param device: The destination GPU device. Defaults to the current CUDA device.
        :type device: torch.device

        :param non_blocking: If True and the source is in pinned memory, the copy will be asynchronous with respect to         the host. Otherwise, the argument has no effect. Default: ``False``.
        :type non_blocking: bool

        """
        for key in self:
            if isinstance(self[key], torch.Tensor):
                if keys_to_move is not None and key not in keys_to_move:
                    continue
                self[key] = self[key]


def data_streams_gather(outputs, target_device, dim=0):
    """
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, DataStreams):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        return type(out)(map(gather_map, zip(*outputs)))
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


def data_streams_scatter(inputs, target_gpus, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        if isinstance(obj, DataStreams) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for _ in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def data_streams_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = data_streams_scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = data_streams_scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class DataStreamsParallel(torch.nn.DataParallel):
    """
    Modified DataParallel wrapper enabling operation on DataStreamss.
    
    .. warning:
        Compatible with PyTorch v1.0.1 !!

    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataStreamsParallel, self).__init__(module, device_ids, output_device, dim)

    def forward(self, *inputs, **kwargs):
        """
        Performs "parallelized forward" pass by scattering batch into several batches, distributing models on different GPUs, performing parallel pass and gathering results into a single (returned) DataStreams.

        ..warning:
            As the "external" operations are changing inputs to tuple of DataStreamss, extension of main DataStreams must be done "outside" of this method.
        """
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        inputs_tuple = []
        for i, item in enumerate(inputs):
            input_dict = DataStreams({key: value for key, value in item.items() if key in self.module.input_data_definitions().keys()})
            inputs_tuple.append(input_dict)
        inputs_tuple = tuple(inputs_tuple)
        inputs_tuple, kwargs = self.scatter(inputs_tuple, kwargs, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs_tuple)])
        self.parallel_apply(replicas, inputs_tuple, kwargs)
        gathered_tuple = self.gather(inputs_tuple, self.output_device)
        return gathered_tuple[0]

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return data_streams_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return data_streams_gather(outputs, output_device, dim=self.dim)

    def add_statistics(self, stat_col):
        """
        Adds statistics for the wrapped model.

        :param stat_col: ``StatisticsCollector``.
        """
        self.module.add_statistics(stat_col)

    def collect_statistics(self, stat_col, data_streams):
        """
        Collects statistics for the wrapped model.

        :param stat_col: :py:class:`ptp.utils.StatisticsCollector`.

        :param data_streams: ``DataStreams`` containing inputs, targets etc.
        :type data_streams: :py:class:`ptp.data_types.DataStreams`
        """
        self.module.collect_statistics(stat_col, data_streams)

    def add_aggregators(self, stat_agg):
        """
        Aggregates statistics for the wrapped model.

        :param stat_agg: ``StatisticsAggregator``.
        """
        self.module.add_aggregators(stat_agg)

    def aggregate_statistics(self, stat_col, stat_agg):
        """
        Aggregates statistics for the wrapped model.

        :param stat_col: ``StatisticsCollector``

        :param stat_agg: ``StatisticsAggregator``
        """
        self.module.aggregate_statistics(stat_col, stat_agg)


class TestModel1(Model):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, datadict):
        input = datadict['index']
        None
        output = self.fc(input)
        None
        datadict.extend({'middle': output})

    def input_data_definitions(self):
        return {'index': DataDefinition(1, 1, 'str')}

    def output_data_definitions(self):
        return {'middle': DataDefinition(1, 1, 'str')}


class TestModel2(Model):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, datadict):
        input = datadict['middle']
        None
        output = self.fc(input)
        None
        datadict.extend({'output': output})

    def input_data_definitions(self):
        return {'middle': DataDefinition(1, 1, 'str')}

    def output_data_definitions(self):
        return {'output': DataDefinition(1, 1, 'str')}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DataStreamsParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_IBM_pytorchpipe(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

