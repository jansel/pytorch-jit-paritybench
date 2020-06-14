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


import torch.nn as nn


import numpy as np


from torch.nn import Module


import torch.nn.functional as F


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*
                args, **kwargs)
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
            self.logger.warning(
                'GPU utilization is demanded but there are no available GPU devices! Using CPUs instead'
                )
        else:
            self.logger.info(
                'GPU utilization is disabled, performing all computations on CPUs'
                )

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
        self.FloatTensor = torch.cuda.FloatTensor
        self.DoubleTensor = torch.cuda.DoubleTensor
        self.HalfTensor = torch.cuda.HalfTensor
        self.ByteTensor = torch.cuda.ByteTensor
        self.CharTensor = torch.cuda.CharTensor
        self.ShortTensor = torch.cuda.ShortTensor
        self.IntTensor = torch.cuda.IntTensor
        self.LongTensor = torch.cuda.LongTensor

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
                raise KeyError(
                    "Global key '{}' already exists and has different value (existing {} vs received {})!"
                    .format(key, self.__globals[key], value))
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
    abs_default_config = os.path.join(AppState().absolute_config_path,
        'default', rel_path) + '.yml'
    if not os.path.isfile(abs_default_config):
        print(
            "ERROR: The default configuration file '{}' for '{}' does not exist"
            .format(abs_default_config, class_type.__module__))
        exit(-1)
    try:
        with open(abs_default_config, 'r') as stream:
            param_dict = yaml.safe_load(stream)
        if param_dict is None:
            print("WARNING: The default configuration file '{}' is empty!".
                format(abs_default_config))
            return {}
        else:
            return param_dict
    except yaml.YAMLError as e:
        print(
            """ERROR: Couldn't properly parse the '{}' default configuration file. YAML error:
  {}"""
            .format(abs_default_config, e))
        exit(-2)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_IBM_pytorchpipe(_paritybench_base):
    pass
