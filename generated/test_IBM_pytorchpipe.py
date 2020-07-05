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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
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


import torchvision.models as models


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import time


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
        print("ERROR: The default configuration file '{}' for '{}' does not exist".format(abs_default_config, class_type.__module__))
        exit(-1)
    try:
        with open(abs_default_config, 'r') as stream:
            param_dict = yaml.safe_load(stream)
        if param_dict is None:
            print("WARNING: The default configuration file '{}' is empty!".format(abs_default_config))
            return {}
        else:
            return param_dict
    except yaml.YAMLError as e:
        print("ERROR: Couldn't properly parse the '{}' default configuration file. YAML error:\n  {}".format(abs_default_config, e))
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
                self[key] = self[key].to(device=device)


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

