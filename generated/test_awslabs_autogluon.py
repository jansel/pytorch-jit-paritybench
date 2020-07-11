import sys
_module = sys.modules[__name__]
del sys
autogluon = _module
contrib = _module
enas = _module
enas_net = _module
enas_scheduler = _module
enas_utils = _module
core = _module
decorator = _module
loss = _module
optimizer = _module
space = _module
task = _module
model_zoo = _module
model_store = _module
models = _module
efficientnet = _module
mbconv = _module
standford_dog_models = _module
utils = _module
scheduler = _module
fifo = _module
hyperband = _module
hyperband_promotion = _module
hyperband_stopping = _module
remote = _module
cli = _module
dask_scheduler = _module
remote_manager = _module
ssh_helper = _module
reporter = _module
resource = _module
dist_manager = _module
manager = _module
nvutil = _module
rl_scheduler = _module
searcher = _module
grid_searcher = _module
rl_controller = _module
searcher_factory = _module
skopt_searcher = _module
base = _module
base_predictor = _module
base_task = _module
image_classification = _module
classifier = _module
dataset = _module
losses = _module
metrics = _module
nets = _module
pipeline = _module
processing_params = _module
object_detection = _module
data_parallel = _module
coco = _module
voc = _module
detector = _module
tabular_prediction = _module
hyperparameter_configs = _module
predictor = _module
presets_configs = _module
text_classification = _module
network = _module
transforms = _module
augment = _module
custom_process = _module
custom_queue = _module
dataloader = _module
defaultdict = _module
deprecate = _module
edict = _module
file_helper = _module
files = _module
learning_rate = _module
miscs = _module
mxutils = _module
pil_transforms = _module
plot_network = _module
plots = _module
serialization = _module
sync_remote = _module
tabular = _module
data = _module
cleaner = _module
label_cleaner = _module
features = _module
abstract_feature_generator = _module
add_datepart_helper = _module
auto_ml_feature_generator = _module
vectorizers = _module
classification_metrics = _module
softclass_metrics = _module
util = _module
ml = _module
constants = _module
learner = _module
abstract_learner = _module
default_learner = _module
abstract = _module
abstract_model = _module
model_trial = _module
catboost = _module
catboost_model = _module
catboost_utils = _module
hyperparameters = _module
parameters = _module
searchspaces = _module
ensemble = _module
bagged_ensemble_model = _module
greedy_weighted_ensemble_model = _module
stacker_ensemble_model = _module
weighted_ensemble_model = _module
knn = _module
knn_model = _module
lgb = _module
callbacks = _module
lgb_trial = _module
lgb_model = _module
lgb_utils = _module
lr = _module
lr_model = _module
lr_preprocessing_utils = _module
rf = _module
rf_model = _module
tabular_nn = _module
categorical_encoders = _module
embednet = _module
tabular_nn_dataset = _module
tabular_nn_model = _module
tabular_nn_trial = _module
trainer = _module
abstract_trainer = _module
auto_trainer = _module
model_presets = _module
presets = _module
presets_rf = _module
tuning = _module
ensemble_selection = _module
feature_pruner = _module
decorators = _module
exceptions = _module
loaders = _module
load_pd = _module
load_pkl = _module
load_pointer = _module
load_s3 = _module
multiprocessing_utils = _module
s3_utils = _module
savers = _module
save_json = _module
save_pd = _module
save_pkl = _module
save_pointer = _module
tqdm = _module
try_import = _module
util_decorator = _module
test = _module
cifar_autogluon = _module
prepare_imagenet = _module
benchmark = _module
blog = _module
data_processing = _module
kaggle_configuration = _module
search_efficientnet = _module
train_enas_imagenet = _module
demo = _module
example_advanced_tabular = _module
example_simple_tabular = _module
example_custom_dataset = _module
example_glue_dataset = _module
train_enas_imagenet = _module
setup = _module
conftest = _module
test_check_style = _module
test_classification_tricks = _module
test_image_classification = _module
test_model_zoo = _module
test_scheduler = _module
test_search_space = _module
test_skoptsearcher = _module
test_tabular = _module
test_text_classification = _module

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


import logging


import pandas as pd


from pandas import DataFrame


import torch


from torch import nn


class Space(object):
    """Basic search space describing set of possible values for hyperparameter.
    """
    pass


class NestedSpace(Space):
    """Nested hyperparameter search space, which is a search space that itself contains multiple search spaces.
    """

    def sample(self, **config):
        """Sample a configuration from this search space.
        """
        pass

    @property
    def cs(self):
        """ ConfigSpace representation of this search space.
        """
        raise NotImplementedError

    @property
    def kwspaces(self):
        """ OrderedDict representation of this search space.
        """
        raise NotImplementedError

    @property
    def default(self):
        """Return default value for hyperparameter corresponding to this search space.
        """
        config = self.cs.get_default_configuration().get_dictionary()
        return self.sample(**config)

    @property
    def rand(self):
        """Randomly sample configuration from this nested search space.
        """
        config = self.cs.sample_configuration().get_dictionary()
        return self.sample(**config)


def _add_hp(cs, hp):
    if hp.name in cs._hyperparameters:
        cs._hyperparameters[hp.name] = hp
    else:
        cs.add_hyperparameter(hp)


def _add_cs(master_cs, sub_cs, prefix, delimiter='.', parent_hp=None):
    new_parameters = []
    for hp in sub_cs.get_hyperparameters():
        new_parameter = copy.deepcopy(hp)
        if new_parameter.name == '':
            new_parameter.name = prefix
        elif not prefix == '':
            new_parameter.name = '%s%s%s' % (prefix, '.', new_parameter.name)
        new_parameters.append(new_parameter)
    for hp in new_parameters:
        _add_hp(master_cs, hp)


def _strip_config_space(config, prefix):
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix) + 1:]] = v
    return new_config


class Categorical(NestedSpace):
    """Nested search space for hyperparameters which are categorical. Such a hyperparameter takes one value out of the discrete set of provided options.

    Parameters
    ----------
    data : Space or python built-in objects
        the choice candidates

    Examples
    --------
    a = ag.space.Categorical('a', 'b', 'c', 'd')
    b = ag.space.Categorical('resnet50', autogluon_obj())
    """

    def __init__(self, *data):
        self.data = [*data]

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    @property
    def cs(self):
        """ ConfigSpace representation of this search space.
        """
        cs = CS.ConfigurationSpace()
        if len(self.data) == 0:
            return CS.ConfigurationSpace()
        hp = CSH.CategoricalHyperparameter(name='choice', choices=range(len(self.data)))
        _add_hp(cs, hp)
        for i, v in enumerate(self.data):
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, str(i))
        return cs

    def sample(self, **config):
        """Sample a configuration from this search space.
        """
        choice = config.pop('choice')
        if isinstance(self.data[choice], NestedSpace):
            min_config = _strip_config_space(config, prefix=str(choice))
            return self.data[choice].sample(**min_config)
        else:
            return self.data[choice]

    @property
    def kwspaces(self):
        """OrderedDict representation of this search space.
        """
        kw_spaces = OrderedDict()
        for idx, obj in enumerate(self.data):
            if isinstance(obj, NestedSpace):
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(idx, sub_k)
                    kw_spaces[new_k] = sub_v
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr


_blocks = []


def enas_net(**kwvars):

    def registered_class(Cls):


        class ENAS_Net(Cls):

            def __init__(self, *args, **kwargs):
                kwvars.update(kwargs)
                super().__init__(*args, **kwvars)
                self._modules = {}
                self._kwspaces = collections.OrderedDict()
                for k, module in kwvars.items():
                    if isinstance(module, (ENAS_Unit, ENAS_Sequential)):
                        self._modules[k] = module
                        if isinstance(module, ENAS_Unit):
                            self._kwspaces[k] = module.kwspaces
                        else:
                            assert isinstance(module, ENAS_Sequential)
                            for key, v in module.kwspaces.items():
                                new_key = '{}.{}'.format(k, key)
                                self._kwspaces[new_key] = v
                self.latency_evaluated = False
                self._avg_latency = 1

            @property
            def nparams(self):
                nparams = 0
                for k, op in self._modules.items():
                    if isinstance(op, (ENAS_Unit, ENAS_Sequential)):
                        nparams += op.nparams
                    else:
                        for _, v in op.collect_params().items():
                            nparams += v.data().size
                return nparams

            @property
            def nodeend(self):
                return list(self._modules.keys())[-1]

            @property
            def nodehead(self):
                return list(self._modules.keys())[0]

            @property
            def graph(self):
                e = Digraph(node_attr={'color': 'lightblue2', 'style': 'filled', 'shape': 'box'})
                pre_node = 'input'
                e.node(pre_node)
                for k, op in self._modules.items():
                    if hasattr(op, 'graph'):
                        e.subgraph(op.graph)
                        e.edge(pre_node, op.nodehead)
                        pre_node = op.nodeend
                    else:
                        if hasattr(op, 'node'):
                            if op.node is None:
                                continue
                            node_info = op.node
                        else:
                            node_info = {'label': op.__class__.__name__}
                        e.node(k, **node_info)
                        e.edge(pre_node, k)
                        pre_node = k
                return e

            @property
            def kwspaces(self):
                return self._kwspaces

            def sample(self, **configs):
                striped_keys = [k.split('.')[0] for k in configs.keys()]
                for k in striped_keys:
                    if isinstance(self._modules[k], ENAS_Unit):
                        self._modules[k].sample(configs[k])
                    else:
                        sub_configs = _strip_config_space(configs, prefix=k)
                        self._modules[k].sample(**sub_configs)

            @property
            def latency(self):
                if not self.latency_evaluated:
                    raise Exception('Latency is not evaluated yet.')
                return self._avg_latency

            @property
            def avg_latency(self):
                if not self.latency_evaluated:
                    raise Exception('Latency is not evaluated yet.')
                return self._avg_latency

            def evaluate_latency(self, x):
                import time
                for k, op in self._modules.items():
                    if hasattr(op, 'evaluate_latency'):
                        x = op.evaluate_latency(x)
                avg_latency = 0.0
                for k, op in self._modules.items():
                    if hasattr(op, 'avg_latency'):
                        avg_latency += op.avg_latency
                self._avg_latency = avg_latency
                self.latency_evaluated = True
        return ENAS_Net
    return registered_class


input_size = 112

