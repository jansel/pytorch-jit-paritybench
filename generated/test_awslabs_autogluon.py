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


from torch import nn


def _strip_config_space(config, prefix):
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix) + 1:]] = v
    return new_config


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_awslabs_autogluon(_paritybench_base):
    pass
