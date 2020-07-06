import sys
_module = sys.modules[__name__]
del sys
environments = _module
custom = _module
openai = _module
envs = _module
bit_flip = _module
random_bitflip = _module
test_openai_gym_atari = _module
docs = _module
conf = _module
actor_critic_cartpole = _module
apex_pong = _module
distributed_ppo_pendulum = _module
dqn_cartpole_with_tf_summaries = _module
impala_cartpole = _module
impala_distributed_dmlab = _module
impala_openai_gym_with_lstm = _module
ppo_cartpole = _module
ppo_or_sac_on_mlagents = _module
sac_pendulum = _module
train_agent_openai = _module
rlgraph = _module
agents = _module
actor_critic_agent = _module
agent = _module
apex_agent = _module
dqfd_agent = _module
dqn_agent = _module
impala_agents = _module
ppo_agent = _module
random_agent = _module
sac_agent = _module
components = _module
action_adapters = _module
action_adapter = _module
action_adapter_utils = _module
bernoulli_distribution_adapter = _module
beta_distribution_adapter = _module
categorical_distribution_adapter = _module
gumbel_softmax_distribution_adapter = _module
normal_distribution_adapter = _module
squashed_normal_distribution_adapter = _module
common = _module
batch_apply = _module
batch_splitter = _module
container_merger = _module
decay_components = _module
environment_stepper = _module
iterative_optimization = _module
multi_gpu_synchronizer = _module
noise_components = _module
repeater_stack = _module
sampler = _module
slice = _module
softmax = _module
staging_area = _module
synchronizable = _module
time_dependent_parameters = _module
component = _module
distributions = _module
bernoulli = _module
beta = _module
categorical = _module
distribution = _module
gumbel_softmax = _module
joint_cumulative_distribution = _module
mixture_distribution = _module
multivariate_normal = _module
normal = _module
squashed_normal = _module
explorations = _module
epsilon_exploration = _module
exploration = _module
helpers = _module
clipping = _module
dynamic_batching = _module
generalized_advantage_estimation = _module
mem_segment_tree = _module
segment_tree = _module
sequence_helper = _module
v_trace_function = _module
layers = _module
layer = _module
nn = _module
activation_functions = _module
concat_layer = _module
conv2d_layer = _module
conv2d_transpose_layer = _module
dense_layer = _module
local_response_normalization_layer = _module
lstm_layer = _module
maxpool2d_layer = _module
multi_lstm_layer = _module
nn_layer = _module
residual_layer = _module
preprocessing = _module
clip = _module
concat = _module
container_splitter = _module
convert_type = _module
grayscale = _module
image_binary = _module
image_crop = _module
image_resize = _module
moving_standardize = _module
multiply_divide = _module
normalize = _module
preprocess_layer = _module
rank_reinterpreter = _module
reshape = _module
sequence = _module
transpose = _module
strings = _module
embedding_lookup = _module
string_layer = _module
string_to_hash_bucket = _module
loss_functions = _module
actor_critic_loss_function = _module
categorical_cross_entropy_loss = _module
container_loss_function = _module
dqfd_loss_function = _module
dqn_loss_function = _module
euclidian_distance_loss = _module
impala_loss_function = _module
loss_function = _module
neg_log_likelihood_loss = _module
ppo_loss_function = _module
sac_loss_function = _module
supervised_loss_function = _module
memories = _module
fifo_queue = _module
mem_prioritized_replay = _module
memory = _module
prioritized_replay = _module
queue_runner = _module
replay_memory = _module
ring_buffer = _module
models = _module
intrinsic_curiosity_world_option_model = _module
model = _module
supervised_model = _module
neural_networks = _module
actor_component = _module
dict_preprocessor_stack = _module
impala = _module
impala_networks = _module
multi_input_stream_neural_network = _module
neural_network = _module
preprocessor_stack = _module
sac = _module
sac_networks = _module
stack = _module
value_function = _module
variational_auto_encoder = _module
optimizers = _module
horovod_optimizer = _module
local_optimizers = _module
optimizer = _module
policies = _module
dueling_policy = _module
dynamic_batching_policy = _module
policy = _module
shared_value_function_policy = _module
queues = _module
deepmind_lab = _module
deterministic_env = _module
environment = _module
gaussian_density_as_reward_env = _module
grid_world = _module
mlagents_env = _module
openai_gym = _module
random_env = _module
sequential_vector_env = _module
vector_env = _module
vizdoom = _module
execution = _module
distributed_tf = _module
impala_worker = _module
environment_sample = _module
ray = _module
apex = _module
apex_executor = _module
apex_memory = _module
ray_memory_actor = _module
ray_actor = _module
ray_executor = _module
ray_policy_worker = _module
ray_util = _module
ray_value_worker = _module
sync_batch_executor = _module
single_threaded_worker = _module
worker = _module
graphs = _module
graph_builder = _module
graph_executor = _module
meta_graph = _module
meta_graph_builder = _module
pytorch_executor = _module
tensorflow_executor = _module
spaces = _module
bool_box = _module
box_space = _module
containers = _module
float_box = _module
int_box = _module
space = _module
space_utils = _module
text_box = _module
tests = _module
agent_functionality = _module
test_all_compile = _module
test_apex_agent_functionality = _module
test_base_agent_functionality = _module
test_dqfd_agent_functionality = _module
test_dqn_agent_functionality = _module
test_impala_agent_functionality = _module
test_ppo_agent_functionality = _module
test_sac_agent_functionality = _module
agent_learning = _module
long_tasks = _module
test_apex_agent_long_task_learning = _module
test_dqn_agent_long_task_learning = _module
test_impala_agent_long_task_learning = _module
short_tasks = _module
test_actor_critic_agent_short_task_learning = _module
test_dqn_agent_short_task_learning = _module
test_impala_agent_short_task_learning = _module
test_ppo_agent_short_task_learning = _module
test_sac_agent_short_task_learning = _module
agent_test = _module
component_test = _module
test_action_adapters = _module
test_actor_components = _module
test_batch_apply = _module
test_batch_splitter = _module
test_component_copy = _module
test_container_merger = _module
test_container_splitter = _module
test_decay_components = _module
test_dict_preprocessor_stack = _module
test_distributions = _module
test_dqn_loss_functions = _module
test_environment_stepper = _module
test_epsilon_exploration = _module
test_explorations = _module
test_fifo_queue = _module
test_generalized_advantage_estimation = _module
test_impala_loss_function = _module
test_local_optimizers = _module
test_multi_input_stream_nn = _module
test_neural_networks = _module
test_neural_networks_keras_style_assembly = _module
test_nn_layers = _module
test_noise_components = _module
test_policies = _module
test_policies_on_container_actions = _module
test_ppo_loss_functions = _module
test_preprocess_layers = _module
test_preprocessor_stacks = _module
test_prioritized_replay = _module
test_python_prioritized_replay = _module
test_replay_memory = _module
test_reshape_preprocessor = _module
test_ring_buffer = _module
test_sac_loss_function = _module
test_sampler_component = _module
test_sequence_helper = _module
test_sequence_preprocessor = _module
test_slice = _module
test_softmax = _module
test_stack = _module
test_staging_area = _module
test_string_layers = _module
test_supervised_loss_functions = _module
test_synchronizable = _module
test_time_dependent_parameters = _module
test_v_trace_function = _module
test_variational_auto_encoders = _module
core = _module
test_api_methods = _module
test_device_placements = _module
test_graph_fns = _module
test_input_incomplete_build = _module
test_input_space_checking = _module
test_pytorch_backend = _module
test_pytorch_util = _module
test_single_components = _module
test_spaces = _module
test_specifiable_server = _module
test_specifiables = _module
dummy_components = _module
dummy_components_with_sub_components = _module
test_deepmind_lab = _module
test_deterministic_env = _module
test_grid_world = _module
test_ml_agents_env = _module
test_random_env = _module
test_readme_example = _module
test_sequential_vector_env = _module
test_apex_executor = _module
test_gpu_strategies = _module
test_ray_policy_worker = _module
test_ray_value_worker = _module
test_single_threaded_worker = _module
test_sync_batch_executor = _module
performance = _module
test_backends = _module
test_multi_gpu_updates = _module
test_python_memory_performance = _module
test_single_threaded_dqn = _module
test_tf_memory_performance = _module
test_time_rank_folding_performance = _module
test_vector_env = _module
test_util = _module
visualization = _module
test_visualizations = _module
utils = _module
config_util = _module
debug_util = _module
decorators = _module
define_by_run_ops = _module
initializer = _module
input_parsing = _module
model_util = _module
numpy = _module
op_records = _module
ops = _module
pytorch_util = _module
rlgraph_errors = _module
specifiable = _module
specifiable_server = _module
tf_util = _module
util = _module
visualization_util = _module
version = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


from math import log


import copy


import inspect


import re


import uuid


from collections import OrderedDict


from functools import partial


from collections import deque


import logging


import time


import random


import math

