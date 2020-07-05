import sys
_module = sys.modules[__name__]
del sys
conf = _module
pyprob = _module
address_dictionary = _module
concurrency = _module
diagnostics = _module
distributions = _module
bernoulli = _module
beta = _module
binomial = _module
categorical = _module
distribution = _module
empirical = _module
exponential = _module
gamma = _module
log_normal = _module
mixture = _module
normal = _module
poisson = _module
truncated_normal = _module
uniform = _module
weibull = _module
graph = _module
model = _module
nn = _module
dataset = _module
embedding_cnn_2d_5c = _module
embedding_cnn_3d_5c = _module
embedding_feedforward = _module
inference_network = _module
inference_network_feedforward = _module
inference_network_lstm = _module
optimizer_larc = _module
proposal_categorical_categorical = _module
proposal_normal_normal = _module
proposal_normal_normal_mixture = _module
proposal_poisson_truncated_normal_mixture = _module
proposal_uniform_beta = _module
proposal_uniform_beta_mixture = _module
proposal_uniform_truncated_normal_mixture = _module
Bernoulli = _module
Beta = _module
Binomial = _module
Categorical = _module
Distribution = _module
Exponential = _module
Gamma = _module
Handshake = _module
HandshakeResult = _module
LogNormal = _module
Message = _module
MessageBody = _module
Normal = _module
Observe = _module
ObserveResult = _module
Poisson = _module
Reset = _module
Run = _module
RunResult = _module
Sample = _module
SampleResult = _module
Tag = _module
TagResult = _module
Tensor = _module
Uniform = _module
Weibull = _module
ppx = _module
remote = _module
state = _module
trace = _module
util = _module
setup = _module
conftest = _module
gum_marsaglia = _module
rejection_sampling = _module
test_dataset = _module
test_diagnostics = _module
test_distributions = _module
test_distributions_remote = _module
test_inference = _module
test_inference_remote = _module
test_model = _module
test_model_remote = _module
test_nn = _module
test_state = _module
test_trace = _module
test_train = _module
test_util = _module

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


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import torch.distributed as dist


from torch.utils.data import DataLoader


import time


import uuid


import copy


import math


import torch.nn.functional as F


import numpy as np


import functools


class EmbeddingCNN2D5C(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv2d(input_channels, 64, 3)
        self._conv2 = nn.Conv2d(64, 64, 3)
        self._conv3 = nn.Conv2d(64, 128, 3)
        self._conv4 = nn.Conv2d(128, 128, 3)
        self._conv5 = nn.Conv2d(128, 128, 3)
        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        self._lin2 = nn.Linear(self._output_dim, self._output_dim)

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        x = nn.MaxPool2d(2)(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        x = torch.relu(self._lin2(x))
        return x.view(torch.Size([-1]) + self._output_shape)


class EmbeddingCNN3D5C(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv3d(input_channels, 64, 3)
        self._conv2 = nn.Conv3d(64, 64, 3)
        self._conv3 = nn.Conv3d(64, 128, 3)
        self._conv4 = nn.Conv3d(128, 128, 3)
        self._conv5 = nn.Conv3d(128, 128, 3)
        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        None
        None
        None

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = nn.MaxPool3d(2)(x)
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        x = nn.MaxPool3d(2)(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        return x.view(torch.Size([-1]) + self._output_shape)


class EmbeddingFeedForward(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=3, activation=torch.relu, activation_last=torch.relu, input_is_one_hot_index=False, input_one_hot_dim=None):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        self._input_dim = util.prod(self._input_shape)
        self._output_dim = util.prod(self._output_shape)
        self._input_is_one_hot_index = input_is_one_hot_index
        self._input_one_hot_dim = input_one_hot_dim
        if input_is_one_hot_index:
            if self._input_dim != 1:
                raise ValueError('If input_is_one_hot_index==True, input_dim should be 1 (the index of one-hot value in a vector of length input_one_hot_dim.)')
            self._input_dim = input_one_hot_dim
        if num_layers < 1:
            raise ValueError('Expecting num_layers >= 1')
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(self._input_dim, self._output_dim))
        else:
            hidden_dim = int((self._input_dim + self._output_dim) / 2)
            layers.append(nn.Linear(self._input_dim, hidden_dim))
            for i in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, self._output_dim))
        self._activation = activation
        self._activation_last = activation_last
        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        if self._input_is_one_hot_index:
            x = torch.stack([util.one_hot(self._input_one_hot_dim, int(v)) for v in x])
        else:
            x = x.view(-1, self._input_dim).float()
        for i in range(len(self._layers)):
            layer = self._layers[i]
            x = layer(x)
            if i == len(self._layers) - 1:
                if self._activation_last is not None:
                    x = self._activation_last(x)
            else:
                x = self._activation(x)
        return x.view(torch.Size([-1]) + self._output_shape)


class Batch:

    def __init__(self, traces):
        self.traces = traces
        self.size = len(traces)
        sub_batches = {}
        total_length_controlled = 0
        for trace in traces:
            tl = trace.length_controlled
            if tl == 0:
                raise ValueError('Trace of length zero.')
            total_length_controlled += tl
            trace_hash = ''.join([variable.address for variable in trace.variables_controlled])
            if trace_hash not in sub_batches:
                sub_batches[trace_hash] = []
            sub_batches[trace_hash].append(trace)
        self.sub_batches = list(sub_batches.values())
        self.mean_length_controlled = total_length_controlled / self.size

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, key):
        return self.traces[key]

    def to(self, device):
        for trace in self.traces:
            trace.to(device=device)


class Optimizer(enum.Enum):
    ADAM = 0
    SGD = 1
    ADAM_LARC = 2
    SGD_LARC = 3


class LearningRateScheduler(enum.Enum):
    NONE = 0
    POLY1 = 1
    POLY2 = 2


class ObserveEmbedding(enum.Enum):
    FEEDFORWARD = 0
    CNN2D5C = 1
    CNN3D5C = 2


class InferenceNetwork(nn.Module):

    def __init__(self, model, observe_embeddings={}, network_type=''):
        super().__init__()
        self._model = model
        self._layers_observe_embedding = nn.ModuleDict()
        self._layers_observe_embedding_final = None
        self._layers_pre_generated = False
        self._layers_initialized = False
        self._observe_embeddings = observe_embeddings
        self._observe_embedding_dim = None
        self._infer_observe = None
        self._infer_observe_embedding = {}
        self._optimizer = None
        self._optimizer_type = None
        self._optimizer_state = None
        self._momentum = None
        self._weight_decay = None
        self._learning_rate_scheduler = None
        self._learning_rate_scheduler_type = None
        self._learning_rate_scheduler_state = None
        self._total_train_seconds = 0
        self._total_train_traces = 0
        self._total_train_traces_end = None
        self._total_train_iterations = 0
        self._learning_rate_init = None
        self._learning_rate_end = None
        self._loss_init = None
        self._loss_min = float('inf')
        self._loss_max = None
        self._loss_previous = float('inf')
        self._history_train_loss = []
        self._history_train_loss_trace = []
        self._history_valid_loss = []
        self._history_valid_loss_trace = []
        self._history_num_params = []
        self._history_num_params_trace = []
        self._distributed_train_loss = util.to_tensor(0.0)
        self._distributed_valid_loss = util.to_tensor(0.0)
        self._distributed_history_train_loss = []
        self._distributed_history_train_loss_trace = []
        self._distributed_history_valid_loss = []
        self._distributed_history_valid_loss_trace = []
        self._modified = util.get_time_str()
        self._updates = 0
        self._on_cuda = False
        self._device = torch.device('cpu')
        self._learning_rate = None
        self._momentum = None
        self._batch_size = None
        self._distributed_backend = None
        self._distributed_world_size = None
        self._network_type = network_type

    def _init_layers_observe_embedding(self, observe_embeddings, example_trace):
        if len(observe_embeddings) == 0:
            raise ValueError('At least one observe embedding is needed to initialize inference network.')
        if isinstance(observe_embeddings, set):
            observe_embeddings = {o: {} for o in observe_embeddings}
        observe_embedding_total_dim = 0
        for name, value in observe_embeddings.items():
            variable = example_trace.named_variables[name]
            if 'reshape' in value:
                input_shape = torch.Size(value['reshape'])
                None
            else:
                input_shape = variable.value.size()
                None
            if 'dim' in value:
                output_shape = torch.Size([value['dim']])
                None
            else:
                None
                output_shape = torch.Size([256])
            if 'embedding' in value:
                embedding = value['embedding']
                None
            else:
                None
                embedding = ObserveEmbedding.FEEDFORWARD
            if embedding == ObserveEmbedding.FEEDFORWARD:
                if 'depth' in value:
                    depth = value['depth']
                    None
                else:
                    None
                    depth = 2
                layer = EmbeddingFeedForward(input_shape=input_shape, output_shape=output_shape, num_layers=depth)
            elif embedding == ObserveEmbedding.CNN2D5C:
                layer = EmbeddingCNN2D5C(input_shape=input_shape, output_shape=output_shape)
            elif embedding == ObserveEmbedding.CNN3D5C:
                layer = EmbeddingCNN3D5C(input_shape=input_shape, output_shape=output_shape)
            else:
                raise ValueError('Unknown embedding: {}'.format(embedding))
            layer
            self._layers_observe_embedding[name] = layer
            observe_embedding_total_dim += util.prod(output_shape)
        self._observe_embedding_dim = observe_embedding_total_dim
        None
        self._layers_observe_embedding_final = EmbeddingFeedForward(input_shape=self._observe_embedding_dim, output_shape=self._observe_embedding_dim, num_layers=2)
        self._layers_observe_embedding_final

    def _embed_observe(self, traces=None):
        embedding = []
        for name, layer in self._layers_observe_embedding.items():
            values = torch.stack([util.to_tensor(trace.named_variables[name].value) for trace in traces]).view(len(traces), -1)
            embedding.append(layer(values))
        embedding = torch.cat(embedding, dim=1)
        embedding = self._layers_observe_embedding_final(embedding)
        return embedding

    def _infer_init(self, observe=None):
        self._infer_observe = observe
        embedding = []
        for name, layer in self._layers_observe_embedding.items():
            value = util.to_tensor(observe[name]).view(1, -1)
            embedding.append(layer(value))
        embedding = torch.cat(embedding, dim=1)
        self._infer_observe_embedding = self._layers_observe_embedding_final(embedding)

    def _init_layers(self):
        raise NotImplementedError()

    def _polymorph(self, batch):
        raise NotImplementedError()

    def _infer_step(self, variable, previous_variable=None, proposal_min_train_iterations=None):
        raise NotImplementedError()

    def _loss(self, batch):
        raise NotImplementedError()

    def _save(self, file_name):
        self._modified = util.get_time_str()
        self._updates += 1
        data = {}
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__
        data['inference_network'] = copy.copy(self)
        data['inference_network']._model = None
        data['inference_network']._optimizer = None
        if self._optimizer is None:
            data['inference_network']._optimizer_state = None
        else:
            data['inference_network']._optimizer_state = self._optimizer.state_dict()
        data['inference_network']._learning_rate_scheduler = None
        if self._learning_rate_scheduler is None:
            data['inference_network']._learning_rate_scheduler_state = None
        else:
            data['inference_network']._learning_rate_scheduler_state = self._learning_rate_scheduler.state_dict()

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_inference_network')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_inference_network')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    @staticmethod
    def _load(file_name):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_inference_network')
            tar.extract('pyprob_inference_network', tmp_dir)
            tar.close()
            if util._cuda_enabled:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
            shutil.rmtree(tmp_dir)
        except Exception as e:
            None
            raise RuntimeError('Cannot load inference network.')
        if data['pyprob_version'] != __version__:
            None
        if data['torch_version'] != torch.__version__:
            None
        ret = data['inference_network']
        if util._cuda_enabled:
            if ret._on_cuda:
                if ret._device != util._device:
                    None
            else:
                None
        elif ret._on_cuda:
            None
        ret
        if not hasattr(ret, '_distributed_train_loss'):
            ret._distributed_train_loss = util.to_tensor(0.0)
        if not hasattr(ret, '_distributed_valid_loss'):
            ret._distributed_valid_loss = util.to_tensor(0.0)
        if not hasattr(ret, '_distributed_history_train_loss'):
            ret._distributed_history_train_loss = []
        if not hasattr(ret, '_distributed_history_train_loss_trace'):
            ret._distributed_history_train_loss_trace = []
        if not hasattr(ret, '_distributed_history_valid_loss'):
            ret._distributed_history_valid_loss = []
        if not hasattr(ret, '_distributed_history_valid_loss_trace'):
            ret._distributed_history_valid_loss_trace = []
        if not hasattr(ret, '_optimizer_state'):
            ret._optimizer_state = None
        if not hasattr(ret, '_learning_rate_scheduler_state'):
            ret._learning_rate_scheduler_state = None
        if not hasattr(ret, '_total_train_traces_end'):
            ret._total_train_traces_end = None
        if not hasattr(ret, '_loss_init'):
            ret._loss_init = None
        if not hasattr(ret, '_learning_rate_init'):
            ret._learning_rate_init = 0
        if not hasattr(ret, '_learning_rate_end'):
            ret._learning_rate_end = 0
        if not hasattr(ret, '_weight_decay'):
            ret._weight_decay = 0
        if not hasattr(ret, '_learning_rate_scheduler_type'):
            ret._learning_rate_scheduler_type = None
        ret._create_optimizer(ret._optimizer_state)
        ret._create_lr_scheduler(ret._learning_rate_scheduler_state)
        return ret

    def to(self, device=None, *args, **kwargs):
        self._device = device
        self._on_cuda = 'cuda' in str(device)
        super()

    def _pre_generate_layers(self, dataset, batch_size=64, save_file_name_prefix=None):
        if not self._layers_initialized:
            self._init_layers_observe_embedding(self._observe_embeddings, example_trace=dataset.__getitem__(0))
            self._init_layers()
            self._layers_initialized = True
        self._layers_pre_generated = True
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: Batch(x))
        util.progress_bar_init('Layer pre-generation...', len(dataset), 'Traces')
        i = 0
        for i_batch, batch in enumerate(dataloader):
            i += len(batch)
            layers_changed = self._polymorph(batch)
            util.progress_bar_update(i)
            if layers_changed and save_file_name_prefix is not None:
                file_name = '{}_00000000_pre_generated.network'.format(save_file_name_prefix)
                None
                self._save(file_name)
        util.progress_bar_end('Layer pre-generation complete')

    def _distributed_sync_parameters(self):
        """ broadcast rank 0 parameter to all ranks """
        for param in self.parameters():
            dist.broadcast(param.data, 0)

    def _distributed_sync_grad(self, world_size):
        """ all_reduce grads from all ranks """
        ttmap = util.to_tensor([(1 if p.grad is not None else 0) for p in self.parameters()])
        pytorch_allreduce_supports_list = True
        try:
            dist.all_reduce([ttmap])
        except:
            pytorch_allreduce_supports_list = False
            dist.all_reduce(ttmap)
        gl = []
        for i, param in enumerate(self.parameters()):
            if param.grad is not None:
                gl.append(param.grad.data)
            elif ttmap[i]:
                param.grad = util.to_tensor(torch.zeros_like(param.data))
                gl.append(param.grad.data)
        if pytorch_allreduce_supports_list:
            dist.all_reduce(gl)
        else:
            for g in gl:
                dist.all_reduce(g)
        for li in gl:
            li /= float(world_size)

    def _distributed_update_train_loss(self, loss, world_size):
        self._distributed_train_loss = util.to_tensor(float(loss))
        dist.all_reduce(self._distributed_train_loss)
        self._distributed_train_loss /= float(world_size)
        self._distributed_history_train_loss.append(float(self._distributed_train_loss))
        self._distributed_history_train_loss_trace.append(self._total_train_traces)
        return self._distributed_train_loss

    def _distributed_update_valid_loss(self, loss, world_size):
        self._distributed_valid_loss = util.to_tensor(float(loss))
        dist.all_reduce(self._distributed_valid_loss)
        self._distributed_valid_loss /= float(world_size)
        self._distributed_history_valid_loss.append(float(self._distributed_valid_loss))
        self._distributed_history_valid_loss_trace.append(self._total_train_traces)
        return self._distributed_valid_loss

    def _create_optimizer(self, state_dict=None):
        if self._optimizer_type is None:
            return
        if self._optimizer_type in [Optimizer.ADAM, Optimizer.ADAM_LARC]:
            self._optimizer = optim.Adam(self.parameters(), lr=self._learning_rate_init, weight_decay=self._weight_decay)
        else:
            self._optimizer = optim.SGD(self.parameters(), lr=self._learning_rate_init, momentum=self._momentum, nesterov=True, weight_decay=self._weight_decay)
        if self._optimizer_type in [Optimizer.ADAM_LARC, Optimizer.SGD_LARC]:
            self._optimizer = LARC(self._optimizer)
        if state_dict is not None:
            self._optimizer.load_state_dict(state_dict)

    def _create_lr_scheduler(self, state_dict=None):
        if self._learning_rate_scheduler_type is None:
            return
        learning_rate_scheduler_type = self._learning_rate_scheduler_type
        iter_end = self._total_train_traces_end
        lr_init = self._learning_rate_init
        lr_end = self._learning_rate_end

        def _poly_decay(iter, power):
            return (lr_init - lr_end) * (1 - iter / iter_end) ** power + lr_end
        if self._optimizer is None:
            self._learning_rate_scheduler = None
        elif learning_rate_scheduler_type == LearningRateScheduler.POLY1:
            self._learning_rate_scheduler = lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda iter: _poly_decay(iter, power=1.0) / lr_init)
        elif learning_rate_scheduler_type == LearningRateScheduler.POLY2:
            self._learning_rate_scheduler = lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda iter: _poly_decay(iter, power=2.0) / lr_init)
        else:
            self._learning_rate_scheduler = None
        if self._learning_rate_scheduler is not None and state_dict is not None:
            self._learning_rate_scheduler.load_state_dict(state_dict)

    def optimize(self, num_traces, dataset, dataset_valid=None, num_traces_end=1000000000.0, batch_size=64, valid_every=None, optimizer_type=Optimizer.ADAM, learning_rate_init=0.0001, learning_rate_end=1e-06, learning_rate_scheduler_type=LearningRateScheduler.NONE, momentum=0.9, weight_decay=1e-05, save_file_name_prefix=None, save_every_sec=600, distributed_backend=None, distributed_params_sync_every_iter=10000, distributed_num_buckets=10, dataloader_offline_num_workers=0, stop_with_bad_loss=False, log_file_name=None):
        if not self._layers_initialized:
            self._init_layers_observe_embedding(self._observe_embeddings, example_trace=dataset.__getitem__(0))
            self._init_layers()
            self._layers_initialized = True
        if distributed_backend is None:
            distributed_world_size = 1
            distributed_rank = 0
        else:
            dist.init_process_group(backend=distributed_backend)
            distributed_world_size = dist.get_world_size()
            distributed_rank = dist.get_rank()
            self._distributed_backend = distributed_backend
            self._distributed_world_size = distributed_world_size
        if isinstance(dataset, OfflineDataset):
            if distributed_world_size == 1:
                dataloader = DataLoader(dataset, batch_sampler=TraceBatchSampler(dataset, batch_size=batch_size, shuffle_batches=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
            else:
                dataloader = DataLoader(dataset, batch_sampler=DistributedTraceBatchSampler(dataset, batch_size=batch_size, num_buckets=distributed_num_buckets, shuffle_batches=True, shuffle_buckets=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=lambda x: Batch(x))
        if dataset_valid is not None:
            if distributed_world_size == 1:
                dataloader_valid = DataLoader(dataset_valid, batch_sampler=TraceBatchSampler(dataset_valid, batch_size=batch_size, shuffle_batches=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
            else:
                dataloader_valid = DataLoader(dataset_valid, batch_sampler=DistributedTraceBatchSampler(dataset_valid, batch_size=batch_size, num_buckets=distributed_num_buckets, shuffle_batches=True, shuffle_buckets=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
            if not self._layers_pre_generated:
                for i_batch, batch in enumerate(dataloader_valid):
                    self._polymorph(batch)
        if distributed_world_size > 1:
            util.init_distributed_print(distributed_rank, distributed_world_size, False)
            if distributed_rank == 0:
                None
                None
                None
                None
                None
                None
                None
                None
        self.train()
        prev_total_train_seconds = self._total_train_seconds
        time_start = time.time()
        time_loss_min = time_start
        time_last_batch = time_start
        if valid_every is None:
            valid_every = max(100, num_traces / 1000)
        last_validation_trace = -valid_every + 1
        valid_loss = 0
        if self._optimizer_type is None:
            self._optimizer_type = optimizer_type
        if self._momentum is None:
            self._momentum = momentum
        if self._weight_decay is None:
            self._weight_decay = weight_decay
        if self._learning_rate_scheduler_type is None:
            self._learning_rate_scheduler_type = learning_rate_scheduler_type
        if self._learning_rate_init is None:
            self._learning_rate_init = learning_rate_init * math.sqrt(distributed_world_size)
        if self._learning_rate_end is None:
            self._learning_rate_end = learning_rate_end
        if self._total_train_traces_end is None:
            self._total_train_traces_end = num_traces_end
        epoch = 0
        trace = 0
        stop = False
        None
        max_print_line_len = 0
        loss_min_str = ''
        time_since_loss_min_str = ''
        loss_init_str = '' if self._loss_init is None else '{:+.2e}'.format(self._loss_init)
        if save_every_sec is not None:
            last_auto_save_time = time_start - save_every_sec
        last_print = time_start - util._print_refresh_rate
        if distributed_rank == 0 and log_file_name is not None:
            log_file = open(log_file_name, mode='w', buffering=1)
            log_file.write('time, iteration, trace, loss, valid_loss, learning_rate, mean_trace_length_controlled, sub_mini_batches, distributed_bucket_id, traces_per_second\n')
        while not stop:
            epoch += 1
            for i_batch, batch in enumerate(dataloader):
                time_batch = time.time()
                if distributed_world_size > 1 and self._total_train_iterations % distributed_params_sync_every_iter == 0:
                    self._distributed_sync_parameters()
                if self._layers_pre_generated:
                    layers_changed = False
                else:
                    layers_changed = self._polymorph(batch)
                if self._optimizer is None or layers_changed:
                    self._create_optimizer()
                    self._create_lr_scheduler()
                self._optimizer.zero_grad()
                success, loss = self._loss(batch)
                if not success:
                    None
                    if stop_with_bad_loss:
                        return
                else:
                    loss.backward()
                    if distributed_world_size > 1:
                        self._distributed_sync_grad(distributed_world_size)
                    self._optimizer.step()
                    loss = float(loss)
                    if distributed_world_size > 1:
                        loss = self._distributed_update_train_loss(loss, distributed_world_size)
                    if self._loss_init is None:
                        self._loss_init = loss
                        self._loss_max = loss
                        loss_init_str = '{:+.2e}'.format(self._loss_init)
                    if loss < self._loss_min:
                        self._loss_min = loss
                        loss_str = colored('{:+.2e}'.format(loss), 'green', attrs=['bold'])
                        loss_min_str = colored('{:+.2e}'.format(self._loss_min), 'green', attrs=['bold'])
                        time_loss_min = time_batch
                        time_since_loss_min_str = colored(util.days_hours_mins_secs_str(0), 'green', attrs=['bold'])
                    elif loss > self._loss_max:
                        self._loss_max = loss
                        loss_str = colored('{:+.2e}'.format(loss), 'red', attrs=['bold'])
                    else:
                        if loss < self._loss_previous:
                            loss_str = colored('{:+.2e}'.format(loss), 'green')
                        elif loss > self._loss_previous:
                            loss_str = colored('{:+.2e}'.format(loss), 'red')
                        else:
                            loss_str = '{:+.2e}'.format(loss)
                        loss_min_str = '{:+.2e}'.format(self._loss_min)
                        time_since_loss_min_str = util.days_hours_mins_secs_str(time_batch - time_loss_min)
                    self._loss_previous = loss
                    self._total_train_iterations += 1
                    trace += batch.size * distributed_world_size
                    self._total_train_traces += batch.size * distributed_world_size
                    self._total_train_seconds = prev_total_train_seconds + (time_batch - time_start)
                    self._history_train_loss.append(loss)
                    self._history_train_loss_trace.append(self._total_train_traces)
                    traces_per_second = batch.size * distributed_world_size / (time_batch - time_last_batch)
                    if dataset_valid is not None:
                        if trace - last_validation_trace > valid_every:
                            None
                            valid_loss = 0
                            with torch.no_grad():
                                for i_batch, batch in enumerate(dataloader_valid):
                                    _, v = self._loss(batch)
                                    valid_loss += v
                            valid_loss = float(valid_loss) / (len(dataloader_valid) / distributed_world_size)
                            if distributed_world_size > 1:
                                valid_loss = self._distributed_update_valid_loss(valid_loss, distributed_world_size)
                            self._history_valid_loss.append(valid_loss)
                            self._history_valid_loss_trace.append(self._total_train_traces)
                            last_validation_trace = trace - 1
                    if distributed_rank == 0 and save_file_name_prefix is not None and save_every_sec is not None:
                        if time_batch - last_auto_save_time > save_every_sec:
                            last_auto_save_time = time_batch
                            file_name = '{}_{}_traces_{}.network'.format(save_file_name_prefix, util.get_time_stamp(), self._total_train_traces)
                            None
                            self._save(file_name)
                    time_last_batch = time_batch
                    if trace >= num_traces:
                        None
                        stop = True
                    if self._total_train_traces >= self._total_train_traces_end:
                        None
                        if self._learning_rate_scheduler is not None:
                            None
                    if self._learning_rate_scheduler is not None:
                        self._learning_rate_scheduler.step(self._total_train_traces)
                    learning_rate_current = self._optimizer.param_groups[0]['lr']
                    learning_rate_current_str = '{:+.2e}'.format(learning_rate_current)
                    if time_batch - last_print > util._print_refresh_rate or stop:
                        last_print = time_batch
                        total_training_seconds_str = util.days_hours_mins_secs_str(self._total_train_seconds)
                        epoch_str = '{:4}'.format('{:,}'.format(epoch))
                        total_train_traces_str = '{:9}'.format('{:,}'.format(self._total_train_traces))
                        traces_per_second_str = '{:,.1f}'.format(traces_per_second)
                        print_line = '{} | {} | {} | {} | {} | {} | {} | {} | {} '.format(total_training_seconds_str, epoch_str, total_train_traces_str, loss_init_str, loss_min_str, loss_str, time_since_loss_min_str, learning_rate_current_str, traces_per_second_str)
                        max_print_line_len = max(len(print_line), max_print_line_len)
                        None
                        sys.stdout.flush()
                    if distributed_rank == 0 and log_file_name is not None:
                        bucket_id = None
                        if isinstance(dataloader.batch_sampler, DistributedTraceBatchSampler):
                            bucket_id = dataloader.batch_sampler._current_bucket_id
                        log_file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(self._total_train_seconds, self._total_train_iterations, self._total_train_traces, loss, valid_loss, learning_rate_current, batch.mean_length_controlled, len(batch.sub_batches), bucket_id, traces_per_second))
                    if stop:
                        break
        if distributed_rank == 0 and log_file_name is not None:
            log_file.close()
        None
        if distributed_rank == 0 and save_file_name_prefix is not None:
            file_name = '{}_{}_traces_{}.network'.format(save_file_name_prefix, util.get_time_stamp(), self._total_train_traces)
            None
            self._save(file_name)


class Distribution:

    def __init__(self, name, address_suffix='', batch_shape=torch.Size(), event_shape=torch.Size(), torch_dist=None):
        self.name = name
        self._address_suffix = address_suffix
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._torch_dist = torch_dist

    @property
    def batch_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.batch_shape
        else:
            return self._batch_shape

    @property
    def event_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.event_shape
        else:
            return self._event_shape

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value, sum=False):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(util.to_tensor(value))
            return torch.sum(lp) if sum else lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        return torch.exp(self.log_prob(util.to_tensor(value)))

    def plot(self, min_val=-10, max_val=10, step_size=0.1, figsize=(10, 5), xlabel=None, ylabel='Probability', xticks=None, yticks=None, log_xscale=False, log_yscale=False, file_name=None, show=True, fig=None, *args, **kwargs):
        if fig is None:
            if not show:
                mpl.rcParams['axes.unicode_minus'] = False
                plt.switch_backend('agg')
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
        xvals = np.arange(min_val, max_val, step_size)
        plt.plot(xvals, [torch.exp(self.log_prob(x)) for x in xvals], *args, **kwargs)
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log', nonposy='clip')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if file_name is not None:
            plt.savefig(file_name)
        if show:
            plt.show()

    @property
    def mean(self):
        if self._torch_dist is not None:
            return self._torch_dist.mean
        else:
            raise NotImplementedError()

    @property
    def variance(self):
        if self._torch_dist is not None:
            return self._torch_dist.variance
        else:
            raise NotImplementedError()

    @property
    def stddev(self):
        return self.variance.sqrt()

    def expectation(self, func):
        raise NotImplementedError()

    @staticmethod
    def kl_divergence(distribution_1, distribution_2):
        if distribution_1._torch_dist is None or distribution_2._torch_dist is None:
            raise ValueError('KL divergence is not currently supported for this pair of distributions.')
        return torch.distributions.kl.kl_divergence(distribution_1._torch_dist, distribution_2._torch_dist)


class Categorical(Distribution):

    def __init__(self, probs=None, logits=None):
        if probs is not None:
            probs = util.to_tensor(probs)
            if probs.dim() == 0:
                raise ValueError('probs cannot be a scalar.')
        if logits is not None:
            logits = util.to_tensor(logits)
            if logits.dim() == 0:
                raise ValueError('logits cannot be a scalar.')
        torch_dist = torch.distributions.Categorical(probs=probs, logits=logits)
        self._probs = torch_dist.probs
        self._logits = torch_dist.logits
        self._num_categories = self._probs.size(-1)
        super().__init__(name='Categorical', address_suffix='Categorical(len_probs:{})'.format(self._probs.size(-1)), torch_dist=torch_dist)

    def __repr__(self):
        return 'Categorical(num_categories: {}, probs:{})'.format(self.num_categories, self.probs)

    @property
    def num_categories(self):
        return self._num_categories

    @property
    def probs(self):
        return self._probs

    @property
    def logits(self):
        return self._logits


class ProposalCategoricalCategorical(nn.Module):

    def __init__(self, input_shape, num_categories, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([num_categories]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        probs = torch.softmax(x, dim=1).view(batch_size, -1) + util._epsilon
        return Categorical(probs)


class Normal(Distribution):

    def __init__(self, loc, scale):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        super().__init__(name='Normal', address_suffix='Normal', torch_dist=torch.distributions.Normal(loc, scale))

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self.mean, self.stddev)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)


class ProposalNormalNormal(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._output_dim].view(batch_size, -1)
        stddevs = torch.exp(x[:, self._output_dim:]).view(batch_size, -1)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(means.size())
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(stddevs.size())
        means = prior_means + means * prior_stddevs
        stddevs = stddevs * prior_stddevs
        means = means.view(self._output_shape)
        stddevs = stddevs.view(self._output_shape)
        return Normal(means, stddevs)


class Mixture(Distribution):

    def __init__(self, distributions, probs=None):
        self._distributions = distributions
        self.length = len(distributions)
        if probs is None:
            self._probs = util.to_tensor(torch.zeros(self.length)).fill_(1.0 / self.length)
        else:
            self._probs = util.to_tensor(probs)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        self._log_probs = torch.log(util.clamp_probs(self._probs))
        event_shape = torch.Size()
        if self._probs.dim() == 1:
            batch_shape = torch.Size()
            self._batch_length = 0
        elif self._probs.dim() == 2:
            batch_shape = torch.Size([self._probs.size(0)])
            self._batch_length = self._probs.size(0)
        else:
            raise ValueError('Expecting a 1d or 2d (batched) mixture probabilities.')
        self._mixing_dist = Categorical(self._probs)
        self._mean = None
        self._variance = None
        super().__init__(name='Mixture', address_suffix='Mixture({})'.format(', '.join([d._address_suffix for d in self._distributions])), batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'Mixture(distributions:({}), probs:{})'.format(', '.join([repr(d) for d in self._distributions]), self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value, sum=False):
        if self._batch_length == 0:
            value = util.to_tensor(value).squeeze()
            lp = torch.logsumexp(self._log_probs + util.to_tensor([d.log_prob(value) for d in self._distributions]), dim=0)
        else:
            value = util.to_tensor(value).view(self._batch_length)
            lp = torch.logsumexp(self._log_probs + torch.stack([d.log_prob(value).squeeze(-1) for d in self._distributions]).view(-1, self._batch_length).t(), dim=1)
        return torch.sum(lp) if sum else lp

    def sample(self):
        if self._batch_length == 0:
            i = int(self._mixing_dist.sample())
            return self._distributions[i].sample()
        else:
            indices = self._mixing_dist.sample()
            dist_samples = []
            for d in self._distributions:
                sample = d.sample()
                if sample.dim() == 0:
                    sample = sample.unsqueeze(-1)
                dist_samples.append(sample)
            ret = []
            for b in range(self._batch_length):
                i = int(indices[b])
                ret.append(dist_samples[i][b])
            return util.to_tensor(ret)

    @property
    def mean(self):
        if self._mean is None:
            means = torch.stack([d.mean for d in self._distributions])
            if self._batch_length == 0:
                self._mean = torch.dot(self._probs, means)
            else:
                self._mean = torch.diag(torch.mm(self._probs, means))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            variances = torch.stack([((d.mean - self.mean).pow(2) + d.variance) for d in self._distributions])
            if self._batch_length == 0:
                self._variance = torch.dot(self._probs, variances)
            else:
                self._variance = torch.diag(torch.mm(self._probs, variances))
        return self._variance


class ProposalNormalNormalMixture(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(batch_size, -1)
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(batch_size, -1)
        prior_means = prior_means.expand_as(means)
        prior_stddevs = prior_stddevs.expand_as(stddevs)
        means = prior_means + means * prior_stddevs
        stddevs = stddevs * prior_stddevs
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        distributions = [Normal(means[:, i:i + 1].view(batch_size), stddevs[:, i:i + 1].view(batch_size)) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class TruncatedNormal(Distribution):

    def __init__(self, mean_non_truncated, stddev_non_truncated, low, high, clamp_mean_between_low_high=False):
        self._mean_non_truncated = util.to_tensor(mean_non_truncated)
        self._stddev_non_truncated = util.to_tensor(stddev_non_truncated)
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        if clamp_mean_between_low_high:
            self._mean_non_truncated = torch.max(torch.min(self._mean_non_truncated, self._high), self._low)
        if self._mean_non_truncated.dim() == 0:
            self._batch_length = 0
        elif self._mean_non_truncated.dim() == 1 or self._mean_non_truncated.dim() == 2:
            self._batch_length = self._mean_non_truncated.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        self._standard_normal_dist = Normal(util.to_tensor(torch.zeros_like(self._mean_non_truncated)), util.to_tensor(torch.ones_like(self._stddev_non_truncated)))
        self._alpha = (self._low - self._mean_non_truncated) / self._stddev_non_truncated
        self._beta = (self._high - self._mean_non_truncated) / self._stddev_non_truncated
        self._standard_normal_cdf_alpha = self._standard_normal_dist.cdf(self._alpha)
        self._standard_normal_cdf_beta = self._standard_normal_dist.cdf(self._beta)
        self._Z = self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha
        self._log_stddev_Z = torch.log(self._stddev_non_truncated * self._Z)
        self._mean = None
        self._variance = None
        batch_shape = self._mean_non_truncated.size()
        event_shape = torch.Size()
        super().__init__(name='TruncatedNormal', address_suffix='TruncatedNormal', batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'TruncatedNormal(mean_non_truncated:{}, stddev_non_truncated:{}, low:{}, high:{})'.format(self._mean_non_truncated, self._stddev_non_truncated, self._low, self._high)

    def log_prob(self, value, sum=False):
        value = util.to_tensor(value)
        lb = value.ge(self._low).type_as(self._low)
        ub = value.le(self._high).type_as(self._low)
        lp = torch.log(lb.mul(ub)) + self._standard_normal_dist.log_prob((value - self._mean_non_truncated) / self._stddev_non_truncated) - self._log_stddev_Z
        if self._batch_length == 1:
            lp = lp.squeeze(0)
        if util.has_nan_or_inf(lp):
            print(colored('Warning: NaN, -Inf, or Inf encountered in TruncatedNormal log_prob.', 'red', attrs=['bold']))
            print('distribution', self)
            print('value', value)
            print('log_prob', lp)
        return torch.sum(lp) if sum else lp

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def mean_non_truncated(self):
        return self._mean_non_truncated

    @property
    def stddev_non_truncated(self):
        return self._stddev_non_truncated

    @property
    def variance_non_truncated(self):
        return self._stddev_non_truncated.pow(2)

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._mean_non_truncated + self._stddev_non_truncated * (self._standard_normal_dist.prob(self._alpha) - self._standard_normal_dist.prob(self._beta)) / self._Z
            if self._batch_length == 1:
                self._mean = self._mean.squeeze(0)
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            standard_normal_prob_alpha = self._standard_normal_dist.prob(self._alpha)
            standard_normal_prob_beta = self._standard_normal_dist.prob(self._beta)
            self._variance = self._stddev_non_truncated.pow(2) * (1 + (self._alpha * standard_normal_prob_alpha - self._beta * standard_normal_prob_beta) / self._Z - ((standard_normal_prob_alpha - standard_normal_prob_beta) / self._Z).pow(2))
            if self._batch_length == 1:
                self._variance = self._variance.squeeze(0)
        return self._variance

    def sample(self):
        shape = self._low.size()
        attempt_count = 0
        ret = util.to_tensor(torch.zeros(shape).fill_(float('NaN')))
        outside_domain = True
        while util.has_nan_or_inf(ret) or outside_domain:
            attempt_count += 1
            if attempt_count == 10000:
                print('Warning: trying to sample from the tail of a truncated normal distribution, which can take a long time. A more efficient implementation is pending.')
            rand = util.to_tensor(torch.zeros(shape).uniform_())
            ret = self._standard_normal_dist.icdf(self._standard_normal_cdf_alpha + rand * (self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha)) * self._stddev_non_truncated + self._mean_non_truncated
            lb = ret.ge(self._low).type_as(self._low)
            ub = ret.lt(self._high).type_as(self._low)
            outside_domain = int(torch.sum(lb.mul(ub))) == 0
        if self._batch_length == 1:
            ret = ret.squeeze(0)
        return ret


class ProposalPoissonTruncatedNormalMixture(nn.Module):

    def __init__(self, input_shape, output_shape, low=0, high=40, num_layers=2, mixture_components=10):
        super().__init__()
        self._low = low
        self._high = high
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        means = torch.sigmoid(means)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        prior_lows = torch.zeros(batch_size).fill_(self._low)
        prior_highs = torch.zeros(batch_size).fill_(self._high)
        means = prior_lows.view(batch_size, -1).expand_as(means) + means * (prior_highs - prior_lows).view(batch_size, -1).expand_as(means)
        distributions = [TruncatedNormal(means[:, i:i + 1].view(batch_size), stddevs[:, i:i + 1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class Beta(Distribution):

    def __init__(self, concentration1, concentration0, low=0, high=1):
        concentration1 = util.to_tensor(concentration1)
        concentration0 = util.to_tensor(concentration0)
        super().__init__(name='Beta', address_suffix='Beta', torch_dist=torch.distributions.Beta(concentration1, concentration0))
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        self._range = self._high - self._low

    def __repr__(self):
        return 'Beta(concentration1:{}, concentration0:{}, low:{}, high:{})'.format(self.concentration1, self.concentration0, self.low, self.high)

    @property
    def concentration1(self):
        return self._torch_dist.concentration1

    @property
    def concentration0(self):
        return self._torch_dist.concentration0

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def sample(self):
        return self._low + super().sample() * self._range

    def log_prob(self, value, sum=False):
        lp = super().log_prob((util.to_tensor(value) - self._low) / self._range, sum=False)
        return torch.sum(lp) if sum else lp

    @property
    def mean(self):
        return self._low + super().mean * self._range

    @property
    def variance(self):
        return super().variance * self._range * self._range


class ProposalUniformBeta(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers, activation=torch.relu, activation_last=torch.relu)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        x = self._ff(x)
        concentration1s = 1.0 + x[:, :self._output_dim].view(self._output_shape)
        concentration0s = 1.0 + x[:, self._output_dim:].view(self._output_shape)
        prior_lows = torch.stack([v.distribution.low for v in prior_variables]).view(concentration1s.size())
        prior_highs = torch.stack([v.distribution.high for v in prior_variables]).view(concentration1s.size())
        return Beta(concentration1s, concentration0s, low=prior_lows, high=prior_highs)


class ProposalUniformBetaMixture(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        concentration1s = x[:, :self._mixture_components].view(batch_size, -1)
        concentration0s = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        concentration1s = 1.0 + torch.relu(concentration1s)
        concentration0s = 1.0 + torch.relu(concentration0s)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_lows = torch.stack([v.distribution.low for v in prior_variables]).view(batch_size)
        prior_highs = torch.stack([v.distribution.high for v in prior_variables]).view(batch_size)
        distributions = [Beta(concentration1s[:, i:i + 1].view(batch_size), concentration0s[:, i:i + 1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class ProposalUniformTruncatedNormalMixture(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        means = torch.sigmoid(means)
        stddevs = torch.sigmoid(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        prior_lows = torch.stack([util.to_tensor(v.distribution.low) for v in prior_variables]).view(batch_size)
        prior_highs = torch.stack([util.to_tensor(v.distribution.high) for v in prior_variables]).view(batch_size)
        prior_range = (prior_highs - prior_lows).view(batch_size, -1)
        means = prior_lows.view(batch_size, -1) + means * prior_range
        stddevs = prior_range / 1000 + stddevs * prior_range * 10
        distributions = [TruncatedNormal(means[:, i:i + 1].view(batch_size), stddevs[:, i:i + 1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)

