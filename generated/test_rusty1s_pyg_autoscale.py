import sys
_module = sys.modules[__name__]
del sys
train_gcn = _module
train_gcn2 = _module
train_gin = _module
main = _module
setup = _module
main = _module
torch_geometric_autoscale = _module
data = _module
history = _module
loader = _module
metis = _module
models = _module
appnp = _module
base = _module
gat = _module
gcn = _module
gcn2 = _module
pna = _module
pna_jk = _module
pool = _module
utils = _module

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


import torch


from torch import Tensor


from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR


from torch.nn import Identity


from torch.nn import Sequential


from torch.nn import Linear


from torch.nn import ReLU


from torch.nn import BatchNorm1d


import time


from torch.__config__ import parallel_info


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from typing import Tuple


from typing import Optional


from typing import NamedTuple


from typing import List


from torch.utils.data import DataLoader


import copy


import torch.nn.functional as F


from torch.nn import ModuleList


from typing import Callable


from typing import Dict


from typing import Any


import warnings


from itertools import product


from torch.cuda import Stream


class History(torch.nn.Module):
    """A historical embedding storage module."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        pin_memory = device is None or str(device) == 'cpu'
        self.emb = torch.empty(num_embeddings, embedding_dim, device=device, pin_memory=pin_memory)
        self._device = torch.device('cpu')
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)

    def _apply(self, fn):
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, n_id: Optional[Tensor]=None) ->Tensor:
        out = self.emb
        if n_id is not None:
            assert n_id.device == self.emb.device
            out = out.index_select(0, n_id)
        return out

    @torch.no_grad()
    def push(self, x, n_id: Optional[Tensor]=None, offset: Optional[Tensor]=None, count: Optional[Tensor]=None):
        if n_id is None and x.size(0) != self.num_embeddings:
            raise ValueError
        elif n_id is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)
        elif offset is None or count is None:
            assert n_id.device == self.emb.device
            self.emb[n_id] = x
        else:
            src_o = 0
            x = x
            for dst_o, c in zip(offset.tolist(), count.tolist()):
                self.emb[dst_o:dst_o + c] = x[src_o:src_o + c]
                src_o += c

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim}, emb_device={self.emb.device}, device={self._device})'


class AsyncIOPool(torch.nn.Module):

    def __init__(self, pool_size: int, buffer_size: int, embedding_dim: int):
        super().__init__()
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.embedding_dim = embedding_dim
        self._device = torch.device('cpu')
        self._pull_queue = []
        self._push_cache = [None] * pool_size
        self._push_streams = [None] * pool_size
        self._pull_streams = [None] * pool_size
        self._cpu_buffers = [None] * pool_size
        self._cuda_buffers = [None] * pool_size
        self._pull_index = -1
        self._push_index = -1

    def _apply(self, fn: Callable) ->None:
        self._device = fn(torch.zeros(1)).device
        return self

    def _pull_stream(self, idx: int) ->Stream:
        if self._pull_streams[idx] is None:
            assert str(self._device)[:4] == 'cuda'
            self._pull_streams[idx] = torch.Stream(self._device)
        return self._pull_streams[idx]

    def _push_stream(self, idx: int) ->Stream:
        if self._push_streams[idx] is None:
            assert str(self._device)[:4] == 'cuda'
            self._push_streams[idx] = torch.Stream(self._device)
        return self._push_streams[idx]

    def _cpu_buffer(self, idx: int) ->Tensor:
        if self._cpu_buffers[idx] is None:
            self._cpu_buffers[idx] = torch.empty(self.buffer_size, self.embedding_dim, pin_memory=True)
        return self._cpu_buffers[idx]

    def _cuda_buffer(self, idx: int) ->Tensor:
        if self._cuda_buffers[idx] is None:
            assert str(self._device)[:4] == 'cuda'
            self._cuda_buffers[idx] = torch.empty(self.buffer_size, self.embedding_dim, device=self._device)
        return self._cuda_buffers[idx]

    @torch.no_grad()
    def async_pull(self, src: Tensor, offset: Optional[Tensor], count: Optional[Tensor], index: Tensor) ->None:
        self._pull_index = (self._pull_index + 1) % self.pool_size
        data = self._pull_index, src, offset, count, index
        self._pull_queue.append(data)
        if len(self._pull_queue) <= self.pool_size:
            self._async_pull(self._pull_index, src, offset, count, index)

    @torch.no_grad()
    def _async_pull(self, idx: int, src: Tensor, offset: Optional[Tensor], count: Optional[Tensor], index: Tensor) ->None:
        with torch.cuda.stream(self._pull_stream(idx)):
            read_async(src, offset, count, index, self._cuda_buffer(idx), self._cpu_buffer(idx))

    @torch.no_grad()
    def synchronize_pull(self) ->Tensor:
        idx = self._pull_queue[0][0]
        synchronize()
        torch.cuda.synchronize(self._pull_stream(idx))
        return self._cuda_buffer(idx)

    @torch.no_grad()
    def free_pull(self) ->None:
        self._pull_queue.pop(0)
        if len(self._pull_queue) >= self.pool_size:
            data = self._pull_queue[self.pool_size - 1]
            idx, src, offset, count, index = data
            self._async_pull(idx, src, offset, count, index)
        elif len(self._pull_queue) == 0:
            self._pull_index = -1

    @torch.no_grad()
    def async_push(self, src: Tensor, offset: Tensor, count: Tensor, dst: Tensor) ->None:
        self._push_index = (self._push_index + 1) % self.pool_size
        self.synchronize_push(self._push_index)
        self._push_cache[self._push_index] = src
        with torch.cuda.stream(self._push_stream(self._push_index)):
            write_async(src, offset, count, dst)

    @torch.no_grad()
    def synchronize_push(self, idx: Optional[int]=None) ->None:
        if idx is None:
            for idx in range(self.pool_size):
                self.synchronize_push(idx)
            self._push_index = -1
        else:
            torch.cuda.synchronize(self._push_stream(idx))
            self._push_cache[idx] = None

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(pool_size={self.pool_size}, buffer_size={self.buffer_size}, embedding_dim={self.embedding_dim}, device={self._device})'


EPS = 1e-05

