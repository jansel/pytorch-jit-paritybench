import sys
_module = sys.modules[__name__]
del sys
modules = _module
functional = _module
_csrc = _module
syncbn = _module
nn = _module
syncbn = _module
test = _module

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


import torch


import torch.cuda.comm as comm


from torch.autograd import Function


from torch.autograd.function import once_differentiable


import torch.nn as nn


from torch.nn import functional as F


from torch.nn.parameter import Parameter


import numpy as np


from torch import nn


class _BatchNorm(nn.Module):
    """
    Customized BatchNorm from nn.BatchNorm
    >> added freeze attribute to enable bn freeze.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.freezed = False
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        compute_stats = not self.freezed and self.training and self.track_running_stats
        ret = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, compute_stats, self.momentum, self.eps)
        return ret

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)


class BatchNorm2dNoSync(_BatchNorm):
    """
    Equivalent to nn.BatchNorm2d
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.join(this_dir, 'csrc')
    main_file = glob.glob(os.path.join(this_dir, '*.cpp'))
    sources_cpu = glob.glob(os.path.join(this_dir, 'cpu', '*.cpp'))
    sources_cuda = glob.glob(os.path.join(this_dir, 'cuda', '*.cu'))
    sources = main_file + sources_cpu
    extra_cflags = []
    extra_cuda_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        sources.extend(sources_cuda)
        extra_cflags = ['-O3', '-DWITH_CUDA']
        extra_cuda_cflags = ['--expt-extended-lambda']
    sources = [os.path.join(this_dir, s) for s in sources]
    extra_include_paths = [this_dir]
    return load(name='ext_lib', sources=sources, extra_cflags=extra_cflags, extra_include_paths=extra_include_paths, extra_cuda_cflags=extra_cuda_cflags)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class BatchNorm2dSyncFunc(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, extra, compute_stats=True, momentum=0.1, eps=1e-05):

        def _parse_extra(ctx, extra):
            ctx.is_master = extra['is_master']
            if ctx.is_master:
                ctx.master_queue = extra['master_queue']
                ctx.worker_queues = extra['worker_queues']
                ctx.worker_ids = extra['worker_ids']
            else:
                ctx.master_queue = extra['master_queue']
                ctx.worker_queue = extra['worker_queue']
        if extra is not None:
            _parse_extra(ctx, extra)
        ctx.compute_stats = compute_stats
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.affine = weight is not None and bias is not None
        if ctx.compute_stats:
            N = _count_samples(x) * (ctx.master_queue.maxsize + 1)
            assert N > 1
            xsum, xsqsum = _backend.syncbn_sum_sqsum(x.detach())
            if ctx.is_master:
                xsums, xsqsums = [xsum], [xsqsum]
                for _ in range(ctx.master_queue.maxsize):
                    xsum_w, xsqsum_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    xsums.append(xsum_w)
                    xsqsums.append(xsqsum_w)
                xsum = comm.reduce_add(xsums)
                xsqsum = comm.reduce_add(xsqsums)
                mean = xsum / N
                sumvar = xsqsum - xsum * mean
                var = sumvar / N
                uvar = sumvar / (N - 1)
                tensors = comm.broadcast_coalesced((mean, uvar, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((xsum, xsqsum))
                mean, uvar, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * uvar)
            ctx.N = N
            ctx.save_for_backward(x, weight, bias, mean, var)
        else:
            mean, var = running_mean, running_var
        z = _backend.syncbn_forward(x, weight, bias, mean, var, ctx.affine, ctx.eps)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, weight, bias, mean, var = ctx.saved_tensors
        dz = dz.contiguous()
        sum_dz, sum_dz_xhat = _backend.syncbn_backward_xhat(dz, x, mean, var, ctx.eps)
        if ctx.is_master:
            sum_dzs, sum_dz_xhats = [sum_dz], [sum_dz_xhat]
            for _ in range(ctx.master_queue.maxsize):
                sum_dz_w, sum_dz_xhat_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                sum_dzs.append(sum_dz_w)
                sum_dz_xhats.append(sum_dz_xhat_w)
            sum_dz = comm.reduce_add(sum_dzs)
            sum_dz_xhat = comm.reduce_add(sum_dz_xhats)
            sum_dz /= ctx.N
            sum_dz_xhat /= ctx.N
            tensors = comm.broadcast_coalesced((sum_dz, sum_dz_xhat), [mean.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            ctx.master_queue.put((sum_dz, sum_dz_xhat))
            sum_dz, sum_dz_xhat = ctx.worker_queue.get()
            ctx.worker_queue.task_done()
        dx, dweight, dbias = _backend.syncbn_backward(dz, x, weight, bias, mean, var, sum_dz, sum_dz_xhat, ctx.affine, ctx.eps)
        return dx, dweight, dbias, None, None, None, None, None, None


batchnorm2d_sync = BatchNorm2dSyncFunc.apply


class BatchNorm2dSync(BatchNorm2dNoSync):
    """
    BatchNorm2d with automatic multi-GPU Sync
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2dSync, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.sync_enabled = True
        self.devices = list(range(torch.cuda.device_count()))
        if len(self.devices) > 1:
            self.worker_ids = self.devices[1:]
            self.master_queue = Queue(len(self.worker_ids))
            self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        compute_stats = not self.freezed and self.training and self.track_running_stats
        if self.sync_enabled and compute_stats and len(self.devices) > 1:
            if x.get_device() == self.devices[0]:
                extra = {'is_master': True, 'master_queue': self.master_queue, 'worker_queues': self.worker_queues, 'worker_ids': self.worker_ids}
            else:
                extra = {'is_master': False, 'master_queue': self.master_queue, 'worker_queue': self.worker_queues[self.worker_ids.index(x.get_device())]}
            return batchnorm2d_sync(x, self.weight, self.bias, self.running_mean, self.running_var, extra, compute_stats, self.momentum, self.eps)
        return super(BatchNorm2dSync, self).forward(x)

    def __repr__(self):
        """repr"""
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},affine={affine}, track_running_stats={track_running_stats},devices={devices})'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm2dNoSync,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm2dSync,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_tamakoji_pytorch_syncbn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

