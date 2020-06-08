import sys
_module = sys.modules[__name__]
del sys
train_net = _module
main_lincls = _module
main_moco = _module
moco = _module
builder = _module
loader = _module

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


import random


import warnings


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import math


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.
        distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp
        =False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.
            encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.
            encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_moco(_paritybench_base):
    pass
