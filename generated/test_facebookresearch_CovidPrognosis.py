import sys
_module = sys.modules[__name__]
del sys
covidprognosis = _module
data = _module
base_dataset = _module
chexpert = _module
collate_fn = _module
combined_datasets = _module
mimic_cxr = _module
nih_chest_xrays = _module
transforms = _module
models = _module
moco_model = _module
plmodules = _module
xray_datamodule = _module
mip_model = _module
train_mip = _module
moco_pretrain = _module
moco_module = _module
train_moco = _module
sip_finetune = _module
train_sip = _module
setup = _module
tests = _module
conftest = _module
test_transforms = _module
test_xray_datasets = _module

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


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import numpy as np


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data._utils.collate import default_collate


from typing import Tuple


import torch


from scipy.ndimage import gaussian_filter


import torch.nn as nn


from torch import Tensor


import math


import torch.nn.functional as F


import torchvision.models as tvmodels


import logging


from warnings import warn


from torchvision import transforms


import torchvision.models as models


import torchvision.transforms as tvt


@torch.no_grad()
def concat_all_gather(tensor: Tensor) ->Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, encoder_q: nn.Module, encoder_k: nn.Module, dim: int=128, K: int=65536, m: float=0.999, T: float=0.07, mlp: bool=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        if mlp:
            if hasattr(self.encoder_q, 'fc'):
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            elif hasattr(self.encoder_q, 'classifier'):
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.classifier)
                self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', nn.functional.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        assert isinstance(self.queue_ptr, Tensor)
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f'batch_size={batch_size}, K={self.K}'
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor) ->Tuple[Tensor, Tensor]:
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
    def _batch_unshuffle_ddp(self, x: Tensor, idx_unshuffle: Tensor) ->Tensor:
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

    def forward(self, im_q: Tensor, im_k: Tensor) ->Tuple[Tensor, Tensor]:
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


class ContinuousPosEncoding(nn.Module):

    def __init__(self, dim, drop=0.1, maxtime=360):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        position = torch.arange(0, maxtime, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(maxtime, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, xs, times):
        ys = xs
        times = times.long()
        for b in range(xs.shape[1]):
            ys[:, b] += self.pe[times[b]]
        return self.dropout(ys)


class MIPModel(nn.Module):

    def __init__(self, image_model, feature_dim, projection_dim, num_classes, num_heads, feedforward_dim, drop_transformer, drop_cpe, pooling, image_shape=(7, 7)):
        super().__init__()
        self.image_shape = image_shape
        self.pooling = pooling
        self.image_model = image_model
        self.group_norm = nn.GroupNorm(32, feature_dim)
        self.projection = nn.Conv2d(feature_dim, projection_dim, (1, 1))
        transformer_dim = projection_dim * image_shape[0] * image_shape[1]
        self.pos_encoding = ContinuousPosEncoding(transformer_dim, drop=drop_cpe)
        self.transformer = nn.TransformerEncoderLayer(d_model=transformer_dim, dim_feedforward=feedforward_dim, nhead=num_heads, dropout=drop_transformer)
        self.classifier = nn.Linear(feature_dim + projection_dim, num_classes)

    def _apply_transformer(self, image_feats: torch.Tensor, times, lens):
        B, N, C, H, W = image_feats.shape
        image_feats = image_feats.flatten(start_dim=2).permute([1, 0, 2])
        image_feats = self.pos_encoding(image_feats, times)
        image_feats = self.transformer(image_feats)
        return image_feats.permute([1, 0, 2]).reshape([B, N, C, H, W])

    def _pool(self, image_feats, lens):
        if self.pooling == 'last_timestep':
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, int(l) - 1])
        elif self.pooling == 'sum':
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, :int(l)].sum(0))
        else:
            raise ValueError(f'Unkown pooling method: {self.pooling}')
        pooled_feats = torch.stack(pooled_feats)
        pooled_feats = F.adaptive_avg_pool2d(pooled_feats, (1, 1))
        return pooled_feats.squeeze(3).squeeze(2)

    def forward(self, images, times, lens):
        B, N, C, H, W = images.shape
        images = images.reshape([B * N, C, H, W])
        image_feats = self.image_model(images)
        image_feats = F.relu(self.group_norm(image_feats))
        image_feats_proj = self.projection(image_feats).reshape([B, N, -1, *self.image_shape])
        image_feats_trans = self._apply_transformer(image_feats_proj, times, lens)
        image_feats = image_feats.reshape([B, N, -1, *self.image_shape])
        image_feats_combined = torch.cat([image_feats, image_feats_trans], dim=2)
        image_feats_pooled = self._pool(image_feats_combined, lens)
        return self.classifier(image_feats_pooled)

