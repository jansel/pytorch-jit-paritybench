import sys
_module = sys.modules[__name__]
del sys
train_net = _module
eval_cls_voc = _module
eval_knn = _module
eval_lincls_imagenet = _module
eval_lincls_places = _module
eval_semisup = _module
hcsc = _module
hcsc = _module
loader = _module
logger = _module
pretrain = _module
utils = _module
options = _module
utils = _module
dataset = _module
main = _module
test = _module
train = _module
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


import numpy as np


import torch.utils.data as data


import random


from torch import nn


import torch.distributed as dist


import torch.backends.cudnn as cudnn


from torchvision import datasets


from torchvision import transforms


from torchvision import models


import warnings


import time


import math


import torch.nn as nn


import torch.nn.parallel


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import logging


from logging import getLogger


from collections import defaultdict


from collections import deque


import pandas as pd


from torch.utils.data import DataLoader


import torch.optim as optim


from sklearn.metrics import average_precision_score


import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class HCSC(nn.Module):
    """
    Our proposed HCSC framework with instance selection
    and prototype selection.

    Args:
        base_encoder (nn.Module class): query encoder model class(use ResNet50 by default)
        dim (int): feature dimension (default: 128)
        queue_length: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: temperature 
        mlp: whether to use mlp projection
        multi_crop: (bool) whether using multi crops augmentation
        instance_selection: (bool) whether enable instance selection
        proto_selection: (bool) whether enable prototype selection
        selection_on_local: (bool) whether apply mining strategy on local views.
        logger: (obj) a logger used to store some mediate variables during training.
    """

    def __init__(self, base_encoder, dim=128, queue_length=16384, m=0.999, T=0.2, mlp=True, multi_crop=False, instance_selection=True, proto_selection=True, selection_on_local=True, logger=None, **kwargs):
        super().__init__()
        self.queue_length = queue_length
        self.m = m
        self.T = T
        self.multi_crop = multi_crop
        self.selection_on_local = selection_on_local
        self.logger = logger
        self.instance_selection = instance_selection
        self.proto_selection = proto_selection
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, queue_length))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_index', torch.arange(0, queue_length))
        self.buffer_dict = dict()
        self.mined_index = list()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def momentum_update_key_encoder(self):
        self._momentum_update_key_encoder()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, index=None):
        keys = concat_all_gather(keys)
        if index is not None:
            index = concat_all_gather(index)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_length % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if index is not None:
            self.queue_index[ptr:ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.queue_length
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

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, centroids, density, index):
        """
        mining based on the clustering results
        """
        queue_p_samples = []
        for layer in range(len(im2cluster)):
            proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids[layer].permute(1, 0))
            density[layer] = density[layer].clamp(min=0.001)
            proto_logit /= density[layer]
            label = im2cluster[layer][index]
            logit = proto_logit.clone().detach().softmax(-1)
            p_sample = 1 - logit[:, label].t()
            queue_p_samples.append(p_sample)
        self.selected_masks = []
        avg_sample_ratios = []
        for p_sample in queue_p_samples:
            neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
            selected_mask = neg_sampler.sample()
            try:
                self.selected_masks.append(selected_mask)
                avg_sample_ratios.append(p_sample.mean())
            except:
                selected_mask = torch.ones([index.shape[0], self.queue.shape[1]])
                self.selected_masks.append(selected_mask)
        return self.selected_masks, avg_sample_ratios

    @torch.no_grad()
    def extract_key_feat(self, im_k):
        self._momentum_update_key_encoder()
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
        k = self.encoder_k(im_k)
        k = nn.functional.normalize(k, dim=1)
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return k

    def extract_feat(self, images, is_eval=False):
        if is_eval:
            k = self.encoder_k(images)
            k = nn.functional.normalize(k, dim=1)
            return k
        im_q, im_k = images[0], images[1]
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        if self.multi_crop:
            k = self.extract_key_feat(im_k)
            local_views = list()
            for n, im_local in enumerate(images[2:]):
                local_q = self.encoder_q(im_local)
                local_q = nn.functional.normalize(local_q, dim=1)
                local_views.append(local_q)
            return q, k, local_views
        else:
            k = self.extract_key_feat(im_k)
            return q, k, None

    def forward(self, images, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            images: a list of images, where
                images[0] as im_q and
                images[1] as im_k 
                others are local views, which are also treated 
                as keys
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        if is_eval:
            return self.extract_feat(images, is_eval)
        else:
            q, k, local_views = self.extract_feat(images, is_eval)
        proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(q, index, cluster_result)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        if proto_labels is not None and self.instance_selection:
            try:
                self.selected_masks, sample_ratios = self.sample_neg_instance(cluster_result['im2cluster'], proto_selected, temp_protos, index)
                self.buffer_dict['avg_sample_ratios'] = sum(sample_ratios) / len(sample_ratios)
                l_neg = list()
                for selected_mask in self.selected_masks:
                    logit = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                    mask = selected_mask.clone().float()
                    l_neg.append(logit * mask)
            except:
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        if isinstance(l_neg, list):
            logits = [(torch.cat([l_pos, l_n], dim=1) / self.T) for l_n in l_neg]
            labels = [torch.zeros(logit.shape[0], dtype=torch.long) for logit in logits]
        else:
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
        local_proto_logits, local_proto_targets = None, None
        local_logits, local_labels = None, None
        if self.multi_crop:
            local_logits, local_labels = self.compute_local_logits(q, k, local_views, index)
            local_proto_logits, local_proto_targets = self.compute_local_proto_logits(local_views, proto_selected, cluster_result, index)
        self._dequeue_and_enqueue(k, index)
        return logits, labels, proto_logits, proto_labels, local_logits, local_labels, local_proto_logits, local_proto_targets

    def compute_local_proto_logits(self, local_views, proto_selected, cluster_result, index):
        """
        Compute prototype logits for local views. 
        """
        if cluster_result is not None:
            local_proto_logits = list()
            local_proto_targets = list()
            for local_view in local_views:
                proto_logits, proto_labels, proto_selected, temp_protos = self.get_protos(local_view, index, cluster_result)
                local_proto_logits.append(proto_logits)
                local_proto_targets.append(proto_labels)
            return local_proto_logits, local_proto_targets
        else:
            return None, None

    def compute_local_logits(self, q, k, local_views, index):
        """
        Args:
            q: (torch.Tensor([N, D]))
            k: (torch.Tensor([N, D]))
            local_views: (list[torch.Tensor([N, D])]) 
                features of local views that could be additional keys or
                queries

        Returns:
            local_logits: (list[torch.Tensor([N, queue_length+1])])
            local_labels: (list[torch.Tensor([N])])
        """
        if self.selection_on_local and hasattr(self, 'selected_masks'):
            l_pos_list = list()
            l_neg_list = list()
            for selected_mask in self.selected_masks:
                l_pos_list.extend([torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views])
                for local_view in local_views:
                    logit = torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()])
                    mask = selected_mask.clone().float()
                    l_neg_list.append(logit * mask)
        else:
            l_pos_list = [torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views]
            l_neg_list = [torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()]) for local_view in local_views]
        local_logits = [(torch.cat([l_pos, l_neg], dim=1) / self.T) for l_pos, l_neg in zip(l_pos_list, l_neg_list)]
        local_labels = [torch.zeros(logit.shape[0], dtype=torch.long) for logit in local_logits]
        return local_logits, local_labels

    def get_protos(self, q, index, cluster_result):
        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            proto_selecteds = []
            temp_protos = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]
                proto_selecteds.append(prototypes)
                temp_protos.append(density)
                all_proto_id = [i for i in range(im2cluster.max())]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                if self.proto_selection:
                    if n == len(cluster_result['im2cluster']) - 1:
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id)
                        neg_prototypes = prototypes[neg_proto_id]
                        logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1), torch.mm(q, neg_prototypes.t())], dim=1)
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1), density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                    else:
                        cluster2cluster = cluster_result['cluster2cluster'][n]
                        prot_logits = cluster_result['logits'][n]
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id)
                        neg_prototypes = prototypes[neg_proto_id]
                        neg_mask = self.sample_neg_protos(im2cluster, cluster2cluster, pos_proto_id, prot_logits, n, cluster_result)
                        neg_logit_mask = neg_mask.clone().float()
                        neg_logits = torch.mm(q, neg_prototypes.t())
                        neg_logits *= neg_logit_mask
                        logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1), neg_logits], dim=1)
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1), density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                else:
                    neg_proto_id = list(neg_proto_id)
                    neg_proto_id = torch.LongTensor(neg_proto_id)
                    neg_prototypes = prototypes[neg_proto_id]
                    logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1), torch.mm(q, neg_prototypes.t())], dim=1)
                    temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1), density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                    logits_proto = logits_proto / temp_map
                labels_proto = torch.zeros(q.shape[0], dtype=torch.long)
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return proto_logits, proto_labels, proto_selecteds, temp_protos
        else:
            return None, None, None, None

    def sample_neg_protos(self, im2cluster, cluster2cluster, pos_proto_id, prot_logits, n, cluster_results):
        """
        Sampling negative prototypes given pos_proto_id and layer

        Args:
            im2cluster: [N_bs]
            pos_proto_id: [N_bs] actually im2cluster[index]
            proto_dist_mat: [N_bs, N_l] used for sampling strategy.
            prot_logits: [N_l, N_{l+1}] proto logits of cucrrent layer
        """
        all_proto_id = [i for i in range(im2cluster.max())]
        neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
        neg_proto_id = torch.LongTensor(list(neg_proto_id))
        upper_pos_proto_id = cluster2cluster[pos_proto_id]
        densities = cluster_results['density'][n + 1] / cluster_results['density'][n + 1].mean() * self.T
        sampling_prob = 1 - (prot_logits / densities).softmax(-1)[neg_proto_id, :][:, upper_pos_proto_id].t()
        neg_sampler = torch.distributions.bernoulli.Bernoulli(sampling_prob.clamp(0.0001, 0.999))
        selected_mask = neg_sampler.sample()
        return selected_mask


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]
            output = torch.cat((output, _out))
            start_idx = end_idx
        return self.head(output)

