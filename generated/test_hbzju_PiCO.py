import sys
_module = sys.modules[__name__]
del sys
model = _module
resnet = _module
train = _module
train_pico_plus = _module
cifar10 = _module
cifar100 = _module
cub200 = _module
randaugment = _module
utils_algo = _module
utils_loss = _module
cifar10 = _module
cifar100 = _module
cub200 = _module
randaugment = _module
utils_algo = _module
utils_loss = _module

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


import torch.nn as nn


from random import sample


import numpy as np


import torch.nn.functional as F


from torchvision import models


import math


import random


import time


import warnings


import torch.nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import copy


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import torchvision.datasets as dsets


from sklearn.preprocessing import OneHotEncoder


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


class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        pretrained = args.dataset == 'cub200'
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer('queue_pseudo', torch.randn(args.moco_queue))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('prototypes', torch.zeros(args.num_class, args.low_dim))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1.0 - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue
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

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        predicted_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)
        for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_b)):
            self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder(args)
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, pseudo_labels_b, args)
        return output, features, pseudo_labels, score_prot


class PiCO_PLUS(PiCO):

    def __init__(self, args, base_encoder):
        super().__init__(args, base_encoder)
        self.register_buffer('queue_rel', torch.zeros(args.moco_queue, dtype=torch.bool))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, is_rel, args):
        super()._dequeue_and_enqueue(keys, labels, args)
        is_rel = concat_all_gather(is_rel)
        batch_size = is_rel.shape[0]
        ptr = int(self.queue_ptr)
        self.queue_rel[ptr:ptr + batch_size] = is_rel

    def forward(self, img_q, im_k=None, Y_ori=None, Y_cor=None, is_rel=None, args=None, eval_only=False):
        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        batch_weight = is_rel.float()
        with torch.no_grad():
            predicetd_scores = torch.softmax(output, dim=1)
            _, within_max_cls = torch.max(predicetd_scores * Y_ori, dim=1)
            _, all_max_cls = torch.max(predicetd_scores, dim=1)
            pseudo_labels_b = batch_weight * within_max_cls + (1 - batch_weight) * all_max_cls
            pseudo_labels_b = within_max_cls
            pseudo_labels_b = pseudo_labels_b.long()
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)
            _, within_max_cls_ori = torch.max(predicetd_scores * Y_ori, dim=1)
            distance_prot = -(q * prototypes[within_max_cls_ori]).sum(dim=1)
            for feat, label in zip(concat_all_gather(q[is_rel]), concat_all_gather(pseudo_labels_b[is_rel])):
                self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            self._momentum_update_key_encoder(args)
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        is_rel_queue = torch.cat((is_rel, is_rel, self.queue_rel.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, pseudo_labels_b, is_rel, args)
        return output, features, pseudo_labels, score_prot, distance_prot, is_rel_queue


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


def Identity(img, v):
    return img


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


model_dict = {'resnet18': [resnet18, 512], 'resnet34': [resnet34, 512], 'resnet50': [resnet50, 2048], 'resnet101': [resnet101, 2048]}


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet18', head='mlp', feat_dim=128, num_class=0, pretrained=False):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        if pretrained:
            model = models.resnet18(pretrained=True)
            model.fc = Identity()
            self.encoder = model
        else:
            self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_class)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError('head not supported: {}'.format(head))
        self.register_buffer('prototypes', torch.zeros(num_class, feat_dim))

    def forward(self, x):
        feat = self.encoder(x)
        feat_c = self.head(feat)
        logits = self.fc(feat)
        return logits, F.normalize(feat_c, dim=1)


def ce_loss(outputs, targets, sel=None):
    targets = targets.detach()
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * targets
    loss_vec = -final_outputs.sum(dim=1)
    if sel is None:
        average_loss = loss_vec.mean()
    else:
        average_loss = loss_vec[sel].mean()
    return loss_vec, average_loss


class partial_loss(nn.Module):

    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.conf_ema_m = conf_ema_m
        self.num_class = confidence.shape[1]

    def set_conf_ema_m(self, args, epoch):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1.0 * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index, is_rel=None):
        confidence = self.confidence[index, :]
        loss_vec, _ = ce_loss(outputs, confidence)
        if is_rel is None:
            average_loss = loss_vec.mean()
        else:
            average_loss = loss_vec[is_rel].mean()
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, self.num_class).float().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
        return None


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1, weights=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if mask is not None:
            mask = mask.float().detach()
            anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1), 0)
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            if weights is None:
                loss = loss.mean()
            else:
                weights = weights.detach()
                loss = (loss * weights).mean()
        else:
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearBatchNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SupConResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (partial_loss,
     lambda: ([], {'confidence': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
]

class Test_hbzju_PiCO(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

