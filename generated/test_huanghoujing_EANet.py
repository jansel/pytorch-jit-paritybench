import sys
_module = sys.modules[__name__]
del sys
package = _module
config = _module
default = _module
data = _module
create_dataset = _module
dataloader = _module
dataset = _module
datasets = _module
coco = _module
cuhk03_np_detected_jpg = _module
cuhk03_np_detected_png = _module
duke = _module
market1501 = _module
msmt17 = _module
partial_ilids = _module
partial_reid = _module
kpt_to_pap_mask = _module
multitask_dataloader = _module
random_identity_sampler = _module
transform = _module
eval = _module
eval_dataloader = _module
eval_feat = _module
extract_feat = _module
metric = _module
np_distance = _module
torch_distance = _module
loss = _module
id_loss = _module
ps_loss = _module
triplet_loss = _module
model = _module
backbone = _module
base_model = _module
global_pool = _module
model = _module
pa_pool = _module
pcb_pool = _module
ps_head = _module
resnet = _module
optim = _module
cft_trainer = _module
eanet_trainer = _module
lr_scheduler = _module
optimizer = _module
reid_trainer = _module
trainer = _module
utils = _module
arg_parser = _module
cfg = _module
file = _module
image = _module
init_path = _module
log = _module
meter = _module
misc = _module
model = _module
rank_list = _module
torch_utils = _module
infer_dataloader_example = _module
remove_optim_lr_s_in_ckpt = _module
visualize_rank_list = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils.data import DataLoader as TorchDataLoader


from torch.utils.data import SequentialSampler


from torch.utils.data import RandomSampler


import numpy as np


from copy import deepcopy


from torch.utils.data import Dataset as TorchDataset


from collections import defaultdict


import torch


from torch.utils.data import Sampler


import torchvision.transforms.functional as F


import random


from sklearn.metrics import average_precision_score


from collections import OrderedDict


import torch.nn as nn


from itertools import chain


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import namedtuple


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import _LRScheduler


import torch.optim as optim


import time


from torch.nn.parallel import DataParallel


class BaseModel(nn.Module):

    def get_ft_and_new_params(self, *args, **kwargs):
        """Get finetune and new parameters, mostly for creating optimizer.
        Return two lists."""
        return [], list(self.parameters())

    def get_ft_and_new_modules(self, *args, **kwargs):
        """Get finetune and new modules, mostly for setting train/eval mode.
        Return two lists."""
        return [], list(self.modules())

    def set_train_mode(self, *args, **kwargs):
        """Set model to train mode for model training, some layers can be fixed and set to eval mode."""
        self.train()


class PartSegHead(nn.Module):

    def __init__(self, cfg):
        super(PartSegHead, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=cfg.in_c, out_channels=cfg.mid_c, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.mid_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=cfg.mid_c, out_channels=cfg.num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(self.deconv(x))))
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=cfg.last_conv_stride)
        self.out_c = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


ArchCfg = namedtuple('ArchCfg', ['block', 'layers'])


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


arch_dict = {'resnet18': ArchCfg(BasicBlock, [2, 2, 2, 2]), 'resnet34': ArchCfg(BasicBlock, [3, 4, 6, 3]), 'resnet50': ArchCfg(Bottleneck, [3, 4, 6, 3]), 'resnet101': ArchCfg(Bottleneck, [3, 4, 23, 3]), 'resnet152': ArchCfg(Bottleneck, [3, 8, 36, 3])}


def load_state_dict(model, src_state_dict, fold_bnt=True):
    """Copy parameters and buffers from `src_state_dict` into `model` and its
    descendants. The `src_state_dict.keys()` NEED NOT exactly match
    `model.state_dict().keys()`. For dict key mismatch, just
    skip it; for copying error, just output warnings and proceed.

    Arguments:
        model: A torch.nn.Module object.
        src_state_dict (dict): A dict containing parameters and persistent buffers.
    Note:
        This is modified from torch.nn.modules.module.load_state_dict(), to make
        the warnings and errors more detailed.
    """
    from torch.nn import Parameter
    dest_state_dict = model.state_dict()
    for name, param in list(src_state_dict.items()):
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception as msg:
            None

    def _fold_nbt(keys):
        nbt_keys = [s for s in keys if s.endswith('.num_batches_tracked')]
        if len(nbt_keys) > 0:
            keys = [s for s in keys if not s.endswith('.num_batches_tracked')] + ['num_batches_tracked  x{}'.format(len(nbt_keys))]
        return keys
    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        None
        if fold_bnt:
            src_missing = _fold_nbt(src_missing)
        for n in src_missing:
            None
    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        None
        if fold_bnt:
            dest_missing = _fold_nbt(dest_missing)
        for n in dest_missing:
            None


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def get_resnet(cfg):
    model = ResNet(arch_dict[cfg.name].block, arch_dict[cfg.name].layers, cfg)
    if cfg.pretrained:
        state_dict = model_zoo.load_url(model_urls[cfg.name], model_dir=cfg.pretrained_model_dir)
        load_state_dict(model, state_dict)
        None
    return model


backbone_factory = {'resnet18': get_resnet, 'resnet34': get_resnet, 'resnet50': get_resnet, 'resnet101': get_resnet, 'resnet152': get_resnet}


def create_backbone(cfg):
    return backbone_factory[cfg.name](cfg)


def create_embedding(in_dim=None, out_dim=None):
    layers = [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Model(BaseModel):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone)
        self.pool = eval('{}(cfg)'.format(cfg.pool_type))
        self.create_em_list()
        if hasattr(cfg, 'num_classes') and cfg.num_classes > 0:
            self.create_cls_list()
        if cfg.use_ps:
            cfg.ps_head.in_c = self.backbone.out_c
            self.ps_head = PartSegHead(cfg.ps_head)
        None

    def create_em_list(self):
        cfg = self.cfg
        self.em_list = nn.ModuleList([create_embedding(self.backbone.out_c, cfg.em_dim) for _ in range(cfg.num_parts)])

    def create_cls_list(self):
        cfg = self.cfg
        self.cls_list = nn.ModuleList([nn.Linear(cfg.em_dim, cfg.num_classes) for _ in range(cfg.num_parts)])
        ori_w = self.cls_list[0].weight.view(-1).detach().numpy().copy()
        self.cls_list.apply(init_classifier)
        new_w = self.cls_list[0].weight.view(-1).detach().numpy().copy()
        import numpy as np
        if np.array_equal(ori_w, new_w):
            None
            None
            None

    def get_ft_and_new_params(self, cft=False):
        """cft: Clustering and Fine Tuning"""
        ft_modules, new_modules = self.get_ft_and_new_modules(cft=cft)
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))
        return ft_params, new_params

    def get_ft_and_new_modules(self, cft=False):
        if cft:
            ft_modules = [self.backbone, self.em_list]
            if hasattr(self, 'ps_head'):
                ft_modules += [self.ps_head]
            new_modules = [self.cls_list] if hasattr(self, 'cls_list') else []
        else:
            ft_modules = [self.backbone]
            new_modules = [self.em_list]
            if hasattr(self, 'cls_list'):
                new_modules += [self.cls_list]
            if hasattr(self, 'ps_head'):
                new_modules += [self.ps_head]
        return ft_modules, new_modules

    def set_train_mode(self, cft=False, fix_ft_layers=False):
        self.train()
        if fix_ft_layers:
            for m in self.get_ft_and_new_modules(cft=cft)[0]:
                m.eval()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im'])

    def reid_forward(self, in_dict):
        pool_out_dict = self.pool(in_dict)
        feat_list = [em(f) for em, f in zip(self.em_list, pool_out_dict['feat_list'])]
        out_dict = {'feat_list': feat_list}
        if hasattr(self, 'cls_list'):
            logits_list = [cls(f) for cls, f in zip(self.cls_list, feat_list)]
            out_dict['logits_list'] = logits_list
        if 'visible' in pool_out_dict:
            out_dict['visible'] = pool_out_dict['visible']
        return out_dict

    def ps_forward(self, in_dict):
        return self.ps_head(in_dict['feat'])

    def forward(self, in_dict, forward_type='reid'):
        in_dict['feat'] = self.backbone_forward(in_dict)
        if forward_type == 'reid':
            out_dict = self.reid_forward(in_dict)
        elif forward_type == 'ps':
            out_dict = {'ps_pred': self.ps_forward(in_dict)}
        elif forward_type == 'ps_reid_parallel':
            out_dict = self.reid_forward(in_dict)
            out_dict['ps_pred'] = self.ps_forward(in_dict)
        elif forward_type == 'ps_reid_serial':
            ps_pred = self.ps_forward(in_dict)
            in_dict['pap_mask'] = gen_pap_mask_from_ps_pred(ps_pred)
            out_dict = self.reid_forward(in_dict)
            out_dict['ps_pred'] = ps_pred
        else:
            raise ValueError('Error forward_type {}'.format(forward_type))
        return out_dict


class TransparentDataParallel(DataParallel):

    def __getattr__(self, name):
        """Forward attribute access to its wrapped module."""
        try:
            return super(TransparentDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        """We only save/load state_dict of the wrapped model. This allows loading
        state_dict of a DataParallelSD model into a non-DataParallel model."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PartSegHead,
     lambda: ([], {'cfg': _mock_config(in_c=4, mid_c=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransparentDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_huanghoujing_EANet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

