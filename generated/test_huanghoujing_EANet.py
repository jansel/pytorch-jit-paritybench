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


from collections import OrderedDict


import torch.nn as nn


from itertools import chain


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import namedtuple


from copy import deepcopy


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
        self.deconv = nn.ConvTranspose2d(in_channels=cfg.in_c, out_channels
            =cfg.mid_c, kernel_size=3, stride=2, padding=1, output_padding=
            1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.mid_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=cfg.mid_c, out_channels=cfg.
            num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(self.deconv(x))))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


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


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=cfg.
            last_conv_stride)
        self.out_c = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), nn.BatchNorm2d(planes * block.
                expansion))
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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_huanghoujing_EANet(_paritybench_base):
    pass

    def test_000(self):
        self._check(PartSegHead(*[], **{'cfg': _mock_config(in_c=4, mid_c=4, num_classes=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
