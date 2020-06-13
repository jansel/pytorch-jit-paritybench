import sys
_module = sys.modules[__name__]
del sys
segmentron = _module
config = _module
settings = _module
data = _module
dataloader = _module
ade = _module
cityscapes = _module
lip_parsing = _module
mscoco = _module
pascal_aug = _module
pascal_voc = _module
sbu_shadow = _module
seg_data_base = _module
utils = _module
downloader = _module
ade20k = _module
models = _module
backbones = _module
build = _module
eespnet = _module
hrnet = _module
mobilenet = _module
resnet = _module
xception = _module
bisenet = _module
ccnet = _module
cgnet = _module
contextnet = _module
dabnet = _module
danet = _module
deeplabv3_plus = _module
denseaspp = _module
dfanet = _module
dunet = _module
edanet = _module
encnet = _module
enet = _module
espnetv2 = _module
fast_scnn = _module
fcn = _module
fpenet = _module
hardnet = _module
hrnet_seg = _module
icnet = _module
lednet = _module
model_zoo = _module
ocnet = _module
pointrend = _module
pspnet = _module
refinenet = _module
segbase = _module
unet = _module
modules = _module
basic = _module
batch_norm = _module
cc_attention = _module
module = _module
syncbn = _module
solver = _module
loss = _module
lovasz_losses = _module
lr_scheduler = _module
optimizer = _module
default_setup = _module
distributed = _module
download = _module
env = _module
filesystem = _module
logger = _module
options = _module
parallel = _module
registry = _module
score = _module
visualize = _module
setup = _module
demo = _module
eval = _module
train = _module

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


import logging


import torch.utils.model_zoo as model_zoo


import math


import torch.nn as nn


import torch.nn.functional as F


import torch._utils


import numpy as np


from collections import OrderedDict


import torch.distributed as dist


from torch import nn


from torch.autograd.function import Function


from torch.autograd.function import once_differentiable


import warnings


from torch.nn.modules.batchnorm import _BatchNorm


from torch.autograd import Variable


from torch import optim


import torch.utils.data as data


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import BatchSampler


import torch.cuda.comm as comm


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel._functions import Broadcast


from torch.autograd import Function


import copy


class DownSampler(nn.Module):

    def __init__(self, in_channels, out_channels, k=4, r_lim=9, reinf=True,
        inp_reinf=3, norm_layer=None):
        super(DownSampler, self).__init__()
        channels_diff = out_channels - in_channels
        self.eesp = EESP(in_channels, channels_diff, stride=2, k=k, r_lim=
            r_lim, down_method='avg', norm_layer=norm_layer)
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(_ConvBNPReLU(inp_reinf,
                inp_reinf, 3, 1, 1), _ConvBN(inp_reinf, out_channels, 1, 1))
        self.act = nn.PReLU(out_channels)

    def forward(self, x, x2=None):
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        output = torch.cat([avg_out, eesp_out], 1)
        if x2 is not None:
            w1 = avg_out.size(2)
            while True:
                x2 = F.avg_pool2d(x2, kernel_size=3, padding=1, stride=2)
                w2 = x2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(x2)
        return self.act(output)


class EESPNet(nn.Module):

    def __init__(self, num_classes=1000, scale=1, reinf=True, norm_layer=nn
        .BatchNorm2d):
        super(EESPNet, self).__init__()
        inp_reinf = 3 if reinf else None
        reps = [0, 3, 7, 3]
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base, levels, base_s = 32, 5, 0
        out_channels = [base] * levels
        for i in range(levels):
            if i == 0:
                base_s = int(base * scale)
                base_s = math.ceil(base_s / K[0]) * K[0]
                out_channels[i] = base if base_s > base else base_s
            else:
                out_channels[i] = base_s * pow(2, i)
        if scale <= 1.5:
            out_channels.append(1024)
        elif scale in [1.5, 2]:
            out_channels.append(1280)
        else:
            raise ValueError('Unknown scale value.')
        self.level1 = _ConvBNPReLU(3, out_channels[0], 3, 2, 1, norm_layer=
            norm_layer)
        self.level2_0 = DownSampler(out_channels[0], out_channels[1], k=K[0
            ], r_lim=r_lim[0], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level3_0 = DownSampler(out_channels[1], out_channels[2], k=K[1
            ], r_lim=r_lim[1], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(out_channels[2], out_channels[2], k=K[2
                ], r_lim=r_lim[2], norm_layer=norm_layer))
        self.level4_0 = DownSampler(out_channels[2], out_channels[3], k=K[2
            ], r_lim=r_lim[2], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(out_channels[3], out_channels[3], k=K[3
                ], r_lim=r_lim[3], norm_layer=norm_layer))
        self.level5_0 = DownSampler(out_channels[3], out_channels[4], k=K[3
            ], r_lim=r_lim[3], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level5 = nn.ModuleList()
        for i in range(reps[2]):
            self.level5.append(EESP(out_channels[4], out_channels[4], k=K[4
                ], r_lim=r_lim[4], norm_layer=norm_layer))
        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[4], 3,
            1, 1, groups=out_channels[4], norm_layer=norm_layer))
        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[5], 1,
            1, 0, groups=K[4], norm_layer=norm_layer))
        self.fc = nn.Linear(out_channels[5], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, seg=True):
        out_l1 = self.level1(x)
        out_l2 = self.level2_0(out_l1, x)
        out_l3_0 = self.level3_0(out_l2, x)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, x)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        if not seg:
            out_l5_0 = self.level5_0(out_l4)
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)
            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = F.dropout(output_g, p=0.2, training=self.training)
            output_1x1 = output_g.view(output_g.size(0), -1)
            return self.fc(output_1x1)
        return out_l1, out_l2, out_l3, out_l4


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=
    1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, groups=groups, bias=bias)


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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
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


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks,
            num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                num_channels[branch_index] * block.expansion))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels
            [branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index
            ] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(
                        num_inchannels[j], num_inchannels[i], 1, 1, 0, bias
                        =False), nn.BatchNorm2d(num_inchannels[i]), nn.
                        Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), nn.BatchNorm2d(
                                num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), nn.BatchNorm2d(
                                num_outchannels_conv3x3), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


_global_config['MODEL'] = 4


class HighResolutionNet(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.stage1_cfg = cfg.MODEL.HRNET.STAGE1
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks,
            norm_layer=norm_layer)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = cfg.MODEL.HRNET.STAGE2
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
            num_channels, norm_layer=norm_layer)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
            num_channels)
        self.stage3_cfg = cfg.MODEL.HRNET.STAGE3
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
            num_channels)
        self.stage4_cfg = cfg.MODEL.HRNET.STAGE4
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg,
            num_channels, multi_scale_output=True)
        self.last_inp_channels = np.int(np.sum(pre_stage_channels))

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels,
                head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=
                in_channels, out_channels=out_channels, kernel_size=3,
                stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU
                (inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(in_channels=head_channels[3] *
            head_block.expansion, out_channels=2048, kernel_size=1, stride=
            1, padding=0), nn.BatchNorm2d(2048), nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer, norm_layer=nn.BatchNorm2d):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(
                        num_channels_pre_layer[i], num_channels_cur_layer[i
                        ], 3, 1, 1, bias=False), norm_layer(
                        num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i
                        ] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels,
                        outchannels, 3, 2, 1, bias=False), norm_layer(
                        outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1,
        norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.
                expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True
        ):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block,
                num_blocks, num_inchannels, num_channels, fuse_method,
                reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return tuple(y_list)

    def init_weights(self, pretrained=''):
        logging.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logging.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logging.info('=> loading {} pretrained model {}'.format(k,
                    pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(MobileNet, self).__init__()
        multiplier = cfg.MODEL.BACKBONE_SCALE
        conv_dw_setting = [[64, 1, 1], [128, 2, 2], [256, 2, 2], [512, 6, 2
            ], [1024, 2, 2]]
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, norm_layer=
            norm_layer)]
        for c, n, s in conv_dw_setting:
            out_channels = int(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_DepthwiseConv(input_channels, out_channels,
                    stride, norm_layer))
                input_channels = out_channels
        self.last_inp_channels = int(1024 * multiplier)
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(int(1024 * multiplier), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), x.size(1)))
        return x


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        self.multiplier = cfg.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
        elif output_stride == 16:
            dilations = [1, 2]
        elif output_stride == 8:
            dilations = [2, 4]
        else:
            raise NotImplementedError
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 
            3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]
            ]
        input_channels = int(32 * self.multiplier
            ) if self.multiplier > 1.0 else 32
        self.conv1 = _ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True,
            norm_layer=norm_layer)
        self.planes = input_channels
        self.block1 = self._make_layer(InvertedResidual, self.planes,
            inverted_residual_setting[0:1], norm_layer=norm_layer)
        self.block2 = self._make_layer(InvertedResidual, self.planes,
            inverted_residual_setting[1:2], norm_layer=norm_layer)
        self.block3 = self._make_layer(InvertedResidual, self.planes,
            inverted_residual_setting[2:3], norm_layer=norm_layer)
        self.block4 = self._make_layer(InvertedResidual, self.planes,
            inverted_residual_setting[3:5], dilations[0], norm_layer=norm_layer
            )
        self.block5 = self._make_layer(InvertedResidual, self.planes,
            inverted_residual_setting[5:], dilations[1], norm_layer=norm_layer)
        self.last_inp_channels = self.planes
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, inverted_residual_setting,
        dilation=1, norm_layer=nn.BatchNorm2d):
        features = list()
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            stride = s if dilation == 1 else 1
            features.append(block(planes, out_channels, stride, t, dilation,
                norm_layer))
            planes = out_channels
            for i in range(n - 1):
                features.append(block(planes, out_channels, 1, t,
                    norm_layer=norm_layer))
                planes = out_channels
        self.planes = planes
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        c1 = self.block2(x)
        c2 = self.block3(c1)
        c3 = self.block4(c2)
        c4 = self.block5(c3)
        return c1, c2, c3, c4


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation,
            dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation,
            dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetV1(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
        zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        scale = cfg.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError
        self.inplanes = int((128 if deep_stem else 64) * scale)
        super(ResNetV1, self).__init__()
        if deep_stem:
            mid_channel = int(64 * scale)
            self.conv1 = nn.Sequential(nn.Conv2d(3, mid_channel, 3, 2, 1,
                bias=False), norm_layer(mid_channel), nn.ReLU(True), nn.
                Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
                norm_layer(mid_channel), nn.ReLU(True), nn.Conv2d(
                mid_channel, self.inplanes, 3, 1, 1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0],
            norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1],
            stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, int(256 * scale), layers[2],
            stride=strides[0], dilation=dilations[0], norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3],
            stride=strides[1], dilation=dilations[1], norm_layer=norm_layer,
            multi_grid=cfg.MODEL.DANET.MULTI_GRID, multi_dilation=cfg.MODEL
            .DANET.MULTI_DILATION)
        self.last_inp_channels = int(512 * block.expansion * scale)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * scale), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=nn.BatchNorm2d, multi_grid=False, multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, 1, stride, bias=False), norm_layer(planes *
                block.expansion))
        layers = []
        if not multi_grid:
            if dilation in (1, 2):
                layers.append(block(self.inplanes, planes, stride, dilation
                    =1, downsample=downsample, previous_dilation=dilation,
                    norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation
                    =2, downsample=downsample, previous_dilation=dilation,
                    norm_layer=norm_layer))
            else:
                raise RuntimeError('=> unknown dilation size: {}'.format(
                    dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=
                multi_dilation[0], downsample=downsample, previous_dilation
                =dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=
                    multi_dilation[i % div], previous_dilation=dilation,
                    norm_layer=norm_layer))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=
                    dilation, previous_dilation=dilation, norm_layer=
                    norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class XceptionBlock(nn.Module):

    def __init__(self, channel_list, stride=1, dilation=1,
        skip_connection_type='conv', relu_first=True, low_feat=False,
        norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat
        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(channel_list[0], channel_list[-1], 1,
                stride=stride, bias=False)
            self.bn = norm_layer(channel_list[-1])
        self.sep_conv1 = SeparableConv2d(channel_list[0], channel_list[1],
            dilation=dilation, relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv2 = SeparableConv2d(channel_list[1], channel_list[2],
            dilation=dilation, relu_first=relu_first, norm_layer=norm_layer)
        self.sep_conv3 = SeparableConv2d(channel_list[2], channel_list[3],
            dilation=dilation, relu_first=relu_first, stride=stride,
            norm_layer=norm_layer)
        self.last_inp_channels = channel_list[3]

    def forward(self, inputs):
        sc1 = self.sep_conv1(inputs)
        sc2 = self.sep_conv2(sc1)
        residual = self.sep_conv3(sc2)
        if self.skip_connection_type == 'conv':
            shortcut = self.conv(inputs)
            shortcut = self.bn(shortcut)
            outputs = residual + shortcut
        elif self.skip_connection_type == 'sum':
            outputs = residual + inputs
        elif self.skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')
        if self.low_feat:
            return outputs, sc2
        else:
            return outputs


class Xception65(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 1
            exit_block_stride = 2
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
            exit_block_stride = 1
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
            exit_block_stride = 1
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = XceptionBlock([64, 128, 128, 128], stride=2,
            norm_layer=norm_layer)
        self.block2 = XceptionBlock([128, 256, 256, 256], stride=2,
            low_feat=True, norm_layer=norm_layer)
        self.block3 = XceptionBlock([256, 728, 728, 728], stride=
            entry_block3_stride, low_feat=True, norm_layer=norm_layer)
        self.block4 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block5 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block6 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block7 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block8 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block9 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block10 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block11 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block12 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block13 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block14 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block15 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block16 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block17 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block18 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block19 = XceptionBlock([728, 728, 728, 728], dilation=
            middle_block_dilation, skip_connection_type='sum', norm_layer=
            norm_layer)
        self.block20 = XceptionBlock([728, 728, 1024, 1024], stride=
            exit_block_stride, dilation=exit_block_dilations[0], norm_layer
            =norm_layer)
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048], dilation=
            exit_block_dilations[1], skip_connection_type='none',
            relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x, c1 = self.block2(x)
        x, c2 = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        c3 = self.block19(x)
        x = self.block20(c3)
        c4 = self.block21(x)
        return c1, c2, c3, c4


class BlockA(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
        norm_layer=None, start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride,
                bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU()
        rep = list()
        inter_channels = out_channels // 4
        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1,
            dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1,
            dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3,
                stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1,
                norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Enc(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, norm_layer=nn.
        BatchNorm2d):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=
            norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=
                norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(nn.Conv2d(1000, in_channels, 1, bias=
            False), norm_layer(in_channels), nn.ReLU(True))

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class XceptionA(nn.Module):

    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            norm_layer(8), nn.ReLU(True))
        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)
        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class _BiSeHead(nn.Module):

    def __init__(self, in_channels, inter_channels, nclass, norm_layer=nn.
        BatchNorm2d):
        super(_BiSeHead, self).__init__()
        self.block = nn.Sequential(_ConvBNReLU(in_channels, inter_channels,
            3, 1, 1, norm_layer=norm_layer), nn.Dropout(0.1), nn.Conv2d(
            inter_channels, nclass, 1))

    def forward(self, x):
        x = self.block(x)
        return x


class SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inter_channels = 64
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3,
            norm_layer=norm_layer)
        self.conv3x3_1 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 
            1, norm_layer=norm_layer)
        self.conv3x3_2 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 
            1, norm_layer=norm_layer)
        self.conv1x1 = _ConvBNReLU(inter_channels, out_channels, 1, 1, 0,
            norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        return x


class _GlobalAvgPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(_GlobalAvgPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class AttentionRefinmentModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(AttentionRefinmentModule, self).__init__()
        self.conv3x3 = _ConvBNReLU(in_channels, out_channels, 3, 1, 1,
            norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels, 1, 1, 0, norm_layer=
            norm_layer), nn.Sigmoid())

    def forward(self, x):
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


class ContextPath(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ContextPath, self).__init__()
        inter_channels = 128
        self.global_context = _GlobalAvgPooling(512, inter_channels, norm_layer
            )
        self.arms = nn.ModuleList([AttentionRefinmentModule(512,
            inter_channels, norm_layer), AttentionRefinmentModule(256,
            inter_channels, norm_layer)])
        self.refines = nn.ModuleList([_ConvBNReLU(inter_channels,
            inter_channels, 3, 1, 1, norm_layer=norm_layer), _ConvBNReLU(
            inter_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer)])

    def forward(self, c1, c2, c3, c4):
        context_blocks = [c4, c3, c2, c1]
        global_context = self.global_context(c4)
        last_feature = global_context
        context_outputs = []
        for i, (feature, arm, refine) in enumerate(zip(context_blocks[:2],
            self.arms, self.refines)):
            feature = arm(feature)
            feature += last_feature
            last_feature = F.interpolate(feature, size=context_blocks[i + 1
                ].size()[2:], mode='bilinear', align_corners=True)
            last_feature = refine(last_feature)
            context_outputs.append(last_feature)
        return context_outputs


class FeatureFusion(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=
        nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = _ConvBNReLU(in_channels, out_channels, 1, 1, 0,
            norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels // reduction, 1, 1, 0,
            norm_layer=norm_layer), _ConvBNReLU(out_channels // reduction,
            out_channels, 1, 1, 0, norm_layer=norm_layer), nn.Sigmoid())

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


class _CCHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer)
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(_RCCAModule, self).__init__()
        self.recurrence = cfg.MODEL.CCNET.RECURRENCE
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            True))
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels +
            inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels), nn.Dropout2d(0.1))

    def forward(self, x):
        out = self.conva(x)
        for i in range(self.recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)
        return out


class _ChannelWiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1):
        super(_ChannelWiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, dilation,
            dilation, groups=in_channels, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class _FGlo(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(_FGlo, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels //
            reduction), nn.ReLU(True), nn.Linear(in_channels // reduction,
            in_channels), nn.Sigmoid())

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.gap(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out


class _InputInjection(nn.Module):

    def __init__(self, ratio):
        super(_InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, 2, 1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class _ConcatInjection(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConcatInjection, self).__init__()
        self.bn = norm_layer(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ContextGuidedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=2, reduction=16,
        down=False, residual=True, norm_layer=nn.BatchNorm2d):
        super(ContextGuidedBlock, self).__init__()
        inter_channels = out_channels // 2 if not down else out_channels
        if down:
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 3, 2, 1,
                norm_layer=norm_layer)
            self.reduce = nn.Conv2d(inter_channels * 2, out_channels, 1,
                bias=False)
        else:
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 1, 1, 0,
                norm_layer=norm_layer)
        self.f_loc = _ChannelWiseConv(inter_channels, inter_channels)
        self.f_sur = _ChannelWiseConv(inter_channels, inter_channels, dilation)
        self.bn = norm_layer(inter_channels * 2)
        self.prelu = nn.PReLU(inter_channels * 2)
        self.f_glo = _FGlo(out_channels, reduction)
        self.down = down
        self.residual = residual

    def forward(self, x):
        out = self.conv(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)
        joi_feat = torch.cat([loc, sur], dim=1)
        joi_feat = self.prelu(self.bn(joi_feat))
        if self.down:
            joi_feat = self.reduce(joi_feat)
        out = self.f_glo(joi_feat)
        if self.residual:
            out = out + x
        return out


class Custom_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=0, **kwargs):
        super(Custom_Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class DepthSepConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthSepConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, dw_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            dw_channels), nn.ReLU(True), nn.Conv2d(dw_channels,
            out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.
            ReLU(True))

    def forward(self, x):
        return self.conv(x)


class DepthConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, out_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(Custom_Conv(in_channels, in_channels * t,
            1), DepthConv(in_channels * t, in_channels * t, stride), nn.
            Conv2d(in_channels * t, out_channels, 1, bias=False), nn.
            BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class Shallow_net(nn.Module):

    def __init__(self, dw_channels1=32, dw_channels2=64, out_channels=128,
        **kwargs):
        super(Shallow_net, self).__init__()
        self.conv = Custom_Conv(3, dw_channels1, 3, 2)
        self.dsconv1 = DepthSepConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DepthSepConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DepthSepConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        return x


class Deep_net(nn.Module):

    def __init__(self, in_channels, block_channels, t, num_blocks, **kwargs):
        super(Deep_net, self).__init__()
        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks
        self.conv_ = Custom_Conv(3, in_channels, 3, 2)
        self.bottleneck1 = self._layer(LinearBottleneck, in_channels,
            block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._layer(LinearBottleneck, block_channels[0],
            block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._layer(LinearBottleneck, block_channels[1],
            block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._layer(LinearBottleneck, block_channels[2],
            block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._layer(LinearBottleneck, block_channels[3],
            block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._layer(LinearBottleneck, block_channels[4],
            block_channels[5], num_blocks[5], t[5], 1)

    def _layer(self, block, in_channels, out_channels, blocks, t, stride):
        layers = []
        layers.append(block(in_channels, out_channels, t, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x


class FeatureFusionModule(nn.Module):

    def __init__(self, highter_in_channels, lower_in_channels, out_channels,
        scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DepthConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(highter_in_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h, w),
            mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = DepthSepConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DepthSepConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels,
            num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1),
        groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=0.001)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class DABModule(nn.Module):

    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1,
            0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0,
            1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(
            1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(
            0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        return output + input


class DownSamplingBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_prelu(output)
        return output


class InputInjection(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class DANetHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 
            3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 
            3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512,
            out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512,
            out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512,
            out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)
        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)
        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


class _DeepLabHead(nn.Module):

    def __init__(self, nclass, c1_channels=256, c4_channels=2048,
        norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.use_aspp = cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP
        self.use_decoder = cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=
                norm_layer)
            last_channels += 48
        self.block = nn.Sequential(SeparableConv2d(last_channels, 256, 3,
            norm_layer=norm_layer, relu_first=False), SeparableConv2d(256, 
            256, 3, norm_layer=norm_layer, relu_first=False), nn.Conv2d(256,
            nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)
        if self.use_decoder:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            return self.block(torch.cat([x, c1], dim=1))
        return self.block(x)


class _DenseASPPHead(nn.Module):

    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d,
        norm_kwargs=None):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64,
            norm_layer, norm_kwargs)
        self.block = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(in_channels +
            5 * 64, nclass, 1))

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):

    def __init__(self, in_channels, inter_channels, out_channels,
        atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None
        ):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3,
            dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **{} if norm_kwargs is
            None else norm_kwargs)),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.
                training)
        return features


class _DenseASPPBlock(nn.Module):

    def __init__(self, in_channels, inter_channels1, inter_channels2,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1,
            inter_channels2, 3, 0.1, norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1,
            inter_channels1, inter_channels2, 6, 0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2,
            inter_channels1, inter_channels2, 12, 0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3,
            inter_channels1, inter_channels2, 18, 0.1, norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4,
            inter_channels1, inter_channels2, 24, 0.1, norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)
        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)
        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)
        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        return x


class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, inter_channels=48, norm_layer=nn.BatchNorm2d):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(nn.Conv2d(512, inter_channels, 1, bias=
            False), norm_layer(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(1024, inter_channels, 1, bias=
            False), norm_layer(inter_channels), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        size = c4.size()[2:]
        c2 = self.conv2(F.interpolate(c2, size, mode='bilinear',
            align_corners=True))
        c3 = self.conv3(F.interpolate(c3, size, mode='bilinear',
            align_corners=True))
        fused_feature = torch.cat([c4, c3, c2], dim=1)
        return fused_feature


class _DUHead(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(_DUHead, self).__init__()
        self.fuse = FeatureFused(norm_layer=norm_layer)
        self.block = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1,
            bias=False), norm_layer(256), nn.ReLU(True), nn.Conv2d(256, 256,
            3, padding=1, bias=False), norm_layer(256), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        fused_feature = self.fuse(c2, c3, c4)
        out = self.block(fused_feature)
        return out


class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor *
            scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (
            self.scale_factor * self.scale_factor))
        x = x.permute(0, 3, 1, 2)
        return x


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        if self.ninput < self.noutput:
            self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3,
                stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2,
                padding=1)
        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)
        output = self.bn(output)
        return F.relu(output)


class EDABlock(nn.Module):

    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super(EDABlock, self).__init__()
        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)
        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)
        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated,
            0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1, 3), stride=1, padding=(0,
            dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x
        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)
        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        output = torch.cat([output, input], 1)
        return output


class _EncHead(nn.Module):

    def __init__(self, in_channels, nclass, se_loss=True, lateral=True,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, 512, 3, padding=1,
            bias=False), norm_layer(512, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True))
        if lateral:
            self.connect = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512,
                1, bias=False), norm_layer(512, **{} if norm_kwargs is None
                 else norm_kwargs), nn.ReLU(True)), nn.Sequential(nn.Conv2d
                (1024, 512, 1, bias=False), norm_layer(512, **{} if 
                norm_kwargs is None else norm_kwargs), nn.ReLU(True))])
            self.fusion = nn.Sequential(nn.Conv2d(3 * 512, 512, 3, padding=
                1, bias=False), norm_layer(512, **{} if norm_kwargs is None
                 else norm_kwargs), nn.ReLU(True))
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv2d(512,
            nclass, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class EncModule(nn.Module):

    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1,
            bias=False), norm_layer(in_channels, **{} if norm_kwargs is
            None else norm_kwargs), nn.ReLU(True), Encoding(D=in_channels,
            K=ncodes), nn.BatchNorm1d(ncodes), nn.ReLU(True), Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.
            Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class Encoding(nn.Module):

    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1.0 / (self.K * self.D) ** (1 / 2)
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        assert X.size(1) == self.D
        B, D = X.size(0), self.D
        if X.dim() == 3:
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x' + str(self.D
            ) + '=>' + str(self.K) + 'x' + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E


class Mean(nn.Module):

    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class InitialBlock(nn.Module):
    """ENet initial block"""

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, 2, 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = norm_layer(out_channels + 3)
        self.act = nn.PReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.maxpool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """Bottlenecks include regular, asymmetric, downsampling, dilated"""

    def __init__(self, in_channels, inter_channels, out_channels, dilation=
        1, asymmetric=False, downsampling=False, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(Bottleneck, self).__init__()
        self.downsamping = downsampling
        if downsampling:
            self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
            self.conv_down = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, 1, bias=False), norm_layer(out_channels))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1,
            bias=False), norm_layer(inter_channels), nn.PReLU())
        if downsampling:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, 2, stride=2, bias=False), norm_layer(
                inter_channels), nn.PReLU())
        elif asymmetric:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, (5, 1), padding=(2, 0), bias=False), nn.
                Conv2d(inter_channels, inter_channels, (1, 5), padding=(0, 
                2), bias=False), norm_layer(inter_channels), nn.PReLU())
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, 3, dilation=dilation, padding=dilation,
                bias=False), norm_layer(inter_channels), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 
            1, bias=False), norm_layer(out_channels), nn.Dropout2d(0.1))
        self.act = nn.PReLU()

    def forward(self, x):
        identity = x
        if self.downsamping:
            identity, max_indices = self.maxpool(identity)
            identity = self.conv_down(identity)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + identity)
        if self.downsamping:
            return out, max_indices
        else:
            return out


class UpsamplingBottleneck(nn.Module):
    """upsampling Block"""

    def __init__(self, in_channels, inter_channels, out_channels,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(UpsamplingBottleneck, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels))
        self.upsampling = nn.MaxUnpool2d(2)
        self.block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1,
            bias=False), norm_layer(inter_channels), nn.PReLU(), nn.
            ConvTranspose2d(inter_channels, inter_channels, 2, 2, bias=
            False), norm_layer(inter_channels), nn.PReLU(), nn.Conv2d(
            inter_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.Dropout2d(0.1))
        self.act = nn.PReLU()

    def forward(self, x, max_indices):
        out_up = self.conv(x)
        out_up = self.upsampling(out_up, max_indices)
        out_ext = self.block(x)
        out = self.act(out_up + out_ext)
        return out


class _PSPModule(nn.Module):

    def __init__(self, in_channels, out_channels=1024, sizes=(1, 2, 4, 8),
        **kwargs):
        super(_PSPModule, self).__init__()
        self.stages = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 3,
            1, 1, groups=in_channels, bias=False) for _ in sizes])
        self.project = _ConvBNPReLU(in_channels * (len(sizes) + 1),
            out_channels, 1, 1, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for stage in self.stages:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(stage(x), size, mode='bilinear',
                align_corners=True)
            feats.append(upsampled)
        return self.project(torch.cat(feats, dim=1))


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64,
        norm_layer=nn.BatchNorm2d):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = SeparableConv2d(dw_channels1, dw_channels2, stride=2,
            relu_first=False, norm_layer=norm_layer)
        self.dsconv2 = SeparableConv2d(dw_channels2, out_channels, stride=2,
            relu_first=False, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
        out_channels=128, t=6, num_blocks=(3, 3, 3), norm_layer=nn.BatchNorm2d
        ):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(InvertedResidual, in_channels,
            block_channels[0], num_blocks[0], t, 2, norm_layer=norm_layer)
        self.bottleneck2 = self._make_layer(InvertedResidual,
            block_channels[0], block_channels[1], num_blocks[1], t, 2,
            norm_layer=norm_layer)
        self.bottleneck3 = self._make_layer(InvertedResidual,
            block_channels[1], block_channels[2], num_blocks[2], t, 1,
            norm_layer=norm_layer)
        self.ppm = PyramidPooling(block_channels[2], norm_layer=norm_layer)
        self.out = _ConvBNReLU(block_channels[2] * 2, out_channels, 1,
            norm_layer=norm_layer)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1,
        norm_layer=nn.BatchNorm2d):
        layers = []
        layers.append(block(inplanes, planes, stride, t, norm_layer=norm_layer)
            )
        for i in range(1, blocks):
            layers.append(block(planes, planes, 1, t, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels,
        scale_factor=4, norm_layer=nn.BatchNorm2d):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _ConvBNReLU(lower_in_channels, out_channels, 1,
            norm_layer=norm_layer)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels,
            out_channels, 1), norm_layer(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(highter_in_channels,
            out_channels, 1), norm_layer(out_channels))
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4,
            mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, norm_layer=nn.
        BatchNorm2d):
        super(Classifer, self).__init__()
        self.dsconv1 = SeparableConv2d(dw_channels, dw_channels, stride=
            stride, relu_first=False, norm_layer=norm_layer)
        self.dsconv2 = SeparableConv2d(dw_channels, dw_channels, stride=
            stride, relu_first=False, norm_layer=norm_layer)
        self.conv = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(dw_channels,
            num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=bias)


class FPEBlock(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=
        1, t=1, scales=4, se=False, norm_layer=None):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, 
            bottleneck_planes // scales, groups=bottleneck_planes // scales,
            dilation=dilat[i], padding=1 * dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for
            _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1])))
                    )
        out = torch.cat(ys, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class MEUModule(nn.Module):

    def __init__(self, channels_high, channels_low, channel_out):
        super(MEUModule, self).__init__()
        self.conv1x1_low = nn.Conv2d(channels_low, channel_out, kernel_size
            =1, bias=False)
        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv1x1_high = nn.Conv2d(channels_high, channel_out,
            kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(channel_out, channel_out, kernel_size=1,
            bias=False)
        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape
        fms_low = self.conv1x1_low(fms_low)
        fms_low = self.bn_low(fms_low)
        sa_avg_out = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1,
            keepdim=True)))
        fms_high = self.conv1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        ca_avg_out = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(
            fms_high))))
        fms_high_up = F.interpolate(fms_high, size=(h, w), mode='bilinear',
            align_corners=True)
        fms_sa_att = sa_avg_out * fms_high_up
        fms_ca_att = ca_avg_out * fms_low
        out = fms_ca_att + fms_sa_att
        return out


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
            )
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=
        False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels,
                growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(ConvLayer(inch, outch))
            if i % 2 == 0 or i == n_layers - 1:
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if i == 0 and self.keepBase or i == t - 1 or i % 2 == 1:
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip, concat=True):
        out = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode=
            'bilinear', align_corners=True)
        if concat:
            out = torch.cat([out, skip], 1)
        return out


class _HRNetHead(nn.Module):

    def __init__(self, nclass, last_inp_channels, norm_layer=nn.BatchNorm2d):
        super(_HRNetHead, self).__init__()
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=
            last_inp_channels, out_channels=last_inp_channels, kernel_size=
            1, stride=1, padding=0), norm_layer(last_inp_channels), nn.ReLU
            (inplace=False), nn.Conv2d(in_channels=last_inp_channels,
            out_channels=nclass, kernel_size=cfg.MODEL.HRNET.
            FINAL_CONV_KERNEL, stride=1, padding=1 if cfg.MODEL.HRNET.
            FINAL_CONV_KERNEL == 3 else 0))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear',
            align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear',
            align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear',
            align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x


class _ICHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_ICHead, self).__init__()
        scale = cfg.MODEL.BACKBONE_SCALE
        self.cff_12 = CascadeFeatureFusion(int(512 * scale), 64, 128,
            nclass, norm_layer)
        self.cff_24 = CascadeFeatureFusion(int(2048 * scale), int(512 *
            scale), 128, nclass, norm_layer)
        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4, size):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        outputs.append(x_12_cls)
        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear',
            align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, size, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        outputs.reverse()
        return outputs


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass,
        norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(nn.Conv2d(low_channels, out_channels,
            3, padding=2, dilation=2, bias=False), norm_layer(out_channels))
        self.conv_high = nn.Sequential(nn.Conv2d(high_channels,
            out_channels, 1, bias=False), norm_layer(out_channels))
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode=
            'bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)
        return x, x_low_cls


class Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2,
            bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2,
            bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool(x1)
        x2 = self.conv2(x)
        x2 = self.pool(x2)
        return torch.cat([x1, x2], dim=1)


class SSnbt(nn.Module):

    def __init__(self, in_channels, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SSnbt, self).__init__()
        inter_channels = in_channels // 2
        self.branch1 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, (3, 1), padding=(1, 0), bias=False), nn.ReLU(
            True), nn.Conv2d(inter_channels, inter_channels, (1, 3),
            padding=(0, 1), bias=False), norm_layer(inter_channels), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (3, 1),
            padding=(dilation, 0), dilation=(dilation, 1), bias=False), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (1, 3),
            padding=(0, dilation), dilation=(1, dilation), bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, (1, 3), padding=(0, 1), bias=False), nn.ReLU(
            True), nn.Conv2d(inter_channels, inter_channels, (3, 1),
            padding=(1, 0), bias=False), norm_layer(inter_channels), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (1, 3),
            padding=(0, dilation), dilation=(1, dilation), bias=False), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (3, 1),
            padding=(dilation, 0), dilation=(dilation, 1), bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.relu = nn.ReLU(True)

    @staticmethod
    def channel_shuffle(x, groups):
        n, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(n, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(n, -1, h, w)
        return x

    def forward(self, x):
        x1, x2 = x.split(x.size(1) // 2, 1)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.relu(out + x)
        out = self.channel_shuffle(out, groups=2)
        return out


class APNModule(nn.Module):

    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(APNModule, self).__init__()
        self.conv1 = _ConvBNReLU(in_channels, in_channels, 3, 2, 1,
            norm_layer=norm_layer)
        self.conv2 = _ConvBNReLU(in_channels, in_channels, 5, 2, 2,
            norm_layer=norm_layer)
        self.conv3 = _ConvBNReLU(in_channels, in_channels, 7, 2, 3,
            norm_layer=norm_layer)
        self.level1 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level2 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level3 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level4 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), _ConvBNReLU(
            in_channels, nclass, 1))

    def forward(self, x):
        w, h = x.size()[2:]
        branch3 = self.conv1(x)
        branch2 = self.conv2(branch3)
        branch1 = self.conv3(branch2)
        out = self.level1(branch1)
        out = F.interpolate(out, ((w + 3) // 4, (h + 3) // 4), mode=
            'bilinear', align_corners=True)
        out = self.level2(branch2) + out
        out = F.interpolate(out, ((w + 1) // 2, (h + 1) // 2), mode=
            'bilinear', align_corners=True)
        out = self.level3(branch3) + out
        out = F.interpolate(out, (w, h), mode='bilinear', align_corners=True)
        out = self.level4(x) * out
        out = self.level5(x) + out
        return out


class _OCHead(nn.Module):

    def __init__(self, nclass, oc_arch, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_OCHead, self).__init__()
        if oc_arch == 'base':
            self.context = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding
                =1, bias=False), norm_layer(512), nn.ReLU(True),
                BaseOCModule(512, 512, 256, 256, scales=[1], norm_layer=
                norm_layer, **kwargs))
        elif oc_arch == 'pyramid':
            self.context = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding
                =1, bias=False), norm_layer(512), nn.ReLU(True),
                PyramidOCModule(512, 512, 256, 512, scales=[1, 2, 3, 6],
                norm_layer=norm_layer, **kwargs))
        elif oc_arch == 'asp':
            self.context = ASPOCModule(2048, 512, 256, 512, norm_layer=
                norm_layer, **kwargs)
        else:
            raise ValueError('Unknown OC architecture!')
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.context(x)
        return self.out(x)


class BaseAttentionBlock(nn.Module):
    """The basic implementation for self-attention block/non-local block."""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BaseAttentionBlock, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(scale)
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels), nn.ReLU(True))
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1
            ).permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1
            ).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.bmm(query, key) * self.key_channels ** -0.5
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.bmm(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(w, h), mode='bilinear',
                align_corners=True)
        return context


class BaseOCModule(nn.Module):
    """Base-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scales=[1], norm_layer=nn.BatchNorm2d, concat=True,
        **kwargs):
        super(BaseOCModule, self).__init__()
        self.stages = nn.ModuleList([BaseAttentionBlock(in_channels,
            out_channels, key_channels, value_channels, scale, norm_layer,
            **kwargs) for scale in scales])
        in_channels = in_channels * 2 if concat else in_channels
        self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1
            ), norm_layer(out_channels), nn.ReLU(True), nn.Dropout2d(0.05))
        self.concat = concat

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        if self.concat:
            context = torch.cat([context, x], 1)
        out = self.project(context)
        return out


class PyramidAttentionBlock(nn.Module):
    """The basic implementation for pyramid self-attention block/non-local block"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidAttentionBlock, self).__init__()
        self.scale = scale
        self.value_channels = value_channels
        self.key_channels = key_channels
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels), nn.ReLU(True))
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        local_x = list()
        local_y = list()
        step_w, step_h = w // self.scale, h // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = step_w * i, step_h * j
                end_x, end_y = min(start_x + step_w, w), min(start_y +
                    step_h, h)
                if i == self.scale - 1:
                    end_x = w
                if j == self.scale - 1:
                    end_y = h
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        local_list = list()
        local_block_cnt = self.scale ** 2 * 2
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:
                local_y[i + 1]]
            w_local, h_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.
                value_channels, -1).permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.
                key_channels, -1).permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.
                key_channels, -1)
            sim_map = torch.bmm(query_local, key_local
                ) * self.key_channels ** -0.5
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.bmm(sim_map, value_local).permute(0, 2, 1
                ).contiguous()
            context_local = context_local.view(batch_size, self.
                value_channels, w_local, h_local)
            local_list.append(context_local)
        context_list = list()
        for i in range(0, self.scale):
            row_tmp = list()
            for j in range(self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = self.W(context)
        return context


class PyramidOCModule(nn.Module):
    """Pyramid-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scales=[1], norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidOCModule, self).__init__()
        self.stages = nn.ModuleList([PyramidAttentionBlock(in_channels,
            out_channels, key_channels, value_channels, scale, norm_layer,
            **kwargs) for scale in scales])
        self.up_dr = nn.Sequential(nn.Conv2d(in_channels, in_channels * len
            (scales), 1), norm_layer(in_channels * len(scales)), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(in_channels * len(scales) * 
            2, out_channels, 1), norm_layer(out_channels), nn.ReLU(True),
            nn.Dropout2d(0.05))

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = [self.up_dr(x)]
        for i in range(len(priors)):
            context += [priors[i]]
        context = torch.cat(context, 1)
        out = self.project(context)
        return out


class ASPOCModule(nn.Module):
    """ASP-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, atrous_rates=(12, 24, 36), norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(ASPOCModule, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=1), norm_layer(out_channels), nn.ReLU(True),
            BaseOCModule(out_channels, out_channels, key_channels,
            value_channels, [2], norm_layer, False, **kwargs))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate1, dilation=rate1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate2, dilation=rate2, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate3, dilation=rate3, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5,
            out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU
            (True), nn.Dropout2d(0.1))

    def forward(self, x):
        feat1 = self.context(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.project(out)
        return out


def point_sample(input, point_coords, **kwargs):
    """
    From Detectron2, point_features.py#19
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    """
    Follows 3.1. Point Selection for Inference and Training
    In Train:, `The sampling strategy selects N points on a feature map to train on.`
    In Inference, `then selects the N most uncertain points`
    Args:
        mask(Tensor): [B, C, H, W]
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2]
    """
    assert mask.dim() == 4, 'Dim must be N(Batch)CHW'
    device = mask.device
    B, _, H, W = mask.shape
    mask, _ = mask.sort(1, descending=True)
    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, (0)] - mask[:, (1)])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)
        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, (0)] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        points[:, :, (1)] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points
    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners
        =False)
    uncertainty_map = -1 * (over_generation_map[:, (0)] -
        over_generation_map[:, (1)])
    _, idx = uncertainty_map.topk(int(beta * N), -1)
    shift = k * N * torch.arange(B, dtype=torch.long, device=device)
    idx += shift[:, (None)]
    importance = over_generation.view(-1, 2)[(idx.view(-1)), :].view(B, int
        (beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)


class PointHead(nn.Module):

    def __init__(self, in_c=275, num_classes=19, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv1d(in_c, 256, kernel_size=1, stride
            =1, padding=0, bias=True), nn.ReLU(True), nn.Conv1d(256, 256,
            kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(True),
            nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0, bias=
            True), nn.ReLU(True), nn.Conv1d(256, num_classes, 1))
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)
        N = x.shape[-1] // 16
        points = sampling_points(out, N * N, self.k, self.beta)
        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)
        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation)
        return {'rend': rend, 'points': points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096
        while out.shape[-1] * 2 < x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode='bilinear',
                align_corners=False)
            points_idx, points = sampling_points(out, num_points, training=
                self.training)
            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(res2, points, align_corners=False)
            feature_representation = torch.cat([coarse, fine], dim=1)
            rend = self.mlp(feature_representation)
            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = out.reshape(B, C, -1).scatter_(2, points_idx, rend).view(B,
                C, H, W)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
            align_corners=False)
        points_idx, points = sampling_points(out, num_points, training=self
            .training)
        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)
        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation)
        B, C, H, W = out.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        out = out.reshape(B, C, -1).scatter_(2, points_idx, rend).view(B, C,
            H, W)
        return {'fine': out}


class _PSPHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=
            norm_kwargs)
        self.block = nn.Sequential(nn.Conv2d(4096, 512, 3, padding=1, bias=
            False), norm_layer(512, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True), nn.Dropout(0.1), nn.Conv2d(512,
            nclass, 1))

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


class _RefineHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_RefineHead, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.p_ims1d2_outl1_dimred = nn.Conv2d(2048, 512, 1, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = nn.Conv2d(512, 256, 1,
            bias=False)
        self.p_ims1d2_outl2_dimred = nn.Conv2d(1024, 256, 1, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = nn.Conv2d(256, 256, 1,
            bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = nn.Conv2d(256, 256, 1,
            bias=False)
        self.p_ims1d2_outl3_dimred = nn.Conv2d(512, 256, 1, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = nn.Conv2d(256, 256, 1,
            bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = nn.Conv2d(256, 256, 1,
            bias=False)
        self.p_ims1d2_outl4_dimred = nn.Conv2d(256, 256, 1, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = nn.Conv2d(256, 256, 1,
            bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.clf_conv = nn.Conv2d(256, nclass, kernel_size=3, stride=1,
            padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, l1, l2, l3, l4):
        l4 = self.do(l4)
        l3 = self.do(l3)
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = F.interpolate(x4, size=l3.size()[2:], mode='bilinear',
            align_corners=True)
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = F.interpolate(x3, size=l2.size()[2:], mode='bilinear',
            align_corners=True)
        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = F.interpolate(x2, size=l1.size()[2:], mode='bilinear',
            align_corners=True)
        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        out = self.clf_conv(x1)
        return out


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'), nn.Conv2d
                (in_planes if i == 0 else out_planes, out_planes, 1, stride
                =1, bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


def _flip_image(img):
    assert img.ndim == 4
    return img.flip(3)


def _pad_image(img, crop_size):
    b, c, h, w = img.shape
    assert c == 3
    padh = crop_size[0] - h if h < crop_size[0] else 0
    padw = crop_size[1] - w if w < crop_size[1] else 0
    if padh == 0 and padw == 0:
        return img
    img_pad = F.pad(img, (0, padh, 0, padw))
    return img_pad


model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c':
    'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth'
    , 'resnet101c':
    'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth'
    , 'resnet152c':
    'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth'
    , 'xception65':
    'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth'
    , 'hrnet_w18_small_v1':
    'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/hrnet-w18-small-v1-08f8ae64.pth'
    , 'mobilenet_v2':
    'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/mobilenetV2-15498621.pth'
    }


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    if overwrite or not os.path.exists(fname) or sha1_hash and not check_sha1(
        fname, sha1_hash):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError('Failed downloading url %s' % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=
                    int(total_length / 1024.0 + 0.5), unit='KB', unit_scale
                    =False, dynamic_ncols=True):
                    f.write(chunk)
        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. The repo may be outdated or download may be incomplete. If the "repo_url" is overridden, consider switching to the default repo.'
                .format(fname))
    return fname


_global_config['PHASE'] = 4


_global_config['TRAIN'] = 4


def load_backbone_pretrained(model, backbone):
    if (cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and not cfg.
        TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.
                BACKBONE_PRETRAINED_PATH), strict=False)
            logging.info(msg)
        elif backbone not in model_urls:
            logging.info('{} has no pretrained model'.format(backbone))
            return
        else:
            logging.info('load backbone pretrained model from url..')
            try:
                msg = model.load_state_dict(model_zoo.load_url(model_urls[
                    backbone]), strict=False)
            except Exception as e:
                logging.warning(e)
                logging.info('Use torch download failed, try custom method!')
                msg = model.load_state_dict(torch.load(download(model_urls[
                    backbone], path=os.path.join(torch.hub._get_torch_home(
                    ), 'checkpoints'))), strict=False)
            logging.info(msg)


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.

    To create a registry (inside segmentron):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".
                format(name, self._name))
        return ret

    def get_list(self):
        return list(self._obj_map.keys())


BACKBONE_REGISTRY = Registry('BACKBONE')


def get_segmentation_backbone(backbone, norm_layer=torch.nn.BatchNorm2d):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = BACKBONE_REGISTRY.get(backbone)(norm_layer)
    load_backbone_pretrained(model, backbone)
    return model


def groupNorm(num_channels, eps=1e-05, momentum=0.1, affine=True):
    return nn.GroupNorm(min(32, num_channels), num_channels, eps=eps,
        affine=affine)


def get_norm(norm):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    support_norm_type = ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
    assert norm in support_norm_type, 'Unknown norm type {}, support norm types are {}'.format(
        norm, support_norm_type)
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': nn.BatchNorm2d, 'SyncBN': NaiveSyncBatchNorm,
            'FrozenBN': FrozenBatchNorm2d, 'GN': groupNorm, 'nnSyncBN': nn.
            SyncBatchNorm}[norm]
    return norm


def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


def _to_tuple(size):
    if isinstance(size, (list, tuple)):
        assert len(size
            ), 'Expect eval crop size contains two element, but received {}'.format(
            len(size))
        return tuple(size)
    elif isinstance(size, numbers.Number):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport datatype: {}'.format(type(size)))


_global_config['AUG'] = 4


_global_config['ROOT_PATH'] = 4


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520,
        crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = os.path.join(cfg.ROOT_PATH, root)
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = self.to_tuple(crop_size)
        self.color_jitter = self._get_color_jitter()

    def to_tuple(self, size):
        if isinstance(size, (list, tuple)):
            return tuple(size)
        elif isinstance(size, (int, float)):
            return tuple((size, size))
        else:
            raise ValueError('Unsupport datatype: {}'.format(type(size)))

    def _get_color_jitter(self):
        color_jitter = cfg.AUG.COLOR_JITTER
        if color_jitter is None:
            return None
        if isinstance(color_jitter, (list, tuple)):
            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        return torchvision.transforms.ColorJitter(*color_jitter)

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = min(outsize)
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        w, h = img.size
        x1 = int(round((w - outsize[1]) / 2.0))
        y1 = int(round((h - outsize[0]) / 2.0))
        img = img.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
        mask = mask.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        if cfg.AUG.MIRROR and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        short_size = random.randint(int(self.base_size * 0.5), int(self.
            base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < min(crop_size):
            padh = crop_size[0] - oh if oh < crop_size[0] else 0
            padw = crop_size[1] - ow if ow < crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=-1)
        w, h = img.size
        x1 = random.randint(0, w - crop_size[1])
        y1 = random.randint(0, h - crop_size[0])
        img = img.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
        mask = mask.crop((x1, y1, x1 + crop_size[1], y1 + crop_size[0]))
        if cfg.AUG.BLUR_PROB > 0 and random.random() < cfg.AUG.BLUR_PROB:
            radius = (cfg.AUG.BLUR_RADIUS if cfg.AUG.BLUR_RADIUS > 0 else
                random.random())
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        if self.color_jitter:
            img = self.color_jitter(img)
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


class COCOSegmentation(SegmentationDataset):
    """COCO Semantic Segmentation Dataset for VOC Pre-training.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/coco'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64,
        20, 63, 7, 72]
    NUM_CLASS = 21

    def __init__(self, root='datasets/coco', split='train', mode=None,
        transform=None, **kwargs):
        super(COCOSegmentation, self).__init__(root, split, mode, transform,
            **kwargs)
        from pycocotools.coco import COCO
        from pycocotools import mask
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root,
                'annotations/instances_train2017.json')
            ids_file = os.path.join(root, 'annotations/train_ids.pkl')
            self.root = os.path.join(root, 'train2017')
        else:
            print('val set')
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')
            ids_file = os.path.join(root, 'annotations/val_ids.pkl')
            self.root = os.path.join(root, 'val2017')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._gen_seg_mask(cocotarget, img_metadata[
            'height'], img_metadata['width']))
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(path)

    def __len__(self):
        return len(self.ids)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * ((np.sum(m, axis=2) > 0) * c
                    ).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        print('Preprocessing mask, this will take a while.' +
            "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                img_metadata['width'])
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.
                format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorcycle', 'person', 'potted-plant', 'sheep',
            'sofa', 'train', 'tv')


def _get_sbu_pairs(folder, split='train'):

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            print(root)
            for filename in files:
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root, filename)
                    maskname = filename.replace('.jpg', '.png')
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath,
                            maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths),
            img_folder))
        return img_paths, mask_paths
    if split == 'train':
        img_folder = os.path.join(folder,
            'SBUTrain4KRecoveredSmall/ShadowImages')
        mask_folder = os.path.join(folder,
            'SBUTrain4KRecoveredSmall/ShadowMasks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        assert split in ('val', 'test')
        img_folder = os.path.join(folder, 'SBU-Test/ShadowImages')
        mask_folder = os.path.join(folder, 'SBU-Test/ShadowMasks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    return img_paths, mask_paths


class SBUSegmentation(SegmentationDataset):
    """SBU Shadow Segmentation Dataset
    """
    NUM_CLASS = 2

    def __init__(self, root='datasets/sbu', split='train', mode=None,
        transform=None, **kwargs):
        super(SBUSegmentation, self).__init__(root, split, mode, transform,
            **kwargs)
        assert os.path.exists(self.root)
        self.images, self.masks = _get_sbu_pairs(self.root, self.split)
        assert len(self.images) == len(self.masks)
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images in subfolders of:' + root + '\n'
                )

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target > 0] = 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit',
                        'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        logging.info('cannot find the mask or image:',
                            imgpath, maskpath)
        logging.info('Found {} images in the folder {}'.format(len(
            img_paths), img_folder))
        return img_paths, mask_paths
    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        logging.info('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder,
            train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder,
            val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


class CitySegmentation(SegmentationDataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/cityscapes'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, root='datasets/cityscapes', split='train', mode=None,
        transform=None, **kwargs):
        super(CitySegmentation, self).__init__(root, split, mode, transform,
            **kwargs)
        assert os.path.exists(self.root
            ), 'Please put dataset in {SEG_ROOT}/datasets/cityscapes'
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert len(self.images) == len(self.mask_paths)
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images in subfolders of:' + root + '\n'
                )
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1,
            2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert value in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')


class VOCSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOC2012'
    NUM_CLASS = 21

    def __init__(self, root='datasets/voc', split='train', mode=None,
        transform=None, **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, transform,
            **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + '.png')
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)
        if split != 'test':
            assert len(self.images) == len(self.masks)
        print('Found {} images in the folder {}'.format(len(self.images),
            _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorcycle', 'person', 'potted-plant', 'sheep',
            'sofa', 'train', 'tv')


class VOCAugSegmentation(SegmentationDataset):
    """Pascal VOC Augmented Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/voc'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCAugSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOCaug/dataset/'
    NUM_CLASS = 21

    def __init__(self, root='datasets/voc', split='train', mode=None,
        transform=None, **kwargs):
        super(VOCAugSegmentation, self).__init__(root, split, mode,
            transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'cls')
        _image_dir = os.path.join(_voc_root, 'img')
        if split == 'train':
            _split_f = os.path.join(_voc_root, 'trainval.txt')
        elif split == 'val':
            _split_f = os.path.join(_voc_root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + '.mat')
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)
        print('Found {} images in the folder {}'.format(len(self.images),
            _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self._load_mat(self.masks[index])
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        elif self.mode == 'testval':
            logging.warn('Use mode of testval, you should set batch size=1')
            img, target = self._img_transform(img), self._mask_transform(target
                )
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(
                self.mode))
        if self.transform is not None:
            img = self.transform(img)
        return img, target, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _load_mat(self, filename):
        mat = sio.loadmat(filename, mat_dtype=True, squeeze_me=True,
            struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorcycle', 'person', 'potted-plant', 'sheep',
            'sofa', 'train', 'tv')


def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith('.jpg'):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask:', maskpath)
    return img_paths, mask_paths


class ADE20KSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 150

    def __init__(self, root='datasets/ade', split='test', mode=None,
        transform=None, **kwargs):
        super(ADE20KSegmentation, self).__init__(root, split, mode,
            transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root
            ), 'Please put the data in {SEG_ROOT}/datasets/ade'
        self.images, self.masks = _get_ade20k_pairs(root, split)
        assert len(self.images) == len(self.masks)
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images in subfolders of:' + root + '\n'
                )
        logging.info('Found {} images in the folder {}'.format(len(self.
            images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ('wall', 'building, edifice', 'sky', 'floor, flooring',
            'tree', 'ceiling', 'road, route', 'bed', 'windowpane, window',
            'grass', 'cabinet', 'sidewalk, pavement',
            'person, individual, someone, somebody, mortal, soul',
            'earth, ground', 'door, double door', 'table',
            'mountain, mount', 'plant, flora, plant life',
            'curtain, drape, drapery, mantle, pall', 'chair',
            'car, auto, automobile, machine, motorcar', 'water',
            'painting, picture', 'sofa, couch, lounge', 'shelf', 'house',
            'sea', 'mirror', 'rug, carpet, carpeting', 'field', 'armchair',
            'seat', 'fence, fencing', 'desk', 'rock, stone',
            'wardrobe, closet, press', 'lamp',
            'bathtub, bathing tub, bath, tub', 'railing, rail', 'cushion',
            'base, pedestal, stand', 'box', 'column, pillar',
            'signboard, sign', 'chest of drawers, chest, bureau, dresser',
            'counter', 'sand', 'sink', 'skyscraper',
            'fireplace, hearth, open fireplace', 'refrigerator, icebox',
            'grandstand, covered stand', 'path', 'stairs, steps', 'runway',
            'case, display case, showcase, vitrine',
            'pool table, billiard table, snooker table', 'pillow',
            'screen door, screen', 'stairway, staircase', 'river',
            'bridge, span', 'bookcase', 'blind, screen',
            'coffee table, cocktail table',
            'toilet, can, commode, crapper, pot, potty, stool, throne',
            'flower', 'book', 'hill', 'bench', 'countertop',
            'stove, kitchen stove, range, kitchen range, cooking stove',
            'palm, palm tree', 'kitchen island',
            'computer, computing machine, computing device, data processor, electronic computer, information processing system'
            , 'swivel chair', 'boat', 'bar', 'arcade machine',
            'hovel, hut, hutch, shack, shanty',
            'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle'
            , 'towel', 'light, light source', 'truck, motortruck', 'tower',
            'chandelier, pendant, pendent', 'awning, sunshade, sunblind',
            'streetlight, street lamp', 'booth, cubicle, stall, kiosk',
            'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box'
            , 'airplane, aeroplane, plane', 'dirt track',
            'apparel, wearing apparel, dress, clothes', 'pole',
            'land, ground, soil',
            'bannister, banister, balustrade, balusters, handrail',
            'escalator, moving staircase, moving stairway',
            'ottoman, pouf, pouffe, puff, hassock', 'bottle',
            'buffet, counter, sideboard',
            'poster, posting, placard, notice, bill, card', 'stage', 'van',
            'ship', 'fountain',
            'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
            'canopy', 'washer, automatic washer, washing machine',
            'plaything, toy', 'swimming pool, swimming bath, natatorium',
            'stool', 'barrel, cask', 'basket, handbasket',
            'waterfall, falls', 'tent, collapsible shelter', 'bag',
            'minibike, motorbike', 'cradle', 'oven', 'ball',
            'food, solid food', 'step, stair', 'tank, storage tank',
            'trade name, brand name, brand, marque',
            'microwave, microwave oven', 'pot, flowerpot',
            'animal, animate being, beast, brute, creature, fauna',
            'bicycle, bike, wheel, cycle', 'lake',
            'dishwasher, dish washer, dishwashing machine',
            'screen, silver screen, projection screen', 'blanket, cover',
            'sculpture', 'hood, exhaust hood', 'sconce', 'vase',
            'traffic light, traffic signal, stoplight', 'tray',
            'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin'
            , 'fan', 'pier, wharf, wharfage, dock', 'crt screen', 'plate',
            'monitor, monitoring device', 'bulletin board, notice board',
            'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag')


datasets = {'ade20k': ADE20KSegmentation, 'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation, 'coco': COCOSegmentation, 'cityscape':
    CitySegmentation, 'sbu': SBUSegmentation}


_global_config['TEST'] = 4


_global_config['SOLVER'] = 4


_global_config['DATASET'] = 4


class SegBaseModel(nn.Module):
    """Base Model for Semantic Segmentation
    """

    def __init__(self, need_backbone=True):
        super(SegBaseModel, self).__init__()
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = cfg.SOLVER.AUX
        self.norm_layer = get_norm(cfg.MODEL.BN_TYPE)
        self.backbone = None
        self.encoder = None
        if need_backbone:
            self.get_backbone()

    def get_backbone(self):
        self.backbone = cfg.MODEL.BACKBONE.lower()
        self.encoder = get_segmentation_backbone(self.backbone, self.norm_layer
            )

    def base_forward(self, x):
        """forwarding backbone network"""
        c1, c2, c3, c4 = self.encoder(x)
        return c1, c2, c3, c4

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred

    def evaluate(self, image):
        """evaluating network with inputs and targets"""
        scales = cfg.TEST.SCALES
        flip = cfg.TEST.FLIP
        crop_size = _to_tuple(cfg.TEST.CROP_SIZE
            ) if cfg.TEST.CROP_SIZE else None
        batch, _, h, w = image.shape
        base_size = max(h, w)
        scores = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
            cur_img = _resize_image(image, height, width)
            if crop_size is not None:
                assert crop_size[0] >= h and crop_size[1] >= w
                crop_size_scaled = int(math.ceil(crop_size[0] * scale)), int(
                    math.ceil(crop_size[1] * scale))
                cur_img = _pad_image(cur_img, crop_size_scaled)
            outputs = self.forward(cur_img)[0][(...), :height, :width]
            if flip:
                outputs += _flip_image(self.forward(_flip_image(cur_img))[0])[(
                    ...), :height, :width]
            score = _resize_image(outputs, h, w)
            if scores is None:
                scores = score
            else:
                scores += score
        return scores


class _UNetHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_UNetHead, self).__init__()
        bilinear = True
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, nclass)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels,
            out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels,
            out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(
            in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, relu_first=True, bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=
            stride, padding=dilation, dilation=dilation, groups=inplanes,
            bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)
        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()), (
                'depthwise', depthwise), ('bn_depth', bn_depth), (
                'pointwise', pointwise), ('bn_point', bn_point)]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise
                ), ('bn_depth', bn_depth), ('relu1', nn.ReLU(inplace=True)),
                ('pointwise', pointwise), ('bn_point', bn_point), ('relu2',
                nn.ReLU(inplace=True))]))

    def forward(self, x):
        return self.block(x)


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d
        ):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ConvBNPReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d):
        super(_ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _BNPReLU(nn.Module):

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d):
        super(_BNPReLU, self).__init__()
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _DepthwiseConv(nn.Module):
    """conv_dw in MobileNet"""

    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(_ConvBNReLU(in_channels, in_channels, 3,
            stride, 1, groups=in_channels, norm_layer=norm_layer),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio,
        dilation=1, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6
                =True, norm_layer=norm_layer))
        layers.extend([_ConvBNReLU(inter_channels, inter_channels, 3,
            stride, dilation, dilation, groups=inter_channels, relu6=True,
            norm_layer=norm_layer), nn.Conv2d(inter_channels, out_channels,
            1, bias=False), norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """
    _version = 3

    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = torch.zeros_like(self
                    .running_mean)
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = torch.ones_like(self.
                    running_var)
        if version is not None and version < 3:
            logging.info('FrozenBatchNorm {} is upgraded to version 3.'.
                format(prefix.rstrip('.')))
            state_dict[prefix + 'running_var'] -= self.eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.
            num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = bn_module.BatchNorm2d, bn_module.SyncBatchNorm
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data + module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.
            get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0
            ] > 0, 'SyncBatchNorm does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])
        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean
            )
        self.running_var += self.momentum * (var.detach() - self.running_var)
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias


class _CAMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, g):
        out = _C.ca_map_forward(weight, g)
        ctx.save_for_backward(weight, g)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        dw, dg = _C.ca_map_backward(dout, weight, g)
        return dw, dg


ca_map = _CAMap.apply


class _CAWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t, f):
        weight = _C.ca_forward(t, f)
        ctx.save_for_backward(t, f)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt, df = _C.ca_backward(dw, t, f)
        return dt, df


ca_weight = _CAWeight.apply


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x
        return out


class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            inplace=True), nn.Dropout(0.1), nn.Conv2d(inter_channels,
            channels, 1))

    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Module):

    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError
        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            in_channels, out_channels, 1, bias=False)), ('bn', nn.
            BatchNorm2d(out_channels)), ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=
            dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=
            dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=
            dilations[2], relu_first=False)
        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.
            AdaptiveAvgPool2d((1, 1))), ('conv', nn.Conv2d(in_channels,
            out_channels, 1, bias=False)), ('bn', nn.BatchNorm2d(
            out_channels)), ('relu', nn.ReLU(inplace=True))]))
        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear',
            align_corners=True)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class PyramidPooling(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6), norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpools.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(_ConvBNReLU(in_channels, out_channels, 1,
                norm_layer=norm_layer, **kwargs))

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for avgpool, conv in zip(self.avgpools, self.convs):
            feats.append(F.interpolate(conv(avgpool(x)), size, mode=
                'bilinear', align_corners=True))
        return torch.cat(feats, dim=1)


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class EESP(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, k=4, r_lim=7,
        down_method='esp', norm_layer=nn.BatchNorm2d):
        super(EESP, self).__init__()
        self.stride = stride
        n = int(out_channels / k)
        n1 = out_channels - (k - 1) * n
        assert down_method in ['avg', 'esp'
            ], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(
            n, n1)
        self.proj_1x1 = _ConvBNPReLU(in_channels, n, 1, stride=1, groups=k,
            norm_layer=norm_layer)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13
            ): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            dilation = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(nn.Conv2d(n, n, 3, stride, dilation,
                dilation=dilation, groups=n, bias=False))
        self.conv_1x1_exp = _ConvBN(out_channels, out_channels, 1, 1,
            groups=k, norm_layer=norm_layer)
        self.br_after_cat = _BNPReLU(out_channels, norm_layer)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == x.size():
            expanded = expanded + x
        return self.module_act(expanded)


class SyncBatchNorm(_BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Parameters:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, sync=True,
        activation='none', slope=0.01, inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum
            =momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        self.slope = slope
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            extra = {'is_master': True, 'master_queue': self.master_queue,
                'worker_queues': self.worker_queues, 'worker_ids': self.
                worker_ids}
        else:
            extra = {'is_master': False, 'master_queue': self.master_queue,
                'worker_queue': self.worker_queues[self.worker_ids.index(x.
                get_device())]}
        if self.inplace:
            return inp_syncbatchnorm(x, self.weight, self.bias, self.
                running_mean, self.running_var, extra, self.sync, self.
                training, self.momentum, self.eps, self.activation, self.slope
                ).view(input_shape)
        else:
            return syncbatchnorm(x, self.weight, self.bias, self.
                running_mean, self.running_var, extra, self.sync, self.
                training, self.momentum, self.eps, self.activation, self.slope
                ).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(self.sync,
                self.activation, self.slope, self.inplace)


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=
            ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target
            )
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds
                [i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target
            )
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i
                ], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).
                forward(*inputs))


class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode=
            'bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode=
            'bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode=
            'bilinear', align_corners=True).squeeze(1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.
            aux_weight)


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000,
        use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            None
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.
                long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return self.criterion(pred, target)


class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index
        =-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = cfg.MODEL.ENCNET.SE_LOSS
        self.se_weight = cfg.MODEL.ENCNET.SE_WEIGHT
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.
                se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), bins=nclass,
                min=0, max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, (0)]
        else:
            class_pred = probas[:, (c)]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(
            fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, classes='present', per_image=False,
    ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0),
            lab.unsqueeze(0), ignore), classes=classes) for prob, lab in
            zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore),
            classes=classes)
    return loss


class LovaszSoftmax(nn.Module):

    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        loss = lovasz_softmax(F.softmax(preds[0], dim=1), target, ignore=
            self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], dim=1), target,
                ignore=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target
            )
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i
                ], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).
                forward(*inputs))


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, weight=None, aux=True,
        aux_weight=0.2, ignore_index=-1, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=
            self.ignore_index)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        loss = self._base_forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _base_forward(self, output, target):
        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1
                ), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1
                ), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = (1 - pt) ** self.gamma * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \\sum{x^p} + \\sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0
            ], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1
            ) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) *
            valid_mask, dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=True, aux_weight=0.4, ignore_index=
        -1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, (i)], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1
                        ], 'Expect weight shape [{}], get[{}]'.format(target
                        .shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss / target.shape[-1]

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))
        loss = self._base_forward(preds[0], target_one_hot, valid_mask)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class PointRendLoss(nn.CrossEntropyLoss):

    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(PointRendLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def forward(self, *inputs, **kwargs):
        result, gt = tuple(inputs)
        pred = F.interpolate(result['coarse'], gt.shape[-2:], mode=
            'bilinear', align_corners=True)
        seg_loss = F.cross_entropy(pred, gt, ignore_index=self.ignore_index)
        gt_points = point_sample(gt.float().unsqueeze(1), result['points'],
            mode='nearest', align_corners=False).squeeze_(1).long()
        points_loss = F.cross_entropy(result['rend'], gt_points,
            ignore_index=self.ignore_index)
        loss = seg_loss + points_loss
        return dict(loss=loss)


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class DataParallelModel(DataParallel):
    """Data parallelism

    Hide the difference of single/multiple GPUs to the user.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    Parameters
    ----------
    module : object
        Network to be parallelized.
    sync : bool
        enable synchronization (default: False).
    Inputs:
        - **inputs**: list of input
    Outputs:
        - **outputs**: list of output
    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        return modules


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
    """Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional), attr:'targets' (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        targets (tensor): targets to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:'targets' :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(*(list(input) + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0]
            )
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutputs):
        return Broadcast.apply(ctx.target_gpus, gradOutputs)


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_LikeLy_Journey_SegmenTron(_paritybench_base):
    pass
    def test_000(self):
        self._check(APNModule(*[], **{'in_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ASPOCModule(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(AttentionRefinmentModule(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BNPReLU(*[], **{'nIn': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(BaseAttentionBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(BaseOCModule(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(BasicBlockV1b(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(BinaryDiceLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(Bottleneck(*[], **{'in_channels': 4, 'inter_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(CAM_Module(*[], **{'in_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(CRPBlock(*[], **{'in_planes': 4, 'out_planes': 4, 'n_stages': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(CascadeFeatureFusion(*[], **{'low_channels': 4, 'high_channels': 4, 'out_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Classifer(*[], **{'dw_channels': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_014(self):
        self._check(Conv(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(ConvLayer(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(Custom_Conv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_017(self):
        self._check(DABModule(*[], **{'nIn': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(DUpsampling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(DepthConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(DepthSepConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(DoubleConv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_022(self):
        self._check(Down(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_023(self):
        self._check(DownSamplingBlock(*[], **{'nIn': 4, 'nOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_024(self):
        self._check(DownsamplerBlock(*[], **{'ninput': 4, 'noutput': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_025(self):
        self._check(Downsampling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_026(self):
        self._check(EDABlock(*[], **{'ninput': 4, 'dilated': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_027(self):
        self._check(EESP(*[], **{'in_channels': 64, 'out_channels': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_028(self):
        self._check(EESPNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_029(self):
        self._check(Enc(*[], **{'in_channels': 4, 'out_channels': 4, 'blocks': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_030(self):
        self._check(EncModule(*[], **{'in_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_031(self):
        self._check(Encoding(*[], **{'D': 4, 'K': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_032(self):
        self._check(FCAttention(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_033(self):
        self._check(FPEBlock(*[], **{'inplanes': 4, 'outplanes': 4, 'dilat': [4, 4, 4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_034(self):
        self._check(FeatureFused(*[], **{}), [torch.rand([4, 512, 64, 64]), torch.rand([4, 1024, 64, 64]), torch.rand([4, 4, 4, 4])], {})

    def test_035(self):
        self._check(FeatureFusion(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    @_fails_compile()
    def test_036(self):
        self._check(FeatureFusionModule(*[], **{'highter_in_channels': 4, 'lower_in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 4, 4])], {})

    def test_037(self):
        self._check(FrozenBatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_038(self):
        self._check(GlobalFeatureExtractor(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_039(self):
        self._check(HarDBlock(*[], **{'in_channels': 4, 'growth_rate': 4, 'grmul': 4, 'n_layers': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_040(self):
        self._check(InitialBlock(*[], **{'out_channels': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_041(self):
        self._check(InputInjection(*[], **{'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_042(self):
        self._check(InvertedResidual(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_043(self):
        self._check(LearningToDownsample(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_044(self):
        self._check(LinearBottleneck(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_045(self):
        self._check(MEUModule(*[], **{'channels_high': 4, 'channels_low': 4, 'channel_out': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_046(self):
        self._check(Mean(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4, 4])], {})

    @_fails_compile()
    def test_047(self):
        self._check(NaiveSyncBatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_048(self):
        self._check(OutConv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_049(self):
        self._check(PAM_Module(*[], **{'in_dim': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_050(self):
        self._check(PyramidAttentionBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_051(self):
        self._check(PyramidOCModule(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_052(self):
        self._check(PyramidPooling(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_053(self):
        self._check(SEModule(*[], **{'channels': 64}), [torch.rand([4, 64, 4, 4])], {})

    @_fails_compile()
    def test_054(self):
        self._check(SSnbt(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_055(self):
        self._check(SeparableConv2d(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_056(self):
        self._check(Shallow_net(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_057(self):
        self._check(SpatialPath(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_058(self):
        self._check(StableBCELoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_059(self):
        self._check(TransitionUp(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_060(self):
        self._check(Up(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    def test_061(self):
        self._check(XceptionA(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_062(self):
        self._check(_BNPReLU(*[], **{'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_063(self):
        self._check(_BiSeHead(*[], **{'in_channels': 4, 'inter_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_064(self):
        self._check(_ChannelWiseConv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_065(self):
        self._check(_ConvBN(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_066(self):
        self._check(_ConvBNPReLU(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_067(self):
        self._check(_ConvBNReLU(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_068(self):
        self._check(_DenseASPPBlock(*[], **{'in_channels': 4, 'inter_channels1': 4, 'inter_channels2': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_069(self):
        self._check(_DenseASPPConv(*[], **{'in_channels': 4, 'inter_channels': 4, 'out_channels': 4, 'atrous_rate': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_070(self):
        self._check(_DenseASPPHead(*[], **{'in_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_071(self):
        self._check(_DepthwiseConv(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_072(self):
        self._check(_FCNHead(*[], **{'in_channels': 4, 'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_073(self):
        self._check(_InputInjection(*[], **{'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_074(self):
        self._check(_PSPModule(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

