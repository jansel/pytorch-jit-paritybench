import sys
_module = sys.modules[__name__]
del sys
config = _module
evaluate = _module
function = _module
inference = _module
loss = _module
dataset = _module
_init_paths = _module
h36m = _module
joints_dataset = _module
mixed_dataset = _module
mpii = _module
multiview_h36m = _module
multiview_mpii = _module
models = _module
multiview_pose_hrnet = _module
multiview_pose_mobilenetv2 = _module
multiview_pose_resnet = _module
pose_hrnet = _module
pose_mobilenetv2 = _module
pose_resnet = _module
body = _module
cameras = _module
cameras_cuda = _module
pictorial = _module
pictorial_cuda = _module
triangulate = _module
utils = _module
pose_utils = _module
transforms = _module
vis = _module
zipreader = _module
train = _module
valid = _module
estimate = _module
estimate_cuda = _module

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


import torch.nn as nn


import torch


import logging


import math


import numpy as np


import torch.nn.functional as F


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1
            )
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, (
                    idx)]), heatmap_gt.mul(target_weight[:, (idx)]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        return loss


class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size, size))
        self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        output = torch.matmul(input_reshape, self.weight)
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        NUM_NETS = 12
        size = int(cfg.NETWORK.HEATMAP_SIZE[0])
        self.weights = weights
        self.aggre = nn.ModuleList()
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size * size))

    def sort_views(self, target, all_views):
        indicator = [(target is item) for item in all_views]
        new_views = [target.clone()]
        for i, item in zip(indicator, all_views):
            if not i:
                new_views.append(item.clone())
        return new_views

    def fuse_with_weights(self, views):
        target = torch.zeros_like(views[0])
        for v, w in zip(views, self.weights):
            target += v * w
        return target

    def forward(self, inputs):
        index = 0
        outputs = []
        nviews = len(inputs)
        for i in range(nviews):
            sorted_inputs = self.sort_views(inputs[i], inputs)
            warped = [sorted_inputs[0]]
            for j in range(1, nviews):
                fc = self.aggre[index]
                fc_output = fc(sorted_inputs[j])
                warped.append(fc_output)
                index += 1
            output = self.fuse_with_weights(warped)
            outputs.append(output)
        return outputs


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, Aggre, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        self.resnet = PoseResNet
        self.aggre_layer = Aggre

    def forward(self, views):
        if isinstance(views, list):
            single_views = []
            for view in views:
                heatmaps = self.resnet(view)
                single_views.append(heatmaps)
            multi_views = []
            if self.config.NETWORK.AGGRE:
                multi_views = self.aggre_layer(single_views)
            return single_views, multi_views
        else:
            return self.resnet(views)


class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size, size))
        self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        output = torch.matmul(input_reshape, self.weight)
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        NUM_NETS = 12
        size = int(cfg.NETWORK.HEATMAP_SIZE[0])
        self.weights = weights
        self.aggre = nn.ModuleList()
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size * size))

    def sort_views(self, target, all_views):
        indicator = [(target is item) for item in all_views]
        new_views = [target.clone()]
        for i, item in zip(indicator, all_views):
            if not i:
                new_views.append(item.clone())
        return new_views

    def fuse_with_weights(self, views):
        target = torch.zeros_like(views[0])
        for v, w in zip(views, self.weights):
            target += v * w
        return target

    def forward(self, inputs):
        index = 0
        outputs = []
        nviews = len(inputs)
        for i in range(nviews):
            sorted_inputs = self.sort_views(inputs[i], inputs)
            warped = [sorted_inputs[0]]
            for j in range(1, nviews):
                fc = self.aggre[index]
                fc_output = fc(sorted_inputs[j])
                warped.append(fc_output)
                index += 1
            output = self.fuse_with_weights(warped)
            outputs.append(output)
        return outputs


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, Aggre, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        self.resnet = PoseResNet
        self.aggre_layer = Aggre

    def forward(self, views):
        if isinstance(views, list):
            single_views = []
            for view in views:
                heatmaps = self.resnet(view)
                single_views.append(heatmaps)
            multi_views = []
            if self.config.NETWORK.AGGRE:
                multi_views = self.aggre_layer(single_views)
            return single_views, multi_views
        else:
            return self.resnet(views)


class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size, size))
        self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        output = torch.matmul(input_reshape, self.weight)
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        NUM_NETS = 12
        size = int(cfg.NETWORK.HEATMAP_SIZE[0])
        self.weights = weights
        self.aggre = nn.ModuleList()
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size * size))

    def sort_views(self, target, all_views):
        indicator = [(target is item) for item in all_views]
        new_views = [target.clone()]
        for i, item in zip(indicator, all_views):
            if not i:
                new_views.append(item.clone())
        return new_views

    def fuse_with_weights(self, views):
        target = torch.zeros_like(views[0])
        for v, w in zip(views, self.weights):
            target += v * w
        return target

    def forward(self, inputs):
        index = 0
        outputs = []
        nviews = len(inputs)
        for i in range(nviews):
            sorted_inputs = self.sort_views(inputs[i], inputs)
            warped = [sorted_inputs[0]]
            for j in range(1, nviews):
                fc = self.aggre[index]
                fc_output = fc(sorted_inputs[j])
                warped.append(fc_output)
                index += 1
            output = self.fuse_with_weights(warped)
            outputs.append(output)
        return outputs


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, Aggre, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        self.resnet = PoseResNet
        self.aggre_layer = Aggre

    def forward(self, views):
        if isinstance(views, list):
            single_views = []
            for view in views:
                heatmaps = self.resnet(view)
                single_views.append(heatmaps)
            multi_views = []
            if self.config.NETWORK.AGGRE:
                multi_views = self.aggre_layer(single_views)
            return single_views, multi_views
        else:
            return self.resnet(views)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM
            )
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


logger = logging.getLogger(__name__)


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
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                num_channels[branch_index] * block.expansion, momentum=
                BN_MOMENTUM))
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
                                num_outchannels_conv3x3), nn.ReLU(True)))
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


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL_EXTRA
        super(PoseHighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.stage2_cfg = cfg['MODEL_EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
            num_channels)
        self.stage3_cfg = cfg['MODEL_EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
            num_channels)
        self.stage4_cfg = cfg['MODEL_EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg,
            num_channels, multi_scale_output=False)
        self.final_layer = nn.Conv2d(in_channels=pre_stage_channels[0],
            out_channels=cfg.NETWORK.NUM_JOINTS, kernel_size=extra.
            FINAL_CONV_KERNEL, stride=1, padding=1 if extra.
            FINAL_CONV_KERNEL == 3 else 0)
        self.pretrained_layers = cfg['MODEL_EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(
                        num_channels_pre_layer[i], num_channels_cur_layer[i
                        ], 3, 1, 1, bias=False), nn.BatchNorm2d(
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
                        outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(
                        outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = self.final_layer(y_list[0])
        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0
                    ] in self.pretrained_layers or self.pretrained_layers[0
                    ] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim,
                oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=
                True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0,
                bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4,
            2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = int(32 * width_mult)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PoseMobileNetV2(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 320
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS
        super(PoseMobileNetV2, self).__init__()
        self.mobilenetv2 = MobileNetV2()
        self.deconv_layers = self._make_deconv_layer(cfg.POSE_RESNET.
            NUM_DECONV_LAYERS, cfg.POSE_RESNET.NUM_DECONV_FILTERS, cfg.
            POSE_RESNET.NUM_DECONV_KERNELS)
        self.final_layer = nn.Conv2d(in_channels=cfg.POSE_RESNET.
            NUM_DECONV_FILTERS[-1], out_channels=cfg.NETWORK.NUM_JOINTS,
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL, stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters
            ), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels
            ), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels
                [i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                out_channels=planes, kernel_size=kernel, stride=2, padding=
                padding, output_padding=output_padding, bias=self.
                deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mobilenetv2(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.
                        format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.
                        format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM
            )
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


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(cfg.POSE_RESNET.
            NUM_DECONV_LAYERS, cfg.POSE_RESNET.NUM_DECONV_FILTERS, cfg.
            POSE_RESNET.NUM_DECONV_KERNELS)
        self.final_layer = nn.Conv2d(in_channels=cfg.POSE_RESNET.
            NUM_DECONV_FILTERS[-1], out_channels=cfg.NETWORK.NUM_JOINTS,
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL, stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters
            ), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels
            ), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels
                [i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                out_channels=planes, kernel_size=kernel, stride=2, padding=
                padding, output_padding=output_padding, bias=self.
                deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.
                        format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.
                        format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_microsoft_multiview_human_pose_estimation_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(JointsMSELoss(*[], **{'use_target_weight': 4}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(MobileNetV2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})
