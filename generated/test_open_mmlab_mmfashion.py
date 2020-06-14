import sys
_module = sys.modules[__name__]
del sys
global_predictor_resnet_attr = _module
global_predictor_vgg_attr = _module
roi_predictor_resnet_attr = _module
roi_predictor_resnet_inshop = _module
roi_predictor_vgg_attr = _module
roi_predictor_vgg_inshop = _module
fashion_parsing_segmentation = _module
demo = _module
inference = _module
mask_rcnn_r50_fpn_1x = _module
mmfashion = _module
type_aware_recommendation_polyvore_disjoint = _module
type_aware_recommendation_polyvore_disjoint_l2_embed = _module
type_aware_recommendation_polyvore_nondisjoint = _module
landmark_detect_resnet = _module
landmark_detect_vgg = _module
roi_retriever_vgg = _module
global_retriever_resnet = _module
global_retriever_vgg = _module
global_retriever_vgg_loss_id = _module
global_retriever_vgg_loss_id_triplet = _module
roi_retriever_resnet = _module
roi_retriever_resnet_loss_id_triplet = _module
roi_retriever_vgg_loss_id = _module
roi_retriever_vgg_loss_id_triplet = _module
prepare_attr_pred = _module
prepare_consumer_to_shop = _module
prepare_in_shop = _module
prepare_landmark_detect = _module
test_fashion_recommender = _module
test_landmark_detector = _module
test_predictor = _module
test_retriever = _module
apis = _module
env = _module
train_fashion_recommender = _module
train_landmark_detector = _module
train_predictor = _module
train_retriever = _module
utils = _module
core = _module
evaluation = _module
attr_predict_demo = _module
attr_predict_eval = _module
cate_predict_eval = _module
landmark_detect_eval = _module
retrieval_demo = _module
retrieval_eval = _module
Attr_Pred = _module
Consumer_to_shop = _module
In_shop = _module
Landmark_Detect = _module
Polyvore_outfit = _module
datasets = _module
builder = _module
dataset_wrappers = _module
loader = _module
build_loader = _module
sampler = _module
registry = _module
models = _module
attr_predictor = _module
attr_predictor = _module
backbones = _module
resnet = _module
vgg = _module
concats = _module
concat = _module
embed_extractor = _module
embed_extract = _module
fashion_recommender = _module
base = _module
type_aware_recommend = _module
global_pool = _module
global_pool = _module
landmark_detector = _module
base = _module
landmark_feature_extractor = _module
landmark_feature_extract = _module
landmark_regression = _module
landmark_regression = _module
losses = _module
bce_with_logit_loss = _module
ce_loss = _module
cosine_embed_loss = _module
loss_norm = _module
margin_ranking_loss = _module
mse_loss = _module
triplet_loss = _module
predictor = _module
base = _module
global_predictor = _module
roi_predictor = _module
retriever = _module
base = _module
global_retriever = _module
roi_retriever = _module
roi_pool = _module
roi_pooling = _module
triplet_net = _module
triplet_net = _module
type_specific_net = _module
type_specific_net = _module
visibility_classifier = _module
visibility_classifier = _module
checkpoint = _module
image = _module
registry = _module
version = _module
setup = _module
extract_features = _module

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


import warnings


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.parallel


import torch.optim


import torch.utils.data


from torch.utils.data.dataset import Dataset


import random


from torch.autograd import Variable


import math


from torch.distributed import get_rank


from torch.distributed import get_world_size


from torch.utils.data import Sampler


from torch.utils.data.distributed import DistributedSampler as _DistributedSampler


import logging


from abc import ABCMeta


from abc import abstractmethod


import torch.nn.functional as F


from collections import OrderedDict


import inspect


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


ATTRPREDICTOR = Registry('attr_predictor')


LOSSES = Registry('loss')


def _build_module(cfg, registry, default_args):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(obj_type,
                registry.name))
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.
            format(type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, registry, default_args)


def build_loss(cfg):
    return build(cfg, LOSSES)


@ATTRPREDICTOR.register_module
class AttrPredictor(nn.Module):

    def __init__(self, inchannels, outchannels, loss_attr=dict(type=
        'BCEWithLogitsLoss', ratio=1, weight=None, size_average=None,
        reduce=None, reduction='mean')):
        super(AttrPredictor, self).__init__()
        self.linear_attr = nn.Linear(inchannels, outchannels)
        self.loss_attr = build_loss(loss_attr)

    def forward_train(self, x, attr):
        attr_pred = self.linear_attr(x)
        loss_attr = self.loss_attr(attr_pred, attr)
        return loss_attr

    def forward_test(self, x):
        attr_pred = torch.sigmoid(self.linear_attr(x))
        return attr_pred

    def forward(self, x, attr=None, return_loss=False):
        if return_loss:
            return self.forward_train(x, attr)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_attr.weight)
        if self.linear_attr.bias is not None:
            self.linear_attr.bias.data.fill_(0.01)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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


BACKBONES = Registry('backbone')


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
        except Exception:
            raise RuntimeError(
                'While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'
                .format(name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict.keys())
    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            print(err_msg)


def load_checkpoint(filename, model, strict=False, logger=None):
    checkpoint = torch.load(filename)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.
            format(filename))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['model_state_dict'].
            items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return model


@BACKBONES.register_module
class ResNet(nn.Module):
    layer_setting = {'resnet50': [3, 4, 6, 3], 'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3]}
    block_setting = {'resnet18': BasicBlock, 'resnet34': BasicBlock,
        'resnet50': Bottleneck}

    def __init__(self, setting='resnet50', zero_init_residual=False, groups
        =1, width_per_group=64, replace_stride_with_dilation=None,
        norm_layer=None):
        super(ResNet, self).__init__()
        block = self.block_setting[setting]
        layers = self.layer_setting[setting]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.zero_init_residual = zero_init_residual

    def init_weights(self, pretrained=None):
        None
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
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


@BACKBONES.register_module
class Vgg(nn.Module):
    setting = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
        512, 512, 512, 'M', 512, 512, 512, 'M']}

    def __init__(self, layer_setting='vgg16', batch_norm=False,
        init_weights=False):
        super(Vgg, self).__init__()
        self.features = self._make_layers(self.setting[layer_setting],
            batch_norm)
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x

    def init_weights(self, pretrained=None):
        None
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
        elif pretrained is None:
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
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


CONCATS = Registry('concat')


@CONCATS.register_module
class Concat(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(Concat, self).__init__()
        self.fc_fusion = nn.Linear(inchannels, outchannels)

    def forward(self, global_x, local_x=None):
        if local_x is not None:
            x = torch.cat((global_x, local_x), 1)
            x = self.fc_fusion(x)
        else:
            x = global_x
        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_fusion.weight)
        if self.fc_fusion.bias is not None:
            self.fc_fusion.bias.data.fill_(0.01)


EMBEDEXTRACTOR = Registry('embed_extractor')


@EMBEDEXTRACTOR.register_module
class EmbedExtractor(nn.Module):

    def __init__(self, inchannels, inter_channels, loss_id=dict(type=
        'CELoss', ratio=1, weight=None, size_average=None, reduce=None,
        reduction='mean'), loss_triplet=dict(type='TripletLoss', method=
        'cosine')):
        super(EmbedExtractor, self).__init__()
        self.embed_linear = nn.Linear(inchannels, inter_channels[0])
        self.bn = nn.BatchNorm1d(inter_channels[0], inter_channels[1])
        self.id_linear = nn.Linear(inter_channels[0], inter_channels[1])
        self.loss_id = build_loss(loss_id)
        if loss_triplet is not None:
            self.loss_triplet = build_loss(loss_triplet)
        else:
            self.loss_triplet = None

    def forward_train(self, x, id, triplet, pos, neg, triplet_pos_label,
        triplet_neg_label):
        embed = self.embed_linear(x)
        id_pred = self.id_linear(embed)
        loss_id = self.loss_id(id_pred, id)
        if triplet:
            pos_embed = self.embed_linear(pos)
            neg_embed = self.embed_linear(neg)
            loss_triplet = self.loss_triplet(embed, pos_embed, neg_embed,
                triplet_pos_label, triplet_neg_label)
            return loss_id + loss_triplet
        else:
            return loss_id

    def forward_test(self, x):
        embed = self.embed_linear(x)
        return embed

    def forward(self, x, id, return_loss=False, triplet=False, pos=None,
        neg=None, triplet_pos_label=None, triplet_neg_label=None):
        if return_loss:
            return self.forward_train(x, id, triplet, pos, neg,
                triplet_pos_label, triplet_neg_label)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed_linear.weight)
        if self.embed_linear.bias is not None:
            self.embed_linear.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.id_linear.weight)
        if self.id_linear.bias is not None:
            self.id_linear.bias.data.fill_(0.01)


class BaseFashionRecommender(nn.Module):
    """ Base class for fashion recommender"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseFashionRecommender, self).__init__()

    @abstractmethod
    def forward_test(self, imgs):
        pass

    @abstractmethod
    def forward_train(self, img, text, has_text, pos_img, pos_text,
        pos_has_text, neg_img, neg_text, neg_has_text, condition):
        pass

    def forward(self, img, text=None, has_text=None, pos_img=None, pos_text
        =None, pos_has_text=None, neg_img=None, neg_text=None, neg_has_text
        =None, condition=None, return_loss=True):
        if return_loss:
            return self.forward_train(img, text, has_text, pos_img,
                pos_text, pos_has_text, neg_img, neg_text, neg_has_text,
                condition)
        else:
            return self.forward_test(img)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))


GLOBALPOOLING = Registry('global_pool')


@GLOBALPOOLING.register_module
class GlobalPooling(nn.Module):

    def __init__(self, inplanes, pool_plane, inter_channels, outchannels):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(inplanes)
        inter_plane = inter_channels[0] * inplanes[0] * inplanes[1]
        if len(inter_channels) > 1:
            self.global_layers = nn.Sequential(nn.Linear(inter_plane,
                inter_channels[1]), nn.ReLU(True), nn.Dropout(), nn.Linear(
                inter_channels[1], outchannels), nn.ReLU(True), nn.Dropout())
        else:
            self.global_layers = nn.Linear(inter_plane, outchannels)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.global_layers(x)
        return global_pool

    def init_weights(self):
        if isinstance(self.global_layers, nn.Linear):
            nn.init.normal_(self.global_layers.weight, 0, 0.01)
            if self.global_layers.bias is not None:
                nn.init.constant_(self.global_layers.bias, 0)
        elif isinstance(self.global_layers, nn.Sequential):
            for m in self.global_layers:
                if type(m) == nn.Linear:
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


class BaseLandmarkDetector(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseLandmarkDetector, self).__init__()

    @abstractmethod
    def simple_test(self, img, landmark):
        pass

    @abstractmethod
    def aug_test(self, img, landmark):
        pass

    def forward_test(self, img):
        num_augs = len(img)
        if num_augs == 1:
            return self.simple_test(img[0])
        else:
            return self.aug_test(img)

    @abstractmethod
    def forward_train(self, img, vis, landmark_for_regreesion,
        landmark_for_roi_pool, attr):
        pass

    def forward(self, img, vis=None, landmark_for_regression=None,
        landmark_for_roi_pool=None, attr=None, return_loss=True):
        if return_loss:
            return self.forward_train(img, vis, landmark_for_regression,
                landmark_for_roi_pool, attr)
        else:
            return self.forward_test(img)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))


LANDMARKFEATUREEXTRACTOR = Registry('landmark_feature_extractor')


@LANDMARKFEATUREEXTRACTOR.register_module
class LandmarkFeatureExtractor(nn.Module):

    def __init__(self, inchannels, feature_dim, landmarks):
        super(LandmarkFeatureExtractor, self).__init__()
        self.linear = nn.Linear(inchannels, landmarks * feature_dim)
        self.landmarks = landmarks
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.landmarks, self.feature_dim)
        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.01)


LANDMARKREGRESSION = Registry('landmark_regression')


@LANDMARKREGRESSION.register_module
class LandmarkRegression(nn.Module):

    def __init__(self, inchannels, outchannels, landmark_num, loss_regress=
        dict(type='MSELoss', ratio=0.0001, reduction='mean')):
        super(LandmarkRegression, self).__init__()
        self.linear = nn.Linear(inchannels, outchannels)
        self.landmark_num = landmark_num
        self.loss_regress = build_loss(loss_regress)

    def forward_train(self, x, pred_vis, vis, landmark):
        pred_lm = self.linear(x).view(-1, self.landmark_num, 2)
        pred_vis = pred_vis.view(-1, self.landmark_num, 1)
        landmark = landmark.view(-1, self.landmark_num, 2)
        vis = vis.view(-1, self.landmark_num, 1)
        loss_regress = self.loss_regress(vis * pred_lm, vis * landmark)
        return loss_regress

    def forward_test(self, x):
        pred_lm = self.linear(x)
        return pred_lm

    def forward(self, x, pred_vis=None, vis=None, landmark=None,
        return_loss=True):
        if return_loss:
            return self.forward_train(x, pred_vis, vis, landmark)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.01)


@LOSSES.register_module
class BCEWithLogitsLoss(nn.Module):

    def __init__(self, ratio, weight, size_average, reduce, reduction):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction
        self.ratio = ratio

    def forward(self, input, target):
        target = target.float()
        return self.ratio * F.binary_cross_entropy_with_logits(input,
            target, self.weight, reduction=self.reduction)


@LOSSES.register_module
class CELoss(nn.Module):

    def __init__(self, ratio=1, weight=None, size_average=None,
        ignore_index=-100, reduce=None, reduction='mean'):
        super(CELoss, self).__init__()
        self.ratio = ratio
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return self.ratio * F.cross_entropy(input, target, weight=self.
            weight, ignore_index=self.ignore_index, reduction=self.reduction)


@LOSSES.register_module
class CosineEmbeddingLoss(nn.Module):

    def __init__(self, margin=0.0, size_average=None, reduce=None,
        reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, margin=self.
            margin, reduction=self.reduction)


@LOSSES.register_module
class L2NormLoss(nn.Module):

    def __init__(self, loss_weight=0.0005):
        super(L2NormLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x1, x2, x3, length):
        x_norm = (x1 + x2 + x3) / 3
        loss_norm = x_norm / np.sqrt(length)
        return self.loss_weight * loss_norm


@LOSSES.register_module
class L1NormLoss(nn.Module):

    def __init__(self, loss_weight=0.0005, average=True):
        super(L1NormLoss, self).__init__()
        self.loss_weight = loss_weight
        self.average = average

    def forward(self, x1, x2, x3, length):
        loss_norm = (x1 + x2 + x3) / 3
        if self.average:
            loss_norm = loss_norm / length
        return self.loss_weight * loss_norm


@LOSSES.register_module
class MarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, loss_weight=5e-05, size_average=None,
        reduce=None, reduction='mean'):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return self.loss_weight * F.margin_ranking_loss(input1, input2,
            target, margin=self.margin, reduction=self.reduction)


@LOSSES.register_module
class SelectiveMarginLoss(nn.Module):

    def __init__(self, loss_weight=5e-05, margin=0.2):
        super(SelectiveMarginLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight

    def forward(self, pos_samples, neg_samples, has_sample):
        margin_diff = torch.clamp(pos_samples - neg_samples + self.margin,
            min=0, max=1000000.0)
        num_sample = max(torch.sum(has_sample), 1)
        return self.loss_weight * (torch.sum(margin_diff * has_sample) /
            num_sample)


@LOSSES.register_module
class MSELoss(nn.Module):

    def __init__(self, ratio=1, size_average=None, reduce=None, reduction=
        'mean'):
        super(MSELoss, self).__init__()
        self.ratio = ratio
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target, avg_factor=None):
        return self.ratio * F.mse_loss(input, target, reduction=self.reduction)


@LOSSES.register_module
class TripletLoss(nn.Module):

    def __init__(self, method='cosine', ratio=1, margin=0.2, use_sigmoid=
        False, reduction='mean', size_average=True):
        super(TripletLoss, self).__init__()
        self.method = method
        self.ratio = ratio
        self.margin = margin
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.size_average = size_average

    def forward(self, anchor, pos, neg, triplet_pos_label, triplet_neg_label):
        if self.use_sigmoid:
            anchor, pos, neg = F.sigmoid(anchor), F.sigmoid(pos), F.sigmoid(neg
                )
        if self.method == 'cosine':
            loss_pos = F.cosine_embedding_loss(anchor, pos,
                triplet_pos_label, margin=self.margin, reduction=self.reduction
                )
            loss_neg = F.cosine_embedding_loss(anchor, neg,
                triplet_neg_label, margin=self.margin, reduction=self.reduction
                )
            losses = loss_pos + loss_neg
        else:
            dist_pos = (anchor - pos).pow(2).sum(1)
            dist_neg = (anchor - neg).pow(2).sum(1)
            losses = self.ratio * F.relu(dist_pos - dist_neg + self.margin)
            if self.size_average:
                losses = losses.mean()
            else:
                losses = losses.sum()
        return losses


class BasePredictor(nn.Module):
    """ Base class for attribute predictors"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BasePredictor, self).__init__()

    @property
    def with_roi_pool(self):
        return hasattr(self, 'roi_pool') and self.roi_pool is not None

    @abstractmethod
    def simple_test(self, img, landmark):
        pass

    @abstractmethod
    def aug_test(self, img, landmark):
        pass

    def forward_test(self, img, landmark=None):
        num_augs = len(img)
        if num_augs == 1:
            return self.simple_test(img[0], landmark[0])
        else:
            return self.aug_test(img, landmark)

    @abstractmethod
    def forward_train(self, img, landmark, attr):
        pass

    def forward(self, img, attr, cate=None, landmark=None, return_loss=True):
        if return_loss:
            return self.forward_train(img, landmark, attr)
        else:
            return self.forward_test(img, landmark)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))


class BaseRetriever(nn.Module):
    """ Base class for fashion retriever"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseRetriever, self).__init__()

    @property
    def with_roi_pool(self):
        return hasattr(self, 'roi_pool') and self.roi_pool is not None

    @abstractmethod
    def simple_test(self, imgs, landmarks):
        pass

    @abstractmethod
    def aug_test(self, imgs, landmarks):
        pass

    def forward_test(self, imgs, landmarks):
        num_augs = len(imgs)
        if num_augs == 1:
            return self.simple_test(imgs, landmarks)
        else:
            return self.aug_test(imgs, landmarks)

    @abstractmethod
    def forward_train(self, img, id, attr, pos, neg, anchor_lm, pos_lm,
        neg_lm, triplet_pos_label, triplet_neg_label):
        pass

    def forward(self, img, landmark=None, id=None, attr=None, pos=None, neg
        =None, pos_lm=None, neg_lm=None, triplet_pos_label=None,
        triplet_neg_label=None, return_loss=True):
        if return_loss:
            return self.forward_train(img, id, attr, pos, neg, landmark,
                pos_lm, neg_lm, triplet_pos_label, triplet_neg_label)
        else:
            return self.forward_test(img, landmark)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))


ROIPOOLING = Registry('roi_pool')


@ROIPOOLING.register_module
class RoIPooling(nn.Module):

    def __init__(self, pool_plane, inter_channels, outchannels, crop_size=7,
        img_size=(224, 224), num_lms=8, roi_size=2):
        super(RoIPooling, self).__init__()
        self.maxpool = nn.MaxPool2d(pool_plane)
        self.linear = nn.Sequential(nn.Linear(num_lms * inter_channels,
            outchannels), nn.ReLU(True), nn.Dropout())
        self.inter_channels = inter_channels
        self.outchannels = outchannels
        self.num_lms = num_lms
        self.crop_size = crop_size
        assert img_size[0] == img_size[1
            ], 'img width should equal to img height'
        self.img_size = img_size[0]
        self.roi_size = roi_size
        self.a = self.roi_size / float(self.crop_size)
        self.b = self.roi_size / float(self.crop_size)

    def forward(self, features, landmarks):
        """batch-wise RoI pooling.

        Args:
            features(tensor): the feature maps to be pooled.
            landmarks(tensor): crop the region of interest based on the
                landmarks(bs, self.num_lms).
        """
        batch_size = features.size(0)
        landmarks = landmarks / self.img_size * self.crop_size
        landmarks = landmarks.view(batch_size, self.num_lms, 2)
        ab = [np.array([[self.a, 0], [0, self.b]]) for _ in range(batch_size)]
        ab = np.stack(ab, axis=0)
        ab = torch.from_numpy(ab).float()
        size = torch.Size((batch_size, features.size(1), self.roi_size,
            self.roi_size))
        pooled = []
        for i in range(self.num_lms):
            tx = -1 + 2 * landmarks[:, (i), (0)] / float(self.crop_size)
            ty = -1 + 2 * landmarks[:, (i), (1)] / float(self.crop_size)
            t_xy = torch.stack((tx, ty)).view(batch_size, 2, 1)
            theta = torch.cat((ab, t_xy), 2)
            flowfield = nn.functional.affine_grid(theta, size)
            one_pooled = nn.functional.grid_sample(features, flowfield.to(
                torch.float32), mode='bilinear', padding_mode='border')
            one_pooled = self.maxpool(one_pooled).view(batch_size, self.
                inter_channels)
            pooled.append(one_pooled)
        pooled = torch.stack(pooled, dim=1).view(batch_size, -1)
        pooled = self.linear(pooled)
        return pooled

    def init_weights(self):
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class EmbedBranch(nn.Module):

    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(feat_dim, embedding_dim), nn.
            BatchNorm1d(embedding_dim, eps=0.001, momentum=0.01), nn.ReLU(
            inplace=True))
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        norm.unsqueeze_(1)
        x = x / norm.expand_as(x)
        return x


TRIPLETNET = Registry('triplet_net')


@TRIPLETNET.register_module
class TripletNet(nn.Module):

    def __init__(self, text_feature_dim, embed_feature_dim, loss_vse=dict(
        type='L1NormLoss', loss_weight=0.005, average=False), loss_triplet=
        dict(type='MarginRankingLoss', margin=0.3, loss_weight=1),
        loss_sim_i=dict(type='MarginRankingLoss', margin=0.3, loss_weight=
        5e-05), loss_selective_margin=dict(type='SelectiveMarginLoss',
        margin=0.3, loss_weight=5e-05), learned_metric=True):
        super(TripletNet, self).__init__()
        self.text_feature_dim = text_feature_dim
        self.embed_feature_dim = embed_feature_dim
        self.text_branch = EmbedBranch(text_feature_dim, embed_feature_dim)
        self.metric_branch = None
        if learned_metric:
            self.metric_branch = nn.Linear(embed_feature_dim, 1, bias=False)
        self.loss_vse = build_loss(loss_vse)
        self.loss_triplet = build_loss(loss_triplet)
        self.loss_sim_i = build_loss(loss_sim_i)
        self.loss_selective_margin = build_loss(loss_selective_margin)

    def image_forward(self, general_x, general_y, general_z):
        """ calculate image similarity loss on the general embedding
            general_x: general feature extracted by backbone
            general_y: far data(Negative)
            general_z: close data(Positive)
        """
        disti_p = F.pairwise_distance(general_y, general_z, 2)
        disti_n1 = F.pairwise_distance(general_y, general_x, 2)
        disti_n2 = F.pairwise_distance(general_z, general_x, 2)
        target = torch.FloatTensor(disti_p.size()).fill_(1)
        loss_sim_i1 = self.loss_sim_i(disti_p, disti_n1, target)
        loss_sim_i2 = self.loss_sim_i(disti_p, disti_n2, target)
        loss_sim_i = (loss_sim_i1 + loss_sim_i2) / 2.0
        return loss_sim_i

    def embed_forward(self, embed_x, embed_y, embed_z):
        """embed_x, mask_norm_x: type_specific net output (Anchor)
           embed_y, mask_norm_y: type_specifc net output (Negative)
           embed_z, mask_norm_z: type_specifi net output (Positive)
           conditions: only x(anchor data) has conditions
        """
        if self.metric_branch is None:
            dist_neg = F.pairwise_distance(embed_x, embed_y, 2)
            dist_pos = F.pairwise_distance(embed_x, embed_z, 2)
        else:
            dist_neg = self.metric_branch(embed_x * embed_y)
            dist_pos = self.metric_branch(embed_x * embed_z)
        target = torch.FloatTensor(dist_neg.size()).fill_(1)
        target = Variable(target)
        loss_type_triplet = self.loss_triplet(dist_neg, dist_pos, target)
        return loss_type_triplet

    def text_forward(self, text_x, text_y, text_z, has_text_x, has_text_y,
        has_text_z):
        desc_x = self.text_branch(text_x)
        desc_y = self.text_branch(text_y)
        desc_z = self.text_branch(text_z)
        distd_p = F.pairwise_distance(desc_y, desc_z, 2)
        distd_n1 = F.pairwise_distance(desc_x, desc_y, 2)
        distd_n2 = F.pairwise_distance(desc_x, desc_z, 2)
        has_text = has_text_x * has_text_y * has_text_z
        loss_sim_t1 = self.loss_selective_margin(distd_p, distd_n1, has_text)
        loss_sim_t2 = self.loss_selective_margin(distd_p, distd_n2, has_text)
        loss_sim_t = (loss_sim_t1 + loss_sim_t2) / 2.0
        return loss_sim_t, desc_x, desc_y, desc_z

    def calc_vse_loss(self, desc_x, general_x, general_y, general_z, has_text):
        """ Both y and z are assumed to be negatives because they are not from the same
            item as x
            desc_x: Anchor language embedding
            general_x: Anchor visual embedding
            general_y: Visual embedding from another item from input triplet
            general_z: Visual embedding from another item from input triplet
            has_text: Binary indicator of whether x had a text description
        """
        distd1_p = F.pairwise_distance(general_x, desc_x, 2)
        distd1_n1 = F.pairwise_distance(general_y, desc_x, 2)
        distd1_n2 = F.pairwise_distance(general_z, desc_x, 2)
        loss_vse_1 = self.loss_selective_margin(distd1_p, distd1_n1, has_text)
        loss_vse_2 = self.loss_selective_margin(distd1_p, distd1_n2, has_text)
        return (loss_vse_1 + loss_vse_2) / 2.0

    def forward(self, general_x, type_embed_x, text_x, has_text_x,
        general_y, type_embed_y, text_y, has_text_y, general_z,
        type_embed_z, text_z, has_text_z):
        """x: Anchor data
           y: Distant(negative) data
           z: Close(positive) data
        """
        loss_sim_i = self.image_forward(general_x, general_y, general_z)
        loss_type_triplet = self.embed_forward(type_embed_x, type_embed_y,
            type_embed_z)
        loss_sim_t, desc_x, desc_y, desc_z = self.text_forward(text_x,
            text_y, text_z, has_text_x, has_text_y, has_text_z)
        loss_vse_x = self.calc_vse_loss(desc_x, general_x, general_y,
            general_z, has_text_x)
        loss_vse_y = self.calc_vse_loss(desc_y, general_y, general_x,
            general_z, has_text_y)
        loss_vse_z = self.calc_vse_loss(desc_z, general_z, general_x,
            general_y, has_text_z)
        loss_vse = self.loss_vse(loss_vse_x, loss_vse_y, loss_vse_z, len(
            general_x))
        return loss_type_triplet, loss_sim_t, loss_vse, loss_sim_i

    def init_weights(self):
        if self.metric_branch is not None:
            weight = torch.zeros(1, self.embed_feature_dim) / float(self.
                embed_feature_dim)
            self.metric_branch.weight = nn.Parameter(weight)


class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


TYPESPECIFICNET = Registry('type_specific_net')


_global_config['num_rand_embed'] = 4


@TYPESPECIFICNET.register_module
class TypeSpecificNet(nn.Module):

    def __init__(self, learned, n_conditions, rand_typespaces=False, use_fc
        =True, l2_embed=False, dim_embed=256, prein=False):
        """init

        Args:
            learned: boolean, indicating whether masks are learned or fixed
            n_conditions: Integer defining number of different similarity
                notions
            use_fc: When true a fully connected layer is learned to transform
                the general embedding to the type specific embedding
            l2_embed: When true we l2 normalize the output type specific
                embeddings
            prein: boolean, indicating whether masks are initialized in equally
                sized disjoint sections or random otherwise
        """
        super(TypeSpecificNet, self).__init__()
        assert learned == True and use_fc == False or learned == False and use_fc == True, 'learn a metric or use fc layer to transform the general embeddings, only one can be true.'
        self.learnedmask = learned
        if rand_typespaces:
            n_conditions = int(np.ceil(n_conditions / float(args.
                num_rand_embed)))
        self.fc_masks = use_fc
        self.l2_norm = l2_embed
        if self.fc_masks:
            masks = []
            for i in range(n_conditions):
                masks.append(nn.Linear(dim_embed, dim_embed))
            self.masks = ListModule(*masks)
        elif self.learnedmask:
            if prein:
                self.masks = nn.Embedding(n_conditions, dim_embed)
                mask_array = np.zeros([n_conditions, dim_embed])
                mask_array.fill(0.1)
                mask_len = int(dim_embed / n_conditions)
                for i in range(n_conditions):
                    mask_array[(i), i * mask_len:(i + 1) * mask_len] = 1
                self.masks.weight = nn.Parameter(torch.Tensor(mask_array),
                    requires_grad=True)
            else:
                self.masks = nn.Embedding(n_conditions, dim_embed)
                self.masks.weight.data.normal_(0.9, 0.7)
        else:
            self.masks = nn.Embedding(n_conditions, dim_embed)
            mask_array = np.zeros([n_conditions, dim_embed])
            mask_len = int(dim_embed / n_conditions)
            for i in range(n_conditions):
                mask_array[(i), i * mask_len:(i + 1) * mask_len] = 1
            self.masks.weight = nn.Parameter(torch.Tensor(mask_array),
                requires_grad=False)

    def forward_test(self, embed_x):
        if self.fc_masks:
            masked_embedding = []
            for mask in self.masks:
                masked_embedding.append(mask(embed_x).unsqueeze(1))
            masked_embedding = torch.cat(masked_embedding, 1)
            embedded_x = embed_x.unsqueeze(1)
        else:
            masks = Variable(self.masks.weight.data)
            masks = masks.unsqueeze(0).repeat(embed_x.size(0), 1, 1)
            embedded_x = embed_x.unsqueeze(1)
            masked_embedding = embedded_x.expand_as(masks) * masks
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=2) + 1e-10
            norm.unsqueeze_(2)
            masked_embedding = masked_embedding / norm.expand_as(
                masked_embedding)
        return torch.cat((masked_embedding, embedded_x), 1)

    def forward_train(self, embed_x, c=None):
        """forward_train.

        Args:
            embed_x: feature embeddings.
            c: type specific embedding to compute for the images, returns all
                embeddings when None including the general embedding
                concatenated onto the end.
        """
        if self.fc_masks:
            mask_norm = 0.0
            masked_embedding = []
            for embed, condition in zip(embed_x, c):
                mask = self.masks[condition]
                masked_embedding.append(mask(embed.unsqueeze(0)))
                mask_norm += mask.weight.norm(1)
            masked_embedding = torch.cat(masked_embedding)
        else:
            self.mask = self.masks(c)
            if self.learnedmask:
                self.mask = torch.nn.functional.relu(self.mask)
            masked_embedding = embed_x * self.mask
            mask_norm = self.mask.norm(1)
        embed_norm = embed_x.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            norm.unsqueeze_(1)
            masked_embedding = masked_embedding / norm
        return masked_embedding, mask_norm, embed_norm

    def forward(self, embed_x, c=None, return_loss=True):
        if return_loss:
            return self.forward_train(embed_x, c)
        else:
            return self.forward_test(embed_x)

    def init_weights(self):
        if isinstance(self.masks, nn.Sequential):
            for m in self.masks:
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)
        elif isinstance(self.masks, nn.Module):
            for m in self.masks.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)


VISIBILITYCLASSIFIER = Registry('visibility_classifier')


@VISIBILITYCLASSIFIER.register_module
class VisibilityClassifier(nn.Module):

    def __init__(self, inchannels, outchannels, landmark_num, loss_vis=dict
        (type='BCEWithLogitsLoss', ratio=1, reduction='none')):
        super(VisibilityClassifier, self).__init__()
        self.linear = nn.Linear(inchannels, 1)
        self.landmark_num = landmark_num
        self.loss_vis = builder.build_loss(loss_vis)

    def forward_train(self, x, vis):
        losses_vis = []
        vis_pred_list = []
        for i in range(self.landmark_num):
            lm_feat = x[:, (i), :]
            vis_pred = F.sigmoid(self.linear(lm_feat))
            lm_vis = vis[:, (i)].unsqueeze(1)
            vis_pred_list.append(lm_vis)
            loss_vis = self.loss_vis(vis_pred, lm_vis)
            losses_vis.append(loss_vis)
        losses_vis_tensor = torch.stack(losses_vis).transpose(1, 0)[:, :, (0)]
        vis_pred_list = torch.stack(vis_pred_list).transpose(1, 0)[:, :, (0)]
        losses_vis_tensor_mean_per_lm = torch.mean(losses_vis_tensor, dim=1,
            keepdim=True)
        losses_vis_tensor_mean_per_batch = torch.mean(
            losses_vis_tensor_mean_per_lm)
        return losses_vis_tensor_mean_per_batch, vis_pred_list

    def forward_test(self, x):
        vis_pred_list = []
        for i in range(self.landmark_num):
            lm_feat = x[:, (i), :]
            vis_pred = F.sigmoid(self.linear(lm_feat))
            vis_pred_list.append(vis_pred)
        vis_pred_list = torch.stack(vis_pred_list).transpose(1, 0)[:, :, (0)]
        return vis_pred_list

    def forward(self, x, vis=None, return_loss=True):
        if return_loss:
            return self.forward_train(x, vis)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.01)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_open_mmlab_mmfashion(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BaseFashionRecommender(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BaseLandmarkDetector(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BasePredictor(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(BaseRetriever(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Concat(*[], **{'inchannels': 4, 'outchannels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(CosineEmbeddingLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(EmbedBranch(*[], **{'feat_dim': 4, 'embedding_dim': 4}), [torch.rand([4, 4, 4])], {})

    def test_008(self):
        self._check(GlobalPooling(*[], **{'inplanes': [4, 4], 'pool_plane': 4, 'inter_channels': [4, 4], 'outchannels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(L1NormLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(L2NormLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(LandmarkFeatureExtractor(*[], **{'inchannels': 4, 'feature_dim': 4, 'landmarks': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(MSELoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(MarginRankingLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(ResNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_015(self):
        self._check(SelectiveMarginLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(TripletLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(Vgg(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

