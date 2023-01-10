import sys
_module = sys.modules[__name__]
del sys
batched_drive = _module
batched_finetune = _module
compute_pose_identity_error = _module
adversarial = _module
perceptual_loss = _module
dice = _module
dis_embed = _module
featmat = _module
idt_embed = _module
l1_rgb = _module
perceptual = _module
augmentation = _module
voxceleb = _module
dataloader = _module
voxceleb2 = _module
voxceleb2_FSTH_crop = _module
voxceleb2_X2Face = _module
voxceleb2_segm = _module
voxceleb2_segmentation_nolandmarks = _module
voxceleb2_segmentation_nolandmarks_X2Face_FAbNet_crops = _module
FSTH = _module
no_landmarks = _module
none = _module
drive = _module
FAbNet_pretrained_embResNeXt = _module
FSTH = _module
X2Face = _module
X2Face_pretrained_embResNeXt = _module
no_pose_encoder = _module
unsupervised_pose_separate_embResNeXt_segmentation = _module
FSTH = _module
FSTH_plus = _module
X2Face = _module
blocks = _module
vector_pose_unsupervised_segmentation_noBottleneck = _module
holycow = _module
train = _module
utils = _module
argparse_utils = _module
crop_as_in_dataset = _module
radam = _module
tensorboard_logging = _module
utils = _module
visualize = _module

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


from torch import nn


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


import torchvision


import numpy as np


import logging


import torch.utils.data


import scipy


import random


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import _SingleProcessDataLoaderIter


from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter


import copy


import math


from torch.nn.utils import spectral_norm


import time


import itertools


from torch.optim import Adam


from abc import ABC


from abc import abstractmethod


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.utils.tensorboard import SummaryWriter


from torchvision.utils import make_grid


from collections import defaultdict


from typing import List


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1)


class PerceptualLoss(nn.Module):

    def __init__(self, weight, vgg_weights_dir, net='caffe', normalize_grad=False):
        super().__init__()
        self.weight = weight
        self.normalize_grad = normalize_grad
        if net == 'pytorch':
            model = torchvision.models.vgg19(pretrained=True).features
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            num_layers = 30
        elif net == 'caffe':
            vgg_weights = torch.load(os.path.join(vgg_weights_dir, 'vgg19-d01eb7cb.pth'))
            map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
            vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])
            model = torchvision.models.vgg19()
            model.classifier = nn.Sequential(Flatten(), *model.classifier._modules.values())
            model.load_state_dict(vgg_weights)
            model = model.features
            mean = torch.tensor([103.939, 116.779, 123.68]) / 255.0
            std = torch.tensor([1.0, 1.0, 1.0]) / 255.0
            num_layers = 30
        elif net == 'face':
            model = torchvision.models.vgg16().features
            model.load_state_dict(torch.load(os.path.join(vgg_weights_dir, 'vgg_face_weights.pth')))
            mean = torch.tensor([103.939, 116.779, 123.68]) / 255.0
            std = torch.tensor([1.0, 1.0, 1.0]) / 255.0
            num_layers = 30
        else:
            raise ValueError(f"Unknown type of PerceptualLoss: expected '{{pytorch,caffe,face}}', got '{net}'")
        self.register_buffer('mean', mean[None, :, None, None])
        self.register_buffer('std', std[None, :, None, None])
        layers_avg_pooling = []
        for weights in model.parameters():
            weights.requires_grad = False
        for module in model.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                layers_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers_avg_pooling.append(module)
            if len(layers_avg_pooling) >= num_layers:
                break
        layers_avg_pooling = nn.Sequential(*layers_avg_pooling)
        self.model = layers_avg_pooling

    def normalize_inputs(self, x):
        return (x - self.mean) / self.std

    def forward(self, input, target):
        input = (input + 1) / 2
        target = (target.detach() + 1) / 2
        loss = 0
        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)
        for layer in self.model:
            features_input = layer(features_input)
            features_target = layer(features_target)
            if layer.__class__.__name__ == 'ReLU':
                if self.normalize_grad:
                    pass
                else:
                    loss = loss + F.l1_loss(features_input, features_target)
        return loss * self.weight


class Criterion(nn.Module):

    def __init__(self, perc_weight, vgg_weights_dir):
        super().__init__()
        self.perceptual_crit = PerceptualLoss(perc_weight, vgg_weights_dir).eval()

    def forward(self, data_dict):
        fake_rgb = data_dict['fake_rgbs']
        real_rgb = data_dict['target_rgbs']
        if len(fake_rgb.shape) > 4:
            fake_rgb = fake_rgb[:, 0]
        if len(real_rgb.shape) > 4:
            real_rgb = real_rgb[:, 0]
        loss_G_dict = {}
        loss_G_dict['VGG'] = self.perceptual_crit(fake_rgb, real_rgb)
        return loss_G_dict


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.finetuning = False

    def enable_finetuning(self, _=None):
        self.finetuning = True

    def forward(self, _):
        pass


class Embedder(nn.Module):

    def __init__(self, identity_embedding_size, pose_embedding_size, average_function):
        super().__init__()
        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size
        import torchvision
        self.identity_encoder = torchvision.models.resnext50_32x4d(num_classes=identity_embedding_size)
        self.pose_encoder = torchvision.models.mobilenet_v2(num_classes=pose_embedding_size)
        self.average_function = average_function
        self.finetuning = False

    def enable_finetuning(self, data_dict=None):
        self.finetuning = True

    def get_identity_embedding(self, data_dict):
        inputs = data_dict['enc_rgbs']
        batch_size, num_faces, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        identity_embeddings = self.identity_encoder(inputs).view(batch_size, num_faces, -1)
        assert identity_embeddings.shape[2] == self.identity_embedding_size
        if self.average_function == 'sum':
            identity_embeddings_aggregated = identity_embeddings.mean(1)
        elif self.average_function == 'max':
            identity_embeddings_aggregated = identity_embeddings.max(1)[0]
        else:
            raise ValueError('Incorrect `average_function` argument, expected `sum` or `max`')
        data_dict['embeds'] = identity_embeddings_aggregated
        data_dict['embeds_elemwise'] = identity_embeddings

    def get_pose_embedding(self, data_dict):
        x = data_dict['pose_input_rgbs'][:, 0]
        data_dict['pose_embedding'] = self.pose_encoder(x)

    def forward(self, data_dict):
        if not self.finetuning:
            self.get_identity_embedding(data_dict)
        self.get_pose_embedding(data_dict)


class Constant(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, *shape))

    def forward(self, batch_size):
        return self.constant.expand((batch_size,) + self.constant.shape[1:])


class Generator(nn.Module):

    def __init__(self, padding, in_channels, out_channels, num_channels, max_num_channels, identity_embedding_size, pose_embedding_size, norm_layer, gen_constant_input_size, gen_num_residual_blocks, output_image_size):
        super().__init__()

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False, norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False, norm_layer=norm_layer)
        if padding == 'zero':
            padding = nn.ZeroPad2d
        elif padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')
        assert math.log2(output_image_size / gen_constant_input_size).is_integer(), '`gen_constant_input_size` must be `image_size` divided by a power of 2'
        num_upsample_blocks = int(math.log2(output_image_size / gen_constant_input_size))
        out_channels_block_nonclamped = num_channels * 2 ** num_upsample_blocks
        out_channels_block = min(out_channels_block_nonclamped, max_num_channels)
        self.constant = Constant(out_channels_block, gen_constant_input_size, gen_constant_input_size)
        current_image_size = gen_constant_input_size
        layers = []
        for i in range(gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        for _ in range(num_upsample_blocks):
            in_channels_block = out_channels_block
            out_channels_block_nonclamped //= 2
            out_channels_block = min(out_channels_block_nonclamped, max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + norm_layer))
        layers.extend([blocks.AdaptiveNorm2d(out_channels_block, norm_layer), nn.ReLU(True), spectral_norm(nn.Conv2d(out_channels_block, out_channels, 3, 1, 1), eps=0.0001), nn.Tanh()])
        self.decoder_blocks = nn.Sequential(*layers)
        self.adains = [module for module in self.modules() if module.__class__.__name__ == 'AdaptiveNorm2d']
        self.identity_embedding_size = identity_embedding_size
        self.pose_embedding_size = pose_embedding_size
        joint_embedding_size = identity_embedding_size + pose_embedding_size
        self.affine_params_projector = nn.Sequential(spectral_norm(nn.Linear(joint_embedding_size, max(joint_embedding_size, 512))), nn.ReLU(True), spectral_norm(nn.Linear(max(joint_embedding_size, 512), self.get_num_affine_params())))
        self.finetuning = False

    def get_num_affine_params(self):
        return sum(2 * module.num_features for module in self.adains)

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == 'AdaptiveNorm2d':
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]
                if m.bias is None:
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)
                if m.weight is None:
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)
                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, data_dict):
        if self.finetuning:
            identity_embedding = self.identity_embedding.expand(len(data_dict['pose_embedding']), -1)
        else:
            identity_embedding = data_dict['embeds']
        pose_embedding = data_dict['pose_embedding']
        joint_embedding = torch.cat((identity_embedding, pose_embedding), dim=1)
        affine_params = self.affine_params_projector(joint_embedding)
        self.assign_affine_params(affine_params)

    def enable_finetuning(self, data_dict=None):
        """
            Make the necessary adjustments to generator architecture to allow fine-tuning.
            For `vanilla` generator, initialize AdaIN parameters from `data_dict['embeds']`
            and flag them as trainable parameters.
            Will require re-initializing optimizer, but only after the first call.

            data_dict:
                dict
                Required contents depend on the specific generator. For `vanilla` generator,
                it is `'embeds'` (1 x `args.embed_channels`).
                If `None`, the module's new parameters will be initialized randomly.
        """
        if data_dict is None:
            some_parameter = next(iter(self.parameters()))
            identity_embedding = torch.rand(1, self.identity_embedding_size)
        else:
            identity_embedding = data_dict['embeds']
        if self.finetuning:
            with torch.no_grad():
                self.identity_embedding.copy_(identity_embedding)
        else:
            self.identity_embedding = nn.Parameter(identity_embedding)
            self.finetuning = True

    def forward(self, data_dict):
        self.assign_embeddings(data_dict)
        batch_size = len(data_dict['pose_embedding'])
        outputs = self.decoder_blocks(self.constant(batch_size))
        rgb, segmentation = outputs[:, :-1], outputs[:, -1:]
        rgb = rgb * 0.75
        rgb += 0.5
        segmentation = segmentation * 0.5
        segmentation += 0.5
        data_dict['fake_rgbs'] = rgb * segmentation
        data_dict['fake_segm'] = segmentation


class AdaptiveNorm2d(nn.Module):

    def __init__(self, num_features, norm_layer='in', eps=0.0001):
        super(AdaptiveNorm2d, self).__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        if 'in' in norm_layer:
            self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        elif 'bn' in norm_layer:
            self.norm_layer = SyncBatchNorm(num_features, momentum=1.0, eps=eps, affine=False)
        self.delete_weight_on_forward = True

    def forward(self, input):
        out = self.norm_layer(input)
        output = out * self.weight[:, :, None, None] + self.bias[:, :, None, None]
        if self.delete_weight_on_forward:
            self.weight = self.bias = None
        return output


class AdaptiveNorm2dTrainable(nn.Module):

    def __init__(self, num_features, norm_layer='in', eps=0.0001):
        super(AdaptiveNorm2dTrainable, self).__init__()
        self.num_features = num_features
        if 'in' in norm_layer:
            self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)

    def forward(self, input):
        out = self.norm_layer(input)
        t = out.shape[0] // self.weight.shape[0]
        output = out * self.weight + self.bias
        return output

    def assign_params(self, weight, bias):
        self.weight = torch.nn.Parameter(weight.view(1, -1, 1, 1))
        self.bias = torch.nn.Parameter(bias.view(1, -1, 1, 1))


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding, upsample, downsample, norm_layer, activation=nn.ReLU, gated=False):
        super(ResBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=0.0001, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=0.0001, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')
        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([nn.Sequential() if padding is nn.ZeroPad2d else padding(1), spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1 if padding is nn.ZeroPad2d else 0, bias=bias), eps=0.0001)])
        if normalize:
            layers.append(norm1)
        layers.extend([activation(inplace=True), nn.Sequential() if padding is nn.ZeroPad2d else padding(1), spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1 if padding is nn.ZeroPad2d else 0, bias=bias), eps=0.0001)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)
        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(nn.Conv2d(in_channels, out_channels, 1), eps=0.0001))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class channelShuffle(nn.Module):

    def __init__(self, groups):
        super(channelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class shuffleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(shuffleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        groups = 4
        block = []
        if in_channels % groups == 0 and out_channels % groups == 0:
            block.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, groups=groups), eps=0.0001))
            block.append(nn.ReLU6(inplace=True))
            block.append(channelShuffle(groups=groups))
            block.append(spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=groups), eps=0.0001))
            block.append(nn.ReLU6(inplace=True))
            block.append(spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, groups=groups), eps=0.0001))
        else:
            block.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1), eps=0.0001))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x


class ResBlockShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, padding, upsample, downsample, norm_layer, activation=nn.ReLU, gated=False):
        super(ResBlockShuffle, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=0.0001, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=0.0001, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')
        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([shuffleConv(in_channels, out_channels, 3, 1, 0, bias=bias)])
        if normalize:
            layers.append(norm1)
        layers.extend([activation(inplace=True), shuffleConv(out_channels, out_channels, 3, 1, 0, bias=bias)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)
        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(shuffleConv(in_channels, out_channels, 1))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class ResBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups, resize_layer, norm_layer, activation):
        super(ResBlockV2, self).__init__()
        upsampling_layers = {'nearest': lambda : nn.Upsample(scale_factor=stride, mode='nearest')}
        downsampling_layers = {'avgpool': lambda : nn.AvgPool2d(stride)}
        norm_layers = {'bn': lambda num_features: SyncBatchNorm(num_features, momentum=1.0, eps=0.0001), 'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=0.0001, affine=True), 'adabn': lambda num_features: AdaptiveNorm2d(num_features, 'bn'), 'adain': lambda num_features: AdaptiveNorm2d(num_features, 'in')}
        normalize = norm_layer != 'none'
        bias = not normalize
        upsample = resize_layer in upsampling_layers
        downsample = resize_layer in downsampling_layers
        if normalize:
            norm_layer = norm_layers[norm_layer]
        layers = []
        if normalize:
            layers.append(norm_layer(in_channels))
        layers.append(activation())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias), eps=0.0001)])
        if normalize:
            layers.append(norm_layer(out_channels))
        layers.extend([activation(), spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias), eps=0.0001)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)
        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(spectral_norm(nn.Conv2d(in_channels, out_channels, 1), eps=0.0001))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class ResBlockV2Shuffle(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups, resize_layer, norm_layer, activation):
        super(ResBlockV2Shuffle, self).__init__()
        upsampling_layers = {'nearest': lambda : nn.Upsample(scale_factor=stride, mode='nearest')}
        downsampling_layers = {'avgpool': lambda : nn.AvgPool2d(stride)}
        norm_layers = {'bn': lambda num_features: SyncBatchNorm(num_features, momentum=1.0, eps=0.0001), 'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=0.0001, affine=True), 'adabn': lambda num_features: AdaptiveNorm2d(num_features, 'bn'), 'adain': lambda num_features: AdaptiveNorm2d(num_features, 'in')}
        normalize = norm_layer != 'none'
        bias = not normalize
        upsample = resize_layer in upsampling_layers
        downsample = resize_layer in downsampling_layers
        if normalize:
            norm_layer = norm_layers[norm_layer]
        layers = []
        if normalize:
            layers.append(norm_layer(in_channels))
        layers.append(activation())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([shuffleConv(in_channels, out_channels, 3, 1, 1, bias=bias)])
        if normalize:
            layers.append(norm_layer(out_channels))
        layers.extend([activation(), shuffleConv(out_channels, out_channels, 3, 1, 1, bias=bias)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)
        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(shuffleConv(in_channels, out_channels, 1))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class GatedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, act_fun, kernel_size, stride=1, padding=0, bias=True):
        super(GatedBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), eps=0.0001)
        self.gate = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), eps=0.0001)
        self.act_fun = act_fun()
        self.gate_act_fun = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.act_fun(out)
        mask = self.gate(x)
        mask = self.gate_act_fun(mask)
        out_masked = out * mask
        return out_masked


class GatedResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding, upsample, downsample, norm_layer, activation=nn.ReLU):
        super(GatedResBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=0.0001, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=0.0001, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')
        main_layers = []
        if normalize:
            main_layers.append(norm0)
        if upsample:
            main_layers.append(nn.Upsample(scale_factor=2))
        main_layers.extend([padding(1), GatedBlock(in_channels, out_channels, activation, 3, 1, 0, bias=bias)])
        if normalize:
            main_layers.append(norm1)
        main_layers.extend([padding(1), GatedBlock(out_channels, out_channels, activation, 3, 1, 0, bias=bias)])
        if downsample:
            main_layers.append(nn.AvgPool2d(2))
        self.main_pipe = nn.Sequential(*main_layers)
        self.skip_pipe = None
        if in_channels != out_channels or upsample or downsample:
            skip_layers = []
            if upsample:
                skip_layers.append(nn.Upsample(scale_factor=2))
            skip_layers.append(GatedBlock(in_channels, out_channels, activation, 1))
            if downsample:
                skip_layers.append(nn.AvgPool2d(2))
            self.skip_pipe = nn.Sequential(*skip_layers)

    def forward(self, input):
        mp_out = self.main_pipe(input)
        if self.skip_pipe is not None:
            output = mp_out + self.skip_pipe(input)
        else:
            output = mp_out + input
        return output


class ResBlockWithoutSpectralNorms(nn.Module):

    def __init__(self, in_channels, out_channels, padding, upsample, downsample, norm_layer, activation=nn.ReLU):
        super(ResBlockWithoutSpectralNorms, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize
        if norm_layer == 'in':
            norm0 = nn.InstanceNorm2d(in_channels, eps=0.0001, affine=True)
            norm1 = nn.InstanceNorm2d(out_channels, eps=0.0001, affine=True)
        elif 'ada' in norm_layer:
            norm0 = AdaptiveNorm2d(in_channels, norm_layer)
            norm1 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm0 = AdaptiveNorm2dTrainable(in_channels, norm_layer)
            norm1 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        elif normalize:
            raise Exception('ResBlock: Incorrect `norm_layer` parameter')
        layers = []
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.extend([padding(1), nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=bias)])
        if normalize:
            layers.append(norm1)
        layers.extend([activation(inplace=True), padding(1), nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=bias)])
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)
        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class MobileNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding, upsample, downsample, norm_layer, activation=nn.ReLU6, expansion_factor=6):
        super(MobileNetBlock, self).__init__()
        normalize = norm_layer != 'none'
        bias = not normalize
        conv0 = nn.Conv2d(in_channels, int(in_channels * expansion_factor), 1)
        dwise = nn.Conv2d(int(in_channels * expansion_factor), int(in_channels * expansion_factor), 3, 2 if downsample else 1, 1, groups=int(in_channels * expansion_factor))
        conv1 = nn.Conv2d(int(in_channels * expansion_factor), out_channels, 1)
        if norm_layer == 'bn':
            pass
        if 'in' in norm_layer:
            norm0 = nn.InstanceNorm2d(int(in_channels * expansion_factor), eps=0.0001, affine=True)
            norm1 = nn.InstanceNorm2d(int(in_channels * expansion_factor), eps=0.0001, affine=True)
            norm2 = nn.InstanceNorm2d(out_channels, eps=0.0001, affine=True)
        if 'ada' in norm_layer:
            norm2 = AdaptiveNorm2d(out_channels, norm_layer)
        elif 'tra' in norm_layer:
            norm2 = AdaptiveNorm2dTrainable(out_channels, norm_layer)
        layers = [conv0]
        if normalize:
            layers.append(norm0)
        layers.append(activation(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
        layers.append(dwise)
        if normalize:
            layers.append(norm1)
        layers.extend([activation(inplace=True), conv1])
        if normalize:
            layers.append(norm2)
        self.block = nn.Sequential(*layers)
        self.skip = None
        if in_channels != out_channels or upsample or downsample:
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
            if downsample:
                layers.append(nn.AvgPool2d(2))
            self.skip = nn.Sequential(*layers)

    def forward(self, input):
        out = self.block(input)
        if self.skip is not None:
            output = out + self.skip(input)
        else:
            output = out + input
        return output


class SelfAttention(nn.Module):

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        b, c, h, w = input.shape
        query = self.query_conv(input).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(input).view(b, -1, h * w)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(input).view(b, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        output = self.gamma * out + input
        return output


class Meter:
    """
    Tracks average and last values of several metrics.
    """

    def __init__(self):
        super().__init__()
        self.sum = defaultdict(float)
        self.num_measurements = defaultdict(int)
        self.last_value = {}

    def add(self, name, value, num_measurements=1):
        """
        Add `num_measurements` measurements for metric `name`, given their average (`value`).
        To add just one measurement, call with `num_measurements = 1` (default).

        name:
            `str`
        value:
            convertible to `float`
        num_measurements:
            `int`
        """
        assert num_measurements >= 0
        if num_measurements == 0:
            return
        value = float(value)
        if value != value:
            self.sum[name] += 0
            self.num_measurements[name] += 0
        else:
            self.sum[name] += value * num_measurements
            self.num_measurements[name] += num_measurements
        self.last_value[name] = value

    def keys(self):
        return self.sum.keys()

    def get_average(self, name):
        return self.sum[name] / max(1, self.num_measurements[name])

    def get_last(self, name):
        return self.last_value[name]

    def get_num_measurements(self, name):
        return self.num_measurements[name]

    def __iadd__(self, other_meter):
        for name in other_meter.sum:
            self.add(name, other_meter.get_average(name), other_meter.get_num_measurements(name))
            self.last_value[name] = other_meter.last_value[name]
        return self


logger = logging.getLogger('tensorboard_logging')


class TrainingModule(torch.nn.Module):

    def __init__(self, embedder, generator, discriminator, criterion_list, metric_list, running_averages={}):
        """
            `embedder`, `generator`, `discriminator`: `nn.Module`s
            `criterion_list`, `metric_list`: a list of `nn.Module`s
            `running_averages`: `None` or a dict of {`str`: `nn.Module`}
                Optional initial states of weights' running averages (useful when resuming training).
                Can provide running averages just for some modules (e.g. for none or only for generator).
                If `None`, don't track running averages at all.
        """
        super().__init__()
        self.embedder = embedder
        self.generator = generator
        self.discriminator = discriminator
        self.criterion_list = nn.ModuleList(criterion_list)
        self.metric_list = nn.ModuleList(metric_list)
        self.compute_losses = True
        self.use_running_averages = False
        self.initialize_running_averages(running_averages)

    def initialize_running_averages(self, initial_values={}):
        """
            Set up weights' running averages for generator and discriminator.

            initial_values: `dict` of `nn.Module`, or `None`
                `None` means do not use running averages at all.

                Otherwise, `initial_values['embedder']` will be used as
                the initla value for embedder's running average. Same for generator.
                If `initial_values['embedder']` is missing, then embedder's running average
                will be initialized to embedder's current weights.
        """
        self.running_averages = {}
        if initial_values is not None:
            for name in ('embedder', 'generator'):
                model = getattr(self, name)
                self.running_averages[name] = copy.deepcopy(model)
                try:
                    initial_value = initial_values[name]
                    self.running_averages[name].load_state_dict(initial_value)
                except KeyError:
                    logger.info(f"No initial value of weights' running averages provided for {name}. Initializing by cloning")
                except:
                    logger.warning(f"Parameters mismatch in {name} and the initial value of weights' running averages. Initializing by cloning")
                    self.running_averages[name].load_state_dict(model.state_dict())
        for module in self.running_averages.values():
            module.eval()
            module.requires_grad_(False)

    def update_running_average(self, alpha=0.999):
        with torch.no_grad():
            for model_name, model_running_avg in self.running_averages.items():
                model_current = getattr(self, model_name)
                for p_current, p_running_average in zip(model_current.parameters(), model_running_avg.parameters()):
                    p_running_average *= alpha
                    p_running_average += p_current * (1 - alpha)
                for p_current, p_running_average in zip(model_current.buffers(), model_running_avg.buffers()):
                    p_running_average.copy_(p_current)

    def set_use_running_averages(self, use_running_averages=True):
        """
            Changes `training_module.use_running_averages` to the specified value.
            Can be used either as a context manager or as a separate method call.

            TODO: migrate to contextlib
        """


        class UseRunningAveragesContextManager:

            def __init__(self, training_module, use_running_averages):
                self.training_module = training_module
                self.old_value = self.training_module.use_running_averages
                self.training_module.use_running_averages = use_running_averages

            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.training_module.use_running_averages = self.old_value
        return UseRunningAveragesContextManager(self, use_running_averages)

    def set_compute_losses(self, compute_losses=True):
        """
            Changes `training_module.compute_losses` to the specified value.
            Can be used either as a context manager or as a separate method call.

            TODO: migrate to contextlib
        """


        class ComputeLossesContextManager:

            def __init__(self, training_module, compute_losses):
                self.training_module = training_module
                self.old_value = self.training_module.compute_losses
                self.training_module.compute_losses = compute_losses

            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.training_module.compute_losses = self.old_value
        return ComputeLossesContextManager(self, compute_losses)

    def forward(self, data_dict, target_dict):
        if self.running_averages and self.use_running_averages:
            embedder = self.running_averages['embedder']
            generator = self.running_averages['generator']
        else:
            embedder = self.embedder
            generator = self.generator
        data_dict = copy.copy(data_dict)
        embedder(data_dict)
        generator(data_dict)
        data_dict.update(target_dict)
        if self.compute_losses:
            self.discriminator(data_dict)
        losses_G_dict = {}
        losses_D_dict = {}
        for criterion in self.criterion_list:
            try:
                crit_out = criterion(data_dict)
            except:
                if self.compute_losses:
                    raise
                else:
                    continue
            if isinstance(crit_out, tuple):
                if len(crit_out) != 2:
                    raise TypeError(f'Unexpected number of outputs in criterion {type(criterion)}: expected 2, got {len(crit_out)}')
                crit_out_G, crit_out_D = crit_out
                losses_G_dict.update(crit_out_G)
                losses_D_dict.update(crit_out_D)
            elif isinstance(crit_out, dict):
                losses_G_dict.update(crit_out)
            else:
                raise TypeError(f'Unexpected type of {type(criterion)} output: expected dict or tuple of two dicts, got {type(crit_out)}')
        return data_dict, losses_G_dict, losses_D_dict

    def compute_metrics(self, data_dict):
        metrics_meter = Meter()
        for metric in self.metric_list:
            metric_out, num_errors = metric(data_dict)
            for metric_output_name, metric_value in metric_out.items():
                metrics_meter.add(metric_output_name, metric_value, num_errors[metric_output_name])
        return metrics_meter


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'act_fun': _mock_layer, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (channelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (shuffleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_shrubb_latent_pose_reenactment(_paritybench_base):
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

