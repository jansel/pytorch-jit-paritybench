import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
aligned_dataset_resized = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
single_dataset = _module
generate_masks = _module
models = _module
InnerFaceShiftTriple = _module
InnerFaceShiftTripleFunction = _module
face_shift_net = _module
face_shiftnet_model = _module
modules = _module
denset_net = _module
discrimators = _module
losses = _module
modules = _module
shift_unet = _module
unet = _module
networks = _module
patch_soft_shift = _module
innerPatchSoftShiftTriple = _module
innerPatchSoftShiftTripleModule = _module
patch_soft_shiftnet_model = _module
res_patch_soft_shift = _module
innerResPatchSoftShiftTriple = _module
res_patch_soft_shiftnet_model = _module
res_shift_net = _module
innerResShiftTriple = _module
shiftnet_model = _module
InnerCos = _module
InnerCosFunction = _module
InnerShiftTriple = _module
InnerShiftTripleFunction = _module
shift_net = _module
base_model = _module
shiftnet_model = _module
notebooks = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
show_map = _module
test = _module
test_acc_shift = _module
train = _module
NonparametricShift = _module
util = _module
html = _module
png = _module
poisson_blending = _module
util = _module
visualizer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch


from torch.nn import functional as F


import time


import torchvision.transforms as transforms


import numpy as np


import re


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import functools


from torch.nn import Parameter


from torch.nn import init


from torch.optim import lr_scheduler


from torchvision import models


import numpy as numpy


import random


import math


from time import time


import inspect


import collections


class Modified_NonparametricShift(object):

    def _extract_patches_from_flag(self, img, patch_size, stride, flag, value):
        input_windows = self._unfold(img, patch_size, stride)
        input_windows = self._filter(input_windows, flag, value)
        return self._norm(input_windows)

    def cosine_similarity(self, former, latter, patch_size, stride, flag,
        with_former=False):
        former_windows = self._unfold(former, patch_size, stride)
        former = self._filter(former_windows, flag, 1)
        latter_windows, i_2, i_3, i_1 = self._unfold(latter, patch_size,
            stride, with_indexes=True)
        latter = self._filter(latter_windows, flag, 0)
        num = torch.einsum('ik,jk->ij', [former, latter])
        norm_latter = torch.einsum('ij,ij->i', [latter, latter])
        norm_former = torch.einsum('ij,ij->i', [former, former])
        den = torch.sqrt(torch.einsum('i,j->ij', [norm_former, norm_latter]))
        if not with_former:
            return num / den, latter_windows, i_2, i_3, i_1
        else:
            return num / den, latter_windows, former_windows, i_2, i_3, i_1

    def _paste(self, input_windows, transition_matrix, i_2, i_3, i_1):
        input_windows = torch.mm(transition_matrix, input_windows)
        input_windows = input_windows.view(i_2, i_3, i_1)
        input_windows = input_windows.permute(2, 0, 1).unsqueeze(0)
        return input_windows

    def _unfold(self, img, patch_size, stride, with_indexes=False):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = input_windows.size()
        if with_indexes:
            input_windows = input_windows.permute(1, 2, 0, 3, 4).contiguous(
                ).view(i_2 * i_3, i_1)
            return input_windows, i_2, i_3, i_1
        else:
            input_windows = input_windows.permute(1, 2, 0, 3, 4).contiguous(
                ).view(i_2 * i_3, i_1, i_4, i_5)
            return input_windows

    def _filter(self, input_windows, flag, value):
        input_window = input_windows[flag == value]
        return input_window.view(input_window.size(0), -1)

    def _norm(self, input_window):
        for i in range(input_window.size(0)):
            input_window[i] = input_window[i] * (1 / (input_window[i].norm(
                2) + 1e-08))
        return input_window


class InnerFaceShiftTripleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, shift_sz, stride, triple_w, flag, flag_flip,
        show_flow, flip_feat=None):
        InnerFaceShiftTripleFunction.ctx = ctx
        assert input.dim() == 4, 'Input Dim has to be 4'
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.flag_flip = flag_flip
        ctx.show_flow = show_flow
        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real
        ctx.ind_lst = torch.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_(
            ).to(input)
        ctx.ind_lst_flip = ctx.ind_lst.clone()
        former_all = input.narrow(1, 0, c // 2)
        latter_all = input.narrow(1, c // 2, c // 2)
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all
            ).zero_()
        if not flip_feat is None:
            assert flip_feat.size() == former_all.size(
                ), 'flip_feat size should be equal to former size'
            ctx.flag = ctx.flag.to(input).long()
            ctx.flag_flip = ctx.flag_flip.to(input).long()
            Nonparm = Modified_NonparametricShift()
            ctx.shift_offsets = []
            for idx in range(ctx.bz):
                flag_cur = ctx.flag[idx]
                flag_cur_flip = ctx.flag_flip[idx]
                latter = latter_all.narrow(0, idx, 1)
                former = former_all.narrow(0, idx, 1)
                cosine, latter_windows, i_2, i_3, i_1 = (Nonparm.
                    cosine_similarity(former.clone().squeeze(), latter.
                    clone().squeeze(), 1, stride, flag_cur))
                cosine_flip, latter_windows_flip, _, _, _ = (Nonparm.
                    cosine_similarity(former.clone().squeeze(), flip_feat.
                    clone().squeeze(), 1, stride, flag_cur_flip))
                cosine_con = torch.cat([cosine, cosine_flip], dim=1)
                _, indexes_con = torch.max(cosine_con, dim=1)
                ori_larger = (indexes_con < cosine.size(1)).long().view(-1, 1)
                _, indexes = torch.max(cosine, dim=1)
                _, indexes_flip = torch.max(cosine_flip, dim=1)
                mask_indexes = (flag_cur == 1).nonzero()
                non_mask_indexes = (flag_cur == 0).nonzero()[indexes]
                mask_indexes_select_index = (mask_indexes.squeeze() *
                    ori_larger.squeeze()).nonzero()
                mask_indexes_select = mask_indexes[mask_indexes_select_index
                    ].squeeze()
                ctx.ind_lst[idx][mask_indexes_select, non_mask_indexes] = 1
                non_mask_indexes_flip = (flag_cur_flip == 0).nonzero()[
                    indexes_flip]
                mask_indexes_flip_select_index = (mask_indexes.squeeze() *
                    (1 - ori_larger.squeeze())).nonzero()
                mask_indexes_flip_select = mask_indexes[
                    mask_indexes_flip_select_index].squeeze()
                ctx.ind_lst_flip[idx][mask_indexes_flip_select,
                    non_mask_indexes_flip] = 1
                ori_tmp = Nonparm._paste(latter_windows, ctx.ind_lst[idx],
                    i_2, i_3, i_1)
                ori_tmp_flip = Nonparm._paste(latter_windows_flip, ctx.
                    ind_lst_flip[idx], i_2, i_3, i_1)
                shift_masked_all[idx] = ori_tmp + ori_tmp_flip
                if ctx.show_flow:
                    shift_offset = torch.stack([non_mask_indexes.squeeze() //
                        ctx.w, non_mask_indexes.squeeze() % ctx.w], dim=-1)
                    ctx.shift_offsets.append(shift_offset)
        if ctx.show_flow:
            ctx.shift_offsets = torch.cat(ctx.shift_offsets, dim=0).float()
            mask_nums = ctx.shift_offsets.size(0) // ctx.bz
            ctx.flow_srcs = torch.zeros(ctx.bz, 3, ctx.h, ctx.w).type_as(input)
            for idx in range(ctx.bz):
                shift_offset = ctx.shift_offsets.narrow(0, idx * mask_nums,
                    mask_nums)
                shift_offsets_map = torch.zeros(1, ctx.h, ctx.w, 2).type_as(
                    input)
                shift_offsets_map[:, ((flag_cur == 1).nonzero().squeeze() //
                    ctx.w), ((flag_cur == 1).nonzero().squeeze() % ctx.w), :
                    ] = shift_offset.unsqueeze(0)
                flow_src = util.highlight_flow(shift_offsets_map, flag_cur.
                    unsqueeze(0))
                ctx.flow_srcs[idx] = flow_src
        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    @staticmethod
    def get_flow_src():
        return InnerFaceShiftTripleFunction.ctx.flow_srcs

    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst
        ind_lst_flip = ctx.ind_lst_flip
        c = grad_output.size(1)
        grad_former_all = grad_output[:, 0:c // 3, :, :]
        grad_latter_all = grad_output[:, c // 3:c * 2 // 3, :, :].clone()
        grad_shifted_all = grad_output[:, c * 2 // 3:c, :, :].clone()
        for idx in range(ctx.bz):
            W_mat_t = ind_lst[idx].t()
            W_mat_t_flip = ind_lst_flip[idx].t()
            grad = grad_shifted_all[idx].view(c // 3, -1).t()
            grad_shifted_weighted = torch.mm(W_mat_t, grad)
            grad_shifted_weighted_flip = torch.mm(W_mat_t_flip, grad)
            grad_shifted_weighted = grad_shifted_weighted.t().contiguous(
                ).view(1, c // 3, ctx.h, ctx.w)
            grad_shifted_weighted_flip = grad_shifted_weighted_flip.t(
                ).contiguous().view(1, c // 3, ctx.h, ctx.w)
            grad_shifted_weighted_all = (grad_shifted_weighted +
                grad_shifted_weighted_flip)
            grad_latter_all[idx] = torch.add(grad_latter_all[idx],
                grad_shifted_weighted_all.mul(ctx.triple_w))
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)
        return grad_input, None, None, None, None, None, None, None


class InnerFaceShiftTriple(nn.Module):

    def __init__(self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1,
        layer_to_last=3, device='gpu'):
        super(InnerFaceShiftTriple, self).__init__()
        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.layer_to_last = layer_to_last
        self.device = device
        self.show_flow = False
        self.flow_srcs = None

    def set_mask(self, mask_global):
        self.mask_all = util.cal_feat_mask(mask_global, self.layer_to_last)

    def _split_mask(self, cur_bsize):
        cur_device = torch.current_device()
        self.cur_mask = self.mask_all[cur_device * cur_bsize:(cur_device + 
            1) * cur_bsize, :, :, :]

    def forward(self, input, flip_feat=None):
        self.bz, self.c, self.h, self.w = input.size()
        if self.device != 'cpu':
            self._split_mask(self.bz)
        else:
            self.cur_mask = self.mask_all
        self.mask = self.cur_mask
        self.mask_flip = torch.flip(self.mask, [3])
        self.flag = util.cal_flag_given_mask_thred(self.mask, self.shift_sz,
            self.stride, self.mask_thred)
        self.flag_flip = util.cal_flag_given_mask_thred(self.mask_flip,
            self.shift_sz, self.stride, self.mask_thred)
        final_out = InnerFaceShiftTripleFunction.apply(input, self.shift_sz,
            self.stride, self.triple_weight, self.flag, self.flag_flip,
            self.show_flow, flip_feat)
        if self.show_flow:
            self.flow_srcs = InnerFaceShiftTripleFunction.get_flow_src()
        innerFeat = input.clone().narrow(1, self.c // 2, self.c // 2)
        return final_out, innerFeat

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__ + '(' + ' ,triple_weight ' + str(self
            .triple_weight) + ')'


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
        use_spectral_norm):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU()),
        self.add_module('conv1', spectral_norm(nn.Conv2d(num_input_features,
            bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            use_spectral_norm)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU()),
        self.add_module('conv2', spectral_norm(nn.Conv2d(bn_size *
            growth_rate, growth_rate, kernel_size=3, stride=1, padding=1,
            bias=False), use_spectral_norm)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, use_spectral_norm):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate, use_spectral_norm)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features,
        use_spectral_norm):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', spectral_norm(nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False),
            use_spectral_norm))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
        use_spectral_norm=True, num_init_features=64, bn_size=4, drop_rate=
        0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', spectral_norm(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
            padding=3, bias=False), use_spectral_norm)), ('norm0', nn.
            BatchNorm2d(num_init_features)), ('relu0', nn.ReLU()), ('pool0',
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate, use_spectral_norm=use_spectral_norm)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2,
                    use_spectral_norm=use_spectral_norm)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.conv_last = spectral_norm(nn.Conv2d(num_features, 256,
            kernel_size=3), use_spectral_norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        features = self.conv_last(features)
        return features


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False, use_spectral_norm=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
            stride=2, padding=padw), use_spectral_norm), nn.LeakyReLU(0.2, 
            True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf *
                nf_mult, kernel_size=kw, stride=2, padding=padw, bias=
                use_bias), use_spectral_norm), norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf *
            nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            use_spectral_norm), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2,
            True)]
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=
            kw, stride=1, padding=padw), use_spectral_norm)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


model_urls = {'densenet121':
    'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169':
    'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201':
    'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161':
    'https://download.pytorch.org/models/densenet161-8d451a50.pth'}


def densenet121(pretrained=False, use_spectral_norm=True, **kwargs):
    """Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6,
        12, 24, 16), use_spectral_norm=use_spectral_norm, **kwargs)
    if pretrained:
        pattern = re.compile(
            '^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$'
            )
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


class DenseNetDiscrimator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False, use_spectral_norm=True):
        super(DenseNetDiscrimator, self).__init__()
        self.model = densenet121(pretrained=True, use_spectral_norm=
            use_spectral_norm)
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.use_sigmoid:
            return self.sigmoid(self.model(input))
        else:
            return self.model(input)


class GANLoss(nn.Module):

    def __init__(self, gan_type='wgan_gp', target_real_label=1.0,
        target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_type = gan_type
        if gan_type == 'wgan_gp':
            self.loss = nn.MSELoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        elif gan_type == 're_s_gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 're_avg_gan':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('GAN type [%s] not recognized.' % gan_type)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_type == 'wgan_gp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss


def spatial_discounting_mask(mask_width, mask_height, discounting_gamma,
    discounting=1):
    """Generate spatial discounting mask constant.
    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Returns:
        tf.Tensor: spatial discounting mask
    """
    gamma = discounting_gamma
    shape = [1, 1, mask_width, mask_height]
    if discounting:
        print('Use spatial discounting l1 loss.')
        mask_values = np.ones((mask_width, mask_height), dtype='float32')
        for i in range(mask_width):
            for j in range(mask_height):
                mask_values[i, j] = max(gamma ** min(i, mask_width - i), 
                    gamma ** min(j, mask_height - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 1)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape, dtype='float32')
    return mask_values


class Discounted_L1(nn.Module):

    def __init__(self, opt):
        super(Discounted_L1, self).__init__()
        self.register_buffer('discounting_mask', torch.tensor(
            spatial_discounting_mask(opt.fineSize // 2 - opt.overlap * 2, 
            opt.fineSize // 2 - opt.overlap * 2, 0.9, opt.discounting)))
        self.L1 = nn.L1Loss()

    def forward(self, input, target):
        self._assert_no_grad(target)
        input_tmp = input * self.discounting_mask
        target_tmp = target * self.discounting_mask
        return self.L1(input_tmp, target_tmp)

    def _assert_no_grad(self, variable):
        assert not variable.requires_grad, "nn criterions don't compute the gradient w.r.t. targets - please mark these variables as volatile or not requiring gradients"


class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        bz, _, h, w = x.size()
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w - 1], 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / bz

    @staticmethod
    def _tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    """
	https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
	"""

    def __init__(self, in_dim, activation, with_attention=False):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attention = with_attention
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
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        if self.with_attention:
            return out, attention
        else:
            return out


class SwitchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9,
        using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1,
                num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1)
                )
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1
                ] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[
                2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class PartialConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim), nn.ReLU(True)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=
            use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGeneratorShiftTriple(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list,
        shift_list, mask_global, ngf=64, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False):
        super(UnetGeneratorShiftTriple, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True,
            use_spectral_norm=use_spectral_norm)
        None
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_shift_block = UnetSkipConnectionShiftBlock(ngf * 2, ngf * 4,
            opt, innerCos_list, shift_list, mask_global, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=
            use_spectral_norm, layer_to_last=3)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_shift_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer, use_spectral_norm=use_spectral_norm)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionShiftBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list,
        mask_global, input_nc, submodule=None, shift_layer=None, outermost=
        False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False, layer_to_last=3):
        super(UnetSkipConnectionShiftBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=
            4, stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        shift = InnerShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
            opt.triple_weight, layer_to_last=layer_to_last, device=device)
        shift.set_mask(mask_global)
        shift_list.append(shift)
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip,
            layer_to_last=layer_to_last, device=device)
        innerCos.set_mask(mask_global)
        innerCos_list.append(innerCos)
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                kernel_size=4, stride=2, padding=1), use_spectral_norm)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 3,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, innerCos, shift, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)


class FaceUnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, innerCos_list, shift_list,
        mask_global, opt, ngf=64, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False):
        super(FaceUnetGenerator, self).__init__()
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf * 2, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e2_norm = norm_layer(ngf * 2)
        self.e3_c = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=6,
            stride=2, padding=2), use_spectral_norm)
        self.e3_norm = norm_layer(ngf * 4)
        self.e4_c = spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e4_norm = norm_layer(ngf * 8)
        self.e5_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e5_norm = norm_layer(ngf * 8)
        self.e6_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e6_norm = norm_layer(ngf * 8)
        self.e7_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e7_norm = norm_layer(ngf * 8)
        self.e8_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.d1_dc = spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d1_norm = norm_layer(ngf * 8)
        self.d2_dc = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d2_norm = norm_layer(ngf * 8)
        self.d3_dc = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d3_norm = norm_layer(ngf * 8)
        self.d4_dc = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d4_norm = norm_layer(ngf * 8)
        self.d5_dc = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d5_norm = norm_layer(ngf * 4)
        self.d6_dc = spectral_norm(nn.ConvTranspose2d(ngf * 4 * 3, ngf * 2,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d6_norm = norm_layer(ngf * 2)
        self.d7_dc = spectral_norm(nn.ConvTranspose2d(ngf * 2 * 2, ngf,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d7_norm = norm_layer(ngf)
        self.d8_dc = spectral_norm(nn.ConvTranspose2d(ngf * 2, output_nc,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        self.shift = InnerFaceShiftTriple(opt.shift_sz, opt.stride, opt.
            mask_thred, opt.triple_weight, layer_to_last=3, device=device)
        self.shift.set_mask(mask_global)
        shift_list.append(self.shift)
        self.innerCos = InnerCos(strength=opt.strength, skip=opt.skip,
            layer_to_last=3, device=device)
        self.innerCos.set_mask(mask_global)
        innerCos_list.append(self.innerCos)

    def forward(self, input, flip_feat=None):
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))
        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))
        d1 = self.d1_norm(self.d1_dc(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_dc(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_dc(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_dc(F.relu_(torch.cat([d3, e5], dim=1))))
        d5 = self.d5_norm(self.d5_dc(F.relu_(torch.cat([d4, e4], dim=1))))
        tmp, innerFeat = self.shift(self.innerCos(F.relu_(torch.cat([d5, e3
            ], dim=1))), flip_feat)
        d6 = self.d6_norm(self.d6_dc(tmp))
        d7 = self.d7_norm(self.d7_dc(F.relu_(torch.cat([d6, e2], dim=1))))
        d8 = self.d8_dc(F.relu_(torch.cat([d7, e1], 1)))
        d8 = torch.tanh(d8)
        return d8, innerFeat


class ResUnetGeneratorShiftTriple(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list,
        shift_list, mask_global, ngf=64, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False):
        super(ResUnetGeneratorShiftTriple, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True,
            use_spectral_norm=use_spectral_norm)
        None
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_shift_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, opt,
            innerCos_list, shift_list, mask_global, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=
            use_spectral_norm, layer_to_last=3)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_shift_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer, use_spectral_norm=use_spectral_norm)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class ResUnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list,
        mask_global, input_nc, submodule=None, shift_layer=None, outermost=
        False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False, layer_to_last=3):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=
            4, stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        shift = InnerResShiftTriple(inner_nc, opt.shift_sz, opt.stride, opt
            .mask_thred, opt.triple_weight, layer_to_last=layer_to_last,
            device=device)
        shift.set_mask(mask_global)
        shift_list.append(shift)
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip,
            layer_to_last=layer_to_last, device=device)
        innerCos.set_mask(mask_global)
        innerCos_list.append(innerCos)
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                kernel_size=4, stride=2, padding=1), use_spectral_norm)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, innerCos, shift, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)


class PatchSoftUnetGeneratorShiftTriple(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list,
        shift_list, mask_global, ngf=64, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False):
        super(PatchSoftUnetGeneratorShiftTriple, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True,
            use_spectral_norm=use_spectral_norm)
        None
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_shift_block = PatchSoftUnetSkipConnectionShiftTriple(ngf * 2, 
            ngf * 4, opt, innerCos_list, shift_list, mask_global, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm, layer_to_last=3)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_shift_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer, use_spectral_norm=use_spectral_norm)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class PatchSoftUnetSkipConnectionShiftTriple(nn.Module):

    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list,
        mask_global, input_nc, submodule=None, shift_layer=None, outermost=
        False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False, layer_to_last=3):
        super(PatchSoftUnetSkipConnectionShiftTriple, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=
            4, stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        shift = InnerPatchSoftShiftTriple(opt.shift_sz, opt.stride, opt.
            mask_thred, opt.triple_weight, opt.fuse, layer_to_last=
            layer_to_last, device=device)
        shift.set_mask(mask_global)
        shift_list.append(shift)
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip,
            layer_to_last=layer_to_last, device=device)
        innerCos.set_mask(mask_global)
        innerCos_list.append(innerCos)
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                kernel_size=4, stride=2, padding=1), use_spectral_norm)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 3,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, innerCos, shift, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)


class ResPatchSoftUnetGeneratorShiftTriple(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list,
        shift_list, mask_global, ngf=64, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False):
        super(ResPatchSoftUnetGeneratorShiftTriple, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True,
            use_spectral_norm=use_spectral_norm)
        None
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_shift_block = ResPatchSoftUnetSkipConnectionShiftTriple(ngf * 
            2, ngf * 4, opt, innerCos_list, shift_list, mask_global,
            input_nc=None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm, layer_to_last=3)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_shift_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer, use_spectral_norm=use_spectral_norm)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class ResPatchSoftUnetSkipConnectionShiftTriple(nn.Module):

    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list,
        mask_global, input_nc, submodule=None, shift_layer=None, outermost=
        False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False, layer_to_last=3):
        super(ResPatchSoftUnetSkipConnectionShiftTriple, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=
            4, stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        shift = InnerResPatchSoftShiftTriple(inner_nc, opt.shift_sz, opt.
            stride, opt.mask_thred, opt.triple_weight, opt.fuse,
            layer_to_last=layer_to_last, device=device)
        shift.set_mask(mask_global)
        shift_list.append(shift)
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip,
            layer_to_last=layer_to_last, device=device)
        innerCos.set_mask(mask_global)
        innerCos_list.append(innerCos)
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                kernel_size=4, stride=2, padding=1), use_spectral_norm)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, innerCos, shift, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=
        nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True,
            use_spectral_norm=use_spectral_norm)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=
            use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer, use_spectral_norm=use_spectral_norm)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_spectral_norm=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=
            4, stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                kernel_size=4, stride=2, padding=1), use_spectral_norm)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2,
                outer_nc, kernel_size=4, stride=2, padding=1),
                use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)


class EasyUnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_spectral_norm=False):
        super(EasyUnetGenerator, self).__init__()
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf * 2, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e2_norm = norm_layer(ngf * 2)
        self.e3_c = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e3_norm = norm_layer(ngf * 4)
        self.e4_c = spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e4_norm = norm_layer(ngf * 8)
        self.e5_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e5_norm = norm_layer(ngf * 8)
        self.e6_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e6_norm = norm_layer(ngf * 8)
        self.e7_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.e7_norm = norm_layer(ngf * 8)
        self.e8_c = spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
            stride=2, padding=1), use_spectral_norm)
        self.d1_c = spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d1_norm = norm_layer(ngf * 8)
        self.d2_c = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d2_norm = norm_layer(ngf * 8)
        self.d3_c = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d3_norm = norm_layer(ngf * 8)
        self.d4_c = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d4_norm = norm_layer(ngf * 8)
        self.d5_c = spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d5_norm = norm_layer(ngf * 4)
        self.d6_c = spectral_norm(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d6_norm = norm_layer(ngf * 2)
        self.d7_c = spectral_norm(nn.ConvTranspose2d(ngf * 2 * 2, ngf,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d7_norm = norm_layer(ngf)
        self.d8_c = spectral_norm(nn.ConvTranspose2d(ngf * 2, output_nc,
            kernel_size=4, stride=2, padding=1), use_spectral_norm)

    def forward(self, input):
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))
        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))
        d1 = self.d1_norm(self.d1_c(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_c(F.relu_(torch.cat([d3, e5], dim=1))))
        d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4], dim=1))))
        d6 = self.d6_norm(self.d6_c(F.relu_(torch.cat([d5, e3], dim=1))))
        d7 = self.d7_norm(self.d7_c(F.relu_(torch.cat([d6, e2], dim=1))))
        d8 = self.d8_c(F.relu_(torch.cat([d7, e1], 1)))
        d8 = torch.tanh(d8)
        return d8


class InnerPatchSoftShiftTriple(nn.Module):

    def __init__(self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1,
        fuse=True, layer_to_last=3):
        super(InnerPatchSoftShiftTriple, self).__init__()
        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.show_flow = False
        self.flow_srcs = None
        self.fuse = fuse
        self.layer_to_last = layer_to_last
        self.softShift = InnerPatchSoftShiftTripleModule()

    def set_mask(self, mask_global):
        mask = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask = mask
        return self.mask

    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        final_out = self.softShift(input, self.stride, self.triple_weight,
            self.mask, self.mask_thred, self.shift_sz, self.show_flow, self
            .fuse)
        if self.show_flow:
            self.flow_srcs = self.softShift.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__ + '(' + ' ,triple_weight ' + str(self
            .triple_weight) + ')'


class InnerPatchSoftShiftTripleModule(nn.Module):

    def forward(self, input, stride, triple_w, mask, mask_thred, shift_sz,
        show_flow, fuse=True):
        assert input.dim() == 4, 'Input Dim has to be 4'
        assert mask.dim() == 4, 'Mask Dim has to be 4'
        self.triple_w = triple_w
        self.mask = mask
        self.mask_thred = mask_thred
        self.show_flow = show_flow
        self.bz, self.c, self.h, self.w = input.size()
        self.Tensor = (torch.FloatTensor if torch.is_available else torch.
            FloatTensor)
        self.ind_lst = self.Tensor(self.bz, self.h * self.w, self.h * self.w
            ).zero_()
        former_all = input.narrow(1, 0, self.c // 2)
        latter_all = input.narrow(1, self.c // 2, self.c // 2)
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all)
        self.mask = self.mask
        latter_all_pad = F.pad(latter_all, [shift_sz // 2, shift_sz // 2, 
            shift_sz // 2, shift_sz // 2], 'constant', 0)
        latter_all_windows = latter_all_pad.unfold(2, shift_sz, stride).unfold(
            3, shift_sz, stride)
        latter_all_windows = latter_all_windows.contiguous().view(self.bz, 
            -1, self.c // 2, shift_sz, shift_sz)
        m_pad = F.pad(self.mask, (shift_sz // 2, shift_sz // 2, shift_sz //
            2, shift_sz // 2), 'constant', 0)
        m = m_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
        m = m.contiguous().view(self.bz, 1, -1, shift_sz, shift_sz)
        m = torch.mean(torch.mean(m, dim=3, keepdim=True), dim=4, keepdim=True)
        mm = m.le(self.mask_thred / (1.0 * shift_sz ** 2)).float()
        fuse_weight = torch.eye(shift_sz).view(1, 1, shift_sz, shift_sz
            ).type_as(input)
        self.shift_offsets = []
        for idx in range(self.bz):
            mm_cur = mm[idx]
            latter_win = latter_all_windows.narrow(0, idx, 1)[0]
            former = former_all.narrow(0, idx, 1)
            latter_den = torch.sqrt(torch.einsum('bcij,bcij->b', [
                latter_win, latter_win]))
            latter_den = torch.max(latter_den, self.Tensor([0.0001]))
            latter_win_normed = latter_win / latter_den.view(-1, 1, 1, 1)
            y_i = F.conv2d(former, latter_win_normed, stride=1, padding=
                shift_sz // 2)
            if fuse:
                y_i = y_i.view(1, 1, self.h * self.w, self.h * self.w)
                y_i = F.conv2d(y_i, fuse_weight, stride=1, padding=1)
                y_i = y_i.contiguous().view(1, self.h, self.w, self.h, self.w)
                y_i = y_i.permute(0, 2, 1, 4, 3)
                y_i = y_i.contiguous().view(1, 1, self.h * self.w, self.h *
                    self.w)
                y_i = F.conv2d(y_i, fuse_weight, stride=1, padding=1)
                y_i = y_i.contiguous().view(1, self.w, self.h, self.w, self.h)
                y_i = y_i.permute(0, 2, 1, 4, 3)
            y_i = y_i.contiguous().view(1, self.h * self.w, self.h, self.w)
            y_i = y_i * mm_cur
            cosine = F.softmax(y_i * 10, dim=1)
            cosine = cosine * mm_cur
            shift_i = F.conv_transpose2d(cosine, latter_win, stride=1,
                padding=shift_sz // 2) / 9.0
            shift_masked_all[idx] = shift_i
        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    def get_flow_src(self):
        return self.flow_srcs


class InnerResPatchSoftShiftTriple(nn.Module):

    def __init__(self, inner_nc, shift_sz=1, stride=1, mask_thred=1,
        triple_weight=1, fuse=True, layer_to_last=3):
        super(InnerResPatchSoftShiftTriple, self).__init__()
        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.show_flow = False
        self.flow_srcs = None
        self.fuse = fuse
        self.layer_to_last = layer_to_last
        self.softShift = InnerPatchSoftShiftTripleModule()
        self.inner_nc = inner_nc
        self.res_net = nn.Sequential(nn.Conv2d(inner_nc * 2, inner_nc,
            kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(inner_nc
            ), nn.ReLU(True), nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
            stride=1, padding=1), nn.InstanceNorm2d(inner_nc))

    def set_mask(self, mask_global):
        mask = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask = mask
        return self.mask

    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        shift_out = self.softShift(input, self.stride, self.triple_weight,
            self.mask, self.mask_thred, self.shift_sz, self.show_flow, self
            .fuse)
        c_out = shift_out.size(1)
        F_c = shift_out.narrow(1, 0, c_out // 3)
        F_s = shift_out.narrow(1, c_out // 3, c_out // 3)
        F_shift = shift_out.narrow(1, c_out * 2 // 3, c_out // 3)
        F_fuse = F_c * F_shift
        F_com = torch.cat([F_c, F_fuse], dim=1)
        res_out = self.res_net(F_com)
        F_c = F_c + res_out
        final_out = torch.cat([F_c, F_s], dim=1)
        if self.show_flow:
            self.flow_srcs = self.softShift.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__ + '(' + ' ,triple_weight ' + str(self
            .triple_weight) + ')'


class Batch_NonShift(object):

    def _extract_patches_from_flag(self, img, patch_size, stride, flag, value):
        input_windows = self._unfold(img, patch_size, stride)
        input_windows = self._filter(input_windows, flag, value)
        return self._norm(input_windows)

    def cosine_similarity(self, former, latter, patch_size, stride, flag,
        with_former=False):
        former_windows = self._unfold(former, patch_size, stride)
        former = self._filter(former_windows, flag, 1)
        latter_windows, i_2, i_3, i_1 = self._unfold(latter, patch_size,
            stride, with_indexes=True)
        latter = self._filter(latter_windows, flag, 0)
        num = torch.einsum('bik,bjk->bij', [former, latter])
        norm_latter = torch.einsum('bij,bij->bi', [latter, latter])
        norm_former = torch.einsum('bij,bij->bi', [former, former])
        den = torch.sqrt(torch.einsum('bi,bj->bij', [norm_former, norm_latter])
            )
        if not with_former:
            return num / den, latter_windows, i_2, i_3, i_1
        else:
            return num / den, latter_windows, former_windows, i_2, i_3, i_1

    def _paste(self, input_windows, transition_matrix, i_2, i_3, i_1):
        bz = input_windows.size(0)
        input_windows = torch.bmm(transition_matrix, input_windows)
        input_windows = input_windows.view(bz, i_2, i_3, i_1)
        input_windows = input_windows.permute(0, 3, 1, 2)
        return input_windows

    def _unfold(self, img, patch_size, stride, with_indexes=False):
        n_dim = 4
        assert img.dim() == n_dim, 'image must be of dimension 4.'
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(2, kH, dH).unfold(3, kW, dW)
        i_0, i_1, i_2, i_3, i_4, i_5 = input_windows.size()
        if with_indexes:
            input_windows = input_windows.permute(0, 2, 3, 1, 4, 5).contiguous(
                ).view(i_0, i_2 * i_3, i_1)
            return input_windows, i_2, i_3, i_1
        else:
            input_windows = input_windows.permute(0, 2, 3, 1, 4, 5).contiguous(
                ).view(i_0, i_2 * i_3, i_1, i_4, i_5)
            return input_windows

    def _filter(self, input_windows, flag, value):
        assert flag.dim() == 2, 'flag should be batch version'
        input_window = input_windows[flag == value]
        bz = flag.size(0)
        return input_window.view(bz, input_window.size(0) // bz, -1)


class InnerShiftTripleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, shift_sz, stride, triple_w, flag, show_flow):
        InnerShiftTripleFunction.ctx = ctx
        assert input.dim() == 4, 'Input Dim has to be 4'
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.show_flow = show_flow
        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real
        ctx.ind_lst = torch.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_(
            ).to(input)
        former_all = input.narrow(1, 0, c // 2)
        latter_all = input.narrow(1, c // 2, c // 2)
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all
            ).zero_()
        ctx.flag = ctx.flag.to(input).long()
        bNonparm = Batch_NonShift()
        ctx.shift_offsets = []
        cosine, latter_windows, i_2, i_3, i_1 = bNonparm.cosine_similarity(
            former_all.clone(), latter_all.clone(), 1, stride, flag)
        _, indexes = torch.max(cosine, dim=2)
        mask_indexes = (flag == 1).nonzero()[:, (1)].view(ctx.bz, -1)
        non_mask_indexes = (flag == 0).nonzero()[:, (1)].view(ctx.bz, -1
            ).gather(1, indexes)
        idx_b = torch.arange(ctx.bz).long().unsqueeze(1).expand(ctx.bz,
            mask_indexes.size(1))
        ctx.ind_lst[idx_b, mask_indexes, non_mask_indexes] = 1
        shift_masked_all = bNonparm._paste(latter_windows, ctx.ind_lst, i_2,
            i_3, i_1)
        if ctx.show_flow:
            assert 1 == 2, 'I do not want maintance the functionality of `show flow`... ^_^'
            ctx.shift_offsets = torch.cat(ctx.shift_offsets, dim=0).float()
            mask_nums = ctx.shift_offsets.size(0) // ctx.bz
            ctx.flow_srcs = torch.zeros(ctx.bz, 3, ctx.h, ctx.w).type_as(input)
            for idx in range(ctx.bz):
                shift_offset = ctx.shift_offsets.narrow(0, idx * mask_nums,
                    mask_nums)
                shift_offsets_map = torch.zeros(1, ctx.h, ctx.w, 2).type_as(
                    input)
                shift_offsets_map[:, ((flag_cur == 1).nonzero().squeeze() //
                    ctx.w), ((flag_cur == 1).nonzero().squeeze() % ctx.w), :
                    ] = shift_offset.unsqueeze(0)
                flow_src = util.highlight_flow(shift_offsets_map, flag_cur.
                    unsqueeze(0))
                ctx.flow_srcs[idx] = flow_src
        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    @staticmethod
    def get_flow_src():
        return InnerShiftTripleFunction.ctx.flow_srcs

    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst
        c = grad_output.size(1)
        grad_former_all = grad_output[:, 0:c // 3, :, :]
        grad_latter_all = grad_output[:, c // 3:c * 2 // 3, :, :].clone()
        grad_shifted_all = grad_output[:, c * 2 // 3:c, :, :].clone()
        W_mat_t = ind_lst.permute(0, 2, 1).contiguous()
        grad = grad_shifted_all.view(ctx.bz, c // 3, -1).permute(0, 2, 1)
        grad_shifted_weighted = torch.bmm(W_mat_t, grad)
        grad_shifted_weighted = grad_shifted_weighted.permute(0, 2, 1
            ).contiguous().view(ctx.bz, c // 3, ctx.h, ctx.w)
        grad_latter_all = torch.add(grad_latter_all, grad_shifted_weighted.
            mul(ctx.triple_w))
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)
        return grad_input, None, None, None, None, None, None


class InnerResShiftTriple(nn.Module):

    def __init__(self, inner_nc, shift_sz=1, stride=1, mask_thred=1,
        triple_weight=1, layer_to_last=3):
        super(InnerResShiftTriple, self).__init__()
        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.show_flow = False
        self.flow_srcs = None
        self.layer_to_last = layer_to_last
        self.inner_nc = inner_nc
        self.res_net = nn.Sequential(nn.Conv2d(inner_nc * 2, inner_nc,
            kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(inner_nc
            ), nn.ReLU(True), nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
            stride=1, padding=1), nn.InstanceNorm2d(inner_nc))

    def set_mask(self, mask_global):
        mask = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask = mask.squeeze()
        return self.mask

    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        self.flag = util.cal_flag_given_mask_thred(self.mask, self.shift_sz,
            self.stride, self.mask_thred)
        shift_out = InnerShiftTripleFunction.apply(input, self.shift_sz,
            self.stride, self.triple_weight, self.flag, self.show_flow)
        c_out = shift_out.size(1)
        F_c = shift_out.narrow(1, 0, c_out // 3)
        F_s = shift_out.narrow(1, c_out // 3, c_out // 3)
        F_shift = shift_out.narrow(1, c_out * 2 // 3, c_out // 3)
        F_fuse = F_c * F_shift
        F_com = torch.cat([F_c, F_fuse], dim=1)
        res_out = self.res_net(F_com)
        F_c = F_c + res_out
        final_out = torch.cat([F_c, F_s], dim=1)
        if self.show_flow:
            self.flow_srcs = InnerShiftTripleFunction.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__ + '(' + ' ,triple_weight ' + str(self
            .triple_weight) + ')'


class InnerCosFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, criterion, strength, target, mask):
        ctx.c = input.size(1)
        ctx.strength = strength
        ctx.criterion = criterion
        if len(target.size()) == 0:
            target = target.expand_as(input.narrow(1, ctx.c // 2, ctx.c // 2)
                ).type_as(input)
        ctx.save_for_backward(input, target, mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            input, target, mask = ctx.saved_tensors
            former = input.narrow(1, 0, ctx.c // 2)
            former_in_mask = torch.mul(former, mask)
            if former_in_mask.size() != target.size():
                target = target.narrow(0, 0, 1).expand_as(former_in_mask
                    ).type_as(former_in_mask)
            former_in_mask_clone = former_in_mask.clone().detach(
                ).requires_grad_(True)
            ctx.loss = ctx.criterion(former_in_mask_clone, target
                ) * ctx.strength
            ctx.loss.backward()
        grad_output[:, 0:ctx.c // 2, :, :] += former_in_mask_clone.grad
        return grad_output, None, None, None, None


class InnerCos(nn.Module):

    def __init__(self, crit='MSE', strength=1, skip=0, layer_to_last=3,
        device='gpu'):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss(
            ) if self.crit == 'MSE' else torch.nn.L1Loss()
        self.strength = strength
        self.skip = skip
        self.layer_to_last = layer_to_last
        self.device = device
        self.target = torch.tensor(1.0)

    def set_mask(self, mask_global):
        mask_all = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask_all = mask_all.float()

    def _split_mask(self, cur_bsize):
        cur_device = torch.current_device()
        self.cur_mask = self.mask_all[cur_device * cur_bsize:(cur_device + 
            1) * cur_bsize, :, :, :]

    def forward(self, in_data):
        self.bz, self.c, _, _ = in_data.size()
        if self.device != 'cpu':
            self._split_mask(self.bz)
        else:
            self.cur_mask = self.mask_all
        self.cur_mask = self.cur_mask
        if not self.skip:
            self.output = InnerCosFunction.apply(in_data, self.criterion,
                self.strength, self.target, self.cur_mask)
            self.target = in_data.narrow(1, self.c // 2, self.c // 2).detach()
        else:
            self.output = in_data
        return self.output

    def __repr__(self):
        skip_str = 'True' if not self.skip else 'False'
        return (self.__class__.__name__ + '(' + 'skip: ' + skip_str +
            'layer ' + str(self.layer_to_last) + ' to last' +
            ' ,strength: ' + str(self.strength) + ')')


class InnerShiftTriple(nn.Module):

    def __init__(self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1,
        layer_to_last=3, device='gpu'):
        super(InnerShiftTriple, self).__init__()
        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.layer_to_last = layer_to_last
        self.device = device
        self.show_flow = False
        self.flow_srcs = None

    def set_mask(self, mask_global):
        self.mask_all = util.cal_feat_mask(mask_global, self.layer_to_last)

    def _split_mask(self, cur_bsize):
        cur_device = torch.current_device()
        self.cur_mask = self.mask_all[cur_device * cur_bsize:(cur_device + 
            1) * cur_bsize, :, :, :]

    def forward(self, input):
        self.bz, self.c, self.h, self.w = input.size()
        if self.device != 'cpu':
            self._split_mask(self.bz)
        else:
            self.cur_mask = self.mask_all
        self.flag = util.cal_flag_given_mask_thred(self.cur_mask, self.
            shift_sz, self.stride, self.mask_thred)
        final_out = InnerShiftTripleFunction.apply(input, self.shift_sz,
            self.stride, self.triple_weight, self.flag, self.show_flow)
        if self.show_flow:
            self.flow_srcs = InnerShiftTripleFunction.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__ + '(' + ' ,triple_weight ' + str(self
            .triple_weight) + ')'


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Zhaoyi_Yan_Shift_Net_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DenseNet(*[], **{}), [torch.rand([4, 3, 128, 128])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DenseNetDiscrimator(*[], **{'input_nc': 4}), [torch.rand([4, 3, 128, 128])], {})

    def test_002(self):
        self._check(EasyUnetGenerator(*[], **{'input_nc': 4, 'output_nc': 4}), [torch.rand([4, 4, 256, 256])], {})

    def test_003(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(Self_Attn(*[], **{'in_dim': 64, 'activation': 4}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(SwitchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(TVLoss(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(UnetGenerator(*[], **{'input_nc': 4, 'output_nc': 4, 'num_downs': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_008(self):
        self._check(VGG16FeatureExtractor(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_009(self):
        self._check(_DenseBlock(*[], **{'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5, 'use_spectral_norm': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(_DenseLayer(*[], **{'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5, 'use_spectral_norm': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(_Transition(*[], **{'num_input_features': 4, 'num_output_features': 4, 'use_spectral_norm': 4}), [torch.rand([4, 4, 4, 4])], {})

