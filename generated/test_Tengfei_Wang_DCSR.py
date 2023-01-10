import sys
_module = sys.modules[__name__]
del sys
master = _module
dataloader = _module
common = _module
dataset = _module
loss = _module
contextual = _module
gaussian = _module
vgg = _module
main = _module
model = _module
alignment = _module
attention = _module
common = _module
dcsr = _module
option = _module
trainer = _module
utility = _module
utils = _module
tools = _module

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


import random


import numpy as np


import torch


import torch.utils.data as data


from torch.utils.data import dataloader


from torch.utils.data import ConcatDataset


import matplotlib


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


import scipy.ndimage


import torchvision.models as models


from collections import namedtuple


import torchvision.models.vgg as vgg


from torch.autograd import Variable


from torch import nn


from torch.nn.utils import spectral_norm as spectral_norm_fn


from torch.nn.utils import weight_norm as weight_norm_fn


from torchvision import transforms


from torchvision import utils as vutils


from torchvision import models


import math


import torch.nn.utils as utils


import time


import torch.optim as optim


import torch.optim.lr_scheduler as lrs


class Loss(nn.modules.loss._Loss):

    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        None
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        gaussian_module = import_module('loss.gaussian')
        self.gaussian_layer = getattr(gaussian_module, 'GaussianLayer')()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('contextual_ref') >= 0:
                module = import_module('loss.contextual')
                loss_function = getattr(module, 'ContextualLoss')()
            elif loss_type.find('contextual_hr') >= 0:
                module = import_module('loss.contextual')
                loss_function = getattr(module, 'ContextualLoss')()
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_function})
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module
        self.gaussian_layer
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args.n_GPUs))
        if args.load != '':
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr, ref):
        losses = []
        for i, l in enumerate(self.loss):
            if 'contextual' not in l['type']:
                if l['function'] is not None:
                    sr_d = sr
                    if hr.shape != sr.shape:
                        sr_d = F.interpolate(sr, size=[sr.shape[2] // 2, sr.shape[3] // 2], mode='bicubic')
                    sr_lf = self.gaussian_layer(sr_d)
                    hr_lf = self.gaussian_layer(hr)
                    loss = l['function'](sr_lf, hr_lf)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
                    self.log[-1, i] += effective_loss.item()
            elif 'contextual_ref' in l['type']:
                loss = l['function'](sr, ref)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif 'contextual_hr' in l['type']:
                loss = l['function'](sr, hr) + l['function'](hr, sr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))
        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **kwargs))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()


class VGG19(nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)
        return out


LOSS_TYPES = ['cosine']


def compute_cosine_distance(x, y):
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)
    y_normalized = y_normalized.reshape(N, C, -1)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)
    dist = 1 - cosine_sim
    return dist


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)
    cx = w / torch.sum(w, dim=2, keepdim=True)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-05)
    return dist_tilde


def contextual_loss(x=torch.Tensor, y=torch.Tensor, band_width=0.5, loss_type='cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
    N, C, H, W = x.size()
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    r_m = torch.max(cx, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / 0.5), 1, r_m[1])
    cx = torch.sum(torch.squeeze(r_m[0] * c, 1), dim=1) / torch.sum(torch.squeeze(c, 1), dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-05))
    return cx_loss


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.
    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, band_width=0.5, loss_type='cosine', use_vgg=True, vgg_layer='relu3_4'):
        super(ContextualLoss, self).__init__()
        self.band_width = band_width
        if use_vgg:
            None
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(name='vgg_mean', tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
            self.register_buffer(name='vgg_std', tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, 'VGG model takes 3 chennel images.'
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)
        return contextual_loss(x, y, self.band_width)


class GaussianLayer(nn.Module):

    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(nn.ReflectionPad2d(2), nn.Conv2d(3, 3, 3, stride=1, padding=0, bias=None, groups=3))
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((3, 3))
        n[1, 1] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=1)
        for name, f in self.named_parameters():
            weight_torch = torch.from_numpy(k)
            f.data.copy_(weight_torch)
            f.requires_grad = False


class Model(nn.Module):

    def __init__(self, args, ckp):
        super(Model, self).__init__()
        None
        self.scale = args.scale
        self.flag_8k = args.flag_8k
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.dir, pre_train=args.pre_train, resume=args.resume, cpu=args.cpu)
        None

    def forward(self, x, ref):
        target = self.get_model()
        if not self.training:
            if not self.flag_8k:
                min_size1 = 126
                min_size2 = 126
            elif x.shape[2] < x.shape[3]:
                min_size1 = 188
                min_size2 = 252
            else:
                min_size1 = 252
                min_size2 = 188
            x = x[:, :, :x.shape[2] // min_size1 * min_size1, :x.shape[3] // min_size2 * min_size2]
            shape_query = x.shape
            num_x = x.shape[3] // min_size2
            num_y = x.shape[2] // min_size1
            x = nn.ZeroPad2d(20)(x)
            sr_list = []
            for j in range(num_y):
                for i in range(num_x):
                    patch_LR = x[:, :, j * min_size1:j * min_size1 + min_size1 + 40, i * min_size2:i * min_size2 + min_size2 + 40]
                    if i > num_x // 4 - 1 and i < 3 * num_x // 4 and j > num_y // 4 - 1 and j < 3 * num_y // 4:
                        patch_ref = ref[:, :, np.maximum((j - num_y // 4) * 2 * min_size1 - 20, 0):np.minimum((j - num_y // 4) * 2 * min_size1 + 2 * min_size1 + 80, ref.shape[2]), np.maximum((i - num_x // 4) * 2 * min_size2 - 20, 0):np.minimum((i - num_x // 4) * 2 * min_size2 + 2 * min_size2 + 80, ref.shape[3])]
                        coarse = False
                    elif i == num_x // 4 - 1 and j >= num_y // 4 - 1 and j < 3 * num_y // 4 or j == num_y // 4 - 1 and i >= num_x // 4 - 1 and i < 3 * num_x // 4:
                        patch_ref = ref[:, :, np.maximum((j - num_y // 4) * 2 * min_size1, 0):np.maximum((j - num_y // 4) * 2 * min_size1, 0) + 2 * min_size1 + 100, np.maximum((i - num_x // 4) * 2 * min_size2, 0):np.maximum((i - num_x // 4) * 2 * min_size2, 0) + 2 * min_size2 + 100]
                        coarse = False
                    elif i == 3 * num_x // 4 and j > num_y // 4 - 1 and j <= 3 * num_y // 4 or j == 3 * num_y // 4 and i > num_x // 4 - 1 and i <= 3 * num_x // 4:
                        patch_ref = ref[:, :, np.minimum((j - num_y // 4) * 2 * min_size1, ref.shape[2] - 2 * min_size1):np.minimum((j - num_y // 4) * 2 * min_size1, ref.shape[2] - 2 * min_size1) + 2 * min_size1, np.minimum((i - num_x // 4) * 2 * min_size2, ref.shape[3] - 2 * min_size2):np.minimum((i - num_x // 4) * 2 * min_size2, ref.shape[3] - 2 * min_size2) + 2 * min_size2]
                        coarse = False
                    elif j == num_y // 4 - 1 and i == 3 * num_x // 4:
                        patch_ref = ref[:, :, 0:2 * min_size1 + 100, ref.shape[3] - 2 * min_size2 - 100:ref.shape[3]]
                        coarse = False
                    elif j == 3 * num_y // 4 and i == num_x // 4 - 1:
                        patch_ref = ref[:, :, ref.shape[2] - 2 * min_size1 - 100:ref.shape[2], 0:2 * min_size2 + 100]
                        coarse = False
                    else:
                        patch_ref = ref
                        coarse = True
                    patch_sr = self.model(patch_LR, patch_ref, coarse)
                    sr_list.append(patch_sr[:, :, 40:-40, 40:-40])
            sr_list = torch.cat(sr_list, dim=0)
            sr_list = sr_list.view(sr_list.shape[0], -1)
            sr_list = sr_list.permute(1, 0)
            sr_list = torch.unsqueeze(sr_list, 0)
            output = F.fold(sr_list, output_size=(shape_query[2] * 2, shape_query[3] * 2), kernel_size=(2 * min_size1, 2 * min_size2), padding=0, stride=(2 * min_size1, 2 * min_size2))
            return output
        else:
            return self.model(x, ref, False)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch):
        target = self.get_model()
        torch.save(target.state_dict(), os.path.join(apath, 'model_latest.pt'))
        if self.save_models:
            torch.save(target.state_dict(), os.path.join(apath, 'model_{}.pt'.format(epoch)))

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if resume == -1:
            self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model_latest.pt'), **kwargs), strict=False)
        elif resume == 0:
            if pre_train != '.':
                None
                self.get_model().load_state_dict(torch.load(pre_train, **kwargs), strict=False)
        else:
            None
            self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model', 'model_{}.pt'.format(resume)), **kwargs), strict=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class AlignedConv2d(nn.Module):

    def __init__(self, inc, outc=1, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(AlignedConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ReflectionPad2d(padding)
        head = [nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True)]
        head2 = [nn.Conv2d(2 * 32, 32, kernel_size=5, padding=2, stride=stride), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1)]
        self.p_conv = nn.Sequential(*head2)
        self.conv1 = nn.Sequential(*head)
        self.p_conv.register_backward_hook(self._set_lr)
        self.conv1.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(2 * inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x, query, ref):
        query = F.interpolate(query, scale_factor=2, mode='bicubic')
        ref = self.conv1(ref)
        query = self.conv1(query)
        affine = self.p_conv(torch.cat((ref, query), 1)) + 1.0
        if self.modulation:
            m = torch.sigmoid(self.m_conv(torch.cat((ref, query), 1)))
        dtype = affine.data.type()
        ks = self.kernel_size
        N = ks * ks
        if self.padding:
            x = self.zero_padding(x)
        affine = torch.clamp(affine, -3, 3)
        p = self._get_p(affine, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        alignment = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(alignment.size(1))], dim=1)
            alignment *= m
        alignment = self._reshape_alignment(alignment, ks)
        return alignment

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-1 * ((self.kernel_size - 1) // 2) - 0.5, (self.kernel_size - 1) // 2 + 0.6, 1.0), torch.arange(-1 * ((self.kernel_size - 1) // 2) - 0.5, (self.kernel_size - 1) // 2 + 0.6, 1.0))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, affine, dtype):
        N, h, w = self.kernel_size * self.kernel_size, affine.size(2), affine.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_n.repeat(affine.size(0), 1, h, w)
        p = p.permute(0, 2, 3, 1)
        affine = affine.permute(0, 2, 3, 1)
        s_x = affine[:, :, :, 0:1]
        s_y = affine[:, :, :, 1:2]
        p[:, :, :, :N] = p[:, :, :, :N].clone() * s_x.type(dtype)
        p[:, :, :, N:] = p[:, :, :, N:].clone() * s_y.type(dtype)
        p = p.view(p.shape[0], p.shape[1], p.shape[2], 1, p.shape[3])
        p = torch.cat((p[:, :, :, :, :N], p[:, :, :, :, N:]), 3)
        p = p.permute(0, 1, 2, 4, 3)
        theta = (affine[:, :, :, 2:] - 1.0) * 1.0472
        rm = torch.cat((torch.cos(theta), torch.sin(theta), -1 * torch.sin(theta), torch.cos(theta)), 3)
        rm = rm.view(affine.shape[0], affine.shape[1], affine.shape[2], 2, 2)
        result = torch.matmul(p, rm)
        result = torch.cat((result[:, :, :, :, 0], result[:, :, :, :, 1]), 3)
        result = result.permute(0, 3, 1, 2) + (self.kernel_size - 1) // 2 + 0.5 + p_0
        return result

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        result = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return result

    @staticmethod
    def _reshape_alignment(alignment, ks):
        b, c, h, w, N = alignment.size()
        alignment = torch.cat([alignment[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        alignment = alignment.contiguous().view(b, c, h * ks, w * ks)
        return alignment


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, In=False, act=nn.PReLU()):
        m = [conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if In:
            m.append(nn.InstanceNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = padding_left, padding_right, padding_top, padding_bottom
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.                Only "same" or "valid" are supported.'.format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches


class FeatureMatching(nn.Module):

    def __init__(self, ksize=3, k_vsize=1, scale=2, stride=1, in_channel=3, out_channel=64, conv=default_conv):
        super(FeatureMatching, self).__init__()
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride
        self.scale = scale
        match0 = BasicBlock(conv, 128, 16, 1, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.feature_extract = torch.nn.Sequential()
        for x in range(7):
            self.feature_extract.add_module(str(x), vgg_pretrained_features[x])
        self.feature_extract.add_module('map', match0)
        for param in self.feature_extract.parameters():
            param.requires_grad = True
        vgg_mean = 0.485, 0.456, 0.406
        vgg_std = 0.229, 0.224, 0.225
        self.sub_mean = MeanShift(1, vgg_mean, vgg_std)
        self.avgpool = nn.AvgPool2d((self.scale, self.scale), (self.scale, self.scale))

    def forward(self, query, key, flag_8k):
        query = self.sub_mean(query)
        if not flag_8k:
            query = F.interpolate(query, scale_factor=self.scale, mode='bicubic', align_corners=True)
        query = self.feature_extract(query)
        shape_query = query.shape
        query = extract_image_patches(query, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        key = self.avgpool(key)
        key = self.sub_mean(key)
        if not flag_8k:
            key = F.interpolate(key, scale_factor=self.scale, mode='bicubic', align_corners=True)
        key = self.feature_extract(key)
        shape_key = key.shape
        w = extract_image_patches(key, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        w = w.permute(0, 2, 1)
        w = F.normalize(w, dim=2)
        query = F.normalize(query, dim=1)
        y = torch.bmm(w, query)
        relavance_maps, hard_indices = torch.max(y, dim=1)
        relavance_maps = relavance_maps.view(shape_query[0], 1, shape_query[2], shape_query[3])
        return relavance_maps, hard_indices


class AlignedAttention(nn.Module):

    def __init__(self, ksize=3, k_vsize=1, scale=1, stride=1, align=False):
        super(AlignedAttention, self).__init__()
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride
        self.scale = scale
        self.align = align
        if align:
            self.align = AlignedConv2d(inc=128, outc=1, kernel_size=self.scale * self.k_vsize, padding=1, stride=self.scale * 1, bias=None, modulation=False)

    def warp(self, input, dim, index):
        views = [input.size(0)] + [(1 if i != dim else -1) for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lr, ref, index_map, value):
        shape_out = list(lr.size())
        kernel = self.scale * self.k_vsize
        unfolded_value = extract_image_patches(value, ksizes=[kernel, kernel], strides=[self.stride * self.scale, self.stride * self.scale], rates=[1, 1], padding='same')
        warpped_value = self.warp(unfolded_value, 2, index_map)
        warpped_features = F.fold(warpped_value, output_size=(shape_out[2] * 2, shape_out[3] * 2), kernel_size=(kernel, kernel), padding=0, stride=self.scale)
        if self.align:
            unfolded_ref = extract_image_patches(ref, ksizes=[kernel, kernel], strides=[self.stride * self.scale, self.stride * self.scale], rates=[1, 1], padding='same')
            warpped_ref = self.warp(unfolded_ref, 2, index_map)
            warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2] * 2, shape_out[3] * 2), kernel_size=(kernel, kernel), padding=0, stride=self.scale)
            warpped_features = self.align(warpped_features, lr, warpped_ref)
        return warpped_features


class PatchSelect(nn.Module):

    def __init__(self, stride=1):
        super(PatchSelect, self).__init__()
        self.stride = stride

    def forward(self, query, key):
        shape_query = query.shape
        shape_key = key.shape
        P = shape_key[3] - shape_query[3] + 1
        key = extract_image_patches(key, ksizes=[shape_query[2], shape_query[3]], strides=[self.stride, self.stride], rates=[1, 1], padding='valid')
        query = query.view(shape_query[0], shape_query[1] * shape_query[2] * shape_query[3], 1)
        y = torch.mean(torch.abs(key - query), 1)
        relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True)
        return hard_indices.view(-1), P, relavance_maps


class Encoder_input(nn.Module):

    def __init__(self, num_res_blocks, n_feats, img_channel, res_scale=1):
        super(Encoder_input, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(img_channel, n_feats)
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class ResList(nn.Module):

    def __init__(self, num_res_blocks, n_feats, res_scale=1):
        super(ResList, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class DCSR(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(DCSR, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        self.flag_8k = args.flag_8k
        m_head1 = [BasicBlock(conv, args.n_colors, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, n_feats, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        m_head2 = [BasicBlock(conv, n_feats, n_feats, kernel_size, stride=2, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, n_feats, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        m_tail = [conv3x3(n_feats, n_feats // 2), nn.LeakyReLU(0.2, inplace=True), conv3x3(n_feats // 2, args.n_colors)]
        fusion1 = [BasicBlock(conv, 2 * n_feats, n_feats, 5, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, n_feats, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        fusion2 = [BasicBlock(conv, 2 * n_feats, n_feats, 5, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, n_feats, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        self.feature_match = FeatureMatching(ksize=3, scale=2, stride=1, in_channel=args.n_colors, out_channel=64)
        self.ref_encoder1 = nn.Sequential(*m_head1)
        self.ref_encoder2 = nn.Sequential(*m_head2)
        self.res1 = ResList(4, n_feats)
        self.res2 = ResList(4, n_feats)
        self.input_encoder = Encoder_input(8, n_feats, args.n_colors)
        self.fusion1 = nn.Sequential(*fusion1)
        self.decoder1 = ResList(8, n_feats)
        self.fusion2 = nn.Sequential(*fusion2)
        self.decoder2 = ResList(4, n_feats)
        self.decoder_tail = nn.Sequential(*m_tail)
        fusion11 = [BasicBlock(conv, 1, 16, 7, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, 16, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        fusion12 = [BasicBlock(conv, 1, 16, 7, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, 16, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        fusion13 = [BasicBlock(conv, 4, 32, 7, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(conv, 32, 3, kernel_size, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        self.alpha1 = nn.Sequential(*fusion11)
        self.alpha2 = nn.Sequential(*fusion12)
        self.alpha3 = nn.Sequential(*fusion13)
        if self.flag_8k:
            self.aa1 = AlignedAttention(scale=4, align=True)
            self.aa2 = AlignedAttention(scale=2, align=False)
            self.aa3 = AlignedAttention(scale=4, align=True)
        else:
            self.aa1 = AlignedAttention(scale=2, align=True)
            self.aa2 = AlignedAttention(scale=1, align=False)
            self.aa3 = AlignedAttention(scale=2, align=True)
        self.avgpool = nn.AvgPool2d((2, 2), (2, 2))
        self.select = PatchSelect()

    def forward(self, input, ref, coarse=False):
        with torch.no_grad():
            if coarse:
                B = input.shape[0]
                ref_ = F.interpolate(ref, scale_factor=1 / 16, mode='bicubic')
                input_ = F.interpolate(input, scale_factor=1 / 8, mode='bicubic')
                i, P, r = self.select(input_, ref_)
                for j in range(B):
                    ref_p = ref[:, :, np.maximum((2 * 8 * (i[j] // P)).cpu(), 0):np.minimum((2 * 8 * (i[j] // P) + 2 * input.shape[2]).cpu(), ref.shape[2]), np.maximum((2 * 8 * (i[j] % P)).cpu(), 0):np.minimum((2 * 8 * (i[j] % P) + 2 * input.shape[3]).cpu(), ref.shape[3])]
            else:
                ref_p = ref
        confidence_map, index_map = self.feature_match(input, ref_p, self.flag_8k)
        ref_downsampled = self.avgpool(ref_p)
        ref_hf = ref_p - F.interpolate(ref_downsampled, scale_factor=2, mode='bicubic')
        ref_hf_aligned = self.aa1(input, ref_p, index_map, ref_hf)
        ref_features1 = self.ref_encoder1(ref_p)
        ref_features1 = self.res1(ref_features1)
        ref_features2 = self.ref_encoder2(ref_features1)
        ref_features2 = self.res2(ref_features2)
        input_down = F.interpolate(input, scale_factor=1 / 2, mode='bicubic')
        ref_features_matched = self.aa2(input_down, ref_p, index_map, ref_features2)
        ref_features_aligned = self.aa3(input, ref_p, index_map, ref_features1)
        input_up = self.input_encoder(input)
        if self.flag_8k:
            confidence_map = F.interpolate(confidence_map, scale_factor=2, mode='bicubic')
        cat_features = torch.cat((ref_features_matched, input_up), 1)
        fused_features2 = self.alpha1(confidence_map) * self.fusion1(cat_features) + input_up
        fused_features2 = self.decoder1(fused_features2)
        fused_features2_up = F.interpolate(fused_features2, scale_factor=2, mode='bicubic')
        confidence_map = F.interpolate(confidence_map, scale_factor=2, mode='bicubic')
        cat_features = torch.cat((ref_features_aligned, fused_features2_up), 1)
        fused_features1 = self.alpha2(confidence_map) * self.fusion2(cat_features) + fused_features2_up
        fused_features1 = self.decoder2(fused_features1)
        result = self.decoder_tail(fused_features1) + ref_hf_aligned * self.alpha3(torch.cat((confidence_map, ref_hf_aligned), 1))
        return result


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Encoder_input,
     lambda: ([], {'num_res_blocks': 4, 'n_feats': 4, 'img_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeatureMatching,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 4, 4]), 0], {}),
     False),
    (GaussianLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (PatchSelect,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResList,
     lambda: ([], {'num_res_blocks': 4, 'n_feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_Tengfei_Wang_DCSR(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

