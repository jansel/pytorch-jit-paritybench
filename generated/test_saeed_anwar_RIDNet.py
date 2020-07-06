import sys
_module = sys.modules[__name__]
del sys
data = _module
benchmark = _module
common = _module
demo = _module
div2k = _module
myimage = _module
srdata = _module
dataloader = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
main = _module
model = _module
common = _module
ops = _module
ridnet = _module
option = _module
template = _module
trainer = _module
utility = _module

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


from torch.utils.data.dataloader import default_collate


import numpy as np


import scipy.misc as misc


import torch


import torch.utils.data as data


import random


from torchvision import transforms


import math


import queue


import collections


import torch.multiprocessing as multiprocessing


from torch._C import _set_worker_signal_handlers


from torch._C import _remove_worker_pids


from torch._C import _error_if_any_worker_fails


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataloader import ExceptionWrapper


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


import torchvision.models as models


from torch.nn import DataParallel


import torch.nn.init as init


import time


from functools import reduce


import torch.optim.lr_scheduler as lrs


class Loss(nn.modules.loss._Loss):

    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        None
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(loss_type[3:], rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args.n_GPUs))
        if args.load != '.':
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
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
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()


class Adversarial(nn.Module):

    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.discriminator = discriminator.Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP':
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(self.discriminator.parameters(), betas=(0, 0.9), eps=1e-08, lr=1e-05)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

    def forward(self, fake, real):
        fake_detach = fake.detach()
        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(), inputs=hat, retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()
            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)
        self.loss /= self.gan_k
        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()
        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        return dict(**state_discriminator, **state_optimizer)


class Discriminator(nn.Module):

    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()
        in_channels = 3
        out_channels = 64
        depth = 7
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        m_features = [common.BasicBlock(args.n_colors, out_channels, 3, bn=bn, act=act)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(common.BasicBlock(in_channels, out_channels, 3, stride=stride, bn=bn, act=act))
        self.features = nn.Sequential(*m_features)
        patch_size = args.patch_size // 2 ** ((depth + 1) // 2)
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), act, nn.Linear(1024, 1)]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))
        return output


class VGG(nn.Module):

    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
        vgg_mean = 0.485, 0.456, 0.406
        vgg_std = 0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):

        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())
        loss = F.mse_loss(vgg_sr, vgg_hr)
        return loss


class Model(nn.Module):

    def __init__(self, args, ckp):
        super(Model, self).__init__()
        None
        self.noise_g = args.noise_g
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
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
        if args.print_model:
            None

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(target.state_dict(), os.path.join(apath, 'model', 'model_latest.pt'))
        if is_best:
            torch.save(target.state_dict(), os.path.join(apath, 'model', 'model_best.pt'))
        if self.save_models:
            torch.save(target.state_dict(), os.path.join(apath, 'model', 'model_{}.pt'.format(epoch)))

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if resume == -1:
            self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs), strict=False)
        elif resume == 0:
            if pre_train != '.':
                None
                self.get_model().load_state_dict(torch.load(pre_train, **kwargs), strict=False)
        else:
            self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model', 'model_{}.pt'.format(resume)), **kwargs), strict=False)

    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [x[:, :, 0:h_size, 0:w_size], x[:, :, 0:h_size, w - w_size:w], x[:, :, h - h_size:h, 0:w_size], x[:, :, h - h_size:h, w - w_size:w]]
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:i + n_GPUs], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.forward_chop(patch, shave=shave, min_size=min_size) for patch in lr_list]
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, w_size - w + w_half:w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, h_size - h + h_half:h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, h_size - h + h_half:h_size, w_size - w + w_half:w_size]
        return output

    def forward_x8(self, x, forward_function):

        def _transform(v, op):
            if self.precision != 'single':
                v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            ret = torch.Tensor(tfnp)
            if self.precision == 'half':
                ret = ret.half()
            return ret
        lr_list = [x]
        for tf in ('v', 'h', 't'):
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if i % 4 % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')
        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)
        return output


class MeanShift(nn.Module):

    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()
        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


def init_weights(modules):
    pass


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.ReLU(inplace=True))
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ResBlock(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class Merge_Run(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run, self).__init__()
        self.body1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.ReLU(inplace=True))
        self.body2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2), nn.ReLU(inplace=True))
        self.body3 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad), nn.ReLU(inplace=True))
        init_weights(self.modules)

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class Merge_Run_dual(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run_dual, self).__init__()
        self.body1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2), nn.ReLU(inplace=True))
        self.body2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4), nn.ReLU(inplace=True))
        self.body3 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad), nn.ReLU(inplace=True))
        init_weights(self.modules)

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class BasicBlockSig(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.Sigmoid())
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, group=1):
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 1, 1, 0))
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class _UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()
        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = ops.BasicBlock(channel, channel // reduction, 1, 1, 0)
        self.c2 = ops.BasicBlockSig(channel // reduction, channel, 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()
        self.r1 = ops.Merge_Run_dual(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels, out_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        out = self.ca(r3)
        return out


class RIDNET(nn.Module):

    def __init__(self, args):
        super(RIDNET, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = ops.BasicBlock(3, n_feats, kernel_size, 1, 1)
        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)
        self.tail = nn.Conv2d(n_feats, 3, kernel_size, 1, 1, 1)

    def forward(self, x):
        s = self.sub_mean(x)
        h = self.head(s)
        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)
        res = self.tail(b_out)
        out = self.add_mean(res)
        f_out = out + x
        return f_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlockSig,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CALayer,
     lambda: ([], {'channel': 64}),
     lambda: ([torch.rand([4, 64, 4, 4])], {}),
     True),
    (EResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanShift,
     lambda: ([], {'mean_rgb': [4, 4, 4], 'sub': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Merge_Run,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Merge_Run_dual,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleBlock,
     lambda: ([], {'n_channels': 4, 'scale': 1.0, 'multi_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (_UpsampleBlock,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_saeed_anwar_RIDNet(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

