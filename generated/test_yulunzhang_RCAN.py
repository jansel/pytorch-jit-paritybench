import sys
_module = sys.modules[__name__]
del sys
code = _module
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
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
option = _module
template = _module
trainer = _module
utility = _module
data = _module
benchmark = _module
common = _module
demo = _module
div2k = _module
srdata = _module
dataloader = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
main = _module
model = _module
common = _module
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
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
xrange = range
wraps = functools.wraps


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
        for l in self.get_loss_module():
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
        self.scale = args.scale
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


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


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


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SEResBlock(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {(2): (6, 2, 2), (4): (8, 4, 2), (8): (12, 8, 2)}[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d
    return conv_f(in_channels, out_channels, kernel_size, stride=stride, padding=padding)


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1), nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels, scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)
        out = a_0.add(a_1)
        return out


class DDBPN(nn.Module):

    def __init__(self, args):
        super(DDBPN, self).__init__()
        scale = args.scale[0]
        n0 = 128
        nr = 32
        self.depth = 6
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0), nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True, i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3, padding=1)]
        self.reconstruction = nn.Sequential(*reconstruction)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.initial(x)
        h_list = []
        l_list = []
        for i in range(self.depth - 1):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        out = self.add_mean(out)
        return out


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CALayer,
     lambda: ([], {'channel': 18}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {'args': _mock_config(n_colors=4, patch_size=16)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RCAB,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer,
     lambda: ([], {'channel': 18}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     True),
    (SEResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yulunzhang_RCAN(_paritybench_base):
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

