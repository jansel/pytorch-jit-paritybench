import sys
_module = sys.modules[__name__]
del sys
src = _module
data = _module
benchmark = _module
common = _module
demo = _module
div2k = _module
div2kjpeg = _module
sr291 = _module
srdata = _module
video = _module
dataloader = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
main = _module
model = _module
common = _module
dcn = _module
deform_conv = _module
setup = _module
ddbpn = _module
edsr = _module
han = _module
matrixmodel = _module
mdsr = _module
ops = _module
rcan = _module
rcan1 = _module
rcan3 = _module
rcan4 = _module
rdn = _module
rdn1 = _module
rdn2 = _module
vdsr = _module
option = _module
template = _module
trainer = _module
utility = _module
videotester = _module

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


from torch.utils.data import dataloader


from torch.utils.data import ConcatDataset


import numpy as np


import torch


import torch.utils.data as data


import random


import torch.multiprocessing as multiprocessing


from torch.utils.data import DataLoader


from torch.utils.data import SequentialSampler


from torch.utils.data import RandomSampler


from torch.utils.data import BatchSampler


from torch.utils.data import _utils


from torch.utils.data._utils import collate


from torch.utils.data._utils import signal_handling


from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL


from torch.utils.data._utils import ExceptionWrapper


from torch.utils.data._utils import IS_WINDOWS


from torch.utils.data._utils.worker import ManagerWatchdog


import matplotlib


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


from types import SimpleNamespace


import torch.optim as optim


import torchvision.models as models


import torch.nn.parallel as P


import torch.utils.model_zoo


import math


import logging


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.nn.init as init


from torch.autograd import Variable


import torch.nn.utils as utils


import time


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
        if args.load != '':
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


class Adversarial(nn.Module):

    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.dis = discriminator.Discriminator(args)
        if gan_type == 'WGAN_GP':
            optim_dict = {'optimizer': 'ADAM', 'betas': (0, 0.9), 'epsilon': 1e-08, 'lr': 1e-05, 'weight_decay': args.weight_decay, 'decay': args.decay, 'gamma': args.gamma}
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args
        self.optimizer = utility.make_optimizer(optim_args, self.dis)

    def forward(self, fake, real):
        self.loss = 0
        fake_detach = fake.detach()
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(), inputs=hat, retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            elif self.gan_type == 'RGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()
            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)
        self.loss /= self.gan_k
        d_fake_bp = self.dis(fake)
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'RGAN':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)
        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss


class Discriminator(nn.Module):
    """
        output is not normalized
    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3, padding=1, stride=stride, bias=False), nn.BatchNorm2d(_out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))
        patch_size = args.patch_size // 2 ** ((depth + 1) // 2)
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
        self.features = nn.Sequential(*m_features)
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
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])
        vgg_mean = 0.485, 0.456, 0.406
        vgg_std = 0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

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
        self.input_large = args.model == 'VDSR'
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
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=args.resume, cpu=args.cpu)
        None

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)
        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'), **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.model.url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format(resume)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)

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

    def forward_x8(self, *args, forward_function=None):

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
        list_x = []
        for a in args:
            x = [a]
            for tf in ('v', 'h', 't'):
                x.extend([_transform(_x, tf) for _x in x])
            list_x.append(x)
        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list):
                y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y):
                    _list_y.append(_y)
        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if i % 4 % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')
        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1:
            y = y[0]
        return y


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

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResBlock(nn.Module):

    def __init__(self, num_channels, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, **kwargs):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(num_channels))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        initialize_weights([self.body], 0.1)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformConvPack(DeformConv):

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


logger = logging.getLogger(__name__)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, *args, extra_offset_mask=False, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


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


url = {'r20f64': ''}


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Dis(nn.Module):

    def __init__(self, loss_type='L1', B=4):
        super(Dis, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'cos':
            self.dot_product, self.square_sum_x, self.square_sum_y = torch.zeros(B), torch.zeros(B), torch.zeros(B)

    def forward(self, x1, x2):
        if self.loss_type == 'L1':
            return self.L1Loss(x1, x2)
        if self.loss_type == 'L2':
            return self.L2Loss(x1, x2)
        if self.loss_type == 'cos':
            return self.cosine_similarity(x1, x2)

    def L1Loss(self, x1, x2):
        loss = torch.sum(torch.abs(x1[:] - x2[:]), dim=1)
        return loss

    def L2Loss(self, x1, x2):
        loss = torch.sum((x1[:] - x2[:]).pow(2), dim=1)
        return loss

    def bit_product_sum(self, x, y):
        return sum([(item[0] * item[1]) for item in zip(x, y)])

    def cosine_similarity(self, x, y, norm=True):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), 'len(x) != len(y)'
        dot_product, square_sum_x, square_sum_y = self.dot_product, self.square_sum_x, self.square_sum_y
        for i in range(x.size()[1]):
            dot_product[:] += x[:, i] * y[:, i]
            square_sum_x[:] += x[:, i] * x[:, i]
            square_sum_y[:] += y[:, i] * y[:, i]
        cos = dot_product / (torch.sqrt(square_sum_x) * torch.sqrt(square_sum_y))
        return 0.5 * cos + 0.5 if norm else cos


class LAM_Module(nn.Module):
    """ Deep attention module"""

    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.dis = Dis('L1')
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
            process:
            reshape x > 2d
            任意两行层特征，求关系置信度。关系置信度定义为 距离求反
            得到置信度矩阵
            矩阵相乘，乘上尺度因子，再与输入相加
            
        """
        m_batchsize, N, C, height, width = x.size()
        energy1 = torch.zeros((4, 11, 11))
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy2 = torch.bmm(proj_query, proj_key)
        for i in range(N):
            for j in range(i, N):
                energy1.data[:, i, j] = self.dis(proj_query[:, i], proj_query[:, j])
                energy1.data[:, j, i] = energy1.data[:, i, j]
        energy = energy1 * energy2
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)
        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""

    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


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


class BFN(nn.Module):

    def __init__(self, num_channels, kernel_size, reduction, n_blocks, block):
        super(BFN, self).__init__()
        branch1 = []
        branch1.append(self._make_blocks(num_channels[0], num_channels[0], kernel_size, reduction, n_blocks, block))
        branch1.append(nn.Conv2d(num_channels[0], num_channels[0], kernel_size, stride=1, padding=1, bias=True))
        branch2 = []
        branch2.append(self._make_blocks(num_channels[1], num_channels[1], kernel_size, reduction, n_blocks, block))
        branch2.append(nn.Conv2d(num_channels[1], num_channels[1], kernel_size, stride=1, padding=1, bias=True))
        branch3 = []
        branch3.append(self._make_blocks(num_channels[2], num_channels[2], kernel_size, reduction, n_blocks, block))
        branch3.append(nn.Conv2d(num_channels[2], num_channels[2], kernel_size, stride=1, padding=1, bias=True))
        self.branch1 = nn.Sequential(*branch1)
        self.branch2 = nn.Sequential(*branch2)
        self.branch3 = nn.Sequential(*branch3)

    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        return nn.Sequential(*blocks)

    def forward(self, x):
        assert type(x) is tuple and len(x) == 3
        res1 = x[0]
        out1 = self.branch1(x[0])
        out1 += res1
        res2 = x[1]
        out2 = self.branch2(x[1])
        out2 += res2
        res3 = x[2]
        out3 = self.branch3(x[2])
        out3 += res3
        return out1, out2, out3


class BFN1(nn.Module):

    def __init__(self, num_channels, kernel_size, reduction, n_blocks, block):
        super(BFN1, self).__init__()
        branch1 = []
        branch1.append(self._make_blocks(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        branch1.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        self.branch1 = nn.Sequential(*branch1)

    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        return nn.Sequential(*blocks)

    def forward(self, x):
        res1 = x
        out1 = self.branch1(x)
        out1 += res1
        return out1


class BFN2(nn.Module):

    def __init__(self, num_channels, kernel_size, reduction, n_blocks, block):
        super(BFN2, self).__init__()
        branch1 = []
        branch1.append(self._make_blocks(num_channels[0], num_channels[0], kernel_size, reduction, n_blocks, block))
        branch1.append(nn.Conv2d(num_channels[0], num_channels[0], kernel_size, stride=1, padding=1, bias=True))
        branch2 = []
        branch2.append(self._make_blocks(num_channels[1], num_channels[1], kernel_size, reduction, n_blocks, block))
        branch2.append(nn.Conv2d(num_channels[1], num_channels[1], kernel_size, stride=1, padding=1, bias=True))
        self.branch1 = nn.Sequential(*branch1)
        self.branch2 = nn.Sequential(*branch2)

    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        return nn.Sequential(*blocks)

    def forward(self, x):
        assert type(x) is tuple and len(x) == 2
        res1 = x[0]
        out1 = self.branch1(x[0])
        out1 += res1
        res2 = x[1]
        out2 = self.branch2(x[1])
        out2 += res2
        return out1, out2


class EoctResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, num_channels, stride=1, downsample=None, res_scale=1, **kwargs):
        super(EoctResBlock, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.downsample = downsample
        self.res_scale = res_scale
        self.conv1 = ops.EoctConv(in_channels, num_channels, stride=stride)
        self.conv2 = ops.EoctConv(num_channels, num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = ops.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = ops.tupleSum(out, residual)
        out = ops.relu(out)
        return out


class EoctBottleneck(nn.Module):

    def __init__(self, in_channels, num_channels, stride=1, downsample=None, res_scale=1, **kwargs):
        super(EoctBottleneck, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.downsample = downsample
        self.res_scale = res_scale
        expand = 6
        linear = 0.8
        self.conv1 = ops.EoctConv(in_channels, ops.tupleMultiply(num_channels, expand), kernel_size=1, padding=1 // 2)
        self.conv2 = ops.EoctConv(ops.tupleMultiply(num_channels, expand), int(ops.tupleMultiply(num_channels, linear)), kernel_size=1, padding=1 // 2)
        self.conv3 = ops.EoctConv(int(ops.tupleMultiply(num_channels, linear)), num_channels, kernel_size=3, padding=kernel_size // 2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = ops.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = ops.tupleSum(out, residual)
        out = ops.relu(out)
        return out


class CAEoctResBlock(nn.Module):

    def __init__(self, in_channels, num_channels, reduction, bias=True, res_scale=1, **kwargs):
        super(CAEoctResBlock, self).__init__()
        self.num_channels = num_channels
        self.res_scale = res_scale
        self.conv1 = ops.EoctConv(in_channels, num_channels, stride=stride)
        self.conv2 = ops.EoctConv(num_channels, num_channels)
        self.caLayer = CAEctBlock(num_channels, num_channels, reduction)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = ops.relu(out)
        out = self.conv2(out)
        out = self.caLayer(out)
        out = ops.tupleSum(out, res)
        out = ops.relu(out)
        return out


class MatrixModel(nn.Module):

    def __init__(self, args):
        super(MatrixModel, self).__init__()
        n_groups = args.n_resgroups
        n_blocks = args.n_resblocks
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = EoctResBlock
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = ops.EoctConv(3, 64)
        modules_body1 = []
        modules_body1.append(self._make_blocks(64, 64, kernel_size, reduction, n_blocks, block))
        modules_body1.append(ops.EoctConv(64, (64, 64), kernel_size))
        modules_body2 = []
        modules_body2.append(self._make_blocks((64, 64), (64, 64), kernel_size, reduction, n_blocks, block))
        modules_body2.append(ops.EoctConv((64, 64), num_channels, kernel_size))
        modules_body3 = []
        modules_body3.append(self._make_blocks(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body3.append(ops.EoctConv(num_channels, 64, kernel_size))
        modules_tail = [ops._UpsampleBlock(num_channels[0], scale=scale), nn.Conv2d(num_channels[0], 3, kernel_size, 1, 1)]
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) for _ in range(n_blocks)]
        blocks.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        res = x
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x += res
        out = self.tail(x)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class RERB(nn.Module):

    def __init__(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        super(RERB, self).__init__()
        blocks = []
        blocks.append(self._make_blocks(in_channels, num_channels, kernel_size, reduction, n_blocks, block))
        blocks.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        self.body = nn.Sequential(*blocks)

    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) for _ in range(n_blocks)]
        blocks.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, x):
        res = x
        x = self.body(x)
        x = ops.tupleSum(x, res)
        x = ops.relu(x)
        return x


blocks_dict = {'BASIC': ResBlock, 'EctBASIC': EoctResBlock, 'EctBOTTLENECK': EoctBottleneck, 'CAEctBASIC': CAEoctResBlock}


class MatrixModelB(nn.Module):

    def __init__(self, args):
        super(MatrixModelB, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        self.fusion_conv1 = ops.EoctConv(num_channels, num_channels, kernel_size)
        self.fusion_conv2 = ops.EoctConv(num_channels, num_channels, kernel_size)
        self.fusion_conv3 = ops.EoctConv(num_channels, num_channels, kernel_size)
        self.conv_last = ops.EoctConv(num_channels, 64, kernel_size)
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        L1_fea = x[0]
        x = self.stage2(x)
        x = self.stage2_conv(x)
        L2_fea = x[1]
        x = self.stage3(x)
        out = self.stage3_conv(x)
        L3_fea = x[2]
        x = L1_fea, L2_fea, L3_fea
        res1 = x
        x = self.fusion_conv1(x)
        x = ops.tupleSum(x, res1)
        res2 = x
        x = self.fusion_conv2(x)
        x = ops.tupleSum(x, res2)
        res3 = x
        x = self.fusion_conv3(x)
        x = ops.tupleSum(x, res3)
        out = self.conv_last(x)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class PDF(nn.Module):
    """ Alignment module using Pyramid, Deformable convolution and Fusion.
    with 3 pyramid levels.
    Bottom-Up.
    """

    def __init__(self, nf=64, groups=8):
        super(PDF, self).__init__()
        self.L1_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_offset_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L3_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv_last = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)

    def forward(self, nbr_fea_l):
        """align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        """
        L1_offset = nbr_fea_l[0]
        L1_offset = self.lrelu(self.L1_offset_conv2(L1_offset))
        L1_fea = self.lrelu(self.L1_dcnpack([nbr_fea_l[0], L1_offset]))
        L1_f = L1_fea
        L2_offset = nbr_fea_l[1]
        L1_offset = self.lrelu(self.L2_offset_conv1(L1_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L1_offset * 2], dim=1)))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L1_fea = self.lrelu(self.L2_offset_conv3(L1_fea))
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L1_fea], dim=1)))
        L2_f = L2_fea
        L3_offset = nbr_fea_l[2]
        L2_offset = self.L3_offset_conv1(L2_offset)
        L3_offset = self.lrelu(self.L3_offset_conv2(torch.cat([L3_offset, L2_offset * 2], dim=1)))
        L3_fea = self.L3_dcnpack([nbr_fea_l[2], L3_offset])
        L2_fea = self.lrelu(self.L3_offset_conv3(L2_fea))
        L3_fea = self.L3_fea_conv(torch.cat([L3_fea, L2_fea], dim=1))
        L3_fea = self.upsample2(L3_fea)
        L2_f = self.upsample(L2_f)
        L_fea = torch.cat([torch.cat([L1_f, L2_f], dim=1), L3_fea], dim=1)
        L_fea = self.lrelu(self.conv_last(L_fea))
        return L_fea


class PD(nn.Module):
    """ module using Pyramid, Deformable convolution
    with 3 pyramid levels.
    Top-down.
    """

    def __init__(self, nf=64, groups=8):
        super(PD, self).__init__()
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, nbr_fea_l):
        """align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        """
        L3_offset = nbr_fea_l[2]
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        L3_f = L3_fea
        L2_offset = nbr_fea_l[1]
        L3_offset = self.upsample(L3_offset)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = self.upsample(L3_fea)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        L2_f = L2_fea
        L1_offset = nbr_fea_l[0]
        L2_offset = self.upsample(L2_offset)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = self.upsample(L2_fea)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        return L1_fea, L2_f, L3_f


class MatrixModelC(nn.Module):

    def __init__(self, args):
        super(MatrixModelC, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        """
        self.pd = PD()
        self.pdf = PDF()
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        L1_fea = x[0]
        x = self.stage2(x)
        x = self.stage2_conv(x)
        L2_fea = x[1]
        x = self.stage3(x)
        x = self.stage3_conv(x)
        L3_fea = x[2]
        x = L1_fea, L2_fea, L3_fea
        x = self.pd(x)
        out = self.pdf(x)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelD(nn.Module):

    def __init__(self, args):
        super(MatrixModelD, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        x = self.stage2(x)
        x = self.stage2_conv(x)
        x = self.stage3(x)
        x = self.stage3_conv(x)
        out = x[0]
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelE(nn.Module):

    def __init__(self, args):
        super(MatrixModelE, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        x = self.stage2(x)
        x = self.stage2_conv(x)
        x = self.stage3(x)
        out = self.stage3_conv(x)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelF(nn.Module):

    def __init__(self, args):
        super(MatrixModelF, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        x = self.stage2(x)
        x = self.stage2_conv(x)
        x = self.stage3(x)
        out = self.stage3_conv(x)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelG(nn.Module):

    def __init__(self, args):
        super(MatrixModelG, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        self.last_conv = nn.Conv2d(64 * 3, 64, kernel_size, 1, 1)
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        out1 = x[0]
        x = self.stage2(x)
        x = self.stage2_conv(x)
        out2 = x[0]
        x = self.stage3(x)
        out = self.stage3_conv(x)
        out2 = torch.cat([out1, out2], dim=1)
        out = torch.cat([out2, out], dim=1)
        out = self.last_conv(out)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class DAM_Module(nn.Module):
    """ Deep attention module"""

    def __init__(self):
        super(DAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)
        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class MatrixModelG2(nn.Module):

    def __init__(self, args):
        super(MatrixModelG2, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        self.da = DAM_Module(64)
        self.last_conv = nn.Conv2d(64 * 3, 64, kernel_size, 1, 1)
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        out1 = x[0].unsqueeze(1)
        x = self.stage2(x)
        x = self.stage2_conv(x)
        out2 = x[0].unsqueeze(1)
        x = self.stage3(x)
        out = self.stage3_conv(x).unsqueeze(1)
        out2 = torch.cat([out1, out2], dim=1)
        out = torch.cat([out2, out], dim=1)
        out = self.da(out)
        out = self.last_conv(out)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


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
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
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
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class MatrixModelF2(nn.Module):

    def __init__(self, args):
        super(MatrixModelF2, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        inter_channels = 64
        self.conv5a = nn.Sequential(nn.Conv2d(64, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv2d(64, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.pa = PAM_Module(inter_channels)
        self.ca = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU())
        self.last_conv = nn.Conv2d(64 * 3, 64, kernel_size, 1, 1)
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        x = self.stage2(x)
        x = self.stage2_conv(x)
        x = self.stage3(x)
        out1 = self.stage3_conv(x)
        feat1 = self.conv5a(out1)
        pa_feat = self.pa(feat1)
        pa_conv = self.conv51(pa_feat)
        feat2 = self.conv5c(out1)
        ca_feat = self.ca(feat2)
        ca_conv = self.conv52(ca_feat)
        feat_sum = torch.cat([pa_conv, ca_conv], dim=1)
        paca_output = torch.cat([feat_sum, out1], dim=1)
        out = self.last_conv(paca_output)
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class GAM_Module(nn.Module):
    """ Global
    attention module"""

    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class MatrixModelH(nn.Module):

    def __init__(self, args):
        super(MatrixModelH, self).__init__()
        num_channels = 64, 64, 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        block = blocks_dict[args.block]
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64, 64), kernel_size)
        modules_stage2 = []
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64, 64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64, 64), num_channels, kernel_size)
        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        self.pa = PAM_Module(64)
        self.pa_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.ca = CAM_Module(64)
        self.ca_conv = nn.Conv2d(64, 64, 3, 1, 1)
        """
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        """
        modules_tail1 = [ops._UpsampleBlock(64, scale=scale), nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        """
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        """

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        x = self.stage1(x)
        x = self.stage1_conv(x)
        x = self.stage2(x)
        x = self.stage2_conv(x)
        x = self.stage3(x)
        out = self.stage3_conv(x)
        pa_out = self.pa(out)
        pa_out = self.pa_conv(pa_out)
        ca_out = self.ca(out)
        ca_out = self.ca_conv(ca_out)
        out = pa_out + ca_out
        out += residual
        out = self.tail1(out)
        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def dataSum(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        assert a.size() == b.size()
        return a + b


class EoctConv(nn.Module):

    def __init__(self, in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True, name=None):
        super(EoctConv, self).__init__()
        self.stride = stride
        if type(in_channels) is tuple and len(in_channels) == 3:
            in_h, in_l, in_ll = in_channels
        elif type(in_channels) is tuple and len(in_channels) == 2:
            in_h, in_l = in_channels
            in_ll = None
        else:
            in_h, in_l, in_ll = in_channels, None, None
        if type(num_channels) is tuple and len(num_channels) == 3:
            num_high, num_low, num_ll = num_channels
        elif type(num_channels) is tuple and len(num_channels) == 2:
            num_high, num_low = num_channels
            num_ll = 0
        else:
            num_high, num_low, num_ll = num_channels, 0, 0
        self.num_high = num_high
        self.num_low = num_low
        self.num_ll = num_ll
        if in_h is not None:
            self.conv2d1 = nn.Conv2d(in_h, num_high, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_high > 0 else None
            self.conv2d2 = nn.Conv2d(in_h, num_low, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_low > 0 else None
            self.conv2d3 = nn.Conv2d(in_h, num_ll, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_ll > 0 else None
        if in_l is not None:
            self.conv2d4 = nn.Conv2d(in_l, num_low, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_low > 0 else None
            self.conv2d5 = nn.Conv2d(in_l, num_high, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_high > 0 else None
            self.conv2d6 = nn.Conv2d(in_l, num_ll, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_ll > 0 else None
        if in_ll is not None:
            self.conv2d7 = nn.Conv2d(in_ll, num_ll, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_ll > 0 else None
            self.conv2d8 = nn.Conv2d(in_ll, num_high, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_high > 0 else None
            self.conv2d9 = nn.Conv2d(in_ll, num_low, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_low > 0 else None
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.pooling1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pooling2 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                nn.init.constant(m.bias, 0)

    def forward(self, data):
        stride = self.stride
        if type(data) is tuple and len(data) == 3:
            data_h, data_l, data_ll = data
        elif type(data) is tuple and len(data) == 2:
            data_h, data_l = data
            data_ll = None
        else:
            data_h, data_l, data_ll = data, None, None
        data_h2l, data_h2h, data_h2ll, data_l2l, data_l2h, data_l2ll, data_ll2ll, data_ll2h, data_ll2l = None, None, None, None, None, None, None, None, None
        if data_h is not None:
            data_h = self.pooling1(data_h) if stride == 2 else data_h
            data_h2h = self.conv2d1(data_h) if self.num_high > 0 else None
            data_h2l = self.pooling1(data_h) if self.num_low > 0 else data_h
            data_h2l = self.conv2d2(data_h2l) if self.num_low > 0 else None
            data_h2ll = self.pooling2(data_h) if self.num_ll > 0 else data_h
            data_h2ll = self.conv2d3(data_h2ll) if self.num_ll > 0 else None
        """processing low frequency group"""
        if data_l is not None:
            data_l2l = self.pooling1(data_l) if self.num_low > 0 and stride == 2 else data_l
            data_l2l = self.conv2d4(data_l2l) if self.num_low > 0 else None
            data_l2h = self.conv2d5(data_l) if self.num_high > 0 else data_l
            data_l2h = self.upsample1(data_l2h) if self.num_high > 0 and stride == 1 else None
            data_l2ll = self.pooling1(data_l) if self.num_ll > 0 else data_l
            data_l2ll = self.conv2d6(data_l2ll) if self.num_ll > 0 else None
        """processing lower frequency group"""
        if data_ll is not None:
            data_ll2ll = self.pooling1(data_ll) if self.num_ll > 0 and stride == 2 else data_ll
            data_ll2ll = self.conv2d7(data_ll2ll) if self.num_ll > 0 else None
            data_ll2h = self.conv2d8(data_ll) if self.num_high > 0 else data_ll
            data_ll2h = self.upsample2(data_ll2h) if self.num_high > 0 and stride == 1 else None
            data_ll2l = self.conv2d9(data_ll) if self.num_low > 0 else data_ll
            data_ll2l = self.upsample1(data_ll2l) if self.num_low > 0 and stride == 1 else None
        """you can force to disable the interaction paths"""
        output = dataSum(dataSum(data_h2h, data_l2h), data_ll2h), dataSum(dataSum(data_h2l, data_l2l), data_ll2l), dataSum(dataSum(data_h2ll, data_l2ll), data_ll2ll)
        if output[2] is None:
            if output[1] is None:
                return output[0]
            else:
                return output[0:2]
        elif output[1] is None:
            return output[0::2]
        else:
            return output


class _UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()
        """
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)"""
        self.conv1 = nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)
        self.conv2 = nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)
        self.relu = nn.ReLU(inplace=True)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pixelshuffle(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pixelshuffle(out)
        return out


class Ada_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, category=2):
        super(Ada_conv, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.category = category
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        mask = self.sigmoid(self.conv0(x.permute(0, 1, 3, 2).contiguous().view(m_batchsize, C, height, width)))
        mask = torch.where(mask < 0.5, torch.full_like(mask, 1), torch.full_like(mask, 0))
        out = self.conv1(x) * mask + self.conv2(x) * (1 - mask)
        return out


class ResAda_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, category=2):
        super(ResAda_conv, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 1, 1, padding=0, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.category = category
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        mask = self.sigmoid(self.conv0(x))
        mask = torch.where(mask < 0.5, torch.full_like(mask, 1), torch.full_like(mask, 0))
        out = self.conv1(x) * mask + self.conv2(x) * (1 - mask)
        return out + x


class FullConvRes(nn.Module):
    """ Full Receptive Field Conv2d Residual Block"""

    def __init__(self, out_channels=64, in_channels=64, K=9):
        super(FullConvRes, self).__init__()
        self.out_channels = out_channels
        self.K = K
        self.gamma = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, K))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        init.xavier_uniform(self.weight)
        init.constant(self.bias, 0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fullconv value + input feature
                attention: B X HW X 9
            process:
            reshape x > 2d
            
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        energy1 = torch.zeros((m_batchsize, height * width, 1))
        for i in range(height * width):
            energy1.data[:, i] = torch.sqrt(energy[:, i, i]).unsqueeze(1)
        energy2 = energy1.permute(0, 2, 1)
        energy_new = energy / energy1.expand_as(energy)
        energy_new = energy_new / energy2.expand_as(energy)
        energy_new = torch.sort(energy_new, dim=-1)[1].float()
        e = torch.chunk(energy_new, self.K, dim=-1)
        for i in range(self.K):
            if i == 0:
                energy_new = e[i][:, :, 0].unsqueeze(2)
            else:
                energy_new = torch.cat([energy_new, e[i][:, :, 0].unsqueeze(2)], dim=2)
        energy_new = energy_new.long()
        ReceptiveField = torch.zeros_like(energy)
        for b in range(m_batchsize):
            for t in range(height * width):
                for k in range(self.K):
                    ReceptiveField.data[b, t, energy_new[b, t, k]] = 1
        out = torch.zeros_like(proj_query)
        for i in range(self.out_channels):
            for j in range(height * width):
                x_out = proj_query[ReceptiveField[:, j].unsqueeze(1).expand_as(proj_query) > 0].view(m_batchsize, C, -1)
                x_K = torch.sum(x_out * self.weight[i].expand_as(x_out), dim=1)
                out.data[:, i, j] = torch.sum(x_K, dim=1) + self.bias[i]
        out = self.relu(out.view(m_batchsize, C, height, width))
        return self.gamma * out + x


class FullConvRes1(nn.Module):
    """ Full Receptive Field Conv2d Residual Block"""

    def __init__(self, out_channels=64, in_channels=64, kernel_size=3):
        super(FullConvRes1, self).__init__()
        self.out_channels = out_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        init.xavier_uniform(self.weight)
        init.constant(self.bias, 0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fullconv value + input feature
                attention: B X HW X 9
            process:
            reshape x > 2d
            
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C // 8, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C // 8, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        max9 = max((1, 9))
        _, ReceptiveFieldIdex = energy.topk(max9, -1, True, False)
        proj_query = x.view(m_batchsize, -1, height * width)
        out = torch.zeros_like(proj_query)
        ReceptiveField = torch.zeros_like(energy)
        for b in range(m_batchsize):
            for t in range(height * width):
                for k in range(9):
                    ReceptiveField.data[b, t, ReceptiveFieldIdex[b, t, k]] = 1
        for i in range(self.out_channels):
            for j in range(height * width):
                x_out = proj_query[ReceptiveField[:, j].unsqueeze(1).expand_as(proj_query) > 0].view(m_batchsize, C, -1)
                x_K = torch.sum(x_out * self.weight[i].expand_as(x_out), dim=1)
                out.data[:, i, j] = torch.sum(x_K, dim=1) + self.bias[i]
        out = self.relu(out.view(m_batchsize, C, height, width))
        return self.gamma * out + x


class FullConv(nn.Module):
    """ Full Receptive Field Conv2d Block"""

    def __init__(self, out_channels=64, in_channels=64, kernel_size=3):
        super(FullConv, self).__init__()
        self.dis = Dis('cos', batchsize=16)
        self.out_channels = out_channels
        self.softmax = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size * kernel_size))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fullconv value + input feature
                attention: B X HW X 9
            process:
            reshape x > 2d
            
        """
        m_batchsize, C, height, width = x.size()
        energy1 = torch.zeros((m_batchsize, height * width, height * width))
        proj_query = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, C, -1)
        energy2 = torch.bmm(proj_query, proj_key)
        for i in range(height * width):
            for j in range(i, height * width):
                energy1.data[:, i, j] = self.dis(proj_query[:, i], proj_query[:, j])
                energy1.data[:, j, i] = energy1.data[:, i, j]
        energy = energy1 * energy2
        maxk = max((1, 9))
        top9, ReceptiveField = energy.topk(maxk, -1, True, False)
        top9 = top9 * ReceptiveField
        score = self.softmax(top9)
        x_in = x.view(m_batchsize, -1, height * width)
        out = x_in
        for i in range(self.out_channels):
            for j in range(height * width):
                x_in = x_in[:, :, ReceptiveField[:, j, :]] * score[:, j, :]
                out[:, i, j] = torch.sum(x_in * self.weight[i].expand_as(x), dim=0)
        out = self.relu(out.view(m_batchsize, C, height, width))
        return out


class MSCALayer(nn.Module):

    def __init__(self):
        pass


class SEDAM_Module(nn.Module):
    """ Deep attention module"""

    def __init__(self, in_dim):
        super(SEDAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.conv_du = nn.Sequential(nn.Conv2d(121, 11, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(11, 121, 1, padding=0, bias=True))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy1 = torch.zeros((m_batchsize, N, 1))
        for i in range(N):
            energy1.data[:, i] = torch.sqrt(energy[:, i, i]).unsqueeze(1)
        energy2 = energy1.permute(0, 2, 1)
        energy = energy / energy1.expand_as(energy)
        energy = energy / energy2.expand_as(energy)
        energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, N, N)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)
        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class MSAM_Module(nn.Module):
    """MultiScale Sptial Attention"""

    def __init__(self, in_dim):
        super(MSAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.conv0 = nn.Conv2d(in_dim, in_dim // 16, 1, 1, 0)
        self.conv = nn.Conv2d(in_dim // 16, in_dim // 16, 3, 1, 1)
        self.atten_conv = nn.Conv2d(in_dim // 16, 1, 1, 1, 0)
        self.last_conv = nn.Conv2d(in_dim // 16 * 4, in_dim, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X HW X HW
        """
        m_batchsize, C, height, width = x.size()
        x1 = self.multi_scale(x)
        proj_query = x1.view(m_batchsize, -1, C * height * width // 16)
        proj_key = x1.view(m_batchsize, -1, C * height * width // 16).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x1.view(m_batchsize, -1, C * height * width // 16)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, -1, height, width)
        out = self.last_conv(out)
        out = self.gamma * out + x
        return out

    def attention(self, x):
        out = self.sigmoid(self.atten_conv(x))
        return out * x + x

    def one_scale(self, x, scale=2):
        m_batchsize, C, height, width = x.size()
        dowsample = nn.AvgPool2d(scale, stride=scale)
        upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        x = dowsample(x)
        x = self.relu(self.conv(x))
        x = upsample(x)
        x = self.attention(x)
        return x

    def multi_scale(self, x):
        x = self.relu(self.conv0(x))
        out = self.conv(x)
        out = out.unsqueeze(1)
        scale_list = [2, 3, 4]
        for scale in scale_list:
            x1 = self.one_scale(x, scale).unsqueeze(1)
            out = torch.cat([out, x1], 1)
        return out


class SAM_Module(nn.Module):
    """SE Sptial Attention"""

    def __init__(self, in_dim):
        super(SAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.pad1 = nn.ReplicationPad2d((0, 0, 1, 0))
        self.pad2 = nn.ReplicationPad2d((1, 0, 0, 0))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X HW X HW
        """
        x, top, left = self.depixel_shuffle(x)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1, height * width)
        proj_key = x.view(m_batchsize, -1, height * width).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        energy1 = torch.zeros((m_batchsize, height * width, 1))
        for i in range(height * width):
            energy1.data[:, i] = torch.sqrt(energy[:, i, i]).unsqueeze(1)
        energy = energy / energy1.expand_as(energy)
        energy1 = energy1.permute(0, 2, 1)
        energy = energy / energy1.expand_as(energy)
        energy = self.softmax(energy)
        out = torch.bmm(proj_query, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)
        out = self.gamma * out + x
        out = self.pixel_shuffle(out)
        if top != 0:
            out = out[:, :, 1:, :]
        if left != 0:
            out = out[:, :, :, 1:]
        return out

    def depixel_shuffle(self, x, upscale_factor=2):
        batch_size, channels, height, width = x.size()
        pdb.set_trace()
        out_channels = channels * upscale_factor ** 2
        top, left = 0, 0
        if height % 2 == 1:
            x = self.pad1(x)
            top = 1
        if width % 2 == 1:
            x = self.pad2(x)
            left = 1
        height = math.ceil(height / upscale_factor)
        width = math.ceil(width / upscale_factor)
        x_view = x.contiguous().view(batch_size, channels, height, upscale_factor, width, upscale_factor)
        shuffle_out = x_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffle_out.view(batch_size, out_channels, height, width), top, left

    def squaremax(self, x, dim=-1):
        x_square = x.pow(2)
        x_sum = torch.sum(x_square, dim=dim, keepdim=True)
        s = x_square / x_sum
        return s

    def logmax(self, x):
        x_log = torch.log(x + 1)
        x_sum = torch.sum(x_log, dim=-1, keepdim=True)
        s = x_log / x_sum
        return s

    def absmax(self, x):
        x_abs = torch.abs(x)
        x_sum = torch.sum(x_abs, dim=-1, keepdim=True)
        s = x_abs / x_sum
        return s


class SECAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(SECAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.conv_du = nn.Sequential(nn.Conv2d(4096, 16, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(16, 4096, 1, padding=0, bias=True))
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
        pdb.set_trace()
        proj_query = x.contiguous().view(m_batchsize, C, -1)
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy1 = torch.zeros((m_batchsize, C, 1))
        for i in range(C):
            energy1.data[:, i] = torch.sqrt(energy[:, i, i]).unsqueeze(1)
        energy2 = energy1.permute(0, 2, 1)
        energy = energy / energy1.expand_as(energy)
        energy = energy / energy2.expand_as(energy)
        energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, C, C)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, -1, height, width)
        out = self.gamma * out + x
        return out


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()])
        self.da = DAM_Module()

    def forward(self, x):
        B, N, C, H, W = x.size()
        x = self.da(x)
        x = x.view(B, N * C, H, W)
        out = self.conv(x).unsqueeze(1)
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):

    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        self.da = DAM_Module()
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        B, N, C, H, W = x.size()
        out = self.da(x)
        out = out.view(B, N * C, H, W)
        out = self.LFF(self.convs(x)).unsqueeze(1) + x
        return out


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.da = DAM_Module()
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1).unsqueeze(1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = torch.cat(RDBs_out, 1)
        B, N, C, H, W = x.size()
        x = self.da(x)
        x = x.view(B, N * C, H, W)
        x = self.GFF(x)
        x += f__1
        return self.UPNet(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'conv': _mock_layer, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CAM_Module,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CSAM_Module,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DAM_Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Dis,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EoctConv,
     lambda: ([], {'in_channels': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FullConvRes1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (GAM_Module,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RCAB,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_UpsampleBlock,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wwlCape_HAN(_paritybench_base):
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

