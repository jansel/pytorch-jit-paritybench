import sys
_module = sys.modules[__name__]
del sys
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
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
rdn = _module
vdsr = _module
option = _module
template = _module
trainer = _module
utility = _module
videotester = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
model = _module
common = _module
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
rdn = _module
vdsr = _module
trainer = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
model = _module
cbam = _module
common = _module
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
rdn = _module
se = _module
vdsr = _module
wdsr = _module
test_SRimages_rename = _module
trainer = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
model = _module
common = _module
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
rdn = _module
vdsr = _module
trainer = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
model = _module
common = _module
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
rdn = _module
vdsr = _module
trainer = _module
loss = _module
adversarial = _module
discriminator = _module
vgg = _module
model = _module
common = _module
ddbpn = _module
edsr = _module
mdsr = _module
rcan = _module
rdn = _module
vdsr = _module
trainer = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.model_zoo


import math


import torch.nn.init as init


import torch.nn.utils as utils


from torch.nn.parameter import Parameter


import torch.backends.cudnn as cudnn


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
                loss_function = getattr(module, 'VGG')(loss_type[3:],
                    rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight),
                'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}
                    )
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args
                .n_GPUs))
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
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
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **
            kwargs))
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
            self.optimizer = optim.Adam(self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-08, lr=1e-05)
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
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake
                    ) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(),
                        inputs=hat, retain_graph=True, create_graph=True,
                        only_inputs=True)[0]
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
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g,
                label_real)
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
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3,
                padding=1, stride=stride, bias=False), nn.BatchNorm2d(
                _out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
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
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn
            .LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
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
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=
            args.resume, cpu=args.cpu)
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
            return self.forward_x8(x, forward_function=forward_function)
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
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        load_from = None
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.get_model()
                    .url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format
                (resume)), **kwargs)
        if load_from:
            self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[a[:, :, 0:h_size, 0:w_size], a[:, :, 0:h_size, w -
            w_size:w], a[:, :, h - h_size:h, 0:w_size], a[:, :, h - h_size:
            h, w - w_size:w]] for a in args]
        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:i + n_GPUs], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.append(_y)
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] = _list_y[1][:, :, :h_half, w_size -
                w + w_half:]
            _y[:, :, h_half:, :w_half] = _list_y[2][:, :, h_size - h +
                h_half:, :w_half]
            _y[:, :, h_half:, w_half:] = _list_y[3][:, :, h_size - h +
                h_half:, w_size - w + w_half:]
        if len(y) == 1:
            y = y[0]
        return y

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
            ret = torch.Tensor(tfnp).to(self.device)
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

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std
        =(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale3 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale4 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )

    def forward(self, x):
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        yn_1 = G_yn * self.scale1
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        yn_2 = Gyn_1 * self.scale2
        yn_2 = yn_2 + yn
        Gyn_2 = self.relu3(yn_2)
        Gyn_2 = self.conv3(Gyn_2)
        yn_3 = Gyn_2 * self.scale3
        yn_3 = yn_3 + yn_1
        Gyn_3 = self.relu4(yn_3)
        Gyn_3 = self.conv4(Gyn_3)
        yn_4 = Gyn_3 * self.scale4
        out = yn_4 + yn_2
        return out


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


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {(2): (6, 2, 2), (4): (8, 4, 2), (8): (
        12, 8, 2)}[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d
    return conv_f(in_channels, out_channels, kernel_size, stride=stride,
        padding=padding)


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels,
            scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])

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
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True,
                i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, 
                False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3,
            padding=1)]
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
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=
        False, act=nn.ReLU(True), res_scale=1):
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

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
        n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=
            True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(
            n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize -
            1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


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
                loss_function = getattr(module, 'VGG')(loss_type[3:],
                    rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight),
                'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}
                    )
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args
                .n_GPUs))
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
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
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **
            kwargs))
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
            self.optimizer = optim.Adam(self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-08, lr=1e-05)
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
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake
                    ) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(),
                        inputs=hat, retain_graph=True, create_graph=True,
                        only_inputs=True)[0]
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
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g,
                label_real)
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
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3,
                padding=1, stride=stride, bias=False), nn.BatchNorm2d(
                _out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
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
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn
            .LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
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
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=
            args.resume, cpu=args.cpu)
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
            return self.forward_x8(x, forward_function=forward_function)
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
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        load_from = None
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.get_model()
                    .url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format
                (resume)), **kwargs)
        if load_from:
            self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[a[:, :, 0:h_size, 0:w_size], a[:, :, 0:h_size, w -
            w_size:w], a[:, :, h - h_size:h, 0:w_size], a[:, :, h - h_size:
            h, w - w_size:w]] for a in args]
        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:i + n_GPUs], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.append(_y)
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] = _list_y[1][:, :, :h_half, w_size -
                w + w_half:]
            _y[:, :, h_half:, :w_half] = _list_y[2][:, :, h_size - h +
                h_half:, :w_half]
            _y[:, :, h_half:, w_half:] = _list_y[3][:, :, h_size - h +
                h_half:, w_size - w + w_half:]
        if len(y) == 1:
            y = y[0]
        return y

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
            ret = torch.Tensor(tfnp).to(self.device)
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

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std
        =(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale3 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale4 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )

    def forward(self, x):
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        yn_1 = G_yn * self.scale1
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        yn_2 = Gyn_1 * self.scale2
        yn_2 = yn_2 + yn
        Gyn_2 = self.relu3(yn_2)
        Gyn_2 = self.conv3(Gyn_2)
        yn_3 = Gyn_2 * self.scale3
        yn_3 = yn_3 + yn_1
        Gyn_3 = self.relu4(yn_3)
        Gyn_3 = self.conv4(Gyn_3)
        yn_4 = Gyn_3 * self.scale4
        out = yn_4 + yn_2
        return out


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


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels,
            scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])

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
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True,
                i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, 
                False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3,
            padding=1)]
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
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=
        False, act=nn.ReLU(True), res_scale=1):
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

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
        n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=
            True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(
            n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize -
            1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


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
            elif loss_type == 'SL1':
                loss_function = nn.SmoothL1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(loss_type[3:],
                    rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight),
                'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}
                    )
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args
                .n_GPUs))
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
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
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **
            kwargs))
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
            self.optimizer = optim.Adam(self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-08, lr=1e-05)
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
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake
                    ) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(),
                        inputs=hat, retain_graph=True, create_graph=True,
                        only_inputs=True)[0]
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
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g,
                label_real)
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
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3,
                padding=1, stride=stride, bias=False), nn.BatchNorm2d(
                _out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
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
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn
            .LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
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
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=
            args.resume, cpu=args.cpu)
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
            return self.forward_x8(x, forward_function=forward_function)
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
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        load_from = None
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.get_model()
                    .url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format
                (resume)), **kwargs)
        if load_from:
            self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[a[:, :, 0:h_size, 0:w_size], a[:, :, 0:h_size, w -
            w_size:w], a[:, :, h - h_size:h, 0:w_size], a[:, :, h - h_size:
            h, w - w_size:w]] for a in args]
        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:i + n_GPUs], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.append(_y)
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] = _list_y[1][:, :, :h_half, w_size -
                w + w_half:]
            _y[:, :, h_half:, :w_half] = _list_y[2][:, :, h_size - h +
                h_half:, :w_half]
            _y[:, :, h_half:, w_half:] = _list_y[3][:, :, h_size - h +
                h_half:, w_size - w + w_half:]
        if len(y) == 1:
            y = y[0]
        return y

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
            ret = torch.Tensor(tfnp).to(self.device)
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


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg',
        'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, 
            gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(
            gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(
                    x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(
                    x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=
                    (x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3
            ).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1)
            .unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg',
        'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
            pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std
        =(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv5 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv6 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.relu5 = nn.PReLU(n_feats, 0.25)
        self.relu6 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True
            )
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad
            =True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True
            )
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]),
            requires_grad=True)
        self.se0 = SE(n_feats)
        self.se1 = SE(n_feats)
        self.se2 = SE(n_feats)

    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        k1 = self.relu4(k1)
        k1 = self.se0(k1)
        k1 = self.conv4(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        k2 = self.relu5(k2)
        k2 = self.se1(k2)
        k2 = self.conv5(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        k3 = self.relu6(k3)
        k3 = self.se2(k3)
        k3 = self.conv6(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out


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


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels,
            scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])

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
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True,
                i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, 
                False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3,
            padding=1)]
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
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=
        False, act=nn.ReLU(True), res_scale=1):
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

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
        n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=
            True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(
            n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize -
            1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg',
        'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp0 = nn.Sequential(Flatten(), nn.Linear(gate_channels, 
            gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(
            gate_channels // reduction_ratio, gate_channels))
        self.mlp1 = nn.Sequential(Flatten(), nn.Linear(gate_channels, 
            gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(
            gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(
                    x.size(2), x.size(3)))
                channel_att_raw = self.mlp0(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(
                    x.size(2), x.size(3)))
                channel_att_raw = self.mlp1(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3
            ).expand_as(x)
        return x * scale


class SE(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg',
        'max'], no_spatial=False):
        super(SE, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
            pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out


class Block(nn.Module):

    def __init__(self, n_feats, kernel_size, block_feats, wn, res_scale=1,
        act=nn.ReLU(True)):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        body.append(wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding
            =kernel_size // 2)))
        body.append(act)
        body.append(wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding
            =kernel_size // 2)))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class WDSR(nn.Module):

    def __init__(self, args):
        super(WDSR, self).__init__()
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([args.
            r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])
        head = []
        head.append(wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3 // 2)))
        body = []
        for i in range(n_resblocks):
            body.append(Block(n_feats, kernel_size, args.block_feats, wn=wn,
                res_scale=args.res_scale, act=act))
        tail = []
        out_feats = scale * scale * args.n_colors
        tail.append(wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2)))
        skip = []
        skip.append(wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5 // 2)))
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 127.5 + self.rgb_mean * 255
        return x


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
                loss_function = getattr(module, 'VGG')(loss_type[3:],
                    rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight),
                'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}
                    )
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args
                .n_GPUs))
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
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
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **
            kwargs))
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
            self.optimizer = optim.Adam(self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-08, lr=1e-05)
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
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake
                    ) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(),
                        inputs=hat, retain_graph=True, create_graph=True,
                        only_inputs=True)[0]
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
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g,
                label_real)
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
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3,
                padding=1, stride=stride, bias=False), nn.BatchNorm2d(
                _out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
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
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn
            .LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
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
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=
            args.resume, cpu=args.cpu)
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
            return self.forward_x8(x, forward_function=forward_function)
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
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        load_from = None
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.get_model()
                    .url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format
                (resume)), **kwargs)
        if load_from:
            self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[a[:, :, 0:h_size, 0:w_size], a[:, :, 0:h_size, w -
            w_size:w], a[:, :, h - h_size:h, 0:w_size], a[:, :, h - h_size:
            h, w - w_size:w]] for a in args]
        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:i + n_GPUs], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.append(_y)
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] = _list_y[1][:, :, :h_half, w_size -
                w + w_half:]
            _y[:, :, h_half:, :w_half] = _list_y[2][:, :, h_size - h +
                h_half:, :w_half]
            _y[:, :, h_half:, w_half:] = _list_y[3][:, :, h_size - h +
                h_half:, w_size - w + w_half:]
        if len(y) == 1:
            y = y[0]
        return y

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
            ret = torch.Tensor(tfnp).to(self.device)
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

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std
        =(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True
            )

    def forward(self, x):
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        G_yn = self.relu3(G_yn)
        G_yn = self.conv3(G_yn)
        yn_1 = G_yn + yn
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        Gyn_1 = self.relu4(Gyn_1)
        Gyn_1 = self.conv4(Gyn_1)
        yn_2 = Gyn_1 + G_yn
        yn_2 = yn_2 * self.scale1
        out = yn_2 + yn
        return out


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


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels,
            scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])

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
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True,
                i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, 
                False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3,
            padding=1)]
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
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=
        False, act=nn.ReLU(True), res_scale=1):
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

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
        n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=
            True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(
            n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize -
            1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


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
                loss_function = getattr(module, 'VGG')(loss_type[3:],
                    rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight),
                'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}
                    )
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args
                .n_GPUs))
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
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
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **
            kwargs))
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
            self.optimizer = optim.Adam(self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-08, lr=1e-05)
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
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake
                    ) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(),
                        inputs=hat, retain_graph=True, create_graph=True,
                        only_inputs=True)[0]
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
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g,
                label_real)
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
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3,
                padding=1, stride=stride, bias=False), nn.BatchNorm2d(
                _out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
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
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn
            .LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
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
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=
            args.resume, cpu=args.cpu)
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
            return self.forward_x8(x, forward_function=forward_function)
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
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        load_from = None
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.get_model()
                    .url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format
                (resume)), **kwargs)
        if load_from:
            self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[a[:, :, 0:h_size, 0:w_size], a[:, :, 0:h_size, w -
            w_size:w], a[:, :, h - h_size:h, 0:w_size], a[:, :, h - h_size:
            h, w - w_size:w]] for a in args]
        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:i + n_GPUs], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.append(_y)
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] = _list_y[1][:, :, :h_half, w_size -
                w + w_half:]
            _y[:, :, h_half:, :w_half] = _list_y[2][:, :, h_size - h +
                h_half:, :w_half]
            _y[:, :, h_half:, w_half:] = _list_y[3][:, :, h_size - h +
                h_half:, w_size - w + w_half:]
        if len(y) == 1:
            y = y[0]
        return y

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
            ret = torch.Tensor(tfnp).to(self.device)
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

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std
        =(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True
            )

    def forward(self, x):
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        G_yn = self.relu3(G_yn)
        G_yn = self.conv3(G_yn)
        yn_1 = G_yn + yn
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        Gyn_1 = self.relu4(Gyn_1)
        Gyn_1 = self.conv4(Gyn_1)
        yn_2 = Gyn_1 + G_yn
        yn_2 = yn_2 * self.scale1
        out = yn_2 + yn
        return out


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


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels,
            scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])

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
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True,
                i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, 
                False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3,
            padding=1)]
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
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=
        False, act=nn.ReLU(True), res_scale=1):
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

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
        n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=
            True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(
            n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize -
            1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


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
                loss_function = getattr(module, 'VGG')(loss_type[3:],
                    rgb_range=args.rgb_range)
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            self.loss.append({'type': loss_type, 'weight': float(weight),
                'function': loss_function})
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None}
                    )
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        for l in self.loss:
            if l['function'] is not None:
                None
                self.loss_module.append(l['function'])
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half':
            self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args
                .n_GPUs))
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
            plt.plot(axis, self.log[:, (i)].numpy(), label=label)
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
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **
            kwargs))
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
            self.optimizer = optim.Adam(self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-08, lr=1e-05)
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
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake
                    ) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(),
                        inputs=hat, retain_graph=True, create_graph=True,
                        only_inputs=True)[0]
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
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g,
                label_real)
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
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3,
                padding=1, stride=stride, bias=False), nn.BatchNorm2d(
                _out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
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
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn
            .LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
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
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=
            args.resume, cpu=args.cpu)
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
            return self.forward_x8(x, forward_function=forward_function)
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
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        load_from = None
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.get_model()
                    .url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format
                (resume)), **kwargs)
        if load_from:
            self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[a[:, :, 0:h_size, 0:w_size], a[:, :, 0:h_size, w -
            w_size:w], a[:, :, h - h_size:h, 0:w_size], a[:, :, h - h_size:
            h, w - w_size:w]] for a in args]
        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:i + n_GPUs], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.append(_y)
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] = _list_y[1][:, :, :h_half, w_size -
                w + w_half:]
            _y[:, :, h_half:, :w_half] = _list_y[2][:, :, h_size - h +
                h_half:, :w_half]
            _y[:, :, h_half:, w_half:] = _list_y[3][:, :, h_size - h +
                h_half:, w_size - w + w_half:]
        if len(y) == 1:
            y = y[0]
        return y

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
            ret = torch.Tensor(tfnp).to(self.device)
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

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std
        =(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True
            )
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True
            )
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad
            =True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True
            )
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]),
            requires_grad=True)

    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out


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


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels,
            scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr,
            scale, up), nn.PReLU(nr)])

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
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True,
                i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, 
                False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3,
            padding=1)]
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
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=
        False, act=nn.ReLU(True), res_scale=1):
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

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
        n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=
            True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(
            n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize -
            1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_HolmesShuan_OISR_PyTorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(MeanShift(*[], **{'rgb_range': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(CALayer(*[], **{'channel': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_002(self):
        self._check(RDB_Conv(*[], **{'inChannels': 4, 'growRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(RDB(*[], **{'growRate0': 4, 'growRate': 4, 'nConvLayers': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(BasicConv(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ChannelPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(SpatialGate(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

