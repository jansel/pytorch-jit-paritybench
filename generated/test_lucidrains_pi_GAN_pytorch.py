import sys
_module = sys.modules[__name__]
del sys
pi_gan_pytorch = _module
coordconv = _module
nerf = _module
pi_gan_pytorch = _module
setup = _module

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


import torch.nn as nn


import torch.nn.functional as F


import math


from functools import partial


from torch import nn


from torch import einsum


from torch.autograd import grad as torch_grad


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.optim import Adam


from torch.optim.lr_scheduler import LambdaLR


import torchvision


from torchvision.utils import save_image


import torchvision.transforms as T


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class Sine(nn.Module):

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def exists(val):
    return val is not None


class Siren(nn.Module):

    def __init__(self, dim_in, dim_out, w0=1.0, c=6.0, is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        w_std = 1 / dim if self.is_first else math.sqrt(c / dim) / w0
        weight.uniform_(-w_std, w_std)
        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma=None, beta=None):
        out = F.linear(x, self.weight, self.bias)
        if exists(gamma):
            out = out * gamma
        if exists(beta):
            out = out + beta
        out = self.activation(out)
        return out


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, lr_mul=0.1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)


class MappingNetwork(nn.Module):

    def __init__(self, *, dim, dim_out, depth=3, lr_mul=0.1):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])
        self.net = nn.Sequential(*layers)
        self.to_gamma = nn.Linear(dim, dim_out)
        self.to_beta = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)


class SirenNet(nn.Module):

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1.0, w0_initial=30.0, use_bias=True, final_activation=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            self.layers.append(Siren(dim_in=layer_dim_in, dim_out=dim_hidden, w0=layer_w0, use_bias=use_bias, is_first=is_first))
        self.last_layer = Siren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)
        return self.last_layer(x)


class SirenGenerator(nn.Module):

    def __init__(self, *, dim, dim_hidden, siren_num_layers=8):
        super().__init__()
        self.mapping = MappingNetwork(dim=dim, dim_out=dim_hidden)
        self.siren = SirenNet(dim_in=3, dim_hidden=dim_hidden, dim_out=dim_hidden, num_layers=siren_num_layers)
        self.to_alpha = nn.Linear(dim_hidden, 1)
        self.to_rgb_siren = Siren(dim_in=dim_hidden, dim_out=dim_hidden)
        self.to_rgb = nn.Linear(dim_hidden, 3)

    def forward(self, latent, coors, batch_size=8192):
        gamma, beta = self.mapping(latent)
        outs = []
        for coor in coors.split(batch_size):
            gamma_, beta_ = map(lambda t: rearrange(t, 'n -> () n'), (gamma, beta))
            x = self.siren(coor, gamma_, beta_)
            alpha = self.to_alpha(x)
            x = self.to_rgb_siren(x, gamma, beta)
            rgb = self.to_rgb(x)
            out = torch.cat((rgb, alpha), dim=-1)
            outs.append(out)
        return torch.cat(outs)


def compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, num_samples, randomize=True):
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples)
    if randomize is True:
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = depth_values + torch.rand(noise_shape) * (far_thresh - near_thresh) / num_samples
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values


def meshgrid_xy(tensor1, tensor2):
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def get_ray_bundle(height, width, focal_length, tform_cam2world):
    ii, jj = meshgrid_xy(torch.arange(width), torch.arange(height))
    directions = torch.stack([(ii - width * 0.5) / focal_length, -(jj - height * 0.5) / focal_length, -torch.ones_like(ii)], dim=-1)
    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def cumprod_exclusive(tensor):
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.0
    return cumprod


def render_volume_density(radiance_field, ray_origins, depth_values):
    sigma_a = F.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([10000000000.0], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)
    return rgb_map, depth_map, acc_map


def get_image_from_nerf_model(model, latents, height, width, focal_length=140, tform_cam2world=torch.eye(4), near_thresh=2.0, far_thresh=6.0, depth_samples_per_ray=32):
    tform_cam2world = tform_cam2world
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length, tform_cam2world)
    query_points, depth_values = compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray)
    flattened_query_points = query_points.reshape((-1, 3))
    images = []
    for latent in latents.unbind(0):
        predictions = []
        predictions.append(model(latent, flattened_query_points))
        radiance_field_flattened = torch.cat(predictions, dim=0)
        unflattened_shape = list(query_points.shape[:-1]) + [4]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
        rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)
        image = rearrange(rgb_predicted, 'h w c -> c h w')
        images.append(image)
    return torch.stack(images)


class Generator(nn.Module):

    def __init__(self, *, image_size, dim, dim_hidden, siren_num_layers):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.nerf_model = SirenGenerator(dim=dim, dim_hidden=dim_hidden, siren_num_layers=siren_num_layers)

    def set_image_size(self, image_size):
        self.image_size = image_size

    def forward(self, latents):
        image_size = self.image_size
        device, b = latents.device, latents.shape[0]
        generated_images = get_image_from_nerf_model(self.nerf_model, latents, image_size, image_size)
        return generated_images


class DiscriminatorBlock(nn.Module):

    def __init__(self, dim, dim_out):
        super().__init__()
        self.res = CoordConv(dim, dim_out, kernel_size=1, stride=2)
        self.net = nn.Sequential(CoordConv(dim, dim_out, kernel_size=3, padding=1), leaky_relu(), CoordConv(dim_out, dim_out, kernel_size=3, padding=1), leaky_relu())
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = self.down(x)
        x = x + res
        return x


class Discriminator(nn.Module):

    def __init__(self, image_size, init_chan=64, max_chan=400, init_resolution=32, add_layer_iters=10000):
        super().__init__()
        resolutions = math.log2(image_size)
        assert resolutions.is_integer(), 'image size must be a power of 2'
        assert math.log2(init_resolution).is_integer(), 'initial resolution must be power of 2'
        resolutions = int(resolutions)
        layers = resolutions - 1
        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        chans = [init_chan, *chans]
        final_chan = chans[-1]
        self.from_rgb_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.image_size = image_size
        self.resolutions = list(map(lambda t: 2 ** (7 - t), range(layers)))
        for resolution, in_chan, out_chan in zip(self.resolutions, chans[:-1], chans[1:]):
            from_rgb_layer = nn.Sequential(CoordConv(3, in_chan, kernel_size=1), leaky_relu()) if resolution >= init_resolution else None
            self.from_rgb_layers.append(from_rgb_layer)
            self.layers.append(DiscriminatorBlock(dim=in_chan, dim_out=out_chan))
        self.final_conv = CoordConv(final_chan, 1, kernel_size=2)
        self.add_layer_iters = add_layer_iters
        self.register_buffer('alpha', torch.tensor(0.0))
        self.register_buffer('resolution', torch.tensor(init_resolution))
        self.register_buffer('iterations', torch.tensor(0.0))

    def increase_resolution_(self):
        if self.resolution >= self.image_size:
            return
        self.alpha += self.alpha + (1 - self.alpha)
        self.iterations.fill_(0.0)
        self.resolution *= 2

    def update_iter_(self):
        self.iterations += 1
        self.alpha -= 1 / self.add_layer_iters
        self.alpha.clamp_(min=0.0)

    def forward(self, img):
        x = img
        for resolution, from_rgb, layer in zip(self.resolutions, self.from_rgb_layers, self.layers):
            if self.resolution < resolution:
                continue
            if self.resolution == resolution:
                x = from_rgb(x)
            if bool(resolution == self.resolution // 2) and bool(self.alpha > 0):
                x_down = F.interpolate(img, scale_factor=0.5)
                x = x * (1 - self.alpha) + from_rgb(x_down) * self.alpha
            x = layer(x)
        out = self.final_conv(x)
        return out


class piGAN(nn.Module):

    def __init__(self, *, image_size, dim, init_resolution=32, generator_dim_hidden=256, siren_num_layers=6, add_layer_iters=10000):
        super().__init__()
        self.dim = dim
        self.G = Generator(image_size=image_size, dim=dim, dim_hidden=generator_dim_hidden, siren_num_layers=siren_num_layers)
        self.D = Discriminator(image_size=image_size, add_layer_iters=add_layer_iters, init_resolution=init_resolution)


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


class ImageDataset(Dataset):

    def __init__(self, folder, image_size, transparent=False, aug_prob=0.0, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'
        self.create_transform(image_size)

    def create_transform(self, image_size):
        self.transform = T.Compose([T.Lambda(partial(resize_to_minimum_size, image_size)), T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def gradient_penalty(images, output, weight=10):
    batch_size, device = images.shape[0], images.device
    gradients = torch_grad(outputs=output, inputs=images, grad_outputs=torch.ones(output.size(), device=device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    l2 = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return weight * l2


def sample_generator(G, batch_size):
    dim = G.dim
    rand_latents = torch.randn(batch_size, dim)
    return G(rand_latents)


def to_value(t):
    return t.clone().detach().item()


class Trainer(nn.Module):

    def __init__(self, *, gan, folder, add_layers_iters=10000, batch_size=8, gradient_accumulate_every=4, sample_every=100, log_every=10, num_train_steps=50000, lr_gen=5e-05, lr_discr=0.0004, target_lr_gen=1e-05, target_lr_discr=0.0001, lr_decay_span=10000):
        super().__init__()
        gan.D.add_layer_iters = add_layers_iters
        self.add_layers_iters = add_layers_iters
        self.gan = gan
        self.optim_D = Adam(self.gan.D.parameters(), betas=(0, 0.9), lr=lr_discr)
        self.optim_G = Adam(self.gan.G.parameters(), betas=(0, 0.9), lr=lr_gen)
        D_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + target_lr_discr / lr_discr * min(i / lr_decay_span, 1)
        G_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + target_lr_gen / lr_gen * min(i / lr_decay_span, 1)
        self.sched_D = LambdaLR(self.optim_D, D_decay_fn)
        self.sched_G = LambdaLR(self.optim_G, G_decay_fn)
        self.iterations = 0
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps
        self.log_every = log_every
        self.sample_every = sample_every
        self.gradient_accumulate_every = gradient_accumulate_every
        self.dataset = ImageDataset(folder=folder, image_size=gan.D.resolution.item())
        self.dataloader = cycle(DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True))
        self.last_loss_D = 0
        self.last_loss_G = 0

    def step(self):
        D, G, batch_size, dim, accumulate_every = self.gan.D, self.gan.G, self.batch_size, self.gan.dim, self.gradient_accumulate_every
        if self.iterations % self.add_layers_iters == 0:
            if self.iterations != 0:
                D.increase_resolution_()
            image_size = D.resolution.item()
            G.set_image_size(image_size)
            self.dataset.create_transform(image_size)
        apply_gp = self.iterations % 4 == 0
        D.train()
        loss_D = 0
        for _ in range(accumulate_every):
            images = next(self.dataloader)
            images = images.requires_grad_()
            real_out = D(images)
            fake_imgs = sample_generator(G, batch_size)
            fake_out = D(fake_imgs.clone().detach())
            divergence = (F.relu(1 + real_out) + F.relu(1 - fake_out)).mean()
            loss = divergence
            if apply_gp:
                gp = gradient_penalty(images, real_out)
                self.last_loss_gp = to_value(gp)
                loss = loss + gp
            (loss / accumulate_every).backward()
            loss_D += to_value(divergence) / accumulate_every
        self.last_loss_D = loss_D
        self.optim_D.step()
        self.optim_D.zero_grad()
        G.train()
        loss_G = 0
        for _ in range(accumulate_every):
            fake_out = sample_generator(G, batch_size)
            loss = D(fake_out).mean()
            (loss / accumulate_every).backward()
            loss_G += to_value(loss) / accumulate_every
        self.last_loss_G = loss_G
        self.optim_G.step()
        self.optim_G.zero_grad()
        self.sched_D.step()
        self.sched_G.step()
        self.iterations += 1
        D.update_iter_()

    def forward(self):
        for _ in trange(self.num_train_steps):
            self.step()
            if self.iterations % self.log_every == 0:
                None
            if self.iterations % self.sample_every == 0:
                i = self.iterations // self.sample_every
                imgs = sample_generator(self.gan.G, 4)
                imgs.clamp_(0.0, 1.0)
                save_image(imgs, f'./{i}.png', nrow=2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {'image_size': 4}),
     lambda: ([torch.rand([4, 400, 64, 64])], {}),
     False),
    (DiscriminatorBlock,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MappingNetwork,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Siren,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SirenNet,
     lambda: ([], {'dim_in': 4, 'dim_hidden': 4, 'dim_out': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_pi_GAN_pytorch(_paritybench_base):
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

