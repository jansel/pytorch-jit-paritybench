import sys
_module = sys.modules[__name__]
del sys
core = _module
checkpoint = _module
data_loader = _module
model = _module
solver = _module
utils = _module
wing = _module
main = _module
metrics = _module
eval = _module
fid = _module
lpips = _module

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


import copy


import math


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from collections import namedtuple


from copy import deepcopy


from functools import partial


from scipy import linalg


class ResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=
        False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, actv=nn.
        LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):

    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1], [-1, 8.0, -1], [-1, -1, -1]]
            ).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1,
            1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):

    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2), nn.Conv2d(dim_in, 3, 1, 1, 0))
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True,
                downsample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim,
                w_hpf=w_hpf, upsample=True))
            dim_in = dim_out
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim,
                w_hpf=w_hpf))
        if w_hpf > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else
                'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if masks is not None and x.size(2) in [32, 64, 128]:
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if masks is not None and x.size(2) in [32, 64, 128]:
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)


class MappingNetwork(nn.Module):

    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.
                ReLU(), nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]
        return s


class StyleEncoder(nn.Module):

    def __init__(self, img_size=256, style_dim=64, num_domains=2,
        max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]
        return s


class Discriminator(nn.Module):

    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]
        return out


def r1_reg(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None,
    masks=None):
    assert (z_trg is None) != (x_ref is None)
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:
            s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(), fake=loss_fake.item(), reg=
        loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=
    None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)
    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    loss = (loss_adv + args.lambda_sty * loss_sty - args.lambda_ds *
        loss_ds + args.lambda_cyc * loss_cyc)
    return loss, Munch(adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds
        .item(), cyc=loss_cyc.item())


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) for ext in [
        'png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class HourGlass(nn.Module):

    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
            out_channels=256, kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))
        self.add_module('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class AddCoordsTh(nn.Module):

    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            x_coords = torch.arange(height).unsqueeze(1).expand(height, width
                ).float()
            y_coords = torch.arange(width).unsqueeze(0).expand(height, width
                ).float()
            x_coords = x_coords / (height - 1) * 2 - 1
            y_coords = y_coords / (width - 1) * 2 - 1
            coords = torch.stack([x_coords, y_coords], dim=0)
            if self.with_r:
                rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2)
                    )
                rr = (rr / torch.max(rr)).unsqueeze(0)
                coords = torch.cat([coords, rr], dim=0)
            self.coords = coords.unsqueeze(0).to(device)
            self.x_coords = x_coords.to(device)
            self.y_coords = y_coords.to(device)

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.repeat(x.size(0), 1, 1, 1)
        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(self.x_coords)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, self
                .x_coords, zero_tensor).to(zero_tensor.device)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, self
                .y_coords, zero_tensor).to(zero_tensor.device)
            coords = torch.cat([coords, xx_boundary_channel,
                yy_boundary_channel], dim=1)
        x_and_coords = torch.cat([x, coords], dim=1)
        return x_and_coords


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""

    def __init__(self, height, width, with_r, with_boundary, in_channels,
        first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(*args, in_channels=in_channels, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1,
            bias=False, dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes), nn.
                ReLU(True), nn.Conv2d(in_planes, out_planes, 1, 1, bias=False))

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


def get_preds_fromhm(hm):
    max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.
        size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[(i), (j), :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor([hm_[pY, pX + 1] - hm_[pY, pX - 1],
                    hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(0.25))
    preds.add_(-0.5)
    return preds


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return torch.where(x < thres, torch.zeros_like(x), x)


def resize(x, p=2):
    """Resize heatmaps."""
    return x ** p


OPPAIR = namedtuple('OPPAIR', 'shift resize')


IDXPAIR = namedtuple('IDXPAIR', 'start end')


class InceptionV3(nn.Module):

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(inception.Conv2d_1a_3x3, inception.
            Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(
            kernel_size=3, stride=2))
        self.block2 = nn.Sequential(inception.Conv2d_3b_1x1, inception.
            Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(in_channels,
            out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)


class LPIPS(nn.Module):

    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1)
        self.sigma = torch.tensor([0.458, 0.448, 0.45]).view(1, 3, 1, 1)

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load('metrics/lpips_weights.ckpt')
        else:
            state_dict = torch.load('metrics/lpips_weights.ckpt',
                map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights
            ):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap) ** 2))
        return lpips_value


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_clovaai_stargan_v2(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ResBlk(*[], **{'dim_in': 4, 'dim_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConvBlock(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Conv1x1(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

