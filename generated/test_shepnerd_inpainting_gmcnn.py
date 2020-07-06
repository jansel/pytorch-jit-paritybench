import sys
_module = sys.modules[__name__]
del sys
data = _module
data = _module
model = _module
basemodel = _module
basenet = _module
layer = _module
loss = _module
net = _module
options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
utils = _module
network = _module
ops = _module
painter_gmcnn = _module

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


import torch


import numpy as np


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import torchvision.models as models


import torch.autograd as autograd


from functools import reduce


from torch.utils.data import DataLoader


from torchvision import transforms


import torchvision.utils as vutils


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.model_folder
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.model_names = []

    def setInput(self, inputData):
        self.input = inputData

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def update_learning_rate(self):
        pass

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, load_path):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                None
                state_dict = torch.load(load_path)
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose=True):
        None
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    None
                None
        None

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoint_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def forward(self, *input):
        return super(BaseNet, self).forward(*input)

    def test(self, *input):
        with torch.no_grad():
            self.forward(*input)

    def save_network(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.cpu().state_dict(), save_path)

    def load_network(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
        else:
            try:
                self.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = self.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    self.load_state_dict(pretrained_dict)
                    None
                except:
                    None
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            None
                    self.load_state_dict(model_dict)


class Conv2d_BN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_BN, self).__init__()
        self.model = nn.Sequential([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.BatchNorm2d(out_channels)])

    def forward(self, *input):
        return self.model(*input)


class upsampling(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale=2):
        super(upsampling, self).__init__()
        assert isinstance(scale, int)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.scale = scale

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        xout = self.conv(F.interpolate(input=x, size=(h, w), mode='nearest', align_corners=True))
        return xout


class PureUpsampling(nn.Module):

    def __init__(self, scale=2, mode='bilinear'):
        super(PureUpsampling, self).__init__()
        assert isinstance(scale, int)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        if self.mode == 'nearest':
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode)
        else:
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode, align_corners=True)
        return xout


def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma - interval / 2, sigma + interval / 2, size + 1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])
    return out_filter


class GaussianBlurLayer(nn.Module):

    def __init__(self, size, sigma, in_channels=1, stride=1, pad=1):
        super(GaussianBlurLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.ch = in_channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        kernel = gauss_kernel(self.size, self.sigma, self.ch, self.ch)
        kernel_tensor = torch.from_numpy(kernel)
        kernel_tensor = kernel_tensor
        x = self.pad(x)
        blurred = F.conv2d(x, kernel_tensor, stride=self.stride)
        return blurred


class ConfidenceDrivenMaskLayer(nn.Module):

    def __init__(self, size=65, sigma=1.0 / 40, iters=7):
        super(ConfidenceDrivenMaskLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagationLayer = GaussianBlurLayer(size, sigma, pad=32)

    def forward(self, mask):
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagationLayer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence


class VGG19(nn.Module):

    def __init__(self, pool='max'):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return out


class VGG19FeatLayer(nn.Module):

    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iteration=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iteration = power_iteration
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iteration):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *input):
        self._update_u_v()
        return self.module.forward(*input)


class PartialConv(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, ksize=3, stride=1):
        super(PartialConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.fnum = 32
        self.padSize = self.ksize // 2
        self.pad = nn.ReflectionPad2d(self.padSize)
        self.eplison = 1e-05
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=ksize)

    def forward(self, x, mask):
        mask_ch = mask.size(1)
        sum_kernel_np = np.ones((mask_ch, mask_ch, self.ksize, self.ksize), dtype=np.float32)
        sum_kernel = torch.from_numpy(sum_kernel_np)
        x = x * mask / (F.conv2d(mask, sum_kernel, stride=1, padding=self.padSize) + self.eplison)
        x = self.pad(x)
        x = self.conv(x)
        mask = F.max_pool2d(mask, self.ksize, stride=self.stride, padding=self.padSize)
        return x, mask


class GatedConv(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, ksize=3, stride=1, act=F.elu):
        super(GatedConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.act = act
        self.padSize = self.ksize // 2
        self.pad = nn.ReflectionPad2d(self.padSize)
        self.convf = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=ksize)
        self.convm = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=ksize, padding=self.padSize)

    def forward(self, x):
        x = self.pad(x)
        x = self.convf(x)
        x = self.act(x)
        m = self.convm(x)
        m = F.sigmoid(m)
        x = x * m
        return x


class GatedDilatedConv(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=2, act=F.elu):
        super(GatedDilatedConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.act = act
        self.padSize = pad
        self.pad = nn.ReflectionPad2d(self.padSize)
        self.convf = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=ksize, dilation=dilation)
        self.convm = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=ksize, dilation=dilation, padding=self.padSize)

    def forward(self, x):
        x = self.pad(x)
        x = self.convf(x)
        x = self.act(x)
        m = self.convm(x)
        m = F.sigmoid(m)
        x = x * m
        return x


class WGANLoss(nn.Module):

    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -input.mean()
        return {'g_loss': g_loss, 'd_loss': d_loss}


class IDMRFLoss(nn.Module):

    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-05
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT
        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)
        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm
        cosine_dist_l = []
        BatchSize = tar.size(0)
        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)
            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [(self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style
        content_loss_list = [(self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
        return self.style_loss + self.content_loss


class StyleLoss(nn.Module):

    def __init__(self, featlayer=VGG19FeatLayer, style_layers=None):
        super(StyleLoss, self).__init__()
        self.featlayer = featlayer()
        if style_layers is not None:
            self.feat_style_layers = style_layers
        else:
            self.feat_style_layers = {'relu2_2': 1.0, 'relu3_2': 1.0, 'relu4_2': 1.0}

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        feats = x.view(b * c, h * w)
        g = torch.mm(feats, feats.t())
        return g.div(b * c * h * w)

    def _l1loss(self, gen, tar):
        return torch.abs(gen - tar).mean()

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [(self.feat_style_layers[layer] * self._l1loss(self.gram_matrix(gen_vgg_feats[layer]), self.gram_matrix(tar_vgg_feats[layer]))) for layer in self.feat_style_layers]
        style_loss = reduce(lambda x, y: x + y, style_loss_list)
        return style_loss


class ContentLoss(nn.Module):

    def __init__(self, featlayer=VGG19FeatLayer, content_layers=None):
        super(ContentLoss, self).__init__()
        self.featlayer = featlayer()
        if content_layers is not None:
            self.feat_content_layers = content_layers
        else:
            self.feat_content_layers = {'relu4_2': 1.0}

    def _l1loss(self, gen, tar):
        return torch.abs(gen - tar).mean()

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        content_loss_list = [(self.feat_content_layers[layer] * self._l1loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_content_layers]
        content_loss = reduce(lambda x, y: x + y, content_loss_list)
        return content_loss


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_x, w_x = x.size()[2:]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        loss = torch.sum(h_tv) + torch.sum(w_tv)
        return loss


class GMCNN(BaseNet):

    def __init__(self, in_channels, out_channels, cnum=32, act=F.elu, norm=F.instance_norm, using_norm=False):
        super(GMCNN, self).__init__()
        self.act = act
        self.using_norm = using_norm
        if using_norm is True:
            self.norm = norm
        else:
            self.norm = None
        ch = cnum
        self.EB1 = []
        self.EB2 = []
        self.EB3 = []
        self.decoding_layers = []
        self.EB1_pad_rec = []
        self.EB2_pad_rec = []
        self.EB3_pad_rec = []
        self.EB1.append(nn.Conv2d(in_channels, ch, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch, ch * 2, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=2))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=4))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=8))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=16))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(PureUpsampling(scale=4))
        self.EB1_pad_rec = [3, 3, 3, 3, 3, 3, 6, 12, 24, 48, 3, 3, 0]
        self.EB2.append(nn.Conv2d(in_channels, ch, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch, ch * 2, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=2))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=4))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=8))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=16))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB2.append(nn.Conv2d(ch * 4, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(PureUpsampling(scale=2))
        self.EB2_pad_rec = [2, 2, 2, 2, 2, 2, 4, 8, 16, 32, 2, 2, 0, 2, 2, 0]
        self.EB3.append(nn.Conv2d(in_channels, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=2))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=4))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=8))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=16))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB3.append(nn.Conv2d(ch * 4, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB3.append(nn.Conv2d(ch * 2, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1))
        self.EB3_pad_rec = [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 1, 0, 1, 1, 0, 1, 1]
        self.decoding_layers.append(nn.Conv2d(ch * 7, ch // 2, kernel_size=3, stride=1))
        self.decoding_layers.append(nn.Conv2d(ch // 2, out_channels, kernel_size=3, stride=1))
        self.decoding_pad_rec = [1, 1]
        self.EB1 = nn.ModuleList(self.EB1)
        self.EB2 = nn.ModuleList(self.EB2)
        self.EB3 = nn.ModuleList(self.EB3)
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        padlen = 49
        self.pads = [0] * padlen
        for i in range(padlen):
            self.pads[i] = nn.ReflectionPad2d(i)
        self.pads = nn.ModuleList(self.pads)

    def forward(self, x):
        x1, x2, x3 = x, x, x
        for i, layer in enumerate(self.EB1):
            pad_idx = self.EB1_pad_rec[i]
            x1 = layer(self.pads[pad_idx](x1))
            if self.using_norm:
                x1 = self.norm(x1)
            if pad_idx != 0:
                x1 = self.act(x1)
        for i, layer in enumerate(self.EB2):
            pad_idx = self.EB2_pad_rec[i]
            x2 = layer(self.pads[pad_idx](x2))
            if self.using_norm:
                x2 = self.norm(x2)
            if pad_idx != 0:
                x2 = self.act(x2)
        for i, layer in enumerate(self.EB3):
            pad_idx = self.EB3_pad_rec[i]
            x3 = layer(self.pads[pad_idx](x3))
            if self.using_norm:
                x3 = self.norm(x3)
            if pad_idx != 0:
                x3 = self.act(x3)
        x_d = torch.cat((x1, x2, x3), 1)
        x_d = self.act(self.decoding_layers[0](self.pads[self.decoding_pad_rec[0]](x_d)))
        x_d = self.decoding_layers[1](self.pads[self.decoding_pad_rec[1]](x_d))
        x_out = torch.clamp(x_d, -1, 1)
        return x_out


class Discriminator(BaseNet):

    def __init__(self, in_channels, cnum=32, fc_channels=8 * 8 * 32 * 4, act=F.elu, norm=None, spectral_norm=True):
        super(Discriminator, self).__init__()
        self.act = act
        self.norm = norm
        self.embedding = None
        self.logit = None
        ch = cnum
        self.layers = []
        if spectral_norm:
            self.layers.append(SpectralNorm(nn.Conv2d(in_channels, ch, kernel_size=5, padding=2, stride=2)))
            self.layers.append(SpectralNorm(nn.Conv2d(ch, ch * 2, kernel_size=5, padding=2, stride=2)))
            self.layers.append(SpectralNorm(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2)))
            self.layers.append(SpectralNorm(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, padding=2, stride=2)))
            self.layers.append(SpectralNorm(nn.Linear(fc_channels, 1)))
        else:
            self.layers.append(nn.Conv2d(in_channels, ch, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.Conv2d(ch, ch * 2, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.Conv2d(ch * 4, ch * 4, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.Linear(fc_channels, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.norm is not None:
                x = self.norm(x)
            x = self.act(x)
        self.embedding = x.view(x.size(0), -1)
        self.logit = self.layers[-1](self.embedding)
        return self.logit


class GlobalLocalDiscriminator(BaseNet):

    def __init__(self, in_channels, cnum=32, g_fc_channels=16 * 16 * 32 * 4, l_fc_channels=8 * 8 * 32 * 4, act=F.elu, norm=None, spectral_norm=True):
        super(GlobalLocalDiscriminator, self).__init__()
        self.act = act
        self.norm = norm
        self.global_discriminator = Discriminator(in_channels=in_channels, fc_channels=g_fc_channels, cnum=cnum, act=act, norm=norm, spectral_norm=spectral_norm)
        self.local_discriminator = Discriminator(in_channels=in_channels, fc_channels=l_fc_channels, cnum=cnum, act=act, norm=norm, spectral_norm=spectral_norm)

    def forward(self, x_g, x_l):
        x_global = self.global_discriminator(x_g)
        x_local = self.local_discriminator(x_l)
        return x_global, x_local


def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=True):
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0 + sz0, of1:of1 + sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[of0, sz0, of1, sz1]], dtype=int)
    return mask, rect


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_stroke_mask(im_size, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask


def generate_mask(type, im_size, mask_size):
    if type == 'rect':
        return generate_rect_mask(im_size, mask_size)
    else:
        return generate_stroke_mask(im_size), None


def init_weights(net, init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    None
    net.apply(init_func)


class InpaintingModel_GMCNN(BaseModel):

    def __init__(self, in_channels, act=F.elu, norm=None, opt=None):
        super(InpaintingModel_GMCNN, self).__init__()
        self.opt = opt
        self.init(opt)
        self.confidence_mask_layer = ConfidenceDrivenMaskLayer()
        self.netGM = GMCNN(in_channels, out_channels=3, cnum=opt.g_cnum, act=act, norm=norm)
        init_weights(self.netGM)
        self.model_names = ['GM']
        if self.opt.phase == 'test':
            return
        self.netD = None
        self.optimizer_G = torch.optim.Adam(self.netGM.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.optimizer_D = None
        self.wganloss = None
        self.recloss = nn.L1Loss()
        self.aeloss = nn.L1Loss()
        self.mrfloss = None
        self.lambda_adv = opt.lambda_adv
        self.lambda_rec = opt.lambda_rec
        self.lambda_ae = opt.lambda_ae
        self.lambda_gp = opt.lambda_gp
        self.lambda_mrf = opt.lambda_mrf
        self.G_loss = None
        self.G_loss_reconstruction = None
        self.G_loss_mrf = None
        self.G_loss_adv, self.G_loss_adv_local = None, None
        self.G_loss_ae = None
        self.D_loss, self.D_loss_local = None, None
        self.GAN_loss = None
        self.gt, self.gt_local = None, None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None
        self.completed, self.completed_local = None, None
        self.completed_logit, self.completed_local_logit = None, None
        self.gt_logit, self.gt_local_logit = None, None
        self.pred = None
        if self.opt.pretrain_network is False:
            if self.opt.mask_type == 'rect':
                self.netD = GlobalLocalDiscriminator(3, cnum=opt.d_cnum, act=act, g_fc_channels=opt.img_shapes[0] // 16 * opt.img_shapes[1] // 16 * opt.d_cnum * 4, l_fc_channels=opt.mask_shapes[0] // 16 * opt.mask_shapes[1] // 16 * opt.d_cnum * 4, spectral_norm=self.opt.spectral_norm)
            else:
                self.netD = GlobalLocalDiscriminator(3, cnum=opt.d_cnum, act=act, spectral_norm=self.opt.spectral_norm, g_fc_channels=opt.img_shapes[0] // 16 * opt.img_shapes[1] // 16 * opt.d_cnum * 4, l_fc_channels=opt.img_shapes[0] // 16 * opt.img_shapes[1] // 16 * opt.d_cnum * 4)
            init_weights(self.netD)
            self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=opt.lr, betas=(0.5, 0.9))
            self.wganloss = WGANLoss()
            self.mrfloss = IDMRFLoss()

    def initVariables(self):
        self.gt = self.input['gt']
        mask, rect = generate_mask(self.opt.mask_type, self.opt.img_shapes, self.opt.mask_shapes)
        self.mask_01 = torch.from_numpy(mask).repeat([self.opt.batch_size, 1, 1, 1])
        self.mask = self.confidence_mask_layer(self.mask_01)
        if self.opt.mask_type == 'rect':
            self.rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
            self.gt_local = self.gt[:, :, self.rect[0]:self.rect[0] + self.rect[1], self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.gt_local = self.gt
        self.im_in = self.gt * (1 - self.mask_01)
        self.gin = torch.cat((self.im_in, self.mask_01), 1)

    def forward_G(self):
        self.G_loss_reconstruction = self.recloss(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss_reconstruction = self.G_loss_reconstruction / torch.mean(self.mask_01)
        self.G_loss_ae = self.aeloss(self.pred * (1 - self.mask_01), self.gt.detach() * (1 - self.mask_01))
        self.G_loss_ae = self.G_loss_ae / torch.mean(1 - self.mask_01)
        self.G_loss = self.lambda_rec * self.G_loss_reconstruction + self.lambda_ae * self.G_loss_ae
        if self.opt.pretrain_network is False:
            self.completed_logit, self.completed_local_logit = self.netD(self.completed, self.completed_local)
            self.G_loss_mrf = self.mrfloss((self.completed_local + 1) / 2.0, (self.gt_local.detach() + 1) / 2.0)
            self.G_loss = self.G_loss + self.lambda_mrf * self.G_loss_mrf
            self.G_loss_adv = -self.completed_logit.mean()
            self.G_loss_adv_local = -self.completed_local_logit.mean()
            self.G_loss = self.G_loss + self.lambda_adv * (self.G_loss_adv + self.G_loss_adv_local)

    def forward_D(self):
        self.completed_logit, self.completed_local_logit = self.netD(self.completed.detach(), self.completed_local.detach())
        self.gt_logit, self.gt_local_logit = self.netD(self.gt, self.gt_local)
        self.D_loss_local = nn.ReLU()(1.0 - self.gt_local_logit).mean() + nn.ReLU()(1.0 + self.completed_local_logit).mean()
        self.D_loss = nn.ReLU()(1.0 - self.gt_logit).mean() + nn.ReLU()(1.0 + self.completed_logit).mean()
        self.D_loss = self.D_loss + self.D_loss_local

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.initVariables()
        self.pred = self.netGM(self.gin)
        self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)
        if self.opt.mask_type == 'rect':
            self.completed_local = self.completed[:, :, self.rect[0]:self.rect[0] + self.rect[1], self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.completed_local = self.completed
        if self.opt.pretrain_network is False:
            for i in range(self.opt.D_max_iters):
                self.optimizer_D.zero_grad()
                self.optimizer_G.zero_grad()
                self.forward_D()
                self.backward_D()
                self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_losses(self):
        l = {'G_loss': self.G_loss.item(), 'G_loss_rec': self.G_loss_reconstruction.item(), 'G_loss_ae': self.G_loss_ae.item()}
        if self.opt.pretrain_network is False:
            l.update({'G_loss_adv': self.G_loss_adv.item(), 'G_loss_adv_local': self.G_loss_adv_local.item(), 'D_loss': self.D_loss.item(), 'G_loss_mrf': self.G_loss_mrf.item()})
        return l

    def get_current_visuals(self):
        return {'input': self.im_in.cpu().detach().numpy(), 'gt': self.gt.cpu().detach().numpy(), 'completed': self.completed.cpu().detach().numpy()}

    def get_current_visuals_tensor(self):
        return {'input': self.im_in.cpu().detach(), 'gt': self.gt.cpu().detach(), 'completed': self.completed.cpu().detach()}

    def evaluate(self, im_in, mask):
        im_in = torch.from_numpy(im_in).type(torch.FloatTensor) / 127.5 - 1
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        im_in = im_in * (1 - mask)
        xin = torch.cat((im_in, mask), 1)
        ret = self.netGM(xin) * mask + im_in * (1 - mask)
        ret = (ret.cpu().detach().numpy() + 1) * 127.5
        return ret.astype(np.uint8)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (ContentLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (Discriminator,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 128, 128])], {}),
     False),
    (GMCNN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     False),
    (GlobalLocalDiscriminator,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 256, 256]), torch.rand([4, 4, 128, 128])], {}),
     False),
    (IDMRFLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (PartialConv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     False),
    (PureUpsampling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyleLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGG19FeatLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_shepnerd_inpainting_gmcnn(_paritybench_base):
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

    def test_011(self):
        self._check(*TESTCASES[11])

