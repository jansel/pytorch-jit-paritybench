import sys
_module = sys.modules[__name__]
del sys
cyclegan = _module
discriminator = _module
discriminator_loss = _module
perceptual_loss = _module
pix2pix = _module
translation_generator = _module
dataset = _module
logs = _module
utils = _module
vgg_utils = _module
test = _module
train = _module

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


from torch import nn


from itertools import chain


from torch.autograd import Variable


from math import log


from torch.autograd import grad


import torch.nn as nn


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision.transforms import Compose


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import Resize


from torchvision.transforms import ToPILImage


from torchvision.transforms import CenterCrop


from torchvision.transforms import Lambda


from torchvision.transforms import Normalize


from torchvision.transforms import ToTensor


from torchvision.transforms import RandomCrop


from torchvision.transforms import functional as F


from torchvision.utils import make_grid


import torch.nn.functional as f


from torchvision.models import vgg19


from torchvision import transforms as T


from torch.optim import Adam


from torch.optim.lr_scheduler import MultiStepLR


class Discriminator(nn.Module):
    """Discriminator with input as features or image"""

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.dis_input_sizes = opt.dis_input_sizes
        self.dis_output_sizes = opt.dis_output_sizes
        num_down_blocks = int(log(opt.dis_input_sizes[0] // max(opt.dis_output_sizes[-1], 4), 2))
        in_channels = opt.dis_input_num_channels[0]
        out_channels = opt.dis_num_channels
        padding = (opt.dis_kernel_size - 1) // 2
        padding_io = opt.dis_kernel_size_io // 2
        spatial_size = opt.dis_input_sizes[0]
        self.blocks = nn.ModuleList()
        self.input_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        for i in range(num_down_blocks):
            self.blocks += [nn.Sequential(nn.Conv2d(in_channels, out_channels, opt.dis_kernel_size, 2, padding), nn.LeakyReLU(0.2, True))]
            in_channels = out_channels
            spatial_size //= 2
            if spatial_size in opt.dis_input_sizes:
                in_channels = opt.dis_input_num_channels[len(self.input_blocks) + 1]
                self.input_blocks += [nn.Sequential(nn.Conv2d(in_channels, out_channels, opt.dis_kernel_size_io, 1, padding_io), nn.LeakyReLU(0.2, True))]
                in_channels = out_channels * 2
            if spatial_size in opt.dis_output_sizes:
                self.output_blocks += [nn.Conv2d(in_channels, 1, opt.dis_kernel_size_io, 1, padding_io)]
            out_channels = min(out_channels * 2, opt.dis_max_channels)
        if opt.dis_output_sizes[-1] == 1:
            self.output_blocks += [nn.Conv2d(out_channels, 1, 4, 4)]
        self.apply(utils.weights_init)

    def forward(self, img_dst, img_src=None, encoder=None):
        input = img_dst
        if img_src is not None:
            input = torch.cat([input, img_src, 0])
        inputs = encoder(input)
        if img_src is not None:
            for i in range(len(inputs)):
                b, c, h, w = inputs[i].shape
                inputs[i] = inputs[i].view(b // 2, c * 2, h, w)
        output = inputs[0]
        spatial_size = output.shape[2]
        input_idx = 0
        output_idx = 0
        preds = []
        for block in self.blocks:
            output = block(output)
            spatial_size //= 2
            if spatial_size in self.dis_input_sizes:
                input = self.input_blocks[input_idx](inputs[input_idx + 1])
                output = torch.cat([output, input], 1)
                input_idx += 1
            if spatial_size in self.dis_output_sizes:
                preds += [self.output_blocks[output_idx](output)]
                output_idx += 1
        if 1 in self.dis_output_sizes:
            preds += [self.output_blocks[output_idx](output)]
        return preds


class DiscriminatorLoss(nn.Module):

    def __init__(self, opt):
        super(DiscriminatorLoss, self).__init__()
        self.gpu_id = opt.gpu_ids[0]
        if opt.dis_adv_loss_type == 'gan':
            self.crit = nn.BCEWithLogitsLoss()
        elif opt.dis_adv_loss_type == 'lsgan':
            self.crit = nn.MSELoss()
        self.labels_real = []
        self.labels_fake = []
        for size in opt.dis_output_sizes:
            shape = opt.batch_size, 1, size, size
            self.labels_real += [Variable(torch.ones(shape))]
            self.labels_fake += [Variable(torch.zeros(shape))]

    def __call__(self, dis, img_real_dst, img_fake_dst=None, aux_real_dst=None, img_real_src=None, enc=None):
        outputs_real = dis(img_real_dst, img_real_src, enc)
        if img_fake_dst is not None:
            outputs_fake = dis(img_fake_dst, img_real_src, enc)
        loss = 0
        losses_adv = []
        for i in range(len(outputs_real)):
            losses_adv += [self.crit(outputs_real[i], self.labels_real[i])]
            if img_fake_dst is not None:
                losses_adv[-1] += self.crit(outputs_fake[i], self.labels_fake[i])
                losses_adv[-1] *= 0.5
            loss += losses_adv[-1]
        losses_adv = [loss_adv.data.item() for loss_adv in losses_adv]
        losses_adv = [sum(losses_adv)]
        return loss, losses_adv


class VGGModified(nn.Module):

    def __init__(self, vgg19_orig, slope):
        super(VGGModified, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module(str(0), vgg19_orig.features[0])
        self.features.add_module(str(1), nn.LeakyReLU(slope, True))
        self.features.add_module(str(2), vgg19_orig.features[2])
        self.features.add_module(str(3), nn.LeakyReLU(slope, True))
        self.features.add_module(str(4), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(5), vgg19_orig.features[5])
        self.features.add_module(str(6), nn.LeakyReLU(slope, True))
        self.features.add_module(str(7), vgg19_orig.features[7])
        self.features.add_module(str(8), nn.LeakyReLU(slope, True))
        self.features.add_module(str(9), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(10), vgg19_orig.features[10])
        self.features.add_module(str(11), nn.LeakyReLU(slope, True))
        self.features.add_module(str(12), vgg19_orig.features[12])
        self.features.add_module(str(13), nn.LeakyReLU(slope, True))
        self.features.add_module(str(14), vgg19_orig.features[14])
        self.features.add_module(str(15), nn.LeakyReLU(slope, True))
        self.features.add_module(str(16), vgg19_orig.features[16])
        self.features.add_module(str(17), nn.LeakyReLU(slope, True))
        self.features.add_module(str(18), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(19), vgg19_orig.features[19])
        self.features.add_module(str(20), nn.LeakyReLU(slope, True))
        self.features.add_module(str(21), vgg19_orig.features[21])
        self.features.add_module(str(22), nn.LeakyReLU(slope, True))
        self.features.add_module(str(23), vgg19_orig.features[23])
        self.features.add_module(str(24), nn.LeakyReLU(slope, True))
        self.features.add_module(str(25), vgg19_orig.features[25])
        self.features.add_module(str(26), nn.LeakyReLU(slope, True))
        self.features.add_module(str(27), nn.AvgPool2d((2, 2), (2, 2)))
        self.features.add_module(str(28), vgg19_orig.features[28])
        self.features.add_module(str(29), nn.LeakyReLU(slope, True))
        self.features.add_module(str(30), vgg19_orig.features[30])
        self.features.add_module(str(31), nn.LeakyReLU(slope, True))
        self.features.add_module(str(32), vgg19_orig.features[32])
        self.features.add_module(str(33), nn.LeakyReLU(slope, True))
        self.features.add_module(str(34), vgg19_orig.features[34])
        self.features.add_module(str(35), nn.LeakyReLU(slope, True))
        self.features.add_module(str(36), nn.AvgPool2d((2, 2), (2, 2)))
        self.classifier = nn.Sequential()
        self.classifier.add_module(str(0), vgg19_orig.classifier[0])
        self.classifier.add_module(str(1), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(2), nn.Dropout2d(p=0.5))
        self.classifier.add_module(str(3), vgg19_orig.classifier[3])
        self.classifier.add_module(str(4), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(5), nn.Dropout2d(p=0.5))
        self.classifier.add_module(str(6), vgg19_orig.classifier[6])

    def forward(self, x):
        return self.classifier(self.features.forward(x))


def get_vgg19(model_name, model_path):
    if model_name == 'vgg19_caffe':
        model = vgg19()
    elif model_name == 'vgg19_pytorch':
        model = vgg19(pretrained=True)
    elif model_name == 'vgg19_pytorch_modified':
        model = VGGModified(vgg19(), 0.2)
        model.load_state_dict(torch.load('%s/%s.pkl' % (model_path, model_name))['state_dict'])
    model.classifier = nn.Sequential(utils.View(), *model.classifier._modules.values())
    vgg = model.features
    vgg_classifier = model.classifier
    names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5', 'torch_view', 'fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8']
    model = nn.Sequential()
    for n, m in zip(names, list(vgg) + list(vgg_classifier)):
        model.add_module(n, m)
    if model_name == 'vgg19_caffe':
        model.load_state_dict(torch.load('%s/%s.pth' % (model_path, model_name)))
    return model


def get_pretrained_net(name, path):
    """ Load pretrained network """
    if name == 'vgg19_caffe':
        os.system('wget -O vgg19_caffe.pth --no-check-certificate -nc https://www.dropbox.com/s/xlbdo688dy4keyk/vgg19-caffe.pth?dl=1')
        vgg = get_vgg19(name, path)
    elif name == 'vgg19_pytorch':
        vgg = get_vgg19(name, path)
    elif name == 'vgg19_pytorch_modified':
        vgg = get_vgg19(name, path)
    else:
        assert False, 'Wrong pretrained network name'
    return vgg


class FeatureExtractor(nn.Module):
    """ 
        Assumes input image is
        if `input_range` is 'sigmoid' -- in range [0,1]
                            'tanh'                [-1, 1]
    """

    def __init__(self, input_range='sigmoid', net_type='vgg19_pytorch_modified', preprocessing_type='corresponding', layers='1,6,11,20,29', net_path='.'):
        super(FeatureExtractor, self).__init__()
        if input_range == 'sigmoid':
            self.preprocess_range = lambda x: x
        elif input_range == 'tanh':
            self.preprocess_range = lambda x: (x + 1.0) / 2.0
        else:
            assert False, 'Wrong input_range'
        self.preprocessing_type = preprocessing_type
        if preprocessing_type == 'corresponding':
            if 'caffe' in net_type:
                self.preprocessing_type = 'caffe'
            elif 'pytorch' in net_type:
                self.preprocessing_type = 'pytorch'
            else:
                assert False, 'Unknown net_type'
        if self.preprocessing_type == 'caffe':
            self.vgg_mean = nn.Parameter(torch.FloatTensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1))
            self.vgg_std = None
        elif self.preprocessing_type == 'pytorch':
            self.vgg_mean = nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.vgg_std = nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            assert False, 'Unknown preprocessing_type'
        net = get_pretrained_net(net_type, net_path)
        self.blocks = nn.ModuleList()
        layers_indices = [int(i) for i in layers.split(',')]
        layers_indices.insert(0, -1)
        for i in range(len(layers_indices) - 1):
            layers_i = []
            for j in range(layers_indices[i] + 1, layers_indices[i + 1] + 1):
                layers_i += [net[j]]
            self.blocks += [nn.Sequential(*layers_i)]
        self.eval()

    def forward(self, input):
        input = input.clone()
        input = self.preprocess_range(input)
        if self.preprocessing_type == 'caffe':
            r, g, b = torch.chunk(input, 3, dim=1)
            bgr = torch.cat([b, g, r], 1)
            out = bgr * 255 - self.vgg_mean
        elif self.preprocessing_type == 'pytorch':
            input = input - self.vgg_mean
            input = input / self.vgg_std
        output = input
        outputs = []
        for block in self.blocks:
            output = block(output)
            outputs.append(output)
        return outputs


class Generator(nn.Module):
    """Translation generator architecture by Johnston et al"""

    def __init__(self, opt):
        super(Generator, self).__init__()
        norm_layer = utils.get_norm_layer(opt.gen_norm_layer)
        upsampling_layer = utils.get_upsampling_layer(opt.gen_upsampling_layer)
        num_down_blocks = int(log(opt.image_size // opt.gen_latent_size, 2))
        num_up_blocks = int(log(opt.image_size // opt.gen_latent_size, 2))
        in_channels = opt.gen_num_channels
        padding = (opt.gen_kernel_size - 1) // 2
        bias = norm_layer != nn.BatchNorm2d
        layers = [nn.Conv2d(3, in_channels, 7, 1, 3, bias=False), nn.ReLU(True)]
        for i in range(num_down_blocks):
            out_channels = min(in_channels * 2, opt.gen_max_channels)
            layers += [nn.Conv2d(in_channels, out_channels, opt.gen_kernel_size, 2, padding, bias), norm_layer(out_channels), nn.ReLU(True)]
            in_channels = out_channels
        for i in range(opt.gen_num_res_blocks):
            layers += [utils.ResBlock(in_channels, norm_layer)]
        for i in range(num_up_blocks):
            out_channels = opt.gen_num_channels * 2 ** (num_up_blocks - i - 1)
            out_channels = max(min(out_channels, opt.gen_max_channels), opt.gen_num_channels)
            layers += upsampling_layer(in_channels, out_channels, opt.gen_kernel_size, 2, bias)
            layers += [norm_layer(out_channels), nn.ReLU(True)]
            in_channels = out_channels
        layers += [nn.Conv2d(out_channels, 3, 7, 1, 3, bias=False), nn.Tanh()]
        self.generator = nn.Sequential(*layers)
        self.apply(utils.weights_init)

    def forward(self, image):
        return self.generator(image)


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.gpu_id = opt.gpu_ids[0]
        self.weights_path = os.path.join('runs', opt.experiment_name, 'checkpoints')
        self.gen_B = Generator(opt)
        if opt.adv_loss_weight:
            self.dis_B = Discriminator(opt)
        utils.load_checkpoint(self, opt.which_epoch, opt.pretrained_gen_path)
        None
        num_params = 0
        for p in self.gen_B.parameters():
            num_params += p.numel()
        None
        None
        self.gen_params = self.gen_B.parameters()
        if opt.adv_loss_weight:
            None
            num_params = 0
            for p in self.dis_B.parameters():
                num_params += p.numel()
            None
            None
            self.dis_params = self.dis_B.parameters()
            self.crit_dis = DiscriminatorLoss(opt)
        self.adv_weight = opt.adv_loss_weight
        if opt.mse_loss_type == 'perceptual' or opt.adv_loss_weight and opt.dis_use_encoder:
            if opt.enc_type[:5] == 'vgg19':
                self.layers = '1,6,11,20,29'
            self.enc = FeatureExtractor(input_range='tanh', net_type=opt.enc_type, layers=self.layers).eval()
            None
            None
            None
        else:
            self.enc = None
        self.crit_mse = utils.get_loss_layer(opt.mse_loss_type, self.enc)
        self.mse_weight = opt.mse_loss_weight
        self.gen_B = nn.DataParallel(self.gen_B, opt.gpu_ids)
        if opt.adv_loss_weight:
            self.dis_B = nn.DataParallel(self.dis_B, opt.gpu_ids)
        if self.enc is not None:
            self.enc = nn.DataParallel(self.enc, opt.gpu_ids)

    def forward(self, inputs):
        real_A, real_B = inputs
        self.real_B = Variable(real_B)
        self.real_A = Variable(real_A)
        self.fake_B = self.gen_B(self.real_A)

    def backward_G(self):
        self.loss_ident_B = self.crit_mse(self.fake_B, self.real_B)
        loss_mse = self.loss_ident_B
        if self.adv_weight:
            loss_dis, _ = self.crit_dis(dis=self.dis_B, img_real_dst=self.fake_B, img_real_src=self.real_A, enc=self.enc)
        else:
            loss_dis = 0
        loss_G = loss_mse * self.mse_weight + loss_dis * self.adv_weight
        if self.training:
            loss_G.backward()
        self.loss_ident_B = self.loss_ident_B.data.item()

    def backward_D(self):
        loss_dis, self.losses_adv_B = self.crit_dis(dis=self.dis_B, img_real_dst=self.real_B, img_fake_dst=self.fake_B.detach(), img_real_src=self.real_A, enc=self.enc)
        loss_D = loss_dis
        if self.training:
            loss_D.backward()

    def train(self, mode=True):
        self.training = mode
        self.gen_B.train(mode)
        if self.adv_weight:
            self.dis_B.train(mode)
        return self


class Matcher(nn.Module):

    def __init__(self, matching_type='features', matching_loss='L1', average_loss=True):
        super(Matcher, self).__init__()
        if matching_type == 'features':
            self.get_stats = self.gram_matrix
        elif matching_type == 'features':
            self.get_stats = lambda x: x
        matching_loss = matching_loss.lower()
        if matching_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif matching_loss == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif matching_loss == 'l1':
            self.criterion = nn.L1Loss()
        self.average_loss = average_loss

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def __call__(self, input_feats, target_feats):
        input_stats = [self.get_stats(features) for features in input_feats]
        target_stats = [self.get_stats(features) for features in target_feats]
        loss = 0
        for input, target in zip(input_stats, target_stats):
            loss += self.criterion(input, target.detach())
        if self.average_loss:
            loss /= len(input_stats)
        return loss


class PerceptualLoss(nn.Module):

    def __init__(self, input_range='sigmoid', net_type='vgg19_pytorch_modified', preprocessing_type='corresponding', matching_loss='L1', match=[{'matching_type': 'features', 'layers': '1,6,11,20,29'}], average_loss=True, extractor=None):
        super(PerceptualLoss, self).__init__()
        self.average_loss = average_loss
        self.matchers = []
        layers = ''
        for m in match:
            self.matchers += [Matcher(m['matching_type'], matching_loss, average_loss)]
            layers += m['layers'] + ','
        layers = layers[:-1]
        layers = np.asarray(layers.split(',')).astype(int)
        layers = np.unique(layers)
        self.layers_idx_m = []
        for m in match:
            layers_m = [int(i) for i in m['layers'].split(',')]
            layers_idx_m = []
            for l in layers_m:
                layers_idx_m += [np.argwhere(layers == l)[0, 0]]
            self.layers_idx_m += [layers_idx_m]
        layers = ','.join(layers.astype(str))
        if extractor is None:
            self.extractor = FeatureExtractor(input_range, net_type, preprocessing_type, layers)
        else:
            self.extractor = extractor

    def forward(self, input, target):
        input_feats = self.extractor(input)
        target_feats = self.extractor(target)
        loss = 0
        for i, m in enumerate(self.matchers):
            input_feats_m = [input_feats[j] for j in self.layers_idx_m[i]]
            target_feats_m = [target_feats[j] for j in self.layers_idx_m[i]]
            loss += m(input_feats_m, target_feats_m)
        if self.average_loss:
            loss /= len(self.matchers)
        return loss


class Identity(nn.Module):

    def __init__(self, num_channels=None):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


class ResBlock(nn.Module):

    def __init__(self, in_channels, norm_layer):
        super(ResBlock, self).__init__()
        norm_layer = Identity if norm_layer is None else norm_layer
        bias = norm_layer == Identity
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias), norm_layer(in_channels), nn.ReLU(True), nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias), norm_layer(in_channels))

    def forward(self, input):
        return input + self.block(input)


class ConcatBlock(nn.Module):

    def __init__(self, enc_channels, out_channels, nonlinear_layer=nn.ReLU, norm_layer=None, norm_layer_cat=None, kernel_size=3):
        super(ConcatBlock, self).__init__()
        norm_layer = Identity if norm_layer is None else norm_layer
        norm_layer_cat = Identity if norm_layer_cat is None else norm_layer_cat
        layers = get_conv_block(enc_channels, out_channels, nonlinear_layer, norm_layer, 'same', False, kernel_size)
        layers += [norm_layer_cat(out_channels)]
        self.enc_block = nn.Sequential(*layers)

    def forward(self, input, vgg_input):
        output_enc = self.enc_block(vgg_input)
        output_dis = input
        output = torch.cat([output_enc, output_dis], 1)
        return output


class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x, size=None):
        if len(x.shape) == 2:
            if size is None:
                return x.view(x.shape[0], -1, 1, 1)
            else:
                b, c = x.shape
                _, _, h, w = size
                return x.view(b, c, 1, 1).expand(b, c, h, w)
        elif len(x.shape) == 4:
            return x.view(x.shape[0], -1)

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (View,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_egorzakharov_PerceptualGAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

