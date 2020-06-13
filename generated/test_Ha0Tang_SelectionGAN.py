import sys
_module = sys.modules[__name__]
del sys
data = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
keypoint = _module
rename = _module
L1_plus_perceptualLoss = _module
losses = _module
ssim = _module
PATN = _module
models = _module
base_model = _module
model_variants = _module
networks = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
ssd_score = _module
compute_ssd_score_fashion = _module
compute_ssd_score_market = _module
test = _module
calPCKH_fashion = _module
calPCKH_market = _module
cmd = _module
compute_coordinates = _module
crop_fashion = _module
crop_market = _module
generate_fashion_datasets = _module
generate_pose_map_fashion = _module
generate_pose_map_market = _module
getMetrics_fashion = _module
getMetrics_market = _module
inception_score = _module
pose_utils = _module
resize_fashion = _module
rm_insnorm_running_vars = _module
train = _module
util = _module
get_data = _module
html = _module
image_pool = _module
png = _module
visualizer = _module
aligned_dataset = _module
base_model = _module
networks = _module
selectiongan_model = _module
KL_model_data = _module
compute_topK_KL = _module
base_model = _module
networks = _module
selectiongan_model = _module
ade20k_dataset = _module
cityscapes_dataset = _module
coco_dataset = _module
custom_dataset = _module
facades_dataset = _module
pix2pix_dataset = _module
coco_generate_instance_map = _module
models = _module
architecture = _module
base_network = _module
discriminator = _module
encoder = _module
generator = _module
loss = _module
normalization = _module
pix2pix_model = _module
trainers = _module
pix2pix_trainer = _module
coco = _module
iter_counter = _module

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


import torch


from torch import nn


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


from math import exp


from collections import OrderedDict


import itertools


import torch.nn as nn


import functools


from torch.nn import init


from torch.optim import lr_scheduler


from torch.autograd import Variable as V


from torch.nn import functional as F


from torch.nn import Module


from torch.nn import Parameter


from torch.nn import Softmax


import torch.nn.utils.spectral_norm as spectral_norm


import re


class L1_plus_perceptualLoss(nn.Module):

    def __init__(self, lambda_L1, lambda_perceptual, perceptual_layers,
        gpu_ids, percep_is_l1):
        super(L1_plus_perceptualLoss, self).__init__()
        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual
        self.gpu_ids = gpu_ids
        self.percep_is_l1 = percep_is_l1
        vgg = models.vgg19(pretrained=True).features
        self.vgg_submodel = nn.Sequential()
        for i, layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i), layer)
            if i == perceptual_layers:
                break
        self.vgg_submodel = torch.nn.DataParallel(self.vgg_submodel,
            device_ids=gpu_ids)
        None

    def forward(self, inputs, targets):
        if self.lambda_L1 == 0 and self.lambda_perceptual == 0:
            return torch.zeros(1), torch.zeros(1), torch.zeros(1)
        loss_l1 = F.l1_loss(inputs, targets) * self.lambda_L1
        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = mean.resize(1, 3, 1, 1)
        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = std.resize(1, 3, 1, 1)
        fake_p2_norm = (inputs + 1) / 2
        fake_p2_norm = (fake_p2_norm - mean) / std
        input_p2_norm = (targets + 1) / 2
        input_p2_norm = (input_p2_norm - mean) / std
        fake_p2_norm = self.vgg_submodel(fake_p2_norm)
        input_p2_norm = self.vgg_submodel(input_p2_norm)
        input_p2_norm_no_grad = input_p2_norm.detach()
        if self.percep_is_l1 == 1:
            loss_perceptual = F.l1_loss(fake_p2_norm, input_p2_norm_no_grad
                ) * self.lambda_perceptual
        else:
            loss_perceptual = F.mse_loss(fake_p2_norm, input_p2_norm_no_grad
                ) * self.lambda_perceptual
        loss = loss_l1 + loss_perceptual
        return loss, loss_l1, loss_perceptual


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 
        sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0
        )
    window = Variable(_2D_window.expand(channel, 1, window_size,
        window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2,
        groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2,
        groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq +
        C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type(
            ) == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.
            size_average)


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        None


class PATBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,
        cated_stream2=False):
        super(PATBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=
            cated_stream2)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias, cated_stream2=False, cal_att=False):
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
        if cated_stream2:
            conv_block += [nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                padding=p, bias=use_bias), norm_layer(dim * 2), nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
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
        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim * 2, dim, kernel_size=3,
                    padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                    bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = torch.sigmoid(x2_out)
        x1_out = x1_out * att
        out = x1 + x1_out
        x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out, x1_out


class PATNModel(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect', n_downsampling=2):
        assert n_blocks >= 0 and type(input_nc) == list
        super(PATNModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model_stream1_down = [nn.ReflectionPad2d(3), nn.Conv2d(self.
            input_nc_s1, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf), nn.ReLU(True)]
        model_stream2_down = [nn.ReflectionPad2d(3), nn.Conv2d(self.
            input_nc_s2, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2), nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        cated_stream2 = [(True) for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(PATBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias, cated_stream2=cated_stream2[i]))
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf *
                mult / 2), kernel_size=3, stride=2, padding=1,
                output_padding=1, bias=use_bias), norm_layer(int(ngf * mult /
                2)), nn.ReLU(True)]
        model_stream1_up2 = []
        model_stream1_up2 += [nn.ReflectionPad2d(3)]
        model_stream1_up2 += [nn.Conv2d(ngf, output_nc, kernel_size=7,
            padding=0)]
        model_stream1_up2 += [nn.Tanh()]
        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)
        self.stream1_up2 = nn.Sequential(*model_stream1_up2)

    def forward(self, input):
        x1, x2 = input
        x1 = self.stream1_down(x1)
        x2 = self.stream2_down(x2)
        for model in self.att:
            x1, x2, _ = model(x1, x2)
        feature = self.stream1_up(x1)
        x1 = self.stream1_up2(feature)
        return x1, feature


class SelectionGANModel(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect', n_downsampling=2):
        assert n_blocks >= 0 and type(input_nc) == list
        super(SelectionGANModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(4, 4))
        self.pool3 = nn.AvgPool2d(kernel_size=(9, 9))
        self.conv106 = nn.Conv2d(106 * 4, 106, kernel_size=3, stride=1,
            padding=1, bias=nn.InstanceNorm2d)
        self.model_attention = nn.Conv2d(106, 10, kernel_size=1, stride=1,
            padding=0)
        self.model_image = nn.Conv2d(106, 30, kernel_size=3, stride=1,
            padding=1)
        self.tanh = torch.nn.Tanh()
        self.convolution_for_attention = torch.nn.Conv2d(10, 1, 1, stride=1,
            padding=0)

    def forward(self, input):
        pool_feature1 = self.pool1(input)
        pool_feature2 = self.pool2(input)
        pool_feature3 = self.pool3(input)
        b, c, h, w = input.size()
        pool_feature1_up = F.upsample(input=pool_feature1, size=(h, w),
            mode='bilinear', align_corners=True)
        pool_feature2_up = F.upsample(input=pool_feature2, size=(h, w),
            mode='bilinear', align_corners=True)
        pool_feature3_up = F.upsample(input=pool_feature3, size=(h, w),
            mode='bilinear', align_corners=True)
        f1 = input * pool_feature1_up
        f2 = input * pool_feature2_up
        f3 = input * pool_feature3_up
        feature_image_combine = torch.cat((f1, f2, f3, input), 1)
        feature_image_combine = self.conv106(feature_image_combine)
        attention = self.model_attention(feature_image_combine)
        image = self.model_image(feature_image_combine)
        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)
        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]
        attention1 = attention1_.repeat(1, 3, 1, 1)
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)
        image = self.tanh(image)
        image1 = image[:, 0:3, :, :]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        image10 = image[:, 27:30, :, :]
        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        output10 = image10 * attention10
        output10 = image10 * attention10
        final = (output1 + output2 + output3 + output4 + output5 + output6 +
            output7 + output8 + output9 + output10)
        sigmoid_ = torch.nn.Sigmoid()
        uncertainty = self.convolution_for_attention(attention)
        uncertainty = sigmoid_(uncertainty)
        uncertainty_map = uncertainty.repeat(1, 3, 1, 1)
        return final, uncertainty_map


class PATNetwork(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect', n_downsampling=2):
        super(PATNetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc
            ) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = PATNModel(input_nc, output_nc, ngf, norm_layer,
            use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=
            n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class SelectionGANNetwork(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
        padding_type='reflect', n_downsampling=2):
        super(SelectionGANNetwork, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = SelectionGANModel(input_nc, output_nc, ngf, norm_layer,
            use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=
            n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None or self.
                real_label_var.numel() != input.numel())
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.
                    real_label)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None or self.
                fake_label_var.numel() != input.numel())
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.
                    fake_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias):
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
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
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


class ResnetDiscriminator(nn.Module):

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d,
        use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',
        use_sigmoid=False, n_downsampling=2):
        assert n_blocks >= 0
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf,
            kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.
            ReLU(True)]
        if n_downsampling <= 2:
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size
                    =3, stride=2, padding=1, bias=use_bias), norm_layer(ngf *
                    mult * 2), nn.ReLU(True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
            mult = 2 ** 1
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
            mult = 2 ** 2
            model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult),
                nn.ReLU(True)]
        if n_downsampling <= 2:
            mult = 2 ** n_downsampling
        else:
            mult = 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf,
            kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.
            ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias):
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
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
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


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=
        nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer)
        self.model = unet_block

    def forward(self, input):
        feature, image = self.model(input)
        return feature, image


class UnetGenerator_a(nn.Module):

    def __init__(self, input_nc, output_nc):
        super(UnetGenerator_a, self).__init__()
        self.deconvolution_1 = torch.nn.ConvTranspose2d(136, 104,
            kernel_size=2, stride=2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(4, 4))
        self.pool3 = nn.AvgPool2d(kernel_size=(9, 9))
        self.model_attention = nn.Conv2d(input_nc, output_nc, kernel_size=1,
            stride=1, padding=0, bias=nn.InstanceNorm2d)
        self.model_image = nn.Conv2d(input_nc, output_nc * 3, kernel_size=3,
            stride=1, padding=1, bias=nn.InstanceNorm2d)
        self.conv330 = nn.Conv2d(330, 110, kernel_size=3, stride=1, padding
            =1, bias=nn.InstanceNorm2d)
        self.conv440 = nn.Conv2d(440, 110, kernel_size=3, stride=1, padding
            =1, bias=nn.InstanceNorm2d)
        self.convolution_for_attention = torch.nn.Conv2d(10, 1, 1, stride=1,
            padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, feature_combine, image_combine):
        output_feature = self.relu(self.dropout(self.deconvolution_1(
            feature_combine)))
        feature_image_combine = torch.cat((output_feature, image_combine), 1)
        pool_feature1 = self.pool1(feature_image_combine)
        pool_feature2 = self.pool2(feature_image_combine)
        pool_feature3 = self.pool3(feature_image_combine)
        pool_feature1_up = F.upsample(input=pool_feature1, size=(256, 256),
            mode='bilinear', align_corners=True)
        pool_feature2_up = F.upsample(input=pool_feature2, size=(256, 256),
            mode='bilinear', align_corners=True)
        pool_feature3_up = F.upsample(input=pool_feature3, size=(256, 256),
            mode='bilinear', align_corners=True)
        f1 = feature_image_combine * pool_feature1_up
        f2 = feature_image_combine * pool_feature2_up
        f3 = feature_image_combine * pool_feature3_up
        feature_image_combine = torch.cat((f1, f2, f3,
            feature_image_combine), 1)
        feature_image_combine = self.conv440(feature_image_combine)
        attention = self.model_attention(feature_image_combine)
        image = self.model_image(feature_image_combine)
        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)
        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]
        attention1 = attention1_.repeat(1, 3, 1, 1)
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)
        image = self.tanh(image)
        image1 = image[:, 0:3, :, :]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        image10 = image[:, 27:30, :, :]
        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        output10 = image10 * attention10
        output11 = (output1 + output2 + output3 + output4 + output5 +
            output6 + output7 + output8 + output9 + output10)
        sigmoid_ = torch.nn.Sigmoid()
        uncertainty = self.convolution_for_attention(attention)
        uncertainty = sigmoid_(uncertainty)
        uncertainty_map = uncertainty.repeat(1, 3, 1, 1)
        return (image1, image2, image3, image4, image5, image6, image7,
            image8, image9, image10, attention1_, attention2_, attention3_,
            attention4_, attention5_, attention6_, attention7_, attention8_,
            attention9_, attention10_, output1, output2, output3, output4,
            output5, output6, output7, output8, output9, output10,
            uncertainty_map, output11)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            down = [downconv]
            self.down = nn.Sequential(*down)
            up = [upconv, nn.Tanh()]
            self.up = nn.Sequential(*up)
            self.submodule = submodule
            self.uprelu = uprelu
            self.use_dropout = use_dropout
            model = [submodule]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1)
            x3 = self.uprelu(x2)
            x4 = self.up(x3)
            if self.use_dropout:
                x5 = nn.Dropout(x4, 0.5)
                return x3, x5
            return x3, x4
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock_a(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        super(UnetSkipConnectionBlock_a, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
            padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1,
            padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2,
            kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(
            ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1,
            kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.
        BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf,
            kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.
            ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult *
                2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=
                use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=
                use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
        use_bias):
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
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
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


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=
        nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=
            None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc
                =None, submodule=unet_block, norm_layer=norm_layer,
                use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=
            None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=
            input_nc, submodule=unet_block, outermost=True, norm_layer=
            norm_layer)
        self.model = unet_block

    def forward(self, input):
        feature, image = self.model(input)
        return feature, image


class UnetGenerator_a(nn.Module):

    def __init__(self, input_nc, output_nc):
        super(UnetGenerator_a, self).__init__()
        self.deconvolution_1 = torch.nn.ConvTranspose2d(136, 104,
            kernel_size=2, stride=2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(4, 4))
        self.pool3 = nn.AvgPool2d(kernel_size=(9, 9))
        self.model_attention = nn.Conv2d(input_nc, output_nc, kernel_size=1,
            stride=1, padding=0, bias=nn.InstanceNorm2d)
        self.model_image = nn.Conv2d(input_nc, output_nc * 3, kernel_size=3,
            stride=1, padding=1, bias=nn.InstanceNorm2d)
        self.conv330 = nn.Conv2d(330, 110, kernel_size=3, stride=1, padding
            =1, bias=nn.InstanceNorm2d)
        self.conv440 = nn.Conv2d(440, 110, kernel_size=3, stride=1, padding
            =1, bias=nn.InstanceNorm2d)
        self.convolution_for_attention = torch.nn.Conv2d(10, 1, 1, stride=1,
            padding=0)
        self.sc = CAM_Module(440)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, feature_combine, image_combine):
        output_feature = self.relu(self.dropout(self.deconvolution_1(
            feature_combine)))
        feature_image_combine = torch.cat((output_feature, image_combine), 1)
        pool_feature1 = self.pool1(feature_image_combine)
        pool_feature2 = self.pool2(feature_image_combine)
        pool_feature3 = self.pool3(feature_image_combine)
        pool_feature1_up = F.upsample(input=pool_feature1, size=(256, 256),
            mode='bilinear', align_corners=True)
        pool_feature2_up = F.upsample(input=pool_feature2, size=(256, 256),
            mode='bilinear', align_corners=True)
        pool_feature3_up = F.upsample(input=pool_feature3, size=(256, 256),
            mode='bilinear', align_corners=True)
        f1 = feature_image_combine * pool_feature1_up
        f2 = feature_image_combine * pool_feature2_up
        f3 = feature_image_combine * pool_feature3_up
        feature_image_combine = torch.cat((f1, f2, f3,
            feature_image_combine), 1)
        feature_image_combine = self.sc(feature_image_combine)
        feature_image_combine = self.conv440(feature_image_combine)
        attention = self.model_attention(feature_image_combine)
        image = self.model_image(feature_image_combine)
        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)
        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]
        attention1 = attention1_.repeat(1, 3, 1, 1)
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)
        image = self.tanh(image)
        image1 = image[:, 0:3, :, :]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        image10 = image[:, 27:30, :, :]
        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        output10 = image10 * attention10
        output11 = (output1 + output2 + output3 + output4 + output5 +
            output6 + output7 + output8 + output9 + output10)
        sigmoid_ = torch.nn.Sigmoid()
        uncertainty = self.convolution_for_attention(attention)
        uncertainty = sigmoid_(uncertainty)
        uncertainty_map = uncertainty.repeat(1, 3, 1, 1)
        return (image1, image2, image3, image4, image5, image6, image7,
            image8, image9, image10, attention1_, attention2_, attention3_,
            attention4_, attention5_, attention6_, attention7_, attention8_,
            attention9_, attention10_, output1, output2, output3, output4,
            output5, output6, output7, output8, output9, output10,
            uncertainty_map, output11)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            down = [downconv]
            self.down = nn.Sequential(*down)
            up = [upconv, nn.Tanh()]
            self.up = nn.Sequential(*up)
            self.submodule = submodule
            self.uprelu = uprelu
            self.use_dropout = use_dropout
            model = [submodule]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1)
            x3 = self.uprelu(x2)
            x4 = self.up(x3)
            if self.use_dropout:
                x5 = nn.Dropout(x4, 0.5)
                return x3, x5
            return x3, x4
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock_a(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
        use_dropout=False):
        super(UnetSkipConnectionBlock_a, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
            padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1,
            padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
        use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1,
            padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2,
            kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(
            ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1,
            kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        query = x.view(m_batchsize, C, -1)
        key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = self.softmax(energy_new)
        value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class SPADEResnetBlock(nn.Module):

    def __init__(self, fin, fout, opt):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class ResnetBlock(nn.Module):

    def __init__(self, dim, norm_layer, activation=nn.ReLU(False),
        kernel_size=3):
        super().__init__()
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(pw), norm_layer(
            nn.Conv2d(dim, dim, kernel_size=kernel_size)), activation, nn.
            ReflectionPad2d(pw), norm_layer(nn.Conv2d(dim, dim, kernel_size
            =kernel_size)))

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class VGG19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True
            ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        None

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or 
                classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=
        0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        elif target_is_real:
            return -input.mean()
        else:
            return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real,
                    for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].
                detach())
        return loss


class KLDLoss(nn.Module):

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=
                False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                '%s is not a recognized param-free norm type in SPADE' %
                param_free_norm_type)
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden,
            kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw
            )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class Pix2PixModel(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu(
            ) else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu(
            ) else torch.ByteTensor
        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.
                FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(input_semantics,
                real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics,
                real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return netG, netD, netE

    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label']
            data['instance'] = data['instance']
            data['image'] = data['image']
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = (self.opt.label_nc + 1 if self.opt.contain_dontcare_label else
            self.opt.label_nc)
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map
                ), dim=1)
        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        fake_image, KLD_loss = self.generate_fake(input_semantics,
            real_image, compute_kld_loss=self.opt.use_vae)
        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
        pred_fake, pred_real = self.discriminate(input_semantics,
            fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
            for_discriminator=False)
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j],
                        pred_real[i][j].detach())
                    GAN_Feat_loss += (unweighted_loss * self.opt.
                        lambda_feat / num_D)
            G_losses['GAN_Feat'] = GAN_Feat_loss
        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image
                ) * self.opt.lambda_vgg
        if not self.opt.no_l1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) * 10
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(input_semantics,
            fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
            for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
            for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False
        ):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        fake_image = self.netG(input_semantics, z=z)
        assert not compute_kld_loss or self.opt.use_vae, 'You cannot compute KLD loss if opt.use_vae == False'
        return fake_image, KLD_loss

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :,
            :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :,
            :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :,
            :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :,
            :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Ha0Tang_SelectionGAN(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SSIM(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BaseModel(*[], **{}), [], {})

    @_fails_compile()
    def test_002(self):
        self._check(UnetGenerator(*[], **{'input_nc': 4, 'output_nc': 4, 'num_downs': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_003(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_004(self):
        self._check(PixelDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(CAM_Module(*[], **{'in_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(KLDLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

