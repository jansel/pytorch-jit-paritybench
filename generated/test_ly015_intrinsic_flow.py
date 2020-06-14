import sys
_module = sys.modules[__name__]
del sys
data = _module
base_dataset = _module
data_loader = _module
flow_dataset = _module
pose_transfer_dataset = _module
models = _module
base_model = _module
flow_regression_model = _module
modules = _module
networks = _module
pose_transfer_model = _module
options = _module
base_options = _module
flow_regression_options = _module
pose_transfer_options = _module
create_flow = _module
create_seg = _module
renderer = _module
run_hmr = _module
fashion_attribute_score = _module
fashion_inception_score = _module
inception_score = _module
masked_inception_score = _module
test_flow_regression_model = _module
test_pose_transfer_model = _module
train_flow_regression_model = _module
train_pose_transfer_model = _module
util = _module
flow_util = _module
image_pool = _module
io = _module
loss_buffer = _module
pose_util = _module
visualizer = _module

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


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


from torch.nn import init


from torch.autograd import Variable


from torch.optim import lr_scheduler


import functools


import numpy as np


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0,
        target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = F.mse_loss
        else:
            self.loss = F.binary_cross_entropy

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids, content_weights=[1.0 / 32, 1.0 / 16, 1.0 / 
        8, 1.0 / 4, 1.0], style_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        shifted_style=False):
        super(VGGLoss, self).__init__()
        self.gpu_ids = gpu_ids
        self.shifted_style = shifted_style
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.shift_delta = [[0, 2, 4, 8, 16], [0, 2, 4, 8], [0, 2, 4], [0, 
            2], [0]]
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
        for param in self.parameters():
            param.requires_grad = False
        if len(gpu_ids) > 0:
            self

    def compute_feature(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def forward(self, X, Y, mask=None, loss_type='content', device_mode=None):
        """
        loss_type: 'content', 'style'
        device_mode: multi, single, sub
        """
        bsz = X.size(0)
        if device_mode is None:
            device_mode = 'multi' if len(self.gpu_ids) > 1 else 'single'
        if device_mode == 'multi':
            if mask is None:
                return nn.parallel.data_parallel(self, (X, Y),
                    module_kwargs={'loss_type': loss_type, 'device_mode':
                    'sub', 'mask': None}).mean(dim=0)
            else:
                return nn.parallel.data_parallel(self, (X, Y, mask),
                    module_kwargs={'loss_type': loss_type, 'device_mode':
                    'sub'}).mean(dim=0)
        else:
            features_x = self.compute_feature(self.normalize(X))
            features_y = self.compute_feature(self.normalize(Y))
            if mask is not None:
                features_x = [(feat * F.adaptive_max_pool2d(mask, (feat.
                    size(2), feat.size(3)))) for feat in features_x]
                features_y = [(feat * F.adaptive_max_pool2d(mask, (feat.
                    size(2), feat.size(3)))) for feat in features_y]
            if loss_type == 'content':
                loss = 0
                for i, (feat_x, feat_y) in enumerate(zip(features_x,
                    features_y)):
                    loss += self.content_weights[i] * F.l1_loss(feat_x,
                        feat_y, reduce=False).view(bsz, -1).mean(dim=1)
            if loss_type == 'style':
                loss = 0
                if self.shifted_style:
                    for i, (feat_x, feat_y) in enumerate(zip(features_x,
                        features_y)):
                        if self.style_weights[i] > 0:
                            for delta in self.shift_delta[i]:
                                if delta == 0:
                                    loss += self.style_weights[i] * F.mse_loss(
                                        self.gram_matrix(feat_x), self.
                                        gram_matrix(feat_y), reduce=False
                                        ).view(bsz, -1).sum(dim=1)
                                else:
                                    loss += 0.5 * self.style_weights[i] * (F
                                        .mse_loss(self.shifted_gram_matrix(
                                        feat_x, delta, 0), self.
                                        shifted_gram_matrix(feat_y, delta, 
                                        0), reduce=False) + F.mse_loss(self
                                        .shifted_gram_matrix(feat_x, 0,
                                        delta), self.shifted_gram_matrix(
                                        feat_y, 0, delta), reduce=False)).view(
                                        bsz, -1).sum(dim=1)
                else:
                    for i, (feat_x, feat_y) in enumerate(zip(features_x,
                        features_y)):
                        if self.style_weights[i] > 0:
                            loss += self.style_weights[i] * F.mse_loss(self
                                .gram_matrix(feat_x), self.gram_matrix(
                                feat_y), reduce=False).view(bsz, -1).sum(dim=1)
            if device_mode == 'single':
                loss = loss.mean(dim=0)
            return loss

    def normalize(self, x):
        mean_1 = x.new([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std_1 = x.new([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        mean_2 = x.new([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_2 = x.new([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x * std_1 + mean_1 - mean_2) / std_2

    def gram_matrix(self, feat):
        bsz, c, h, w = feat.size()
        feat = feat.view(bsz, c, h * w)
        feat_T = feat.transpose(1, 2)
        g = torch.matmul(feat, feat_T) / (c * h * w)
        return g

    def shifted_gram_matrix(self, feat, shift_x, shift_y):
        bsz, c, h, w = feat.size()
        assert shift_x < w and shift_y < h
        feat1 = feat[:, :, shift_y:, shift_x:].contiguous().view(bsz, c, -1)
        feat2 = feat[:, :, :h - shift_y, :w - shift_x].contiguous().view(bsz,
            c, -1)
        g = torch.matmul(feat1, feat2.transpose(1, 2)) / (c * h * w)
        return g


def EPE(input_flow, target_flow, vis_mask):
    """
    compute endpoint-error
    input_flow: (N,C=2,H,W)
    target_flow: (N,C=2,H,W)
    vis_mask: (N,1,H,W)
    """
    bsz = input_flow.size(0)
    epe = (target_flow - input_flow).norm(dim=1, p=2, keepdim=True) * vis_mask
    count = vis_mask.view(bsz, -1).sum(dim=1, keepdim=True)
    return (epe.view(bsz, -1) / (count * bsz + 1e-08)).sum()


def L1(input_flow, target_flow, vis_mask):
    """
    compute l1-loss
    input_flow: (N,C=2,H,W)
    target_flow: (N,C=2,H,W)
    vis_mask: (N,1,H,W)
    """
    bsz = input_flow.size(0)
    err = (target_flow - input_flow).abs() * vis_mask
    count = vis_mask.view(bsz, -1).sum(dim=1, keepdim=True)
    return (err.view(bsz, -1) / (count * bsz * 2 + 1e-08)).sum()


def L2(input_flow, target_flow, vis_mask):
    """
    compute l1-loss
    input_flow: (N,C=2,H,W)
    target_flow: (N,C=2,H,W)
    vis_mask: (N,1,H,W)
    """
    bsz = input_flow.size(0)
    err = (target_flow - input_flow).norm(dim=1, p=2, keepdim=True) * vis_mask
    count = vis_mask.view(bsz, -1).sum(dim=1, keepdim=True)
    return (err.view(bsz, -1) / (count * bsz + 1e-08)).sum()


class MultiScaleFlowLoss(nn.Module):
    """
    Derived from NVIDIA/flownet2-pytorch repo.
    """

    def __init__(self, start_scale=2, num_scale=5, l_weight=0.32, loss_type
        ='l1'):
        super(MultiScaleFlowLoss, self).__init__()
        self.start_scale = start_scale
        self.num_scale = num_scale
        self.loss_weights = [(l_weight / 2 ** scale) for scale in range(
            self.num_scale)]
        self.loss_type = loss_type
        self.div_flow = 0.05
        self.avg_pools = [nn.AvgPool2d(self.start_scale * 2 ** scale, self.
            start_scale * 2 ** scale) for scale in range(num_scale)]
        self.max_pools = [nn.MaxPool2d(self.start_scale * 2 ** scale, self.
            start_scale * 2 ** scale) for scale in range(num_scale)]
        if loss_type == 'l1':
            self.loss_func = L1
        elif loss_type == 'l2':
            self.loss_func = L2

    def forward(self, input_flows, target_flow, vis_mask):
        loss = 0
        epe = 0
        target_flow = self.div_flow * target_flow
        for i, input_ in enumerate(input_flows):
            target_ = self.avg_pools[i](target_flow)
            mask_ = self.max_pools[i](vis_mask)
            assert input_.is_same_size(target_
                ), 'scale %d size mismatch: input(%s) vs. target(%s)' % (i,
                input_.size(), target_.size())
            loss += self.loss_weights[i] * self.loss_func(input_, target_,
                mask_)
            epe += self.loss_weights[i] * EPE(input_, target_, mask_)
        return loss, epe


def warp_acc_flow(x, flow, mode='bilinear', mask=None, mask_value=-1):
    """
    warp an image/tensor according to given flow.
    Input:
        x: (bsz, c, h, w)
        flow: (bsz, c, h, w)
        mask: (bsz, 1, h, w). 1 for valid region and 0 for invalid region. invalid region will be fill with "mask_value" in the output images.
    Output:
        y: (bsz, c, h, w)
    """
    bsz, c, h, w = x.size()
    xx = x.new_tensor(range(w)).view(1, -1).repeat(h, 1)
    yy = x.new_tensor(range(h)).view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(bsz, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(bsz, 1, 1, 1)
    grid = torch.cat((xx, yy), dim=1).float()
    grid = grid + flow
    grid[:, (0), :, :] = 2.0 * grid[:, (0), :, :] / max(w - 1, 1) - 1.0
    grid[:, (1), :, :] = 2.0 * grid[:, (1), :, :] / max(h - 1, 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, grid, mode=mode, padding_mode='zeros')
    if mask is not None:
        output = torch.where(mask > 0.5, output, output.new_ones(1).mul_(
            mask_value))
    return output


class SS_FlowLoss(nn.Module):
    """
    segmentation sensitive flow loss
    this loss function only penalize pixels where the flow points to a wrong segmentation area
    """

    def __init__(self, loss_type='l1'):
        super(SS_FlowLoss, self).__init__()
        self.div_flow = 0.05
        self.loss_type = loss_type

    def forward(self, input_flow, target_flow, seg_1, seg_2, vis_2):
        """
        input_flow: (bsz, 2, h, w)
        target_flow: (bsz, 2, h, w) note that there is scale factor between input_flow and target_flow, which is self.div_flow
        seg_1, seg_2: (bsz, ns, h, w) channel-0 should be background
        vis_2: (bsz, 1, h, w) visibility map of image_2
        """
        with torch.no_grad():
            seg_1 = seg_1[:, 1:, (...)]
            seg_2 = seg_2[:, 1:, (...)]
            seg_1w = warp_acc_flow(seg_1, input_flow)
            seg_1w = (seg_1w > 0).float()
            mask = (seg_2 * (1 - seg_1w)).sum(dim=1, keepdim=True)
            mask = mask * (vis_2 == 0).float()
        err = (input_flow - target_flow).mul(self.div_flow) * mask
        if self.loss_type == 'l1':
            loss = err.abs().mean()
        elif self.loss_type == 'l2':
            loss = err.norm(p=2, dim=1).mean()
        return loss


class PSNR(nn.Module):

    def forward(self, images_1, images_2):
        numpy_imgs_1 = images_1.cpu().detach().numpy().transpose(0, 2, 3, 1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0, 255).astype(np
            .uint8)
        numpy_imgs_2 = images_2.cpu().detach().numpy().transpose(0, 2, 3, 1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0, 255).astype(np
            .uint8)
        psnr_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            psnr_score.append(compare_psnr(img_2, img_1))
        return Variable(images_1.data.new(1).fill_(np.mean(psnr_score)))


class SSIM(nn.Module):

    def forward(self, images_1, images_2, mask=None):
        numpy_imgs_1 = images_1.cpu().detach().numpy().transpose(0, 2, 3, 1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0, 255).astype(np
            .uint8)
        numpy_imgs_2 = images_2.cpu().detach().numpy().transpose(0, 2, 3, 1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0, 255).astype(np
            .uint8)
        if mask is not None:
            mask = mask.cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np
                .uint8)
            numpy_imgs_1 = numpy_imgs_1 * mask
            numpy_imgs_2 = numpy_imgs_2 * mask
        ssim_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            ssim_score.append(compare_ssim(img_1, img_2, multichannel=True))
        return Variable(images_1.data.new(1).fill_(np.mean(ssim_score)))


class Identity(nn.Module):

    def __init__(self, dim=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
    model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
        stride, padding, dilation, bias=bias), norm_layer(out_channels))
    return model


def channel_mapping(in_channels, out_channels, norm_layer=nn.BatchNorm2d,
    bias=False):
    return conv(in_channels, out_channels, kernel_size=1, norm_layer=
        norm_layer, bias=bias)


class ResidualBlock(nn.Module):
    """
    Derived from Variational UNet.
    """

    def __init__(self, dim, dim_a, norm_layer=nn.BatchNorm2d, use_bias=
        False, activation=nn.ReLU(False), use_dropout=False, no_end_norm=False
        ):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.activation = activation
        if dim_a <= 0 or dim_a is None:
            if no_end_norm:
                self.conv = conv(in_channels=dim, out_channels=dim,
                    kernel_size=3, padding=1, norm_layer=Identity, bias=True)
            else:
                self.conv = conv(in_channels=dim, out_channels=dim,
                    kernel_size=3, padding=1, norm_layer=norm_layer, bias=
                    use_bias)
        else:
            self.conv_a = channel_mapping(in_channels=dim_a, out_channels=
                dim, norm_layer=norm_layer, bias=use_bias)
            if no_end_norm:
                self.conv = conv(in_channels=dim * 2, out_channels=dim,
                    kernel_size=3, padding=1, norm_layer=Identity, bias=True)
            else:
                self.conv = conv(in_channels=dim * 2, out_channels=dim,
                    kernel_size=3, padding=1, norm_layer=norm_layer, bias=
                    use_bias)

    def forward(self, x, a=None):
        if a is None:
            residual = x
        else:
            a = self.conv_a(self.activation(a))
            residual = torch.cat((x, a), dim=1)
        residual = self.conv(self.activation(residual))
        out = x + residual
        if self.use_dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        return out


class GateBlock(nn.Module):

    def __init__(self, dim, dim_a, activation=nn.ReLU(False)):
        super(GateBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=dim_a, out_channels=dim,
            kernel_size=1)

    def forward(self, x, a):
        """
        x: (bsz, dim, h, w)
        a: (bsz, dim_a, h, w)
        """
        a = self.activation(a)
        g = F.sigmoid(self.conv(a))
        return x * g


class UnetGenerator(nn.Module):
    """
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    """

    def __init__(self, input_nc, output_nc, nf=64, max_nf=256, num_scales=7,
        n_residual_blocks=2, norm='batch', activation=nn.ReLU(False),
        use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        self.pre_conv = channel_mapping(input_nc, nf, norm_layer, use_bias)
        for l in range(num_scales):
            c_in = min(nf * (l + 1), max_nf)
            c_out = min(nf * (l + 2), max_nf)
            for i in range(n_residual_blocks):
                self.__setattr__('enc_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, None, norm_layer, use_bias, activation,
                    use_dropout=False))
            downsample = nn.Sequential(activation, nn.Conv2d(c_in, c_out,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out))
            self.__setattr__('enc_%d_downsample' % l, downsample)
            upsample = nn.Sequential(activation, nn.Conv2d(c_out, c_in * 4,
                kernel_size=3, padding=1, bias=use_bias), nn.PixelShuffle(2
                ), norm_layer(c_in))
            self.__setattr__('dec_%d_upsample' % l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, c_in, norm_layer, use_bias, activation, use_dropout))
        self.dec_output = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(nf,
            output_nc, kernel_size=7, padding=0, bias=True))

    def forward(self, x, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, x, module_kwargs={
                'single_device': True})
        else:
            hiddens = []
            x = self.pre_conv(x)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x = self.__getattr__('enc_%d_res_%d' % (l, i))(x)
                    hiddens.append(x)
                x = self.__getattr__('enc_%d_downsample' % l)(x)
            for l in range(self.num_scales - 1, -1, -1):
                x = self.__getattr__('dec_%d_upsample' % l)(x)
                for i in range(self.n_residual_blocks - 1, -1, -1):
                    h = hiddens.pop()
                    x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, h)
            out = self.dec_output(x)
            return out


class UnetGenerator_MultiOutput(nn.Module):
    """
    A variation of UnetGenerator that support multiple output branches
    """

    def __init__(self, input_nc, output_nc=[3], nf=64, max_nf=256,
        num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU
        (False), use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_MultiOutput, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc if isinstance(output_nc, list) else [
            output_nc]
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        self.pre_conv = channel_mapping(input_nc, nf, norm_layer, use_bias)
        for l in range(num_scales):
            c_in = min(nf * (l + 1), max_nf)
            c_out = min(nf * (l + 2), max_nf)
            for i in range(n_residual_blocks):
                self.__setattr__('enc_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, None, norm_layer, use_bias, activation,
                    use_dropout=False))
            downsample = nn.Sequential(activation, nn.Conv2d(c_in, c_out,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out))
            self.__setattr__('enc_%d_downsample' % l, downsample)
            upsample = nn.Sequential(activation, nn.Conv2d(c_out, c_in * 4,
                kernel_size=3, padding=1, bias=use_bias), nn.PixelShuffle(2
                ), norm_layer(c_in))
            self.__setattr__('dec_%d_upsample' % l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, c_in, norm_layer, use_bias, activation, use_dropout))
        for i, c_out in enumerate(output_nc):
            dec_output_i = nn.Sequential(channel_mapping(nf, nf, norm_layer,
                use_bias), activation, nn.ReflectionPad2d(3), nn.Conv2d(nf,
                c_out, kernel_size=7, padding=0, bias=True))
            self.__setattr__('dec_output_%d' % i, dec_output_i)

    def forward(self, x, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, x, module_kwargs={
                'single_device': True})
        else:
            hiddens = []
            x = self.pre_conv(x)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x = self.__getattr__('enc_%d_res_%d' % (l, i))(x)
                    hiddens.append(x)
                x = self.__getattr__('enc_%d_downsample' % l)(x)
            for l in range(self.num_scales - 1, -1, -1):
                x = self.__getattr__('dec_%d_upsample' % l)(x)
                for i in range(self.n_residual_blocks - 1, -1, -1):
                    h = hiddens.pop()
                    x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, h)
            out = []
            for i in range(len(self.output_nc)):
                out.append(self.__getattr__('dec_output_%d' % i)(x))
            return out


class DualUnetGenerator(nn.Module):
    """
    A variation of Unet architecture, similar to deformable gan. It contains two encoders: one for target pose and one for appearance. The feature map of appearance encoder will be warped to target pose, guided
    by input flow. There are skip connections from both encoders to the decoder.
    """

    def __init__(self, pose_nc, appearance_nc, output_nc, aux_output_nc=[],
        nf=32, max_nf=128, num_scales=7, num_warp_scales=5,
        n_residual_blocks=2, norm='batch', vis_mode='none', activation=nn.
        ReLU(False), use_dropout=False, no_end_norm=False, gpu_ids=[]):
        """
        vis_mode: ['none', 'hard_gate', 'soft_gate', 'residual']
        no_end_norm: remove normalization layer at the start and the end.
        """
        super(DualUnetGenerator, self).__init__()
        self.pose_nc = pose_nc
        self.appearance_nc = appearance_nc
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.num_warp_scales = num_warp_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout
        self.vis_mode = vis_mode
        self.vis_expand_mult = 2
        self.aux_output_nc = aux_output_nc
        self.no_end_norm = no_end_norm
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        if not no_end_norm:
            self.encp_pre_conv = channel_mapping(pose_nc, nf, norm_layer,
                use_bias)
            self.enca_pre_conv = channel_mapping(appearance_nc, nf,
                norm_layer, use_bias)
        else:
            self.encp_pre_conv = channel_mapping(pose_nc, nf, Identity, True)
            self.enca_pre_conv = channel_mapping(appearance_nc, nf,
                Identity, True)
        for l in range(num_scales):
            c_in = min(nf * (l + 1), max_nf)
            c_out = min(nf * (l + 2), max_nf)
            for i in range(n_residual_blocks):
                self.__setattr__('encp_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, None, norm_layer, use_bias, activation,
                    use_dropout=False))
            p_downsample = nn.Sequential(activation, nn.Conv2d(c_in, c_out,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out))
            self.__setattr__('encp_%d_downsample' % l, p_downsample)
            for i in range(n_residual_blocks):
                self.__setattr__('enca_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, None, norm_layer, use_bias, activation,
                    use_dropout=False))
                if l < num_warp_scales:
                    if vis_mode == 'hard_gate':
                        pass
                    elif vis_mode == 'soft_gate':
                        self.__setattr__('enca_%d_vis_%d' % (l, i),
                            GateBlock(c_in, c_in * self.vis_expand_mult,
                            activation))
                    elif vis_mode == 'residual':
                        self.__setattr__('enca_%d_vis_%d' % (l, i),
                            ResidualBlock(c_in, c_in * self.vis_expand_mult,
                            norm_layer, use_bias, activation, use_dropout=
                            False))
                    elif vis_mode == 'res_no_vis':
                        self.__setattr__('enca_%d_vis_%d' % (l, i),
                            ResidualBlock(c_in, None, norm_layer, use_bias,
                            activation, use_dropout=False))
            a_downsample = nn.Sequential(activation, nn.Conv2d(c_in, c_out,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out))
            self.__setattr__('enca_%d_downsample' % l, p_downsample)
            if l == num_scales - 1:
                self.dec_fuse = channel_mapping(c_out * 2, c_out,
                    norm_layer, use_bias)
            upsample = nn.Sequential(activation, nn.Conv2d(c_out, c_in * 4,
                kernel_size=3, padding=1, bias=use_bias), nn.PixelShuffle(2
                ), norm_layer(c_in))
            self.__setattr__('dec_%d_upsample' % l, upsample)
            for i in range(n_residual_blocks):
                if l == num_scales - 1 and i == n_residual_blocks - 1:
                    self.__setattr__('dec_%d_res_%d' % (l, i),
                        ResidualBlock(c_in, c_in * 2, norm_layer, use_bias,
                        activation, use_dropout, no_end_norm=no_end_norm))
                else:
                    self.__setattr__('dec_%d_res_%d' % (l, i),
                        ResidualBlock(c_in, c_in * 2, norm_layer, use_bias,
                        activation, use_dropout))
        self.dec_output = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(nf,
            output_nc, kernel_size=7, padding=0, bias=True))
        for i, a_nc in enumerate(aux_output_nc):
            dec_aux_output = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d
                (nf, a_nc, kernel_size=7, padding=0, bias=True))
            self.__setattr__('dec_aux_output_%d' % i, dec_aux_output)

    def _vis_expand(self, feat, vis):
        """
        expand feature from n channels to n*vis_expand_mult channels
        """
        feat_exp = [(feat * (vis == i).float()) for i in range(self.
            vis_expand_mult)]
        return torch.cat(feat_exp, dim=1)

    def forward(self, x_p, x_a, flow=None, vis=None, output_feats=False,
        single_device=False):
        """
        x_p: (bsz, pose_nc, h, w), pose input
        x_a: (bsz, appearance_nc, h, w), appearance input
        vis: (bsz, 1, h, w), 0-visible, 1-invisible, 2-background
        flow: (bsz, 2, h, w) or None. if flow==None, feature warping will not be performed
        """
        if len(self.gpu_ids) > 1 and not single_device:
            if flow is not None:
                assert vis is not None
                return nn.parallel.data_parallel(self, (x_p, x_a, flow, vis
                    ), module_kwargs={'single_device': True, 'output_feats':
                    output_feats})
            else:
                return nn.parallel.data_parallel(self, (x_p, x_a),
                    module_kwargs={'flow': None, 'vis': None,
                    'single_device': True, 'output_feats': output_feats})
        else:
            use_fw = flow is not None
            if use_fw:
                vis = vis.round()
            hidden_p = []
            hidden_a = []
            x_p = self.encp_pre_conv(x_p)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x_p = self.__getattr__('encp_%d_res_%d' % (l, i))(x_p)
                    hidden_p.append(x_p)
                x_p = self.__getattr__('encp_%d_downsample' % l)(x_p)
            x_a = self.enca_pre_conv(x_a)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x_a = self.__getattr__('enca_%d_res_%d' % (l, i))(x_a)
                    if use_fw and l < self.num_warp_scales:
                        if i == 0:
                            flow_l = F.avg_pool2d(flow, kernel_size=2 ** l
                                ).div_(2 ** l) if l > 0 else flow
                            vis_l = -F.max_pool2d(-vis, kernel_size=2 ** l
                                ) if l > 0 else vis
                        x_w = warp_acc_flow(x_a, flow_l)
                        if self.vis_mode == 'none':
                            pass
                        elif self.vis_mode == 'hard_gate':
                            x_w = x_w * (vis_l < 2).float()
                        elif self.vis_mode == 'soft_gate':
                            x_we = self._vis_expand(x_w, vis_l)
                            x_w = self.__getattr__('enca_%d_vis_%d' % (l, i))(
                                x_w, x_we)
                        elif self.vis_mode == 'residual':
                            x_we = self._vis_expand(x_w, vis_l)
                            x_w = self.__getattr__('enca_%d_vis_%d' % (l, i))(
                                x_w, x_we)
                        elif self.vis_mode == 'res_no_vis':
                            x_w = self.__getattr__('enca_%d_vis_%d' % (l, i))(
                                x_w)
                        hidden_a.append(x_w)
                    else:
                        hidden_a.append(x_a)
                x_a = self.__getattr__('enca_%d_downsample' % l)(x_a)
            x = self.dec_fuse(torch.cat((x_p, x_a), dim=1))
            feats = [x]
            for l in range(self.num_scales - 1, -1, -1):
                x = self.__getattr__('dec_%d_upsample' % l)(x)
                feats = [x] + feats
                for i in range(self.n_residual_blocks - 1, -1, -1):
                    h_p = hidden_p.pop()
                    h_a = hidden_a.pop()
                    x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, torch
                        .cat((h_p, h_a), dim=1))
            out = self.dec_output(x)
            if self.aux_output_nc or output_feats:
                aux_out = []
                if self.aux_output_nc:
                    for i in range(len(self.aux_output_nc)):
                        aux_out.append(self.__getattr__('dec_aux_output_%d' %
                            i)(x))
                if output_feats:
                    aux_out.append(feats)
                return out, aux_out
            else:
                return out


class UnetDecoder(nn.Module):
    """
    Decoder that decodes hierarachical features. Support multi-task output. Used as an external decoder of a DualUnetGenerator network
    """

    def __init__(self, output_nc=[], nf=32, max_nf=128, num_scales=7,
        n_residual_blocks=2, norm='batch', activation=nn.ReLU(False),
        gpu_ids=[]):
        super(UnetDecoder, self).__init__()
        output_nc = output_nc if isinstance(output_nc, list) else [output_nc]
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        for l in range(num_scales):
            c_in = min(nf * (l + 1), max_nf)
            c_out = min(nf * (l + 2), max_nf)
            upsample = nn.Sequential(activation, nn.Conv2d(c_out, c_in * 4,
                kernel_size=3, padding=1, bias=use_bias), nn.PixelShuffle(2
                ), norm_layer(c_in))
            self.__setattr__('dec_%d_upsample' % l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, c_in if i == 0 else None, norm_layer, use_bias,
                    activation))
        for i, c_out in enumerate(output_nc):
            dec_output_i = nn.Sequential(channel_mapping(nf, nf, norm_layer,
                use_bias), activation, nn.ReflectionPad2d(3), nn.Conv2d(nf,
                c_out, kernel_size=7))
            self.__setattr__('dec_output_%d' % i, dec_output_i)

    def forward(self, feats, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            nn.parallel.data_parallel(self, feats, module_kwargs={
                'single_device': True})
        else:
            x, hiddens = feats[-1], feats[:-1]
            for l in range(self.num_scales - 1, -1, -1):
                x = self.__getattr__('dec_%d_upsample' % l)(x)
                for i in range(self.n_residual_blocks):
                    if i == 0:
                        h = hiddens.pop()
                        x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, h)
                    else:
                        x = self.__getattr__('dec_%d_res_%d' % (l, i))(x)
            out = []
            for i in range(len(self.output_nc)):
                out.append(self.__getattr__('dec_output_%d' % i)(x))
            return out


class FlowUnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
        outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(FlowUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
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
            down = [downconv, downnorm]
            up = [uprelu, upconv, upnorm]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size
                =4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule
        self.predict_flow = nn.Sequential(nn.LeakyReLU(0.1), nn.Conv2d(
            outer_nc, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        if self.outermost:
            x_ = self.down(x)
            x_, x_pyramid, flow_pyramid = self.submodule(x_)
            x_ = self.up(x_)
            x_out = x_
        elif self.innermost:
            x_pyramid = []
            flow_pyramid = []
            x_ = self.up(self.down(x))
            x_out = torch.cat((x, x_), dim=1)
        else:
            x_ = self.down(x)
            x_, x_pyramid, flow_pyramid = self.submodule(x_)
            x_ = self.up(x_)
            x_out = torch.cat((x, x_), dim=1)
        flow = self.predict_flow(x_)
        x_pyramid = [x_] + x_pyramid
        flow_pyramid = [flow] + flow_pyramid
        return x_out, x_pyramid, flow_pyramid


class FlowUnet(nn.Module):

    def __init__(self, input_nc, nf=16, start_scale=2, num_scale=5, norm=
        'batch', gpu_ids=[], max_nf=512):
        super(FlowUnet, self).__init__()
        self.gpu_ids = gpu_ids
        self.nf = nf
        self.norm = norm
        self.start_scale = 2
        self.num_scale = 5
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        conv_downsample = [nn.Conv2d(input_nc, nf, kernel_size=7, padding=3,
            bias=use_bias), norm_layer(nf), nn.LeakyReLU(0.1)]
        nc = nf
        for i in range(np.log2(start_scale).astype(np.int)):
            conv_downsample += [nn.Conv2d(nc, 2 * nc, kernel_size=3, stride
                =2, padding=1, bias=use_bias), norm_layer(2 * nc), nn.
                LeakyReLU(0.1)]
            nc = nc * 2
        self.conv_downsample = nn.Sequential(*conv_downsample)
        unet_block = None
        for l in range(num_scale)[::-1]:
            outer_nc = min(max_nf, nc * 2 ** l)
            inner_nc = min(max_nf, nc * 2 ** (l + 1))
            innermost = l == num_scale - 1
            outermost = l == 0
            unet_block = FlowUnetSkipConnectionBlock(outer_nc, inner_nc,
                input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                innermost=innermost, outermost=outermost)
        self.unet_block = unet_block
        self.nf_out = min(max_nf, nc)
        self.predict_vis = nn.Sequential(nn.LeakyReLU(0.1), nn.Conv2d(min(
            max_nf, nc), 3, kernel_size=3, stride=1, padding=1))

    def forward(self, input, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, input, module_kwargs={
                'single_device': True})
        else:
            x = self.conv_downsample(input)
            feat_out, x_pyr, flow_pyr = self.unet_block(x)
            vis = self.predict_vis(feat_out)
            flow_out = F.upsample(flow_pyr[0], scale_factor=self.
                start_scale, mode='bilinear', align_corners=False)
            vis = F.upsample(vis, scale_factor=self.start_scale, mode=
                'bilinear', align_corners=False)
            return flow_out, vis, flow_pyr, feat_out


class FlowUnet_v2(nn.Module):
    """
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    """

    def __init__(self, input_nc, nf=64, max_nf=256, start_scale=2,
        num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU
        (False), use_dropout=False, gpu_ids=[]):
        super(FlowUnet_v2, self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.max_nf = max_nf
        self.start_scale = start_scale
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        start_level = np.log2(start_scale).astype(np.int)
        pre_conv = [channel_mapping(input_nc, nf, norm_layer, use_bias)]
        for i in range(start_level):
            c_in = min(nf * (i + 1), max_nf)
            c_out = min(nf * (i + 2), max_nf)
            pre_conv += [ResidualBlock(c_in, None, norm_layer, use_bias,
                activation, use_dropout=use_dropout), activation, nn.Conv2d
                (c_in, c_out, kernel_size=3, stride=2, padding=1, bias=
                use_bias), norm_layer(c_out)]
        self.pre_conv = nn.Sequential(*pre_conv)
        for l in range(num_scales):
            c_in = min(nf * (start_level + l + 1), max_nf)
            c_out = min(nf * (start_level + l + 2), max_nf)
            for i in range(n_residual_blocks):
                self.__setattr__('enc_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, None, norm_layer, use_bias, activation,
                    use_dropout=use_dropout))
            downsample = nn.Sequential(activation, nn.Conv2d(c_in, c_out,
                kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out))
            self.__setattr__('enc_%d_downsample' % l, downsample)
            upsample = nn.Sequential(activation, nn.Conv2d(c_out, c_in * 4,
                kernel_size=3, padding=1, bias=use_bias), nn.PixelShuffle(2
                ), norm_layer(c_in))
            self.__setattr__('dec_%d_upsample' % l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d' % (l, i), ResidualBlock(
                    c_in, c_in, norm_layer, use_bias, activation, use_dropout))
            pred_flow = nn.Sequential(activation, nn.Conv2d(c_in, 2,
                kernel_size=3, padding=1, bias=True))
            self.__setattr__('pred_flow_%d' % l, pred_flow)
        self.pred_vis = nn.Sequential(activation, nn.Conv2d(nf * (1 +
            start_level), 3, kernel_size=3, padding=1, bias=True))

    def forward(self, x, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, x, module_kwargs={
                'single_device': True})
        else:
            hiddens = []
            flow_pyr = []
            x = self.pre_conv(x)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x = self.__getattr__('enc_%d_res_%d' % (l, i))(x)
                    hiddens.append(x)
                x = self.__getattr__('enc_%d_downsample' % l)(x)
            for l in range(self.num_scales - 1, -1, -1):
                x = self.__getattr__('dec_%d_upsample' % l)(x)
                for i in range(self.n_residual_blocks - 1, -1, -1):
                    h = hiddens.pop()
                    x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, h)
                flow_pyr = [self.__getattr__('pred_flow_%d' % l)(x)] + flow_pyr
            feat_out = x
            flow_out = F.upsample(flow_pyr[0], scale_factor=self.
                start_scale, mode='bilinear', align_corners=False)
            vis_out = F.upsample(self.pred_vis(x), scale_factor=self.
                start_scale, mode='bilinear', align_corners=False)
            return flow_out, vis_out, flow_pyr, feat_out


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.
        BatchNorm2d, use_sigmoid=False, output_bias=True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
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
            padding=padw, bias=output_bias)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor
            ):
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ly015_intrinsic_flow(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(FlowUnet(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_001(self):
        self._check(GateBlock(*[], **{'dim': 4, 'dim_a': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(NLayerDiscriminator(*[], **{'input_nc': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(SS_FlowLoss(*[], **{}), [torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4])], {})

