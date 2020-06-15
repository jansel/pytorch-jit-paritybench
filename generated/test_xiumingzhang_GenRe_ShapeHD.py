import sys
_module = sys.modules[__name__]
del sys
datasets = _module
shapenet = _module
test = _module
Progbar = _module
loggers = _module
models = _module
depth_pred_with_sph_inpaint = _module
genre_full_model = _module
marrnet = _module
marrnet1 = _module
marrnet2 = _module
marrnetbase = _module
netinterface = _module
shapehd = _module
wgangp = _module
networks = _module
networks = _module
revresnet = _module
uresnet = _module
options = _module
options_test = _module
options_train = _module
toolbox = _module
build = _module
calc_prob = _module
functions = _module
setup = _module
cam_bp = _module
cam_back_projection = _module
get_surface_mask = _module
sperical_to_tdf = _module
Spherical_backproj = _module
modules = _module
camera_backprojection_module = _module
nnd = _module
nnd = _module
spherical_proj = _module
train = _module
util = _module
util_cam_para = _module
util_camera = _module
util_img = _module
util_io = _module
util_loadlib = _module
util_print = _module
util_reproj = _module
util_sph = _module
util_voxel = _module
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from scipy.ndimage.morphology import binary_erosion


from torch import nn


import time


import torch.optim as optim


from torch.nn import init


from torch import FloatTensor


from torch import tensor


from torch import cat


from torch.autograd import Variable


from torch.nn import Module


class Net(nn.Module):
    """
       MarrNet-1    MarrNet-2
    RGB ------> 2.5D ------> 3D
         fixed      finetuned
    """

    def __init__(self, marrnet1_path=None, marrnet2_path=None,
        pred_silhou_thres=0.3):
        super().__init__()
        self.marrnet1 = Marrnet1([3, 1, 1], ['normal', 'depth', 'silhou'],
            pred_depth_minmax=True)
        if marrnet1_path:
            state_dict = torch.load(marrnet1_path)['nets'][0]
            self.marrnet1.load_state_dict(state_dict)
        self.marrnet2 = Marrnet2(4)
        if marrnet2_path:
            state_dict = torch.load(marrnet2_path)['nets'][0]
            self.marrnet2.load_state_dict(state_dict)
        for p in self.marrnet1.parameters():
            p.requires_grad = False
        for p in self.marrnet2.parameters():
            p.requires_grad = True
        self.pred_silhou_thres = pred_silhou_thres

    def forward(self, input_struct):
        with torch.no_grad():
            pred = self.marrnet1(input_struct)
        depth = pred['depth']
        normal = pred['normal']
        silhou = pred['silhou']
        is_bg = silhou < self.pred_silhou_thres
        depth[is_bg] = 0
        normal[is_bg.repeat(1, 3, 1, 1)] = 0
        x = torch.cat((depth, normal), 1)
        latent_vec = self.marrnet2.encoder(x)
        vox = self.marrnet2.decoder(latent_vec)
        pred['voxel'] = vox
        return pred


class Net(nn.Module):
    """
    2.5D maps to 3D voxel
    """

    def __init__(self, in_planes, encode_dims=200, silhou_thres=0):
        super().__init__()
        self.encoder = ImageEncoder(in_planes, encode_dims=encode_dims)
        self.decoder = VoxelDecoder(n_dims=encode_dims, nf=512)
        self.silhou_thres = silhou_thres

    def forward(self, input_struct):
        depth = input_struct.depth
        normal = input_struct.normal
        silhou = input_struct.silhou
        is_bg = silhou <= self.silhou_thres
        depth[is_bg] = 0
        normal[is_bg.repeat(1, 3, 1, 1)] = 0
        x = torch.cat((depth, normal), 1)
        latent_vec = self.encoder(x)
        vox = self.decoder(latent_vec)
        return vox


class ImageEncoder(nn.Module):
    """
    Used for 2.5D maps to 3D voxels
    """

    def __init__(self, input_nc, encode_dims=200):
        super().__init__()
        resnet_m = resnet18(pretrained=True)
        resnet_m.conv1 = nn.Conv2d(input_nc, 64, 7, stride=2, padding=3,
            bias=False)
        resnet_m.avgpool = nn.AdaptiveAvgPool2d(1)
        resnet_m.fc = nn.Linear(512, encode_dims)
        self.main = nn.Sequential(resnet_m)

    def forward(self, x):
        return self.main(x)


def batchnorm3d(n_feat):
    return nn.BatchNorm3d(n_feat, eps=1e-05, momentum=0.1, affine=True)


def deconv3d_2x(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(n_ch_in, n_ch_out, 4, stride=2, padding=1,
        dilation=1, groups=1, bias=bias)


def deconv3d_add3(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(n_ch_in, n_ch_out, 4, stride=1, padding=0,
        dilation=1, groups=1, bias=bias)


def relu():
    return nn.ReLU(inplace=True)


class VoxelDecoder(nn.Module):
    """
    Used for 2.5D maps to 3D voxels
    """

    def __init__(self, n_dims=200, nf=512):
        super().__init__()
        self.main = nn.Sequential(deconv3d_add3(n_dims, nf, True),
            batchnorm3d(nf), relu(), deconv3d_2x(nf, nf // 2, True),
            batchnorm3d(nf // 2), relu(), nn.Sequential(), nn.Sequential(),
            deconv3d_2x(nf // 2, nf // 4, True), batchnorm3d(nf // 4), relu
            (), deconv3d_2x(nf // 4, nf // 8, True), batchnorm3d(nf // 8),
            relu(), deconv3d_2x(nf // 8, nf // 16, True), batchnorm3d(nf //
            16), relu(), deconv3d_2x(nf // 16, 1, True))

    def forward(self, x):
        x_vox = x.view(x.size(0), -1, 1, 1, 1)
        return self.main(x_vox)


class VoxelGenerator(nn.Module):

    def __init__(self, nz=200, nf=64, bias=False, res=128):
        super().__init__()
        layers = [deconv3d_add3(nz, nf * 8, bias), batchnorm3d(nf * 8),
            relu(), deconv3d_2x(nf * 8, nf * 4, bias), batchnorm3d(nf * 4),
            relu(), deconv3d_2x(nf * 4, nf * 2, bias), batchnorm3d(nf * 2),
            relu(), deconv3d_2x(nf * 2, nf, bias), batchnorm3d(nf), relu()]
        if res == 64:
            layers.append(deconv3d_2x(nf, 1, bias))
        elif res == 128:
            layers += [deconv3d_2x(nf, nf, bias), batchnorm3d(nf), relu(),
                deconv3d_2x(nf, 1, bias)]
        else:
            raise NotImplementedError(res)
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def conv3d_half(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1,
        groups=1, bias=bias)


def conv3d_minus3(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1,
        groups=1, bias=bias)


def relu_leaky():
    return nn.LeakyReLU(0.2, inplace=True)


class VoxelDiscriminator(nn.Module):

    def __init__(self, nf=64, bias=False, res=128):
        super().__init__()
        layers = [conv3d_half(1, nf, bias), relu_leaky(), conv3d_half(nf, 
            nf * 2, bias), relu_leaky(), conv3d_half(nf * 2, nf * 4, bias),
            relu_leaky(), conv3d_half(nf * 4, nf * 8, bias), relu_leaky(),
            conv3d_minus3(nf * 8, 1, bias)]
        if res == 64:
            pass
        elif res == 128:
            extra_layers = [conv3d_half(nf, nf, bias), relu_leaky()]
            layers = layers[:2] + extra_layers + layers[2:]
        else:
            raise NotImplementedError(res)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        y = self.main(x)
        return y.view(-1, 1).squeeze(1)


class Unet_3D(nn.Module):

    def __init__(self, nf=20, in_channel=2, no_linear=False):
        super(Unet_3D, self).__init__()
        self.nf = nf
        self.enc1 = Conv3d_block(in_channel, nf, 8, 2, 3)
        self.enc2 = Conv3d_block(nf, 2 * nf, 4, 2, 1)
        self.enc3 = Conv3d_block(2 * nf, 4 * nf, 4, 2, 1)
        self.enc4 = Conv3d_block(4 * nf, 8 * nf, 4, 2, 1)
        self.enc5 = Conv3d_block(8 * nf, 16 * nf, 4, 2, 1)
        self.enc6 = Conv3d_block(16 * nf, 32 * nf, 4, 1, 0)
        self.full_conv_block = nn.Sequential(nn.Linear(32 * nf, 32 * nf),
            nn.LeakyReLU())
        self.dec1 = Deconv3d_skip(32 * 2 * nf, 16 * nf, 4, 1, 0, 0)
        self.dec2 = Deconv3d_skip(16 * 2 * nf, 8 * nf, 4, 2, 1, 0)
        self.dec3 = Deconv3d_skip(8 * 2 * nf, 4 * nf, 4, 2, 1, 0)
        self.dec4 = Deconv3d_skip(4 * 2 * nf, 2 * nf, 4, 2, 1, 0)
        self.dec5 = Deconv3d_skip(4 * nf, nf, 8, 2, 3, 0)
        self.dec6 = Deconv3d_skip(2 * nf, 1, 4, 2, 1, 0, is_activate=False)
        self.no_linear = no_linear

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        if not self.no_linear:
            flatten = enc6.view(enc6.size()[0], self.nf * 32)
            bottleneck = self.full_conv_block(flatten)
            bottleneck = bottleneck.view(enc6.size()[0], self.nf * 32, 1, 1, 1)
            dec1 = self.dec1(bottleneck, enc6)
        else:
            dec1 = self.dec1(enc6, enc6)
        dec2 = self.dec2(dec1, enc5)
        dec3 = self.dec3(dec2, enc4)
        dec4 = self.dec4(dec3, enc3)
        dec5 = self.dec5(dec4, enc2)
        out = self.dec6(dec5, enc1)
        return out


class Conv3d_block(nn.Module):

    def __init__(self, ncin, ncout, kernel_size, stride, pad, dropout=False):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(ncin, ncout, kernel_size, stride,
            pad), nn.BatchNorm3d(ncout), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class Deconv3d_skip(nn.Module):

    def __init__(self, ncin, ncout, kernel_size, stride, pad, extra=0,
        is_activate=True):
        super(Deconv3d_skip, self).__init__()
        if is_activate:
            self.net = nn.Sequential(nn.ConvTranspose3d(ncin, ncout,
                kernel_size, stride, pad, extra), nn.BatchNorm3d(ncout), nn
                .LeakyReLU())
        else:
            self.net = nn.ConvTranspose3d(ncin, ncout, kernel_size, stride,
                pad, extra)

    def forward(self, x, skip_in):
        y = cat((x, skip_in), dim=1)
        return self.net(y)


class ViewAsLinear(nn.Module):

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=
        stride, padding=1, bias=False, output_padding=output_padding)


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(RevBasicBlock, self).__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes, stride=stride,
            output_padding=1 if stride > 1 else 0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(RevBottleneck, self).__init__()
        bottleneck_planes = int(inplanes / 4)
        self.deconv1 = nn.ConvTranspose2d(inplanes, bottleneck_planes,
            kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv2 = nn.ConvTranspose2d(bottleneck_planes,
            bottleneck_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv3 = nn.ConvTranspose2d(bottleneck_planes, planes,
            kernel_size=1, bias=False, stride=stride, output_padding=1 if 
            stride > 0 else 0)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class RevResNet(nn.Module):

    def __init__(self, block, layers, planes, inplanes=None, out_planes=5):
        """
        planes: # output channels for each block
        inplanes: # input channels for the input at each layer
            If missing, it will be inferred.
        """
        if inplanes is None:
            inplanes = [512]
        self.inplanes = inplanes[0]
        super(RevResNet, self).__init__()
        inplanes_after_blocks = inplanes[4] if len(inplanes) > 4 else planes[3]
        self.deconv1 = nn.ConvTranspose2d(inplanes_after_blocks, planes[3],
            kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(planes[3], out_planes,
            kernel_size=7, stride=2, padding=3, bias=False, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes[3])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0], stride=2)
        if len(inplanes) > 1:
            self.inplanes = inplanes[1]
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        if len(inplanes) > 2:
            self.inplanes = inplanes[2]
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        if len(inplanes) > 3:
            self.inplanes = inplanes[3]
        self.layer4 = self._make_layer(block, planes[3], layers[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(nn.ConvTranspose2d(self.inplanes,
                planes, kernel_size=1, stride=stride, bias=False,
                output_padding=1 if stride > 1 else 0), nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x


def revuresnet18(**kwargs):
    """
    Reverse ResNet-18 compatible with the U-Net setting
    """
    model = RevResNet(RevBasicBlock, [2, 2, 2, 2], [256, 128, 64, 64],
        inplanes=[512, 512, 256, 128, 128], **kwargs)
    return model


class Net(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, out_planes, layer_names, input_planes=3):
        super().__init__()
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        module_list.append(nn.Sequential(resnet.conv1 if input_planes == 3 else
            in_conv, resnet.bn1, resnet.relu, resnet.maxpool))
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list.append(revresnet.layer1)
            module_list.append(revresnet.layer2)
            module_list.append(revresnet.layer3)
            module_list.append(revresnet.layer4)
            module_list.append(nn.Sequential(revresnet.deconv1, revresnet.
                bn1, revresnet.relu, revresnet.deconv2))
            module_list = nn.ModuleList(module_list)
            setattr(self, 'decoder_' + layer_name, module_list)
            self.decoders[layer_name] = module_list

    def forward(self, im):
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs


class Net_inpaint(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, out_planes, layer_names, input_planes=3):
        super().__init__()
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        module_list.append(nn.Sequential(resnet.conv1 if input_planes == 3 else
            in_conv, resnet.bn1, resnet.relu, resnet.maxpool))
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=2,
            padding=3, bias=False, output_padding=0)
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list.append(revresnet.layer1)
            module_list.append(revresnet.layer2)
            module_list.append(revresnet.layer3)
            module_list.append(revresnet.layer4)
            module_list.append(nn.Sequential(revresnet.deconv1, revresnet.
                bn1, revresnet.relu, self.deconv2))
            module_list = nn.ModuleList(module_list)
            setattr(self, 'decoder_' + layer_name, module_list)
            self.decoders[layer_name] = module_list

    def forward(self, im):
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs


class spherical_backprojection(nn.Module):

    def __init__(self, grid, vox_res=128):
        super(camera_backprojection, self).__init__()
        self.vox_res = vox_res
        self.backprojection_layer = SphericalBackProjection()
        assert type(grid) == torch.FloatTensor
        self.grid = Variable(grid)

    def forward(self, spherical):
        return self.backprojection_layer(spherical, self.grid, self.vox_res)


class Camera_back_projection_layer(nn.Module):

    def __init__(self, res=128):
        super(Camera_back_projection_layer, self).__init__()
        assert res == 128
        self.res = 128

    def forward(self, depth_t, fl=418.3, cam_dist=2.2, shift=True):
        n = depth_t.size(0)
        if type(fl) == float:
            fl_v = fl
            fl = torch.FloatTensor(n, 1)
            fl.fill_(fl_v)
        if type(cam_dist) == float:
            cmd_v = cam_dist
            cam_dist = torch.FloatTensor(n, 1)
            cam_dist.fill_(cmd_v)
        df = CameraBackProjection.apply(depth_t, fl, cam_dist, self.res)
        return self.shift_tdf(df) if shift else df

    @staticmethod
    def shift_tdf(input_tdf, res=128):
        out_tdf = 1 - res * input_tdf
        return out_tdf


class camera_backprojection(nn.Module):

    def __init__(self, vox_res=128):
        super(camera_backprojection, self).__init__()
        self.vox_res = vox_res
        self.backprojection_layer = CameraBackProjection()

    def forward(self, depth, fl, camdist):
        return self.backprojection_layer(depth, fl, camdist, self.voxel_res)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_xiumingzhang_GenRe_ShapeHD(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv3d_block(*[], **{'ncin': 4, 'ncout': 4, 'kernel_size': 4, 'stride': 1, 'pad': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_001(self):
        self._check(Deconv3d_skip(*[], **{'ncin': 4, 'ncout': 4, 'kernel_size': 4, 'stride': 1, 'pad': 4}), [torch.rand([4, 1, 64, 64, 64]), torch.rand([4, 3, 64, 64, 64])], {})

    def test_002(self):
        self._check(RevBasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ViewAsLinear(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

