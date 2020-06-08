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


import torch.optim as optim


from torch.nn import init


from torch import FloatTensor


from torch import tensor


from torch import cat


from torch.autograd import Variable


from torch.nn import Module


def render_model(mesh, sgrid):
    index_tri, index_ray, loc = mesh.ray.intersects_id(ray_origins=sgrid,
        ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))
    grid_hits = sgrid[index_ray]
    dist = np.linalg.norm(grid_hits - loc, axis=-1)
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    im = dist_im
    return im


def resize(im, target_size, which_dim, interpolation='bicubic', clamp=None):
    """
    Resize one dimension of the image to a certain size while maintaining the aspect ratio

    Args:
        im: Image to resize
            Any type that cv2.resize() accepts
        target_size: Target horizontal or vertical dimension
            Integer
        which_dim: Which dimension to match target_size
            'horizontal' or 'vertical'
        interpolation: Interpolation method
            'bicubic'
            Optional; defaults to 'bicubic'
        clamp: Clamp the resized image with minimum and maximum values
            Array_likes of one smaller float and another larger float
            Optional; defaults to None (no clamping)

    Returns:
        im_resized: Resized image
            Numpy array with new horizontal and vertical dimensions
    """
    h, w = im.shape[:2]
    if interpolation == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        raise NotImplementedError(interpolation)
    if which_dim == 'horizontal':
        scale_factor = target_size / w
    elif which_dim == 'vertical':
        scale_factor = target_size / h
    else:
        raise ValueError(which_dim)
    im_resized = cv2.resize(im, None, fx=scale_factor, fy=scale_factor,
        interpolation=interpolation)
    if clamp is not None:
        min_val, max_val = clamp
        im_resized[im_resized < min_val] = min_val
        im_resized[im_resized > max_val] = max_val
    return im_resized


def make_sgrid(b, alpha, beta, gamma):
    res = b * 2
    pi = np.pi
    phi = np.linspace(0, 180, res * 2 + 1)[1::2]
    theta = np.linspace(0, 360, res + 1)[:-1]
    grid = np.zeros([res, res, 3])
    for idp, p in enumerate(phi):
        for idt, t in enumerate(theta):
            grid[idp, idt, 2] = np.cos(p * pi / 180)
            proj = np.sin(p * pi / 180)
            grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
            grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
    grid = np.reshape(grid, (res * res, 3))
    return grid


def depth_to_mesh_df(depth_im, th, jitter, upsample=0.6, cam_dist=2.0):
    from util.util_camera import tsdf_renderer
    depth = depth_im[:, :, (0)]
    mask = np.where(depth == 0, -1.0, 1.0)
    depth = 1 - depth
    t = tsdf_renderer()
    thl = th[0]
    thh = th[1]
    if jitter:
        th = th + (np.random.rand(2) - 0.5) * 0.1
        thl = np.min(th)
        thh = np.max(th)
    scale = thh - thl
    depth = depth * scale
    t.depth = (depth + thl) * mask
    t.camera.focal_length = 0.05
    t.camera.sensor_width = 0.03059411708155671
    t.camera.position = np.array([-cam_dist, 0, 0])
    t.camera.res = [480, 480]
    t.camera.rx = np.array([0, 0, 1])
    t.camera.ry = np.array([0, 1, 0])
    t.camera.rz = -np.array([1, 0, 0])
    t.back_project_ptcloud(upsample=upsample)
    tdf = np.ones([128, 128, 128]) / 128
    cnt = np.zeros([128, 128, 128])
    for pts in t.ptcld:
        pt = pts
        ids = np.floor((pt + 0.5) * 128).astype(int)
        if np.any(np.abs(pt) >= 0.5):
            continue
        center = (ids + 0.5) * 1 / 128 - 0.5
        dist = ((center[0] - pt[0]) ** 2 + (center[1] - pt[1]) ** 2 + (
            center[2] - pt[2]) ** 2) ** 0.5
        n = cnt[ids[0], ids[1], ids[2]]
        tdf[ids[0], ids[1], ids[2]] = (tdf[ids[0], ids[1], ids[2]] * n + dist
            ) / (n + 1)
        cnt[ids[0], ids[1], ids[2]] += 1
    return tdf


def render_spherical(data, mask, obj_path=None, debug=False):
    depth_im = data['depth'][(0), (0), :, :]
    th = data['depth_minmax']
    depth_im = resize(depth_im, 480, 'vertical')
    im = resize(mask, 480, 'vertical')
    gt_sil = np.where(im > 0.95, 1, 0)
    depth_im = depth_im * gt_sil
    depth_im = depth_im[:, :, (np.newaxis)]
    b = 64
    tdf = depth_to_mesh_df(depth_im, th, False, 1.0, 2.2)
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(tdf,
            0.999 / 128, spacing=(1 / 128, 1 / 128, 1 / 128))
        mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
        sgrid = make_sgrid(b, 0, 0, 0)
        im_depth = render_model(mesh, sgrid)
        im_depth = im_depth.reshape(2 * b, 2 * b)
        im_depth = np.where(im_depth > 1, 1, im_depth)
    except:
        im_depth = np.ones([128, 128])
        return im_depth
    return im_depth


def sph_pad(sph_tensor, padding_margin=16):
    F = torch.nn.functional
    pad2d = padding_margin, padding_margin, padding_margin, padding_margin
    rep_padded_sph = F.pad(sph_tensor, pad2d, mode='replicate')
    _, _, h, w = rep_padded_sph.shape
    rep_padded_sph[:, :, :, 0:padding_margin] = rep_padded_sph[:, :, :, w -
        2 * padding_margin:w - padding_margin]
    rep_padded_sph[:, :, :, h - padding_margin:] = rep_padded_sph[:, :, :,
        padding_margin:2 * padding_margin]
    return rep_padded_sph


def gen_sph_grid(res=128):
    pi = np.pi
    phi = np.linspace(0, 180, res * 2 + 1)[1::2]
    theta = np.linspace(0, 360, res + 1)[:-1]
    grid = np.zeros([res, res, 3])
    for idp, p in enumerate(phi):
        for idt, t in enumerate(theta):
            grid[idp, idt, 2] = np.cos(p * pi / 180)
            proj = np.sin(p * pi / 180)
            grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
            grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
    grid = np.reshape(grid, (1, 1, res, res, 3))
    return torch.from_numpy(grid).float()


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


def relu():
    return nn.ReLU(inplace=True)


def deconv3d_2x(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(n_ch_in, n_ch_out, 4, stride=2, padding=1,
        dilation=1, groups=1, bias=bias)


def deconv3d_add3(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(n_ch_in, n_ch_out, 4, stride=1, padding=0,
        dilation=1, groups=1, bias=bias)


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


def conv3d_minus3(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1,
        groups=1, bias=bias)


def relu_leaky():
    return nn.LeakyReLU(0.2, inplace=True)


def conv3d_half(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1,
        groups=1, bias=bias)


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
