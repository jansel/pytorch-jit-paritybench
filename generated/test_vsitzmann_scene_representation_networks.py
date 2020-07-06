import sys
_module = sys.modules[__name__]
del sys
custom_layers = _module
data_util = _module
dataio = _module
geometry = _module
hyperlayers = _module
srns = _module
test = _module
train = _module
util = _module

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


import torchvision


import torch


from torch import nn


import numpy as np


from torch.nn import functional as F


import torch.nn as nn


import functools


import time


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import math


import torch.nn.functional as F


class DepthSampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, xy, depth, cam2world, intersection_net, intrinsics):
        self.logs = list()
        batch_size, _, _ = cam2world.shape
        intersections = geometry.world_from_xy_depth(xy=xy, depth=depth, cam2world=cam2world, intrinsics=intrinsics)
        depth = geometry.depth_from_world(intersections, cam2world)
        if self.training:
            None
        return intersections, depth


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not 'bias' in name:
            continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.0)


class Raymarcher(nn.Module):

    def __init__(self, num_feature_channels, raymarch_steps):
        super().__init__()
        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps
        hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels, hidden_size=hidden_size)
        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)
        self.out_layer = nn.Linear(hidden_size, 1)
        self.counter = 0

    def forward(self, cam2world, phi, uv, intrinsics):
        batch_size, num_samples, _ = uv.shape
        log = list()
        ray_dirs = geometry.get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)
        initial_depth = torch.zeros((batch_size, num_samples, 1)).normal_(mean=0.05, std=0.0005)
        init_world_coords = geometry.world_from_xy_depth(uv, initial_depth, intrinsics=intrinsics, cam2world=cam2world)
        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]
        for step in range(self.steps):
            v = phi(world_coords[-1])
            state = self.lstm(v.view(-1, self.n_feature_channels), states[-1])
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))
            signed_distance = self.out_layer(state[0]).view(batch_size, num_samples, 1)
            new_world_coords = world_coords[-1] + ray_dirs * signed_distance
            states.append(state)
            world_coords.append(new_world_coords)
            depth = geometry.depth_from_world(world_coords[-1], cam2world)
            if self.training:
                None
            depths.append(depth)
        if not self.counter % 100:
            drawing_depths = torch.stack(depths, dim=0)[:, (0), :, :]
            drawing_depths = util.lin2img(drawing_depths).repeat(1, 3, 1, 1)
            log.append(('image', 'raycast_progress', torch.clamp(torchvision.utils.make_grid(drawing_depths, scale_each=False, normalize=True), 0.0, 5), 100))
            fig = util.show_images([util.lin2img(signed_distance)[(i), :, :, :].detach().cpu().numpy().squeeze() for i in range(batch_size)])
            log.append(('figure', 'stopping_distances', fig, 100))
        self.counter += 1
        return world_coords[-1], depths[-1], log


class DeepvoxelsRenderer(nn.Module):

    def __init__(self, nf0, in_channels, input_resolution, img_sidelength):
        super().__init__()
        self.nf0 = nf0
        self.in_channels = in_channels
        self.input_resolution = input_resolution
        self.img_sidelength = img_sidelength
        self.num_down_unet = util.num_divisible_by_2(input_resolution)
        self.num_upsampling = util.num_divisible_by_2(img_sidelength) - self.num_down_unet
        self.build_net()

    def build_net(self):
        self.net = [pytorch_prototyping.Unet(in_channels=self.in_channels, out_channels=3 if self.num_upsampling <= 0 else 4 * self.nf0, outermost_linear=True if self.num_upsampling <= 0 else False, use_dropout=True, dropout_prob=0.1, nf0=self.nf0 * 2 ** self.num_upsampling, norm=nn.BatchNorm2d, max_channels=8 * self.nf0, num_down=self.num_down_unet)]
        if self.num_upsampling > 0:
            self.net += [pytorch_prototyping.UpsamplingNet(per_layer_out_ch=self.num_upsampling * [self.nf0], in_channels=4 * self.nf0, upsampling_mode='transpose', use_dropout=True, dropout_prob=0.1), pytorch_prototyping.Conv2dSame(self.nf0, out_channels=self.nf0 // 2, kernel_size=3, bias=False), nn.BatchNorm2d(self.nf0 // 2), nn.ReLU(True), pytorch_prototyping.Conv2dSame(self.nf0 // 2, 3, kernel_size=3)]
        self.net += [nn.Tanh()]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        batch_size, _, ch = input.shape
        input = input.permute(0, 2, 1).view(batch_size, ch, self.img_sidelength, self.img_sidelength)
        out = self.net(input)
        return out.view(batch_size, 3, -1).permute(0, 2, 1)


class BatchLinear(nn.Module):

    def __init__(self, weights, biases):
        """Implements a batch linear layer.

        :param weights: Shape: (batch, out_ch, in_ch)
        :param biases: Shape: (batch, 1, out_ch)
        """
        super().__init__()
        self.weights = weights
        self.biases = biases

    def __repr__(self):
        return 'BatchLinear(in_ch=%d, out_ch=%d)' % (self.weights.shape[-1], self.weights.shape[-2])

    def forward(self, input):
        output = input.matmul(self.weights.permute(*[i for i in range(len(self.weights.shape) - 2)], -1, -2))
        output += self.biases
        return output


class LookupLinear(nn.Module):

    def __init__(self, in_ch, out_ch, num_objects):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hypo_params = nn.Embedding(num_objects, in_ch * out_ch + out_ch)
        for i in range(num_objects):
            nn.init.kaiming_normal_(self.hypo_params.weight.data[(i), :self.in_ch * self.out_ch].view(self.out_ch, self.in_ch), a=0.0, nonlinearity='relu', mode='fan_in')
            self.hypo_params.weight.data[(i), self.in_ch * self.out_ch:].fill_(0.0)

    def forward(self, obj_idx):
        hypo_params = self.hypo_params(obj_idx)
        weights = hypo_params[(...), :self.in_ch * self.out_ch]
        biases = hypo_params[(...), self.in_ch * self.out_ch:self.in_ch * self.out_ch + self.out_ch]
        biases = biases.view(*biases.size()[:-1], 1, self.out_ch)
        weights = weights.view(*weights.size()[:-1], self.out_ch, self.in_ch)
        return BatchLinear(weights=weights, biases=biases)


class LookupLayer(nn.Module):

    def __init__(self, in_ch, out_ch, num_objects):
        super().__init__()
        self.out_ch = out_ch
        self.lookup_lin = LookupLinear(in_ch, out_ch, num_objects=num_objects)
        self.norm_nl = nn.Sequential(nn.LayerNorm([self.out_ch], elementwise_affine=False), nn.ReLU(inplace=True))

    def forward(self, obj_idx):
        net = nn.Sequential(self.lookup_lin(obj_idx), self.norm_nl)
        return net


class LookupFC(nn.Module):

    def __init__(self, hidden_ch, num_hidden_layers, num_objects, in_ch, out_ch, outermost_linear=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LookupLayer(in_ch=in_ch, out_ch=hidden_ch, num_objects=num_objects))
        for i in range(num_hidden_layers):
            self.layers.append(LookupLayer(in_ch=hidden_ch, out_ch=hidden_ch, num_objects=num_objects))
        if outermost_linear:
            self.layers.append(LookupLinear(in_ch=hidden_ch, out_ch=out_ch, num_objects=num_objects))
        else:
            self.layers.append(LookupLayer(in_ch=hidden_ch, out_ch=out_ch, num_objects=num_objects))

    def forward(self, obj_idx):
        net = []
        for i in range(len(self.layers)):
            net.append(self.layers[i](obj_idx))
        return nn.Sequential(*net)


def last_hyper_layer_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data *= 0.1


class HyperLinear(nn.Module):
    """A hypernetwork that predicts a single linear layer (weights & biases)."""

    def __init__(self, in_ch, out_ch, hyper_in_ch, hyper_num_hidden_layers, hyper_hidden_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hypo_params = pytorch_prototyping.FCBlock(in_features=hyper_in_ch, hidden_ch=hyper_hidden_ch, num_hidden_layers=hyper_num_hidden_layers, out_features=in_ch * out_ch + out_ch, outermost_linear=True)
        self.hypo_params[-1].apply(last_hyper_layer_init)

    def forward(self, hyper_input):
        hypo_params = self.hypo_params(hyper_input)
        weights = hypo_params[(...), :self.in_ch * self.out_ch]
        biases = hypo_params[(...), self.in_ch * self.out_ch:self.in_ch * self.out_ch + self.out_ch]
        biases = biases.view(*biases.size()[:-1], 1, self.out_ch)
        weights = weights.view(*weights.size()[:-1], self.out_ch, self.in_ch)
        return BatchLinear(weights=weights, biases=biases)


class HyperLayer(nn.Module):
    """A hypernetwork that predicts a single Dense Layer, including LayerNorm and a ReLU."""

    def __init__(self, in_ch, out_ch, hyper_in_ch, hyper_num_hidden_layers, hyper_hidden_ch):
        super().__init__()
        self.hyper_linear = HyperLinear(in_ch=in_ch, out_ch=out_ch, hyper_in_ch=hyper_in_ch, hyper_num_hidden_layers=hyper_num_hidden_layers, hyper_hidden_ch=hyper_hidden_ch)
        self.norm_nl = nn.Sequential(nn.LayerNorm([out_ch], elementwise_affine=False), nn.ReLU(inplace=True))

    def forward(self, hyper_input):
        """
        :param hyper_input: input to hypernetwork.
        :return: nn.Module; predicted fully connected network.
        """
        return nn.Sequential(self.hyper_linear(hyper_input), self.norm_nl)


def partialclass(cls, *args, **kwds):


    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewCls


class HyperFC(nn.Module):
    """Builds a hypernetwork that predicts a fully connected neural network.
    """

    def __init__(self, hyper_in_ch, hyper_num_hidden_layers, hyper_hidden_ch, hidden_ch, num_hidden_layers, in_ch, out_ch, outermost_linear=False):
        super().__init__()
        PreconfHyperLinear = partialclass(HyperLinear, hyper_in_ch=hyper_in_ch, hyper_num_hidden_layers=hyper_num_hidden_layers, hyper_hidden_ch=hyper_hidden_ch)
        PreconfHyperLayer = partialclass(HyperLayer, hyper_in_ch=hyper_in_ch, hyper_num_hidden_layers=hyper_num_hidden_layers, hyper_hidden_ch=hyper_hidden_ch)
        self.layers = nn.ModuleList()
        self.layers.append(PreconfHyperLayer(in_ch=in_ch, out_ch=hidden_ch))
        for i in range(num_hidden_layers):
            self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=hidden_ch))
        if outermost_linear:
            self.layers.append(PreconfHyperLinear(in_ch=hidden_ch, out_ch=out_ch))
        else:
            self.layers.append(PreconfHyperLayer(in_ch=hidden_ch, out_ch=out_ch))

    def forward(self, hyper_input):
        """
        :param hyper_input: Input to hypernetwork.
        :return: nn.Module; Predicted fully connected neural network.
        """
        net = []
        for i in range(len(self.layers)):
            net.append(self.layers[i](hyper_input))
        return nn.Sequential(*net)


class SRNsModel(nn.Module):

    def __init__(self, num_instances, latent_dim, tracing_steps, has_params=False, fit_single_srn=False, use_unet_renderer=False, freeze_networks=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.has_params = has_params
        self.num_hidden_units_phi = 256
        self.phi_layers = 4
        self.rendering_layers = 5
        self.sphere_trace_steps = tracing_steps
        self.freeze_networks = freeze_networks
        self.fit_single_srn = fit_single_srn
        if self.fit_single_srn:
            self.phi = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=self.phi_layers - 2, in_features=3, out_features=self.num_hidden_units_phi)
        else:
            self.latent_codes = nn.Embedding(num_instances, latent_dim)
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
            self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim, hyper_num_hidden_layers=1, hyper_hidden_ch=self.latent_dim, hidden_ch=self.num_hidden_units_phi, num_hidden_layers=self.phi_layers - 2, in_ch=3, out_ch=self.num_hidden_units_phi)
        self.ray_marcher = custom_layers.Raymarcher(num_feature_channels=self.num_hidden_units_phi, raymarch_steps=self.sphere_trace_steps)
        if use_unet_renderer:
            self.pixel_generator = custom_layers.DeepvoxelsRenderer(nf0=32, in_channels=self.num_hidden_units_phi, input_resolution=128, img_sidelength=128)
        else:
            self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=self.rendering_layers - 1, in_features=self.num_hidden_units_phi, out_features=3, outermost_linear=True)
        if self.freeze_networks:
            all_network_params = list(self.pixel_generator.parameters()) + list(self.ray_marcher.parameters()) + list(self.hyper_phi.parameters())
            for param in all_network_params:
                param.requires_grad = False
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.logs = list()
        None
        None
        util.print_network(self)

    def get_regularization_loss(self, prediction, ground_truth):
        """Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        """
        _, depth = prediction
        neg_penalty = torch.min(depth, torch.zeros_like(depth)) ** 2
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, prediction, ground_truth):
        """Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']
        trgt_imgs = trgt_imgs
        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """
        if self.fit_single_srn:
            self.latent_reg_loss = 0
        else:
            self.latent_reg_loss = torch.mean(self.z ** 2)
        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        """Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']
        trgt_imgs = trgt_imgs
        batch_size = pred_imgs.shape[0]
        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()
        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()
        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)
            p = p / 2.0 + 0.5
            p = np.clip(p, a_min=0.0, a_max=1.0)
            trgt = trgt / 2.0 + 0.5
            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)
            psnrs.append(psnr)
            ssims.append(ssim)
        return psnrs, ssims

    def get_comparisons(self, model_input, prediction, ground_truth=None):
        predictions, depth_maps = prediction
        batch_size = predictions.shape[0]
        intrinsics = model_input['intrinsics']
        uv = model_input['uv'].float()
        x_cam = uv[:, :, (0)].view(batch_size, -1)
        y_cam = uv[:, :, (1)].view(batch_size, -1)
        z_cam = depth_maps.view(batch_size, -1)
        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode='constant', value=1.0)
        predictions = util.lin2img(predictions)
        if ground_truth is not None:
            trgt_imgs = ground_truth['rgb']
            trgt_imgs = util.lin2img(trgt_imgs)
            return torch.cat((normals.cpu(), predictions.cpu(), trgt_imgs.cpu()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), predictions.cpu()), dim=3).numpy()

    def get_output_img(self, prediction):
        pred_imgs, _ = prediction
        return util.lin2img(pred_imgs)

    def write_updates(self, writer, predictions, ground_truth, iter, prefix=''):
        """Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        """
        predictions, depth_maps = predictions
        trgt_imgs = ground_truth['rgb']
        trgt_imgs = trgt_imgs
        batch_size, num_samples, _ = predictions.shape
        for type, name, content, every_n in self.logs:
            name = prefix + name
            if not iter % every_n:
                if type == 'image':
                    writer.add_image(name, content.detach().cpu().numpy(), iter)
                    writer.add_scalar(name + '_min', content.min(), iter)
                    writer.add_scalar(name + '_max', content.max(), iter)
                elif type == 'figure':
                    writer.add_figure(name, content, iter, close=True)
                elif type == 'histogram':
                    writer.add_histogram(name, content.detach().cpu().numpy(), iter)
                elif type == 'scalar':
                    writer.add_scalar(name, content.detach().cpu().numpy(), iter)
                elif type == 'embedding':
                    writer.add_embedding(mat=content, global_step=iter)
        if not iter % 100:
            output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
            output_vs_gt = util.lin2img(output_vs_gt)
            writer.add_image(prefix + 'Output_vs_gt', torchvision.utils.make_grid(output_vs_gt, scale_each=False, normalize=True).cpu().detach().numpy(), iter)
            rgb_loss = ((predictions.float() - trgt_imgs.float()) ** 2).mean(dim=2, keepdim=True)
            rgb_loss = util.lin2img(rgb_loss)
            fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze() for i in range(batch_size)])
            writer.add_figure(prefix + 'rgb_error_fig', fig, iter, close=True)
            depth_maps_plot = util.lin2img(depth_maps)
            writer.add_image(prefix + 'pred_depth', torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1), scale_each=True, normalize=True).cpu().detach().numpy(), iter)
        writer.add_scalar(prefix + 'out_min', predictions.min(), iter)
        writer.add_scalar(prefix + 'out_max', predictions.max(), iter)
        writer.add_scalar(prefix + 'trgt_min', trgt_imgs.min(), iter)
        writer.add_scalar(prefix + 'trgt_max', trgt_imgs.max(), iter)
        if iter:
            writer.add_scalar(prefix + 'latent_reg_loss', self.latent_reg_loss, iter)

    def forward(self, input, z=None):
        self.logs = list()
        instance_idcs = input['instance_idx'].long()
        pose = input['pose']
        intrinsics = input['intrinsics']
        uv = input['uv'].float()
        if self.fit_single_srn:
            phi = self.phi
        else:
            if self.has_params:
                if z is None:
                    self.z = input['param']
                else:
                    self.z = z
            else:
                self.z = self.latent_codes(instance_idcs)
            phi = self.hyper_phi(self.z)
        points_xyz, depth_maps, log = self.ray_marcher(cam2world=pose, intrinsics=intrinsics, uv=uv, phi=phi)
        self.logs.extend(log)
        v = phi(points_xyz)
        novel_views = self.pixel_generator(v)
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, (0)].view(batch_size, -1)
            y_cam = uv[:, :, (1)].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)
            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(('image', 'normals', torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))
        if not self.fit_single_srn:
            self.logs.append(('embedding', '', self.latent_codes.weight, 500))
            self.logs.append(('scalar', 'embed_min', self.z.min(), 1))
            self.logs.append(('scalar', 'embed_max', self.z.max(), 1))
        return novel_views, depth_maps


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LookupFC,
     lambda: ([], {'hidden_ch': 4, 'num_hidden_layers': 1, 'num_objects': 4, 'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (LookupLayer,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'num_objects': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (LookupLinear,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'num_objects': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_vsitzmann_scene_representation_networks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

