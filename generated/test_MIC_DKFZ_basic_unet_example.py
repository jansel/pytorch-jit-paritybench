import sys
_module = sys.modules[__name__]
del sys
master = _module
Config_unet = _module
Config_unet_spleen = _module
configs = _module
datasets = _module
data_loader = _module
example_dataset = _module
create_splits = _module
preprocessing = _module
spleen = _module
preprocessing = _module
NumpyDataLoader = _module
three_dim = _module
data_augmentation = _module
two_dim = _module
utils = _module
evaluation = _module
evaluator = _module
metrics = _module
UNetExperiment = _module
UNetExperiment3D = _module
experiments = _module
ND_Crossentropy = _module
loss_functions = _module
dice_loss = _module
topk_loss = _module
RecursiveUNet = _module
RecursiveUNet3D = _module
UNET = _module
run_preprocessing = _module
run_train_pipeline = _module
runner = _module
segment_a_spleen = _module
train = _module
train3D = _module
utilities = _module
file_and_folder_operations = _module

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


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from collections import defaultdict


import numpy as np


import torch


import torch.optim as optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as F


from collections import OrderedDict


from torch import nn


import torch.nn as nn


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]
        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1
        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        target = target.view(-1)
        return super(CrossentropyND, self).forward(inp, target)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def soft_dice(net_output, gt, smooth=1.0, smooth_in_nom=1.0):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (-((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result


def soft_dice_per_batch_2(net_output, gt, smooth=1.0, smooth_in_nom=1.0, background_weight=1, rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:]
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(tp.shape)
    weights[0] = background_weight
    if net_output.device.type == 'cuda':
        weights = weights
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == 'cuda':
            rebalance_weights = rebalance_weights
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights
    result = (-((2 * tp + smooth_in_nom) / (2 * tp + fp + fn + smooth)) * weights).mean()
    return result


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1.0, apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True, background_weight=1, rebalance_weights=None):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        if not do_bg:
            assert background_weight == 1, 'if there is no bg, then set background weight to 1 you dummy'
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        if smooth_in_nom:
            self.smooth_in_nom = smooth
        else:
            self.smooth_in_nom = 0
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        shp_y = y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        y_onehot = torch.zeros(shp_x)
        if x.device.type == 'cuda':
            y_onehot = y_onehot
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or self.rebalance_weights is not None:
                raise NotImplementedError('nah son')
            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        else:
            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom, background_weight=self.background_weight, rebalance_weights=self.rebalance_weights)
        return l


class MultipleOutputLoss(nn.Module):

    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs that should predict the same y
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), 'x must be either tuple or list'
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        l = weights[0] * self.loss(x[0], y)
        for i in range(1, len(x)):
            l += weights[i] * self.loss(x[i], y)
        return l


def softmax_helper(x):
    rpt = [(1) for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class DC_and_CE_loss(nn.Module):

    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate='sum'):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == 'sum':
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError('nah son')
        return result


class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """

    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1,)), int(num_voxels // self.k), sorted=False)
        return res.mean()


class DC_and_topk_loss(nn.Module):

    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate='sum'):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == 'sum':
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError('nah son')
        return result


class CrossentropyWithLossMask(nn.CrossEntropyLoss):

    def __init__(self, k=None):
        """
        This implementation ignores weight, ignore_index (use loss mask!) and reduction!
        :param k:
        """
        super(CrossentropyWithLossMask, self).__init__(weight=None, ignore_index=-100, reduction='none')
        self.k = k

    def forward(self, inp, target, loss_mask=None):
        target = target.long()
        inp = inp.float()
        if loss_mask is not None:
            loss_mask = loss_mask.float()
        num_classes = inp.size()[1]
        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1
        if not inp.is_contiguous():
            inp = inp.contiguous()
        inp = inp.view(target.shape[0], -1, num_classes)
        target = target.view(target.shape[0], -1)
        if loss_mask is not None:
            loss_mask = loss_mask.view(target.shape[0], -1)
        if self.k is not None:
            if loss_mask is not None:
                num_sel = torch.stack(tuple([(i.sum() / self.k) for i in torch.unbind(loss_mask, 0)]), 0).long()
                loss = torch.stack(tuple([torch.topk(super(CrossentropyWithLossMask, self).forward(inp[i], target[i])[loss_mask[i].byte()], num_sel[i], sorted=False)[0].mean() for i in range(target.shape[0])]))
            else:
                num_sel = [np.prod(inp.shape[2:]) / self.k] * inp.shape[0]
                loss = torch.stack(tuple([torch.topk(super(CrossentropyWithLossMask, self).forward(inp[i], target[i]), num_sel[i], sorted=False)[0].mean() for i in range(target.shape[0])]))
        elif loss_mask is not None:
            loss = torch.stack(tuple([super(CrossentropyWithLossMask, self).forward(inp[i], target[i])[loss_mask[i].byte()].mean() for i in range(target.shape[0])]))
        else:
            loss = torch.stack(tuple([super(CrossentropyWithLossMask, self).forward(inp[i], target[i]).mean() for i in range(target.shape[0])]))
        loss = loss.mean()
        return loss


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()
        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size, instancenorm=do_instancenorm)
        self.center = nn.Sequential(nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, 2, stride=2), nn.ReLU(inplace=True))
        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2, stride=2)
        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)
        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2)
        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)
        self.softmax = torch.nn.Softmax2d()
        self.output_reconstruction_map = nn.Conv2d(initial_filter_size, out_channels=1, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:xy1 + target_width, xy2:xy2 + target_height]

    def forward(self, x, enable_concat=True, print_layer_shapes=False):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)
        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)
        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)
        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)
        center = self.center(pool)
        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop * concat_weight], 1)
        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)
        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)
        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)
        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))
        if enable_concat:
            output = self.final(expand)
        if not enable_concat:
            output = self.output_reconstruction_map(expand)
        return output


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        pool = nn.MaxPool3d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv3 = self.expand(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        if outermost:
            final = nn.Conv3d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=2, stride=2)
            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm3d):
        layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=1), norm_layer(out_channels), nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=1), nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def center_crop(layer, target_depth, target_width, target_height):
        batch_size, n_channels, layer_depth, layer_width, layer_height = layer.size()
        xy0 = (layer_depth - target_depth) // 2
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy0:xy0 + target_depth, xy1:xy1 + target_width, xy2:xy2 + target_height]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3], x.size()[4])
            return torch.cat([x, crop], 1)


class UNet3D(nn.Module):

    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=3, norm_layer=nn.InstanceNorm3d):
        super(UNet3D, self).__init__()
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs - 1), out_channels=initial_filter_size * 2 ** num_downs, num_classes=num_classes, kernel_size=kernel_size, norm_layer=norm_layer, innermost=True)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs - (i + 1)), out_channels=initial_filter_size * 2 ** (num_downs - i), num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size, num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer, outermost=True)
        self.model = unet_block

    def forward(self, x):
        return self.model(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MultipleOutputLoss,
     lambda: ([], {'loss': MSELoss()}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TopKLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (UNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
]

class Test_MIC_DKFZ_basic_unet_example(_paritybench_base):
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

