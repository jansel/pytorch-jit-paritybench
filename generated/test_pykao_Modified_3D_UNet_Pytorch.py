import sys
_module = sys.modules[__name__]
del sys
N4ITKforBRATS2017 = _module
braintools = _module
data_loader = _module
dataset = _module
loss = _module
main = _module
model = _module
paths = _module
run_preporcessing = _module
test = _module

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


import time


import logging


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.optim


from torch.utils.data import DataLoader


class Modified3DUNet(nn.Module):

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.
            base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter *
            2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter *
            2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.
            base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter *
            4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.
            base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter *
            8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.
            base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter *
            16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = (self.
            norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16, 
            self.base_n_filter * 8))
        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.
            base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter *
            16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.
            base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = (self.
            norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8, self
            .base_n_filter * 4))
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter *
            8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.
            base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = (self.
            norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4, self
            .base_n_filter * 2))
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter *
            4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.
            base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = (self.
            norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2, self
            .base_n_filter))
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter *
            2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.
            n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.
            n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.Conv3d(feat_in, feat_out, kernel_size=3,
            stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn
            .Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1,
            bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out,
            kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn
            .Upsample(scale_factor=2, mode='nearest'), nn.Conv3d(feat_in,
            feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.
            InstanceNorm3d(feat_out), nn.LeakyReLU())

    def forward(self, x):
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(
            ds1_ds2_sum_upscale_ds3_sum)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
        out = self.softmax(out)
        return out, seg_layer


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pykao_Modified_3D_UNet_Pytorch(_paritybench_base):
    pass
