import sys
_module = sys.modules[__name__]
del sys
msra_hand = _module
gen_gt = _module
main = _module
show_acc = _module
accuracy = _module
compare_acc = _module
loss = _module
main = _module
model = _module
plot = _module
progressbar = _module
sampler = _module
solver = _module
v2v_model = _module
v2v_util = _module
v2v_model = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
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


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


class SoftmaxCrossEntropyWithLogits(nn.Module):
    """
    Similar to tensorflow's tf.nn.softmax_cross_entropy_with_logits
    ref: https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f

    The 'input' is unnormalized scores.
    The 'target' is a probability distribution.

    Shape:
        Input: (N, C), batch size N, with C classes
        Target: (N, C), batch size N, with C classes
    """

    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()

    def forward(self, input, target):
        loss = torch.sum(-target * F.log_softmax(input, -1), -1)
        mean_loss = torch.mean(loss)
        return mean_loss


class MixedLoss(nn.Module):
    """
    ref: https://github.com/mks0601/PoseFix_RELEASE/blob/master/main/model.py

    input: {
        'heatmap': (N, C, X, Y, Z), unnormalized
        'coord': (N, C, 3)
    }

    target: {
        'heatmap': (N, C, X, Y, Z), normalized
        'coord': (N, C, 3)
    }

    """

    def __init__(self, heatmap_weight=0.5):
        super(MixedLoss, self).__init__()
        self.w1 = heatmap_weight
        self.w2 = 1 - self.w1
        self.cross_entropy_loss = SoftmaxCrossEntropyWithLogits()

    def forward(self, input, target):
        pred_heatmap, pred_coord = input['heatmap'], input['coord']
        gt_heatmap, gt_coord = target['heatmap'], target['coord']
        N, C = pred_heatmap.shape[0:2]
        pred_heatmap = pred_heatmap.view(N * C, -1)
        gt_heatmap = gt_heatmap.view(N * C, -1)
        hm_loss = self.cross_entropy_loss(pred_heatmap, gt_heatmap)
        l1_loss = torch.mean(torch.abs(pred_coord - gt_coord))
        return self.w1 * hm_loss + self.w2 * l1_loss


class VolumetricSoftmax(nn.Module):
    """
    TODO: soft-argmax: norm coord to [-1, 1], instead of [0, N]

    ref: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
    """

    def __init__(self, channel, sizes):
        super(VolumetricSoftmax, self).__init__()
        self.channel = channel
        self.xsize, self.ysize, self.zsize = sizes[0], sizes[1], sizes[2]
        self.volume_size = self.xsize * self.ysize * self.zsize
        pos_x, pos_y, pos_z = np.meshgrid(np.arange(self.xsize), np.arange(self.ysize), np.arange(self.zsize), indexing='ij')
        pos_x = torch.from_numpy(pos_x.reshape(-1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(-1)).float()
        pos_z = torch.from_numpy(pos_z.reshape(-1)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)

    def forward(self, x):
        x = x.view(-1, self.volume_size)
        p = F.softmax(x, dim=1)
        expected_x = torch.sum(self.pos_x * p, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * p, dim=1, keepdim=True)
        expected_z = torch.sum(self.pos_z * p, dim=1, keepdim=True)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        out = expected_xyz.view(-1, self.channel, 3)
        return out


class Model(nn.Module):

    def __init__(self, in_channels, out_channels, output_res=44):
        super(Model, self).__init__()
        self.output_res = output_res
        self.basic_model = V2VModel(in_channels, out_channels)
        self.spatial_softmax = VolumetricSoftmax(out_channels, (self.output_res, self.output_res, self.output_res))

    def forward(self, x):
        heatmap = self.basic_model(x)
        coord = self.spatial_softmax(heatmap)
        output = {'heatmap': heatmap, 'coord': coord}
        return output


class Basic3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(out_planes), nn.ReLU(True), nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(out_planes))
        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(out_planes))

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):

    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert kernel_size == 2
        assert stride == 2
        self.block = nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):

    def __init__(self):
        super(EncoderDecorder, self).__init__()
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.mid_res = Res3DBlock(128, 128)
        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)
        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        x = self.mid_res(x)
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x


class V2VModel(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(V2VModel, self).__init__()
        self.front_layers = nn.Sequential(Basic3DBlock(input_channels, 16, 7), Pool3DBlock(2), Res3DBlock(16, 32), Res3DBlock(32, 32), Res3DBlock(32, 32))
        self.encoder_decoder = EncoderDecorder()
        self.back_layers = nn.Sequential(Res3DBlock(32, 32), Basic3DBlock(32, 32, 1), Basic3DBlock(32, 32, 1))
        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)
        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class Basic3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(out_planes), nn.ReLU(True), nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(out_planes))
        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(out_planes))

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):

    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert kernel_size == 2
        assert stride == 2
        self.block = nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):

    def __init__(self):
        super(EncoderDecorder, self).__init__()
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.mid_res = Res3DBlock(128, 128)
        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)
        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        x = self.mid_res(x)
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x


class V2VModel(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(V2VModel, self).__init__()
        self.front_layers = nn.Sequential(Basic3DBlock(input_channels, 16, 7), Pool3DBlock(2), Res3DBlock(16, 32), Res3DBlock(32, 32), Res3DBlock(32, 32))
        self.encoder_decoder = EncoderDecorder()
        self.back_layers = nn.Sequential(Res3DBlock(32, 32), Basic3DBlock(32, 32, 1), Basic3DBlock(32, 32, 1))
        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)
        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Basic3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (Pool3DBlock,
     lambda: ([], {'pool_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Res3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (SoftmaxCrossEntropyWithLogits,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 2, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (VolumetricSoftmax,
     lambda: ([], {'channel': 4, 'sizes': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dragonbook_V2V_PoseNet_pytorch(_paritybench_base):
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

