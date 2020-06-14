import sys
_module = sys.modules[__name__]
del sys
attention_augmented_conv = _module
attention_augmented_wide_resnet = _module
main = _module
preprocess = _module
attention_augmented_conv = _module
attention_augmented_conv = _module

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


import torch.optim as optim


import time


class AugmentedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh,
        shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        assert self.Nh != 0, 'integer division or modulo by zero, Nh >= 1'
        assert self.dk % self.Nh == 0, 'dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)'
        assert self.dv % self.Nh == 0, 'dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)'
        assert stride in [1, 2], str(stride) + ' Up to 2 strides are allowed.'
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels -
            self.dv, self.kernel_size, stride=stride, padding=self.padding)
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv,
            kernel_size=self.kernel_size, stride=stride, padding=self.padding)
        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, 
                dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, 
                dk // Nh), requires_grad=True))

    def forward(self, x):
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk,
            self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self
            .Nh, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)
        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = batch, Nh, channels // Nh, height, width
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = batch, Nh * dv, H, W
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, 'w'
            )
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3),
            self.key_rel_h, W, H, Nh, 'h')
        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))
        if case == 'w':
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == 'h':
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5
                ).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, shape, stride=1, v=
        0.2, k=2, Nh=4):
        super(wide_basic, self).__init__()
        if stride == 2:
            original_shape = shape * 2
        else:
            original_shape = shape
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = AugmentedConv(in_planes, planes, kernel_size=3, dk=k *
            planes, dv=int(v * planes), Nh=Nh, relative=True, shape=
            original_shape)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = AugmentedConv(planes, planes, kernel_size=3, dk=k *
            planes, dv=int(v * planes), Nh=Nh, stride=stride, relative=True,
            shape=shape)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(AugmentedConv(in_planes, planes,
                kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh,
                relative=True, stride=stride, shape=shape))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        short = self.shortcut(x)
        out += short
        return out


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class Wide_ResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, shape):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 20
        self.shape = shape
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        dv_v = 0.2
        dk_k = 2
        Nh = 4
        None
        n_Stages = [20, 20 * k, 40 * k, 60 * k]
        self.conv1 = AugmentedConv(in_channels=3, out_channels=n_Stages[0],
            kernel_size=3, dk=dk_k * n_Stages[0], dv=int(dv_v * n_Stages[0]
            ), shape=shape, Nh=Nh, relative=True)
        self.layer1 = nn.Sequential(self._wide_layer(wide_basic, n_Stages[1
            ], n, dropout_rate, stride=1, shape=shape))
        self.layer2 = nn.Sequential(self._wide_layer(wide_basic, n_Stages[2
            ], n, dropout_rate, stride=2, shape=shape // 2))
        self.layer3 = nn.Sequential(self._wide_layer(wide_basic, n_Stages[3
            ], n, dropout_rate, stride=2, shape=shape // 4))
        self.bn1 = nn.BatchNorm2d(n_Stages[3], momentum=0.9)
        self.linear = nn.Linear(n_Stages[3], num_classes)
        self.apply(_weights_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride,
        shape):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate=
                dropout_rate, stride=stride, shape=shape))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class AugmentedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh,
        shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        assert self.Nh != 0, 'integer division or modulo by zero, Nh >= 1'
        assert self.dk % self.Nh == 0, 'dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)'
        assert self.dv % self.Nh == 0, 'dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)'
        assert stride in [1, 2], str(stride) + ' Up to 2 strides are allowed.'
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels -
            self.dv, self.kernel_size, stride=stride, padding=self.padding)
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv,
            kernel_size=self.kernel_size, stride=stride, padding=self.padding)
        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, 
                dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, 
                dk // Nh), requires_grad=True))

    def forward(self, x):
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk,
            self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self
            .Nh, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)
        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = batch, Nh, channels // Nh, height, width
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = batch, Nh * dv, H, W
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, 'w'
            )
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3),
            self.key_rel_h, W, H, Nh, 'h')
        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))
        if case == 'w':
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == 'h':
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5
                ).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


use_cuda = torch.cuda.is_available()


device = torch.device('cuda' if use_cuda else 'cpu')


class AugmentedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh,
        relative):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels -
            self.dv, self.kernel_size, padding=1)
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv,
            kernel_size=1)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)

    def forward(self, x):
        batch, _, height, width = x.size()
        conv_out = self.conv_out(x)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk,
            self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self
            .Nh, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)
        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = batch, Nh, channels // Nh, height, width
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = batch, Nh * dv, H, W
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        key_rel_w = nn.Parameter(torch.randn((2 * W - 1, dk), requires_grad
            =True)).to(device)
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, 'w')
        key_rel_h = nn.Parameter(torch.randn((2 * H - 1, dk), requires_grad
            =True)).to(device)
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3),
            key_rel_h, W, H, Nh, 'h')
        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))
        if case == 'w':
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == 'h':
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5
                ).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_leaderj1001_Attention_Augmented_Conv2d(_paritybench_base):
    pass
