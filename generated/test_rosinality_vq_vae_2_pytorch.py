import sys
_module = sys.modules[__name__]
del sys
dataset = _module
extract_code = _module
pixelsnail = _module
pixelsnail_mnist = _module
sample = _module
scheduler = _module
train_pixelsnail = _module
train_vqvae = _module
vqvae = _module

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


from math import sqrt


from functools import partial


from functools import lru_cache


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


class WNConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding=0, bias=True, activation=None):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel,
            kernel_size, stride=stride, padding=padding, bias=bias))
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.activation = activation

    def forward(self, input):
        out = self.conv(input)
        if self.activation is not None:
            out = self.activation(out)
        return out


class CausalConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding='downright', activation=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2
        self.kernel_size = kernel_size
        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2
            pad = [pad, pad, kernel_size[0] - 1, 0]
        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2
        self.pad = nn.ZeroPad2d(pad)
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, stride=
            stride, padding=0, activation=activation)

    def forward(self, input):
        out = self.pad(input)
        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, (-1), self.causal:].zero_()
        out = self.conv(out)
        return out


class GatedResBlock(nn.Module):

    def __init__(self, in_channel, channel, kernel_size, conv='wnconv2d',
        activation=nn.ELU, dropout=0.1, auxiliary_channel=0, condition_dim=0):
        super().__init__()
        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)
        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')
        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')
        self.activation = activation(inplace=True)
        self.conv1 = conv_module(in_channel, channel, kernel_size)
        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)
        if condition_dim > 0:
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1,
                bias=False)
        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))
        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if condition is not None:
            condition = self.condition(condition)
            out += condition
        out = self.gate(out)
        out += input
        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0
    return torch.from_numpy(mask).unsqueeze(0), torch.from_numpy(start_mask
        ).unsqueeze(1)


def wn_linear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))


class CausalAttention(nn.Module):

    def __init__(self, query_channel, key_channel, channel, n_head=8,
        dropout=0.1):
        super().__init__()
        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)
        self.dim_head = channel // n_head
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(
                1, 2)
        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))
        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        attn = attn.masked_fill(mask == 0, -10000.0)
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)
        out = attn @ value
        out = out.transpose(1, 2).reshape(batch, height, width, self.
            dim_head * self.n_head)
        out = out.permute(0, 3, 1, 2)
        return out


class PixelBlock(nn.Module):

    def __init__(self, in_channel, channel, kernel_size, n_res_block,
        attention=True, dropout=0.1, condition_dim=0):
        super().__init__()
        resblocks = []
        for i in range(n_res_block):
            resblocks.append(GatedResBlock(in_channel, channel, kernel_size,
                conv='causal', dropout=dropout, condition_dim=condition_dim))
        self.resblocks = nn.ModuleList(resblocks)
        self.attention = attention
        if attention:
            self.key_resblock = GatedResBlock(in_channel * 2 + 2,
                in_channel, 1, dropout=dropout)
            self.query_resblock = GatedResBlock(in_channel + 2, in_channel,
                1, dropout=dropout)
            self.causal_attention = CausalAttention(in_channel + 2, 
                in_channel * 2 + 2, in_channel // 2, dropout=dropout)
            self.out_resblock = GatedResBlock(in_channel, in_channel, 1,
                auxiliary_channel=in_channel // 2, dropout=dropout)
        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input
        for resblock in self.resblocks:
            out = resblock(out, condition=condition)
        if self.attention:
            key_cat = torch.cat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)
        else:
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)
        return out


class CondResNet(nn.Module):

    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()
        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=
            kernel_size // 2)]
        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, :input.shape[2], :]


def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, :input.shape[3]]


class PixelSNAIL(nn.Module):

    def __init__(self, shape, n_class, channel, kernel_size, n_block,
        n_res_block, res_channel, attention=True, dropout=0.1,
        n_cond_res_block=0, cond_res_channel=0, cond_res_kernel=3,
        n_out_res_block=0):
        super().__init__()
        height, width = shape
        self.n_class = n_class
        if kernel_size % 2 == 0:
            kernel = kernel_size + 1
        else:
            kernel = kernel_size
        self.horizontal = CausalConv2d(n_class, channel, [kernel // 2,
            kernel], padding='down')
        self.vertical = CausalConv2d(n_class, channel, [(kernel + 1) // 2, 
            kernel // 2], padding='downright')
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(PixelBlock(channel, res_channel, kernel_size,
                n_res_block, attention=attention, dropout=dropout,
                condition_dim=cond_res_channel))
        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(n_class, cond_res_channel,
                cond_res_kernel, n_cond_res_block)
        out = []
        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))
        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])
        self.out = nn.Sequential(*out)

    def forward(self, input, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        input = F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self
            .background)
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical
        background = self.background[:, :, :height, :].expand(batch, 2,
            height, width)
        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]
            else:
                condition = F.one_hot(condition, self.n_class).permute(0, 3,
                    1, 2).type_as(self.background)
                condition = self.cond_resnet(condition)
                condition = F.interpolate(condition, scale_factor=2)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]
        for block in self.blocks:
            out = block(out, background, condition=condition)
        out = self.out(out)
        return out, cache


class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True
            ) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim
            =True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(1 - self.decay,
                embed_onehot.sum(0))
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum
                )
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.
                n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):

    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(
            in_channel, channel, 3, padding=1), nn.ReLU(inplace=True), nn.
            Conv2d(channel, in_channel, 1))

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride
        ):
        super().__init__()
        if stride == 4:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2,
                padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2,
                channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn
                .Conv2d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2,
                padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel // 2,
                channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):

    def __init__(self, in_channel, out_channel, channel, n_res_block,
        n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([nn.ConvTranspose2d(channel, channel // 2, 4,
                stride=2, padding=1), nn.ReLU(inplace=True), nn.
                ConvTranspose2d(channel // 2, out_channel, 4, stride=2,
                padding=1)])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4,
                stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):

    def __init__(self, in_channel=3, channel=128, n_res_block=2,
        n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block,
            n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel,
            stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block,
            n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4,
            stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel,
            n_res_block, n_res_channel, stride=4)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_t, quant_b)
        return dec


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rosinality_vq_vae_2_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(CausalConv2d(*[], **{'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Decoder(*[], **{'in_channel': 4, 'out_channel': 4, 'channel': 4, 'n_res_block': 1, 'n_res_channel': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Quantize(*[], **{'dim': 4, 'n_embed': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ResBlock(*[], **{'in_channel': 4, 'channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(VQVAE(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_005(self):
        self._check(WNConv2d(*[], **{'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

