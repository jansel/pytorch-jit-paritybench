import sys
_module = sys.modules[__name__]
del sys
maxvit = _module
maxvit = _module
setup = _module

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


from typing import Type


from typing import Callable


from typing import Tuple


from typing import Optional


from typing import Set


from typing import List


from typing import Union


import torch


import torch.nn as nn


def _gelu_ignore_parameters(*args, **kwargs) ->nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConv(nn.Module):
    """ MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(self, in_channels: int, out_channels: int, downscale: bool=False, act_layer: Type[nn.Module]=nn.GELU, norm_layer: Type[nn.Module]=nn.BatchNorm2d, drop_path: float=0.0) ->None:
        """ Constructor method """
        super(MBConv, self).__init__()
        self.drop_path_rate: float = drop_path
        if not downscale:
            assert in_channels == out_channels, 'If downscaling is utilized input and output channels must be equal.'
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        self.main_path = nn.Sequential(norm_layer(in_channels), DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1, act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path), SqueezeExcite(in_chs=out_channels, rd_ratio=0.25), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1)))
        self.skip_path = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.0:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


def get_relative_position_index(win_h: int, win_w: int) ->torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.

    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, in_channels: int, num_heads: int=32, grid_window_size: Tuple[int, int]=(7, 7), attn_drop: float=0.0, drop: float=0.0) ->None:
        """ Constructor method """
        super(RelativeSelfAttention, self).__init__()
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))
        self.register_buffer('relative_position_index', get_relative_position_index(grid_window_size[0], grid_window_size[1]))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _get_relative_positional_bias(self) ->torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        B_, N, C = input.shape
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    """ MaxViT Transformer block.

        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))

        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(self, in_channels: int, partition_function: Callable, reverse_function: Callable, num_heads: int=32, grid_window_size: Tuple[int, int]=(7, 7), attn_drop: float=0.0, drop: float=0.0, drop_path: float=0.0, mlp_ratio: float=4.0, act_layer: Type[nn.Module]=nn.GELU, norm_layer: Type[nn.Module]=nn.LayerNorm) ->None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(in_channels=in_channels, num_heads=num_heads, grid_window_size=grid_window_size, attn_drop=attn_drop, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels), act_layer=act_layer, drop=drop)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        B, C, H, W = input.shape
        input_partitioned = self.partition_function(input, self.grid_window_size)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output


def grid_partition(input: torch.Tensor, grid_size: Tuple[int, int]=(7, 7)) ->torch.Tensor:
    """ Grid partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    B, C, H, W = input.shape
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(grid: torch.Tensor, original_size: Tuple[int, int], grid_size: Tuple[int, int]=(7, 7)) ->torch.Tensor:
    """ Reverses the grid partition.

    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    (H, W), C = original_size, grid.shape[-1]
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


def window_partition(input: torch.Tensor, window_size: Tuple[int, int]=(7, 7)) ->torch.Tensor:
    """ Window partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)

    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    B, C, H, W = input.shape
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows: torch.Tensor, original_size: Tuple[int, int], window_size: Tuple[int, int]=(7, 7)) ->torch.Tensor:
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    H, W = original_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


class MaxViTBlock(nn.Module):
    """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true spatial downscaling is performed. Default: False
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(self, in_channels: int, out_channels: int, downscale: bool=False, num_heads: int=32, grid_window_size: Tuple[int, int]=(7, 7), attn_drop: float=0.0, drop: float=0.0, drop_path: float=0.0, mlp_ratio: float=4.0, act_layer: Type[nn.Module]=nn.GELU, norm_layer: Type[nn.Module]=nn.BatchNorm2d, norm_layer_transformer: Type[nn.Module]=nn.LayerNorm) ->None:
        """ Constructor method """
        super(MaxViTBlock, self).__init__()
        self.mb_conv = MBConv(in_channels=in_channels, out_channels=out_channels, downscale=downscale, act_layer=act_layer, norm_layer=norm_layer, drop_path=drop_path)
        self.block_transformer = MaxViTTransformerBlock(in_channels=out_channels, partition_function=window_partition, reverse_function=window_reverse, num_heads=num_heads, grid_window_size=grid_window_size, attn_drop=attn_drop, drop=drop, drop_path=drop_path, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer_transformer)
        self.grid_transformer = MaxViTTransformerBlock(in_channels=out_channels, partition_function=grid_partition, reverse_function=grid_reverse, num_heads=num_heads, grid_window_size=grid_window_size, attn_drop=attn_drop, drop=drop, drop_path=drop_path, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer_transformer)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
        """
        output = self.grid_transformer(self.block_transformer(self.mb_conv(input)))
        return output


class MaxViTStage(nn.Module):
    """ Stage of the MaxViT.

    Args:
        depth (int): Depth of the stage.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
    """

    def __init__(self, depth: int, in_channels: int, out_channels: int, num_heads: int=32, grid_window_size: Tuple[int, int]=(7, 7), attn_drop: float=0.0, drop: float=0.0, drop_path: Union[List[float], float]=0.0, mlp_ratio: float=4.0, act_layer: Type[nn.Module]=nn.GELU, norm_layer: Type[nn.Module]=nn.BatchNorm2d, norm_layer_transformer: Type[nn.Module]=nn.LayerNorm) ->None:
        """ Constructor method """
        super(MaxViTStage, self).__init__()
        self.blocks = nn.Sequential(*[MaxViTBlock(in_channels=in_channels if index == 0 else out_channels, out_channels=out_channels, downscale=index == 0, num_heads=num_heads, grid_window_size=grid_window_size, attn_drop=attn_drop, drop=drop, drop_path=drop_path if isinstance(drop_path, float) else drop_path[index], mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, norm_layer_transformer=norm_layer_transformer) for index in range(depth)])

    def forward(self, input=torch.Tensor) ->torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2].
        """
        output = self.blocks(input)
        return output


class MaxViT(nn.Module):
    """ Implementation of the MaxViT proposed in:
        https://arxiv.org/pdf/2204.01697.pdf

    Args:
        in_channels (int, optional): Number of input channels to the convolutional stem. Default 3
        depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 2, 5, 2)
        channels (Tuple[int, ...], optional): Number of channels in each network stage. Default (64, 128, 256, 512)
        num_classes (int, optional): Number of classes to be predicted. Default 1000
        embed_dim (int, optional): Embedding dimension of the convolutional stem. Default 64
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        global_pool (str, optional): Global polling type to be utilized. Default "avg"
    """

    def __init__(self, in_channels: int=3, depths: Tuple[int, ...]=(2, 2, 5, 2), channels: Tuple[int, ...]=(64, 128, 256, 512), num_classes: int=1000, embed_dim: int=64, num_heads: int=32, grid_window_size: Tuple[int, int]=(7, 7), attn_drop: float=0.0, drop=0.0, drop_path=0.0, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, norm_layer_transformer=nn.LayerNorm, global_pool: str='avg') ->None:
        """ Constructor method """
        super(MaxViT, self).__init__()
        assert len(depths) == len(channels), 'For each stage a channel dimension must be given.'
        assert global_pool in ['avg', 'max'], f'Only avg and max is supported but {global_pool} is given'
        self.num_classes: int = num_classes
        self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), act_layer(), nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), act_layer())
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        self.stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages.append(MaxViTStage(depth=depth, in_channels=embed_dim if index == 0 else channels[index - 1], out_channels=channel, num_heads=num_heads, grid_window_size=grid_window_size, attn_drop=attn_drop, drop=drop, drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])], mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, norm_layer_transformer=norm_layer_transformer))
        self.global_pool: str = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    @torch.jit.ignore
    def no_weight_decay(self) ->Set[str]:
        """ Gets the names of parameters to not apply weight decay to.

        Returns:
            nwd (Set[str]): Set of parameter names to not apply weight decay to.
        """
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    def reset_classifier(self, num_classes: int, global_pool: Optional[str]=None) ->None:
        """Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str, optional): If not global pooling is updated
        """
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, input: torch.Tensor) ->torch.Tensor:
        """ Forward pass of feature extraction.

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Image features of the backbone.
        """
        output = input
        for stage in self.stages:
            output = stage(output)
        return output

    def forward_head(self, input: torch.Tensor, pre_logits: bool=False):
        """ Forward pass of classification head.

        Args:
            input (torch.Tensor): Input features
            pre_logits (bool, optional): If true pre-logits are returned

        Returns:
            output (torch.Tensor): Classification output of the shape [B, num_classes].
        """
        if self.global_pool == 'avg':
            input = input.mean(dim=(2, 3))
        elif self.global_pool == 'max':
            input = torch.amax(input, dim=(2, 3))
        return input if pre_logits else self.head(input)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """ Forward pass

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Classification output of the shape [B, num_classes].
        """
        output = self.forward_features(self.stem(input))
        output = self.forward_head(output)
        return output

