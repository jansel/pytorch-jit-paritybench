import sys
_module = sys.modules[__name__]
del sys
examples = _module
albef = _module
data = _module
retrieval_datamodule = _module
retrieval_dataset = _module
transforms = _module
vqa_datamodules = _module
vqa_dataset = _module
finetune_retrieval = _module
finetune_vqa = _module
model = _module
utils = _module
cnn_encoder = _module
cnn_lstm = _module
lstm_encoder = _module
test_cnn_encoder = _module
test_cnn_lstm = _module
test_lstm_encoder = _module
common = _module
multidata = _module
flava = _module
callbacks = _module
multimodal_eval = _module
coco_zero_shot = _module
datamodules = _module
imagenet_zeroshot_data = _module
transforms = _module
definitions = _module
finetune = _module
model = _module
native = _module
data = _module
model = _module
train = _module
utils = _module
convert_weights = _module
datamodule = _module
dataset = _module
flickr_eval = _module
postprocessors = _module
transforms = _module
loss = _module
matcher = _module
optimizer = _module
phrase_grounding = _module
test_loss = _module
test_matcher = _module
test_postprocessors = _module
args_parse = _module
dist = _module
metrics = _module
misc = _module
vqa_eval = _module
vqa_finetune = _module
audio_utils = _module
construct_from_json = _module
game = _module
generate_text_desc = _module
mugen_datamodules = _module
mugen_dataset = _module
video_utils = _module
text_video_gpt = _module
video_vqvae = _module
eval = _module
model = _module
video_clip = _module
test_text_video_gpt = _module
test_video_vqvae = _module
test_video_clip = _module
omnivore = _module
data_builder = _module
datasets = _module
presets = _module
rand_aug3d = _module
transforms = _module
train = _module
utils = _module
common = _module
setup = _module
tests = _module
models = _module
test_albef = _module
test_image_encoder = _module
test_multimodal_encoder = _module
clip = _module
test_clip = _module
test_image_encoder = _module
test_text_encoder = _module
test_checkpoint = _module
test_flava = _module
test_image_encoder = _module
test_text_encoder = _module
mdetr = _module
test_image_encoder = _module
test_mdetr = _module
test_text_encoder = _module
test_transformer = _module
test_gpt = _module
test_late_fusion = _module
test_omnivore = _module
test_two_tower = _module
test_video_gpt = _module
test_video_vqvae = _module
test_vqvae = _module
modules = _module
encoders = _module
test_bert_text_encoder = _module
test_embedding_encoder = _module
test_mil_encoder = _module
test_swin_transformer_3d_encoder = _module
test_weighted_embedding_encoder = _module
fusions = _module
test_attention_fusion = _module
test_deepset_fusion = _module
layers = _module
test_activation = _module
test_attention = _module
test_codebook = _module
test_conv = _module
test_mlp = _module
test_normalizations = _module
test_position_embedding = _module
test_text_embedding = _module
test_transformer = _module
losses = _module
test_albef = _module
test_commitment = _module
test_contrastive_loss_with_temperature = _module
test_mdetr_losses = _module
test_utils = _module
test_bert_text_transform = _module
test_clip_transform = _module
test_flava_transform = _module
test_video_transform = _module
test_assertion = _module
test_attention_utils = _module
test_ckpt_load = _module
test_common = _module
test_distributed = _module
test_generate = _module
torchmultimodal = _module
image_encoder = _module
model = _module
multimodal_encoder = _module
image_encoder = _module
model = _module
text_encoder = _module
image_encoder = _module
model = _module
text_encoder = _module
transformer = _module
gpt = _module
late_fusion = _module
image_encoder = _module
model = _module
text_encoder = _module
transformer = _module
omnivore = _module
two_tower = _module
video_gpt = _module
video_vqvae = _module
vqvae = _module
bert_text_encoder = _module
embedding_encoder = _module
mil_encoder = _module
swin_transformer_3d_encoder = _module
weighted_embedding_encoder = _module
attention_fusion = _module
concat_fusion = _module
deepset_fusion = _module
activation = _module
attention = _module
codebook = _module
conv = _module
mlp = _module
normalizations = _module
position_embedding = _module
text_embedding = _module
transformer = _module
albef = _module
contrastive_loss_with_temperature = _module
flava = _module
mdetr = _module
vqvae = _module
bert_text_transform = _module
clip_transform = _module
flava_transform = _module
video_transform = _module
assertion = _module
attention = _module
common = _module
distributed = _module
file_io = _module
generate = _module
version = _module

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


from typing import List


from typing import Optional


from typing import Tuple


import torch


from torch import Tensor


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import DistributedSampler


from typing import Callable


from typing import Union


import re


from torchtext.transforms import PadTransform


from torchtext.transforms import Sequential


from torchtext.transforms import ToTensor


from torchtext.transforms import Truncate


from torchvision import transforms


import random


import time


import torch.backends.cudnn as cudnn


import torch.distributed as dist


from torch.optim import AdamW


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import copy


from typing import Any


from typing import Dict


import torch.nn.functional as F


from torch import nn


import warnings


from functools import partial


import logging


from torchvision.datasets import CocoCaptions


import torchvision


from torch.utils.data.distributed import DistributedSampler


import numpy as np


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.elastic.multiprocessing.errors import record


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from torch import distributed as dist


from torchvision.datasets import CocoDetection


from typing import Sequence


from torchvision.ops.boxes import box_iou


from torchvision.ops.boxes import box_convert


from typing import Iterable


import torchvision.transforms.functional as F


from torchvision import transforms as T


from collections import OrderedDict


from scipy.optimize import linear_sum_assignment


from torchvision.ops.boxes import generalized_box_iou


from copy import deepcopy


import functools


from collections import defaultdict


from collections import deque


import math


import torch.utils.data as data


from torchtext.transforms import CharBPETokenizer


from torchvision.models.video import S3D


import torchvision.datasets.samplers as video_samplers


from torch.utils.data.dataloader import default_collate


from torchvision.transforms.functional import InterpolationMode


import scipy.io


import torchvision.transforms as T


from torchvision.datasets.vision import VisionDataset


from torchvision.transforms import autoaugment


from torchvision.transforms import functional as F


from torchvision.transforms import InterpolationMode


import torch.utils.data


from torchvision.models.vision_transformer import VisionTransformer


from math import inf


from torch.nn import functional as F


from torchvision.models.video.swin_transformer import PatchEmbed3d


from itertools import repeat


from torch import tensor


from itertools import product


from itertools import chain


import torch.multiprocessing as mp


import torch.nn as nn


import torch.optim as optim


from typing import NamedTuple


from torchvision.transforms import ToPILImage


from torch.utils.checkpoint import checkpoint


from torchvision.models.feature_extraction import create_feature_extractor


from collections import namedtuple


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import ResNet


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


from torchvision.models._api import Weights


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.models.resnet import resnet101


from torchvision.models.resnet import ResNet101_Weights


from typing import cast


from torch import Size


from torchvision.models.video.swin_transformer import PatchMerging


from torchvision.models.video.swin_transformer import SwinTransformer3d as TVSwinTransformer3d


from typing import Mapping


import itertools


from typing import OrderedDict


from torchtext.transforms import AddToken


from torchtext.transforms import BERTTokenizer


from torchtext.transforms import StrToIntTransform


from torchtext import transforms as text_transforms


from torchtext.transforms import CLIPTokenizer


from torchvision import transforms as image_transforms


from torchvision.transforms.functional import normalize


from torchvision.transforms.functional import resize


from torch.distributed import all_gather as all_gather_no_backprop


from torch.distributed.nn.functional import all_gather as all_gather_with_backprop


class PredictionHead(nn.Module):
    """
    Predict the following token autoregressively.

    Args:
        vocab_size (int): The number of different tokens the prediction_head can predict.
        hidden_size (int): The hidden size of the prediction_head.
        layer_norm_eps (float): The epsilon used by the prediction_head normalization layer.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function in the prediction_head.

    Inputs:
        hidden_states (Tensor): The hidden states of preceding tokens.

    Returns:
        Tensor: Prediction scores for the following token.
    """

    def __init__(self, vocab_size: int=30522, hidden_size: int=768, layer_norm_eps: float=1e-12, transform_act_fn: Callable[[Tensor], Tensor]=nn.functional.gelu) ->None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: Tensor) ->Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args: Any, **kwargs: Any) ->None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) ->Tensor:
        output = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(x)


class MLP(nn.Module):
    """A multi-layer perceptron module.

    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Optional[Union[int, List[int]]]=None, dropout: float=0.5, activation: Callable[..., nn.Module]=nn.ReLU, normalization: Optional[Callable[..., nn.Module]]=None) ->None:
        super().__init__()
        layers = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = []
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.model(x)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, attn_dropout: float=0.0) ->Tuple[Tensor, Tensor]:
    """Similar to PyTorch Core's _scaled_dot_product_attention but generalized
    to handle n-dimensional input tokens (images, video) and support multihead.
    Computes attention as described in Attention Is All You Need (Vaswani et al. 2017)

    Args:
        q (Tensor): Query of shape ``(b, h, d1, ..., dn, dim_qk)`` or ``(b, h, seq_len, dim_qk)`` where
            ``h`` is number of attention heads, ``d1, ..., dn`` are latent dimensions and ``dim_qk` is
            the embedding dim of the query tensor.
        k (Tensor): Key of shape ``(b, h, d1', ...., dn', dim_qk)`` or ``(b, h, seq_len', dim_qk)`` where
            ``h`` is the number of attention heads, ``d1', ..., dn'` are latent dimensions and ``dim_qk``
            is the key embedding dim aligned with query embedding dim,
            see :class:`~torchmultimodal.modules.layers.attention.MultiHeadAttention`.
        v (Tensor): Value of shape ``(b, h, d1', ..., dn', dim_v)`` or ``(b, h, seq_len', dim_v)`` where
            ``h`` is the number of attention heads, ``d1', ..., dn'`` are latent dimensions and ``dim_v``
            is the embedding dim of the value tensor.
        attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions. Applied before softmax.
        head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions.
            Applied after dropout, before matrix multiplication with values.
        attn_dropout (float): Probability of dropout after softmax. Default is ``0.0``.

    Returns:
        A tuple of output tensor and attention probabilities.
    """
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1]))
    if attention_mask is not None:
        attn = attn.masked_fill(attention_mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)
    attn = F.dropout(attn, p=attn_dropout)
    if head_mask is not None:
        attn = attn * head_mask
    a = torch.matmul(attn, v)
    return a, attn


def shift_dim(x: Tensor, src_dim: int=-1, dest_dim: int=-1, make_contiguous: bool=True) ->Tensor:
    """Permutes tensor x by moving src_dim to dest_dim.
    i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

    Args:
        x (Tensor): input Tensor you want to permute
        src_dim (int, optional): the axis you want to move. Negative indexing supported. Defaults to -1.
        dest_dim (int, optional): the axis you want to move to. Negative indexing supported. Defaults to -1.
        make_contiguous (bool, optional): if you want the output tensor to be contiguous in memory. Defaults to True.

    Returns:
        Tensor: permuted Tensor
    """
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims
    dims = list(range(n_dims))
    del dims[src_dim]
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


class AxialAttention(nn.Module):
    """Computes attention over a single axis of the input. Other dims are flattened into the batch dimension.

    Args:
        axial_dim (int): Dimension to compute attention on, indexed by input dimensions
            (i.e., ``0`` for first input dimension, ``1`` for second).
        attn_dropout (float): Probability of dropout after softmax. Default is ``0.0``.
    """

    def __init__(self, axial_dim: int, attn_dropout: float=0.0) ->None:
        super().__init__()
        self.axial_dim = axial_dim + 2
        self.attn_dropout = attn_dropout

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """
        Args:
            q (Tensor): Query input of shape ``(b, h, d1, ..., dn, dim_q)`` where ``h`` is the number of
                attention heads, ``(d1, ..., dn)`` are the query latent dimensions and ``dim_q`` is the dimension
                of the query embeddings.
            k, v (Tensor): Key/value input of shape ``(b, h, d1', ..., dn', dim_kv)`` where ``h`` is the number
                of attention heads, ``(d1', ..., dn')`` are the key/value latent dimensions and ``dim_kv`` is
                the dimension of the key/value embeddings.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)`` where ``q_dn`` is
                the dimension of the axis to compute attention on of the query and ``k_dn`` that of the key.
                Contains 1s for positions to attend to and 0s for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.

        Returns:
            A tuple of output tensor and attention probabilities.
        """
        if self.axial_dim >= len(q.shape) - 1:
            raise ValueError('axial dim does not match input shape')
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)
        out, attn_probs = scaled_dot_product_attention(q, k, v, attention_mask=attention_mask, head_mask=head_mask, attn_dropout=self.attn_dropout if self.training else 0.0)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out, attn_probs


class SelfAttention(nn.Module):
    """Computes attention over the entire n-dimensional input.

    Args:
        attn_dropout (float, optional): Probability of dropout after softmax. Default is ``0.0``.
    """

    def __init__(self, attn_dropout: float=0.0) ->None:
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """
        Args:
            q (Tensor): Query input of shape ``(b, h, d1, ..., dn, dim_q)`` where ``h`` is the number of
                attention heads, ``(d1, ..., dn)`` are the query latent dimensions and ``dim_q`` is the dimension
                of the query embeddings.
            k, v (Tensor): Key/value input of shape ``(b, h, d1', ..., dn', dim_kv)`` where ``h`` is the number
                of attention heads, ``(d1', ..., dn')`` are the key/value latent dimensions and ``dim_kv`` is
                the dimension of the key/value embeddings.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)`` where ``q_dn`` is the
                dimension of the flattened query input along its latent dimensions and ``k_dn`` that of the
                flattened key input. Contains 1s for positions to attend to and 0s for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.

        Returns:
            A tuple of output tensor and attention probabilities.
        """
        _, _, *shape, _ = q.shape
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)
        out, attn_probs = scaled_dot_product_attention(q, k, v, attention_mask=attention_mask, head_mask=head_mask, attn_dropout=self.attn_dropout if self.training else 0.0)
        return out.unflatten(2, shape), attn_probs


def merge_multihead(x: Tensor) ->Tensor:
    """Moves head dim back to original location and concatenates heads
    (b, n_head, d1, ..., dn, c // n_head) -> (b, d1, ..., dn, c)"""
    return shift_dim(x, 1, -2).flatten(start_dim=-2)


def split_multihead(x: Tensor, n_head: int) ->Tensor:
    """Splits channel dimension of input tensor of size (b, d1, ..., dn, c)
    into multiple heads, (b, n_head, d1, ..., dn, c // n_head)"""
    x = x.unflatten(-1, (n_head, -1))
    x = shift_dim(x, -2, 1)
    return x


class MultiHeadAttention(nn.Module):
    """Computes multihead attention with flexible attention mechanism and caching for fast decoding.

    Multihead attention linearly projects and divides queries, keys, and values into
    multiple 'heads'. This enables the computation of attention multiple times in
    parallel, creating more varied representations and allows the model to jointly
    attend to information from different representation subspaces at different positions,
    as described in `"Attention Is All You Need (Vaswani et al. 2017)"<https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        dim_q (int): Dimensionality of query embedding vector.
        dim_kv (int): Dimensionality of key/value embedding vector.
        n_head (int): Number of attention heads.
        attn_module (nn.Module): Module of attention mechanism to use. Default is ``SelfAttention``.
            See :class:`~torchmultimodal.modules.layers.attention.SelfAttention` for API details.
        add_bias (bool): Whether to add bias to the q, k, v, linear layers or not. Default is ``True``.

    Attributes:
        cache (Dict[str, Tensor]): Dictionary that stores past key/value vectors.

    Raises:
        ValueError: When ``dim_q`` or ``dim_kv`` is not divisible by ``n_head``.
    """

    def __init__(self, dim_q: int, dim_kv: int, n_head: int, attn_module: nn.Module=SelfAttention(), add_bias: bool=True) ->None:
        super().__init__()
        if dim_q % n_head != 0 or dim_kv % n_head != 0:
            raise ValueError('The hidden size of q, k, v must be a multiple of the number of attention heads.')
        self.d_qk = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head
        self.query = nn.Linear(dim_q, n_head * self.d_qk, bias=add_bias)
        self.key = nn.Linear(dim_kv, n_head * self.d_qk, bias=add_bias)
        self.value = nn.Linear(dim_kv, n_head * self.d_v, bias=add_bias)
        self.output = nn.Linear(n_head * self.d_v, dim_q, bias=True)
        self.attn = attn_module
        self.cache: Optional[Dict[str, Tensor]] = None

    def forward(self, q: Tensor, kv: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, return_attn_weights: bool=False, use_cache: bool=False, causal: bool=False) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            q (Tensor): Query of shape ``(b, d1, ..., dn, dim_q)`` or ``(b, seq_len, dim_q)``
                (for autoregressive decoding it's typical to pass in flattened tensors).
            kv (Tensor, optional): Key (and value) of shape ``(b, d1', ..., dn', dim_kv)`` or
                ``(b, seq_len', dim_kv)``. If this argument is specified, cross-attention will be applied.
                Default is ``None``.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)`` where ``q_dn`` is
                the dimension of the axis to compute attention on of the query and ``k_dn`` that of the key.
                If the input tensors are flattened across the entire latent dimensions, ``q_dn = d1 x ... x dn``
                and ``k_dn = d1' x ... x dn'``. Contains 1s for positions to attend to and 0s
                for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.
            use_cache (bool): If ``True``, caches past ``k`` and ``v`` tensors for faster decoding.
                If ``False``, recomputes ``k`` and ``v`` for each decoding step. Default is ``False``.
            causal (bool): Whether to use causal attention or not. Default is ``False``.

        Returns:
            * If ``return_attn_weights`` is ``True``: A tuple of output tensor and attention probabilities.
            * If ``return_attn_weights`` is ``False``: A single output tensor.

        Raises:
            TypeError: An error occurred when ``causal`` is ``True`` and ``attn_module`` is ``AxialAttention``.
        """
        if isinstance(self.attn, AxialAttention) and causal:
            raise TypeError('Causal axial attention is not supported.')
        k = v = q if kv is None else kv
        q = split_multihead(self.query(q), self.n_head)
        if causal or not self.cache:
            k = split_multihead(self.key(k), self.n_head)
            v = split_multihead(self.value(v), self.n_head)
        if use_cache:
            if not self.cache:
                self.cache = dict(k=k.clone(), v=v.clone())
            else:
                if causal:
                    k_, v_ = self.cache['k'], self.cache['v']
                    self.cache['k'] = torch.cat([k_, k], dim=2)
                    self.cache['v'] = torch.cat([v_, v], dim=2)
                k, v = self.cache['k'], self.cache['v']
        a, attn_probs = self.attn(q, k, v, attention_mask, head_mask)
        a = merge_multihead(a)
        a = self.output(a)
        if return_attn_weights:
            return a, attn_probs
        else:
            return a


class TransformerCrossAttentionLayer(nn.Module):
    """Transformer layer with self-attention on inputs and cross-attention on an encoder's outputs.
    Can be used in a transformer decoder or an encoder with cross-attention. Similar to
    ``nn.TransformerDecoderLayer``, but generalized for use in an encoder with cross-attention as well.
    Uses a custom ``MultiHeadAttention`` that supports n-dimensional inputs including sequences,
    images, video.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        encoder_hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate
            cross-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``.
            See ``MultiHeadAttention`` for shape requirements.
        cross_attention_mask (Tensor, optional): mask to be applied to cross-attention inputs,
            ``encoder_hidden_states``. See ``MultiHeadAttention`` for shape requirements.
    """

    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float=0.0, activation: Callable[..., nn.Module]=nn.ReLU, layer_norm_eps: float=1e-12, norm_first: bool=False) ->None:
        super().__init__()
        self.attention = MultiHeadAttention(dim_q=d_model, dim_kv=d_model, n_head=n_head, attn_module=SelfAttention(dropout))
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(dim_q=d_model, dim_kv=d_model, n_head=n_head, attn_module=SelfAttention(dropout))
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feedforward = MLP(d_model, d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(self, hidden_states: Tensor, attention_mask: Optional[Tensor]=None) ->Tensor:
        output = self.attention(hidden_states, attention_mask=attention_mask, return_attn_weights=False)
        output = self.attention_dropout(output)
        return output

    def _cross_attention_block(self, hidden_states: Tensor, encoder_hidden_states: Tensor, cross_attention_mask: Optional[Tensor]=None) ->Tensor:
        output = self.cross_attention(hidden_states, encoder_hidden_states, attention_mask=cross_attention_mask, return_attn_weights=False)
        output = self.cross_attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: Tensor) ->Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(self, hidden_states: Tensor, encoder_hidden_states: Tensor, attention_mask: Optional[Tensor]=None, cross_attention_mask: Optional[Tensor]=None) ->Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._self_attention_block(inputs, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.cross_attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(attn_norm_output, kv, cross_attention_mask)
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.feedforward_layernorm(cross_attention_residual)
        ff_residual = cross_attention_norm_output + self._feedforward_block(cross_attention_norm_output)
        return ff_residual

    def _forward_postnorm(self, hidden_states: Tensor, encoder_hidden_states: Tensor, attention_mask: Optional[Tensor]=None, cross_attention_mask: Optional[Tensor]=None) ->Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        attn_output = self._self_attention_block(x, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(attn_norm_output, kv, cross_attention_mask)
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.cross_attention_layernorm(cross_attention_residual)
        ff_residual = cross_attention_norm_output + self._feedforward_block(cross_attention_norm_output)
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, attention_mask: Optional[Tensor]=None, cross_attention_mask: Optional[Tensor]=None) ->Tensor:
        if self.norm_first:
            return self._forward_prenorm(hidden_states, encoder_hidden_states, attention_mask, cross_attention_mask)
        else:
            return self._forward_postnorm(hidden_states, encoder_hidden_states, attention_mask, cross_attention_mask)


def get_extended_attention_mask(attention_mask: Tensor) ->Tensor:
    """Makes attention masks broadcastable along head and sequence dimensions.

    Accepting two types of attention masks:
        - Causal: masks that prevent attending to future positions of dimensions
            ``(batch_size, query_seq_len, key_seq_len)``
        - Padding: masks that prevent attending to token paddings of dimensions
            ``(batch_size, seq_len)``

    Args:
        attention_mask (Tensor):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.

    Returns:
        extended_attention_mask (Tensor):
            The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
    """
    if attention_mask.dim() == 4:
        extended_attention_mask = attention_mask
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError('Wrong shape for attention_mask (shape {})'.format(attention_mask.shape))
    extended_attention_mask = extended_attention_mask
    return extended_attention_mask


class ALBEFMultimodalEncoder(nn.Module):
    """
    Construct multimodal embeddings from image embeddings, text embeddings, and text attention mask.

    The ALBEFMultimodalEncoder is similar to ALBEFTextEncoder, with the addition of image-text cross attention in encoder layers.

    Args:
        hidden_size (int): Dimensionality of the encoder layers.
            Default is 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            Default is 6.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            Default is 12.
        intermediate_size (int): Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
            Default is 3072.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
            Default is 1e-12.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function for the Transformer encoder layer.
            Default is GELU.

    Inputs:
        hidden_states (Tensor of shape (batch_size, seq_len, hidden_size)):
            Unimodal input hidden states.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Optional[Tensor] of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.
        is_decoder (bool): Whether this module is used as a decoder. Default is False.
    """

    def __init__(self, hidden_size: int=768, num_hidden_layers: int=6, num_attention_heads: int=12, intermediate_size: int=3072, layer_norm_eps: float=1e-12, transform_act_fn: Callable[..., nn.Module]=nn.GELU) ->None:
        super().__init__()
        self.layer = nn.ModuleList([TransformerCrossAttentionLayer(d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=transform_act_fn, layer_norm_eps=layer_norm_eps) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, encoder_hidden_states: Tensor, encoder_attention_mask: Optional[Tensor]=None) ->Tensor:
        attention_mask = get_extended_attention_mask(attention_mask)
        if encoder_attention_mask is not None:
            encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask)
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, cross_attention_mask=encoder_attention_mask)
        return hidden_states


class BERTTextEmbeddings(nn.Module):
    """Construct word, position, and token type embeddings following BERT, similar to HuggingFace BertEmbeddings

    Attributes:
        hidden_size (int): size of embedding space. Default is 768.
        vocab_size (int): size of vocabulary. Default is 30522.
        pad_token_id (int): id used for padding token. Default is 0.
        max_position_embeddings (int): the highest position id number, or max sequence length. Default is 512.
        type_vocab_size (int): the highest token type id number. Default is 2.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        dropout (float): dropout probability after all embeddings and layernorm
        offset_pos_ids (bool): if True, shift position ids by one for the padding token. Used in RoBERTa.
            Default is False.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere
    """

    def __init__(self, hidden_size: int=768, vocab_size: int=30522, pad_token_id: int=0, max_position_embeddings: int=512, type_vocab_size: int=2, layer_norm_eps: float=1e-12, dropout: float=0.0, offset_pos_ids: bool=False) ->None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id
        self.offset_pos_ids = offset_pos_ids

    def create_position_ids_from_input_ids(self, input_ids: Tensor) ->Tensor:
        """
        Replace non-padding symbols with their position numbers.
        Position numbers begin at pad_token_id+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Inputs: input_ids (Tensor): Tensor from which to create position IDs.
                pad_token_id (int): Padding index
                    (determines starting point of position IDs).
        """
        mask = input_ids.ne(self.pad_token_id).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + self.pad_token_id

    def forward(self, input_ids: Optional[Tensor]=None, token_type_ids: Optional[Tensor]=None, position_ids: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None) ->Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError('input_ids or inputs_embeds must not be None')
        seq_length = input_shape[1]
        if position_ids is None:
            if self.offset_pos_ids:
                position_ids = self.create_position_ids_from_input_ids(input_ids)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_causal_attention_mask(tgt_seq_len: int, src_seq_len: Optional[int]=None) ->Tensor:
    """
    Generates causal attention masks of dimensions (target_sequence_length, source_sequence_length).
    """
    if src_seq_len is None:
        src_seq_len = tgt_seq_len
    return torch.tril(torch.ones(tgt_seq_len, src_seq_len))


class ALBEFDecoder(nn.Module):
    """
    Generate the prediction scores for answers from image and question hidden states.

    Args:
        text_embeddings (ALBEFTextEmbeddings): Instantiated ALBEFTextEmbeddings.
        multimodal_encoder (ALBEFMultimodalEncoder): Instantiated ALBEFMultimodalEncoder.
        prediction_head (PredictionHead): Instantiated PredictionHead.

    Inputs:
        input_ids (Tensor of shape (batch_size, seq_len)):
            Input ids for input text tokens.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Tensor of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.

    Returns:
        Tensor: Prediction scores for answers.
    """

    def __init__(self, text_embeddings: BERTTextEmbeddings, multimodal_encoder: ALBEFMultimodalEncoder, prediction_head: PredictionHead) ->None:
        super().__init__()
        self.text_embeddings = text_embeddings
        self.multimodal_encoder = multimodal_encoder
        self.prediction_head = prediction_head

    def get_extended_attention_mask_for_decoder(self, attention_mask: Tensor) ->Tensor:
        """
        Apply a causal mask in addition to the padding mask and make the mask broadcastable,
        such that future and masked tokens are ignored.

        Args:
            attention_mask (Tensor):
                Padding mask with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            extended_attention_mask (Tensor):
                The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
        """
        device = attention_mask.device
        batch_size, seq_length = attention_mask.shape
        causal_mask = get_causal_attention_mask(seq_length)
        causal_mask = causal_mask.repeat(batch_size, 1).view(batch_size, seq_length, seq_length)
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask
        return extended_attention_mask

    def forward(self, input_ids: Tensor, attention_mask: Tensor, encoder_hidden_states: Tensor, encoder_attention_mask: Tensor) ->Tensor:
        hidden_states = self.text_embeddings(input_ids)
        attention_mask = self.get_extended_attention_mask_for_decoder(attention_mask)
        decoder_output = self.multimodal_encoder(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        prediction_scores = self.prediction_head(decoder_output)
        return prediction_scores


ALBEFOutput = namedtuple('ALBEFOutput', ['image_embeddings', 'image_embeddings_m', 'text_embeddings', 'text_embeddings_m', 'multimodal_embeddings', 'multimodal_embeddings_m'], defaults=(None, None, None, None, None, None))


@torch.no_grad()
def momentum_update(model: nn.Module, model_m: nn.Module, momentum: float) ->None:
    for param, param_m in zip(model.parameters(), model_m.parameters()):
        param_m.data = param_m.data * momentum + param.data * (1 - momentum)


@torch.no_grad()
def remove_grad(model: nn.Module) ->None:
    for param in model.parameters():
        param.requires_grad = False


class ALBEFModel(nn.Module):
    """
    ALBEF is a model to ALign the image and text representations BEfore Fusing
    (ALBEF) them through cross-modal attention, which enables more grounded vision
    and language representation learning. (https://arxiv.org/pdf/2107.07651.pdf)

    Args:   vision_encoder (nn.Module): Instantiated vision encoder.
            text_encoder (nn.Module): Instantiated text encoder.
            multimodal_encoder (nn.Module): Instantiated multimodal encoder.
            momentum (float): Momentum parameter. Default is 0.995.

    Inputs: image (Tensor): Tensor of shape (B, C, H, W) containing image features.
            text (Tensor): Tensor of shape (B, L) containing text features.
            text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
    """

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, multimodal_encoder: nn.Module, momentum: float=0.995) ->None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.multimodal_encoder_m = copy.deepcopy(multimodal_encoder)
        remove_grad(self.vision_encoder_m)
        remove_grad(self.text_encoder_m)
        remove_grad(self.multimodal_encoder_m)
        self.momentum = momentum

    def forward(self, image: Tensor, text: Tensor, text_atts: Tensor) ->ALBEFOutput:
        image_embeds = self.vision_encoder(image)
        text_embeds = self.text_encoder(text, text_atts)
        multimodal_embeddings = self.multimodal_encoder(hidden_states=text_embeds.last_hidden_state, attention_mask=text_atts, encoder_hidden_states=image_embeds)
        with torch.no_grad():
            momentum_update(self.vision_encoder, self.vision_encoder_m, self.momentum)
            momentum_update(self.text_encoder, self.text_encoder_m, self.momentum)
            momentum_update(self.multimodal_encoder, self.multimodal_encoder_m, self.momentum)
            image_embeds_m = self.vision_encoder_m(image)
            text_embeds_m = self.text_encoder_m(text, text_atts)
            multimodal_embeddings_m = self.multimodal_encoder_m(hidden_states=text_embeds_m.last_hidden_state, attention_mask=text_atts, encoder_hidden_states=image_embeds_m)
        return ALBEFOutput(image_embeddings=image_embeds, image_embeddings_m=image_embeds_m, text_embeddings=text_embeds.last_hidden_state, text_embeddings_m=text_embeds_m.last_hidden_state, multimodal_embeddings=multimodal_embeddings, multimodal_embeddings_m=multimodal_embeddings_m)


class CausalLanguageModelingLoss(nn.Module):
    """
    Compute the autoregressive masked language modeling loss by predicting the next token, as used in VQA.
    Support loss distillation for non-zero alpha. Compute standard mlm loss for zero alpha.

    Args:
        mask_token_id (int): The token id indicating a masked token. Default is -100.

    Inputs:
        labels (Tensor of shape (batch_size, seq_length)): The masked output tokens.
        prediction_scores (Tensor of shape (batch_size, seq_length, vocab_size)):
            The prediction scores from a prediction head.
        prediction_scores_m (Optional[Tensor] of shape (batch_size, seq_length, vocab_size)):
            The prediction scores from a momentum prediction head.
            Required if alpha is non-zero.
        alpha (float): The interpolation value between mlm_loss and loss_distill. Default is 0.
    """

    def __init__(self, mask_token_id: int=-100) ->None:
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, labels: Tensor, prediction_scores: Tensor, prediction_scores_m: Optional[Tensor]=None, alpha: Optional[float]=0.0) ->Tensor:
        batch_size = labels.size(0)
        prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        mlm_loss = F.cross_entropy(prediction_scores.view(-1, prediction_scores.shape[-1]), labels.view(-1), reduction='none')
        mlm_loss = mlm_loss.view(batch_size, -1).sum(1)
        if alpha != 0:
            assert prediction_scores_m is not None, 'prediction_scores_m cannot be None for non-zero alpha'
            with torch.no_grad():
                prediction_scores_m = prediction_scores_m[:, :-1, :].contiguous()
            loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=-1) * F.softmax(prediction_scores_m, dim=-1), dim=-1)
            loss_distill = (loss_distill * (labels != self.mask_token_id)).sum(1)
            mlm_loss = (1 - alpha) * mlm_loss + alpha * loss_distill
        return mlm_loss


class ALBEFModelForVQA(nn.Module):
    """
    ALBEF Model for VQA finetuning and inference.

    Args:
        model (ALBEFModel): Instantiated ALBEFModel.
        answer_decoder (ALBEFDecoder): Instantiated ALBEFDecoder.
        loss (CausalLanguageModelingLoss): Instantiated CausalLanguageModelingLoss.

    Inputs:
        image (Tensor of shape (B, C, H, W)): Image features.
        question (Tensor of shape (B, L)): Question text features.
        question_atts (Tensor of shape (B, L)): Question attention mask.
        answers (Tensor of shape (N, M)): Answer text features.
        answers_atts (Tensor of shape (N, M)): Answer attention mask.
        ans_weights (Optional[Tensor] of shape (N)): Weights for each answer.
            Required if is_train is True.
        ans_lengths (Optional[List[int]] of length B): Number of answers for each question.
            ans_lengths should sum to N.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Required if is_train is True.
        k (Optional[int]): The number of answers to return for inference.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.

    Returns:
        is_train is True:
            Tensor: The masked language modeling loss for input.
        is_train is False:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
    """

    def __init__(self, model: ALBEFModel, answer_decoder: ALBEFDecoder, loss: CausalLanguageModelingLoss) ->None:
        super().__init__()
        self.model = model
        self.answer_decoder = answer_decoder
        self.loss = loss
        self.answer_decoder_m = copy.deepcopy(self.answer_decoder)
        remove_grad(self.answer_decoder_m)

    def _train_forward(self, image: Tensor, question: Tensor, question_atts: Tensor, answers: Tensor, answers_atts: Tensor, ans_weights: Tensor, ans_lengths: List[int], alpha: float) ->Tensor:
        """
        Forward step for training. Encode the inputs with the ALBEFModel.
        Generate pseudo-targets using answer_decoder_m (momentum decoder model).
        Generate answer predictions using answer_decoder.
        Compute masked language modeling loss of the predictions using answers as labels,
            pseudo-targets as soft-labels, and alpha as their interpolation value.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answers_atts (Tensor of shape (N, M)): Answer attention mask.
            ans_weights (Tensor of shape (N)): Weights for each answer.
            ans_lengths (List[int] of length B): Number of answers for each question.
                ans_lengths should sum to N.
            alpha (float): The interpolation value between clm_loss and loss_distill.

        Returns:
            Tensor: The masked language modeling loss for input.
        """
        encoder_outputs = self.model(image, question, question_atts)
        encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask = self._encoder_hidden_states(encoder_outputs.multimodal_embeddings, encoder_outputs.multimodal_embeddings_m, question_atts, ans_lengths)
        with torch.no_grad():
            momentum_update(self.answer_decoder, self.answer_decoder_m, self.model.momentum)
            prediction_scores_m = self.answer_decoder_m(input_ids=answers, attention_mask=answers_atts, encoder_hidden_states=encoder_hidden_states_m, encoder_attention_mask=encoder_attention_mask)
        prediction_scores = self.answer_decoder(input_ids=answers, attention_mask=answers_atts, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        labels = answers.masked_fill(answers == 0, self.loss.mask_token_id)
        loss = self.loss(labels, prediction_scores, prediction_scores_m, alpha)
        loss = ans_weights * loss
        loss = loss.sum() / image.size(0)
        return loss

    def _eval_forward(self, image: Tensor, question: Tensor, question_atts: Tensor, answers: Tensor, answer_atts: Tensor, k: int=128) ->Tuple[Tensor, Tensor]:
        """
        Forward step for evaluation. Encode the inputs with the ALBEFModel.
        Generate answer autoregressively using the decoder, starting with the [CLS] token.
        Compute the answer ids and their perspective probabilities of the top k predictions.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answer_atts (Tensor of shape (N, M)): Answer attention mask.
            k (int): The number of answers to return for inference.

        Returns:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
        """
        encoder_outputs = self.model(image, question, question_atts)
        num_ques = question.size(0)
        start_ids = answers[0, 0].repeat(num_ques, 1)
        atts = torch.ones(start_ids.shape)
        prediction_scores = self.answer_decoder(input_ids=start_ids, attention_mask=atts, encoder_hidden_states=encoder_outputs.multimodal_embeddings, encoder_attention_mask=question_atts)
        logits = prediction_scores[:, 0, :]
        answer_first_token = answers[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)
        input_ids = []
        input_atts = []
        for topk_id in topk_ids:
            input_ids.append(answers.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids)
        input_atts = torch.cat(input_atts)
        targets_ids = input_ids.masked_fill(input_ids == 0, self.loss.mask_token_id)
        question_states = encoder_outputs.multimodal_embeddings.repeat_interleave(k, dim=0)
        question_atts = question_atts.repeat_interleave(k, dim=0)
        prediction_scores = self.answer_decoder(input_ids=input_ids, attention_mask=input_atts, encoder_hidden_states=question_states, encoder_attention_mask=question_atts)
        answer_loss = self.loss(targets_ids, prediction_scores)
        answer_loss = answer_loss.view(input_ids.size(0), -1)
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs

    def _encoder_hidden_states(self, multimodal_embeds: Tensor, multimodal_embeds_m: Tensor, question_atts: Tensor, ans_lengths: List[int]) ->Tuple[Tensor, Tensor, Tensor]:
        """
        Repeat each image-question input, repeat its embedding and mask to match the number of answers it has.

        Args:
            multimodal_embeds (Tensor): Image-question embeddings.
            multimodal_embeds_m (Tensor): Image-question embeddings from the momentum model.
            question_atts (Tensor): Question attention mask.
            ans_lengths (List[int]): The number of answers each image-question input has.

        Returns:
            encoder_hidden_states (Tensor): Image-question embeddings after the repetition.
            encoder_hidden_states_m (Tensor): Image-question embeddings from the momentum model after the repetition.
            encoder_attention_mask (Tensor): Question attention mask after the repetition.
        """
        encoder_hidden_states = []
        encoder_attention_mask = []
        for b, n in enumerate(ans_lengths):
            encoder_hidden_states += [multimodal_embeds[b]] * n
            encoder_attention_mask += [question_atts[b]] * n
        encoder_hidden_states = torch.stack(encoder_hidden_states)
        encoder_attention_mask = torch.stack(encoder_attention_mask)
        with torch.no_grad():
            encoder_hidden_states_m = []
            for b, n in enumerate(ans_lengths):
                encoder_hidden_states_m += [multimodal_embeds_m[b]] * n
            encoder_hidden_states_m = torch.stack(encoder_hidden_states_m)
        return encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask

    def forward(self, image: Tensor, question: Tensor, question_atts: Tensor, answers: Tensor, answers_atts: Tensor, ans_weights: Optional[Tensor]=None, ans_lengths: Optional[List[int]]=None, alpha: Optional[float]=0.0, k: Optional[int]=128, is_train: Optional[bool]=True) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(image, question, question_atts, answers, answers_atts, ans_weights, ans_lengths, alpha)
        else:
            return self._eval_forward(image, question, question_atts, answers, answers_atts, k)


ALBEFSimilarity = namedtuple('ALBEFSimilarity', ['sim_i2t', 'sim_t2i', 'sim_i2t_m', 'sim_t2i_m'], defaults=(None, None, None, None))


ALBEFWithSimilarityOutput = namedtuple('ALBEFWithSimilarityOutput', ['image_embeddings', 'text_embeddings', 'multimodal_embeddings', 'multimodal_embeddings_neg', 'similarity', 'sim_targets'], defaults=(None, None, None, None, None, None))


def _gather_embeddings(embeddings: torch.Tensor) ->torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return embeddings
    embeddings_all_gpus = [torch.zeros_like(embeddings) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(embeddings_all_gpus, embeddings)
    return torch.cat(embeddings_all_gpus)


class ALBEFModelWithSimilarity(nn.Module):
    """
    ALBEFModelWithSimilarity outputs image embeddings, text embeddings, multimodal embeddings,
    negative image-text pairs multimodal embeddings, and image-text similarity, as used in ITC
    and ITM losses.

    Args:   albef_model (ALBEFModel): Instantiated ALBEF model.
            vision_proj (nn.Module): Instantiated vision projection layer.
            text_proj (nn.Module): Instantiated text projection layer.
            embed_size (int): Embedding size of the vision and text projection layers. Default is 256.
            queue_size (int): Size of image and text queues for momentum distillation. Default is 65536.
            masked_token_id (int): The token id indicating a masked token. Default is -100.
            temp (float): Temperature parameter. Default is 0.07.

    Inputs: image (Tensor): Tensor of shape (B, C, H, W) containing image features.
            text (Tensor): Tensor of shape (B, L) containing text features.
            text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
            idx (Tensor): Tensor of shape (B) containing unique identifiers for each sample.
    """

    def __init__(self, albef_model: ALBEFModel, vision_proj: nn.Module, text_proj: nn.Module, embed_size: int=256, queue_size: int=65536, mask_token_id: int=-100, temp: float=0.07) ->None:
        super().__init__()
        self.albef_model = albef_model
        self.vision_proj = vision_proj
        self.text_proj = text_proj
        self.vision_proj_m = copy.deepcopy(vision_proj)
        self.text_proj_m = copy.deepcopy(text_proj)
        remove_grad(self.vision_proj_m)
        remove_grad(self.text_proj_m)
        self.queue_size = queue_size
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.register_buffer('image_queue', torch.randn(embed_size, queue_size))
        self.register_buffer('text_queue', torch.randn(embed_size, queue_size))
        self.register_buffer('idx_queue', torch.full((1, self.queue_size), mask_token_id))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.image_queue: Tensor
        self.text_queue: Tensor
        self.idx_queue: Tensor
        self.queue_ptr: Tensor
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image: Tensor, text: Tensor, text_atts: Tensor, idx: Tensor) ->ALBEFWithSimilarityOutput:
        outputs = self.albef_model(image, text, text_atts)
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.detach().clone()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        similarity = self._similarity(outputs.image_embeddings, outputs.image_embeddings_m, outputs.text_embeddings, outputs.text_embeddings_m, idx)
        image_embeds_neg, text_embeds_neg, text_atts_neg = self._neg_embeddings(outputs.image_embeddings, outputs.text_embeddings, text_atts, similarity)
        multimodal_embeddings_neg = self.albef_model.multimodal_encoder(torch.cat([outputs.text_embeddings, text_embeds_neg], dim=0), torch.cat([text_atts, text_atts_neg], dim=0), torch.cat([image_embeds_neg, outputs.image_embeddings], dim=0))
        return ALBEFWithSimilarityOutput(image_embeddings=outputs.image_embeddings, text_embeddings=outputs.text_embeddings, multimodal_embeddings=outputs.multimodal_embeddings, multimodal_embeddings_neg=multimodal_embeddings_neg, similarity=similarity, sim_targets=sim_targets)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat_m: Tensor, text_feat_m: Tensor, idx: Tensor) ->None:
        image_feats = _gather_embeddings(image_feat_m)
        text_feats = _gather_embeddings(text_feat_m)
        idxs = _gather_embeddings(idx)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, 'queue_size should be divisible by batch_size'
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _similarity(self, image_embeds: Tensor, image_embeds_m: Tensor, text_embeds: Tensor, text_embeds_m: Tensor, idx: Tensor) ->ALBEFSimilarity:
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        with torch.no_grad():
            momentum_update(self.vision_proj, self.vision_proj_m, self.albef_model.momentum)
            momentum_update(self.text_proj, self.text_proj_m, self.albef_model.momentum)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.detach().clone()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.detach().clone()], dim=1)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        return ALBEFSimilarity(sim_i2t=sim_i2t, sim_t2i=sim_t2i, sim_i2t_m=sim_i2t_m, sim_t2i_m=sim_t2i_m)

    def _neg_embeddings(self, image_embeds: Tensor, text_embeds: Tensor, text_atts: Tensor, similarity: ALBEFSimilarity) ->Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(similarity.sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(similarity.sim_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
        image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_t2i[b], 1).item())
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_i2t[b], 1).item())
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg, text_atts_neg


class ImageTextContrastiveLoss(nn.Module):
    """
    Compute the image-text contrastive loss from image-text similarity, as used in ALBEF.
    Support loss distillation with pseudo-targets for non-zero alpha. Compute standard contrastive loss for zero alpha.

    Inputs:
        image_to_text_sim (Tensor): Image to text similarity.
        text_to_image_sim (Tensor): Text to image similarity.
        image_to_text_sim_m (Optional[Tensor]): Image to text similarity from momentum models.
            Required if alpha is non-zero.
        text_to_image_sim_m (Optional[Tensor]): Text to image similarity from momentum models.
            Required if alpha is non-zero.
        sim_targets (Optional[Tensor]): Similarity pseudo-targets from momentum models. Default is the diagonal matrix.
            Requires all Tensor inputs to have the same size.
        alpha (Optional[float]): The interpolation value of momentum similarity and sim_targets. Default is 0.
    """

    def __init__(self) ->None:
        super().__init__()

    def forward(self, image_to_text_sim: Tensor, text_to_image_sim: Tensor, image_to_text_sim_m: Optional[Tensor]=None, text_to_image_sim_m: Optional[Tensor]=None, sim_targets: Optional[Tensor]=None, alpha: Optional[float]=0.0) ->Tensor:
        if sim_targets is None:
            sim_targets = torch.zeros(image_to_text_sim.size())
            sim_targets.fill_diagonal_(1)
        if alpha != 0:
            assert image_to_text_sim_m is not None and text_to_image_sim_m is not None, 'sim_i2t_m and sim_t2i_m cannot be none for non-zero alpha'
            with torch.no_grad():
                image_to_text_sim_targets = alpha * F.softmax(image_to_text_sim_m, dim=1) + (1 - alpha) * sim_targets
                text_to_image_sim_targets = alpha * F.softmax(text_to_image_sim_m, dim=1) + (1 - alpha) * sim_targets
        else:
            image_to_text_sim_targets = sim_targets
            text_to_image_sim_targets = sim_targets
        loss_i2t = -torch.sum(F.log_softmax(image_to_text_sim, dim=1) * image_to_text_sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(text_to_image_sim, dim=1) * text_to_image_sim_targets, dim=1).mean()
        loss_itc = (loss_i2t + loss_t2i) / 2
        return loss_itc


class ALBEFModelForRetrieval(nn.Module):
    """
    ALBEF Model for Retrieval finetuning and inference.
    In training mode, the forward step computes image-text contrastive loss and
    image-text matching loss.
    In evaluation mode, the forward step takes 3 types of input:
        image: encode image input, project and normalize the embeddings.
        text: encode text input, project and normalize the embeddings.
        multimodal: create multimodal embeddings from image and text
            embeddings, and compute image-text matching scores.

    Args:
        model_with_similarity (ALBEFModelWithSimilarity): Instantiated ALBEFModelWithSimilarity.
        itc_loss (ImageTextContrastiveLoss): Instantiated ImageTextContrastiveLoss.
        hidden_size (int): Dimensionality of encoder outputs.

    Inputs:
        image (Optional[Tensor] of shape (B, C, H, W)): Image features.
            Required if is_train is True.
            Required if input_type is "image" or "multimodal".
        text (Optional[Tensor] of shape (B, L)): Text features.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        text_atts (Tensor of shape (B, L)): Text attention mask.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        idx (Tensor of shape (B)): Identifier for each image sample.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Default is 0.
        input_type (Optional[str]): "image", "text", or "multimodal" indicating the encoding type.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.
            Default is True.

    Returns:
        is_train is True:
            Tensor: The sum of itc loss and itm loss.
        is_train is False:
            input_type is "image":
                Tuple[Tensor, Tensor]: Image embeddings and projected image features.
            input_type is "text":
                Tuple[Tensor, Tensor]: Text embeddings and projected text features.
            input_type is "multimodal"
                Tensor: Scores for the retrieval task.
    """

    def __init__(self, model_with_similarity: ALBEFModelWithSimilarity, itc_loss: ImageTextContrastiveLoss, hidden_size: int) ->None:
        super().__init__()
        self.model_with_similarity = model_with_similarity
        self.itc_loss = itc_loss
        self.itm_head = nn.Linear(hidden_size, 2)

    def _train_forward(self, image: Tensor, text: Tensor, text_atts: Tensor, idx: Tensor, alpha: float) ->Tensor:
        encoder_output = self.model_with_similarity(image, text, text_atts, idx)
        similarity_outputs = encoder_output.similarity
        similarity_targets = encoder_output.sim_targets
        itc_loss = self.itc_loss(similarity_outputs.sim_i2t, similarity_outputs.sim_t2i, similarity_outputs.sim_i2t_m, similarity_outputs.sim_t2i_m, similarity_targets, alpha)
        pos_embeddings = encoder_output.multimodal_embeddings[:, 0, :]
        neg_embeddings = encoder_output.multimodal_embeddings_neg[:, 0, :]
        vl_embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(pos_embeddings.size(0), dtype=torch.long), torch.zeros(neg_embeddings.size(0), dtype=torch.long)], dim=0)
        itm_loss = F.cross_entropy(vl_output, itm_labels)
        loss = itc_loss + itm_loss
        return loss

    def _encode_image(self, image: Tensor) ->Tuple[Tensor, Tensor]:
        image_embed = self.model_with_similarity.albef_model.vision_encoder(image)
        image_feat = F.normalize(self.model_with_similarity.vision_proj(image_embed[:, 0, :]), dim=-1)
        return image_embed, image_feat

    def _encode_text(self, text: Tensor, text_atts: Tensor) ->Tuple[Tensor, Tensor]:
        text_embed = self.model_with_similarity.albef_model.text_encoder(text, text_atts).last_hidden_state
        text_feat = F.normalize(self.model_with_similarity.text_proj(text_embed[:, 0, :]), dim=-1)
        return text_embed, text_feat

    def _image_text_matching_score(self, image: Tensor, text: Tensor, text_atts: Tensor) ->Tensor:
        multimodal_embeds = self.model_with_similarity.albef_model.multimodal_encoder(text, text_atts, image)
        score = self.itm_head(multimodal_embeds[:, 0, :])[:, 1]
        return score

    def _eval_forward(self, input_type: str, image: Optional[Tensor], text: Optional[Tensor], text_atts: Optional[Tensor]) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if input_type == 'image':
            assert image is not None, 'image input tensor cannot be None'
            return self._encode_image(image)
        elif input_type == 'text':
            assert text is not None and text_atts is not None, 'text and text attention mask cannot be None'
            return self._encode_text(text, text_atts)
        elif input_type == 'multimodal':
            assert image is not None and text is not None and text_atts is not None, 'image embeddings, text embeddings, and text attention mask cannot be None'
            return self._image_text_matching_score(image, text, text_atts)
        else:
            raise ValueError('input_type must be image, text, or multimodal')

    def forward(self, image: Optional[Tensor]=None, text: Optional[Tensor]=None, text_atts: Optional[Tensor]=None, idx: Optional[Tensor]=None, alpha: Optional[Tensor]=0.0, input_type: Optional[str]=None, is_train: Optional[bool]=True) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(image, text, text_atts, idx, alpha)
        else:
            return self._eval_forward(input_type, image, text, text_atts)


class CNNEncoder(nn.Module):
    """A CNN encoder.

    Stacks n layers of (Conv2d, MaxPool2d, BatchNorm2d), where n is determined
    by the length of the input args.

    Args:
        input_dims (List[int]): List of input dimensions.
        output_dims (List[int]): List of output dimensions. Should match
            input_dims offset by one.
        kernel_sizes (List[int]): Kernel sizes for convolutions. Should match
            the sizes of cnn_input_dims and cnn_output_dims.

    Inputs:
        x (Tensor): Tensor containing a batch of images.
    ​
    """

    def __init__(self, input_dims: List[int], output_dims: List[int], kernel_sizes: List[int]):
        super().__init__()
        conv_layers: List[nn.Module] = []
        assert len(input_dims) == len(output_dims) and len(output_dims) == len(kernel_sizes), 'input_dims, output_dims, and kernel_sizes should all have the same length'
        assert input_dims[1:] == output_dims[:-1], 'output_dims should match input_dims offset by one'
        for in_channels, out_channels, kernel_size in zip(input_dims, output_dims, kernel_sizes):
            padding_size = kernel_size // 2
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
            max_pool2d = nn.MaxPool2d(2, stride=2)
            batch_norm_2d = nn.BatchNorm2d(out_channels)
            conv_layers.append(nn.Sequential(conv, nn.LeakyReLU(), max_pool2d, batch_norm_2d))
        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.cnn(x)


class LSTMEncoder(nn.Module):
    """An LSTM encoder. Stacks an LSTM on an embedding layer.

    Args:
        vocab_size (int): The size of the vocab for embeddings.
        embedding_dim (int): The size of each embedding vector.
        input_size (int): The number of features in the LSTM input.
        hidden_size (int): The number of features in the hidden state.
        bidirectional (bool): Whether to use bidirectional LSTM.
        batch_first (bool): Whether to provide batches as (batch, seq, feature)
            or (seq, batch, feature).

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(self, vocab_size: int, embedding_dim: int, input_size: int, hidden_size: int, bidirectional: bool, batch_first: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        _, x = self.lstm(self.embedding(x))
        x = x[0].transpose(0, 1)
        assert x.size(1) == 2, 'hidden state (final) should have 1st dim as 2'
        x = torch.cat([x[:, 0, :], x[:, 1, :]], dim=-1)
        return x


CKPT_KEY = 'flava_full'


class DalleConv2d(nn.Module):

    def __init__(self, n_in: int, n_out: int, kw: int) ->None:
        super().__init__()
        w = torch.empty((n_out, n_in, kw, kw), dtype=torch.float32)
        w.normal_(std=1 / math.sqrt(n_in * kw ** 2))
        b = torch.zeros((n_out,), dtype=torch.float32)
        self.w, self.b = nn.Parameter(w), nn.Parameter(b)
        self.kw = kw

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return nn.functional.conv2d(x, self.w, self.b, padding=(self.kw - 1) // 2)


class DalleEncoderBlock(nn.Module):

    def __init__(self, n_in: int, n_out: int, n_layers: int) ->None:
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / n_layers ** 2
        self.id_path = DalleConv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([('relu_1', nn.ReLU()), ('conv_1', DalleConv2d(n_in, n_hid, 3)), ('relu_2', nn.ReLU()), ('conv_2', DalleConv2d(n_hid, n_hid, 3)), ('relu_3', nn.ReLU()), ('conv_3', DalleConv2d(n_hid, n_hid, 3)), ('relu_4', nn.ReLU()), ('conv_4', DalleConv2d(n_hid, n_out, 1))]))

    def forward(self, x: Tensor) ->Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class DalleEncoder(nn.Module):

    def __init__(self, group_count: int=4, n_hid: int=256, n_blk_per_group: int=2, input_channels: int=3, vocab_size: int=8192, **kwargs: Any) ->None:
        super().__init__()
        self.input_channels = input_channels
        n_layers = group_count * n_blk_per_group
        output_conv = DalleConv2d(8 * n_hid, vocab_size, 1)
        self.blocks = nn.Sequential(OrderedDict([('input', DalleConv2d(input_channels, 1 * n_hid, 7)), ('group_1', self._create_group(n_layers, n_blk_per_group, 1 * n_hid, 1 * n_hid)), ('group_2', self._create_group(n_layers, n_blk_per_group, 1 * n_hid, 2 * n_hid)), ('group_3', self._create_group(n_layers, n_blk_per_group, 2 * n_hid, 4 * n_hid)), ('group_4', self._create_group(n_layers, n_blk_per_group, 4 * n_hid, 8 * n_hid, use_pool=False)), ('output', nn.Sequential(OrderedDict([('relu', nn.ReLU()), ('conv', output_conv)])))]))

    def _create_group(self, n_layers: int, n_blk_per_group: int, n_in: int, n_hid: int, use_pool: bool=True) ->nn.Module:
        make_blk = partial(DalleEncoderBlock, n_layers=n_layers)
        blk_range = range(n_blk_per_group)
        blocks: OrderedDict[str, nn.Module] = OrderedDict()
        for i in blk_range:
            if i == 0:
                blocks[f'block_{i + 1}'] = make_blk(n_in, n_hid)
            else:
                blocks[f'block_{i + 1}'] = make_blk(n_hid, n_hid)
        if use_pool:
            blocks['pool'] = nn.MaxPool2d(kernel_size=2)
        return nn.Sequential(blocks)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        return self.blocks(x)


class DalleVAEEncoder(nn.Module):

    def __init__(self, image_size: Union[int, Tuple[int, int]]=112, pretrained: bool=True):
        super().__init__()
        self.image_size = image_size
        self.encoder = DalleEncoder()
        if pretrained:
            self.load_model()

    def load_model(self) ->Any:
        encoder_state_dict = torch.hub.load_state_dict_from_url('https://cdn.openai.com/dall-e/encoder.pkl')
        self.encoder.load_state_dict(encoder_state_dict.state_dict())
        return self.state_dict()

    def get_codebook_indices(self, images: Tensor) ->Tensor:
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images: Tensor) ->Tensor:
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob: Tensor) ->Tensor:
        return self.get_codebook_indices(img_seq_prob)


FLAVAOutput = namedtuple('FLAVAOutput', ['image', 'image_masked', 'text', 'text_masked', 'multimodal', 'multimodal_masked', 'projected_image_embeddings', 'projected_text_embeddings'], defaults=(None, None, None, None, None, None, None, None))


class TransformerOutput(NamedTuple):
    last_hidden_state: Optional[Tensor] = None
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
    image_labels: Optional[Tensor] = None


def gather_tensor(tensor: Tensor, backprop_in_gather: bool=True) ->List[Tensor]:
    """Gathers a tensor across all GPUs.

    Args:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_in_gather (bool): Whether to backpropagate the gradients from
            all_gather to all workers (versus just the local worker). Defaults
            to {\\double back-quote}True{\\double quote}.

    Returns:
        List[Tensor]: List of gathered tensors across all GPUs.
    """
    world_size = torch.distributed.get_world_size()
    if backprop_in_gather:
        return all_gather_with_backprop(tensor)
    else:
        tensor_all_gpus = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather_no_backprop(tensor_all_gpus, tensor)
        tensor_all_gpus[torch.distributed.get_rank()] = tensor
        return tensor_all_gpus


def _gather_embeddings_and_labels(embeddings_a: Tensor, embeddings_b: Tensor, backprop_in_gather: bool=True) ->Tuple[Tensor, Tensor, Tensor]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        return embeddings_a, embeddings_b, labels
    embeddings_a_all_gpus = gather_tensor(embeddings_a, backprop_in_gather)
    embeddings_b_all_gpus = gather_tensor(embeddings_b, backprop_in_gather)
    local_batch_size = embeddings_a.size(0)
    labels = local_batch_size * torch.distributed.get_rank() + torch.arange(local_batch_size, device=embeddings_a.device)
    return torch.cat(embeddings_a_all_gpus), torch.cat(embeddings_b_all_gpus), labels


class ModelOutput(OrderedDict):

    def keys(self) ->Any:
        for field in fields(self):
            yield field.name

    def __getitem__(self, key: Any) ->Any:
        return getattr(self, key)

    def __iter__(self) ->Any:
        yield from self.keys()

    def values(self) ->Any:
        for field in fields(self):
            yield getattr(self, field.name)

    def items(self) ->Any:
        for field in fields(self):
            yield field.name, getattr(self, field.name)


class Pooler(nn.Module):

    def __init__(self, hidden_size: int=768, **kwargs: Any):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) ->Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TwoWayHead(nn.Module):

    def __init__(self, hidden_size: int=768, **kwargs: Any):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: Tensor) ->Tensor:
        return self.seq_relationship(pooled_output)


def assert_labels_are_present(labels: Optional[Tensor], category: str='labels') ->None:
    assert labels is not None, f'Model is in training model but {category} are not passed'


class MaskedPredictionHead(nn.Module):

    def __init__(self, hidden_size: int=768, vocab_size: int=30522, transform_act_fn: Callable[[Tensor], Tensor]=nn.functional.gelu, layer_norm_eps: float=1e-05, use_fp32_layer_norm: bool=True, **kwargs: Any):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm: nn.LayerNorm
        if use_fp32_layer_norm:
            self.layer_norm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) ->Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


FLAVA_FOR_PRETRAINED_MAPPING = {CKPT_KEY: 'https://download.pytorch.org/models/multimodal/flava/flava_for_pretraining_unified_text_encoder.pt'}


FLAVA_MODEL_MAPPING = {CKPT_KEY: 'https://download.pytorch.org/models/multimodal/flava/flava_model_unified_text_encoder.pt'}


def to_2tuple(x: int) ->Tuple[int, int]:
    return x, x


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size: int=224, patch_size: int=16, num_channels: int=3, embed_dim: int=768) ->None:
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, pixel_values: Tensor, interpolate_pos_encoding: bool=False) ->Tensor:
        _, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ImageEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, image_size: int=224, patch_size: int=16, num_channels: int=3, hidden_size: int=768, hidden_dropout_prob: float=0.0, use_image_masking: bool=True) ->None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(image_size=image_size, patch_size=patch_size, num_channels=num_channels, embed_dim=hidden_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)
        if use_image_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.mask_token = None

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) ->Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        npatch = embeddings.shape[1] - 1
        n = self.position_embeddings.shape[1] - 1
        if npatch == n and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_embeddings.patch_size[0]
        w0 = width // self.patch_embeddings.patch_size[1]
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(n)), int(math.sqrt(n)), dim).permute(0, 3, 1, 2), scale_factor=(h0 / math.sqrt(n), w0 / math.sqrt(n)), mode='bicubic', align_corners=False)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: Tensor, image_patches_mask: Optional[Tensor]=None, interpolate_pos_encoding: bool=False) ->Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        _, seq_len, _ = embeddings.size()
        if image_patches_mask is not None:
            if self.mask_token is not None:
                mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
                w = image_patches_mask.unsqueeze(-1).type_as(mask_tokens)
                embeddings = embeddings * (1 - w) + mask_tokens * w
            else:
                warnings.warn('image_patches_mask passed but use_image_masking in init was false. Ignoring.')
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def init_transformer_weights(module: nn.Module, initializer_range: float) ->None:
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ImageTransformer(nn.Module):

    def __init__(self, embeddings: nn.Module, encoder: nn.Module, layernorm: nn.Module, pooler: nn.Module, weight_init_fn: Optional[Callable]=None, initializer_range: float=0.02, **kwargs: Any) ->None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if weight_init_fn is None:
            weight_init_fn = partial(init_transformer_weights, initializer_range=initializer_range)
        self.apply(weight_init_fn)

    def forward(self, pixel_values: Optional[Tensor]=None, image_patches_mask: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None) ->TransformerOutput:
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values, image_patches_mask=image_patches_mask)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask, return_attn_weights=True, return_hidden_states=True)
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return TransformerOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


def flava_image_encoder(hidden_size: int=768, num_attention_heads: int=12, num_hidden_layers: int=12, use_image_masking: bool=False, dropout: float=0.0, intermediate_size: int=3072, intermediate_activation: Callable[..., nn.Module]=nn.GELU, layer_norm_eps: float=1e-12, image_size: int=224, patch_size: int=16, num_channels: int=3) ->ImageTransformer:
    embeddings = ImageEmbeddings(image_size=image_size, patch_size=patch_size, num_channels=num_channels, hidden_size=hidden_size, hidden_dropout_prob=dropout, use_image_masking=use_image_masking)
    encoder = TransformerEncoder(n_layer=num_hidden_layers, d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=intermediate_activation, layer_norm_eps=layer_norm_eps, dropout=dropout, norm_first=True)
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)
    return ImageTransformer(embeddings=embeddings, encoder=encoder, layernorm=layernorm, pooler=pooler)


class FLAVATransformerWithoutEmbeddings(nn.Module):

    def __init__(self, encoder: nn.Module, layernorm: nn.Module, pooler: nn.Module, hidden_size: int=768, weight_init_fn: Optional[Callable]=None, initializer_range: float=0.02, use_cls_token: bool=True, **kwargs: Any):
        super().__init__()
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.cls_token = None
        if weight_init_fn is None:
            weight_init_fn = partial(init_transformer_weights, initializer_range=initializer_range)
        self.apply(weight_init_fn)

    def forward(self, hidden_states: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None) ->TransformerOutput:
        if hidden_states is None:
            raise ValueError('You have to specify hidden_states')
        if self.cls_token is not None:
            batch_size = hidden_states.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        encoder_output = self.encoder(hidden_states, attention_mask=attention_mask, return_hidden_states=True, return_attn_weights=True)
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return TransformerOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


def flava_multimodal_encoder(hidden_size: int=768, num_attention_heads: int=12, num_hidden_layers: int=12, dropout: float=0.0, intermediate_size: int=3072, intermediate_activation: Callable[..., nn.Module]=nn.GELU, layer_norm_eps: float=1e-12) ->FLAVATransformerWithoutEmbeddings:
    encoder = TransformerEncoder(n_layer=num_hidden_layers, d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=intermediate_activation, layer_norm_eps=layer_norm_eps, dropout=dropout, norm_first=True)
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)
    return FLAVATransformerWithoutEmbeddings(encoder=encoder, layernorm=layernorm, pooler=pooler, hidden_size=hidden_size)


class BERTTextEncoder(nn.Module):
    """
    General text transformer encoder with embeddings, following BERT.
    Can be constructed with any user-provided embeddings and encoder.

    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870

    Attributes:
        embeddings (nn.Module): Module that projects text token ids into embeddings.
            See :py:class: `torchmultimodal.modules.layers.text_embedding.BERTTextEmbeddings` for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class:
            `torchmultimodal.modules.layers.transformer.TransformerEncoder` for interface.
        layernorm (nn.Module, optional): Module for layernorm to be applied after encoder. Defaults to ``None``.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: `torchmultimodal.models.flava.transformer.init_transformer_weights`
            as an example. Defaults to ``None``.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape [batch, seq_len]
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere

    Raises:
        ValueError: if input_ids and inputs_embeds are both ``None``.
    """

    def __init__(self, embeddings: nn.Module, encoder: nn.Module, layernorm: Optional[nn.Module]=None, pooler: Optional[nn.Module]=None, weight_init_fn: Optional[Callable]=None) ->None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(self, input_ids: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, token_type_ids: Optional[Tensor]=None, position_ids: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, return_attn_weights: bool=False, return_hidden_states: bool=False) ->TransformerOutput:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError('input_ids or inputs_embeds must not be None')
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            if hasattr(self.embeddings, 'pad_token_id'):
                attention_mask[input_ids == self.embeddings.pad_token_id] = 0
        attention_mask = get_extended_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask, return_attn_weights=return_attn_weights, return_hidden_states=return_hidden_states)
        last_hidden_state = encoder_output.last_hidden_state
        pooled_output = encoder_output.pooler_output
        if self.layernorm:
            last_hidden_state = self.layernorm(last_hidden_state)
        if self.pooler:
            pooled_output = self.pooler(last_hidden_state)
        return TransformerOutput(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


def flava_text_encoder(num_hidden_layers: int=12, hidden_size: int=768, num_attention_heads: int=12, intermediate_size: int=3072, intermediate_activation: Callable[..., nn.Module]=nn.GELU, layer_norm_eps: float=1e-12, dropout: float=0.0, vocab_size: int=30522, pad_token_id: int=0, type_vocab_size: int=2, max_position_embeddings: int=512, initializer_range: float=0.02) ->BERTTextEncoder:
    embeddings = BERTTextEmbeddings(hidden_size=hidden_size, vocab_size=vocab_size, pad_token_id=pad_token_id, type_vocab_size=type_vocab_size, max_position_embeddings=max_position_embeddings, layer_norm_eps=layer_norm_eps, dropout=dropout)
    encoder = TransformerEncoder(n_layer=num_hidden_layers, d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=intermediate_activation, layer_norm_eps=layer_norm_eps, dropout=dropout, norm_first=True)
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)
    weight_init_fn = partial(init_transformer_weights, initializer_range=initializer_range)
    return BERTTextEncoder(embeddings=embeddings, encoder=encoder, layernorm=layernorm, pooler=pooler, weight_init_fn=weight_init_fn)


def load_module_from_url(model: nn.Module, url: str, strict: bool=True, progress: bool=True) ->None:
    local_path = _PATH_MANAGER.get_local_path(url)
    if not torch.cuda.is_available():
        state_dict = torch.load(local_path, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(local_path)
    model.load_state_dict(state_dict, strict=strict)


class FLAVAPreTrainModule(nn.Module):

    def __init__(self, use_bf16: bool=True, **flava_pretraining_kwargs: Any):
        super().__init__()
        self.model = flava_model_for_pretraining(**flava_pretraining_kwargs)
        self.use_bf16 = use_bf16

    def forward(self, batch, action=None):
        if action == 'encode_text':
            return self.model.encode_text(batch)
        elif action == 'encode_image':
            return self.model.encode_image(batch)
        if 'image' in batch and ('text' in batch or 'text_masked' in batch):
            required_embedding = 'mm'
        elif 'image' in batch:
            required_embedding = 'image'
        elif 'text' in batch or 'text_masked' in batch:
            required_embedding = 'text'
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")
        output = self.model(image=batch.get('image'), image_for_codebook=batch.get('image_for_codebook'), image_patches_mask=batch.get('image_patches_mask'), text=batch.get('text'), text_masked=batch.get('text_masked'), mlm_labels=batch.get('mlm_labels'), itm_labels=batch.get('itm_labels'), required_embedding=required_embedding)
        return output

    def encode_text(self, *args, **kwargs):
        return self.model.encode_text(*args, **kwargs)


class BoxLosses(NamedTuple):
    l1_loss: torch.Tensor
    giou_loss: torch.Tensor


class MDETRLoss(nn.Module):

    def __init__(self, soft_token_loss: Callable[..., Tensor], box_losses: Callable[..., BoxLosses], contrastive_alignment_loss: Optional[nn.Module]=None, vqa_losses: Optional[Iterable[Callable[..., Dict[str, Tensor]]]]=None):
        super().__init__()
        self.soft_token_loss = soft_token_loss
        self.box_losses = box_losses
        self.contrastive_alignment_loss = contrastive_alignment_loss
        self.vqa_losses = vqa_losses

    def get_average_num_boxes_across_workers(self, num_boxes: Tensor):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return torch.clamp(num_boxes, min=1).item()
        torch.distributed.all_reduce(num_boxes)
        num_boxes_all_workers = torch.clamp(num_boxes / torch.distributed.get_world_size(), min=1).item()
        return num_boxes_all_workers

    def total_losses_with_weights(self, loss_dict: Dict[str, Tensor], weight_dict: Optional[Dict[str, float]]=None) ->torch.Tensor:
        for k in weight_dict.keys():
            if k not in loss_dict.keys():
                raise ValueError(f'Weight dict contains invalid key {k}')
        return sum([(weight_dict[k] * loss_dict[k]) for k in weight_dict.keys()])

    def forward(self, pred_logits: Tensor, pred_boxes: Tensor, targets: List[Dict[str, Any]], positive_map, indices: List[Tuple[Tensor, Tensor]], contrastive_query_embeddings: Optional[Tensor]=None, contrastive_token_embeddings: Optional[Tensor]=None, tokenized: Optional[Any]=None, vqa_preds: Optional[Dict[str, Tensor]]=None, vqa_labels: Optional[Dict[str, Tensor]]=None, vqa_masks: Optional[Dict[str, Tensor]]=None, weight_dict: Optional[Dict[str, float]]=None) ->Dict[str, Tensor]:
        target_boxes = [t['boxes'] for t in targets]
        target_tokens = [t['tokens_positive'] for t in targets]
        n_target_boxes = [len(t) for t in target_boxes]
        num_boxes = sum(n_target_boxes)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)
        num_boxes_all_workers = self.get_average_num_boxes_across_workers(num_boxes)
        self.pred_logits = pred_logits
        self.n_target_boxes = n_target_boxes
        self.positive_map = positive_map
        self.indices = indices
        self.num_boxes_all_workers = num_boxes_all_workers
        soft_token_loss = self.soft_token_loss(pred_logits, n_target_boxes, positive_map, indices, num_boxes_all_workers)
        box_losses = self.box_losses(pred_boxes, target_boxes, indices, num_boxes_all_workers)
        loss_dict = {'soft_token_loss': soft_token_loss, 'l1_loss': box_losses.l1_loss, 'giou_loss': box_losses.giou_loss}
        if self.contrastive_alignment_loss is not None:
            if contrastive_query_embeddings is None or contrastive_token_embeddings is None or tokenized is None:
                raise ValueError('For contrastive alignment loss must pass contrastive query/token embeddings and tokenized text')
            contrastive_alignment_loss = self.contrastive_alignment_loss(contrastive_query_embeddings, contrastive_token_embeddings, target_tokens, indices, num_boxes_all_workers, tokenized)
            loss_dict.update(contrastive_alignment_loss=contrastive_alignment_loss)
        if self.vqa_losses is not None:
            if vqa_preds is None or vqa_labels is None:
                raise ValueError('For QA loss qa_preds and qa_labels must not be None')
            for vqa_loss in self.vqa_losses:
                loss_dict.update(vqa_loss(vqa_preds, vqa_labels, vqa_masks))
        if weight_dict is not None:
            total_loss = self.total_losses_with_weights(loss_dict, weight_dict)
            loss_dict.update(total_loss=total_loss)
        return loss_dict


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions, while the others are un-matched (and thus treated
    as non-objects). This implementation is based on the MDETR repo:
    https://github.com/ashkamath/mdetr/blob/main/models/matcher.py#L13

    Attributes:
        cost_class (float): Relative weight of the classification error in the
            matching cost. Default: ``1``
        cost_bbox (float): Relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: ``1``
        cost_giou (float): Relative weight of the giou loss of the bounding box in
            the matching cost. Default: ``1``


    Args:
        pred_logits (Tensor): Classification logits.
            Size: (batch_size, num_queries, num_classes)
        pred_boxes (Tensor): Predicted box coordinates.
            Size: (batch_size, num_queries, 4)
        target_boxes_per_sample (List[Tensor]): A list of target bounding boxes.
            Length = batch_size.
            Each element is a tensor of size (n_boxes_for_sample, 4).
        positive_map (Tensor): :math:`	ext{positive_map}[i,j] = 1` when box i maps to class j.
            Size: (total_boxes, num_classes) where total_boxes is the sum of
            n_boxes_for_sample over every sample in the batch.

    Returns:
        A list of size batch_size, containing tuples of ``(index_i, index_j)`` where:
            - ``index_i`` is the indices of the selected predictions (in order)
            - ``index_j`` is the indices of the corresponding selected targets
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    Raises:
        ValueError: If all costs are zero or first dim of target boxes and positive map
            don't match or classification cost and bbox cost shapes don't match.
    """

    def __init__(self, cost_class: float=1, cost_bbox: float=5, cost_giou: float=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError('At least one cost must be nonzero')

    @torch.no_grad()
    def forward(self, pred_logits: Tensor, pred_boxes: Tensor, target_boxes_per_sample: List[Tensor], positive_map: Tensor) ->List[Tuple[Tensor, Tensor]]:
        bs, num_queries = pred_logits.shape[:2]
        target_boxes = torch.cat(target_boxes_per_sample)
        out_prob = F.softmax(pred_logits.flatten(0, 1), dim=-1)
        out_bbox = pred_boxes.flatten(0, 1)
        if target_boxes.size(0) != positive_map.size(0):
            raise ValueError('Total of target boxes should match first dim of positive map')
        cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)
        cost_bbox = torch.cdist(out_bbox, target_boxes, p=1)
        if cost_class.shape != cost_bbox.shape:
            raise ValueError(f"""
            Classification and bounding box cost shapes do not match.
            Classification cost shape: {cost_class.shape},
            Bounding box cost shape: {cost_bbox.shape}
            """)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, in_fmt='cxcywh', out_fmt='xyxy'), box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy'))
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()
        sizes = [x.size(0) for x in target_boxes_per_sample]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TextTokenizer(nn.Module):
    """Converts between text and tokens / embedings

    Wrapper around the tokenizer to be consistent with the API required by
    :py:class:`torchmultimodal.models.gpt.MultimodalGPT`. It also contains the
    embedding layer to enable lookup by token ids.
    """

    def __init__(self, context_len: int, d_model: int, tokenizer: nn.Module) ->None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.encode('[PAD]')[0]
        self.vocab_size = self.tokenizer.vocab_size
        self.context_len = context_len
        self.num_text_tokens = self.vocab_size + context_len
        self.embedding = nn.Embedding(self.num_text_tokens, d_model)

    def text_to_tokens(self, sentences: List[str]) ->Tensor:
        """Pads the sentences to be of equal lengths"""
        tokens = [self.tokenizer.encode(sentence.strip().lower() + ' [SEP]') for sentence in sentences]
        token_ids = [t[:self.context_len] for t in tokens]
        for i, t in enumerate(token_ids):
            t += [self.pad_id] * (self.context_len - len(t))
            token_ids[i] = t
        return torch.Tensor(token_ids).type(torch.int64)

    def encode(self, sentences: List[str], device: str) ->Tensor:
        """Encodes sentences to token ids"""
        token_ids = self.text_to_tokens(sentences)
        unique_pad_ids = torch.arange(self.context_len, device=device) + self.vocab_size
        token_ids = torch.where(token_ids == self.pad_id, unique_pad_ids, token_ids)
        return token_ids

    def _filter_token_ids(self, token_ids: List[int]) ->List[Optional[int]]:
        """Filters out token ids out side of vocab"""
        return [token_id for token_id in token_ids if token_id > 0 and token_id <= self.vocab_size]

    def decode(self, token_ids: Tensor) ->List[str]:
        """Decodes token ids back to sentences"""
        sentences = []
        for _token_ids in token_ids:
            _token_ids = self._filter_token_ids(_token_ids.tolist())
            sentence = self.tokenizer.decode(_token_ids)
            sentences.append(sentence)
        return sentences

    def lookup(self, token_ids: Tensor) ->Tensor:
        return self.embedding(token_ids)


class TextEncoder(nn.Module):
    """Encode tokenized text to the last hidden state representation of the CLS token using
        DistilBERT. DistilBERT prepends a CLS (classification) token to every text so the
        token's hidden state represents the entire text.

    Adapted from MUGEN's text encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Args:
        model_config (Optional[Dict[str, Any]]): model config for DistilBERT.
            Defaults to ``None``, indicating the default DistilBERT config.
        padding_value (int): value that was used to pad the input text.
            Defaults to ``0``, Hugging Face's BERT pad token.

    Inputs:
        input_ids (Tensor): tensor of (batch, text_length) tokenized text

    Returns:
        Tensor: encoded text with dimensions (batch, ``model_config.dim``).
            Default ``model_config.dim`` is ``768``.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]]=None, padding_value: int=0):
        super().__init__()
        self.padding_value = padding_value
        self.target_token_idx = 0
        distilbert_config = DistilBertConfig(**model_config) if model_config else DistilBertConfig()
        self.model = DistilBertModel(config=distilbert_config)
        self.out_dim = self.model.config.dim

    def build_attention_mask(self, input_ids: torch.Tensor) ->torch.Tensor:
        return input_ids != self.padding_value

    def forward(self, input_ids: torch.Tensor) ->torch.Tensor:
        attention_mask = self.build_attention_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class AxialAttentionBlock(nn.Module):
    """Computes multihead axial attention across all dims of the input.

    Axial attention is an alternative to standard full attention, where instead
    of computing attention across the entire flattened input, you compute it for
    each dimension. To capture the global context that full attention does, stacking
    multiple axial attention layers will allow information to propagate among the
    multiple dimensions of the input. This enables attention calculations on high
    dimensional inputs (images, videos) where full attention would be computationally
    expensive and unfeasible. For more details, see `"Axial Attention in
    Multidimensional Transformers (Ho et al. 2019)"<https://arxiv.org/pdf/1912.12180.pdf>`_
    and `"CCNet: Criss-Cross Attention for Semantic Segmentation (Huang et al. 2019)
    "<https://arxiv.org/pdf/1811.11721.pdf>`_.

    Follows implementation by VideoGPT:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        n_dims (int): Dimensionality of input data, not including batch or embedding dims.
        qkv_dim (int): Dimensionality of query/key/value embedding vectors.
        n_head (int): Number of heads in multihead attention. Must divide into ``qkv_dim``
            evenly.
    """

    def __init__(self, n_dims: int, qkv_dim: int, n_head: int) ->None:
        super().__init__()
        self.qkv_dim = qkv_dim
        self.mha_attns = nn.ModuleList([MultiHeadAttention(dim_q=qkv_dim, dim_kv=qkv_dim, n_head=n_head, attn_module=AxialAttention(d), add_bias=False) for d in range(n_dims)])

    def forward(self, x: Tensor) ->Tensor:
        n_channel = x.shape[1]
        if n_channel != self.qkv_dim:
            raise ValueError(f'Input channel dimension is {n_channel}, expected {self.qkv_dim}')
        h = shift_dim(x, 1, -1)
        attn_out = torch.zeros_like(h)
        for mha_attn in self.mha_attns:
            attn_out += mha_attn(h)
        h = attn_out
        h = shift_dim(h, -1, 1)
        return h


def calculate_same_padding(kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], input_shape: Union[Size, Tuple[int, ...]]) ->Tuple:
    """Calculates padding amount on each dimension based on given kernel size and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are halved. If
    stride does not divide into input evenly, then output = ceil(input / stride), following
    the TensorFlow implementation explained here:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        kernel_size (int or Tuple[int, ...]): Size of convolutional kernel.
        stride (int or Tuple[int, ...]): Stride amount of kernel.
        input_shape (Size or Tuple[int, ...]): Shape of input, without batch or channel dimension.

    Returns:
        A tuple of the padding amount in a tuple of tuples for each dimension.
    """
    n_dims = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n_dims))
    if isinstance(stride, int):
        stride = tuple(repeat(stride, n_dims))
    if not len(kernel_size) == len(stride) == len(input_shape):
        raise ValueError('dims for kernel, stride, and input must match')
    total_pad = []
    for k, s, d in zip(kernel_size, stride, input_shape):
        if d % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - d % s, 0)
        total_pad.append(pad)
    pad_input = []
    for p in total_pad[::-1]:
        pad_input.append(p // 2 + p % 2)
        pad_input.append(p // 2)
    pad_input = tuple(pad_input)
    return pad_input


class SamePadConv3d(nn.Module):
    """Performs a same padded convolution on a 3D input.

    This maintains input shape with unit stride, and divides input dims by non-unit stride.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (int): Number of channels in input, same as ``nn.Conv3d``.
        out_channels (int): Number of channels for output, same as ``nn.Conv3d``.
        kernel_size (int or Tuple[int, int, int]): Size of convolutional filter, same as ``nn.Conv3d``.
        stride (int or Tuple[int, int, int], optional): Stride for convolution, same as ``nn.Conv3d``.
        bias (bool, optional): If ``True`` use a bias for convolutional layer or not,
            same as ``nn.Conv3d``. Defaults to ``True``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]]=1, bias: bool=True, **kwargs: Any) ->None:
        super().__init__()
        self.pad_input: Tuple = None
        self.kernel_size = kernel_size
        self.stride = stride
        if 'padding' in kwargs:
            warnings.warn('Padding was specified but will not be used in favor of same padding,                 use Conv3d directly for custom padding')
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size, stride=self.stride, bias=bias, **kwargs)

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(self.kernel_size, self.stride, x.shape[2:])
        return self.conv(F.pad(x, self.pad_input))


class AttentionResidualBlock(nn.Module):
    """Residual block with axial attention.

    Implements the component as proposed in `"VideoGPT: Video Generation using VQ-VAE and
    Transformers (Yan et al. 2022)"<https://arxiv.org/pdf/2104.10157.pdf>`_.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        hidden_dim (int, optional): Size of channel dim of input. Default is ``240``.
        n_head (int, optional): Number of heads in multihead attention. Must divide into hidden_dim evenly.
            Default is ``2``.

    Raises:
        ValueError: If ``hidden_dim`` is less than ``2``.
    """

    def __init__(self, hidden_dim: int=240, n_head: int=2) ->None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError('hidden dim must be at least 2')
        self.block = nn.Sequential(nn.BatchNorm3d(hidden_dim), nn.ReLU(), SamePadConv3d(hidden_dim, hidden_dim // 2, 3, bias=False), nn.BatchNorm3d(hidden_dim // 2), nn.ReLU(), SamePadConv3d(hidden_dim // 2, hidden_dim, 1, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU(), AxialAttentionBlock(3, hidden_dim, n_head))

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        return x + self.block(x)


class VideoEncoder(nn.Module):
    """Encoder for Video VQVAE.

    Stacks specified number of ``SamePadConv3d`` layers
    followed by a stack of ``AttentionResidualBlocks`` and a final ``SamePadConv3d``
    layer before the codebook. The residual blocks use Axial Attention to enhance
    representations of video data without significantly increasing computational
    cost.

    Follows VideoGPT's implementation:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channel_dims (Tuple[int, ...]): Input channel dimension for each layer in conv stack.
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): Kernel sizes for each layer in conv stack.
        strides (Tuple[Tuple[int, int, int], ...]): Strides for each layer in conv stack.
        output_dim (int): Size of hidden dimension of final output.
        n_res_layers (int, optional): Number of ``AttentionResidualBlocks`` to include. Default is ``4``.
        attn_hidden_dim (int, optional): Size of hidden dimension in attention block. Default is ``240``.
        kwargs (Any): Keyword arguments to be passed into ``SamePadConv3d`` and used by ``nn.Conv3d``.

    Raises:
        ValueError: If the lengths of ``in_channel_dims``, ``kernel_sizes``, and ``strides`` are not
            all equivalent.
    """

    def __init__(self, in_channel_dims: Tuple[int, ...], kernel_sizes: Tuple[Tuple[int, int, int], ...], strides: Tuple[Tuple[int, int, int], ...], output_dim: int, n_res_layers: int=4, attn_hidden_dim: int=240, **kwargs: Any):
        super().__init__()
        assert_equal_lengths(in_channel_dims, kernel_sizes, strides, msg='in_channel_dims, kernel_sizes, and strides must be same length.')
        convolutions: List[nn.Module] = []
        n_conv_layers = len(in_channel_dims)
        for i in range(n_conv_layers):
            in_channel = in_channel_dims[i]
            out_channel = in_channel_dims[i + 1] if i < n_conv_layers - 1 else attn_hidden_dim
            kernel = kernel_sizes[i]
            stride = strides[i]
            convolutions.append(SamePadConv3d(in_channel, out_channel, kernel, stride, bias=True, **kwargs))
            if i < n_conv_layers - 1:
                convolutions.append(nn.ReLU())
        self.convs = nn.Sequential(*convolutions)
        self.res_stack = nn.Sequential(*[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)], nn.BatchNorm3d(attn_hidden_dim), nn.ReLU())
        self.conv_out = SamePadConv3d(attn_hidden_dim, output_dim, kernel_size=1, stride=1)

    def get_latent_shape(self, input_shape: Union[Tuple, Size]) ->Tuple:
        """Return shape of encoder output based on number of downsampling conv layers"""
        latent_shape = list(input_shape)
        for layer in self.convs:
            if isinstance(layer, SamePadConv3d):
                latent_shape = [(latent_shape[dim] // layer.conv.stride[dim]) for dim in range(len(input_shape))]
        return tuple(latent_shape)

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x (Tensor): Input video data with shape ``(b, c, d1, d2, d3)``.
        """
        in_channel = x.shape[1]
        if in_channel != self.convs[0].conv.in_channels:
            raise ValueError(f'expected input channel dim to be {self.convs[0].conv.in_channels}, but got {in_channel}')
        h = self.convs(x)
        h = self.res_stack(h)
        h = self.conv_out(h)
        return h


class Projection(nn.Module):
    """Project embeddings to a fixed dimension by adding the hidden-layer output and final output of a MLP.

    Args:
        in_dim (int): dimension of input.
        out_dim (int): dimension of output.
            Defaults to ``256``, the value used by MUGEN.
        dropout_prob (float): dropout probability.
            Defaults to ``0.1``, the value used by MUGEN.

    Inputs:
        x (Tensor): embeddings (batch, dim_in)

    Returns:
        Tensor: projected embeddings (batch, dim_out)

    """

    def __init__(self, in_dim, out_dim=256, dropout_prob=0.1) ->None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.drop = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.gelu(embed1)
        embed2 = self.linear2(embed2)
        embed2 = self.drop(embed2)
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class RandomMixup(torch.torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float=0.5, alpha: float=1.0, inplace: bool=False) ->None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f'Please provide a valid positive value for the num_classes. Got num_classes={num_classes}')
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (..., H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim < 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return batch, target
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return batch, target

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha}, inplace={self.inplace})'
        return s


class RandomCutmix(torch.torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float=0.5, alpha: float=1.0, inplace: bool=False) ->None:
        super().__init__()
        if num_classes < 1:
            raise ValueError('Please provide a valid positive value for the num_classes.')
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (..., H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim < 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return batch, target
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        h, w = batch.shape[-2:]
        r_x = torch.randint(w, (1,))
        r_y = torch.randint(h, (1,))
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * w)
        r_h_half = int(r * h)
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=w))
        y2 = int(torch.clamp(r_y + r_h_half, max=h))
        batch[..., y1:y2, x1:x2] = batch_rolled[..., y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (w * h))
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return batch, target

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha}, inplace={self.inplace})'
        return s


class Unsqueeze(torch.torch.nn.Module):

    def __init__(self, pos=0):
        super().__init__()
        self.pos = pos

    def forward(self, x):
        return x.unsqueeze(self.pos)


class ConvertTCHWtoCTHW(torch.nn.Module):
    """Convert tensor from (T, C, H, W) to (C, T, H, W)"""

    def forward(self, vid: torch.Tensor) ->torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class DropChannels(torch.nn.Module):
    """
    Drops Channels with predefined probability values.
    Pads the dropped channels with `pad_value`.
    Channels can be tied using `tie_channels`
    For example, for RGBD input, RGB can be tied by using `tie_channels=[0,1,2]`.
    In this case, channels [0,1,2] will be dropped all at once or not at all.
    Assumes input is of the form CxHxW or TxCxHxW
    """

    def __init__(self, channel_probs, fill_values, tie_channels=None, all_channel_drop=False):
        """
        channel_probs: List of probabilities
        fill_values: List of values to fill the dropped channels with
        tie_channels: List of indices. Tie dropping of certain channels.
        all_channel_drop: Bool variable to prevent cases where all channels are dropped.
        """
        super().__init__()
        channel_probs = np.array(channel_probs, dtype=np.float32)
        self.channel_probs = channel_probs
        self.fill_values = fill_values
        self.tie_channels = tie_channels
        self.all_channel_drop = all_channel_drop
        if tie_channels is not None:
            tie_probs = [channel_probs[x] for x in tie_channels]
            assert len(set(tie_probs)) == 1, 'All tie_channel probs must be equal'

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        if x.ndim == 3:
            num_channels = x.shape[0]
            channel_index = 0
        elif x.ndim == 4:
            num_channels = x.shape[1]
            channel_index = 1
        else:
            raise ValueError(f'Unexpected number of dims {x.ndim}. Expected 3 or 4.')
        assert num_channels == len(self.channel_probs), f'channel_probs is {len(self.channel_probs)} but got {num_channels} channels'
        to_drop = [(np.random.random() < self.channel_probs[c]) for c in range(num_channels)]
        if self.tie_channels is not None:
            first_drop = to_drop[self.tie_channels[0]]
            for idx in self.tie_channels[1:]:
                to_drop[idx] = first_drop
        if all(to_drop) and self.all_channel_drop is False:
            to_drop = [(False) for _ in range(num_channels)]
        for c in range(num_channels):
            if not to_drop[c]:
                continue
            if channel_index == 0:
                x[c, ...] = self.fill_values[c]
            elif channel_index == 1:
                x[:, c, ...] = self.fill_values[c]
            else:
                raise NotImplementedError()
        return x


class DepthNorm(torch.nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(self, max_depth: float, clamp_max_before_scale: bool=False, min_depth: float=0.01):
        """
        Args:
            max_depth (float): The max value of depth for the dataset
            clamp_max (bool): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        if max_depth < 0.0:
            raise ValueError('max_depth must be > 0; got %.2f' % max_depth)
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def forward(self, image: torch.Tensor):
        c, h, w = image.shape
        if c != 4:
            err_msg = f'This transform is for 4 channel RGBD input only; got {image.shape}'
            raise ValueError(err_msg)
        color_img = image[:3, ...]
        depth_img = image[3:4, ...]
        depth_img = depth_img.clamp(min=self.min_depth)
        if self.clamp_max_before_scale:
            depth_img = depth_img.clamp(max=self.max_depth)
        depth_img /= self.max_depth
        img = torch.cat([color_img, depth_img], dim=0)
        return img


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device='cpu'):

        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=True)


class DummyEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(8, 2, batch_first=True), num_layers=1, norm=nn.LayerNorm(8))

    def forward(self, x):
        return self.transformer(x)[:, 0, :]


class ALBEFVisionEncoder(nn.Module):
    """
    Modified VisionTransformer used by ALBEF.

    This class returns the output of the encoder ('encoder.ln'), without passing it to the heads.

    Args:
        image_size (int): The size (resolution) of each image.
            Default is 256.
        patch_size (int) The size (resolution) of each patch.
            Default is 16.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            Default is 12.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            Default is 12.
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
            Default is 768.
        mlp_dim (int): Dimensionality of the MLP Block in the encoder layers.
            Default is 3072.
        dropout (float): The dropout ratio for the encoder probabilities.
            Default is 0.
        attention_dropout (float): The dropout ratio for the attention probabilities.
            Default is 0.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
            Default is 1e-6.

    Inputs:
        x (Tensor): Tensor of size (n, c, image_size, image_size) containing image features
    """

    def __init__(self, image_size: int=256, patch_size: int=16, num_hidden_layers: int=12, num_attention_heads: int=12, hidden_size: int=768, mlp_dim: int=3072, dropout: float=0.0, attention_dropout: float=0.0, layer_norm_eps: float=1e-06) ->None:
        super().__init__()
        vision_transformer = VisionTransformer(image_size, patch_size, num_hidden_layers, num_attention_heads, hidden_size, mlp_dim, dropout, attention_dropout, norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps))
        self.encoder_layer_name = 'encoder.ln'
        self.encoder = create_feature_extractor(vision_transformer, [self.encoder_layer_name])

    def forward(self, x: Tensor) ->Tensor:
        return self.encoder(x)[self.encoder_layer_name]


class SiLU(nn.Module):
    """Sigmoid Linear Unit

    .. math:: \\text{SiLU}(x) = x * \\sigma(1.702 * x)

    where :math:`\\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU for greater forward speed. Note that this is different from
    ``torch.nn.SiLU`` by the coefficient ``1.702`` from the paper:
    `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x: Tensor) ->Tensor:
        return torch.sigmoid(1.702 * x) * x


class CLIPViTEncoder(nn.Module):
    """
    Vision transformer encoder for CLIP.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        patch_size (int): The dimension of each patch
        image_size(int): The size (width==height) of input image
        width (int): Dimensionality of the encoder layers and the pooler layer
        heads (int): Number of attention heads for each attention layer in the Transformer encoder
        layers (int): Number of hidden layers in the Transformer encoder

    Inputs:
        x (Tensor): image tensor with dimensions B x C(3) x image_size x image_size
    """

    def __init__(self, embedding_dim: int, patch_size: int, image_size: int, width: int, heads: int, layers: int):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.conv = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.image_size = image_size
        scale = width ** -0.5
        self.cls_token_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = Fp32LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dropout=0.0, activation=SiLU(), norm_first=True, dim_feedforward=4 * width, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_post = Fp32LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, embedding_dim))

    def forward(self, x: Tensor) ->Tensor:
        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(f'Expected input with width and height as {self.image_size}, found {x.size(2)} by {x.size(3)} ')
        if x.size(1) != 3:
            raise ValueError(f'Expected 3 channels found {x.size(1)}')
        x = self.conv(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.encoder(x)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.projection
        return x


EXPANSION = 4


class ResNetForCLIPBottleneck(nn.Module):

    def __init__(self, inplanes: int, planes: int, stride: int=1):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * EXPANSION, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * EXPANSION)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * EXPANSION:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * EXPANSION, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * EXPANSION))]))

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: Tensor) ->Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ResNetForCLIP(nn.Module):
    """Modified ResNet used by CLIP.

    Based on https://github.com/openai/CLIP/blob/main/clip/model.py#L93, this class
    differs from Torchvision's ResNet in the following ways:
    - There are now 3 "stem" convolutions as opposed to 1, with an
        average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
        prepended to convolutions with stride > 1.
    - The final pooling layer is a QKV attention instead of an average pool.

    Args:
        layers (Tuple[int]):
        output_dim (int): dimension of output tensor
        heads (int): number of heads in the attention pooling layer
        input_resolution (int): resolution of image input to encoder
        width (int): ResNet width
        use_clip_init (bool): Whether to use CLIP-specific initialization.

    Inputs:
        x (Tensor): Tensor containing image features
    """

    def __init__(self, layers: Tuple[int, int, int, int]=(3, 4, 6, 3), output_dim: int=512, heads: int=1024, input_resolution: int=224, width: int=64, use_clip_init: bool=True):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        if use_clip_init:
            self.initialize_parameters()

    def _make_layer(self, planes: int, blocks: int, stride: int=1) ->nn.Module:
        layers = [ResNetForCLIPBottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * EXPANSION
        for _ in range(1, blocks):
            layers.append(ResNetForCLIPBottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def initialize_parameters(self) ->None:
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith('bn3.weight'):
                    nn.init.zeros_(param)

    def forward(self, x: Tensor) ->Tensor:

        def stem(x: Tensor) ->Tensor:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class CLIPOutput(NamedTuple):
    embeddings_a: torch.Tensor
    embeddings_b: torch.Tensor


class CLIP(nn.Module):
    """CLIP is a model for contrastive pretraining between two modalities.

    CLIP (https://arxiv.org/pdf/2103.00020.pdf) jointly trains an image encoder
    (either ResNet or ViT) and a text encoder (Transformer) to predict correct
    (image, text) pairings via a contrastive loss function. This module contains the
    encoders, while the loss is implemented in ContrastiveLossWithTemperature.


    Args:   encoder_a (nn.Module): Instantiated encoder for modality A.
                See e.g. ResNetForCLIP class.
            encoder_b (nn.Module): Instantiated encoder for modality B.
                See e.g. CLIPTextEncoder class.

    Inputs: features_a (Tensor): Tensor containing features of modality A.
            features_b (Tensor): Tensor containing features of modality B.
    """

    def __init__(self, encoder_a: nn.Module, encoder_b: nn.Module):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) ->CLIPOutput:
        embeddings_a = self.encoder_a(features_a)
        embeddings_b = self.encoder_b(features_b)
        embeddings_a = F.normalize(embeddings_a)
        embeddings_b = F.normalize(embeddings_b)
        return CLIPOutput(embeddings_a=embeddings_a, embeddings_b=embeddings_b)


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder class. Should be instantiated and passed to
    CLIP (models/clip.py)

    As in CLIP, the text encoder follows a Transformer architecture.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        context_length (int): Maximum sequence length for Transforer.
        vocab_size (int): Vocab size.
        width (int): Embedding dimension for Transformer encoder.
        heads (int): Number of heads in Transformer encoder.
        layers (int): Number of layers in Transformer encoder.
        use_clip_init (bool): Whether to use CLIP-specific initialization.

    Inputs: text (Tensor): Tensor containing text features.
    """
    TOKEN_EMBEDDING_INIT_STD = 0.02
    POS_EMBEDDING_INIT_STD = 0.01

    def __init__(self, embedding_dim: int=512, context_length: int=77, vocab_size: int=49408, width: int=512, heads: int=8, layers: int=12, use_clip_init: bool=True):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.token_embedding = torch.nn.Embedding(vocab_size, width)
        self.positional_embedding = torch.nn.Parameter(torch.empty(context_length, width))
        encoder_layer = TransformerEncoderLayer(d_model=width, nhead=heads, dropout=0.0, activation=SiLU(), norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=layers)
        self.width = width
        self.context_length = context_length
        self.ln_final = Fp32LayerNorm(width)
        self.projection = nn.Linear(width, embedding_dim, bias=False)
        if use_clip_init:
            self.initialize_parameters()

    def initialize_parameters(self) ->None:
        nn.init.normal_(self.token_embedding.weight, std=self.TOKEN_EMBEDDING_INIT_STD)
        nn.init.normal_(self.positional_embedding, std=self.POS_EMBEDDING_INIT_STD)
        proj_std = self.width ** -0.5 * (2 * self.encoder.num_layers) ** -0.5
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for layer in self.encoder.layers:
            nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(layer.linear1.weight, std=fc_std)
            nn.init.normal_(layer.linear2.weight, std=proj_std)
        nn.init.normal_(self.projection.weight, std=self.width ** -0.5)

    def build_attention_mask(self) ->Tensor:
        mask = torch.full((self.context_length, self.context_length), float('-inf')).triu(1)
        return mask

    def forward(self, text: Tensor) ->Tensor:
        if text.size(1) != self.context_length:
            raise ValueError(f'length of input should be {self.context_length} but found {text.size(1)}')
        embeddings = self.token_embedding(text)
        embeddings = embeddings + self.positional_embedding
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.encoder(embeddings, mask=self.build_attention_mask())
        embeddings = torch.permute(embeddings, (1, 0, 2))
        embeddings = self.ln_final(embeddings)
        embeddings = self.projection(embeddings[torch.arange(embeddings.shape[0]), text.argmax(dim=-1)])
        return embeddings


class ImageTransformerWithVAE(nn.Module):

    def __init__(self, image_transformer: nn.Module, vae: nn.Module, **kwargs: Dict[str, Any]) ->None:
        super().__init__()
        self.image_transformer = image_transformer
        self.vae = vae

    def forward(self, pixel_values: Optional[Tensor]=None, image_patches_mask: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None) ->TransformerOutput:
        image_labels = self.vae(pixel_values).flatten(1)
        image_patches_mask = image_patches_mask.flatten(1)
        image_labels[image_patches_mask == False] = -1
        output = self.image_transformer(pixel_values=pixel_values, image_patches_mask=image_patches_mask, attention_mask=attention_mask)
        return TransformerOutput(last_hidden_state=output.last_hidden_state, pooler_output=output.pooler_output, hidden_states=output.hidden_states, attentions=output.attentions)


class TransformerDecoderOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.gpt.TransformerDecoder`.

    Attributes:
        last_hidden_states (Tensor): Output from the last layer of the transformer.
        hidden_states (Tuple[Tensor, ...], optional): Outputs from all layers of the transformer.
            Defaults to ``None``.
        attention_weights (Tuple[Tensor, ...], optional): Attention probabilities from all layers of the
            transformer. Defaults to ``None``.
        past_key_values (Tuple[Dict[str, Tensor], ...]], optional): If ``use_cache`` is on, contains
            key/value tensors prior to the current step along the sequence. Defaults to ``None``.
    """
    last_hidden_states: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attention_weights: Optional[Tuple[Tensor, ...]] = None
    past_key_values: Optional[Tuple[Dict[str, Tensor], ...]] = None


class MultimodalGPTOutput(NamedTuple):
    """Outputs from :meth:`~torchmultimodal.models.gpt.MultimodalGPT.forward`.

    Attributes:
        decoder_output (TransformerDeocoderOutput): Contains output from the multimodal transformer decoder.
            See :class:`MultimodalTransformerDecoder`.
        logits (Tensor): Logits computed from the last hidden state of the multimodal transformer decoder.
    """
    decoder_output: TransformerDecoderOutput
    logits: Tensor


class MultimodalGPT(nn.Module):
    """Extends the GPT (Generative Pre-Training) model for cross-modality generation.

    This module implements the GPT model for generation of one modality given another
    following the paper `"Improving Language Understanding by Generative Pre-Training
    "<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>`_.

    Args:
        d_model (int): Embedding dimension of the transformer decoder.
        num_in_tokens (int): Number of unique token states for the input modality.
        num_out_tokens (int): Number of unique token states for the output modality.
        latent_shape ([Tuple[int, ...]): Shape of the latent space of the output modality tokenizer. Used to reshape
            sequence of generated tokens to be decoded back to data.
        in_tokenizer (nn.Module): Tokenizer for the input modality. Must have methods ``encode``, ``lookup``.
        out_tokenizer (nn.Module): Tokenizer for the output modality. Must have methods ``encode``, ``decode``.
        mm_decoder (nn.Module): Multimodal transformer decoder. An instace of
            :py:class:`MultimodalTransformerDecoder`.
        in_projection (nn.Module, optional): Projects the input modality token embeddings to match size of the
            transformer decoder. Defaults to ``None``.
        out_projection (nn.Module, optional): Projects the output modality token embeddings to match size of the
            transformer decoder. Defaults to ``None``.
        norm_layer (Callable[..., nn.Module], optional): Which normalization layer to use. Supports ``nn.Module`` or
            partial. If ``None``, ``nn.LayerNorm`` will be used as the default.
        use_gpt_init (bool): Whether to use GPT model specific initialization. Defaults to ``True``.

    Raises:
        AttributeError: If input tokenizer does not implement methods ``encode`` and ``lookup`` or if output
        tokenizer does not implement methods ``encode``, ``lookup`` and ``decode``.
    """

    def __init__(self, d_model: int, num_in_tokens: int, num_out_tokens: int, latent_shape: Tuple[int, ...], in_tokenizer: nn.Module, out_tokenizer: nn.Module, mm_decoder: nn.Module, in_projection: Optional[nn.Module]=None, out_projection: Optional[nn.Module]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, use_gpt_init: bool=True) ->None:
        super().__init__()
        if not all([hasattr(in_tokenizer, attr_name) for attr_name in ['encode', 'lookup']]):
            raise AttributeError("Input modality tokenizer must have methods 'encode' and 'lookup'.")
        if not all([hasattr(out_tokenizer, attr_name) for attr_name in ['encode', 'lookup', 'decode']]):
            raise AttributeError("Output modality tokenizer must have methods 'encode', 'lookup' and 'decode'.")
        num_tokens = num_in_tokens + num_out_tokens
        self.num_in_tokens = num_in_tokens
        self.num_out_tokens = num_out_tokens
        self.latent_shape = latent_shape
        self.in_tokenizer = in_tokenizer
        self.out_tokenizer = out_tokenizer
        self.mm_decoder = mm_decoder
        self.in_projection = in_projection
        self.out_projection = out_projection
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-05)
        self.norm = norm_layer(normalized_shape=d_model)
        self.to_logit = nn.Linear(d_model, num_tokens, bias=False)
        self.to_logit.weight.data.copy_(torch.zeros(num_tokens, d_model))
        if use_gpt_init:
            self.initialize_parameters()

    def initialize_parameters(self) ->None:
        if hasattr(self.in_projection, 'weight'):
            self.in_projection.weight.data.normal_(std=0.02)
        if hasattr(self.out_projection, 'weight'):
            self.out_projection.weight.data.normal_(std=0.02)

    def forward(self, in_tokens: Optional[Tensor]=None, out_tokens: Optional[Tensor]=None, in_pos_ids: Optional[Tensor]=None, out_pos_ids: Optional[Tensor]=None, attn_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, logits_mask: Optional[Tensor]=None, use_cache: bool=False, causal: bool=False, right_shift: bool=False, return_attn_weights: bool=False, return_hidden_states: bool=False) ->MultimodalGPTOutput:
        """
        Args:
            in_tokens (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing tokens
                for the input modality. Defaults to ``None``.
            out_tokens (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing tokens
                for the output modality. Defaults to ``None``.
            in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
                input modality position embeddings. Defaults to ``None``.
            out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
                output modality position embeddings. Defaults to ``None``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            logits_mask (Tensor, optional): Tensor of dimension ``(seq_len, num_tokens)`` or
                ``(b, seq_len, num_tokens)`` to ensure we only calculate probabilities from tokens of the
                corresponding modality sequence.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
                recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            right_shift (bool): If ``True``, shifts the embedding vectors to the right and prepends it with start of
                sentence token. Defaults to ``False``. This option is disregarded during training mode
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instance of :class:`~torchmultimodal.models.gpt.MultimodalGPTOutput`.
        """
        decoder_output = self.fwd(in_tokens=in_tokens, out_tokens=out_tokens, in_pos_ids=in_pos_ids, out_pos_ids=out_pos_ids, attn_mask=attn_mask, head_mask=head_mask, use_cache=use_cache, causal=causal, right_shift=right_shift, return_attn_weights=return_attn_weights, return_hidden_states=return_hidden_states)
        hidden_states = decoder_output.last_hidden_states
        logits = self.logit_projection(hidden_states, logits_mask)
        return MultimodalGPTOutput(decoder_output, logits)

    def fwd(self, in_tokens: Optional[Tensor]=None, out_tokens: Optional[Tensor]=None, in_pos_ids: Optional[Tensor]=None, out_pos_ids: Optional[Tensor]=None, attn_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, use_cache: bool=False, causal: bool=False, right_shift: bool=False, return_attn_weights: bool=False, return_hidden_states: bool=False) ->TransformerDecoderOutput:
        if in_tokens is None and out_tokens is None:
            raise ValueError('input-modality token and output-modality token sequences cannot be both empty')
        in_modality = out_modality = None
        if in_tokens is not None:
            in_modality = self.lookup(in_tokens, 'in')
            if self.in_projection is not None:
                in_modality = self.in_projection(in_modality)
        if out_tokens is not None:
            out_modality = self.lookup(out_tokens, 'out')
            if self.out_projection is not None:
                out_modality = self.out_projection(out_modality)
        return self.mm_decoder(in_modality=in_modality, out_modality=out_modality, in_pos_ids=in_pos_ids, out_pos_ids=out_pos_ids, attn_mask=attn_mask, head_mask=head_mask, use_cache=use_cache, causal=causal, right_shift=right_shift, return_attn_weights=return_attn_weights, return_hidden_states=return_hidden_states)

    def logit_projection(self, hidden_states: Tensor, logits_mask: Optional[Tensor]=None) ->Tensor:
        if logits_mask is not None and logits_mask.dim() == 2:
            logits_mask = logits_mask.unsqueeze(0)
        hidden_states = self.norm(hidden_states)
        logits = self.to_logit(hidden_states)
        max_neg_value = -torch.finfo(logits.dtype).max
        if logits_mask is not None:
            logits.masked_fill_(logits_mask == 0, max_neg_value)
        return logits

    def encode(self, x: Any, modality: str, **kwargs: Any) ->Tensor:
        """Converts data to token ids.

        Although this is not part of the forward pass, it is used to generate labels for training
        as well as inputs for autoregressive decoding.

        Args:
            x (Any): Data to be encoded, e.g., ``List[str]`` for text, ``Tensor`` of shape
                ``(b, c, d1, ..., dn)`` for audio/image/video.
            modality (str): Input or output modality string used to select the encoder.
            kwargs (Any): Other keyword arguments suitable for the encoder.

        Returns:
            A tensor of token ids of shape ``(b, seq_len)``.

        Raises:
            ValueError: If ``modality`` is neither ``in`` nor ``out``.
        """
        if modality == 'in':
            encoder = self.in_tokenizer.encode
        elif modality == 'out':
            encoder = self.out_tokenizer.encode
        else:
            raise ValueError(f'Invalid modality parameter: {modality}')
        token_ids = encoder(x, **kwargs)
        return token_ids.flatten(start_dim=1, end_dim=-1)

    def decode(self, token_ids: Tensor, **kwargs: Any) ->Any:
        """Converts out-modality tokens ids back to data during generation.

        Args:
            token_ids (Tensor): Token ID sequence ``(b, seq_len)`` to be decoded.
            kwargs (Any): Other keywords arguments suitable for the decoder.

        Returns:
            The decoded data, e.g., ``List[str]`` for text, a tensor of shape ``(b, c, d1. ,,, dn)`` for
                audio/image/video.

        Raises:
            ValueError: If the shape of ``token_ids`` is not of dimension two.
            ValueError: If the sequence dim of ``token_ids`` does not match that inferred from ``latent_shape``.
        """
        if len(token_ids.shape) != 2:
            raise ValueError(f"Shape of token ids should be '(batch_size, sequence_length)' but got {token_ids.shape}")
        latent_seq_len = torch.prod(torch.tensor(self.latent_shape)).item()
        if token_ids.shape[1] != latent_seq_len:
            raise ValueError(f'Sequence to decode does not match that inferred from the tokenizer: {latent_seq_len}')
        token_ids = token_ids.view(token_ids.shape[0], *self.latent_shape)
        return self.out_tokenizer.decode(token_ids, **kwargs)

    def lookup(self, token_ids: Tensor, modality: str) ->Tensor:
        """Looks up the latent embeddings corresponding to the token ids during generation.

        We ask each tokenizer to implement this method. An example is :class:`torchmultimodal.models.vqvae.VQVAE`.

        Args:
            token_ids (Tensor): Token ID sequence ``(b, seq_len)``.
            modality (str): The modality at which this method is performed.

        Returns:
            A tensor of embeddings corresponding to the token ids.

        Raises:
            ValueError: If ``modality`` is neither ``in`` nor ``out``.
        """
        if modality == 'in':
            tokenizer = self.in_tokenizer
        elif modality == 'out':
            tokenizer = self.out_tokenizer
        else:
            raise ValueError(f'Invalid modality parameter: {modality}')
        return tokenizer.lookup(token_ids)


class MultimodalTransformerDecoder(nn.Module):
    """A transformer decoder for two modalities

    The token- and position- embedding layers are per modality:
        * During training both modalities are fed into the module and concatenated as a single sequence of
            tokenized embedding vectors
        * During generation the future data points are predicted step-wise from the past. The input modality
            is processed before the output modality (see ``torchmultimodal.utils.common.generate``). Therefore,
            at any point in time the input data contains only one modality.

    Args:
        in_pos_emb (nn.Module): Input modality position embedding layer.
        out_pos_emb (nn.Module): Output modality position embedding layer.
        decoder (nn.Module): The transformer decoder. An instance of :py:class:`TransformerDecoder`.
        right_shift (nn.Module): Layer that shifts the embedding vectors to the right and prepends it with
            start of sentence token (SOS). An instance of :py:class:`RightShift`.

    Note:
        * During training mode, the SOS token is prepended to the left of the concatenated input and
            output modality sequence;
        * During generation mode, the SOS token is only required for the input modality sequence as
            the initial token to be learnt from. Right shift should be turned off
            (``right_shift = False``, see args) when we start to generate the output modality samples.
    """

    def __init__(self, in_pos_emb: nn.Module, out_pos_emb: nn.Module, decoder: nn.Module, right_shift: nn.Module) ->None:
        super().__init__()
        self.in_pos_emb = in_pos_emb
        self.out_pos_emb = out_pos_emb
        self.decoder = decoder
        self.right_shift = right_shift

    def forward(self, in_modality: Optional[Tensor]=None, out_modality: Optional[Tensor]=None, in_pos_ids: Optional[Tensor]=None, out_pos_ids: Optional[Tensor]=None, attn_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, use_cache: bool=False, causal: bool=False, right_shift: bool=False, return_attn_weights: bool=False, return_hidden_states: bool=False) ->TransformerDecoderOutput:
        """
        Args:
            in_modality (Tensor, optional): Tensor of dimension ``(b, in_seq_len, d_model)`` containing tokenized
                embeddings for the input modality. Defaults to ``None``.
            out_modality (Tensor, optional): Tensor of dimension ``(b, out_seq_len, d_model')`` containing tokenized
                embeddings for the output modality. Defaults to ``None``.
            in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
                input modality position embeddings. Defaults to ``None``.
            out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
                output modality position embeddings. Defaults to ``None``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding.
                If ``False``, recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            right_shift (bool): If ``True``, shifts the embedding vectors to the right and prepends it with start of
                sentence token. Defaults to ``False``. This option is disregarded during training mode
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instace of :class:`~torchmultimodal.models.gpt.TransformerDecoderOutput`.
        """
        if in_modality is None and out_modality is None:
            raise ValueError('in_modality and out_modality sequences cannot be both empty')
        if in_modality is None:
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids)
            x = out_modality + self.out_pos_emb(out_pos_ids)
        elif out_modality is None:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids)
            x = in_modality + self.in_pos_emb(in_pos_ids)
        else:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids)
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids)
            x_in = in_modality + self.in_pos_emb(in_pos_ids)
            x_out = out_modality + self.out_pos_emb(out_pos_ids)
            x = torch.cat((x_in, x_out), dim=1)
        if self.training or right_shift:
            x = self.right_shift(x)
        return self.decoder(x, attn_mask, head_mask, use_cache, causal, return_attn_weights, return_hidden_states)

    def _norm_pos_ids(self, x: Tensor, pos_ids: Optional[Tensor]=None) ->Tensor:
        _, seq_len, _ = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)[None, :]
        if pos_ids.shape[1] != seq_len:
            raise ValueError(f'Input sequence and position ids must be equal in length: {pos_ids.shape[1]} != {seq_len}')
        return pos_ids


def get_clones(module: nn.Module, n: int) ->nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    A transformer decoder.

    Args:   decoder_layer (nn.Module): Module for an individual decoder layer.
            num_layers (int): Number of decoder layers.
            norm (Optional[nn.Module]): Normalization applied after last decoder layer.
                Default: None
            return_intermediate (bool): Whether to return intermediate decoder outputs.
                Default: True

    Inputs: tgt (Tensor): The sequence to the decoder layer.
            memory (Tensor): The sequence from the last layer of the decoder.
            tgt_mask (Optional[Tensor]) The mask for the tgt sequence. Default: None
            memory_mask (Optional[Tensor]): The mask for the memory sequence.
                Default: None
            tgt_key_padding_mask (Optional[Tensor]): The mask for the tgt keys per batch.
                Default: None
            memory_key_padding_mask (Optional[Tensor]):
            pos (Optional[Tensor]): Positional embeddings applied to Q and K
                self-attention matrices. Default: None
            query_pos (Optional[Tensor]): Positional embeddings applied to Q
                cross-attention matrix. Default: None
    """

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module]=None, return_intermediate: bool=True):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None) ->Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        if self.norm is not None:
            return self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    A single layer from a transformer decoder.

    Args:   d_model (int): Number of features in the input.
            num_heads (int): Number of heads in multi-head attention.
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., nn.Module]): The activation function of the
                intermediate layer. Default: nn.ReLU

    Inputs: tgt (Tensor): The sequence to the decoder layer.
            memory (Tensor): The sequence from the last layer of the decoder.
            tgt_mask (Optional[Tensor]) The mask for the tgt sequence. Default: None
            memory_mask (Optional[Tensor]): The mask for the memory sequence.
                Default: None
            tgt_key_padding_mask (Optional[Tensor]): The mask for the tgt keys per batch.
                Default: None
            memory_key_padding_mask (Optional[Tensor]):
            pos (Optional[Tensor]): Positional embeddings applied to Q and K
                self-attention matrices. Default: None
            query_pos (Optional[Tensor]): Positional embeddings applied to Q
                cross-attention matrix. Default: None
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int=2048, dropout: float=0.1, activation: Callable[..., nn.Module]=nn.ReLU):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.mlp = MLP(d_model, d_model, [dim_feedforward], dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = activation

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) ->Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None) ->Tensor:
        x = tgt
        q = k = self.with_pos_embed(x, query_pos)
        self_attention_outputs = self.self_attn(q, k, value=x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        x = x + self.dropout1(self_attention_outputs)
        x = self.norm1(x)
        cross_attention_outputs = self.cross_attn_image(query=self.with_pos_embed(x, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        x = x + self.dropout3(cross_attention_outputs)
        x = self.norm3(x)
        mlp_outputs = self.mlp(x)
        x = x + self.dropout4(mlp_outputs)
        x = self.norm4(x)
        return x


class RightShift(nn.Module):
    """Shifts the embedding vectors along the sequence dimension to the right.

    Since the decoder progresses by taking the token it generates in the previous step, before it
    has generated anything it needs a token to start with. Hence, the start-of-sentence (SOS) token.
    The SOS token is a learnable parameter of the decoder and the choice of its initialization is taken
    from VideoGPT: https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/attention.py#L517

    Args:
        embedding_dim (int): Dimension of the embedding vector for each token along the sequence.

    Attributes:
        sos (nn.Parameter): The starting token to be prepended to the sequence.
    """

    def __init__(self, embedding_dim: int) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sos = nn.Parameter(torch.FloatTensor(embedding_dim).normal_(std=0.02))

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x (Tensor): An input tensor of shape ``(b, seq_len, emb_dim)``.

        Returns;
            A tensor of the same shape as that of the input with the ``sos`` token prepended.
        """
        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2)
        sos = self.sos.unsqueeze(0).unsqueeze(1).repeat(x_shape[0], 1, 1)
        x = torch.cat((sos.data, x[:, :-1, :]), dim=1)
        x = x.view(*x_shape)
        return x


class LateFusion(nn.Module):
    """A generic architecture for late fusion multimodal models.

    A late fusion model contains separate encoders for each modality,
    followed by a fusion layer and then a head module. For an example of a
    late fusion model, see the TorchMultimodal implementation of the cnn-lstm
    multimodal classifier (cnn_lstm.py)

    Args:
        encoders (ModuleDict): Dictionary mapping modalities to their respective
            encoders.

    Inputs:
        modalities (Dict[str, Tensor]): A dictionary mapping modalities to
            their tensor representations.
    """

    def __init__(self, encoders: nn.ModuleDict, fusion_module: nn.Module, head_module: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleDict({k: encoders[k] for k in sorted(encoders.keys())})
        self.fusion_module = fusion_module
        self.head_module = head_module

    def forward(self, modalities: Dict[str, torch.Tensor]) ->torch.Tensor:
        embeddings = {}
        for key, encoder in self.encoders.items():
            assert key in modalities, f'{key} missing in input'
            embeddings[key] = encoder(modalities[key])
        fused = self.fusion_module(embeddings)
        return self.head_module(fused)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copied from torchvision.ops.misc with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans. This module is a useful replacement for BatchNorm2d in the
    case of very small batches, see https://bit.ly/3xQvmiJ.


    Args:   n (int): Number of features ``C`` from expected input size ``(N, C, H, W)``
            eps (float): Value added to denominator for numerical stability.
                Default = 1e-5

    Inputs: x (Tensor): Tensor to be normalized
    """

    def __init__(self, n: int, eps: float=1e-05):
        super().__init__()
        self.eps = eps
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x: Tensor) ->Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbedding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper (https://arxiv.org/abs/1706.03762),
    generalized to work on images.

    Args:   num_pos_feats (int): Number of positional features
                (should be half the output embedding size). Default = 64
            temperature (int): Base for generating frequency mesh. Default = 10000
            scale (float): Scaling factor when performing normalization. Setting
                scale = s will rescale values to fall in [0, s].
                Default = None (no normalization)

    Inputs: mask (Tensor): Padding mask (used to infer size of each image in batch).
                Input size: (batch_size, height, width)

    Returns: Tensor of size (batch_size, 2 * num_pos_feats, height, width)
    """

    def __init__(self, num_pos_feats: int=64, temperature: int=10000, scale: float=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def forward(self, mask: Tensor) ->Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.scale is not None:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MaskedIntermediateLayer(nn.Module):
    """
    This class wraps a backbone returning an intermediate layer (e.g. a ResNet
    where we do not want to perform pooling) while casting masks to the appropriate
    sizes.

    Note: for simplicity we only support returning a single intermediate layer.

    Args:   body (nn.Module): The module to return the intermediate layer from.
            intermediate_layer (str): Name of the layer to return from body.

    Inputs: images (Tensor): Batch of images to pass to the backbone.
            image_masks (Tensor): Masks to cast to backbone output size.
    """

    def __init__(self, body: nn.Module, intermediate_layer: str):
        super().__init__()
        self.body = IntermediateLayerGetter(body, return_layers={intermediate_layer: 0})

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) ->Tuple[Tensor, Tensor]:
        out = self.body(images)
        tensor = out[next(iter(out))]
        mask = F.interpolate(image_masks[None].float(), size=tensor.shape[-2:]).bool()[0]
        return tensor, mask


class MDETRTransformerOutput(NamedTuple):
    decoder_hidden_states: torch.Tensor
    text_memory: torch.Tensor


class MDETRModelOutput(NamedTuple):
    transformer_output: MDETRTransformerOutput
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    extra_embeddings: Optional[torch.Tensor]


class MDETR(nn.Module):
    """
    MDETR (https://arxiv.org/abs/2104.12763) is a modulated detection model
    used to detect objects in an image conditioned on text or captions.
    This class contains the entire MDETR architecture, including the
    image backbone, text encoder, and multimodal transformer. (Note that the
    matcher and losses are provided elsewhere.)

    Args:   image_backbone (nn.Module): Torch module of the backbone to be used.
                See image_encoder.py.
            text_encoder (nn.Module): Torch module of the text encoder to be used.
                See text_encoder.py.
            transformer (nn.Module): The multimodal transformer module. See the
                Transformer class in this file.
            pos_embed (nn.Module): Module for positional embedding of images.
            text_projection (nn.Module): Module to resize text encoder outputs before feeding
                them to the multimodal transformer.
            image_projection (nn.Module): Projection module applied to image embeddings
                prior to the multimodal transformer.
            query_embed (nn.Module): Learned object query embeddings (used in
                transformer decoder).
            bbox_embed (nn.Module): Embedding mapping transformer outputs to
                bounding boxes.
            class_embed (nn.Module): Embedding mapping transformer outputs to classes.
            extra_query_embeddings (Optional[nn.Embedding]): Additional query embeddings,
                as used in e.g. VQA. Default: None

    Inputs: images (List[Tensor]): A list of image Tensors (possibly of different sizes).
            text (List[Tensor]): A list of Tensors of tokenized texts (possibly of different lengths).

    Returns:
        A dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
    """

    def __init__(self, image_backbone: nn.Module, text_encoder: nn.Module, transformer: nn.Module, pos_embed: nn.Module, text_projection: nn.Module, image_projection: nn.Module, query_embed: nn.Embedding, bbox_embed: nn.Module, class_embed: nn.Module, extra_query_embeddings: Optional[nn.Embedding]=None):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.transformer = transformer
        self.pos_embed = pos_embed
        self.image_projection = image_projection
        self.query_embed = query_embed
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed
        self.extra_query_embeddings = extra_query_embeddings

    def _pad_images(self, images: List[Tensor]) ->Tuple[Tensor, Tensor]:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        b, _, h, w = batch_shape
        dtype = images[0].dtype
        device = images[0].device
        padded_images = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(images, padded_images, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
        return padded_images, mask

    def _pad_text(self, text: List[Tensor], padding_idx: int=1) ->Tuple[Tensor, Tensor]:
        padded_text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=padding_idx)
        mask = padded_text == padding_idx
        return padded_text, mask

    def forward(self, images: List[Tensor], text: List[Tensor]) ->MDETRModelOutput:
        images, image_mask = self._pad_images(images)
        text, text_attention_mask = self._pad_text(text)
        encoded_text = self.text_encoder(text, text_attention_mask)
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        image_embeddings, image_mask = self.image_backbone(images, image_mask)
        pos = self.pos_embed(image_mask)
        query_embed = self.query_embed.weight
        if self.extra_query_embeddings is not None:
            n_extra_embeddings = self.extra_query_embeddings.num_embeddings
            query_embed = torch.cat([query_embed, self.extra_query_embeddings.weight])
        text_memory_resized = self.text_projection(text_memory)
        transformer_output = self.transformer(self.image_projection(image_embeddings), image_mask, query_embed, pos, text_memory=text_memory_resized, text_attention_mask=text_attention_mask)
        if self.extra_query_embeddings is not None:
            extra_embeddings = transformer_output.decoder_hidden_states[0, :, -n_extra_embeddings:]
            decoder_hidden_states_truncated = transformer_output.decoder_hidden_states[:, :, :-n_extra_embeddings]
            transformer_output = transformer_output._replace(decoder_hidden_states=decoder_hidden_states_truncated)
        else:
            extra_embeddings = None
        final_hidden_state = transformer_output.decoder_hidden_states[-1]
        outputs_class = self.class_embed(final_hidden_state)
        outputs_coord = self.bbox_embed(final_hidden_state).sigmoid()
        return MDETRModelOutput(transformer_output, outputs_class, outputs_coord, extra_embeddings)


class ContrastiveEmbeddingsOutput(NamedTuple):
    query_embeddings: Tensor
    token_embeddings: Tensor


class MDETRVQAOutput(NamedTuple):
    model_output: MDETRModelOutput
    vqa_preds: Dict[str, Tensor]
    contrastive_embeddings: ContrastiveEmbeddingsOutput


class MDETRForVQA(nn.Module):

    def __init__(self, model: MDETR, vqa_heads: nn.ModuleDict, contrastive_alignment_image_projection: nn.Module, contrastive_alignment_text_projection: nn.Module):
        super().__init__()
        self.model = model
        self.vqa_heads = vqa_heads
        if self.model.extra_query_embeddings is None:
            raise ValueError('MDETRForVQA requires extra query embeddings ')
        if self.model.extra_query_embeddings.num_embeddings != len(self.vqa_heads.keys()):
            raise ValueError('Number of heads must match number of QA embeddings')
        self.contrastive_alignment_image_projection = contrastive_alignment_image_projection
        self.contrastive_alignment_text_projection = contrastive_alignment_text_projection

    def forward(self, images: List[Tensor], text: List[Tensor]) ->MDETRVQAOutput:
        model_output = self.model(images, text)
        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]
        contrastive_query_embeddings = F.normalize(self.contrastive_alignment_image_projection(final_hidden_state), p=2, dim=-1)
        contrastive_token_embeddings = F.normalize(self.contrastive_alignment_text_projection(model_output.transformer_output.text_memory).transpose(0, 1), p=2, dim=-1)
        contrastive_outputs = ContrastiveEmbeddingsOutput(contrastive_query_embeddings, contrastive_token_embeddings)
        answer_preds = OrderedDict()
        vqa_embeddings = model_output.extra_embeddings.transpose(0, 1)
        for (head_name, head), embedding in zip(self.vqa_heads.items(), vqa_embeddings):
            answer_preds[head_name] = head(embedding)
        return MDETRVQAOutput(model_output, answer_preds, contrastive_outputs)


class MDETRPhraseGroundingOutput(NamedTuple):
    model_output: MDETRModelOutput
    contrastive_embeddings: ContrastiveEmbeddingsOutput


class MDETRForPhraseGrounding(nn.Module):

    def __init__(self, model: MDETR, contrastive_alignment_image_projection: nn.Module, contrastive_alignment_text_projection: nn.Module):
        super().__init__()
        self.model = model
        self.contrastive_alignment_image_projection = contrastive_alignment_image_projection
        self.contrastive_alignment_text_projection = contrastive_alignment_text_projection

    def forward(self, images: List[Tensor], text: List[Tensor]) ->MDETRPhraseGroundingOutput:
        model_output = self.model(images, text)
        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]
        contrastive_query_embeddings = F.normalize(self.contrastive_alignment_image_projection(final_hidden_state), p=2, dim=-1)
        contrastive_token_embeddings = F.normalize(self.contrastive_alignment_text_projection(model_output.transformer_output.text_memory).transpose(0, 1), p=2, dim=-1)
        contrastive_outputs = ContrastiveEmbeddingsOutput(contrastive_query_embeddings, contrastive_token_embeddings)
        return MDETRPhraseGroundingOutput(model_output, contrastive_outputs)


class ModifiedTransformerEncoder(nn.Module):
    """
    Modified version of TorchText's RoBERTa transformer encoder
    taking in embeddings instead of input IDs.

    Args:   embedding_dim (int): Number of features in the input.
            num_encoder_layers  (int): Number of layers in the encoder.
            num_attention_heads (int): Number of heads in multi-head attention.
            ffn_dimension (int): Dimension of feedforward network inside
                attention layers.
            dropout (float): dropout value in each layer. Default: 0.1.
            normalize_before (bool): Whether to do PreNorm in encoder layers.
                Default: False
            return_all_layers (bool) Whether to return all layers (or just the last
                one). Default: False

    Inputs: embeddings (Tensor): Tensor of embeddings of a batch of input IDs.
            attention_mask (Optional[Tensor]) Batch attention mask returned from
                tokenizer (applied as padding mask inside self-attention).
                Default: None
    """

    def __init__(self, embedding_dim: int, num_encoder_layers: int, num_attention_heads: int, ffn_dimension: int, dropout: float=0.1, normalize_before: bool=False):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=ffn_dimension, dropout=dropout, activation='gelu', batch_first=True, norm_first=normalize_before)
        self.layers = torch.nn.TransformerEncoder(encoder_layer=layer, num_layers=num_encoder_layers)
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: Tensor, attention_mask: Optional[Tensor]=None, return_attn_weights: bool=False, return_hidden_states: bool=False) ->TransformerOutput:
        encoded = embeddings
        batch_size, seq_len = embeddings.size()[:2]
        mask = attention_mask.reshape(batch_size, seq_len)
        for layer in self.layers.layers:
            encoded = layer(encoded, src_key_padding_mask=mask)
        return TransformerOutput(last_hidden_state=encoded)


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    Args:   input_feat_size (int): Dimension of input features.
            output_feat_size (int): Dimension of output features.
            dropout (float): Dropout probability for final features. Default: 0.1
            do_ln (bool): Whether to perform layer normalization after the linear layer.
    Inputs: encoder_features (Tensor): Features to be resized.
    """

    def __init__(self, input_feat_size: int, output_feat_size: int, dropout: float=0.1, do_ln: bool=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12) if do_ln else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: Tensor) ->Tensor:
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class MDETRTransformer(nn.Module):
    """
    Transformer class for MDETR model.

    Args:   d_model (int): Number of features in the input.
            num_heads (int): Number of heads in multi-head attention.
            num_encoder_layers (int): Number of layers in the encoder. Default: 6
            num_decoder_layers (int): Number of layers in the decoder. Default: 6
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., nn.Module]): The activation function of the
                intermediate layer. Default: nn.ReLU
            normalize_before (bool): Whether to do PreNorm. Default: False
            return_intermediate_dec (bool): Whether to return intermediate decoder outputs.
                Default: True

    Inputs: image_embeddings Tensor: The image input.
            image_mask (Tensor) The mask for the image sequence.
            query_embed (Tensor): Positional embeddings applied to Q
                cross-attention matrix in decoder.
            pos_embed (Tensor): Positional embeddings applied to Q and K
                self-attention matrices in decoder.
            text_memory (Tensor): Text input.
            text_attention_mask (Tensor): Attention mask for text input.
    """

    def __init__(self, d_model: int=512, num_heads: int=8, num_encoder_layers: int=6, num_decoder_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1, activation: Callable[..., nn.Module]=nn.ReLU, normalize_before: bool=False, return_intermediate_dec: bool=True):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation, normalize_before)
        encoder_final_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_final_norm)
        decoder_layer = TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self._init_parameters()

    def _init_parameters(self) ->None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_embeddings: Tensor, image_mask: Tensor, query_embed: Tensor, pos_embed: Tensor, text_memory: Tensor, text_attention_mask: Tensor) ->MDETRTransformerOutput:
        bs = image_embeddings.size(0)
        image_embeddings = image_embeddings.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        image_mask = image_mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        mm_embeddings = torch.cat([image_embeddings, text_memory], dim=0)
        image_mask = torch.cat([image_mask, text_attention_mask], dim=1)
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory)], dim=0)
        mm_memory = self.encoder(mm_embeddings, src_key_padding_mask=image_mask, pos=pos_embed)
        text_memory = mm_memory[-len(text_memory):]
        assert mm_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
        assert mm_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
        hs = self.decoder(tgt, mm_memory, memory_key_padding_mask=image_mask, pos=pos_embed, query_pos=query_embed)
        return MDETRTransformerOutput(decoder_hidden_states=hs.transpose(1, 2), text_memory=text_memory)


class TransformerEncoder(nn.Module):

    def __init__(self, n_layer: int, d_model: int, n_head: int, dim_feedforward: int, dropout: float=0.0, activation: Callable[..., nn.Module]=nn.ReLU, layer_norm_eps: float=1e-12, norm_first: bool=False, final_layer_norm_eps: Optional[float]=None):
        super().__init__()
        self.layer = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(n_layer)])
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, return_attn_weights: bool=False, return_hidden_states: bool=False) ->TransformerOutput:
        all_hidden_states: Tuple[Tensor, ...] = () if return_hidden_states else None
        all_self_attentions: Tuple[Tensor, ...] = () if return_attn_weights else None
        for layer_module in self.layer:
            if return_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, head_mask=head_mask, return_attn_weights=return_attn_weights)
            if return_attn_weights:
                hidden_states = layer_outputs[0]
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs
        if return_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        return TransformerOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer is made up of multihead self-attention and feedforward blocks,
    based on the architecture in "Attention Is All You Need" (Vaswani et al. 2017). Similar to
    ``nn.TransformerEncoderLayer``, but uses a custom ``MultiHeadAttention`` that supports
    n-dimensional inputs (including sequences, images, video) and head-masking.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``. See
            ``MultiHeadAttention`` for shape requirements.
        head_mask (Tensor, optional): mask to be applied to self-attention inputs after softmax and dropout,
            before matrix multiplication with values. See ``MultiHeadAttention`` for shape requirements.
        return_attn_weights (bool, optional): return attention probabilities in addition to attention output.
            Defaults to False.
    """

    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float=0.0, activation: Callable[..., nn.Module]=nn.ReLU, layer_norm_eps: float=1e-12, norm_first: bool=False) ->None:
        super().__init__()
        self.attention = MultiHeadAttention(dim_q=d_model, dim_kv=d_model, n_head=n_head, attn_module=SelfAttention(dropout))
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward = MLP(d_model, d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _attention_block(self, hidden_states: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        output, attn_weights = self.attention(hidden_states, attention_mask=attention_mask, head_mask=head_mask, return_attn_weights=True)
        output = self.attention_dropout(output)
        return output, attn_weights

    def _feedforward_block(self, hidden_states: Tensor) ->Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(self, hidden_states: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, return_attn_weights: bool=False) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        inputs = self.attention_layernorm(x)
        attn_output, attn_weights = self._attention_block(inputs, attention_mask=attention_mask, head_mask=head_mask)
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(self.feedforward_layernorm(attn_residual))
        if return_attn_weights:
            return ff_residual, attn_weights
        else:
            return ff_residual

    def _forward_postnorm(self, hidden_states: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, return_attn_weights: bool=False) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        attn_output, attn_weights = self._attention_block(x, attention_mask=attention_mask, head_mask=head_mask)
        attn_residual = attn_output + x
        attn_residual = self.attention_layernorm(attn_residual)
        ff_residual = attn_residual + self._feedforward_block(attn_residual)
        outputs = self.feedforward_layernorm(ff_residual)
        if return_attn_weights:
            return outputs, attn_weights
        else:
            return outputs

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor]=None, head_mask: Optional[Tensor]=None, return_attn_weights: bool=False) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.norm_first:
            return self._forward_prenorm(hidden_states, attention_mask, head_mask, return_attn_weights)
        else:
            return self._forward_postnorm(hidden_states, attention_mask, head_mask, return_attn_weights)


class Omnivore(nn.Module):
    """Omnivore is a model that accept multiple vision modality.

    Omnivore (https://arxiv.org/abs/2201.08377) is a single model that able to do classification
    on images, videos, and single-view 3D data using the same shared parameters of the encoder.

    Args:
        encoder (nn.Module): Instantiated encoder. It generally accept a video backbone.
            The paper use SwinTransformer3d for the encoder.
        heads (Optional[nn.ModuleDict]): Dictionary of multiple heads for each dataset type

    Inputs:
        x (Tensor): 5 Dimensional batched video tensor with format of B C D H W
            where B is batch, C is channel, D is time, H is height, and W is width.
        input_type (str): The dataset type of the input, this will used to choose
            the correct head.
    """

    def __init__(self, encoder: nn.Module, heads: nn.ModuleDict):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, x: torch.Tensor, input_type: str) ->torch.Tensor:
        x = self.encoder(x)
        assert input_type in self.heads, f'Unsupported input_type: {input_type}, please use one of {list(self.heads.keys())}'
        x = self.heads[input_type](x)
        return x


class PatchEmbedOmnivore(nn.Module):
    """Patch Embedding strategy for Omnivore model
    It will use common PatchEmbed3d for image and video,
    for single view depth image it will have separate embedding for the depth channel
    and add the embedding result with the RGB channel
    reference: https://arxiv.org/abs/2201.08377

    Args:
        patch_size (Tuple[int, int, int]): Patch token size. Default: ``(2, 4, 4)``
        embed_dim (int): Number of linear projection output channels. Default: ``96``
        norm_layer (nn.Module, optional): Normalization layer. Default: ``None``
    """

    def __init__(self, patch_size: List[int], embed_dim: int=96, norm_layer: Optional[Callable[..., nn.Module]]=None):
        super().__init__()
        self.patch_embed = PatchEmbed3d(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.depth_patch_embed = PatchEmbed3d(patch_size=patch_size, in_channels=1, embed_dim=embed_dim, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        assert x.ndim == 5
        has_depth = x.shape[1] == 4
        if has_depth:
            x_rgb = self.patch_embed(x[:, :3, ...])
            x_d = self.depth_patch_embed(x[:, 3:, ...])
            x = x_rgb + x_d
        else:
            x = self.patch_embed(x)
        return x


class TwoTowerOutput(NamedTuple):
    output: Tensor
    tower_embeddings: Dict[str, Tensor]


class TwoTower(nn.Module):
    """
    A two tower architecture with a pair of late fusion models
    (for now, can be extended) followed by a fusion for output of each tower.
    Args:
        tower_id_to_tower (Dict[str, LateFusion]): mapping of tower id
        to tower model. Size should be 2, same tower should be passed in
        for shared towers
        tower fusion (nn.Module): Module fusing list of tensors (tower outputs)
        into single output
        shared_tower_id_to_channel_mapping (Optional[Dict[str, Dict[str, str]]]): Dict
        of shared tower id to mapping of channel names of the shared tower
         to the original input channel name
    Inputs:
        channel_to_input (Dict[str,Tensor]) : Channel name to input tensor dict
    """

    def __init__(self, tower_id_to_tower: Dict[str, LateFusion], tower_fusion: nn.Module, shared_tower_id_to_channel_mapping: Optional[Dict[str, Dict[str, str]]]=None):
        super().__init__()
        if len(tower_id_to_tower) != 2:
            raise ValueError(f'Two tower needs 2 towers but found                 {len(tower_id_to_tower)} towers')
        self.tower_id_to_tower = nn.ModuleDict(tower_id_to_tower)
        self.tower_fusion = tower_fusion
        if shared_tower_id_to_channel_mapping is not None:
            towers = list(tower_id_to_tower.values())
            if towers[0] != towers[1]:
                raise ValueError('Towers should be shared if channel mapping is passed in')
        self.shared_tower_id_to_channel_mapping: Optional[Dict[str, Dict[str, str]]] = shared_tower_id_to_channel_mapping

    def forward(self, channel_to_input: Dict[str, Tensor]) ->TwoTowerOutput:
        tower_embeddings = OrderedDict()
        for tower_id, tower in self.tower_id_to_tower.items():
            tower_input = self._get_tower_input(tower_id, list(tower.encoders.keys()), channel_to_input)
            tower_embeddings[tower_id] = tower(tower_input)
        final_out = self.tower_fusion(list(tower_embeddings.values()))
        return TwoTowerOutput(output=final_out, tower_embeddings=tower_embeddings)

    def _get_tower_input(self, tower_id: str, tower_channels: List[str], channel_to_input: Dict[str, Tensor]) ->Dict[str, Tensor]:
        tower_input = {}
        channel_name_mapping: Dict[str, str] = {}
        if self.shared_tower_id_to_channel_mapping is not None:
            if self.shared_tower_id_to_channel_mapping.get(tower_id) is not None:
                channel_name_mapping = self.shared_tower_id_to_channel_mapping[tower_id]
        for channel in tower_channels:
            if channel_name_mapping.get(channel) is not None:
                input_channel_name = channel_name_mapping[channel]
            else:
                input_channel_name = channel
            tower_input[channel] = channel_to_input[input_channel_name]
        return tower_input


def calculate_transpose_padding(kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], input_shape: Union[Size, Tuple[int, ...]], input_pad: Union[int, Tuple[int, ...]]=0) ->Tuple[Tuple, Tuple]:
    """Calculates padding for transposed convolution based on input dims, kernel size, and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are doubled.

    The 'padding' argument in ConvTranspose effectively trims the output, and the 'output_padding'
    argument effectively expands the output. These two knobs are adjusted to meet desired output dim.

    Args:
        kernel_size (int or Tuple[int, ...]): Size of convolutional kernel.
        stride (int or Tuple[int, ...]): Stride amount of kernel.
        input_shape (Size or Tuple[int, ...]): Shape of input, without batch or channel dimension.
        input_pad (int or Tuple[int, ...]): Amount of padding added to input, must be twice length of
            kernel/stride/input_shape.

    Returns:
        A tuple of padding and output_padding to be used in ConvTranspose layers
    """
    n_dims = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n_dims))
    if isinstance(stride, int):
        stride = tuple(repeat(stride, n_dims))
    if isinstance(input_pad, int):
        input_pad = tuple(repeat(input_pad, n_dims * 2))
    if not len(kernel_size) == len(stride) == len(input_shape):
        raise ValueError('dims for kernel, stride, and input must match')
    if len(input_pad) % 2 != 0 or len(input_pad) // 2 != len(input_shape):
        raise ValueError('input_pad length must be twice the number of dims')
    transpose_pad = []
    output_pad = []
    for i, (d, k, s) in enumerate(zip(input_shape, kernel_size, stride)):
        output_shape_actual = k + (d + input_pad[2 * i] + input_pad[2 * i + 1] - 1) * s
        output_shape_expected = d * s
        transpose_pad.append(max((output_shape_actual - output_shape_expected + 1) // 2, 0))
        output_pad.append(output_shape_expected - (output_shape_actual - transpose_pad[-1] * 2))
    transpose_pad = tuple(transpose_pad)
    output_pad = tuple(output_pad)
    return transpose_pad, output_pad


class SamePadConvTranspose3d(nn.Module):
    """Performs a same padded transposed convolution on a 3D input.

    This ensures output shape in input shape multiplied by stride.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (int): Number of channels in input, same as Conv3d
        out_channels (int): Number of channels for output, same as Conv3d
        kernel_size (int or Tuple[int, int, int]): Size of convolutional filter, same as Conv3d
        stride (int or Tuple[int, int, int], optional): Stride for convolution, same as Conv3d
        bias (bool, optional): If ``True`` use a bias for convolutional layer or not,
            same as ``nn.Conv3d``. Defaults to ``True``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]]=1, bias: bool=True, **kwargs: Any) ->None:
        super().__init__()
        self.pad_input: Tuple = None
        self.kernel_size = kernel_size
        self.stride = stride
        if 'padding' in kwargs:
            warnings.warn('Padding was specified but will not be used in favor of same padding,                 use ConvTranspose3d directly for custom padding')
        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, **kwargs)

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(self.kernel_size, self.stride, x.shape[2:])
            self.convt.padding, self.convt.output_padding = calculate_transpose_padding(self.kernel_size, self.stride, x.shape[2:], self.pad_input[::-1])
        return self.convt(F.pad(x, self.pad_input))


class VideoDecoder(nn.Module):
    """Decoder for Video VQVAE.

    Takes quantized output from codebook and applies a ``SamePadConv3d`` layer, a stack of
    ``AttentionResidualBlocks``, followed by a specified number of ``SamePadConvTranspose3d``
    layers. The residual blocks use Axial Attention to enhance representations of video data
    without significantly increasing computational cost.

    Follows VideoGPT's implementation:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        out_channel_dims (Tuple[int, ...]): Output channel dimension for each layer in conv stack.
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): Kernel sizes for each layer in conv stack.
        strides (Tuple[Tuple[int, int, int], ...]): Strides for each layer in conv stack
        input_dim (int): Input channel dimension for first conv layer before attention stack
        n_res_layers (int): Number of ``AttentionResidualBlocks`` to include. Default is ``4``.
        attn_hidden_dim (int): Size of hidden dimension in attention block. Default is ``240``.
        kwargs (Any): Keyword arguments to be passed into ``SamePadConvTranspose3d`` and used by
            ``nn.ConvTranspose3d``.

    Raises:
        ValueError: If the lengths of ``out_channel_dims``, ``kernel_sizes``, and ``strides`` are not
            all equivalent.
    """

    def __init__(self, out_channel_dims: Tuple[int, ...], kernel_sizes: Tuple[Tuple[int, int, int], ...], strides: Tuple[Tuple[int, int, int], ...], input_dim: int, n_res_layers: int=4, attn_hidden_dim: int=240, **kwargs: Any):
        super().__init__()
        assert_equal_lengths(out_channel_dims, kernel_sizes, strides, msg='out_channel_dims, kernel_sizes, and strides must be same length.')
        self.conv_in = SamePadConv3d(input_dim, attn_hidden_dim, kernel_size=1, stride=1)
        self.res_stack = nn.Sequential(*[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)], nn.BatchNorm3d(attn_hidden_dim), nn.ReLU())
        transpose_convolutions: List[nn.Module] = []
        n_conv_layers = len(out_channel_dims)
        for i in range(n_conv_layers):
            in_channel = out_channel_dims[i - 1] if i > 0 else attn_hidden_dim
            out_channel = out_channel_dims[i]
            kernel = kernel_sizes[i]
            stride = strides[i]
            transpose_convolutions.append(SamePadConvTranspose3d(in_channel, out_channel, kernel, stride, bias=True, **kwargs))
            if i < n_conv_layers - 1:
                transpose_convolutions.append(nn.ReLU())
        self.convts = nn.Sequential(*transpose_convolutions)

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x (Tensor): Input quantized embeddings with shape ``(b, emb_dim, d1, d2, d3)``.
        """
        in_channel = x.shape[1]
        if in_channel != self.conv_in.conv.in_channels:
            raise ValueError(f'expected input channel dim to be {self.conv_in.conv.in_channels}, but got {in_channel}')
        h = self.conv_in(x)
        h = self.res_stack(h)
        h = self.convts(h)
        return h


class CodebookOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.modules.layers.codebook.Codebook`.

    Attributes:
        encoded_flat (Tensor): The flattened encoder output of shape ``(b x d1 x ... x dn, c)``.
        quantized_flat (Tensor): The nearest embeddings for the encoded of shape ``(b x d1 x ... x dn, emb_dim)``.
        codebook_indices (Tensor): Indices of the nearest embeddings of shape ``(b, d1, d2, ..., dn)``.
        quantized (Tensor): The nearest embeddings reshaped back to ``(b, emb_dim, d1, ..., dn)``.
    """
    encoded_flat: Tensor
    quantized_flat: Tensor
    codebook_indices: Tensor
    quantized: Tensor


class Codebook(nn.Module):
    """Bottleneck layer of VQVAE model

    Codebook provides an embedding layer that takes in the output of an encoder
    and performs a nearest-neighbor lookup in the embedding space.
    Vector quantization was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)
    to generate high-fidelity images, videos, and audio data.
    The embedding weights are trained with exponential moving average updates as described
    in original paper.

    Code was largely inspired by a PyTorch implementation of the author's original code, found here:
    https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    and by the implementation in MUGEN (Hayes et al. 2022), found here:
    https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/video_vqvae/vqvae.py

    Args:
        num_embeddings (int): Number of vectors in the embedding space.
        embedding_dim (int): Dimensionality of the embedding vectors.
        decay (float, optional): Factor used in exponential moving average update of the embeddings.
            Defaults to ``0.99``.
        codebook_usage_threshold (float, optional): Threshold for the average number of times an embedding vector
            is chosen below which it will be re-initialized. Defaults to ``1.0``.
        epsilon (float, optional): Noise used in Laplace smoothing of codebook usage. Defaults to ``1e-7``.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float=0.99, codebook_usage_threshold: float=1.0, epsilon: float=1e-07) ->None:
        super().__init__()
        randn_init_embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', randn_init_embedding.clone())
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        self.register_buffer('code_avg', randn_init_embedding.clone())
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self._decay = decay
        self._epsilon = epsilon
        self.codebook_usage_threshold = codebook_usage_threshold
        self._is_embedding_init = False

    def _load_from_state_dict(self, state_dict: Mapping[str, Any], prefix: str, local_metadata: Mapping, strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) ->None:
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._is_embedding_init = True

    def _tile(self, x: Tensor, n: int) ->Tensor:
        num_vectors, num_channels = x.shape
        if num_vectors < n:
            num_repeats = (n + num_vectors - 1) // num_vectors
            std = 0.01 / torch.sqrt(torch.tensor(num_channels))
            x = x.repeat(num_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _get_random_vectors(self, x: Tensor, n: int) ->Tensor:
        x_tiled = self._tile(x, n)
        idx = torch.randperm(x_tiled.shape[0])
        x_rand = x_tiled[idx][:n]
        return x_rand

    def _preprocess(self, encoded: Tensor) ->Tuple[Tensor, Size]:
        encoded_permuted = shift_dim(encoded, 1, -1)
        permuted_shape = encoded_permuted.shape
        encoded_flat = encoded_permuted.view(-1, permuted_shape[-1])
        if encoded_flat.shape[-1] != self.embedding_dim:
            raise ValueError(f'Expected {encoded_flat.shape[-1]} to be embedding size of {self.embedding_dim}')
        return encoded_flat, permuted_shape

    def _postprocess(self, quantized_flat: Tensor, permuted_shape: Union[Size, Tuple]) ->Tensor:
        quantized_permuted = quantized_flat.view(permuted_shape)
        quantized = shift_dim(quantized_permuted, -1, 1)
        return quantized

    def _init_embedding(self, encoded_flat: Tensor) ->None:
        self._is_embedding_init = True
        encoded_flat_rand = self._get_random_vectors(encoded_flat, self.num_embeddings)
        self.embedding = encoded_flat_rand
        self.code_avg = encoded_flat_rand
        self.code_usage = torch.ones(self.num_embeddings)

    def _ema_update_embedding(self, encoded_flat: Tensor, codebook_indices: Tensor) ->None:
        codebook_onehot = nn.functional.one_hot(codebook_indices, num_classes=self.num_embeddings).type(torch.float)
        codebook_selection_count = torch.sum(codebook_onehot, 0)
        self.code_usage.mul_(self._decay).add_(codebook_selection_count, alpha=1 - self._decay)
        n = torch.sum(self.code_usage)
        self.code_usage.add_(self._epsilon).divide_(n + self.num_embeddings * self._epsilon).mul_(n)
        encoded_per_codebook = torch.matmul(codebook_onehot.t(), encoded_flat)
        self.code_avg.mul_(self._decay).add_(encoded_per_codebook, alpha=1 - self._decay)
        self.embedding = self.code_avg / self.code_usage.unsqueeze(1)
        encoded_flat_rand = self._get_random_vectors(encoded_flat, self.num_embeddings)
        self.embedding = torch.where(self.code_usage.unsqueeze(1) >= self.codebook_usage_threshold, self.embedding, encoded_flat_rand)

    def _quantize(self, encoded_flat: Tensor) ->Tuple[Tensor, Tensor]:
        distances = torch.cdist(encoded_flat, self.embedding, p=2.0) ** 2
        codebook_indices_flat = torch.argmin(distances, dim=1)
        quantized_flat = F.embedding(codebook_indices_flat, self.embedding)
        if self.training:
            self._ema_update_embedding(encoded_flat, codebook_indices_flat)
        quantized_flat = encoded_flat + (quantized_flat - encoded_flat).detach()
        return quantized_flat, codebook_indices_flat

    def forward(self, z: Tensor) ->CodebookOutput:
        """
        Args:
            z (Tensor): Tensor containing a batch of encoder outputs of shape ``(b, c, d1, ..., dn)``.

        Returns:
            An instance of :class:`~torchmultimodal.modules.layers.codebook.CodebookOutput`.
        """
        encoded_flat, permuted_shape = self._preprocess(z)
        if not self._is_embedding_init and self.training:
            self._init_embedding(encoded_flat)
        quantized_flat, codebook_indices_flat = self._quantize(encoded_flat)
        quantized = self._postprocess(quantized_flat, permuted_shape)
        codebook_indices = codebook_indices_flat.view(z.shape[0], *z.shape[2:])
        return CodebookOutput(encoded_flat, quantized_flat, codebook_indices, quantized)

    def extra_repr(self) ->str:
        return 'num_embeddings={}, embedding_dim={}'.format(self.num_embeddings, self.embedding_dim)

    def lookup(self, indices: Tensor) ->Tensor:
        return F.embedding(indices, self.embedding)


class VQVAEOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.vqvae.VQVAE`.

    Attributes:
        decoded (Tensor): Output of the decoder.
        codebook_output (CodebookOutput): Output of codebook layer to be used in loss calculations.
    """
    decoded: Tensor
    codebook_output: CodebookOutput


class VQVAE(nn.Module):
    """General model for VQVAE that provides codebook layer to link user specified
    encoder and decoder.

    Vector Quantized Variational Autoencoder is a type of autoencoder that defines
    an embedding of discrete vectors as the latent variables in the bottleneck layer
    instead of normally distributed latent variables as in a standard VAE. This enables
    high-fidelity reconstruction of input data. It was first introduced in "Neural
    Discrete Representation Learning" (Oord et al. 2017) and has since seen success in
    tokenizing and generating high-resolution image, audio, and video data.

    Args:
        encoder (nn.Module): Model that accepts single Tensor as input in forward, ``encoder(x)``.
            Will be used to project input into codebook layer. Expects channel
            dim of encoder output to match ``embedding_dim`` of codebook.
            See :class:`~torchmultimodal.modules.layers.codebook.Codebook`.
        decoder (nn.Module): Model that accepts single Tensor as input in forward, ``decoder(x)``.
            Should be able to accept output shape of codebook layer, which matches output shape of
            the encoder.
        num_embeddings (int): Number of embedding vectors in codebook.
        embedding_dim (int): Dimensionality of embedding vectors in codebook.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, num_embeddings: int, embedding_dim: int) ->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = Codebook(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def latent_shape(self, input_shape: Union[Size, Tuple]) ->Tuple[int, ...]:
        """Returns the downsampled shape of the encoder output: (d1, ..., dn)"""
        if not hasattr(self.encoder, 'get_latent_shape'):
            raise AttributeError(f"Missing attribute 'get_latent_shape' of the encoder {self.encoder}")
        return self.encoder.get_latent_shape(input_shape)

    def encode(self, x: Tensor, return_embeddings: bool=False) ->Union[Tuple[Tensor, Tensor], Tensor]:
        """Converts input data to token ids

        Args:
            x (Tensor): Input data of shape ``(b, c, d1, ..., dn)``.
            return_embeddings (bool): Flag to return also the quantized embeddings. Defaults to ``False``.

        Returns:
            * A tensor of token ids: ``(b, d1, ...., dn)``
            * A tuple of token ids and quantized embeddings ``(b, emb_dim, d1, ..., dn)``.
        """
        encoded = self.encoder(x)
        out = self.codebook(encoded)
        indices = out.codebook_indices
        quantized = out.quantized
        if return_embeddings:
            return indices, quantized
        return indices

    def decode(self, indices: Tensor) ->Tensor:
        """Converts token ids back to data"""
        quantized = self.lookup(indices)
        quantized = shift_dim(quantized, -1, 1)
        return self.decoder(quantized)

    def lookup(self, indices: Tensor) ->Tensor:
        if not hasattr(self.codebook, 'lookup'):
            raise AttributeError(f"Missing attribute 'lookup' of the codebook {self.codebook}")
        return self.codebook.lookup(indices)

    def forward(self, x: Tensor) ->VQVAEOutput:
        """
        Args:
            x (Tensor): Input data of shape ``(b, c, d1, ..., dn)``.

        Returns:
            An instance of :class:`~torchmultimodal.models.vqvae.VQVAEOutput`.
        """
        encoded = self.encoder(x)
        codebook_output = self.codebook(encoded)
        decoded = self.decoder(codebook_output.quantized)
        return VQVAEOutput(decoded, codebook_output)


POOLING_TYPES = ['sum', 'mean', 'max']


class EmbeddingEncoder(nn.Module):
    """Combine embeddings for tensor representing list of indices based on pooling type

    Args:
        embedding (nn.Embedding): embedding module
        pooling_type (str): pooling function to combine the embeddings like sum. Choose
        from pooling_types
        pooling_dim (int) : dimension along which the pooling function is applied
        use_hash (bool): if hashing based on embedding vocab size if applied to input
        before embedding layer

    Inputs:
        x (Tensor): Tensor bsz x max seq length representing (padded) list of indices
        for embedding

    """

    def __init__(self, embedding: nn.Embedding, pooling_type: str, pooling_dim: int=1, use_hash: bool=False):
        super().__init__()
        self.embedding = embedding
        if pooling_type not in POOLING_TYPES:
            raise ValueError(f'pooling type should be in {POOLING_TYPES}, found {pooling_type}')
        self.pooling_type = pooling_type
        self.pooling_dim = pooling_dim
        self.use_hash = use_hash

    def forward(self, x: Tensor) ->Tensor:
        if self.use_hash:
            x = x % (self.embedding.num_embeddings - 1) + 1
        out = self.embedding(x)
        if self.pooling_type == 'sum':
            out = torch.sum(out, dim=self.pooling_dim)
        elif self.pooling_type == 'mean':
            out = torch.mean(out, dim=self.pooling_dim)
        else:
            out = torch.max(out, dim=self.pooling_dim).values
        return out


class DeepsetFusionModule(nn.Module):
    """
    Fuse embeddings through stacking followed by pooling strategy and MLP
    See https://arxiv.org/pdf/2003.01607.pdf

    Args:
        channel_to_encoder_dim (Dict[str, int]): mapping of channel name to the        encoding dimension
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).        Use MLP for mlp_classifier for default mlp.
        pooling_function (Callable): Pooling function to combine the tensors,        like torch.median        apply_attention (bool): If self attention (2 layer net) is applied before        stacking embeddings, defaults to False.
        attention_dim (int): intermediate dim for attention layer.        defaults to projection dim / 2
        modality_normalize (bool): If normalization is applied along the modality axis,        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim         is applied to the embeddings. defaults to False

    """

    def __init__(self, channel_to_encoder_dim: Dict[str, int], mlp: nn.Module, pooling_function: Callable, apply_attention: bool=False, attention_dim: Optional[int]=None, modality_normalize: bool=False, norm_factor: float=2.0, use_auto_mapping: bool=False):
        super().__init__()
        self.apply_attention = apply_attention
        self.modality_normalize = modality_normalize
        self.norm_factor = norm_factor
        self.use_auto_mapping = use_auto_mapping
        projection_dim = DeepsetFusionModule.get_projection_dim(channel_to_encoder_dim, use_auto_mapping)
        if self.use_auto_mapping:
            self.projections = nn.ModuleDict({channel: nn.Linear(dim, projection_dim) for channel, dim in channel_to_encoder_dim.items()})
        else:
            self.projections = nn.ModuleDict({channel: nn.Identity() for channel in channel_to_encoder_dim})
        if self.apply_attention:
            self.attention: nn.Module
            if attention_dim is None:
                attention_dim = projection_dim // 2
            self.attention = nn.Sequential(nn.Linear(projection_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1), nn.Softmax(dim=-2))
        else:
            self.attention = nn.Identity()
        self.pooling_function = pooling_function
        self.mlp = mlp

    def forward(self, embeddings: Dict[str, Tensor]) ->Tensor:
        projections = {}
        for channel, projection in self.projections.items():
            projections[channel] = projection(embeddings[channel])
        embedding_list = [projections[k] for k in sorted(projections.keys())]
        stacked_embeddings = torch.stack(embedding_list, dim=1)
        if self.apply_attention:
            attn_weights = self.attention(stacked_embeddings)
            stacked_embeddings = stacked_embeddings * attn_weights
        if self.modality_normalize:
            normalized_embeddings = F.normalize(stacked_embeddings, p=self.norm_factor, dim=1)
        else:
            normalized_embeddings = F.normalize(stacked_embeddings, p=self.norm_factor, dim=2)
        pooled_features = self._pool_features(normalized_embeddings)
        fused = self.mlp(pooled_features)
        return fused

    @classmethod
    def get_projection_dim(cls, channel_to_encoder_dim: Dict[str, int], use_auto_mapping: bool) ->int:
        if use_auto_mapping:
            projection_dim = min(channel_to_encoder_dim.values())
        else:
            encoder_dim = set(channel_to_encoder_dim.values())
            if len(encoder_dim) != 1:
                raise ValueError('Encoder dimension should be same for all channels                     if use_auto_mapping is set to false')
            projection_dim = encoder_dim.pop()
        return projection_dim

    def _pool_features(self, embeddings: Tensor) ->Tensor:
        pooled_embeddings = self.pooling_function(embeddings, dim=1)
        if torch.jit.isinstance(pooled_embeddings, Tuple[Tensor, Tensor]):
            return pooled_embeddings.values
        if not isinstance(pooled_embeddings, Tensor):
            raise ValueError(f'Result from pooling function should be a tensor.             {self.pooling_function} does not satisfy that')
        return pooled_embeddings


class DeepsetFusionWithTransformer(DeepsetFusionModule):

    def __init__(self, channel_to_encoder_dim: Dict[str, int], mlp: nn.Module, pooling_function: nn.TransformerEncoder, apply_attention: bool=False, attention_dim: Optional[int]=None, modality_normalize: bool=False, norm_factor: float=2.0, use_auto_mapping: bool=False):
        super().__init__(channel_to_encoder_dim, mlp, pooling_function, apply_attention, attention_dim, modality_normalize, norm_factor, use_auto_mapping)

    def _pool_features(self, embeddings: Tensor) ->Tensor:
        pooled = self.pooling_function(embeddings)
        return pooled[:, 0, :]


class MILEncoder(nn.Module):
    """
    Multi instance learning encoder that partitions the input into a set of inputs
    and uses a shared encoder followed by deepset
    fusion to get a pooled representation of the entire input. Example use is to build a
    single representation from embeddings of all images in a post.

    Args:
        partition_sizes (List[int]): list of size for each partition of the input
        shared_encoder (nn.Module): Shared encoder for each partition of the input.
        shared_encoder_dim (int) : Output dimension of the encoders
        Following fields are same as the params for deepset fusion
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).        Use MLP from mlp_classifier for default mlp implementation.
        pooling_function (Callable): Pooling function to combine the tensors,        like torch.median
        apply_attention (bool): If self attention is applied before        stacking embeddings, defaults to False
        modality_normalize (bool): If normalization is applied along the modality axis,        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim         is applied to the embeddings. defaults to False

    """

    def __init__(self, partition_sizes: List[int], shared_encoder: nn.Module, shared_encoder_dim: int, mlp: nn.Module, pooling_function: Callable, apply_attention: bool=False, attention_dim: Optional[int]=None, modality_normalize: bool=False, norm_factor: float=2.0, use_auto_mapping: bool=False):
        super().__init__()
        self.partition_sizes = partition_sizes
        self.shared_encoder = shared_encoder
        channel_to_encoder_dim = {}
        for i in range(len(partition_sizes)):
            channel_to_encoder_dim[self.get_channel_name(i)] = shared_encoder_dim
        deepset_fusion_cls = DeepsetFusionWithTransformer if isinstance(pooling_function, nn.TransformerEncoder) else DeepsetFusionModule
        self.deepset_fusion: Union[DeepsetFusionWithTransformer, DeepsetFusionModule] = deepset_fusion_cls(channel_to_encoder_dim=channel_to_encoder_dim, mlp=mlp, pooling_function=pooling_function, apply_attention=apply_attention, attention_dim=attention_dim, modality_normalize=modality_normalize, norm_factor=norm_factor, use_auto_mapping=use_auto_mapping)

    def get_channel_name(self, id: int) ->str:
        return f'mil_{id}'

    def forward(self, x: Tensor) ->Tensor:
        idx = 0
        input_size = x.size(dim=1)
        if input_size != sum(self.partition_sizes):
            raise ValueError(f'partition sizes should sum to the input size {input_size}')
        partitioned_input = torch.split(x, self.partition_sizes, dim=1)
        encoded_input = {}
        for idx, input in enumerate(partitioned_input):
            key = self.get_channel_name(idx)
            encoded_input[key] = self.shared_encoder(input)
        return self.deepset_fusion(encoded_input)


class WeightedEmbeddingEncoder(nn.Module):
    """Combine weighted embeddings for tensor representing list of indices based on
    pooling type.

    Args:
        embedding (nn.Embedding): embedding module
        pooling_function (Callable[[Tensor, int], Union[Tensor, Tuple]]): pooling function to combine the weighted embeddings,        example: torch.sum function should return a tensor or namedtuple containing the tensor in the values field like torch.max
        pooling_dim (int) : dimension along which the pooling function is applied

    Inputs:
        weights (Tensor): A float tensor of shape [batch_size x num_categories] containing the weights of a categorical feature.            The weights represent multiplier factors for the corresponding category embedding vectors.

    """

    def __init__(self, embedding: nn.Embedding, pooling_function: Callable[[Tensor, int], Union[Tensor, Tuple]], pooling_dim: int=1) ->None:
        super().__init__()
        self.embedding = embedding
        self.pooling_function = pooling_function
        self.pooling_dim = pooling_dim

    def forward(self, weights: Tensor) ->Tensor:
        index = torch.arange(0, weights.size(1), dtype=torch.int)
        index = index
        weighted_embeddings = self.embedding(index) * weights.unsqueeze(-1)
        pooled_embeddings = self.pooling_function(weighted_embeddings, self.pooling_dim)
        if isinstance(pooled_embeddings, Tensor):
            output: Tensor = pooled_embeddings
        else:
            assert hasattr(pooled_embeddings, 'values'), 'pooled embeddings should be Tensor or tuple with values field as Tensor'
            output = pooled_embeddings.values
        return output


class AttentionFusionModule(nn.Module):
    """
    Fuse embeddings through weighted sum of the corresponding linear projections.
    Linear layer for learning the weights.

    Args:
        channel_to_encoder_dim: mapping of channel name to the encoding dimension
        encoding_projection_dim: common dimension to project the encodings to.
        defaults to min of the encoder dim if not set

    """

    def __init__(self, channel_to_encoder_dim: Dict[str, int], encoding_projection_dim: Optional[int]=None):
        super().__init__()
        attn_in_dim = sum(channel_to_encoder_dim.values())
        self.attention = nn.Sequential(nn.Linear(attn_in_dim, len(channel_to_encoder_dim)), nn.Softmax(-1))
        if encoding_projection_dim is None:
            encoding_projection_dim = min(channel_to_encoder_dim.values())
        encoding_projection = {}
        for channel in sorted(channel_to_encoder_dim.keys()):
            encoding_projection[channel] = nn.Linear(channel_to_encoder_dim[channel], encoding_projection_dim)
        self.encoding_projection = nn.ModuleDict(encoding_projection)

    def forward(self, embeddings: Dict[str, Tensor]) ->Tensor:
        concatenated_in = torch.cat([embeddings[k] for k in sorted(embeddings.keys())], dim=-1)
        attention_weights = self.attention(concatenated_in)
        projected_embeddings: List[Tensor] = []
        for channel, projection in self.encoding_projection.items():
            projected_embedding = projection(embeddings[channel])
            projected_embeddings.append(projected_embedding)
        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
        fused = torch.sum(torch.stack(projected_embeddings), dim=0)
        return fused


class ConcatFusionModule(nn.Module):
    """Module to fuse modalities via concatenation. Sorted by keys for consistency.

    Inputs:
        embeddings (Dict[str, Tensor]): A dictionary mapping modalities to their
            tensor representations.

    """

    def __init__(self, projection: nn.Module=None):
        super().__init__()
        if projection:
            self.projection = projection
        else:
            self.projection = nn.Identity()

    def forward(self, embeddings: Dict[str, torch.Tensor]) ->torch.Tensor:
        concatenated_in = torch.cat([embeddings[k] for k in sorted(embeddings.keys())], dim=-1)
        return self.projection(concatenated_in)


class BroadcastedPositionEmbedding(nn.Module):
    """Spatiotemporal broadcasted positional embeddings.

    Based on broadcasted position embedding algorithm in codebase:
        https://github.com/wilson1yan/VideoGPT/blob/c21cc7e2579f820cb2b90097406d72cf69a46474/videogpt/attention.py#L458

        Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \\prod_{j>i}\\text{dim}[j]`.

    Args:
        latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
        embedding_dim (int): The size of each embedding vector.

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``.
    """

    def __init__(self, latent_shape: Tuple[int, ...], embedding_dim: int) ->None:
        """
        Args:
            latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
            embedding_dim (int): The size of each embedding vector.

        Raises:
            ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``
        """
        super().__init__()
        if embedding_dim % len(latent_shape) != 0:
            raise ValueError(f'Embedding dim {embedding_dim} modulo len(latent_shape) {len(latent_shape)} is not zero')
        self.latent_shape = latent_shape
        self.n_dim = n_dim = len(self.latent_shape)
        self.embedding_dim = embedding_dim
        self.embedding = nn.ParameterDict({f'd_{i}': nn.Parameter(torch.randn(self.latent_shape[i], embedding_dim // n_dim) * 0.01) for i in range(n_dim)})

    @property
    def indices(self) ->Tensor:
        """Returns broadcasted indices of the data

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 3), embedding_dim=6)
            >>> pos_emb.indices
            tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        """
        return torch.cartesian_prod(*[torch.arange(s) for s in self.latent_shape])

    def _broadcast(self, i: int) ->Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.latent_shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.latent_shape``. The embedding dim is not touched.

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 4), embedding_dim=6)
            >>> print(pos_emb.embedding["d_0"].shape)
            torch.Size([2, 3])
            >>> pos_emb.embedding["d_0"] = nn.Parameter(torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
            >>> out = pos_emb._broadcast(i=0)
            >>> print(out)
            tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]])
            >>> print(out.shape)
            (2, 4, 3)

        The input is broadcasted along the second dim ``4`` since it's the ``0``-th embedding constructed w.r.t the
        first dim ``2``.
        """
        emb = self.embedding[f'd_{i}']
        emb = emb.view(*itertools.repeat(1, i), self.latent_shape[i], *itertools.repeat(1, self.n_dim - i - 1), -1)
        emb = emb.expand(*self.latent_shape, -1)
        return emb

    def forward(self, position_ids: Tensor) ->Tensor:
        """
        Args:
            position_ids (Tensor): batches of of 1D integer tensors indicating locations of the broadcasted
                position embeddings to be returned.

        Returns:
            A tensor with the position embeddings selected by position ids.

        Raises:
            IndexError: If any position id(s) provided is outside of the indices range.
        """
        invalid_ids = position_ids[torch.logical_or(position_ids >= len(self.indices), position_ids < -1)]
        if len(invalid_ids):
            raise IndexError(f'Invalid position ids: {invalid_ids}')
        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=-1)
        indices = [*self.indices[position_ids].permute(2, 1, 0)]
        embeddings = embeddings[indices].transpose(0, 1)
        return embeddings


DEFAULT_LOGIT_SCALE = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))


class ContrastiveLossWithTemperature(nn.Module):
    """Contrastive loss with a temperature parameter, as used in CLIP and FLAVA.
    CLIP: https://arxiv.org/pdf/2103.00020.pdf
    FLAVA: https://arxiv.org/pdf/2112.04482.pdf


    A contrastive loss over pairs of input embeddings a and b. For each input_a
    embedding, we compute a weighted cosine similarity with all input_b embeddings,
    then calculate the cross entropy loss against the true (input_a, input_b) pairing.
    Each input_b embedding is evaluated against all input_a embeddings similarly.
    The batch's loss is the average cross entropy over all input_a and input_b embeddings
    in the batch.

    Temperature is a learned parameter clamped to ``[1, 100]`` and
    initialized to 1 / 0.07 as in the CLIP paper.


    Args:
        logit_scale (Union[float, nn.Module]): Log of the learnable temperature parameter value
            A nn.Parameter instantiation can also be passed directly in case parent class
            is handling the initialization.
            Defaults to ``ln(1/0.07)``, as in the CLIP paper.
        logit_scale_min (Optional[float]): Log of the minimum temperature value.
            If ``None``, then temperature will not be clamped to a minimum value.
            Defaults to ``ln(1)``, as in the CLIP paper.
        logit_scale_max (Optional[float]): Log of the maximum temperature value.
            If ``None``, then temperature will not be clamped to a maximum value.
            Defaults to ``ln(100)``, as in the CLIP paper.

    Inputs: embeddings_a (Tensor): Tensor containing features from the first input or modality.
                (In the CLIP model, these are the outputs of the image encoder.)
            embeddings_b (Tensor): Tensor containing features from the second input or modality.
                (In the CLIP model, these are the outputs of the text encoder.)
            backprop_in_gather (bool): Whether to backpropagate the gradients from
                all_gather to all workers (versus just the local worker).
            cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)
    """

    def __init__(self, logit_scale: Union[float, nn.Parameter]=DEFAULT_LOGIT_SCALE, logit_scale_min: Optional[float]=math.log(1), logit_scale_max: Optional[float]=math.log(100)):
        super().__init__()
        if not logit_scale_min and not logit_scale_max:
            raise ValueError('Only one of `logit_scale_min` and `logit_scale_max` can be None.')
        self.logit_scale_min = logit_scale_min
        self.logit_scale_max = logit_scale_max
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def forward(self, embeddings_a: Tensor, embeddings_b: Tensor, backprop_in_gather: bool=True, cross_entropy_kwargs: Optional[Dict[str, Any]]=None) ->Tensor:
        self.logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)
        return contrastive_loss_with_temperature(embeddings_a=embeddings_a, embeddings_b=embeddings_b, logit_scale=self.logit_scale, backprop_in_gather=backprop_in_gather, cross_entropy_kwargs=cross_entropy_kwargs).loss


class CommitmentLoss(nn.Module):
    """Commitment loss calculates the mean Euclidean distance between pairs of encoder output vectors
    and their corresponding quantized vectors. It encourages an encoder to generate outputs closer to an embedding.
    This is the beta in Eq. 3 of Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)

    Args:
        commitment_cost (float): multiplicative weight for the commitment loss value
    """

    def __init__(self, commitment_cost: float=1.0, **kwargs: Any) ->None:
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, quantized: Tensor, encoded: Tensor) ->Tensor:
        loss = F.mse_loss(quantized.detach(), encoded) * self.commitment_cost
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BroadcastedPositionEmbedding,
     lambda: ([], {'latent_shape': [4, 4], 'embedding_dim': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64)], {}),
     False),
    (CLIP,
     lambda: ([], {'encoder_a': _mock_layer(), 'encoder_b': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CNNEncoder,
     lambda: ([], {'input_dims': [4, 4], 'output_dims': [4, 4], 'kernel_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Codebook,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CommitmentLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvertTCHWtoCTHW,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DalleConv2d,
     lambda: ([], {'n_in': 4, 'n_out': 4, 'kw': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DalleEncoderBlock,
     lambda: ([], {'n_in': 4, 'n_out': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthNorm,
     lambda: ([], {'max_depth': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (FeatureResizer,
     lambda: ([], {'input_feat_size': 4, 'output_feat_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageTextContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'dim_q': 4, 'dim_kv': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Projection,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetForCLIPBottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RightShift,
     lambda: ([], {'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SamePadConv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerCrossAttentionLayer,
     lambda: ([], {'d_model': 4, 'n_head': 4, 'dim_feedforward': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (TransformerEncoder,
     lambda: ([], {'n_layer': 1, 'd_model': 4, 'n_head': 4, 'dim_feedforward': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'n_head': 4, 'dim_feedforward': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Unsqueeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VQVAE,
     lambda: ([], {'encoder': _mock_layer(), 'decoder': _mock_layer(), 'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_facebookresearch_multimodal(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

