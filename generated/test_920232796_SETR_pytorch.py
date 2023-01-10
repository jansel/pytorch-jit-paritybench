import sys
_module = sys.modules[__name__]
del sys
transformer_model = _module
transformer_seg = _module
main = _module
task_minst = _module
tast_car_seg = _module

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


import logging


import math


import torch


from torch import nn


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import numpy as np


import torchvision


import torch.nn as nn


from torchvision import datasets


from torchvision import transforms


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


class TransLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class TransEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[:2])
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransConfig(object):

    def __init__(self, patch_size, in_channels, out_channels, sample_rate=4, hidden_size=768, num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, initializer_range=0.02, layer_norm_eps=1e-12):
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class TransSelfAttention(nn.Module):

    def __init__(self, config: TransConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class TransSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = TransSelfAttention(config)
        self.output = TransSelfOutput(config)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


def gelu(x):
    """ 
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish, 'mish': mish}


class TransIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = TransAttention(config)
        self.intermediate = TransIntermediate(config)
        self.output = TransOutput(config)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TransEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([TransLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            layer_output = layer_module(hidden_states)
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class InputDense2d(nn.Module):

    def __init__(self, config):
        super(InputDense2d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class InputDense3d(nn.Module):

    def __init__(self, config):
        super(InputDense3d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * config.in_channels, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TransModel2d(nn.Module):

    def __init__(self, config):
        super(TransModel2d, self).__init__()
        self.config = config
        self.dense = InputDense2d(config)
        self.embeddings = TransEmbeddings(config)
        self.encoder = TransEncoder(config)

    def forward(self, input_ids, output_all_encoded_layers=True):
        dense_out = self.dense(input_ids)
        embedding_output = self.embeddings(input_ids=dense_out)
        encoder_layers = self.encoder(embedding_output, output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoder_layers[-1]
        if not output_all_encoded_layers:
            encoder_layers = encoder_layers[-1]
        return encoder_layers


class TransModel3d(nn.Module):

    def __init__(self, config):
        super(TransModel3d, self).__init__()
        self.config = config
        self.dense = InputDense3d(config)
        self.embeddings = TransEmbeddings(config)
        self.encoder = TransEncoder(config)

    def forward(self, input_ids, output_all_encoded_layers=True):
        dense_out = self.dense(input_ids)
        embedding_output = self.embeddings(input_ids=dense_out)
        encoder_layers = self.encoder(embedding_output, output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoder_layers[-1]
        if not output_all_encoded_layers:
            encoder_layers = encoder_layers[-1]
        return encoder_layers


class Encoder2D(nn.Module):

    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % sample_v ** 2 == 0, '不能除尽'
        self.final_dense = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // sample_v ** 2)
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v
        self.is_segmentation = is_segmentation

    def forward(self, x):
        b, c, h, w = x.shape
        assert self.config.in_channels == c, 'in_channels != 输入图像channel'
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        if h % p1 != 0:
            None
            os._exit(0)
        if w % p2 != 0:
            None
            os._exit(0)
        hh = h // p1
        ww = w // p2
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)
        encode_x = self.bert_model(x)[-1]
        if not self.is_segmentation:
            return encode_x
        x = self.final_dense(encode_x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.hh, p2=self.ww, h=hh, w=ww, c=self.config.hidden_size)
        return encode_x, x


class PreTrainModel(nn.Module):

    def __init__(self, patch_size, in_channels, out_class, hidden_size=1024, num_hidden_layers=8, num_attention_heads=16, decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size, in_channels=in_channels, out_channels=0, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out


class Vit(nn.Module):

    def __init__(self, patch_size, in_channels, out_class, hidden_size=1024, num_hidden_layers=8, num_attention_heads=16, sample_rate=4):
        super().__init__()
        config = TransConfig(patch_size=patch_size, in_channels=in_channels, out_channels=0, sample_rate=sample_rate, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out


class Decoder2D(nn.Module):

    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(nn.Conv2d(in_channels, features[0], 3, padding=1), nn.BatchNorm2d(features[0]), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder_2 = nn.Sequential(nn.Conv2d(features[0], features[1], 3, padding=1), nn.BatchNorm2d(features[1]), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder_3 = nn.Sequential(nn.Conv2d(features[1], features[2], 3, padding=1), nn.BatchNorm2d(features[2]), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder_4 = nn.Sequential(nn.Conv2d(features[2], features[3], 3, padding=1), nn.BatchNorm2d(features[3]), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class SETRModel(nn.Module):

    def __init__(self, patch_size=(32, 32), in_channels=3, out_channels=1, hidden_size=1024, num_hidden_layers=8, num_attention_heads=16, decode_features=[512, 256, 128, 64], sample_rate=4):
        super().__init__()
        config = TransConfig(patch_size=patch_size, in_channels=in_channels, out_channels=out_channels, sample_rate=sample_rate, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels, features=decode_features)

    def forward(self, x):
        _, final_x = self.encoder_2d(x)
        x = self.decoder_2d(final_x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransEmbeddings,
     lambda: ([], {'config': _mock_config(max_position_embeddings=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransLayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_920232796_SETR_pytorch(_paritybench_base):
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

