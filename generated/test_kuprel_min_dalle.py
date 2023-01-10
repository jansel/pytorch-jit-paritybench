import sys
_module = sys.modules[__name__]
del sys
image_from_text = _module
min_dalle = _module
min_dalle = _module
models = _module
dalle_bart_decoder = _module
dalle_bart_encoder = _module
vqgan_detokenizer = _module
text_tokenizer = _module
predictor = _module
setup = _module
tkinter_ui = _module

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


import torch


import numpy


from torch import LongTensor


from torch import FloatTensor


import torch.backends.cudnn


import torch.backends.cuda


from typing import Iterator


from typing import Tuple


from typing import List


from torch import nn


from torch import BoolTensor


from math import sqrt


import string


class AttentionBase(nn.Module):

    def __init__(self, head_count: int, embed_count: int):
        super().__init__()
        self.head_count = head_count
        self.embed_count = embed_count
        self.k_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.v_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.q_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.out_proj = nn.Linear(embed_count, embed_count, bias=False)

    def forward(self, keys: FloatTensor, values: FloatTensor, queries: FloatTensor, attention_mask: BoolTensor) ->FloatTensor:
        keys = keys.reshape(keys.shape[:2] + (self.head_count, -1))
        values = values.reshape(values.shape[:2] + (self.head_count, -1))
        queries = queries.reshape(queries.shape[:2] + (self.head_count, -1))
        queries /= queries.shape[-1] ** 0.5
        attention_bias = (1 - attention_mask) * -1000000000000.0
        attention_weights: FloatTensor = torch.einsum('bqhc,bkhc->bhqk', queries, keys)
        attention_weights += attention_bias
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum('bhqk,bkhc->bqhc', attention_weights, values)
        shape = attention_output.shape[:2] + (self.embed_count,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj.forward(attention_output)
        return attention_output


class DecoderCrossAttention(AttentionBase):

    def forward(self, decoder_state: FloatTensor, encoder_state: FloatTensor, attention_mask: BoolTensor) ->FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):

    def __init__(self, head_count: int, embed_count: int):
        super().__init__(head_count, embed_count)

    def forward(self, decoder_state: FloatTensor, attention_state: FloatTensor, attention_mask: BoolTensor, token_index: LongTensor) ->Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        token_count = token_index.shape[1]
        if token_count == 1:
            batch_count = decoder_state.shape[0]
            attn_state_new = torch.cat([keys, values])
            attention_state[:, token_index[0]] = attn_state_new
            keys = attention_state[:batch_count]
            values = attention_state[batch_count:]
        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, attention_state


class GLU(nn.Module):

    def __init__(self, count_in_out: int, count_middle: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.ln0 = nn.LayerNorm(count_in_out)
        self.ln1 = nn.LayerNorm(count_middle)
        self.fc0 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc1 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc2 = nn.Linear(count_middle, count_in_out, bias=False)

    def forward(self, z: FloatTensor) ->FloatTensor:
        z = self.ln0.forward(z)
        w = self.fc0.forward(z)
        w = self.gelu.forward(w)
        v = self.fc1.forward(z)
        z = self.ln1.forward(w * v)
        z = self.fc2.forward(z)
        return z


IMAGE_TOKEN_COUNT = 256


class DecoderLayer(nn.Module):

    def __init__(self, head_count: int, embed_count: int, glu_embed_count: int, device: str):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttention(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)

    def forward(self, decoder_state: FloatTensor, encoder_state: FloatTensor, attention_state: FloatTensor, attention_mask: BoolTensor, token_index: LongTensor) ->Tuple[FloatTensor, FloatTensor]:
        token_count = token_index.shape[1]
        if token_count == 1:
            self_attn_mask = self.token_indices <= token_index
            self_attn_mask = self_attn_mask[:, None, None, :]
        else:
            self_attn_mask = self.token_indices[None, None, :token_count] <= token_index[:, :, None]
            self_attn_mask = self_attn_mask[:, None, :, :]
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(decoder_state=decoder_state, attention_state=attention_state, attention_mask=self_attn_mask, token_index=token_index)
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(decoder_state=decoder_state, encoder_state=encoder_state, attention_mask=attention_mask)
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state
        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):

    def __init__(self, image_vocab_count: int, embed_count: int, attention_head_count: int, glu_embed_count: int, layer_count: int, device: str):
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.image_vocab_count = image_vocab_count
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embed_count)
        self.layers: List[DecoderLayer] = nn.ModuleList([DecoderLayer(head_count=attention_head_count, embed_count=embed_count, glu_embed_count=glu_embed_count, device=device) for _ in range(layer_count)])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)

    def forward(self, attention_mask: BoolTensor, encoder_state: FloatTensor, attention_state: FloatTensor, prev_tokens: LongTensor, token_index: LongTensor) ->Tuple[FloatTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index = token_index.unsqueeze(0).repeat(image_count * 2, 1)
        prev_tokens = prev_tokens.repeat(2, 1)
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(decoder_state, encoder_state, attention_state[i], attention_mask, token_index)
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        return logits, attention_state

    def sample_tokens(self, settings, **kwargs) ->Tuple[LongTensor, FloatTensor]:
        logits, attention_state = self.forward(**kwargs)
        image_count = logits.shape[0] // 2
        temperature = settings[[0]]
        top_k = settings[[1]]
        supercondition_factor = settings[[2]]
        logits = logits[:, -1, :2 ** 14]
        logits: FloatTensor = logits[:image_count] * (1 - supercondition_factor) + logits[image_count:] * supercondition_factor
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        logits -= logits_sorted[:, [0]]
        logits /= temperature
        logits.exp_()
        logits *= is_kept
        image_tokens = torch.multinomial(logits, 1)[:, 0]
        return image_tokens, attention_state


class EncoderSelfAttention(AttentionBase):

    def forward(self, encoder_state: FloatTensor, attention_mask: BoolTensor) ->FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(encoder_state)
        return super().forward(keys, values, queries, attention_mask)


class EncoderLayer(nn.Module):

    def __init__(self, embed_count: int, head_count: int, glu_embed_count: int):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = EncoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)

    def forward(self, encoder_state: FloatTensor, attention_mask: BoolTensor) ->FloatTensor:
        residual = encoder_state
        encoder_state = self.pre_self_attn_layer_norm.forward(encoder_state)
        encoder_state = self.self_attn.forward(encoder_state, attention_mask)
        encoder_state = self.self_attn_layer_norm.forward(encoder_state)
        encoder_state = residual + encoder_state
        residual = encoder_state
        encoder_state = self.glu.forward(encoder_state)
        encoder_state = residual + encoder_state
        return encoder_state


class DalleBartEncoder(nn.Module):

    def __init__(self, layer_count: int, embed_count: int, attention_head_count: int, text_vocab_count: int, text_token_count: int, glu_embed_count: int, device: str):
        super().__init__()
        self.text_vocab_count = text_vocab_count
        self.embed_tokens = nn.Embedding(text_vocab_count, embed_count)
        self.embed_positions = nn.Embedding(text_token_count, embed_count)
        self.layers: List[EncoderLayer] = nn.ModuleList([EncoderLayer(embed_count=embed_count, head_count=attention_head_count, glu_embed_count=glu_embed_count) for _ in range(layer_count)])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        token_indices = torch.arange(text_token_count, device=device)
        self.pose_tokens = torch.stack([token_indices] * 2)

    def forward(self, text_tokens: LongTensor) ->FloatTensor:
        attention_mask = text_tokens.not_equal(1)[:, None, None, :]
        encoder_state = self.embed_tokens.forward(text_tokens) + self.embed_positions.forward(self.pose_tokens)
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, attention_mask)
        encoder_state = self.final_ln.forward(encoder_state)
        return encoder_state


class ResnetBlock(nn.Module):

    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2 ** 5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2 ** 5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) ->FloatTensor:
        h = x
        h = self.norm1.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(nn.Module):

    def __init__(self):
        super().__init__()
        n = 2 ** 9
        self.norm = nn.GroupNorm(2 ** 5, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x: FloatTensor) ->FloatTensor:
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        token_count = int(sqrt(h.shape[-1]))
        h = h.reshape(m, n, token_count, token_count)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)

    def forward(self, h: FloatTensor) ->FloatTensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):

    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) ->FloatTensor:
        x = self.upsample.forward(x)
        x = self.conv.forward(x)
        return x


class UpsampleBlock(nn.Module):

    def __init__(self, log2_count_in: int, log2_count_out: int, has_attention: bool, has_upsample: bool):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        self.block = nn.ModuleList([ResnetBlock(log2_count_in, log2_count_out), ResnetBlock(log2_count_out, log2_count_out), ResnetBlock(log2_count_out, log2_count_out)])
        if has_attention:
            self.attn = nn.ModuleList([AttentionBlock(), AttentionBlock(), AttentionBlock()])
        if has_upsample:
            self.upsample = Upsample(log2_count_out)

    def forward(self, h: FloatTensor) ->FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2 ** 8, 2 ** 9, 3, padding=1)
        self.mid = MiddleLayer()
        self.up = nn.ModuleList([UpsampleBlock(7, 7, False, False), UpsampleBlock(8, 7, False, True), UpsampleBlock(8, 8, False, True), UpsampleBlock(9, 8, False, True), UpsampleBlock(9, 9, True, True)])
        self.norm_out = nn.GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = nn.Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: FloatTensor) ->FloatTensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)
        for i in reversed(range(5)):
            z = self.up[i].forward(z)
        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(nn.Module):

    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.embedding = nn.Embedding(vocab_count, embed_count)
        self.post_quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()

    def forward(self, is_seamless: bool, z: LongTensor) ->FloatTensor:
        grid_size = int(sqrt(z.shape[0]))
        token_count = grid_size * 2 ** 4
        if is_seamless:
            z = z.view([grid_size, grid_size, 2 ** 4, 2 ** 4])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
            z = z.flatten().unsqueeze(1)
            z = self.embedding.forward(z)
            z = z.view((1, token_count, token_count, 2 ** 8))
        else:
            z = self.embedding.forward(z)
            z = z.view((z.shape[0], 2 ** 4, 2 ** 4, 2 ** 8))
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255
        if is_seamless:
            z = z[0]
        else:
            z = z.view([grid_size, grid_size, 2 ** 8, 2 ** 8, 3])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
        return z


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionBase,
     lambda: ([], {'head_count': 4, 'embed_count': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     False),
    (DecoderCrossAttention,
     lambda: ([], {'head_count': 4, 'embed_count': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (DecoderSelfAttention,
     lambda: ([], {'head_count': 4, 'embed_count': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EncoderLayer,
     lambda: ([], {'embed_count': 4, 'head_count': 4, 'glu_embed_count': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (EncoderSelfAttention,
     lambda: ([], {'head_count': 4, 'embed_count': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (GLU,
     lambda: ([], {'count_in_out': 4, 'count_middle': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'log2_count': 4}),
     lambda: ([torch.rand([4, 16, 4, 4])], {}),
     True),
]

class Test_kuprel_min_dalle(_paritybench_base):
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

