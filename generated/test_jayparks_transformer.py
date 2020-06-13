import sys
_module = sys.modules[__name__]
del sys
data = _module
data_utils = _module
dataset = _module
preprocess = _module
train = _module
transformer = _module
beam = _module
layers = _module
models = _module
modules = _module
optimizer = _module
sublayers = _module
translator = _module
translate = _module

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


import math


import torch


import torch.nn as nn


from torch.nn.utils import clip_grad_norm


import torch.optim as optim


import numpy as np


import torch.nn.init as init


from torch.autograd import Variable


class EncoderLayer(nn.Module):

    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads,
            dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
            enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class WeightedEncoderLayer(nn.Module):

    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff,
            n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
            attn_mask=self_attn_mask)


class DecoderLayer(nn.Module):

    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads,
            dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads,
            dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs,
            dec_inputs, dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs,
            enc_outputs, enc_outputs, attn_mask=enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class WeightedDecoderLayer(nn.Module):

    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model,
            n_branches, dropout)
        self.dec_enc_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff,
            n_branches, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs,
            dec_inputs, dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs,
            enc_outputs, enc_outputs, attn_mask=enc_attn_mask)
        return dec_outputs, dec_self_attn, dec_enc_attn


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)
    return pad_attn_mask.expand(b_size, len_q, len_k)


class Encoder(nn.Module):

    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
        max_seq_len, src_vocab_size, dropout=0.1, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=
            data_utils.PAD)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model)
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = (EncoderLayer if not weighted else
            WeightedEncoderLayer)
        self.layers = nn.ModuleList([self.layer_type(d_k, d_v, d_model,
            d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs += self.pos_emb(enc_inputs_len)
        enc_outputs = self.dropout_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class Decoder(nn.Module):

    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
        max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=
            data_utils.PAD)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model)
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = (DecoderLayer if not weighted else
            WeightedDecoderLayer)
        self.layers = nn.ModuleList([self.layer_type(d_k, d_v, d_model,
            d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs,
        return_attn=False):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs += self.pos_emb(dec_inputs_len)
        dec_outputs = self.dropout_emb(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask +
            dec_self_attn_subsequent_mask, 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs,
                enc_outputs, self_attn_mask=dec_self_attn_mask,
                enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


def proj_prob_simplex(inputs):
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i + 1].sum() - 1) / (i + 1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs - t, min=0.0)


class Transformer(nn.Module):

    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = Encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model,
            opt.d_ff, opt.n_heads, opt.max_src_seq_len, opt.src_vocab_size,
            opt.dropout, opt.weighted_model)
        self.decoder = Decoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model,
            opt.d_ff, opt.n_heads, opt.max_tgt_seq_len, opt.tgt_vocab_size,
            opt.dropout, opt.weighted_model)
        self.tgt_proj = Linear(opt.d_model, opt.tgt_vocab_size, bias=False)
        self.weighted_model = opt.weighted_model
        if opt.share_proj_weight:
            None
            self.tgt_proj.weight = self.decoder.tgt_emb.weight
        if opt.share_embs_weight:
            None
            assert opt.src_vocab_size == opt.tgt_vocab_size, 'To share word embeddings, the vocabulary size of src/tgt should be the same'
            self.encoder.src_emb.weight = self.decoder.tgt_emb.weight

    def trainable_params(self):
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)
        return param_groups

    def encode(self, enc_inputs, enc_inputs_len, return_attn=False):
        return self.encoder(enc_inputs, enc_inputs_len, return_attn)

    def decode(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs,
        return_attn=False):
        return self.decoder(dec_inputs, dec_inputs_len, enc_inputs,
            enc_outputs, return_attn)

    def forward(self, enc_inputs, enc_inputs_len, dec_inputs,
        dec_inputs_len, return_attn=False):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs,
            enc_inputs_len, return_attn)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs,
            dec_inputs_len, enc_inputs, enc_outputs, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)
            ), enc_self_attns, dec_self_attns, dec_enc_attns

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1000000000.0)
        attn = self.dropout(self.softmax(scores))
        context = torch.matmul(attn, v)
        return context, attn


class LayerNormalization(nn.Module):

    def __init__(self, d_hid, eps=1e-06):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta
        return ln_out


class PosEncoding(nn.Module):

    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array([[(pos / np.power(10000, 2.0 * (j // 2) /
            d_word_vec)) for j in range(d_word_vec)] for pos in range(
            max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc),
            requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        tensor = (torch.cuda.LongTensor if input_len.is_cuda else torch.
            LongTensor)
        input_pos = tensor([(list(range(1, len + 1)) + [0] * (max_len - len
            )) for len in input_len])
        return self.pos_enc(input_pos)


class _MultiHeadAttention(nn.Module):

    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = Linear([d_model, d_k * n_heads])
        self.w_k = Linear([d_model, d_k * n_heads])
        self.w_v = Linear([d_model, d_v * n_heads])
        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, q, k, v, attn_mask):
        b_size = q.size(0)
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(
            1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(
            1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(
            1, 2)
        if attn_mask:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(b_size, -1, 
            self.n_heads * self.d_v)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model,
            n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        residual = q
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)
        output = self.dropout(self.proj(context))
        return self.layer_norm(residual + output), attn


class MultiBranchAttention(nn.Module):

    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_branches = n_branches
        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model,
            n_branches, dropout)
        self.w_o = nn.ModuleList([Linear(d_v, d_model) for _ in range(
            n_branches)])
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp / self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a / self.w_a.sum())
        self.pos_ffn = nn.ModuleList([PoswiseFeedForwardNet(d_model, d_ff //
            n_branches, dropout) for _ in range(n_branches)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)
        init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask):
        residual = q
        context, attn = self.multih_attn(q, k, v, attn_mask=attn_mask)
        context = context.split(self.d_v, dim=-1)
        outputs = [self.w_o[i](context[i]) for i in range(self.n_branches)]
        outputs = [(kappa * output) for kappa, output in zip(self.w_kp,
            outputs)]
        outputs = [pos_ffn(output) for pos_ffn, output in zip(self.pos_ffn,
            outputs)]
        outputs = [(alpha * output) for alpha, output in zip(self.w_a, outputs)
            ]
        output = self.dropout(torch.stack(outputs).sum(dim=0))
        return self.layer_norm(residual + output), attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff,
            kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model,
            kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return self.layer_norm(residual + output)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jayparks_transformer(_paritybench_base):
    pass
    def test_000(self):
        self._check(LayerNormalization(*[], **{'d_hid': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Linear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(PoswiseFeedForwardNet(*[], **{'d_model': 4, 'd_ff': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ScaledDotProductAttention(*[], **{'d_k': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

