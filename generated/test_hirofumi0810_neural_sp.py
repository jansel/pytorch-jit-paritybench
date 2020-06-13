import sys
_module = sys.modules[__name__]
del sys
remove_disfluency = _module
remove_pos = _module
pre_filter = _module
text_post_process = _module
text_pre_process = _module
format_acronyms_dict = _module
format_acronyms_dict_fisher_swbd = _module
map_acronyms_ctm = _module
map_acronyms_transcripts = _module
join_suffix = _module
neural_sp = _module
bin = _module
args_asr = _module
args_lm = _module
asr = _module
eval = _module
plot_attention = _module
plot_ctc = _module
train = _module
eval_utils = _module
lm = _module
plot_cache = _module
train = _module
plot_utils = _module
train_utils = _module
datasets = _module
token_converter = _module
character = _module
phone = _module
word = _module
wordpiece = _module
evaluators = _module
accuracy = _module
edit_distance = _module
ppl = _module
resolving_unk = _module
wordpiece_bleu = _module
models = _module
base = _module
criterion = _module
data_parallel = _module
build = _module
gated_convlm = _module
lm_base = _module
rnnlm = _module
transformer_xl = _module
transformerlm = _module
modules = _module
attention = _module
causal_conv = _module
cif = _module
gelu = _module
glu = _module
gmm_attention = _module
initialization = _module
mocha = _module
multihead_attention = _module
positinal_embedding = _module
positionwise_feed_forward = _module
relative_multihead_attention = _module
sync_bidir_multihead_attention = _module
transformer = _module
zoneout = _module
seq2seq = _module
decoders = _module
beam_search = _module
ctc = _module
decoder_base = _module
fwd_bwd_attention = _module
las = _module
rnn_transducer = _module
transformer = _module
encoders = _module
conv = _module
encoder_base = _module
gated_conv = _module
rnn = _module
tds = _module
transformer = _module
frontends = _module
frame_stacking = _module
gaussian_noise = _module
sequence_summary = _module
spec_augment = _module
splicing = _module
speech2text = _module
torch_utils = _module
trainers = _module
lr_scheduler = _module
model_name = _module
optimizer = _module
reporter = _module
utils = _module
setup = _module
test_encoder = _module
test_las_decoder = _module
test_rnn_transducer_decoder = _module
test_rnnlm = _module
test_transformer_decoder = _module
test_transformer_xl_lm = _module
test_transformerlm = _module
compute_oov_rate = _module
make_tsv = _module
map2phone = _module
text2dict = _module
trn2ctm = _module

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


import copy


import logging


import numpy as np


import torch


import functools


import torch.nn as nn


from torch.nn.utils import vector_to_parameters


from torch.nn.utils import parameters_to_vector


import math


import torch.nn.functional as F


from torch.nn import DataParallel


from torch.nn.parallel.scatter_gather import gather


from collections import OrderedDict


import random


from itertools import groupby


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


logger = logging.getLogger(__name__)


class ModelBase(nn.Module):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.info('Overriding ModelBase class.')

    @property
    def torch_version(self):
        return float('.'.join(torch.__version__.split('.')[:2]))

    @property
    def num_params_dict(self):
        if not hasattr(self, '_nparams_dict'):
            self._nparams_dict = {}
            for n, p in self.named_parameters():
                self._nparams_dict[n] = p.view(-1).size(0)
        return self._nparams_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_nparams'):
            self._nparams = 0
            for n, p in self.named_parameters():
                self._nparams += p.view(-1).size(0)
        return self._nparams

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters())).idx

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1. See detail in

            https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

        """
        for n, p in self.named_parameters():
            if p.dim() == 1 and 'bias_ih' in n:
                dim = p.size(0)
                start, end = dim // 4, dim // 2
                p.data[start:end].fill_(1.0)
                logger.info('Initialize %s with 1 (bias in forget gate)' % n)

    def add_weight_noise(self, std=0.075):
        """Add variational weight noise to weight parametesr.

        Args:
            std (float): standard deviation

        """
        with torch.no_grad():
            param_vector = parameters_to_vector(self.parameters())
            normal_dist = torch.distributions.Normal(loc=torch.tensor([0.0]
                ), scale=torch.tensor([std]))
            noise = normal_dist.sample(param_vector.size())
            if self.device_id >= 0:
                noise = noise
            param_vector.add_(noise[0])
        vector_to_parameters(param_vector, self.parameters())

    def cudnn_setting(self, deterministic=False, benchmark=True):
        """CuDNN setting.

        Args:
            deterministic (bool):
            benchmark (bool):

        """
        assert self.use_cuda
        if benchmark:
            torch.backends.cudnn.benchmark = True
        elif deterministic:
            torch.backends.cudnn.enabled = False
        logger.info('torch.backends.cudnn.benchmark: %s' % torch.backends.
            cudnn.benchmark)
        logger.info('torch.backends.cudnn.enabled: %s' % torch.backends.
            cudnn.enabled)


class CustomDataParallel(DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(CustomDataParallel, self).__init__(module, device_ids,
            output_device, dim)

    def gather(self, outputs, output_device):
        n_returns = len(outputs[0])
        n_gpus = len(outputs)
        if n_returns == 2:
            losses = [output[0] for output in outputs]
            observation_mean = {}
            for output in outputs:
                for k, v in output[1].items():
                    if v is None:
                        continue
                    if k not in observation_mean.keys():
                        observation_mean[k] = v
                    else:
                        observation_mean[k] += v
                observation_mean = {k: (v / n_gpus) for k, v in
                    observation_mean.items()}
            return gather(losses, output_device, dim=self.dim).mean(
                ), observation_mean
        else:
            raise ValueError(n_returns)


NEG_INF = float(np.finfo(np.float32).min)


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        atype (str): type of attention mechanisms
        adim: (int) dimension of the attention space
        sharpening_factor (float): sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): number of channles of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): size of kernel.
            This must be the odd number.
        dropout (float): dropout probability for attention weights
        lookahead (int): lookahead frames for triggered attention

    """

    def __init__(self, kdim, qdim, adim, atype, sharpening_factor=1,
        sigmoid_smoothing=False, conv_out_channels=10, conv_kernel_size=201,
        dropout=0.0, lookahead=2):
        super(AttentionMechanism, self).__init__()
        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.atype = atype
        self.adim = adim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = 1
        self.lookahead = lookahead
        self.reset()
        self.dropout = nn.Dropout(p=dropout)
        if atype == 'no':
            raise NotImplementedError
        elif atype in ['add', 'triggered_attention']:
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)
        elif atype == 'location':
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.w_conv = nn.Linear(conv_out_channels, adim, bias=False)
            self.conv = nn.Conv2d(in_channels=1, out_channels=
                conv_out_channels, kernel_size=(1, conv_kernel_size),
                stride=1, padding=(0, (conv_kernel_size - 1) // 2), bias=False)
            self.v = nn.Linear(adim, 1, bias=False)
        elif atype == 'dot':
            self.w_key = nn.Linear(kdim, adim, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)
        elif atype == 'luong_dot':
            assert kdim == qdim
        elif atype == 'luong_general':
            self.w_key = nn.Linear(kdim, qdim, bias=False)
        elif atype == 'luong_concat':
            self.w = nn.Linear(kdim + qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)
        else:
            raise ValueError(atype)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, value, query, mask=None, aw_prev=None, cache=
        False, mode='', trigger_point=None):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            klens (IntTensor): `[B]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            cache (bool): cache key and mask
            mode: dummy interface for MoChA
            trigger_point (IntTensor): `[B]`
        Returns:
            cv (FloatTensor): `[B, 1, vdim]`
            aw (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            beta: dummy interface for MoChA

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        if aw_prev is None:
            aw_prev = key.new_zeros(bs, 1, klen)
        else:
            aw_prev = aw_prev.squeeze(1)
        if self.key is None or not cache:
            if self.atype in ['add', 'trigerred_attention', 'location',
                'dot', 'luong_general']:
                self.key = self.w_key(key)
            else:
                self.key = key
            self.mask = mask
            if mask is not None:
                assert self.mask.size() == (bs, 1, klen), (self.mask.size(),
                    (bs, 1, klen))
        if self.key.size(0) != query.size(0):
            self.key = self.key[0:1, :, :].repeat([query.size(0), 1, 1])
        if self.atype == 'no':
            raise NotImplementedError
        elif self.atype in ['add', 'triggered_attention']:
            tmp = self.key.unsqueeze(1) + self.w_query(query).unsqueeze(2)
            e = self.v(torch.tanh(tmp)).squeeze(3)
        elif self.atype == 'location':
            conv_feat = self.conv(aw_prev.unsqueeze(1)).squeeze(2)
            conv_feat = conv_feat.transpose(2, 1).contiguous().unsqueeze(1)
            tmp = self.key.unsqueeze(1) + self.w_query(query).unsqueeze(2)
            e = self.v(torch.tanh(tmp + self.w_conv(conv_feat))).squeeze(3)
        elif self.atype == 'dot':
            e = torch.bmm(self.w_query(query), self.key.transpose(2, 1))
        elif self.atype in ['luong_dot', 'luong_general']:
            e = torch.bmm(query, self.key.transpose(2, 1))
        elif self.atype == 'luong_concat':
            query = query.repeat([1, klen, 1])
            e = self.v(torch.tanh(self.w(torch.cat([self.key, query], dim=-1)))
                ).transpose(2, 1)
        assert e.size() == (bs, qlen, klen), (e.size(), (bs, qlen, klen))
        if self.atype == 'triggered_attention':
            assert trigger_point is not None
            for b in range(bs):
                e[(b), :, trigger_point[b] + self.lookahead + 1:] = NEG_INF
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        if self.sigmoid_smoothing:
            aw = torch.sigmoid(e) / torch.sigmoid(e).sum(-1).unsqueeze(-1)
        else:
            aw = torch.softmax(e * self.sharpening_factor, dim=-1)
        aw = self.dropout(aw)
        cv = torch.bmm(aw, value)
        return cv, aw.unsqueeze(1), None


class CausalConv1d(nn.Module):
    """1D dilated causal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, in_channels]`
        Returns:
            xs (FloatTensor): `[B, T, out_channels]`

        """
        xs = xs.transpose(2, 1)
        xs = self.conv1d(xs)
        if self.padding != 0:
            xs = xs[:, :, :-self.padding]
        xs = xs.transpose(2, 1).contiguous()
        return xs


class CIF(nn.Module):
    """docstring for CIF."""

    def __init__(self, enc_dim, conv_out_channels, conv_kernel_size,
        threshold=0.9):
        super(CIF, self).__init__()
        self.threshold = threshold
        self.channel = conv_out_channels
        self.n_heads = 1
        self.conv = nn.Conv1d(in_channels=enc_dim, out_channels=
            conv_out_channels, kernel_size=conv_kernel_size * 2 + 1, stride
            =1, padding=conv_kernel_size)
        self.proj = Linear(conv_out_channels, 1)

    def forward(self, eouts, elens, ylens=None, max_len=200):
        """

        Args:
            eouts (FloatTensor): `[B, T, enc_dim]`
            elens (IntTensor): `[B]`
            ylens (IntTensor): `[B]`
            max_len (int): the maximum length of target sequence
        Returns:
            eouts_fired (FloatTensor): `[B, T, enc_dim]`
            alpha (FloatTensor): `[B, T]`
            aws (FloatTensor): `[B, 1 (head), L. T]`

        """
        bs, xtime, enc_dim = eouts.size()
        conv_feat = self.conv(eouts.transpose(2, 1))
        conv_feat = conv_feat.transpose(2, 1)
        alpha = torch.sigmoid(self.proj(conv_feat)).squeeze(2)
        if ylens is not None:
            alpha_norm = alpha / alpha.sum(1).unsqueeze(1) * ylens.unsqueeze(1)
        else:
            alpha_norm = alpha
        if ylens is not None:
            max_len = ylens.max().int()
        eouts_fired = eouts.new_zeros(bs, max_len + 1, enc_dim)
        aws = eouts.new_zeros(bs, 1, max_len + 1, xtime)
        n_tokens = torch.zeros(bs, dtype=torch.int32)
        state = eouts.new_zeros(bs, self.channel)
        alpha_accum = eouts.new_zeros(bs)
        for t in range(xtime):
            alpha_accum += alpha_norm[:, (t)]
            for b in range(bs):
                if t > elens[b] - 1:
                    continue
                if ylens is not None and n_tokens[b] >= ylens[b].item():
                    continue
                if alpha_accum[b] >= self.threshold:
                    ak1 = 1 - alpha_accum[b]
                    ak2 = alpha_norm[b, t] - ak1
                    aws[b, 0, n_tokens[b], t] += ak1
                    eouts_fired[b, n_tokens[b]] = state[b] + ak1 * eouts[b, t]
                    n_tokens[b] += 1
                    state[b] = ak2 * eouts[b, t]
                    alpha_accum[b] = ak2
                    aws[b, 0, n_tokens[b], t] += ak2
                else:
                    state[b] += alpha_norm[b, t] * eouts[b, t]
                    aws[b, 0, n_tokens[b], t] += alpha_norm[b, t]
            if ylens is None and t == elens[b] - 1:
                if alpha_accum[b] >= 0.5:
                    n_tokens[b] += 1
                    eouts_fired[b, n_tokens[b]] = state[b]
        eouts_fired = eouts_fired[:, :max_len]
        aws = aws[:, :max_len]
        return eouts_fired, alpha, aws


class LinearGLUBlock(nn.Module):
    """A linear GLU block.

    Args:
        size (int): input and output dimension

    """

    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size * 2)

    def forward(self, xs):
        return F.glu(self.fc(xs), dim=-1)


class ConvGLUBlock(nn.Module):
    """A convolutional GLU block.

    Args:
        kernel_size (int): kernel size
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        bottlececk_dim (int): dimension of the bottleneck layers for computational efficiency
        dropout (float): dropout probability

    """

    def __init__(self, kernel_size, in_ch, out_ch, bottlececk_dim=0,
        dropout=0.0):
        super().__init__()
        self.conv_residual = None
        if in_ch != out_ch:
            self.conv_residual = nn.utils.weight_norm(nn.Conv2d(in_channels
                =in_ch, out_channels=out_ch, kernel_size=(1, 1)), name=
                'weight', dim=0)
            self.dropout_residual = nn.Dropout(p=dropout)
        self.pad_left = nn.ConstantPad2d((0, 0, kernel_size - 1, 0), 0)
        layers = OrderedDict()
        if bottlececk_dim == 0:
            layers['conv'] = nn.utils.weight_norm(nn.Conv2d(in_channels=
                in_ch, out_channels=out_ch * 2, kernel_size=(kernel_size, 1
                )), name='weight', dim=0)
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = nn.GLU()
        elif bottlececk_dim > 0:
            layers['conv_in'] = nn.utils.weight_norm(nn.Conv2d(in_channels=
                in_ch, out_channels=bottlececk_dim, kernel_size=(1, 1)),
                name='weight', dim=0)
            layers['dropout_in'] = nn.Dropout(p=dropout)
            layers['conv_bottleneck'] = nn.utils.weight_norm(nn.Conv2d(
                in_channels=bottlececk_dim, out_channels=bottlececk_dim,
                kernel_size=(kernel_size, 1)), name='weight', dim=0)
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = nn.GLU()
            layers['conv_out'] = nn.utils.weight_norm(nn.Conv2d(in_channels
                =bottlececk_dim, out_channels=out_ch * 2, kernel_size=(1, 1
                )), name='weight', dim=0)
            layers['dropout_out'] = nn.Dropout(p=dropout)
        self.layers = nn.Sequential(layers)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`
        """
        residual = xs
        if self.conv_residual is not None:
            residual = self.dropout_residual(self.conv_residual(residual))
        xs = self.pad_left(xs)
        xs = self.layers(xs)
        xs = xs + residual
        return xs


class GMMAttention(nn.Module):

    def __init__(self, kdim, qdim, adim, n_mixtures, vfloor=1e-06):
        """GMM attention.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of query
            adim: (int) dimension of the attention layer
            n_mixtures (int): number of mixtures
            vfloor (float):

        """
        super(GMMAttention, self).__init__()
        self.n_mix = n_mixtures
        self.n_heads = 1
        self.vfloor = vfloor
        self.mask = None
        self.myu = None
        self.ffn_gamma = nn.Linear(qdim, n_mixtures)
        self.ffn_beta = nn.Linear(qdim, n_mixtures)
        self.ffn_kappa = nn.Linear(qdim, n_mixtures)

    def reset(self):
        self.mask = None
        self.myu = None

    def forward(self, key, value, query, mask=None, aw_prev=None, cache=
        False, mode='', trigger_point=None):
        """Soft monotonic attention during training.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qmax, klen]`
            aw_prev (FloatTensor): `[B, klen, 1]`
            cache (bool): cache key and mask
            mode: dummy interface for MoChA
            trigger_point: dummy interface for MoChA
        Return:
            cv (FloatTensor): `[B, 1, vdim]`
            alpha (FloatTensor): `[B, klen, 1]`
            beta: dummy interface for MoChA

        """
        bs, klen = key.size()[:2]
        if self.myu is None:
            myu_prev = key.new_zeros(bs, 1, self.n_mix)
        else:
            myu_prev = self.myu
        self.mask = mask
        if self.mask is None:
            assert self.mask.size() == (bs, 1, klen), (self.mask.size(), (
                bs, 1, klen))
        w = torch.softmax(self.ffn_gamma(query), dim=-1)
        v = torch.exp(self.ffn_beta(query))
        myu = torch.exp(self.ffn_kappa(query)) + myu_prev
        self.myu = myu
        js = torch.arange(klen).unsqueeze(0).unsqueeze(2).repeat([bs, 1,
            self.n_mix]).float()
        device_id = torch.cuda.device_of(next(self.parameters())).idx
        if device_id >= 0:
            js = js.float()
        numerator = torch.exp(-torch.pow(js - myu, 2) / (2 * v + self.vfloor))
        denominator = torch.pow(2 * math.pi * v + self.vfloor, 0.5)
        aw = w * numerator / denominator
        aw = aw.sum(2).unsqueeze(1)
        if self.mask is not None:
            aw = aw.masked_fill_(self.mask == 0, NEG_INF)
        cv = torch.bmm(aw, value)
        return cv, aw.unsqueeze(2), None


def init_with_xavier_dist(n, p):
    if p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() in [2, 3]:
        nn.init.xavier_uniform_(p)
        logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
    else:
        raise ValueError(n)


class MonotonicEnergy(nn.Module):

    def __init__(self, kdim, qdim, adim, atype, n_heads, init_r, conv1d=
        False, conv_kernel_size=5, bias=True, param_init=''):
        """Energy function for the monotonic attenion.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of heads
            init_r (int): initial value for offset r
            conv1d (bool): use 1D causal convolution for energy calculation
            conv_kernel_size (int): kernel size for 1D convolution
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method

        """
        super().__init__()
        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.key = None
        self.mask = None
        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(adim)
        if atype == 'add':
            self.w_key = nn.Linear(kdim, adim)
            self.v = nn.Linear(adim, n_heads, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)
        elif atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        else:
            raise NotImplementedError(atype)
        self.r = nn.Parameter(torch.Tensor([init_r]))
        logger.info('init_r is initialized with %d' % init_r)
        self.conv1d = None
        if conv1d:
            self.conv1d = CausalConv1d(in_channels=kdim, out_channels=kdim,
                kernel_size=conv_kernel_size)
        if atype == 'add':
            self.v = nn.utils.weight_norm(self.v, name='weight', dim=0)
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == 'scaled_dot':
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)
        if self.conv1d is not None:
            logger.info(
                '===== Initialize %s with Xavier uniform distribution =====' %
                self.conv1d.__class__.__name__)
            for n, p in self.conv1d.named_parameters():
                init_with_xavier_dist(n, p)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False):
        """Compute monotonic energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Return:
            e (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)
        if self.key is None or not cache:
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key))
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads,
                    1, 1])
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), (
                    self.mask.size(), (bs, self.n_heads, qlen, klen))
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        query = query.transpose(2, 1).contiguous()
        if self.atype == 'add':
            key = self.key.unsqueeze(2)
            query = query.unsqueeze(3)
            e = torch.relu(key + query)
            e = e.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e = self.v(e).permute(0, 3, 1, 2)
        elif self.atype == 'scaled_dot':
            e = torch.matmul(query, self.key.transpose(3, 2)) / self.scale
        if self.r is not None:
            e = e + self.r
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        assert e.size() == (bs, self.n_heads, qlen, klen), (e.size(), (bs,
            self.n_heads, qlen, klen))
        return e


class ChunkEnergy(nn.Module):

    def __init__(self, kdim, qdim, adim, atype, n_heads=1, bias=True,
        param_init=''):
        """Energy function for the chunkwise attention.

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of quary
            adim (int): dimension of attention space
            atype (str): type of attention mechanism
            n_heads (int): number of heads
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method

        """
        super().__init__()
        self.key = None
        self.mask = None
        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(adim)
        if atype == 'add':
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)
        elif atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)
        else:
            raise NotImplementedError(atype)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False):
        """Compute chunkwise energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
        Return:
            energy (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, klen, kdim = key.size()
        qlen = query.size(1)
        if self.key is None or not cache:
            key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.key = key.transpose(2, 1).contiguous()
            self.mask = mask
            if mask is not None:
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads,
                    1, 1])
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), (
                    self.mask.size(), (bs, self.n_heads, qlen, klen))
        if self.atype == 'add':
            key = self.key.unsqueeze(2)
            query = self.w_query(query).unsqueeze(1).unsqueeze(3)
            energy = torch.relu(key + query)
            energy = self.v(energy).squeeze(4)
        elif self.atype == 'scaled_dot':
            query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
            query = query.transpose(2, 1).contiguous()
            energy = torch.matmul(query, self.key.transpose(3, 2)) / self.scale
        if self.mask is not None:
            energy = energy.masked_fill_(self.mask == 0, NEG_INF)
        assert energy.size() == (bs, self.n_heads, qlen, klen), (energy.
            size(), (bs, self.n_heads, qlen, klen))
        return energy


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.

    Args:
        x (FloatTensor): `[B, H_mono, H_chunk, qlen, klen]`
        back (int):
        forward (int):

    Returns:
        x_sum (FloatTensor): `[B, H_mono, H_chunk, qlen, klen]`

    """
    bs, n_heads_mono, n_heads_chunk, qlen, klen = x.size()
    x = x.view(-1, klen)
    x_padded = F.pad(x, pad=[back, forward])
    x_padded = x_padded.unsqueeze(1)
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum


def efficient_chunkwise_attention(alpha, e, mask, chunk_size, n_heads,
    sharpening_factor, chunk_len_dist=None):
    """Compute chunkwise attention distribution efficiently by clipping logits.

    Args:
        alpha (FloatTensor): `[B, H_mono, qlen, klen]`
        e (FloatTensor): `[B, H_chunk, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        n_heads (int): number of heads for chunkwise attention
        sharpening_factor (float):
        chunk_len_dist (IntTensor): `[B, H_mono, qlen]`
    Return
        beta (FloatTensor): `[B, H_mono * H_chunk, qlen, klen]`

    """
    bs, _, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)
    e = e.unsqueeze(1)
    if n_heads > 1:
        alpha = alpha.repeat([1, 1, n_heads, 1, 1])
    e -= torch.max(e, dim=-1, keepdim=True)[0]
    softmax_exp = torch.clamp(torch.exp(e), min=1e-05)
    if chunk_len_dist is not None:
        raise NotImplementedError
    elif chunk_size == -1:
        softmax_denominators = torch.cumsum(softmax_exp, dim=-1)
        beta = softmax_exp * moving_sum(alpha * sharpening_factor /
            softmax_denominators, back=0, forward=klen - 1)
    else:
        softmax_denominators = moving_sum(softmax_exp, back=chunk_size - 1,
            forward=0)
        beta = softmax_exp * moving_sum(alpha * sharpening_factor /
            softmax_denominators, back=0, forward=chunk_size - 1)
    return beta.view(bs, -1, qlen, klen)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), x.size(1), x.size
        (2), 1), x[:, :, :, :-1]], dim=-1), dim=-1)


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumsum(torch.cat([x.new_zeros(x.size(0), x.size(1), x.size
        (2), 1), x[:, :, :, :-1]], dim=-1), dim=-1)


def safe_cumprod(x, eps):
    """Numerically stable cumulative product by cumulative sum in log-space.
        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max
        =1.0))))


def add_gaussian_noise(xs, std):
    """Additive gaussian nosie to encourage discreteness."""
    noise = xs.new_zeros(xs.size()).normal_(std=std)
    return xs + noise


class MoChA(nn.Module):

    def __init__(self, kdim, qdim, adim, atype, chunk_size, n_heads_mono=1,
        n_heads_chunk=1, conv1d=False, init_r=-4, eps=1e-06, noise_std=1.0,
        no_denominator=False, sharpening_factor=1.0, dropout=0.0,
        dropout_head=0.0, bias=True, param_init='', decot=False, lookahead=2):
        """Monotonic chunk-wise attention.

            "Monotonic Chunkwise Attention" (ICLR 2018)
            https://openreview.net/forum?id=Hko85plCW
            "Monotonic Multihead Attention" (ICLR 2020)
            https://openreview.net/forum?id=Hyg96gBKPS

            if chunk_size == 1, this is equivalent to Hard monotonic attention
                "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                 https://arxiv.org/abs/1704.00784
            if chunk_size == -1, this is equivalent to Monotonic infinite lookback attention (Milk)
                "Monotonic Infinite Lookback Attention for Simultaneous Machine Translation" (ACL 2019)
                 https://arxiv.org/abs/1906.05218

        Args:
            kdim (int): dimension of key
            qdim (int): dimension of query
            adim: (int) dimension of the attention layer
            atype (str): type of attention mechanism
            chunk_size (int): window size for chunkwise attention
            n_heads_mono (int): number of heads for monotonic attention
            n_heads_chunk (int): number of heads for chunkwise attention
            conv1d (bool): apply 1d convolution for energy calculation
            init_r (int): initial value for parameter 'r' used for monotonic attention
            eps (float): epsilon parameter to avoid zero division
            noise_std (float): standard deviation for input noise
            no_denominator (bool): set the denominator to 1 in the alpha recurrence
            sharpening_factor (float): sharping factor for beta calculation
            dropout (float): dropout probability for attention weights
            dropout_head (float): dropout probability for heads
            bias (bool): use bias term in linear layers
            param_init (str): parameter initialization method
            decot (bool): delay constrainted training (DeCoT)
            lookahead (int): lookahead frames for DeCoT

        """
        super(MoChA, self).__init__()
        self.atype = atype
        assert adim % (n_heads_mono * n_heads_chunk) == 0
        self.d_k = adim // (n_heads_mono * n_heads_chunk)
        self.chunk_size = chunk_size
        self.milk = chunk_size == -1
        self.n_heads = n_heads_mono
        self.n_heads_mono = n_heads_mono
        self.n_heads_chunk = n_heads_chunk
        self.eps = eps
        self.noise_std = noise_std
        self.no_denom = no_denominator
        self.sharpening_factor = sharpening_factor
        self.decot = decot
        self.lookahead = lookahead
        self.monotonic_energy = MonotonicEnergy(kdim, qdim, adim, atype,
            n_heads_mono, init_r, conv1d, bias=bias, param_init=param_init)
        self.chunk_energy = ChunkEnergy(kdim, qdim, adim, atype,
            n_heads_chunk, bias, param_init
            ) if chunk_size > 1 or self.milk else None
        if n_heads_mono * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_out = nn.Linear(adim, kdim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_value.bias, 0.0)
        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

    def reset(self):
        self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()

    def forward(self, key, value, query, mask=None, aw_prev=None, mode=
        'hard', cache=False, trigger_point=None, eps_wait=-1,
        boundary_rightmost=None):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, H, 1, klen]`
            mode (str): recursive/parallel/hard
            cache (bool): cache key and mask
            trigger_point (IntTensor): `[B]`
            eps_wait (int): acceptable delay for MMA
            boundary_rightmost (int):
        Return:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_mono, qlen, klen]`
            beta (FloatTensor): `[B, H_chunk, qlen, klen]`

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        if aw_prev is None:
            aw_prev = key.new_zeros(bs, self.n_heads_mono, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.n_heads_mono, 1, 1)
        e_mono = self.monotonic_energy(key, query, mask, cache=cache)
        if mode == 'recursive':
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std)
                )
            alpha = []
            for i in range(qlen):
                shifted_1mp_choose = torch.cat([key.new_ones(bs, self.
                    n_heads_mono, 1, 1), 1 - p_choose[:, :, i:i + 1, :-1]],
                    dim=-1)
                q = key.new_zeros(bs, self.n_heads_mono, 1, klen + 1)
                for j in range(klen):
                    q[:, :, i:i + 1, (j + 1)] = shifted_1mp_choose[:, :, i:
                        i + 1, (j)].clone() * q[:, :, i:i + 1, (j)].clone(
                        ) + aw_prev[:, :, :, (j)].clone()
                aw_prev = p_choose[:, :, i:i + 1] * q[:, :, i:i + 1, 1:]
                alpha.append(aw_prev)
            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]
            alpha_masked = alpha.clone()
        elif mode == 'parallel':
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std)
                )
            cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=self.eps)
            alpha = []
            for i in range(qlen):
                denom = 1 if self.no_denom else torch.clamp(cumprod_1mp_choose
                    [:, :, i:i + 1], min=self.eps, max=1.0)
                aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :,
                    i:i + 1] * torch.cumsum(aw_prev / denom, dim=-1)
                if self.decot and trigger_point is not None:
                    for b in range(bs):
                        aw_prev[(b), :, :, trigger_point[b] + self.
                            lookahead + 1:] = 0
                alpha.append(aw_prev)
            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]
            alpha_masked = alpha.clone()
            if (self.n_heads_mono > 1 and self.dropout_head > 0 and self.
                training):
                n_effective_heads = self.n_heads_mono
                head_mask = alpha.new_ones(alpha.size()).byte()
                for h in range(self.n_heads_mono):
                    if random.random() < self.dropout_head:
                        head_mask[:, (h)] = 0
                        n_effective_heads -= 1
                alpha_masked = alpha_masked.masked_fill_(head_mask == 0, 0)
                if n_effective_heads > 0:
                    alpha_masked = alpha_masked * (self.n_heads_mono /
                        n_effective_heads)
        elif mode == 'hard':
            assert qlen == 1
            p_choose_i = (torch.sigmoid(e_mono) >= 0.5).float()[:, :, 0:1]
            p_choose_i *= torch.cumsum(aw_prev[:, :, 0:1], dim=-1)
            alpha = p_choose_i * exclusive_cumprod(1 - p_choose_i)
            vertical_latency = False
            if eps_wait > 0:
                for b in range(bs):
                    first_mma_layer = boundary_rightmost is None
                    if first_mma_layer or not vertical_latency:
                        boundary_threshold = alpha.size(-1) - 1
                    else:
                        boundary_threshold = min(alpha.size(-1) - 1, 
                            boundary_rightmost + eps_wait)
                    if alpha[b].sum() == 0:
                        if (vertical_latency and boundary_threshold < alpha
                            .size(-1) - 1):
                            alpha[(b), :, (0), (boundary_threshold)] = 1
                        continue
                    leftmost = alpha[(b), :, (0)].nonzero()[:, (-1)].min(
                        ).item()
                    rightmost = alpha[(b), :, (0)].nonzero()[:, (-1)].max(
                        ).item()
                    for h in range(self.n_heads_mono):
                        if alpha[b, h, 0].sum().item() == 0:
                            if first_mma_layer or not vertical_latency:
                                alpha[b, h, 0, min(rightmost, leftmost +
                                    eps_wait)] = 1
                            elif boundary_threshold < alpha.size(-1) - 1:
                                alpha[b, h, 0, boundary_threshold] = 1
                            continue
                        if first_mma_layer or not vertical_latency:
                            if alpha[b, h, 0].nonzero()[:, (-1)].min().item(
                                ) >= leftmost + eps_wait:
                                alpha[(b), (h), (0), :] = 0
                                alpha[b, h, 0, leftmost + eps_wait] = 1
                        elif alpha[b, h, 0].nonzero()[:, (-1)].min().item(
                            ) > boundary_threshold:
                            alpha[(b), (h), (0), :] = 0
                            alpha[b, h, 0, boundary_threshold] = 1
            alpha_masked = alpha.clone()
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'."
                )
        beta = None
        if self.chunk_size > 1 or self.milk:
            e_chunk = self.chunk_energy(key, query, mask, cache=cache)
            beta = efficient_chunkwise_attention(alpha_masked, e_chunk,
                mask, self.chunk_size, self.n_heads_chunk, self.
                sharpening_factor)
            beta = self.dropout(beta)
        if self.n_heads_mono * self.n_heads_chunk > 1:
            value = self.w_value(value).view(bs, -1, self.n_heads_mono *
                self.n_heads_chunk, self.d_k)
            value = value.transpose(2, 1).contiguous()
            if self.chunk_size == 1:
                cv = torch.matmul(alpha, value)
            else:
                cv = torch.matmul(beta, value)
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.
                n_heads_mono * self.n_heads_chunk * self.d_k)
            cv = self.w_out(cv)
        elif self.chunk_size == 1:
            cv = torch.bmm(alpha.squeeze(1), value)
        else:
            cv = torch.bmm(beta.squeeze(1), value)
        assert alpha.size() == (bs, self.n_heads_mono, qlen, klen), (alpha.
            size(), (bs, self.n_heads_mono, qlen, klen))
        if self.chunk_size > 1 or self.milk:
            assert beta.size() == (bs, self.n_heads_mono * self.
                n_heads_chunk, qlen, klen), (beta.size(), (bs, self.
                n_heads_mono * self.n_heads_chunk, qlen, klen))
        return cv, alpha, beta


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        atype (str): type of attention mechanism
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method

    """

    def __init__(self, kdim, qdim, adim, n_heads, dropout, atype=
        'scaled_dot', bias=True, param_init=''):
        super(MultiheadAttentionMechanism, self).__init__()
        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.reset()
        self.dropout = nn.Dropout(p=dropout)
        if atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        elif atype == 'add':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            self.v = nn.Linear(adim, n_heads, bias=bias)
        else:
            raise NotImplementedError(atype)
        self.w_out = nn.Linear(adim, kdim, bias=bias)
        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_value.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)
        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

    def reset(self):
        self.key = None
        self.value = None
        self.mask = None

    def forward(self, key, value, query, mask, aw_prev=None, cache=False,
        mode='', trigger_point=None, eps_wait=-1, boundary_rightmost=None):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev: dummy interface
            cache (bool): cache key, value, and mask
            mode: dummy interface for MoChA
            trigger_point: dummy interface for MoChA
            eps_wait: dummy interface for MMA
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen]`
            beta: dummy interface for MoChA

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        if self.key is None or not cache:
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.value = self.w_value(value).view(bs, -1, self.n_heads,
                self.d_k)
            self.mask = mask
            if self.mask is not None:
                self.mask = self.mask.unsqueeze(3).repeat([1, 1, 1, self.
                    n_heads])
                assert self.mask.size() == (bs, qlen, klen, self.n_heads), (
                    self.mask.size(), (bs, qlen, klen, self.n_heads))
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        if self.atype == 'scaled_dot':
            e = torch.einsum('bihd,bjhd->bijh', (query, self.key)) / self.scale
        elif self.atype == 'add':
            key = self.key.unsqueeze(1)
            query = query.unsqueeze(2)
            e = self.v(torch.tanh(key + query)).squeeze(4)
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        aw = torch.softmax(e, dim=2)
        aw = self.dropout(aw)
        cv = torch.einsum('bijh,bjhd->bihd', (aw, self.value))
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)
        return cv, aw, None


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        dropout (float): dropout probability
        pe_type (str): type of positional encoding
        param_init (str): parameter initialization method
        max_len (int):
        conv_kernel_size (int):
        layer_norm_eps (float):

    """

    def __init__(self, d_model, dropout, pe_type, param_init, max_len=5000,
        conv_kernel_size=3, layer_norm_eps=1e-12):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe_type = pe_type
        self.scale = math.sqrt(self.d_model)
        if '1dconv' in pe_type:
            causal_conv1d = CausalConv1d(in_channels=d_model, out_channels=
                d_model, kernel_size=conv_kernel_size)
            layers = []
            conv_nlayers = int(pe_type.replace('1dconv', '')[0])
            for l in range(conv_nlayers):
                layers.append(copy.deepcopy(causal_conv1d))
                layers.append(nn.LayerNorm(d_model, eps=layer_norm_eps))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            self.pe = nn.Sequential(*layers)
            if param_init == 'xavier_uniform':
                self.reset_parameters()
        elif pe_type != 'none':
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(
                1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(
                math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.dropout = nn.Dropout(p=dropout)
        logger.info('Positional encoding: %s' % pe_type)

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        for layer in self.pe:
            if isinstance(layer, CausalConv1d):
                for n, p in layer.named_parameters():
                    init_with_xavier_dist(n, p)

    def forward(self, xs, scale=True):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        if scale:
            xs = xs * self.scale
        if self.pe_type == 'none':
            return xs
        elif self.pe_type == 'add':
            xs = xs + self.pe[:, :xs.size(1)]
            xs = self.dropout(xs)
        elif self.pe_type == 'concat':
            xs = torch.cat([xs, self.pe[:, :xs.size(1)]], dim=-1)
            xs = self.dropout(xs)
        elif '1dconv' in self.pe_type:
            xs = self.pe(xs)
        else:
            raise NotImplementedError(self.pe_type)
        return xs


class XLPositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout):
        """Positional embedding for TransformerXL."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1 / 10000 ** (torch.arange(0.0, d_model, 2.0) / d_model)
        self.register_buffer('inv_freq', inv_freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, positions, device_id):
        """Forward computation.

        Args:
            positions (LongTensor): `[L]`
        Returns:
            pos_emb (LongTensor): `[L, 1 d_model]`

        """
        if device_id >= 0:
            positions = positions
        sinusoid_inp = torch.einsum('i,j->ij', positions.float(), self.inv_freq
            )
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = self.dropout(pos_emb)
        return pos_emb.unsqueeze(1)


def gelu_accurate(x):
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 *
        torch.pow(x, 3))))


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network (FFN) layer.

    Args:
        d_model (int): input and output dimension
        d_ff (int): hidden dimension
        dropout (float): dropout probability
        activation: non-linear function
        param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, dropout, activation, param_init):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = lambda x: gelu(x)
        elif activation == 'gelu_accurate':
            self.activation = lambda x: gelu_accurate(x)
        elif activation == 'glu':
            self.activation = LinearGLUBlock(d_ff)
        else:
            raise NotImplementedError(activation)
        logger.info('FFN activation: %s' % activation)
        if param_init == 'xavier_uniform':
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.constant_(self.w_1.bias, 0.0)
        nn.init.constant_(self.w_2.bias, 0.0)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class RelativeMultiheadAttentionMechanism(nn.Module):
    """Relative multi-head attention layer for TransformerXL.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method

    """

    def __init__(self, kdim, qdim, adim, n_heads, dropout, bias=True,
        param_init=''):
        super(RelativeMultiheadAttentionMechanism, self).__init__()
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.dropout = nn.Dropout(p=dropout)
        self.w_key = nn.Linear(kdim, adim, bias=bias)
        self.w_value = nn.Linear(kdim, adim, bias=bias)
        self.w_query = nn.Linear(qdim, adim, bias=bias)
        self.w_position = nn.Linear(qdim, adim, bias=bias)
        self.w_out = nn.Linear(adim, kdim, bias=bias)
        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_value.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)
        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

    def _rel_shift(self, xs):
        """Calculate relative positional attention efficiently.

        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`

        """
        bs, qlen, klen, n_heads = xs.size()
        xs = xs.permute(1, 2, 0, 3).contiguous().view(qlen, klen, bs * n_heads)
        zero_pad = xs.new_zeros((qlen, 1, bs * n_heads))
        xs_shifted = torch.cat([zero_pad, xs], dim=1).view(klen + 1, qlen, 
            bs * n_heads)[1:].view_as(xs)
        return xs_shifted.view(qlen, klen, bs, n_heads).permute(2, 0, 1, 3)

    def forward(self, key, query, memory, pos_embs, mask, u=None, v=None):
        """Forward computation.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            memory (FloatTensor): `[B, mlen, d_model]`
            mask (ByteTensor): `[B, qlen, klen+mlen]`
            pos_embs (LongTensor): `[qlen, 1, d_model]`
            u (nn.Parameter): `[H, d_k]`
            v (nn.Parameter): `[H, d_k]`
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen+mlen]`

        """
        bs, qlen = query.size()[:2]
        klen = key.size(1)
        mlen = memory.size(1) if memory.dim() > 1 else 0
        if mlen > 0:
            key = torch.cat([memory, key], dim=1)
        value = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)
        key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + klen, self.n_heads), (mask
                .size(), (bs, qlen, klen + mlen, self.n_heads))
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        pos_embs = self.w_position(pos_embs)
        pos_embs = pos_embs.view(-1, self.n_heads, self.d_k)
        if u is not None:
            AC = torch.einsum('bihd,bjhd->bijh', (query + u[None, None], key))
        else:
            AC = torch.einsum('bihd,bjhd->bijh', (query, key))
        if v is not None:
            BD = torch.einsum('bihd,jhd->bijh', (query + v[None, None],
                pos_embs))
        else:
            BD = torch.einsum('bihd,jhd->bijh', (query, pos_embs))
        BD = self._rel_shift(BD)
        e = (AC + BD) / self.scale
        if mask is not None:
            e = e.masked_fill_(mask == 0, NEG_INF)
        aw = torch.softmax(e, dim=2)
        aw = self.dropout(aw)
        cv = torch.einsum('bijh,bjhd->bihd', (aw, value))
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)
        return cv, aw


class SyncBidirMultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention space
        n_heads (int): number of heads
        dropout (float): dropout probability
        bias (bool): use bias term in linear layers
        atype (str): type of attention mechanisms
        param_init (str): parameter initialization method
        future_weight (float):

    """

    def __init__(self, kdim, qdim, adim, n_heads, dropout, atype=
        'scaled_dot', bias=True, param_init='', future_weight=0.1):
        super(SyncBidirMultiheadAttentionMechanism, self).__init__()
        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.future_weight = future_weight
        self.reset()
        self.dropout = nn.Dropout(p=dropout)
        if atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
        elif atype == 'add':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            self.v = nn.Linear(adim, n_heads, bias=bias)
        else:
            raise NotImplementedError(atype)
        self.w_out = nn.Linear(adim, kdim, bias=bias)
        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info(
            '===== Initialize %s with Xavier uniform distribution =====' %
            self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_value.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)
        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

    def reset(self):
        self.key_fwd = None
        self.key_bwd = None
        self.value_fwd = None
        self.value_bwd = None
        self.tgt_mask = None
        self.identity_mask = None

    def forward(self, key_fwd, value_fwd, query_fwd, key_bwd, value_bwd,
        query_bwd, tgt_mask, identity_mask, mode='', cache=True,
        trigger_point=None):
        """Forward computation.

        Args:
            key_fwd (FloatTensor): `[B, klen, kdim]`
            value_fwd (FloatTensor): `[B, klen, vdim]`
            query_fwd (FloatTensor): `[B, qlen, qdim]`
            key_bwd (FloatTensor): `[B, klen, kdim]`
            value_bwd (FloatTensor): `[B, klen, vdim]`
            query_bwd (FloatTensor): `[B, qlen, qdim]`
            tgt_mask (ByteTensor): `[B, qlen, klen]`
            identity_mask (ByteTensor): `[B, qlen, klen]`
            mode: dummy interface for MoChA
            cache (bool): cache key, value, and tgt_mask
            trigger_point (IntTensor): dummy
        Returns:
            cv_fwd (FloatTensor): `[B, qlen, vdim]`
            cv_bwd (FloatTensor): `[B, qlen, vdim]`
            aw_fwd_h (FloatTensor): `[B, H, qlen, klen]`
            aw_fwd_f (FloatTensor): `[B, H, qlen, klen]`
            aw_bwd_h (FloatTensor): `[B, H, qlen, klen]`
            aw_bwd_f (FloatTensor): `[B, H, qlen, klen]`

        """
        bs, klen = key_fwd.size()[:2]
        qlen = query_fwd.size(1)
        if self.key_fwd is None or not cache:
            key_fwd = self.w_key(key_fwd).view(bs, -1, self.n_heads, self.d_k)
            self.key_fwd = key_fwd.transpose(2, 1).contiguous()
            value_fwd = self.w_value(value_fwd).view(bs, -1, self.n_heads,
                self.d_k)
            self.value_fwd = value_fwd.transpose(2, 1).contiguous()
            self.tgt_mask = tgt_mask
            self.identity_mask = identity_mask
            if tgt_mask is not None:
                self.tgt_mask = tgt_mask.unsqueeze(1).repeat([1, self.
                    n_heads, 1, 1])
                assert self.tgt_mask.size() == (bs, self.n_heads, qlen, klen)
            if identity_mask is not None:
                self.identity_mask = identity_mask.unsqueeze(1).repeat([1,
                    self.n_heads, 1, 1])
                assert self.identity_mask.size() == (bs, self.n_heads, qlen,
                    klen)
        if self.key_bwd is None or not cache:
            key_bwd = self.w_key(key_bwd).view(bs, -1, self.n_heads, self.d_k)
            self.key_bwd = key_bwd.transpose(2, 1).contiguous()
            value_bwd = self.w_value(value_bwd).view(bs, -1, self.n_heads,
                self.d_k)
            self.value_bwd = value_bwd.transpose(2, 1).contiguous()
        query_fwd = self.w_query(query_fwd).view(bs, -1, self.n_heads, self.d_k
            )
        query_fwd = query_fwd.transpose(2, 1).contiguous()
        query_bwd = self.w_query(query_bwd).view(bs, -1, self.n_heads, self.d_k
            )
        query_bwd = query_bwd.transpose(2, 1).contiguous()
        if self.atype == 'scaled_dot':
            e_fwd_h = torch.matmul(query_fwd, self.key_fwd.transpose(3, 2)
                ) / self.scale
            e_fwd_f = torch.matmul(query_fwd, self.key_bwd.transpose(3, 2)
                ) / self.scale
            e_bwd_h = torch.matmul(query_bwd, self.key_bwd.transpose(3, 2)
                ) / self.scale
            e_bwd_f = torch.matmul(query_bwd, self.key_fwd.transpose(3, 2)
                ) / self.scale
        elif self.atype == 'add':
            e_fwd_h = torch.tanh(self.key_fwd.unsqueeze(2) + query_fwd.
                unsqueeze(3))
            e_fwd_h = e_fwd_h.permute(0, 2, 3, 1, 4).contiguous().view(bs,
                qlen, klen, -1)
            e_fwd_h = self.v(e_fwd_h).permute(0, 3, 1, 2)
            e_fwd_f = torch.tanh(self.key_bwd.unsqueeze(2) + query_fwd.
                unsqueeze(3))
            e_fwd_f = e_fwd_f.permute(0, 2, 3, 1, 4).contiguous().view(bs,
                qlen, klen, -1)
            e_fwd_f = self.v(e_fwd_f).permute(0, 3, 1, 2)
            e_bwd_h = torch.tanh(self.key_bwd.unsqueeze(2) + query_bwd.
                unsqueeze(3))
            e_bwd_h = e_bwd_h.permute(0, 2, 3, 1, 4).contiguous().view(bs,
                qlen, klen, -1)
            e_bwd_h = self.v(e_bwd_h).permute(0, 3, 1, 2)
            e_bwd_f = torch.tanh(self.key_fwd.unsqueeze(2) + query_bwd.
                unsqueeze(3))
            e_bwd_f = e_bwd_f.permute(0, 2, 3, 1, 4).contiguous().view(bs,
                qlen, klen, -1)
            e_bwd_f = self.v(e_bwd_f).permute(0, 3, 1, 2)
        if self.tgt_mask is not None:
            e_fwd_h = e_fwd_h.masked_fill_(self.tgt_mask == 0, NEG_INF)
            e_bwd_h = e_bwd_h.masked_fill_(self.tgt_mask == 0, NEG_INF)
        if self.identity_mask is not None:
            e_fwd_f = e_fwd_f.masked_fill_(self.identity_mask == 0, NEG_INF)
            e_bwd_f = e_bwd_f.masked_fill_(self.identity_mask == 0, NEG_INF)
        aw_fwd_h = self.dropout(torch.softmax(e_fwd_h, dim=-1))
        aw_fwd_f = self.dropout(torch.softmax(e_fwd_f, dim=-1))
        aw_bwd_h = self.dropout(torch.softmax(e_bwd_h, dim=-1))
        aw_bwd_f = self.dropout(torch.softmax(e_bwd_f, dim=-1))
        cv_fwd_h = torch.matmul(aw_fwd_h, self.value_fwd)
        cv_fwd_f = torch.matmul(aw_fwd_f, self.value_bwd)
        cv_bwd_h = torch.matmul(aw_bwd_h, self.value_bwd)
        cv_bwd_f = torch.matmul(aw_bwd_f, self.value_fwd)
        cv_fwd_h = cv_fwd_h.transpose(2, 1).contiguous().view(bs, -1, self.
            n_heads * self.d_k)
        cv_fwd_h = self.w_out(cv_fwd_h)
        cv_fwd_f = cv_fwd_f.transpose(2, 1).contiguous().view(bs, -1, self.
            n_heads * self.d_k)
        cv_fwd_f = self.w_out(cv_fwd_f)
        cv_bwd_h = cv_bwd_h.transpose(2, 1).contiguous().view(bs, -1, self.
            n_heads * self.d_k)
        cv_bwd_h = self.w_out(cv_bwd_h)
        cv_bwd_f = cv_bwd_f.transpose(2, 1).contiguous().view(bs, -1, self.
            n_heads * self.d_k)
        cv_bwd_f = self.w_out(cv_bwd_f)
        cv_fwd = cv_fwd_h + self.future_weight * torch.tanh(cv_fwd_f)
        cv_bwd = cv_bwd_h + self.future_weight * torch.tanh(cv_bwd_f)
        return cv_fwd, cv_bwd, aw_fwd_h, aw_fwd_f, aw_bwd_h, aw_bwd_f


class TransformerEncoderBlock(nn.Module):
    """A single layer of the Transformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        atype (str): type of attention mechanism
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probabilities for layers
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        memory_transformer (bool): streaming TransformerXL encoder

    """

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att,
        dropout_layer, layer_norm_eps, ffn_activation, param_init,
        memory_transformer=False):
        super(TransformerEncoderBlock, self).__init__()
        self.n_heads = n_heads
        self.memory_transformer = memory_transformer
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model, qdim=d_model, adim=d_model,
            n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation,
            param_init)
        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer

    def forward(self, xs, xx_mask=None, pos_embs=None, memory=None, u=None,
        v=None):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T, T]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            memory (FloatTensor): `[B, L_prev, d_model]`
            u (FloatTensor): global parameter for TransformerXL
            v (FloatTensor): global parameter for TransformerXL
        Returns:
            xs (FloatTensor): `[B, T, d_model]`
            xx_aws (FloatTensor): `[B, H, T, T]`

        """
        if self.dropout_layer > 0 and self.training and random.random(
            ) >= self.dropout_layer:
            return xs, None
        residual = xs
        xs = self.norm1(xs)
        if self.memory_transformer:
            xs, xx_aws = self.self_attn(xs, xs, memory, pos_embs, xx_mask, u, v
                )
        elif memory is not None:
            xs_memory = torch.cat([memory, xs], dim=1)
            xs, xx_aws, _ = self.self_attn(xs_memory, xs_memory, xs, mask=
                xx_mask)
        else:
            xs, xx_aws, _ = self.self_attn(xs, xs, xs, mask=xx_mask)
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm2(xs)
        xs = self.feed_forward(xs)
        xs = self.dropout(xs) + residual
        return xs, xx_aws


class TransformerDecoderBlock(nn.Module):
    """A single layer of the Transformer decoder.

        Args:
            d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
            d_ff (int): hidden dimension of PositionwiseFeedForward
            atype (str): type of attention mechanism
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            dropout_layer (float): LayerDrop probabilities for layers
            dropout_head (float): HeadDrop probabilities for attention heads
            layer_norm_eps (float): epsilon parameter for layer normalization
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str): parameter initialization method
            src_tgt_attention (bool): if False, ignore source-target attention
            memory_transformer (bool): TransformerXL decoder
            mocha_chunk_size (int): chunk size for MoChA. -1 means infinite lookback.
            mocha_n_heads_mono (int): number of heads for monotonic attention
            mocha_n_heads_chunk (int): number of heads for chunkwise attention
            mocha_init_r (int):
            mocha_eps (float):
            mocha_std (float):
            mocha_no_denominator (bool):
            mocha_1dconv (bool):
            lm_fusion (bool):

    """

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att,
        dropout_layer, layer_norm_eps, ffn_activation, param_init,
        src_tgt_attention=True, memory_transformer=False, mocha_chunk_size=
        0, mocha_n_heads_mono=1, mocha_n_heads_chunk=1, mocha_init_r=2,
        mocha_eps=1e-06, mocha_std=1.0, mocha_no_denominator=False,
        mocha_1dconv=False, dropout_head=0, lm_fusion=False):
        super(TransformerDecoderBlock, self).__init__()
        self.atype = atype
        self.n_heads = n_heads
        self.src_tgt_attention = src_tgt_attention
        self.memory_transformer = memory_transformer
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model, qdim=d_model, adim=d_model,
            n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if 'mocha' in atype:
                self.n_heads = mocha_n_heads_monoNone
                self.src_attn = MoChA(kdim=d_model, qdim=d_model, adim=
                    d_model, atype='scaled_dot', chunk_size=
                    mocha_chunk_size, n_heads_mono=mocha_n_heads_mono,
                    n_heads_chunk=mocha_n_heads_chunk, init_r=mocha_init_r,
                    eps=mocha_eps, noise_std=mocha_std, no_denominator=
                    mocha_no_denominator, conv1d=mocha_1dconv, dropout=
                    dropout_att, dropout_head=dropout_head, param_init=
                    param_init)
            else:
                self.src_attn = MHA(kdim=d_model, qdim=d_model, adim=
                    d_model, n_heads=n_heads, dropout=dropout_att,
                    param_init=param_init)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation,
            param_init)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_layer = dropout_layer
        self.lm_fusion = lm_fusion
        if lm_fusion:
            self.norm_lm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.linear_lm_feat = nn.Linear(d_model, d_model)
            self.linear_lm_gate = nn.Linear(d_model * 2, d_model)
            self.linear_lm_fusion = nn.Linear(d_model * 2, d_model)
            if 'attention' in lm_fusion:
                self.lm_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model,
                    n_heads=n_heads, dropout=dropout_att, param_init=param_init
                    )

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None,
        xy_aws_prev=None, mode='hard', lmout=None, pos_embs=None, memory=
        None, u=None, v=None, eps_wait=-1, boundary_rightmost=None):
        """Transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L (query), L (key)]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            xy_aws_prev (FloatTensor): `[B, H, L, T]`
            mode (str):
            lmout (FloatTensor): `[B, L, d_model]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            memory (FloatTensor): `[B, L_prev, d_model]`
            u (FloatTensor): global parameter for TransformerXL
            v (FloatTensor): global parameter for TransformerXL
            eps_wait (int):
            boundary_rightmost (int):
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aws (FloatTensor)`[B, H, L, L]`
            xy_aws (FloatTensor): `[B, H, L, T]`
            xy_aws_beta (FloatTensor): `[B, H, L, T]`

        """
        if self.dropout_layer > 0 and self.training and random.random(
            ) >= self.dropout_layer:
            xy_aws = None
            if self.src_tgt_attention:
                bs, qlen, klen = xy_mask.size()
                xy_aws = ys.new_zeros(bs, self.n_heads, qlen, klen)
            return ys, None, xy_aws, None, None
        residual = ys
        ys = self.norm1(ys)
        if cache is not None:
            ys_q = ys[:, -1:]
            residual = residual[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys
        yy_aws = None
        if self.memory_transformer:
            if cache is not None:
                pos_embs = pos_embs[-ys_q.size(1):]
            out, yy_aws = self.self_attn(ys, ys_q, memory, pos_embs,
                yy_mask, u, v)
        else:
            out, yy_aws, _ = self.self_attn(ys, ys, ys_q, mask=yy_mask)
        out = self.dropout(out) + residual
        xy_aws, xy_aws_beta = None, None
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, xy_aws, xy_aws_beta = self.src_attn(xs, xs, out, mask=
                xy_mask, aw_prev=xy_aws_prev, mode=mode, eps_wait=eps_wait,
                boundary_rightmost=boundary_rightmost)
            out = self.dropout(out) + residual
        yy_aws_lm = None
        if self.lm_fusion:
            residual = out
            out = self.norm_lm(out)
            lmout = self.linear_lm_feat(lmout)
            if 'attention' in self.lm_fusion:
                out, yy_aws_lm, _ = self.lm_attn(lmout, lmout, out, mask=
                    yy_mask)
            gate = torch.sigmoid(self.linear_lm_gate(torch.cat([out, lmout],
                dim=-1)))
            gated_lmout = gate * lmout
            out = self.linear_lm_fusion(torch.cat([out, gated_lmout], dim=-1))
            out = self.dropout(out) + residual
        residual = out
        out = self.norm3(out)
        out = self.feed_forward(out)
        out = self.dropout(out) + residual
        if cache is not None:
            out = torch.cat([cache, out], dim=1)
        return out, yy_aws, xy_aws, xy_aws_beta, yy_aws_lm


class ZoneoutCell(nn.Module):

    def __init__(self, cell, zoneout_prob_h, zoneout_prob_c):
        super(ZoneoutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        if not isinstance(cell, nn.RNNCellBase):
            raise TypeError('The cell is not a LSTMCell or GRUCell!')
        if isinstance(cell, nn.LSTMCell):
            self.prob = zoneout_prob_h, zoneout_prob_c
        else:
            self.prob = zoneout_prob_h

    def forward(self, inputs, state):
        return self.zoneout(state, self.cell(inputs, state), self.prob)

    def zoneout(self, state, next_state, prob):
        if isinstance(state, tuple):
            return self.zoneout(state[0], next_state[0], prob[0]
                ), self.zoneout(state[1], next_state[1], prob[1])
        mask = state.new(state.size()).bernoulli_(prob)
        if self.training:
            return mask * next_state + (1 - mask) * state
        else:
            return prob * next_state + (1 - prob) * state


class LayerNorm2D(nn.Module):
    """Layer normalization for CNN outputs."""

    def __init__(self, dim, eps=1e-12):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, xs):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, out_ch, T, feat_dim]`
        Returns:
            xs (FloatTensor): `[B, out_ch, T, feat_dim]`

        """
        bs, out_ch, xmax, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, xmax, out_ch * feat_dim)
        xs = self.norm(xs)
        xs = xs.view(bs, xmax, out_ch, feat_dim).transpose(2, 1)
        return xs


class Padding(nn.Module):
    """Padding variable length of sequences."""

    def __init__(self, bidirectional_sum_fwd_bwd):
        super(Padding, self).__init__()
        self.bidir_sum = bidirectional_sum_fwd_bwd

    def forward(self, xs, xlens, rnn, prev_state=None):
        xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
        xs, state = rnn(xs, hx=prev_state)
        xs = pad_packed_sequence(xs, batch_first=True)[0]
        if self.bidir_sum:
            assert rnn.bidirectional
            half = xs.size(-1) // 2
            xs = xs[:, :, :half] + xs[:, :, half:]
        return xs, state


class MaxpoolSubsampler(nn.Module):
    """Subsample by max-pooling input frames."""

    def __init__(self, factor):
        super(MaxpoolSubsampler, self).__init__()
        self.factor = factor
        if factor > 1:
            self.max_pool = nn.MaxPool1d(1, stride=factor, ceil_mode=True)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens
        xs = self.max_pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()
        xlens //= self.factor
        return xs, xlens


class Conv1dSubsampler(nn.Module):
    """Subsample by 1d convolution and max-pooling."""

    def __init__(self, factor, n_units, conv_kernel_size=5):
        super(Conv1dSubsampler, self).__init__()
        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.factor = factor
        if factor > 1:
            self.conv1d = nn.Conv1d(in_channels=n_units, out_channels=
                n_units, kernel_size=conv_kernel_size, stride=1, padding=(
                conv_kernel_size - 1) // 2)
            self.max_pool = nn.MaxPool1d(1, stride=factor, ceil_mode=True)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens
        xs = torch.relu(self.conv1d(xs.transpose(2, 1)))
        xs = self.max_pool(xs).transpose(2, 1).contiguous()
        xlens //= self.factor
        return xs, xlens


class DropSubsampler(nn.Module):
    """Subsample by droping input frames."""

    def __init__(self, factor):
        super(DropSubsampler, self).__init__()
        self.factor = factor

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens
        xs = xs[:, ::self.factor, :]
        xlens = [max(1, (i + self.factor - 1) // self.factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class ConcatSubsampler(nn.Module):
    """Subsample by concatenating successive input frames."""

    def __init__(self, factor, n_units):
        super(ConcatSubsampler, self).__init__()
        self.factor = factor
        if factor > 1:
            self.proj = nn.Linear(n_units * factor, n_units)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens
        xs = xs.transpose(1, 0).contiguous()
        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.factor - 1,
            -1, -1)], dim=-1) for t in range(xs.size(0)) if (t + 1) % self.
            factor == 0]
        xs = torch.cat(xs, dim=0).transpose(1, 0)
        xs = torch.relu(self.proj(xs))
        xlens //= self.factor
        return xs, xlens


class NiN(nn.Module):
    """Network in network."""

    def __init__(self, dim):
        super(NiN, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim,
            kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(dim)

    def forward(self, xs):
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)
        xs = torch.relu(self.batch_norm(self.conv(xs)))
        xs = xs.transpose(2, 1).squeeze(3)
        return xs


class TDSBlock(nn.Module):
    """TDS block.

    Args:
        channel (int):
        kernel_size (int):
        in_freq (int):
        dropout (float):

    """

    def __init__(self, channel, kernel_size, in_freq, dropout):
        super().__init__()
        self.channel = channel
        self.in_freq = in_freq
        self.conv2d = nn.Conv2d(in_channels=channel, out_channels=channel,
            kernel_size=(kernel_size, 1), stride=(1, 1), padding=(
            kernel_size // 2, 0))
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(in_freq * channel, eps=1e-06)
        self.conv1d_1 = nn.Conv2d(in_channels=in_freq * channel,
            out_channels=in_freq * channel, kernel_size=1, stride=1, padding=0)
        self.dropout2_1 = nn.Dropout(p=dropout)
        self.conv1d_2 = nn.Conv2d(in_channels=in_freq * channel,
            out_channels=in_freq * channel, kernel_size=1, stride=1, padding=0)
        self.dropout2_2 = nn.Dropout(p=dropout)
        self.layer_norm2 = nn.LayerNorm(in_freq * channel, eps=1e-06)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`

        """
        bs, _, time, _ = xs.size()
        residual = xs
        xs = self.conv2d(xs)
        xs = torch.relu(xs)
        self.dropout1(xs)
        xs = xs + residual
        bs, out_ch, time, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)
        xs = self.layer_norm1(xs)
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)
        residual = xs
        xs = self.conv1d_1(xs)
        xs = torch.relu(xs)
        self.dropout2_1(xs)
        xs = self.conv1d_2(xs)
        self.dropout2_2(xs)
        xs = xs + residual
        xs = xs.unsqueeze(3)
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)
        xs = self.layer_norm2(xs)
        xs = xs.view(bs, time, out_ch, feat_dim).contiguous().transpose(2, 1)
        return xs


class SubsampelBlock(nn.Module):

    def __init__(self, in_channel, out_channel, in_freq, dropout):
        super().__init__()
        self.conv1d = nn.Conv2d(in_channels=in_channel, out_channels=
            out_channel, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(in_freq * out_channel, eps=1e-06)

    def forward(self, xs):
        """Forward computation.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`

        """
        bs, _, time, _ = xs.size()
        xs = self.conv1d(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        bs, out_ch, time, feat_dim = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)
        xs = self.layer_norm(xs)
        xs = xs.view(bs, time, out_ch, feat_dim).contiguous().transpose(2, 1)
        return xs


def make_pad_mask(seq_lens, device_id=-1):
    """Make mask for padding.

    Args:
        seq_lens (IntTensor): `[B]`
        device_id (int):
    Returns:
        mask (IntTensor): `[B, T]`

    """
    bs = seq_lens.size(0)
    max_time = max(seq_lens)
    seq_range = torch.arange(0, max_time, dtype=torch.int32)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
    seq_length_expand = seq_range_expand.new(seq_lens).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand
    if device_id >= 0:
        mask = mask.cuda(device_id)
    return mask


class SequenceSummaryNetwork(nn.Module):
    """Sequence summary network.

    Args:
        input_dim (int): dimension of input features
        n_units (int):
        n_layers (int):
        bottleneck_dim (int): dimension of the last bottleneck layer
        dropout (float): dropout probability
        param_init (float):

    """

    def __init__(self, input_dim, n_units, n_layers, bottleneck_dim,
        dropout, param_init=0.1):
        super(SequenceSummaryNetwork, self).__init__()
        self.n_layers = n_layers
        self.ssn = nn.ModuleList()
        self.ssn += [nn.Linear(input_dim, n_units, bias=False)]
        self.ssn += [nn.Dropout(p=dropout)]
        for l in range(1, n_layers - 1):
            self.ssn += [nn.Linear(n_units, bottleneck_dim if l == n_layers -
                2 else n_units, bias=False)]
            self.ssn += [nn.Dropout(p=dropout)]
        self.p = nn.Linear(bottleneck_dim, input_dim, bias=False)
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' %
            self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant',
                    0.0))
            elif p.dim() == 2:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform',
                    param_init))
            else:
                raise ValueError(n)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T', input_dim]`

        """
        bs, time = xs.size()[:2]
        s = xs.clone()
        for l in range(self.n_layers - 1):
            s = torch.tanh(self.ssn[l](s))
        s = self.ssn[self.n_layers - 1](s)
        device_id = torch.cuda.device_of(next(self.parameters())).idx
        mask = make_pad_mask(xlens, device_id).unsqueeze(2)
        s = s.masked_fill_(mask == 0, 0)
        s = s.sum(1) / xlens.float().unsqueeze(1)
        xs = xs + self.p(s).unsqueeze(1)
        return xs


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hirofumi0810_neural_sp(_paritybench_base):
    pass
    def test_000(self):
        self._check(CausalConv1d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(LinearGLUBlock(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MaxpoolSubsampler(*[], **{'factor': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Conv1dSubsampler(*[], **{'factor': 4, 'n_units': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(DropSubsampler(*[], **{'factor': 4}), [torch.rand([4, 4, 4, 4]), [4, 4]], {})

    @_fails_compile()
    def test_005(self):
        self._check(ConcatSubsampler(*[], **{'factor': 4, 'n_units': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(NiN(*[], **{'dim': 4}), [torch.rand([4, 4, 4])], {})

    def test_007(self):
        self._check(SubsampelBlock(*[], **{'in_channel': 4, 'out_channel': 4, 'in_freq': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

