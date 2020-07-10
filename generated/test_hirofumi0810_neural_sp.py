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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import copy


import logging


import numpy as np


import time


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
            normal_dist = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([std]))
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
        logger.info('torch.backends.cudnn.benchmark: %s' % torch.backends.cudnn.benchmark)
        logger.info('torch.backends.cudnn.enabled: %s' % torch.backends.cudnn.enabled)


class CustomDataParallel(DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(CustomDataParallel, self).__init__(module, device_ids, output_device, dim)

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
                observation_mean = {k: (v / n_gpus) for k, v in observation_mean.items()}
            return gather(losses, output_device, dim=self.dim).mean(), observation_mean
        else:
            raise ValueError(n_returns)


def compute_accuracy(logits, ys_ref, pad):
    """Compute teacher-forcing accuracy.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys_ref (LongTensor): `[B, T]`
        pad (int): index for padding
    Returns:
        acc (float): teacher-forcing accuracy

    """
    pad_pred = logits.view(ys_ref.size(0), ys_ref.size(1), logits.size(-1)).argmax(2)
    mask = ys_ref != pad
    numerator = torch.sum(pad_pred.masked_select(mask) == ys_ref.masked_select(mask))
    denominator = torch.sum(mask)
    acc = float(numerator) * 100 / float(denominator)
    return acc


def cross_entropy_lsm(logits, ys, lsm_prob, ignore_index, training, normalize_length=False):
    """Compute cross entropy loss for label smoothing of sequence-to-sequence models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`
        lsm_prob (float): label smoothing probability
        ignore_index (int): index for padding
        normalize_length (bool): normalize XE loss by target sequence length
    Returns:
        loss_mean (FloatTensor): `[1]`
        ppl (float): perplexity

    """
    bs, _, vocab = logits.size()
    ys = ys.view(-1)
    logits = logits.view((-1, logits.size(2)))
    if lsm_prob == 0 or not training:
        loss = F.cross_entropy(logits, ys, ignore_index=ignore_index, reduction='mean')
        ppl = np.exp(loss.item())
        if not normalize_length:
            loss *= (ys != ignore_index).sum() / bs
    else:
        with torch.no_grad():
            target_dist = logits.new_zeros(logits.size())
            target_dist.fill_(lsm_prob / (vocab - 1))
            mask = ys == ignore_index
            ys_masked = ys.masked_fill(mask, 0)
            target_dist.scatter_(1, ys_masked.unsqueeze(1), 1 - lsm_prob)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss_sum = -torch.mul(target_dist, log_probs)
        n_tokens = len(ys) - mask.sum().item()
        denom = n_tokens if normalize_length else bs
        loss = loss_sum.masked_fill(mask.unsqueeze(1), 0).sum() / denom
        ppl = np.exp(loss.item()) if normalize_length else np.exp(loss.item() * bs / n_tokens)
    return loss, ppl


def np2tensor(array, device_id=-1):
    """Convert form np.ndarray to torch.Tensor.

    Args:
        array (np.ndarray): A tensor of any sizes
        device_id (int): ht index of the device
    Returns:
        tensor (FloatTensor/IntTensor/LongTensor):

    """
    tensor = torch.from_numpy(array)
    if device_id >= 0:
        tensor = tensor
    return tensor


def pad_list(xs, pad_value=0.0, pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which concains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, *xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[(b), -xs[b].size(0):] = xs[b]
        else:
            xs_pad[(b), :xs[b].size(0)] = xs[b]
    return xs_pad


class LMBase(ModelBase):
    """Base class for language models."""

    def __init__(self, args):
        super(ModelBase, self).__init__()
        logger.info(self.__class__.__name__)
        logger.info('Overriding LMBase class.')

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def forward(self, ys, state=None, is_eval=False, n_caches=0, ylens=[], predict_last=False):
        """Forward computation.

        Args:
            ys (list): length `B`, each of which contains arrays of size `[L]`
            state (tuple or list):
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int): number of cached states
            ylens (list): not used
            predict_last (bool): used for TransformerLM and GatedConvLM
        Returns:
            loss (FloatTensor): `[1]`
            new_state (tuple or list):
            observation (dict):

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, state, observation = self._forward(ys, state, n_caches, predict_last)
        else:
            self.train()
            loss, state, observation = self._forward(ys, state)
        return loss, state, observation

    def _forward(self, ys, state, n_caches=0, predict_last=False):
        ys = [np2tensor(y, self.device_id) for y in ys]
        ys = pad_list(ys, self.pad)
        ys_in, ys_out = ys[:, :-1], ys[:, 1:]
        logits, out, new_state = self.decode(ys_in, state=state, mems=state)
        if predict_last:
            ys_out = ys_out[:, (-1)].unsqueeze(1)
            logits = logits[:, (-1)].unsqueeze(1)
        if n_caches > 0 and len(self.cache_ids) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            if self.adaptive_softmax is None:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = self.adaptive_softmax.log_prob(logits).exp()
            cache_probs = probs.new_zeros(probs.size())
            self.cache_ids = self.cache_ids[-n_caches:]
            self.cache_keys = self.cache_keys[-n_caches:]
            cache_attn = torch.softmax(self.cache_theta * torch.matmul(torch.cat(self.cache_keys, dim=1), out.transpose(2, 1)).squeeze(2), dim=1)
            if len(self.cache_ids) == n_caches:
                self.cache_attn += [cache_attn.cpu().numpy()]
                self.cache_attn = self.cache_attn[-n_caches:]
            for offset, idx in enumerate(self.cache_ids):
                cache_probs[:, :, (idx)] += cache_attn[:, (offset)]
            probs = (1 - self.cache_lambda) * probs + self.cache_lambda * cache_probs
            loss = -torch.log(probs[:, :, (ys_out[:, (-1)])])
        elif self.adaptive_softmax is None:
            loss, ppl = cross_entropy_lsm(logits, ys_out.contiguous(), self.lsm_prob, self.pad, self.training, normalize_length=True)
        else:
            loss = self.adaptive_softmax(logits.view((-1, logits.size(2))), ys_out.contiguous().view(-1)).loss
            ppl = np.exp(loss.item())
        if n_caches > 0:
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [out]
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(logits.view((-1, logits.size(2)))), ys_out, pad=self.pad)
        observation = {'loss.lm': loss.item(), 'acc.lm': acc, 'ppl.lm': ppl}
        return loss, new_state, observation

    def repackage_state(self, state):
        return state

    def reset_length(self, mem_len):
        self.mem_len = mem_len

    def decode(self, ys, state=None, mems=None, incremental=False):
        raise NotImplementedError

    def predict(self, ys, state=None, mems=None, cache=None):
        """Precict function for ASR.

        Args:
            ys (LongTensor): `[B, L]`
            state:
                - RNNLM: dict
                    hxs (FloatTensor): `[n_layers, B, n_units]`
                    cxs (FloatTensor): `[n_layers, B, n_units]`
                - TransformerLM (LongTensor): `[B, L]`
                - TransformerXL (list): length `n_layers + 1`, each of which contains a tensor`[B, L, d_model]`
            mems (list):
            cache (list):
        Returns:
            lmout (FloatTensor): `[B, L, vocab]`, used for LM integration such as cold fusion
            state:
                - RNNLM: dict
                    hxs (FloatTensor): `[n_layers, B, n_units]`
                    cxs (FloatTensor): `[n_layers, B, n_units]`
                - TransformerLM (LongTensor): `[B, L]`
                - TransformerXL (list): length `n_layers + 1`, each of which contains a tensor`[B, L, d_model]`
            log_probs (FloatTensor): `[B, L, vocab]`

        """
        logits, lmout, new_state = self.decode(ys, state, mems=mems, cache=cache, incremental=True)
        log_probs = torch.log_softmax(logits, dim=-1)
        return lmout, new_state, log_probs

    def plot_attention(self):
        pass


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


def repeat(module, n_layers):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


class RNNLM(LMBase):
    """RNN language model."""

    def __init__(self, args, save_path=None):
        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)
        self.lm_type = args.lm_type
        self.save_path = save_path
        self.emb_dim = args.emb_dim
        self.rnn_type = args.lm_type
        assert args.lm_type in ['lstm', 'gru']
        self.n_units = args.n_units
        self.n_projs = args.n_projs
        self.n_layers = args.n_layers
        self.residual = args.residual
        self.n_units_cv = args.n_units_null_context
        self.lsm_prob = args.lsm_prob
        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        self.cache_theta = 0.2
        self.cache_lambda = 0.2
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []
        self.embed = nn.Embedding(self.vocab, args.emb_dim, padding_idx=self.pad)
        self.dropout_embed = nn.Dropout(p=args.dropout_in)
        rnn = nn.LSTM if args.lm_type == 'lstm' else nn.GRU
        self.rnn = nn.ModuleList()
        self.dropout = nn.Dropout(p=args.dropout_hidden)
        if args.n_projs > 0:
            self.proj = repeat(nn.Linear(args.n_units, args.n_projs), args.n_layers)
        rnn_idim = args.emb_dim + args.n_units_null_context
        for _ in range(args.n_layers):
            self.rnn += [rnn(rnn_idim, args.n_units, 1, batch_first=True)]
            rnn_idim = args.n_units
            if args.n_projs > 0:
                rnn_idim = args.n_projs
        self.glu = None
        if args.use_glu:
            self.glu = LinearGLUBlock(rnn_idim)
        self._odim = rnn_idim
        self.adaptive_softmax = None
        self.output_proj = None
        self.output = None
        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(rnn_idim, self.vocab, cutoffs=[self.vocab // 25, self.vocab // 5], div_value=4.0)
        elif args.tie_embedding:
            if rnn_idim != args.emb_dim:
                self.output_proj = nn.Linear(rnn_idim, args.emb_dim)
                rnn_idim = args.emb_dim
                self._odim = rnn_idim
            self.output = nn.Linear(rnn_idim, self.vocab)
            self.output.weight = self.embed.weight
        else:
            self.output = nn.Linear(rnn_idim, self.vocab)
        self.reset_parameters(args.param_init)

    @property
    def output_dim(self):
        return self._odim

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() == 2:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)

    def decode(self, ys, state, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (FloatTensor): `[B, L]`
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            cache: dummy interfance for TransformerLM/TransformerXL
            incremental: dummy interfance for TransformerLM/TransformerXL
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            ys_emb (FloatTensor): `[B, L, n_units]` (for cache)
            new_state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            new_mems: dummy interfance for TransformerXL

        """
        bs, ymax = ys.size()
        ys_emb = self.dropout_embed(self.embed(ys.long()))
        if state is None:
            state = self.zero_state(bs)
        new_state = {'hxs': None, 'cxs': None}
        if self.n_units_cv > 0:
            cv = ys.new_zeros(bs, ymax, self.n_units_cv).float()
            ys_emb = torch.cat([ys_emb, cv], dim=-1)
        residual = None
        new_hxs, new_cxs = [], []
        for lth in range(self.n_layers):
            self.rnn[lth].flatten_parameters()
            if self.rnn_type == 'lstm':
                ys_emb, (h, c) = self.rnn[lth](ys_emb, hx=(state['hxs'][lth:lth + 1], state['cxs'][lth:lth + 1]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru':
                ys_emb, h = self.rnn[lth](ys_emb, hx=state['hxs'][lth:lth + 1])
            new_hxs.append(h)
            ys_emb = self.dropout(ys_emb)
            if self.n_projs > 0:
                ys_emb = torch.tanh(self.proj[lth](ys_emb))
            if self.residual and lth > 0:
                ys_emb = ys_emb + residual
            residual = ys_emb
        new_state['hxs'] = torch.cat(new_hxs, dim=0)
        if self.rnn_type == 'lstm':
            new_state['cxs'] = torch.cat(new_cxs, dim=0)
        if self.glu is not None:
            if self.residual:
                residual = ys_emb
            ys_emb = self.glu(ys_emb)
            if self.residual:
                ys_emb = ys_emb + residual
        if self.adaptive_softmax is None:
            if self.output_proj is not None:
                ys_emb = self.output_proj(ys_emb)
            logits = self.output(ys_emb)
        else:
            logits = ys_emb
        return logits, ys_emb, new_state

    def zero_state(self, batch_size):
        """Initialize hidden state.

        Args:
            batch_size (int): batch size
        Returns:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        w = next(self.parameters())
        state = {'hxs': None, 'cxs': None}
        state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        if self.rnn_type == 'lstm':
            state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.n_units)
        return state

    def repackage_state(self, state):
        """Wraps hidden states in new Tensors, to detach them from their history.

        Args:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
        Returns:
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`

        """
        state['hxs'] = state['hxs'].detach()
        if self.rnn_type == 'lstm':
            state['cxs'] = state['cxs'].detach()
        return state


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

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, src_tgt_attention=True, memory_transformer=False, mocha_chunk_size=0, mocha_n_heads_mono=1, mocha_n_heads_chunk=1, mocha_init_r=2, mocha_eps=1e-06, mocha_std=1.0, mocha_no_denominator=False, mocha_1dconv=False, dropout_head=0, lm_fusion=False):
        super(TransformerDecoderBlock, self).__init__()
        self.atype = atype
        self.n_heads = n_heads
        self.src_tgt_attention = src_tgt_attention
        self.memory_transformer = memory_transformer
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if 'mocha' in atype:
                self.n_heads = mocha_n_heads_mono
                self.src_attn = MoChA(kdim=d_model, qdim=d_model, adim=d_model, atype='scaled_dot', chunk_size=mocha_chunk_size, n_heads_mono=mocha_n_heads_mono, n_heads_chunk=mocha_n_heads_chunk, init_r=mocha_init_r, eps=mocha_eps, noise_std=mocha_std, no_denominator=mocha_no_denominator, conv1d=mocha_1dconv, dropout=dropout_att, dropout_head=dropout_head, param_init=param_init)
            else:
                self.src_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_layer = dropout_layer
        self.lm_fusion = lm_fusion
        if lm_fusion:
            self.norm_lm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.linear_lm_feat = nn.Linear(d_model, d_model)
            self.linear_lm_gate = nn.Linear(d_model * 2, d_model)
            self.linear_lm_fusion = nn.Linear(d_model * 2, d_model)
            if 'attention' in lm_fusion:
                self.lm_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None, xy_aws_prev=None, mode='hard', lmout=None, pos_embs=None, memory=None, u=None, v=None, eps_wait=-1, boundary_rightmost=None):
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
        if self.dropout_layer > 0 and self.training and random.random() >= self.dropout_layer:
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
            out, yy_aws = self.self_attn(ys, ys_q, memory, pos_embs, yy_mask, u, v)
        else:
            out, yy_aws, _ = self.self_attn(ys, ys, ys_q, mask=yy_mask)
        out = self.dropout(out) + residual
        xy_aws, xy_aws_beta = None, None
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, xy_aws, xy_aws_beta = self.src_attn(xs, xs, out, mask=xy_mask, aw_prev=xy_aws_prev, mode=mode, eps_wait=eps_wait, boundary_rightmost=boundary_rightmost)
            out = self.dropout(out) + residual
        yy_aws_lm = None
        if self.lm_fusion:
            residual = out
            out = self.norm_lm(out)
            lmout = self.linear_lm_feat(lmout)
            if 'attention' in self.lm_fusion:
                out, yy_aws_lm, _ = self.lm_attn(lmout, lmout, out, mask=yy_mask)
            gate = torch.sigmoid(self.linear_lm_gate(torch.cat([out, lmout], dim=-1)))
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
        sinusoid_inp = torch.einsum('i,j->ij', positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = self.dropout(pos_emb)
        return pos_emb.unsqueeze(1)


def init_with_normal_dist(n, p, std):
    if 'norm' in n and 'weight' in n:
        assert p.dim() == 1
        nn.init.normal_(p, 1.0, std)
        logger.info('Initialize %s with %s / (1.0, %.3f)' % (n, 'normal', std))
    elif p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() == 2:
        nn.init.normal_(p, mean=0, std=std)
        logger.info('Initialize %s with %s / (0.0, %.3f)' % (n, 'normal', std))
    else:
        raise ValueError(n)


def mkdir_join(path, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new direcory if the direcory does not exist.
    Args:
        path (str): path to a diretcory
        dir_name (str): a direcory name
    Returns:
        path to the new directory
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in range(len(dir_name)):
        if i < len(dir_name) - 1:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        elif '.' not in dir_name[i]:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        else:
            path = os.path.join(path, dir_name[i])
    return path


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.

    Args:
        x (Tensor):
    Returns:
        np.ndarray

    """
    return x.cpu().numpy()


class TransformerXL(LMBase):
    """TransformerXL language model."""

    def __init__(self, args, save_path=None):
        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)
        self.lm_type = args.lm_type
        self.save_path = save_path
        self.d_model = args.transformer_d_model
        self.n_layers = args.n_layers
        self.n_heads = args.transformer_n_heads
        self.lsm_prob = args.lsm_prob
        if args.mem_len > 0:
            self.mem_len = args.mem_len
        else:
            self.mem_len = args.bptt
        if args.recog_mem_len > 0:
            self.mem_len = args.recog_mem_len
        self.zero_center_offset = args.zero_center_offset
        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        self.cache_theta = 0.2
        self.cache_lambda = 0.2
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []
        self.pos_emb = XLPositionalEmbedding(self.d_model, args.dropout_in)
        self.u = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.scale = math.sqrt(self.d_model)
        self.dropout_emb = nn.Dropout(p=args.dropout_in)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(self.d_model, args.transformer_d_ff, args.transformer_attn_type, self.n_heads, args.dropout_hidden, args.dropout_att, args.dropout_layer, args.transformer_layer_norm_eps, args.transformer_ffn_activation, args.transformer_param_init, src_tgt_attention=False, memory_transformer=True)) for lth in range(self.n_layers)])
        self.norm_out = nn.LayerNorm(self.d_model, eps=args.transformer_layer_norm_eps)
        self.adaptive_softmax = None
        self.output = None
        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(self.d_model, self.vocab, cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)], div_value=4.0)
        else:
            self.output = nn.Linear(self.d_model, self.vocab)
            if args.tie_embedding:
                self.output.weight = self.embed.weight
        self.reset_parameters()

    @property
    def output_dim(self):
        return self.d_model

    def reset_parameters(self):
        """Initialize parameters with normal distribution."""
        logging.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_normal_dist(n, p, std=0.02)

    def init_memory(self):
        """Initialize memory."""
        if self.device_id >= 0:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]
        else:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): length `n_layers`, each of which contains `[B, mlen, d_model]`
            hidden_states (list): length `n_layers`, each of which contains `[B, L, d_model]`
        Returns:
            new_mems (list): length `n_layers`, each of which contains `[B, mlen, d_model]`

        """
        if memory_prev is None:
            memory_prev = self.init_memory()
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)
                new_mems.append(cat[:, start_idx:end_idx].detach())
        return new_mems

    def decode(self, ys, state=None, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (list): dummy interfance for RNNLM
            mems (list): length `n_layers`, each of which contains a FloatTensor `[B, mlen, d_model]`
            cache (list): length `L`, each of which contains a FloatTensor `[B, L-1, d_model]`
            incremental (bool): ASR decoding mode
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]`
            new_cache (list): length `n_layers`, each of which contains a FloatTensor `[B, L, d_model]`

        """
        if cache is None:
            cache = [None] * self.n_layers
        if mems is None:
            mems = self.init_memory()
            mlen = 0
        else:
            mlen = mems[0].size(1)
        bs, ylen = ys.size()[:2]
        if incremental and cache[0] is not None:
            ylen = cache[0].size(1) + 1
        causal_mask = ys.new_ones(ylen, ylen + mlen).byte()
        causal_mask = torch.tril(causal_mask, diagonal=0 + mlen, out=causal_mask).unsqueeze(0)
        causal_mask = causal_mask.repeat([bs, 1, 1])
        out = self.dropout_emb(self.embed(ys.long()) * self.scale)
        if self.zero_center_offset:
            pos_idxs = torch.arange(mlen - 1, -ylen - 1, -1.0, dtype=torch.float)
        else:
            pos_idxs = torch.arange(ylen + mlen - 1, -1, -1.0, dtype=torch.float)
        pos_embs = self.pos_emb(pos_idxs, self.device_id)
        new_mems = [None] * self.n_layers
        new_cache = [None] * self.n_layers
        hidden_states = [out]
        for lth, (mem, layer) in enumerate(zip(mems, self.layers)):
            if incremental and mlen > 0 and mem.size(0) != bs:
                mem = mem.repeat([bs, 1, 1])
            out, yy_aws = layer(out, causal_mask, cache=cache[lth], pos_embs=pos_embs, memory=mem, u=self.u, v=self.v)[:2]
            if incremental:
                new_cache[lth] = out
            elif lth < self.n_layers - 1:
                hidden_states.append(out)
            if not self.training and yy_aws is not None:
                setattr(self, 'yy_aws_layer%d' % lth, tensor2np(yy_aws))
        out = self.norm_out(out)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out
        if incremental:
            return logits, out, new_cache
        else:
            new_mems = self.update_memory(mems, hidden_states)
            return logits, out, new_mems

    def plot_attention(self, n_cols=4):
        """Plot attention for each head in all layers."""
        save_path = mkdir_join(self.save_path, 'att_weights')
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        for lth in range(self.n_layers):
            if not hasattr(self, 'yy_aws_layer%d' % lth):
                continue
            yy_aws = getattr(self, 'yy_aws_layer%d' % lth)
            plt.clf()
            fig, axes = plt.subplots(self.n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.n_heads):
                if self.n_heads > n_cols:
                    ax = axes[h // n_cols, h % n_cols]
                else:
                    ax = axes[h]
                ax.imshow(yy_aws[(-1), (h), :, :], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % lth), dvi=500)
            plt.close()


class CausalConv1d(nn.Module):
    """1D dilated causal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

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


def init_with_xavier_dist(n, p):
    if p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() in [2, 3]:
        nn.init.xavier_uniform_(p)
        logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
    else:
        raise ValueError(n)


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

    def __init__(self, d_model, dropout, pe_type, param_init, max_len=5000, conv_kernel_size=3, layer_norm_eps=1e-12):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe_type = pe_type
        self.scale = math.sqrt(self.d_model)
        if '1dconv' in pe_type:
            causal_conv1d = CausalConv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel_size)
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
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.dropout = nn.Dropout(p=dropout)
        logger.info('Positional encoding: %s' % pe_type)

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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


class TransformerLM(LMBase):
    """Transformer language model."""

    def __init__(self, args, save_path=None):
        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)
        self.lm_type = args.lm_type
        self.save_path = save_path
        self.d_model = args.transformer_d_model
        self.n_layers = args.n_layers
        self.n_heads = args.transformer_n_heads
        self.lsm_prob = args.lsm_prob
        self.mem_len = args.mem_len
        if args.recog_mem_len > 0:
            self.mem_len = args.recog_mem_len
        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        self.cache_theta = 0.2
        self.cache_lambda = 0.2
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []
        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.pos_enc = PositionalEncoding(self.d_model, args.dropout_in, args.transformer_pe_type, args.transformer_param_init)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(self.d_model, args.transformer_d_ff, args.transformer_attn_type, self.n_heads, args.dropout_hidden, args.dropout_att, args.dropout_layer, args.transformer_layer_norm_eps, args.transformer_ffn_activation, args.transformer_param_init, src_tgt_attention=False)) for lth in range(self.n_layers)])
        self.norm_out = nn.LayerNorm(self.d_model, eps=args.transformer_layer_norm_eps)
        self.adaptive_softmax = None
        self.output = None
        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(self.d_model, self.vocab, cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)], div_value=4.0)
        else:
            self.output = nn.Linear(self.d_model, self.vocab)
            if args.tie_embedding:
                self.output.weight = self.embed.weight
        self.reset_parameters()

    @property
    def output_dim(self):
        return self.d_model

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logging.info('===== Initialize %s =====' % self.__class__.__name__)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.constant_(self.embed.weight[self.pad], 0)
        if self.output is not None:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.constant_(self.output.bias, 0.0)

    def init_memory(self):
        """Initialize memory."""
        if self.device_id >= 0:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]
        else:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): length `n_layers`, each of which contains `[B, mlen, d_model]`
            hidden_states (list): length `n_layers`, each of which contains `[B, L, d_model]`
        Returns:
            new_mems (list): length `n_layers`, each of which contains `[B, mlen, d_model]`

        """
        if memory_prev is None:
            memory_prev = self.init_memory()
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)
                new_mems.append(cat[:, start_idx:end_idx].detach())
        return new_mems

    def decode(self, ys, state=None, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (list): dummy interfance for RNNLM
            mems (list): length `n_layers`, each of which contains a FloatTensor `[B, mlen, d_model]`
            cache (list): length `L`, each of which contains a FloatTensor `[B, L-1, d_model]`
            incremental (bool): ASR decoding mode
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]`
            new_cache (list): length `n_layers`, each of which contains a FloatTensor `[B, L, d_model]`

        """
        if cache is None:
            cache = [None] * self.n_layers
        if mems is None:
            mems = self.init_memory()
        bs, ylen = ys.size()[:2]
        if incremental and cache[0] is not None:
            ylen = cache[0].size(1) + 1
        causal_mask = ys.new_ones(ylen, ylen).byte()
        causal_mask = torch.tril(causal_mask, diagonal=0, out=causal_mask).unsqueeze(0)
        causal_mask = causal_mask.repeat([bs, 1, 1])
        out = self.pos_enc(self.embed(ys.long()))
        new_mems = [None] * self.n_layers
        new_cache = [None] * self.n_layers
        hidden_states = [out]
        for lth, (mem, layer) in enumerate(zip(mems, self.layers)):
            out, yy_aws = layer(out, causal_mask, cache=cache[lth], memory=mem)[:2]
            if incremental:
                new_cache[lth] = out
            elif lth < self.n_layers - 1:
                hidden_states.append(out)
            if not self.training and yy_aws is not None:
                setattr(self, 'yy_aws_layer%d' % lth, tensor2np(yy_aws))
        out = self.norm_out(out)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out
        if incremental:
            return logits, out, new_cache
        elif self.mem_len > 0:
            new_mems = self.update_memory(mems, hidden_states)
            return logits, out, new_mems
        else:
            return logits, out, mems

    def plot_attention(self, n_cols=4):
        """Plot attention for each head in all layers."""
        save_path = mkdir_join(self.save_path, 'att_weights')
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        for lth in range(self.n_layers):
            if not hasattr(self, 'yy_aws_layer%d' % lth):
                continue
            yy_aws = getattr(self, 'yy_aws_layer%d' % lth)
            plt.clf()
            fig, axes = plt.subplots(self.n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.n_heads):
                if self.n_heads > n_cols:
                    ax = axes[h // n_cols, h % n_cols]
                else:
                    ax = axes[h]
                ax.imshow(yy_aws[(-1), (h), :, :], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % lth), dvi=500)
            plt.close()


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

    def __init__(self, kdim, qdim, adim, atype, sharpening_factor=1, sigmoid_smoothing=False, conv_out_channels=10, conv_kernel_size=201, dropout=0.0, lookahead=2):
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
            self.conv = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(1, conv_kernel_size), stride=1, padding=(0, (conv_kernel_size - 1) // 2), bias=False)
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

    def forward(self, key, value, query, mask=None, aw_prev=None, cache=False, mode='', trigger_point=None):
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
            if self.atype in ['add', 'trigerred_attention', 'location', 'dot', 'luong_general']:
                self.key = self.w_key(key)
            else:
                self.key = key
            self.mask = mask
            if mask is not None:
                assert self.mask.size() == (bs, 1, klen), (self.mask.size(), (bs, 1, klen))
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
            e = self.v(torch.tanh(self.w(torch.cat([self.key, query], dim=-1)))).transpose(2, 1)
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


class CIF(nn.Module):
    """docstring for CIF."""

    def __init__(self, enc_dim, conv_out_channels, conv_kernel_size, threshold=0.9):
        super(CIF, self).__init__()
        self.threshold = threshold
        self.channel = conv_out_channels
        self.n_heads = 1
        self.conv = nn.Conv1d(in_channels=enc_dim, out_channels=conv_out_channels, kernel_size=conv_kernel_size * 2 + 1, stride=1, padding=conv_kernel_size)
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


class ConvGLUBlock(nn.Module):
    """A convolutional GLU block.

    Args:
        kernel_size (int): kernel size
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        bottlececk_dim (int): dimension of the bottleneck layers for computational efficiency
        dropout (float): dropout probability

    """

    def __init__(self, kernel_size, in_ch, out_ch, bottlececk_dim=0, dropout=0.0):
        super().__init__()
        self.conv_residual = None
        if in_ch != out_ch:
            self.conv_residual = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1)), name='weight', dim=0)
            self.dropout_residual = nn.Dropout(p=dropout)
        self.pad_left = nn.ConstantPad2d((0, 0, kernel_size - 1, 0), 0)
        layers = OrderedDict()
        if bottlececk_dim == 0:
            layers['conv'] = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=(kernel_size, 1)), name='weight', dim=0)
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = nn.GLU()
        elif bottlececk_dim > 0:
            layers['conv_in'] = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch, out_channels=bottlececk_dim, kernel_size=(1, 1)), name='weight', dim=0)
            layers['dropout_in'] = nn.Dropout(p=dropout)
            layers['conv_bottleneck'] = nn.utils.weight_norm(nn.Conv2d(in_channels=bottlececk_dim, out_channels=bottlececk_dim, kernel_size=(kernel_size, 1)), name='weight', dim=0)
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = nn.GLU()
            layers['conv_out'] = nn.utils.weight_norm(nn.Conv2d(in_channels=bottlececk_dim, out_channels=out_ch * 2, kernel_size=(1, 1)), name='weight', dim=0)
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

    def forward(self, key, value, query, mask=None, aw_prev=None, cache=False, mode='', trigger_point=None):
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
            assert self.mask.size() == (bs, 1, klen), (self.mask.size(), (bs, 1, klen))
        w = torch.softmax(self.ffn_gamma(query), dim=-1)
        v = torch.exp(self.ffn_beta(query))
        myu = torch.exp(self.ffn_kappa(query)) + myu_prev
        self.myu = myu
        js = torch.arange(klen).unsqueeze(0).unsqueeze(2).repeat([bs, 1, self.n_mix]).float()
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


class MonotonicEnergy(nn.Module):

    def __init__(self, kdim, qdim, adim, atype, n_heads, init_r, conv1d=False, conv_kernel_size=5, bias=True, param_init=''):
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
            self.conv1d = CausalConv1d(in_channels=kdim, out_channels=kdim, kernel_size=conv_kernel_size)
        if atype == 'add':
            self.v = nn.utils.weight_norm(self.v, name='weight', dim=0)
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == 'scaled_dot':
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)
        if self.conv1d is not None:
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.conv1d.__class__.__name__)
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
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), (self.mask.size(), (bs, self.n_heads, qlen, klen))
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
        assert e.size() == (bs, self.n_heads, qlen, klen), (e.size(), (bs, self.n_heads, qlen, klen))
        return e


class ChunkEnergy(nn.Module):

    def __init__(self, kdim, qdim, adim, atype, n_heads=1, bias=True, param_init=''):
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
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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
                self.mask = self.mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                assert self.mask.size() == (bs, self.n_heads, qlen, klen), (self.mask.size(), (bs, self.n_heads, qlen, klen))
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
        assert energy.size() == (bs, self.n_heads, qlen, klen), (energy.size(), (bs, self.n_heads, qlen, klen))
        return energy


def add_gaussian_noise(xs):
    noise = torch.normal(torch.zeros(xs.shape[-1]), 0.075)
    if xs.is_cuda:
        noise = noise
    xs.data += noise
    return xs


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


def efficient_chunkwise_attention(alpha, e, mask, chunk_size, n_heads, sharpening_factor, chunk_len_dist=None):
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
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators, back=0, forward=klen - 1)
    else:
        softmax_denominators = moving_sum(softmax_exp, back=chunk_size - 1, forward=0)
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators, back=0, forward=chunk_size - 1)
    return beta.view(bs, -1, qlen, klen)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), x.size(1), x.size(2), 1), x[:, :, :, :-1]], dim=-1), dim=-1)


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumsum(torch.cat([x.new_zeros(x.size(0), x.size(1), x.size(2), 1), x[:, :, :, :-1]], dim=-1), dim=-1)


def safe_cumprod(x, eps):
    """Numerically stable cumulative product by cumulative sum in log-space.
        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max=1.0))))


class MoChA(nn.Module):

    def __init__(self, kdim, qdim, adim, atype, chunk_size, n_heads_mono=1, n_heads_chunk=1, conv1d=False, init_r=-4, eps=1e-06, noise_std=1.0, no_denominator=False, sharpening_factor=1.0, dropout=0.0, dropout_head=0.0, bias=True, param_init='', decot=False, lookahead=2):
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
        self.monotonic_energy = MonotonicEnergy(kdim, qdim, adim, atype, n_heads_mono, init_r, conv1d, bias=bias, param_init=param_init)
        self.chunk_energy = ChunkEnergy(kdim, qdim, adim, atype, n_heads_chunk, bias, param_init) if chunk_size > 1 or self.milk else None
        if n_heads_mono * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_out = nn.Linear(adim, kdim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters(bias)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head

    def reset_parameters(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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

    def forward(self, key, value, query, mask=None, aw_prev=None, mode='hard', cache=False, trigger_point=None, eps_wait=-1, boundary_rightmost=None):
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
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))
            alpha = []
            for i in range(qlen):
                shifted_1mp_choose = torch.cat([key.new_ones(bs, self.n_heads_mono, 1, 1), 1 - p_choose[:, :, i:i + 1, :-1]], dim=-1)
                q = key.new_zeros(bs, self.n_heads_mono, 1, klen + 1)
                for j in range(klen):
                    q[:, :, i:i + 1, (j + 1)] = shifted_1mp_choose[:, :, i:i + 1, (j)].clone() * q[:, :, i:i + 1, (j)].clone() + aw_prev[:, :, :, (j)].clone()
                aw_prev = p_choose[:, :, i:i + 1] * q[:, :, i:i + 1, 1:]
                alpha.append(aw_prev)
            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]
            alpha_masked = alpha.clone()
        elif mode == 'parallel':
            p_choose = torch.sigmoid(add_gaussian_noise(e_mono, self.noise_std))
            cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=self.eps)
            alpha = []
            for i in range(qlen):
                denom = 1 if self.no_denom else torch.clamp(cumprod_1mp_choose[:, :, i:i + 1], min=self.eps, max=1.0)
                aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :, i:i + 1] * torch.cumsum(aw_prev / denom, dim=-1)
                if self.decot and trigger_point is not None:
                    for b in range(bs):
                        aw_prev[(b), :, :, trigger_point[b] + self.lookahead + 1:] = 0
                alpha.append(aw_prev)
            alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]
            alpha_masked = alpha.clone()
            if self.n_heads_mono > 1 and self.dropout_head > 0 and self.training:
                n_effective_heads = self.n_heads_mono
                head_mask = alpha.new_ones(alpha.size()).byte()
                for h in range(self.n_heads_mono):
                    if random.random() < self.dropout_head:
                        head_mask[:, (h)] = 0
                        n_effective_heads -= 1
                alpha_masked = alpha_masked.masked_fill_(head_mask == 0, 0)
                if n_effective_heads > 0:
                    alpha_masked = alpha_masked * (self.n_heads_mono / n_effective_heads)
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
                        boundary_threshold = min(alpha.size(-1) - 1, boundary_rightmost + eps_wait)
                    if alpha[b].sum() == 0:
                        if vertical_latency and boundary_threshold < alpha.size(-1) - 1:
                            alpha[(b), :, (0), (boundary_threshold)] = 1
                        continue
                    leftmost = alpha[(b), :, (0)].nonzero()[:, (-1)].min().item()
                    rightmost = alpha[(b), :, (0)].nonzero()[:, (-1)].max().item()
                    for h in range(self.n_heads_mono):
                        if alpha[b, h, 0].sum().item() == 0:
                            if first_mma_layer or not vertical_latency:
                                alpha[b, h, 0, min(rightmost, leftmost + eps_wait)] = 1
                            elif boundary_threshold < alpha.size(-1) - 1:
                                alpha[b, h, 0, boundary_threshold] = 1
                            continue
                        if first_mma_layer or not vertical_latency:
                            if alpha[b, h, 0].nonzero()[:, (-1)].min().item() >= leftmost + eps_wait:
                                alpha[(b), (h), (0), :] = 0
                                alpha[b, h, 0, leftmost + eps_wait] = 1
                        elif alpha[b, h, 0].nonzero()[:, (-1)].min().item() > boundary_threshold:
                            alpha[(b), (h), (0), :] = 0
                            alpha[b, h, 0, boundary_threshold] = 1
            alpha_masked = alpha.clone()
        else:
            raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
        beta = None
        if self.chunk_size > 1 or self.milk:
            e_chunk = self.chunk_energy(key, query, mask, cache=cache)
            beta = efficient_chunkwise_attention(alpha_masked, e_chunk, mask, self.chunk_size, self.n_heads_chunk, self.sharpening_factor)
            beta = self.dropout(beta)
        if self.n_heads_mono * self.n_heads_chunk > 1:
            value = self.w_value(value).view(bs, -1, self.n_heads_mono * self.n_heads_chunk, self.d_k)
            value = value.transpose(2, 1).contiguous()
            if self.chunk_size == 1:
                cv = torch.matmul(alpha, value)
            else:
                cv = torch.matmul(beta, value)
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.n_heads_mono * self.n_heads_chunk * self.d_k)
            cv = self.w_out(cv)
        elif self.chunk_size == 1:
            cv = torch.bmm(alpha.squeeze(1), value)
        else:
            cv = torch.bmm(beta.squeeze(1), value)
        assert alpha.size() == (bs, self.n_heads_mono, qlen, klen), (alpha.size(), (bs, self.n_heads_mono, qlen, klen))
        if self.chunk_size > 1 or self.milk:
            assert beta.size() == (bs, self.n_heads_mono * self.n_heads_chunk, qlen, klen), (beta.size(), (bs, self.n_heads_mono * self.n_heads_chunk, qlen, klen))
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

    def __init__(self, kdim, qdim, adim, n_heads, dropout, atype='scaled_dot', bias=True, param_init=''):
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
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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

    def forward(self, key, value, query, mask, aw_prev=None, cache=False, mode='', trigger_point=None, eps_wait=-1, boundary_rightmost=None):
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
            self.value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)
            self.mask = mask
            if self.mask is not None:
                self.mask = self.mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                assert self.mask.size() == (bs, qlen, klen, self.n_heads), (self.mask.size(), (bs, qlen, klen, self.n_heads))
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


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_accurate(x):
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


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
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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

    def __init__(self, kdim, qdim, adim, n_heads, dropout, bias=True, param_init=''):
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
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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
        xs_shifted = torch.cat([zero_pad, xs], dim=1).view(klen + 1, qlen, bs * n_heads)[1:].view_as(xs)
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
            assert mask.size() == (bs, qlen, mlen + klen, self.n_heads), (mask.size(), (bs, qlen, klen + mlen, self.n_heads))
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        pos_embs = self.w_position(pos_embs)
        pos_embs = pos_embs.view(-1, self.n_heads, self.d_k)
        if u is not None:
            AC = torch.einsum('bihd,bjhd->bijh', (query + u[None, None], key))
        else:
            AC = torch.einsum('bihd,bjhd->bijh', (query, key))
        if v is not None:
            BD = torch.einsum('bihd,jhd->bijh', (query + v[None, None], pos_embs))
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

    def __init__(self, kdim, qdim, adim, n_heads, dropout, atype='scaled_dot', bias=True, param_init='', future_weight=0.1):
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
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
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

    def forward(self, key_fwd, value_fwd, query_fwd, key_bwd, value_bwd, query_bwd, tgt_mask, identity_mask, mode='', cache=True, trigger_point=None):
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
            value_fwd = self.w_value(value_fwd).view(bs, -1, self.n_heads, self.d_k)
            self.value_fwd = value_fwd.transpose(2, 1).contiguous()
            self.tgt_mask = tgt_mask
            self.identity_mask = identity_mask
            if tgt_mask is not None:
                self.tgt_mask = tgt_mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                assert self.tgt_mask.size() == (bs, self.n_heads, qlen, klen)
            if identity_mask is not None:
                self.identity_mask = identity_mask.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                assert self.identity_mask.size() == (bs, self.n_heads, qlen, klen)
        if self.key_bwd is None or not cache:
            key_bwd = self.w_key(key_bwd).view(bs, -1, self.n_heads, self.d_k)
            self.key_bwd = key_bwd.transpose(2, 1).contiguous()
            value_bwd = self.w_value(value_bwd).view(bs, -1, self.n_heads, self.d_k)
            self.value_bwd = value_bwd.transpose(2, 1).contiguous()
        query_fwd = self.w_query(query_fwd).view(bs, -1, self.n_heads, self.d_k)
        query_fwd = query_fwd.transpose(2, 1).contiguous()
        query_bwd = self.w_query(query_bwd).view(bs, -1, self.n_heads, self.d_k)
        query_bwd = query_bwd.transpose(2, 1).contiguous()
        if self.atype == 'scaled_dot':
            e_fwd_h = torch.matmul(query_fwd, self.key_fwd.transpose(3, 2)) / self.scale
            e_fwd_f = torch.matmul(query_fwd, self.key_bwd.transpose(3, 2)) / self.scale
            e_bwd_h = torch.matmul(query_bwd, self.key_bwd.transpose(3, 2)) / self.scale
            e_bwd_f = torch.matmul(query_bwd, self.key_fwd.transpose(3, 2)) / self.scale
        elif self.atype == 'add':
            e_fwd_h = torch.tanh(self.key_fwd.unsqueeze(2) + query_fwd.unsqueeze(3))
            e_fwd_h = e_fwd_h.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_fwd_h = self.v(e_fwd_h).permute(0, 3, 1, 2)
            e_fwd_f = torch.tanh(self.key_bwd.unsqueeze(2) + query_fwd.unsqueeze(3))
            e_fwd_f = e_fwd_f.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_fwd_f = self.v(e_fwd_f).permute(0, 3, 1, 2)
            e_bwd_h = torch.tanh(self.key_bwd.unsqueeze(2) + query_bwd.unsqueeze(3))
            e_bwd_h = e_bwd_h.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
            e_bwd_h = self.v(e_bwd_h).permute(0, 3, 1, 2)
            e_bwd_f = torch.tanh(self.key_fwd.unsqueeze(2) + query_bwd.unsqueeze(3))
            e_bwd_f = e_bwd_f.permute(0, 2, 3, 1, 4).contiguous().view(bs, qlen, klen, -1)
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
        cv_fwd_h = cv_fwd_h.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_fwd_h = self.w_out(cv_fwd_h)
        cv_fwd_f = cv_fwd_f.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_fwd_f = self.w_out(cv_fwd_f)
        cv_bwd_h = cv_bwd_h.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv_bwd_h = self.w_out(cv_bwd_h)
        cv_bwd_f = cv_bwd_f.transpose(2, 1).contiguous().view(bs, -1, self.n_heads * self.d_k)
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

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, memory_transformer=False):
        super(TransformerEncoderBlock, self).__init__()
        self.n_heads = n_heads
        self.memory_transformer = memory_transformer
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)
        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer

    def forward(self, xs, xx_mask=None, pos_embs=None, memory=None, u=None, v=None):
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
        if self.dropout_layer > 0 and self.training and random.random() >= self.dropout_layer:
            return xs, None
        residual = xs
        xs = self.norm1(xs)
        if self.memory_transformer:
            xs, xx_aws = self.self_attn(xs, xs, memory, pos_embs, xx_mask, u, v)
        elif memory is not None:
            xs_memory = torch.cat([memory, xs], dim=1)
            xs, xx_aws, _ = self.self_attn(xs_memory, xs_memory, xs, mask=xx_mask)
        else:
            xs, xx_aws, _ = self.self_attn(xs, xs, xs, mask=xx_mask)
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm2(xs)
        xs = self.feed_forward(xs)
        xs = self.dropout(xs) + residual
        return xs, xx_aws


class SyncBidirTransformerDecoderBlock(nn.Module):
    """A single layer of the synchronous bidirectional Transformer decoder.

        Args:
            d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
            d_ff (int): hidden dimension of PositionwiseFeedForward
            n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            dropout_layer (float): LayerDrop probabilities for layers
            layer_norm_eps (float): epsilon parameter for layer normalization
            ffn_activation (str): nonolinear function for PositionwiseFeedForward
            param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init):
        super(SyncBidirTransformerDecoderBlock, self).__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = SyncBidirMHA(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.src_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ys, ys_bwd, yy_mask, identity_mask, xs, xy_mask, cache=None, cache_bwd=None):
        """Synchronous bidirectional Transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            ys_bwd (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L, L]`
            identity_mask (ByteTensor): `[B, L, L]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            cache_bwd (FloatTensor): `[B, L-1, d_model]`
        Returns:
            out (FloatTensor): `[B, L, d_model]`
            yy_aws_h (FloatTensor)`[B, L, L]`
            yy_aws_f (FloatTensor)`[B, L, L]`
            yy_aws_bwd_h (FloatTensor)`[B, L, L]`
            yy_aws_bwd_f (FloatTensor)`[B, L, L]`
            xy_aws (FloatTensor): `[B, L, T]`
            xy_aws_bwd (FloatTensor): `[B, L, T]`

        """
        residual = ys
        residual_bwd = ys_bwd
        ys = self.norm1(ys)
        ys_bwd = self.norm1(ys_bwd)
        if cache is not None:
            assert cache_bwd is not None
            ys_q = ys[:, -1:]
            ys_bwd_q = ys_bwd[:, -1:]
            residual = residual[:, -1:]
            residual_bwd = residual_bwd[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys
            ys_bwd_q = ys_bwd
        out, out_bwd, yy_aws_h, yy_aws_f, yy_aws_bwd_h, yy_aws_bwd_f = self.self_attn(ys, ys, ys_q, ys_bwd, ys_bwd, ys_bwd_q, tgt_mask=yy_mask, identity_mask=identity_mask)
        out = self.dropout(out) + residual
        out_bwd = self.dropout(out_bwd) + residual_bwd
        residual = out
        out = self.norm2(out)
        out, xy_aws, _ = self.src_attn(xs, xs, out, mask=xy_mask)
        out = self.dropout(out) + residual
        residual_bwd = out_bwd
        out_bwd = self.norm2(out_bwd)
        out_bwd, xy_aws_bwd, _ = self.src_attn(xs, xs, out_bwd, mask=xy_mask)
        out_bwd = self.dropout(out_bwd) + residual_bwd
        residual = out
        out = self.norm3(out)
        out = self.feed_forward(out)
        out = self.dropout(out) + residual
        residual_bwd = out_bwd
        out_bwd = self.norm3(out_bwd)
        out_bwd = self.feed_forward(out_bwd)
        out_bwd = self.dropout(out_bwd) + residual_bwd
        if cache is not None:
            out = torch.cat([cache, out], dim=1)
            out_bwd = torch.cat([cache_bwd, out_bwd], dim=1)
        return out, out_bwd, yy_aws_h, yy_aws_f, yy_aws_bwd_h, yy_aws_bwd_f, xy_aws, xy_aws_bwd


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
            return self.zoneout(state[0], next_state[0], prob[0]), self.zoneout(state[1], next_state[1], prob[1])
        mask = state.new(state.size()).bernoulli_(prob)
        if self.training:
            return mask * next_state + (1 - mask) * state
        else:
            return prob * next_state + (1 - prob) * state


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):
        super(ModelBase, self).__init__()
        logger.info('Overriding DecoderBase class.')

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def reset_session(self):
        self.new_session = True

    def greedy(self, eouts, elens, max_len_ratio):
        raise NotImplementedError

    def beam_search(self, eouts, elens, params, idx2token):
        raise NotImplementedError

    def _plot_attention(self):
        raise NotImplementedError

    def decode_ctc(self, eouts, elens, params, idx2token, lm=None, lm_2nd=None, lm_2nd_rev=None, nbest=1, refs_id=None, utt_ids=None, speakers=None):
        """Decoding with CTC scores in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            params (dict):
                recog_beam_width (int): size of beam
                recog_length_penalty (float): length penalty
                recog_lm_weight (float): weight of first path LM score
                recog_lm_second_weight (float): weight of second path LM score
                recog_lm_rev_weight (float): weight of second path backward LM score
            lm: firsh path LM
            lm_2nd: second path LM
            lm_2nd_rev: secoding path backward LM
        Returns:
            probs (FloatTensor): `[B, T, vocab]`
            topk_ids (LongTensor): `[B, T, topk]`
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`

        """
        if params['recog_beam_width'] == 1:
            best_hyps = self.ctc.greedy(eouts, elens)
        else:
            best_hyps = self.ctc.beam_search(eouts, elens, params, idx2token, lm, lm_2nd, lm_2nd_rev, nbest, refs_id, utt_ids, speakers)
        return best_hyps

    def ctc_probs(self, eouts, temperature=1.0):
        """Return CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.softmax(self.ctc.output(eouts) / temperature, dim=-1)

    def ctc_log_probs(self, eouts, temperature=1.0):
        """Return log-scale CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            log_probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.log_softmax(self.ctc.output(eouts) / temperature, dim=-1)

    def ctc_probs_topk(self, eouts, temperature=1.0, topk=None):
        """Get CTC top-K probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            temperature (float): softmax temperature
            topk (int):
        Returns:
            probs (FloatTensor): `[B, T, vocab]`
            topk_ids (LongTensor): `[B, T, topk]`

        """
        probs = torch.softmax(self.ctc.output(eouts) / temperature, dim=-1)
        if topk is None:
            topk = probs.size(-1)
        _, topk_ids = torch.topk(probs, k=topk, dim=-1, largest=True, sorted=True)
        return probs, topk_ids

    def lm_rescoring(self, hyps, lm, lm_weight, reverse=False, tag=''):
        for i in range(len(hyps)):
            ys = hyps[i]['hyp']
            if reverse:
                ys = ys[::-1]
            ys = [np2tensor(np.fromiter(ys, dtype=np.int64), self.device_id)]
            ys_in = pad_list([y[:-1] for y in ys], -1)
            ys_out = pad_list([y[1:] for y in ys], -1)
            lmout, lmstate, scores_lm = lm.predict(ys_in, None)
            score_lm = sum([scores_lm[0, t, ys_out[0, t]] for t in range(ys_out.size(1))])
            score_lm /= ys_out.size(1)
            hyps[i]['score'] += score_lm * lm_weight
            hyps[i]['score_lm_' + tag] = score_lm


class BeamSearch(object):

    def __init__(self, beam_width, eos, ctc_weight, device_id, beam_width_bwd=0):
        super(BeamSearch, self).__init__()
        self.beam_width = beam_width
        self.beam_width_bwd = beam_width_bwd
        self.eos = eos
        self.device_id = device_id
        self.ctc_weight = ctc_weight

    def remove_complete_hyp(self, hyps_sorted, end_hyps, prune=True, backward=False):
        new_hyps = []
        is_finish = False
        for hyp in hyps_sorted:
            if not backward and len(hyp['hyp']) > 1 and hyp['hyp'][-1] == self.eos:
                end_hyps += [hyp]
            elif backward and len(hyp['hyp_bwd']) > 1 and hyp['hyp_bwd'][-1] == self.eos:
                end_hyps += [hyp]
            else:
                new_hyps += [hyp]
        if len(end_hyps) >= self.beam_width + self.beam_width_bwd:
            if prune:
                end_hyps = end_hyps[:self.beam_width + self.beam_width_bwd]
            is_finish = True
        return new_hyps, end_hyps, is_finish

    def add_ctc_score(self, hyp, topk_ids, ctc_state, total_scores_topk, ctc_prefix_scorer, new_chunk=False, backward=False):
        beam_width = self.beam_width_bwd if backward else self.beam_width
        if ctc_prefix_scorer is None:
            return None, topk_ids.new_zeros(beam_width), total_scores_topk
        ctc_scores, new_ctc_states = ctc_prefix_scorer(hyp, tensor2np(topk_ids[0]), ctc_state, new_chunk=new_chunk)
        total_scores_ctc = torch.from_numpy(ctc_scores)
        if self.device_id >= 0:
            total_scores_ctc = total_scores_ctc
        total_scores_topk += total_scores_ctc * self.ctc_weight
        total_scores_topk, joint_ids_topk = torch.topk(total_scores_topk, k=beam_width, dim=1, largest=True, sorted=True)
        topk_ids = topk_ids[:, (joint_ids_topk[0])]
        new_ctc_states = new_ctc_states[joint_ids_topk[0].cpu().numpy()]
        return new_ctc_states, total_scores_ctc, total_scores_topk

    def add_lm_score(self):
        raise NotImplementedError


LOG_0 = float(np.finfo(np.float32).min)


LOG_1 = 0


def _flip_label_probability(log_probs, xlens):
    """Flips a label probability matrix.
    This function rotates a label probability matrix and flips it.
    ``log_probs[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = log_probs[i + xlens[b], b, l]``

    Args:
        cum_log_prob (FloatTensor): `[T, B, vocab]`
        xlens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, vocab]`

    """
    xmax, bs, vocab = log_probs.size()
    rotate = (torch.arange(xmax, dtype=torch.int64)[:, (None)] + xlens) % xmax
    return torch.flip(log_probs[rotate[:, :, (None)], torch.arange(bs, dtype=torch.int64)[(None), :, (None)], torch.arange(vocab, dtype=torch.int64)[(None), (None), :]], dims=[0])


def _flip_path(path, path_lens):
    """Flips label sequence.
    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_lens[b]]``
    .. ::
       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g

    Args:
        path (FloatTensor): `[B, 2*L+1]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[B, 2*L+1]`

    """
    bs = path.size(0)
    max_path_len = path.size(1)
    rotate = (torch.arange(max_path_len) + path_lens[:, (None)]) % max_path_len
    return torch.flip(path[torch.arange(bs, dtype=torch.int64)[:, (None)], rotate], dims=[1])


def _flip_path_probability(cum_log_prob, xlens, path_lens):
    """Flips a path probability matrix.
    This function returns a path probability matrix and flips it.
    ``cum_log_prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = cum_log_prob[i + xlens[j], j, k + path_lens[j]]``

    Args:
        cum_log_prob (FloatTensor): `[T, B, 2*L+1]`
        xlens (LongTensor): `[B]`
        path_lens (LongTensor): `[B]`
    Returns:
        FloatTensor: `[T, B, 2*L+1]`

    """
    xmax, bs, max_path_len = cum_log_prob.size()
    rotate_input = (torch.arange(xmax, dtype=torch.int64)[:, (None)] + xlens) % xmax
    rotate_label = (torch.arange(max_path_len, dtype=torch.int64) + path_lens[:, (None)]) % max_path_len
    return torch.flip(cum_log_prob[rotate_input[:, :, (None)], torch.arange(bs, dtype=torch.int64)[(None), :, (None)], rotate_label], dims=[0, 2])


def _label_to_path(labels, blank):
    path = labels.new_zeros(labels.size(0), labels.size(1) * 2 + 1).fill_(blank).long()
    path[:, 1::2] = labels
    return path


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
        mask = mask
    return mask


class CTCForcedAligner(object):

    def __init__(self, blank=0):
        self.blank = blank
        self.log0 = LOG_0

    def _computes_transition(self, prev_log_prob, path, path_lens, cum_log_prob, y, skip_accum=False):
        bs, max_path_len = path.size()
        mat = prev_log_prob.new_zeros(3, bs, max_path_len).fill_(self.log0)
        mat[(0), :, :] = prev_log_prob
        mat[(1), :, 1:] = prev_log_prob[:, :-1]
        mat[(2), :, 2:] = prev_log_prob[:, :-2]
        same_transition = path[:, :-2] == path[:, 2:]
        mat[(2), :, 2:][same_transition] = self.log0
        log_prob = torch.logsumexp(mat, dim=0)
        outside = torch.arange(max_path_len, dtype=torch.int64) >= path_lens.unsqueeze(1)
        log_prob[outside] = self.log0
        if not skip_accum:
            cum_log_prob += log_prob
        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        log_prob += y[batch_index, path]
        return log_prob

    def align(self, logits, elens, ys, ylens):
        bs, xmax, vocab = logits.size()
        device_id = torch.cuda.device_of(logits).idx
        mask = make_pad_mask(elens, device_id)
        mask = mask.unsqueeze(2).repeat([1, 1, vocab])
        logits = logits.masked_fill_(mask == 0, self.log0)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        path = _label_to_path(ys, self.blank)
        path_lens = 2 * ylens.long() + 1
        ymax = ys.size(1)
        max_path_len = path.size(1)
        assert ys.size() == (bs, ymax), ys.size()
        assert path.size() == (bs, ymax * 2 + 1)
        alpha = log_probs.new_zeros(bs, max_path_len).fill_(self.log0)
        alpha[:, (0)] = LOG_1
        beta = alpha.clone()
        gamma = alpha.clone()
        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        seq_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[seq_index, batch_index, path]
        for t in range(xmax):
            alpha = self._computes_transition(alpha, path, path_lens, log_probs_fwd_bwd[t], log_probs[t])
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(log_probs, elens.long())
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            beta = self._computes_transition(beta, r_path, path_lens, log_probs_fwd_bwd[t], log_probs_inv[t])
        best_lattices = log_probs.new_zeros((bs, xmax), dtype=torch.int64)
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            gamma = self._computes_transition(gamma, path, path_lens, log_probs_fwd_bwd[t], log_probs[t], skip_accum=True)
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(gamma == self.log0, self.log0)
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_lattices[b, t] = token_idx
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(self.log0)
            for b in range(bs):
                gamma[b, offsets[b]] = LOG_1
        trigger_lattices = torch.zeros((bs, xmax), dtype=torch.int64)
        trigger_points = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)
        for b in range(bs):
            n_triggers = 0
            trigger_points[b, ylens[b]] = elens[b] - 1
            for t in range(elens[b]):
                token_idx = best_lattices[b, t]
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_lattices[b, t - 1]):
                    continue
                trigger_lattices[b, t] = token_idx
                trigger_points[b, n_triggers] = t
                n_triggers += 1
        assert ylens.sum() == (trigger_lattices != 0).sum()
        return trigger_points


def kldiv_lsm_ctc(logits, ylens):
    """Compute KL divergence loss for label smoothing of CTC and Transducer models.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits.size()
    log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = torch.mul(probs, log_probs - log_uniform)
    loss_mean = np.sum([loss[(b), :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean


class CTC(DecoderBase):
    """Connectionist temporal classificaiton (CTC).

    Args:
        eos (int): index for <eos> (shared with <sos>)
        blank (int): index for <blank>
        enc_n_units (int):
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probability for the RNN layer
        lsm_prob (float): label smoothing probability
        fc_list (list):
        param_init (float): parameter initialization method
        backward (bool): flip the output sequence

    """

    def __init__(self, eos, blank, enc_n_units, vocab, dropout=0.0, lsm_prob=0.0, fc_list=None, param_init=0.1, backward=False):
        super(CTC, self).__init__()
        self.eos = eos
        self.blank = blank
        self.vocab = vocab
        self.lsm_prob = lsm_prob
        self.bwd = backward
        self.space = -1
        if fc_list is not None and len(fc_list) > 0:
            _fc_list = [int(fc) for fc in fc_list.split('_')]
            fc_layers = OrderedDict()
            for i in range(len(_fc_list)):
                input_dim = enc_n_units if i == 0 else _fc_list[i - 1]
                fc_layers['fc' + str(i)] = nn.Linear(input_dim, _fc_list[i])
                fc_layers['dropout' + str(i)] = nn.Dropout(p=dropout)
            fc_layers['fc' + str(len(_fc_list))] = nn.Linear(_fc_list[-1], vocab)
            self.output = nn.Sequential(fc_layers)
        else:
            self.output = nn.Linear(enc_n_units, vocab)
        self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)
        self.forced_aligner = CTCForcedAligner()

    def forward(self, eouts, elens, ys, forced_align=False):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length B
            ys (list): A list of length B, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[B, L, vocab]`

        """
        ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
        ys_ctc = torch.cat([np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int32)) for y in ys], dim=0)
        logits = self.output(eouts)
        loss = self.warpctc_loss(logits.transpose(1, 0), ys_ctc, elens.cpu(), ylens)
        if self.device_id >= 0:
            loss = loss
        if self.lsm_prob > 0:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits, elens) * self.lsm_prob
        trigger_points = None
        if forced_align:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id) for y in ys]
            ys_in_pad = pad_list(ys, 0)
            trigger_points = self.forced_aligner.align(logits.clone(), elens, ys_in_pad, ylens)
        return loss, trigger_points

    def trigger_points(self, eouts, elens):
        """Extract trigger points.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
        Returns:
            hyps (IntTensor): `[B, L]`

        """
        bs, xmax, _ = eouts.size()
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        best_paths = log_probs.argmax(-1)
        hyps = []
        for b in range(bs):
            indices = [best_paths[b, t].item() for t in range(elens[b])]
            collapsed_indices = [x[0] for x in groupby(indices)]
            best_hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            hyps.append(best_hyp)
        ymax = max([len(h) for h in hyps])
        trigger_points = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)
        for b in range(bs):
            n_triggers = 0
            for t in range(elens[b]):
                token_idx = best_paths[b, t]
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_paths[b, t - 1]):
                    continue
                trigger_points[b, n_triggers] = t
                n_triggers += 1
        return trigger_points

    def greedy(self, eouts, elens):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (np.ndarray): `[B]`
        Returns:
            hyps (np.ndarray): Best path hypothesis. `[B, L]`

        """
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        best_paths = log_probs.argmax(-1)
        hyps = []
        for b in range(eouts.size(0)):
            indices = [best_paths[b, t].item() for t in range(elens[b])]
            collapsed_indices = [x[0] for x in groupby(indices)]
            best_hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            hyps.append(np.array(best_hyp))
        return np.array(hyps)

    def beam_search(self, eouts, elens, params, idx2token, lm=None, lm_second=None, lm_second_rev=None, nbest=1, refs_id=None, utt_ids=None, speakers=None):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (list): A list of length B
            params (dict):
                recog_beam_width (int): size of beam
                recog_length_penalty (float): length penalty
                recog_lm_weight (float): weight of first path LM score
                recog_lm_second_weight (float): weight of second path LM score
                recog_lm_bwd_weight (float): weight of second path backward LM score
            idx2token (): converter from index to token
            lm: firsh path LM
            lm_second: second path LM
            lm_second_rev: secoding path backward LM
            nbest (int):
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            best_hyps (list): Best path hypothesis. `[B, L]`

        """
        bs = eouts.size(0)
        beam_width = params['recog_beam_width']
        lp_weight = params['recog_length_penalty']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        if lm_second is not None:
            assert lm_weight_second > 0
            lm_second.eval()
        best_hyps = []
        log_probs = torch.log_softmax(self.output(eouts), dim=-1)
        for b in range(bs):
            beam = [{'hyp': [self.eos], 'p_b': LOG_1, 'p_nb': LOG_0, 'score_lm': LOG_1, 'lmstate': None}]
            for t in range(elens[b]):
                new_beam = []
                log_probs_topk, topk_ids = torch.topk(log_probs[b:b + 1, (t)], k=min(beam_width, self.vocab), dim=-1, largest=True, sorted=True)
                for i_beam in range(len(beam)):
                    hyp = beam[i_beam]['hyp'][:]
                    p_b = beam[i_beam]['p_b']
                    p_nb = beam[i_beam]['p_nb']
                    score_lm = beam[i_beam]['score_lm']
                    new_p_b = np.logaddexp(p_b + log_probs[b, t, self.blank].item(), p_nb + log_probs[b, t, self.blank].item())
                    if len(hyp) > 1:
                        new_p_nb = p_nb + log_probs[b, t, hyp[-1]].item()
                    else:
                        new_p_nb = LOG_0
                    score_ctc = np.logaddexp(new_p_b, new_p_nb)
                    score_lp = len(hyp[1:]) * lp_weight
                    new_beam.append({'hyp': hyp, 'score': score_ctc + score_lm + score_lp, 'p_b': new_p_b, 'p_nb': new_p_nb, 'score_ctc': score_ctc, 'score_lm': score_lm, 'score_lp': score_lp, 'lmstate': beam[i_beam]['lmstate']})
                    if lm_weight > 0 and lm is not None:
                        _, lmstate, lm_log_probs = lm.predict(eouts.new_zeros(1, 1).fill_(hyp[-1]), beam[i_beam]['lmstate'])
                    else:
                        lmstate = None
                    new_p_b = LOG_0
                    for c in tensor2np(topk_ids)[0]:
                        p_t = log_probs[b, t, c].item()
                        if c == self.blank:
                            continue
                        c_prev = hyp[-1] if len(hyp) > 1 else None
                        if c == c_prev:
                            new_p_nb = p_b + p_t
                        else:
                            new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                            if c == self.space:
                                pass
                        score_ctc = np.logaddexp(new_p_b, new_p_nb)
                        score_lp = (len(hyp[1:]) + 1) * lp_weight
                        if lm_weight > 0 and lm is not None:
                            local_score_lm = lm_log_probs[0, 0, c].item() * lm_weight
                            score_lm += local_score_lm
                        new_beam.append({'hyp': hyp + [c], 'score': score_ctc + score_lm + score_lp, 'p_b': new_p_b, 'p_nb': new_p_nb, 'score_ctc': score_ctc, 'score_lm': score_lm, 'score_lp': score_lp, 'lmstate': lmstate})
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_width]
            if lm_second is not None:
                new_beam = []
                for i_beam in range(len(beam)):
                    ys = [np2tensor(np.fromiter(beam[i_beam]['hyp'], dtype=np.int64), self.device_id)]
                    ys_pad = pad_list(ys, lm_second.pad)
                    _, _, lm_log_probs = lm_second.predict(ys_pad, None)
                    score_ctc = np.logaddexp(beam[i_beam]['p_b'], beam[i_beam]['p_nb'])
                    score_lm = lm_log_probs.sum() * lm_weight_second
                    score_lp = len(beam[i_beam]['hyp'][1:]) * lp_weight
                    new_beam.append({'hyp': beam[i_beam]['hyp'], 'score': score_ctc + score_lm + score_lp, 'score_ctc': score_ctc, 'score_lp': score_lp, 'score_lm': score_lm})
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)
            best_hyps.append(np.array(beam[0]['hyp'][1:]))
            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            logger.info('Hyp: %s' % idx2token(beam[0]['hyp'][1:]))
            logger.info('log prob (hyp): %.7f' % beam[0]['score'])
            logger.info('log prob (CTC): %.7f' % beam[0]['score_ctc'])
            logger.info('log prob (lp): %.7f' % beam[0]['score_lp'])
            if lm is not None:
                logger.info('log prob (hyp, lm): %.7f' % beam[0]['score_lm'])
        return np.array(best_hyps)


class CTCPrefixScore(object):
    """Compute CTC label sequence scores.

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously

    [Reference]:
        https://github.com/espnet/espnet
    """

    def __init__(self, log_probs, blank, eos, truncate=False):
        """
        Args:
            log_probs (np.ndarray):
            blank (int): index of <blank>
            eos (int): index of <eos>
            truncate (bool): restart prefix search from the previous CTC spike

        """
        self.blank = blank
        self.eos = eos
        self.xlen_prev = 0
        self.xlen = len(log_probs)
        self.log_probs = log_probs
        self.log0 = LOG_0
        self.truncate = truncate
        self.offset = 0

    def initial_state(self):
        """Obtain an initial CTC state

        Returns:
            ctc_states (np.ndarray): `[T, 2]`

        """
        r = np.full((self.xlen, 2), self.log0, dtype=np.float32)
        r[0, 1] = self.log_probs[0, self.blank]
        for i in range(1, self.xlen):
            r[i, 1] = r[i - 1, 1] + self.log_probs[i, self.blank]
        return r

    def register_new_chunk(self, log_probs_chunk):
        self.xlen_prev = self.xlen
        self.log_probs = np.concatenate([self.log_probs, log_probs_chunk], axis=0)
        self.xlen = len(self.log_probs)

    def __call__(self, hyp, cs, r_prev, new_chunk=False):
        """Compute CTC prefix scores for next labels.

        Args:
            hyp (list): prefix label sequence
            cs (np.ndarray): array of next labels. A tensor of size `[beam_width]`
            r_prev (np.ndarray): previous CTC state `[T, 2]`
        Returns:
            ctc_scores (np.ndarray): `[beam_width]`
            ctc_states (np.ndarray): `[beam_width, T, 2]`

        """
        beam_width = len(cs)
        ylen = len(hyp) - 1
        r = np.ndarray((self.xlen, 2, beam_width), dtype=np.float32)
        xs = self.log_probs[:, (cs)]
        if ylen == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.log0
        else:
            r[ylen - 1] = self.log0
        if new_chunk and self.xlen_prev > 0:
            xlen_prev = r_prev.shape[0]
            r_new = np.full((self.xlen - xlen_prev, 2), self.log0, dtype=np.float32)
            r_new[0, 1] = r_prev[xlen_prev - 1, 1] + self.log_probs[xlen_prev, self.blank]
            for i in range(xlen_prev + 1, self.xlen):
                r_new[i - xlen_prev, 1] = r_new[i - xlen_prev - 1, 1] + self.log_probs[i, self.blank]
            r_prev = np.concatenate([r_prev, r_new], axis=0)
        r_sum = np.logaddexp(r_prev[:, (0)], r_prev[:, (1)])
        last = hyp[-1]
        if ylen > 0 and last in cs:
            log_phi = np.ndarray((self.xlen, beam_width), dtype=np.float32)
            for k in range(beam_width):
                log_phi[:, (k)] = r_sum if cs[k] != last else r_prev[:, (1)]
        else:
            log_phi = r_sum
        start = max(ylen, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.xlen):
            r[t, 0] = np.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = np.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.log_probs[t, self.blank]
            log_psi = np.logaddexp(log_psi, log_phi[t - 1] + xs[t])
        eos_pos = np.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]
        return log_psi, np.rollaxis(r, 2)


class MBR(torch.autograd.Function):
    """Minimum Bayes Risk (MBR) training.

    Args:
        vocab (int): number of nodes in softmax layer

    """

    @staticmethod
    def forward(ctx, log_probs, hyps, exp_risk, grad):
        """Forward pass.

        Args:
            log_probs (FloatTensor): `[N_best, L, vocab]`
            hyps (LongTensor): `[N_best, L]`
            exp_risk (FloatTensor): `[1]` (for forward)
            grad (FloatTensor): `[1]` (for backward)
        Returns:
            loss (FloatTensor): `[1]`

        """
        device_id = torch.cuda.device_of(log_probs).idx
        onehot = torch.eye(log_probs.size(-1))[hyps]
        grads = grad * onehot
        log_probs = log_probs.requires_grad_()
        ctx.save_for_backward(log_probs, grads)
        return exp_risk

    @staticmethod
    def backward(ctx, grad_output):
        input, grads = ctx.saved_tensors
        input.grad = grads
        return input, None, None, None


def append_sos_eos(xs, ys, sos, eos, pad, bwd=False, replace_sos=False):
    """Append <sos> and <eos> and return padded sequences.

    Args:
        xs (Tensor): for GPU id extraction
        ys (list): A list of length `[B]`, which contains a list of size `[L]`
        sos (int):
        eos (int):
        pad (int):
        bwd (bool):
        replace_sos (bool):
    Returns:
        ys_in (LongTensor): `[B, L]`
        ys_out (LongTensor): `[B, L]`
        ylens (IntTensor): `[B]`

    """
    device_id = torch.cuda.device_of(xs.data).idx
    _eos = xs.new_zeros(1).fill_(eos).long()
    ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64), device_id) for y in ys]
    if replace_sos:
        ylens = np2tensor(np.fromiter([(y[1:].size(0) + 1) for y in ys], dtype=np.int32))
        ys_in = pad_list([y for y in ys], pad)
        ys_out = pad_list([torch.cat([y[1:], _eos], dim=0) for y in ys], pad)
    else:
        _sos = xs.new_zeros(1).fill_(sos).long()
        ylens = np2tensor(np.fromiter([(y.size(0) + 1) for y in ys], dtype=np.int32))
        ys_in = pad_list([torch.cat([_sos, y], dim=0) for y in ys], pad)
        ys_out = pad_list([torch.cat([y, _eos], dim=0) for y in ys], pad)
    return ys_in, ys_out, ylens


def compute_wer(ref, hyp, normalize=False):
    """Compute Word Error Rate.

        [Reference]
            https://martin-thoma.com/word-error-rate-calculation/
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        wer (float): Word Error Rate between ref and hyp
        n_sub (int): the number of substitution
        n_ins (int): the number of insertion
        n_del (int): the number of deletion

    """
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)
    wer = d[len(ref)][len(hyp)]
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x > 0 and y > 0:
            if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                error_list.append('C')
                x = x - 1
                y = y - 1
            elif d[x][y] == d[x][y - 1] + 1:
                error_list.append('I')
                y = y - 1
            elif d[x][y] == d[x - 1][y - 1] + 1:
                error_list.append('S')
                x = x - 1
                y = y - 1
            else:
                error_list.append('D')
                x = x - 1
        elif x == 0 and y > 0:
            if d[x][y] == d[x][y - 1] + 1:
                error_list.append('I')
                y = y - 1
            else:
                error_list.append('D')
                x = x - 1
        elif y == 0 and x > 0:
            error_list.append('D')
            x = x - 1
        else:
            raise ValueError
    n_sub = error_list.count('S')
    n_ins = error_list.count('I')
    n_del = error_list.count('D')
    n_cor = error_list.count('C')
    assert wer == n_sub + n_ins + n_del
    assert n_cor == len(ref) - n_sub - n_del
    if normalize:
        wer /= len(ref)
    return wer * 100, n_sub * 100, n_ins * 100, n_del * 100


def distillation(logits_student, logits_teacher, ylens, temperature=5.0):
    """Compute cross entropy loss for knowledge distillation of sequence-to-sequence models.

    Args:
        logits_student (FloatTensor): `[B, T, vocab]`
        logits_teacher (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
        temperature (float):
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs, _, vocab = logits_student.size()
    log_probs_student = torch.log_softmax(logits_student, dim=-1)
    probs_teacher = torch.softmax(logits_teacher / temperature, dim=-1).data
    loss = -torch.mul(probs_teacher, log_probs_student)
    loss_mean = np.sum([loss[(b), :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean


class RNNDecoder(DecoderBase):
    """RNN decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of the encoder outputs
        attn_type (str): type of attention mechanism
        rnn_type (str): lstm/gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of the embedding and output layers
        attn_dim (int): dimension of attention space
        attn_sharpening_factor (float):
        attn_sigmoid_smoothing (bool):
        attn_conv_out_channels (int):
        attn_conv_kernel_size (int):
        attn_n_heads (int): number of attention heads
        dropout (float): dropout probability for the RNN layer
        dropout_emb (float): dropout probability for the embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        ss_prob (float): scheduled sampling probability
        ss_type (str): constant/ramp
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        mbr_training (bool): MBR training
        mbr_ce_weight (float): CE weight for regularization during MBR training
        external_lm (RNNLM):
        lm_fusion (str): type of LM fusion
        lm_init (bool):
        backward (bool): decode in the backward order
        global_weight (float):
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (float):
        mocha_chunk_size (int): chunk size for MoChA
        mocha_n_heads_mono (int):
        mocha_init_r (int):
        mocha_eps (float):
        mocha_std (float):
        mocha_no_denominator (bool):
        mocha_1dconv (bool): 1dconv for MoChA
        mocha_quantity_loss_weight (float):
        latency_metric (str): latency metric
        latency_loss_weight (float):
        gmm_attn_n_mixtures (int): number of mixtures for GMM attention
        replace_sos (bool): replace <sos> with special tokens
        distillation_weight (float): soft label weight for knowledge distillation
        discourse_aware (str): state_carry_over

    """

    def __init__(self, special_symbols, enc_n_units, attn_type, rnn_type, n_units, n_projs, n_layers, bottleneck_dim, emb_dim, vocab, tie_embedding, attn_dim, attn_sharpening_factor, attn_sigmoid_smoothing, attn_conv_out_channels, attn_conv_kernel_size, attn_n_heads, dropout, dropout_emb, dropout_att, lsm_prob, ss_prob, ss_type, ctc_weight, ctc_lsm_prob, ctc_fc_list, mbr_training, mbr_ce_weight, external_lm, lm_fusion, lm_init, backward, global_weight, mtl_per_batch, param_init, mocha_chunk_size, mocha_n_heads_mono, mocha_init_r, mocha_eps, mocha_std, mocha_no_denominator, mocha_1dconv, mocha_quantity_loss_weight, latency_metric, latency_loss_weight, gmm_attn_n_mixtures, replace_sos, distillation_weight, discourse_aware):
        super(RNNDecoder, self).__init__()
        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ss_prob = ss_prob
        self.ss_type = ss_type
        if ss_type == 'constant':
            self._ss_prob = ss_prob
        elif ss_type == 'ramp':
            self._ss_prob = 0
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
        self.lm_fusion = lm_fusion
        self.bwd = backward
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos
        self.distillation_weight = distillation_weight
        self.quantity_loss_weight = mocha_quantity_loss_weight
        self._quantity_loss_weight = mocha_quantity_loss_weight
        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        self.ctc_trigger = self.latency_metric in ['ctc_sync', 'ctc_dal'] or attn_type == 'triggered_attention'
        if self.ctc_trigger:
            assert 0 < self.ctc_weight < 1
        self.mbr_ce_weight = mbr_ce_weight
        self.mbr = MBR.apply if mbr_training else None
        self.discourse_aware = discourse_aware
        self.dstate_prev = None
        self.new_session = False
        self.prev_spk = ''
        self.dstates_final = None
        self.lmstate_final = None
        self.lmmemory = None
        self.aws_dict = {}
        self.data_dict = {}
        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos, blank=self.blank, enc_n_units=enc_n_units, vocab=vocab, dropout=dropout, lsm_prob=ctc_lsm_prob, fc_list=ctc_fc_list, param_init=param_init)
        if self.att_weight > 0:
            qdim = n_units if n_projs == 0 else n_projs
            if attn_type == 'mocha':
                assert attn_n_heads == 1
                self.score = MoChA(enc_n_units, qdim, attn_dim, atype='add', chunk_size=mocha_chunk_size, n_heads_mono=mocha_n_heads_mono, init_r=mocha_init_r, eps=mocha_eps, noise_std=mocha_std, no_denominator=mocha_no_denominator, conv1d=mocha_1dconv, sharpening_factor=attn_sharpening_factor, decot=latency_metric == 'decot', lookahead=2)
            elif attn_type == 'gmm':
                self.score = GMMAttention(enc_n_units, qdim, attn_dim, n_mixtures=gmm_attn_n_mixtures)
            elif attn_n_heads > 1:
                assert attn_type == 'add'
                self.score = MultiheadAttentionMechanism(enc_n_units, qdim, attn_dim, n_heads=attn_n_heads, dropout=dropout_att, atype='add')
            else:
                self.score = AttentionMechanism(enc_n_units, qdim, attn_dim, attn_type, sharpening_factor=attn_sharpening_factor, sigmoid_smoothing=attn_sigmoid_smoothing, conv_out_channels=attn_conv_out_channels, conv_kernel_size=attn_conv_kernel_size, dropout=dropout_att, lookahead=2)
            self.rnn = nn.ModuleList()
            cell = nn.LSTMCell if rnn_type == 'lstm' else nn.GRUCell
            if self.n_projs > 0:
                self.proj = repeat(nn.Linear(n_units, n_projs), n_layers)
            self.dropout = nn.Dropout(p=dropout)
            dec_odim = enc_n_units + emb_dim
            for l in range(n_layers):
                self.rnn += [cell(dec_odim, n_units)]
                dec_odim = n_units
                if self.n_projs > 0:
                    dec_odim = n_projs
            if external_lm is not None and lm_fusion:
                self.linear_dec_feat = nn.Linear(dec_odim + enc_n_units, n_units)
                if lm_fusion in ['cold', 'deep']:
                    self.linear_lm_feat = nn.Linear(external_lm.output_dim, n_units)
                    self.linear_lm_gate = nn.Linear(n_units * 2, n_units)
                elif lm_fusion == 'cold_prob':
                    self.linear_lm_feat = nn.Linear(external_lm.vocab, n_units)
                    self.linear_lm_gate = nn.Linear(n_units * 2, n_units)
                else:
                    raise ValueError(lm_fusion)
                self.output_bn = nn.Linear(n_units * 2, bottleneck_dim)
            else:
                self.output_bn = nn.Linear(dec_odim + enc_n_units, bottleneck_dim)
            self.embed = nn.Embedding(vocab, emb_dim, padding_idx=self.pad)
            self.dropout_emb = nn.Dropout(p=dropout_emb)
            assert bottleneck_dim > 0, 'bottleneck_dim must be larger than zero.'
            self.output = nn.Linear(bottleneck_dim, vocab)
            if tie_embedding:
                if emb_dim != bottleneck_dim:
                    raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                self.output.weight = self.embed.weight
        self.reset_parameters(param_init)
        self.lm = external_lm
        if lm_init:
            assert lm_init.vocab == vocab
            assert lm_init.n_units == n_units
            assert lm_init.emb_dim == emb_dim
            logger.info('===== Initialize the decoder with pre-trained RNNLM')
            assert lm_init.n_projs == 0
            assert lm_init.n_units_null_context == enc_n_units
            for l in range(lm_init.n_layers):
                for n, p in lm_init.rnn[l].named_parameters():
                    assert getattr(self.rnn[l], n).size() == p.size()
                    getattr(self.rnn[l], n).data = p.data
                    logger.info('Overwrite %s' % n)
            assert self.embed.weight.size() == lm_init.embed.weight.size()
            self.embed.weight.data = lm_init.embed.weight.data
            logger.info('Overwrite %s' % 'embed.weight')

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'score.monotonic_energy.v.weight_g' in n or 'score.monotonic_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'score.chunk_energy.v.weight_g' in n or 'score.chunk_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if p.dim() == 1:
                if 'linear_lm_gate.fc.bias' in n:
                    nn.init.constant_(p, -1.0)
                    logger.info('Initialize %s with %s / %.3f' % (n, 'constant', -1.0))
                else:
                    nn.init.constant_(p, 0.0)
                    logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() in [2, 3, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)

    def start_scheduled_sampling(self):
        self._ss_prob = self.ss_prob

    def forward(self, eouts, elens, ys, task='all', teacher_logits=None, recog_params={}, idx2token=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None, 'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros(1)
        trigger_points = None
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            forced_align = self.ctc_trigger and self.training or self.attn_type == 'triggered_attention'
            loss_ctc, trigger_points = self.ctc(eouts, elens, ys, forced_align=forced_align)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
        if self.att_weight > 0 and (task == 'all' or 'ctc' not in task) and self.mbr is None:
            loss_att, acc_att, ppl_att, loss_quantity, loss_latency = self.forward_att(eouts, elens, ys, teacher_logits=teacher_logits, trigger_points=trigger_points)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.attn_type == 'mocha':
                if self._quantity_loss_weight > 0:
                    loss_att += loss_quantity * self._quantity_loss_weight
                observation['loss_quantity'] = loss_quantity.item()
            if self.latency_metric:
                observation['loss_latency'] = loss_latency.item() if self.training else 0
                if self.latency_metric != 'decot' and self.latency_loss_weight > 0:
                    loss_att += loss_latency * self.latency_loss_weight
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * self.att_weight
        if self.mbr is not None and (task == 'all' or 'mbr' not in task):
            N_best = recog_params['recog_beam_width']
            alpha = 1.0
            assert N_best >= 2
            loss_mbr = 0.0
            loss_ce = 0.0
            bs = eouts.size(0)
            for b in range(bs):
                self.eval()
                with torch.no_grad():
                    nbest_hyps_id, _, log_scores = self.beam_search(eouts[b:b + 1], elens[b:b + 1], params=recog_params, nbest=N_best, exclude_eos=True)
                nbest_hyps_id_b = [np.fromiter(y, dtype=np.int64) for y in nbest_hyps_id[0]]
                log_scores_b = np2tensor(np.array(log_scores[0], dtype=np.float32), self.device_id)
                scores_b_norm = torch.softmax(alpha * log_scores_b, dim=-1)
                wers_b = np2tensor(np.array([(compute_wer(ref=idx2token(ys[b]).split(' '), hyp=idx2token(nbest_hyps_id_b[n]).split(' '))[0] / 100) for n in range(N_best)], dtype=np.float32), self.device_id)
                exp_wer_b = (scores_b_norm * wers_b).sum()
                grad_b = (scores_b_norm * (wers_b - exp_wer_b)).sum()
                self.train()
                logits_b = self.forward_mbr(eouts[b:b + 1].repeat([N_best, 1, 1]), elens[b:b + 1].repeat([N_best]), nbest_hyps_id_b)
                log_probs_b = torch.log_softmax(logits_b, dim=-1)
                _eos = eouts.new_zeros(1).fill_(self.eos).long()
                nbest_hyps_id_b_pad = pad_list([torch.cat([np2tensor(y, self.device_id), _eos], dim=0) for y in nbest_hyps_id_b], self.pad)
                loss_mbr += self.mbr(log_probs_b, nbest_hyps_id_b_pad, exp_wer_b, grad_b)
                loss_ce += self.forward_att(eouts[b:b + 1], elens[b:b + 1], ys[b:b + 1])[0]
            loss = loss_mbr + loss_ce * self.mbr_ce_weight
            observation['loss_mbr'] = loss_mbr.item()
            observation['loss_att'] = loss_ce.item()
        observation['loss'] = loss.item()
        return loss, observation

    def forward_mbr(self, eouts, elens, ys_hyp):
        """Compute XE loss for the attention-based decoder.

        Args:
            eouts (FloatTensor): `[N_best, T, enc_n_units]`
            elens (IntTensor): `[N_best]`
            ys_hyp (list): length `N_best`, each of which contains a list of size `[L]`
        Returns:
            logits (FloatTensor): `[N_best, L, vocab]`

        """
        bs, xmax = eouts.size()[:2]
        ys_in, ys_out, ylens = append_sos_eos(eouts, ys_hyp, self.eos, self.eos, self.pad)
        dstates = self.zero_state(bs)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw, aws = None, []
        betas = []
        lmout, lmstate = None, None
        ys_emb = self.dropout_emb(self.embed(ys_in))
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1)
        logits = []
        for t in range(ys_in.size(1)):
            is_sample = t > 0 and self._ss_prob > 0 and random.random() < self._ss_prob
            if self.lm is not None:
                y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in[:, t:t + 1]
                lmout, lmstate, _ = self.lm.predict(y_lm, lmstate)
            y_emb = self.dropout_emb(self.embed(self.output(logits[-1]).detach().argmax(-1))) if is_sample else ys_emb[:, t:t + 1]
            dstates, cv, aw, attn_v, beta = self.decode_step(eouts, dstates, cv, y_emb, src_mask, aw, lmout, mode='parallel')
            aws.append(aw)
            if beta is not None:
                betas.append(beta)
            logits.append(attn_v)
        with torch.no_grad():
            aws = torch.cat(aws, dim=2)
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.aws_dict['xy_aws'] = tensor2np(aws)
            if len(betas) > 0:
                betas = torch.cat(betas, dim=2)
                self.aws_dict['xy_aws_beta'] = tensor2np(betas)
        logits = self.output(torch.cat(logits, dim=1))
        return logits

    def forward_att(self, eouts, elens, ys, return_logits=False, teacher_logits=None, trigger_points=None):
        """Compute XE loss for the attention-based decoder.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
            trigger_points (IntTensor): `[B, T]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            loss_quantity (FloatTensor): `[1]`
            loss_latency (FloatTensor): `[1]`

        """
        bs, xmax = eouts.size()[:2]
        ys_in, ys_out, ylens = append_sos_eos(eouts, ys, self.eos, self.eos, self.pad, self.bwd)
        ymax = ys_in.size(1)
        dstates = self.zero_state(bs)
        if self.training:
            if self.discourse_aware and not self.new_session:
                dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
            self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
            self.new_session = False
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw, aws = None, []
        betas = []
        lmout, lmstate = None, None
        ys_emb = self.dropout_emb(self.embed(ys_in))
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1)
        tgt_mask = (ys_out != self.pad).unsqueeze(2)
        logits = []
        for t in range(ymax):
            is_sample = t > 0 and self._ss_prob > 0 and random.random() < self._ss_prob
            if self.lm is not None:
                self.lm.eval()
                with torch.no_grad():
                    y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in[:, t:t + 1]
                    lmout, lmstate, _ = self.lm.predict(y_lm, lmstate)
            y_emb = self.dropout_emb(self.embed(self.output(logits[-1]).detach().argmax(-1))) if is_sample else ys_emb[:, t:t + 1]
            dstates, cv, aw, attn_v, beta = self.decode_step(eouts, dstates, cv, y_emb, src_mask, aw, lmout, mode='parallel', trigger_point=trigger_points[:, (t)] if trigger_points is not None else None)
            aws.append(aw)
            if beta is not None:
                betas.append(beta)
            logits.append(attn_v)
            if self.training and self.discourse_aware:
                for b in [b for b, ylen in enumerate(ylens.tolist()) if t == ylen - 1]:
                    self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1].detach()
                    if self.rnn_type == 'lstm':
                        self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1].detach()
        if self.training and self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = torch.cat(self.dstate_prev['hxs'], dim=1)
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = torch.cat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs'][0]
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = self.dstate_prev['cxs'][0]
        logits = self.output(torch.cat(logits, dim=1))
        if return_logits:
            return logits
        aws = torch.cat(aws, dim=2)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.aws_dict['xy_aws'] = tensor2np(aws)
            if len(betas) > 0:
                betas = torch.cat(betas, dim=2)
                self.aws_dict['xy_aws_beta'] = tensor2np(betas)
        n_heads = aws.size(1)
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)
        if self.attn_type == 'mocha' or trigger_points is not None:
            aws = aws.masked_fill_(src_mask.unsqueeze(1).repeat([1, n_heads, 1, 1]) == 0, 0)
            aws = aws.masked_fill_(tgt_mask.unsqueeze(1).repeat([1, n_heads, 1, 1]) == 0, 0)
        loss_quantity = 0.0
        if self.attn_type == 'mocha':
            n_tokens_pred = aws.sum(3).sum(2).sum(1) / n_heads
            n_tokens_ref = tgt_mask.squeeze(2).sum(1).float()
            loss_quantity = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))
        loss_latency = 0.0
        if self.latency_metric == 'interval':
            assert trigger_points is None
            assert aws.size(1) == 1
            aws_prev = torch.cat([aws.new_zeros(aws.size())[:, :, -1:], aws.clone()[:, :, :-1]], dim=2)
            aws_mat = aws_prev.unsqueeze(3) * aws.unsqueeze(4)
            delay_mat = aws.new_ones(xmax, xmax).float()
            delay_mat = torch.tril(delay_mat, diagonal=-1, out=delay_mat)
            delay_mat = torch.cumsum(delay_mat, dim=-2).unsqueeze(0)
            delay_mat = delay_mat.unsqueeze(1).unsqueeze(2).expand_as(aws_mat)
            loss_latency = torch.pow((aws_mat * delay_mat).sum(-1), 2).sum(-1)
            loss_latency = torch.mean(loss_latency.squeeze(1))
        elif trigger_points is not None:
            js = torch.arange(xmax).float()
            if self.device_id >= 0:
                js = js
            js = js.repeat([bs, n_heads, ymax, 1])
            exp_trigger_points = (js * aws).sum(3)
            trigger_points = trigger_points.float()
            if self.device_id >= 0:
                trigger_points = trigger_points
            trigger_points = trigger_points.unsqueeze(1)
            if self.latency_metric == 'ctc_sync':
                loss_latency = torch.abs(exp_trigger_points - trigger_points)
            elif self.latency_metric == 'ctc_dal':
                loss_latency = torch.abs(exp_trigger_points - trigger_points)
            else:
                raise NotImplementedError(self.latency_metric)
            loss_latency = loss_latency.sum() / ylens.sum()
        if teacher_logits is not None:
            kl_loss = distillation(logits, teacher_logits, ylens, temperature=5.0)
            loss = loss * (1 - self.distillation_weight) + kl_loss * self.distillation_weight
        acc = compute_accuracy(logits, ys_out, self.pad)
        return loss, acc, ppl, loss_quantity, loss_latency

    def decode_step(self, eouts, dstates, cv, y_emb, mask, aw, lmout, mode='hard', trigger_point=None, cache=True):
        dstates = self.recurrency(torch.cat([y_emb, cv], dim=-1), dstates['dstate'])
        cv, aw, beta = self.score(eouts, eouts, dstates['dout_score'], mask, aw, cache=cache, mode=mode, trigger_point=trigger_point)
        attn_v = self.generate(cv, dstates['dout_gen'], lmout)
        return dstates, cv, aw, attn_v, beta

    def zero_state(self, bs):
        """Initialize decoder state.

        Args:
            bs (int): batch size
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        dstates = {'dstate': None}
        w = next(self.parameters())
        hxs = w.new_zeros(self.n_layers, bs, self.dec_n_units)
        cxs = w.new_zeros(self.n_layers, bs, self.dec_n_units) if self.rnn_type == 'lstm' else None
        dstates['dstate'] = hxs, cxs
        return dstates

    def recurrency(self, inputs, dstate):
        """Recurrency function.

        Args:
            inputs (FloatTensor): `[B, 1, emb_dim + enc_n_units]`
            dstate (tuple): A tuple of (hxs, cxs)
        Returns:
            new_dstates (dict):
                dout_score (FloatTensor): `[B, 1, dec_n_units]`
                dout_gen (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        hxs, cxs = dstate
        dout = inputs.squeeze(1)
        new_dstates = {'dout_score': None, 'dout_gen': None, 'dstate': None}
        new_hxs, new_cxs = [], []
        for lth in range(self.n_layers):
            if self.rnn_type == 'lstm':
                h, c = self.rnn[lth](dout, (hxs[lth], cxs[lth]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru':
                h = self.rnn[lth](dout, hxs[lth])
            new_hxs.append(h)
            dout = self.dropout(h)
            if self.n_projs > 0:
                dout = torch.tanh(self.proj[lth](dout))
            if lth == 0:
                new_dstates['dout_score'] = dout.unsqueeze(1)
        new_hxs = torch.stack(new_hxs, dim=0)
        if self.rnn_type == 'lstm':
            new_cxs = torch.stack(new_cxs, dim=0)
        new_dstates['dout_gen'] = dout.unsqueeze(1)
        new_dstates['dstate'] = new_hxs, new_cxs
        return new_dstates

    def generate(self, cv, dout, lmout):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_n_units]`
            lmout (FloatTensor): `[B, 1, lm_n_units]`
        Returns:
            attn_v (FloatTensor): `[B, 1, vocab]`

        """
        gated_lmout = None
        if self.lm is not None:
            dec_feat = self.linear_dec_feat(torch.cat([dout, cv], dim=-1))
            if self.lm_fusion in ['cold', 'deep']:
                lmout = self.linear_lm_feat(lmout)
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmout], dim=-1)))
                gated_lmout = gate * lmout
            elif self.lm_fusion == 'cold_prob':
                lmout = self.linear_lm_feat(self.lm.output(lmout))
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmout], dim=-1)))
                gated_lmout = gate * lmout
            out = self.output_bn(torch.cat([dec_feat, gated_lmout], dim=-1))
        else:
            out = self.output_bn(torch.cat([dout, cv], dim=-1))
        attn_v = torch.tanh(out)
        return attn_v

    def _plot_attention(self, save_path, n_cols=1):
        """Plot attention for each head."""
        if self.att_weight == 0:
            return 0
        _save_path = mkdir_join(save_path, 'dec_att_weights')
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)
        elens = self.data_dict['elens']
        ylens = self.data_dict['ylens']
        for k, aw in self.aws_dict.items():
            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp, figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                ax.imshow(aw[(-1), (h), :ylens[-1], :elens[-1]], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()

    def greedy(self, eouts, elens, max_len_ratio, idx2token, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, trigger_points=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            trigger_points (IntTensor): `[B, T]`
        Returns:
            hyps (list): length `B`, each of which contains arrays of size `[L]`
            aws (list): length `B`, each of which contains arrays of size `[H, L, T]`

        """
        bs, xmax, _ = eouts.size()
        dstates = self.zero_state(bs)
        if self.discourse_aware and not self.new_session:
            dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
        self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
        self.new_session = False
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw = None
        lmout, lmstate = None, None
        y = eouts.new_zeros(bs, 1).fill_(refs_id[0][0] if self.replace_sos else self.eos).long()
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1)
        if self.attn_type == 'triggered_attention':
            assert trigger_points is not None
        hyps_batch, aws_batch = [], []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        ymax = int(math.floor(xmax * max_len_ratio)) + 1
        for t in range(ymax):
            if self.lm is not None:
                lmout, lmstate = self.lm.decode(self.lm(y), lmstate)
            y_emb = self.dropout_emb(self.embed(y))
            dstates, cv, aw, attn_v, _ = self.decode_step(eouts, dstates, cv, y_emb, src_mask, aw, lmout, trigger_point=trigger_points[:, (t)] if trigger_points is not None else None)
            aws_batch += [aw]
            y = self.output(attn_v).argmax(-1)
            hyps_batch += [y]
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                        if self.discourse_aware:
                            self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1]
                            if self.rnn_type == 'lstm':
                                self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1]
                    ylens[b] += 1
            if sum(eos_flags) == bs:
                break
            if t == ymax - 1:
                break
        if self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = torch.cat(self.dstate_prev['hxs'], dim=1)
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = torch.cat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs']
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = self.dstate_prev['cxs']
        self.lmstate_final = lmstate
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        aws_batch = tensor2np(torch.cat(aws_batch, dim=2))
        if self.bwd:
            hyps = [hyps_batch[(b), :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_batch[(b), :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[(b), :ylens[b]] for b in range(bs)]
            aws = [aws_batch[(b), :, :ylens[b]] for b in range(bs)]
        if exclude_eos:
            if self.bwd:
                hyps = [(hyps[b][1:] if eos_flags[b] else hyps[b]) for b in range(bs)]
            else:
                hyps = [(hyps[b][:-1] if eos_flags[b] else hyps[b]) for b in range(bs)]
        if idx2token is not None:
            for b in range(bs):
                if utt_ids is not None:
                    logger.debug('Utt-id: %s' % utt_ids[b])
                if refs_id is not None and self.vocab == idx2token.vocab:
                    logger.debug('Ref: %s' % idx2token(refs_id[b]))
                if self.bwd:
                    logger.debug('Hyp: %s' % idx2token(hyps[b][::-1]))
                else:
                    logger.debug('Hyp: %s' % idx2token(hyps[b]))
        return hyps, aws

    def beam_search(self, eouts, elens, params, idx2token=None, lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None, nbest=1, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[], cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
            idx2token (): converter from index to token
            lm: firsh path LM
            lm_second: second path LM
            lm_second_bwd: secoding path backward LM
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
            cache_states (bool): cache TransformerLM/TransformerXL states for fast decoding
        Returns:
            nbest_hyps_idx (list): length `B`, each of which contains list of N hypotheses
            aws (list): length `B`, each of which contains arrays of size `[H, L, T]`
            scores (list):

        """
        bs, xmax, _ = eouts.size()
        n_models = len(ensmbl_decs) + 1
        beam_width = params['recog_beam_width']
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        cp_weight = params['recog_coverage_penalty']
        cp_threshold = params['recog_coverage_threshold']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        gnmt_decoding = params['recog_gnmt_decoding']
        eos_threshold = params['recog_eos_threshold']
        asr_state_CO = params['recog_asr_state_carry_over']
        lm_state_CO = params['recog_lm_state_carry_over']
        softmax_smoothing = params['recog_softmax_smoothing']
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        if lm_second is not None:
            assert lm_weight_second > 0
            lm_second.eval()
        if lm_second_bwd is not None:
            assert lm_weight_second_bwd > 0
            lm_second_bwd.eval()
        trfm_lm = isinstance(lm, TransformerLM) or isinstance(lm, TransformerXL)
        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)
        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            self.score.reset()
            dstates = self.zero_state(1)
            lmstate = None
            ys = eouts.new_zeros(1, 1).fill_(self.eos).long()
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)
            ensmbl_dstate, ensmbl_cv = [], []
            if n_models > 1:
                for dec in ensmbl_decs:
                    ensmbl_dstate += [dec.zero_state(1)]
                    ensmbl_cv += [eouts.new_zeros(1, 1, dec.enc_n_units)]
                    dec.score.reset()
            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if asr_state_CO:
                        dstates = self.dstates_final
                    if lm_state_CO:
                        if isinstance(lm, RNNLM):
                            lmstate = self.lmstate_final
                        elif isinstance(lm, TransformerLM):
                            ys_prev = self.lmstate_final
                            _, lmstate, _ = lm.predict(ys_prev)
                            ys = torch.cat([ys_prev, ys], dim=1)
                else:
                    self.dstates_final = None
                    self.lmstate_final = None
                    self.lmmemory = None
                self.prev_spk = speakers[b]
            helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id)
            end_hyps = []
            hyps = [{'hyp': [self.eos], 'ys': ys, 'score': 0.0, 'score_att': 0.0, 'score_ctc': 0.0, 'score_lm': 0.0, 'dstates': dstates, 'cv': eouts.new_zeros(1, 1, self.enc_n_units), 'aws': [None], 'lmstate': lmstate, 'ensmbl_dstate': ensmbl_dstate, 'ensmbl_cv': ensmbl_cv, 'ensmbl_aws': [[None]] * (n_models - 1), 'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None}]
            ymax = int(math.floor(elens[b] * max_len_ratio)) + 1
            for t in range(ymax):
                y = eouts.new_zeros(len(hyps), 1).long()
                for j, beam in enumerate(hyps):
                    if self.replace_sos and t == 0:
                        prev_idx = refs_id[0][0]
                    else:
                        prev_idx = beam['hyp'][-1]
                    y[j, 0] = prev_idx
                cv = torch.cat([beam['cv'] for beam in hyps], dim=0)
                aw = torch.cat([beam['aws'][-1] for beam in hyps], dim=0) if t > 0 else None
                hxs = torch.cat([beam['dstates']['dstate'][0] for beam in hyps], dim=1)
                if self.rnn_type == 'lstm':
                    cxs = torch.cat([beam['dstates']['dstate'][1] for beam in hyps], dim=1)
                dstates = {'dstate': (hxs, cxs)}
                lmout, lmstate, scores_lm = None, None, None
                if lm is not None or self.lm is not None:
                    if trfm_lm:
                        ys = eouts.new_zeros(len(hyps), beam['ys'].size(1)).long()
                        for j, cand in enumerate(hyps):
                            ys[(j), :] = cand['ys']
                    else:
                        ys = y
                    if t > 0 or t == 0 and trfm_lm and lm_state_CO and self.lmstate_final is not None:
                        if isinstance(lm, RNNLM):
                            lmstate = {'hxs': torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1), 'cxs': torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)}
                        elif trfm_lm:
                            if isinstance(lm, TransformerLM):
                                lmstate = [torch.cat([beam['lmstate'][l] for beam in hyps], dim=0) for l in range(lm.n_layers)]
                            elif t > 0:
                                lmstate = [torch.cat([beam['lmstate'][l] for beam in hyps], dim=0) for l in range(lm.n_layers)]
                    if self.lm is not None:
                        lmout, lmstate, scores_lm = self.lm.predict(y, lmstate)
                    elif lm is not None:
                        lmout, lmstate, scores_lm = lm.predict(ys, lmstate, mems=self.lmmemory, cache=lmstate if cache_states else None)
                dstates, cv, aw, attn_v, _ = self.decode_step(eouts[b:b + 1, :elens[b]].repeat([cv.size(0), 1, 1]), dstates, cv, self.dropout_emb(self.embed(y)), None, aw, lmout)
                probs = torch.softmax(self.output(attn_v).squeeze(1) * softmax_smoothing, dim=1)
                ensmbl_dstate, ensmbl_cv, ensmbl_aws = [], [], []
                if n_models > 1:
                    for i_e, dec in enumerate(ensmbl_decs):
                        cv_e = torch.cat([beam['ensmbl_cv'][i_e] for beam in hyps], dim=0)
                        aw_e = torch.cat([beam['ensmbl_aws'][i_e][-1] for beam in hyps], dim=0) if t > 0 else None
                        hxs_e = torch.cat([beam['ensmbl_dstate'][i_e]['dstate'][0] for beam in hyps], dim=1)
                        if self.rnn_type == 'lstm':
                            cxs_e = torch.cat([beam['dstates'][i_e]['dstate'][1] for beam in hyps], dim=1)
                        dstates_e = {'dstate': (hxs_e, cxs_e)}
                        dstate_e, cv_e, aw_e, attn_v_e, _ = dec.decode_step(ensmbl_eouts[i_e][b:b + 1, :ensmbl_elens[i_e][b]].repeat([cv_e.size(0), 1, 1]), dstates_e, cv_e, dec.dropout_emb(dec.embed(y)), None, aw_e, lmout)
                        ensmbl_dstate += [{'dstate': (beam['dstates'][i_e]['dstate'][0][:, j:j + 1], beam['dstates'][i_e]['dstate'][1][:, j:j + 1])}]
                        ensmbl_cv += [cv_e[j:j + 1]]
                        ensmbl_aws += [beam['ensmbl_aws'][i_e] + [aw_e[j:j + 1]]]
                        probs += torch.softmax(dec.output(attn_v_e).squeeze(1), dim=1)
                scores_att = torch.log(probs / n_models)
                new_hyps = []
                for j, beam in enumerate(hyps):
                    total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                    total_scores = total_scores_att * (1 - ctc_weight)
                    total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(beam_width)
                    if lp_weight > 0:
                        if gnmt_decoding:
                            lp = math.pow(6 + len(beam['hyp'][1:]), lp_weight) / math.pow(6, lp_weight)
                            total_scores_topk /= lp
                        else:
                            total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight
                    if cp_weight > 0:
                        aw_mat = torch.cat(beam['aws'][1:] + [aw[j:j + 1]], dim=2)
                        aw_mat = aw_mat[:, (0), :, :]
                        if gnmt_decoding:
                            aw_mat = torch.log(aw_mat.sum(-1))
                            cp = torch.where(aw_mat < 0, aw_mat, aw_mat.new_zeros(aw_mat.size())).sum()
                            total_scores_topk += cp * cp_weight
                        else:
                            if cp_threshold == 0:
                                cp = aw_mat.sum() / self.score.n_heads
                            else:
                                cp = torch.where(aw_mat > cp_threshold, aw_mat, aw_mat.new_zeros(aw_mat.size())).sum() / self.score.n_heads
                            total_scores_topk += cp * cp_weight
                    else:
                        cp = 0.0
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(beam['hyp'], topk_ids, beam['ctc_state'], total_scores_topk, ctc_prefix_scorer)
                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        length_norm_factor = 1.0
                        if length_norm:
                            length_norm_factor = len(beam['hyp'][1:]) + 1
                        total_score = total_scores_topk[0, k].item() / length_norm_factor
                        if idx == self.eos:
                            if len(beam['hyp']) - 1 < elens[b] * min_len_ratio:
                                continue
                            max_score_no_eos = scores_att[(j), :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_att[(j), idx + 1:].max(0)[0].item())
                            if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue
                        new_lmstate = None
                        if lmstate is not None:
                            if isinstance(lm, RNNLM):
                                new_lmstate = {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]}
                            elif trfm_lm:
                                new_lmstate = [lmstate_l[j:j + 1] for lmstate_l in lmstate]
                            else:
                                raise ValueError(type(lm))
                        ys = torch.cat([beam['ys'], eouts.new_zeros(1, 1).fill_(idx).long()], dim=-1)
                        new_hyps.append({'hyp': beam['hyp'] + [idx], 'ys': ys, 'score': total_score, 'score_att': total_scores_att[0, idx].item(), 'score_cp': cp, 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[k].item(), 'dstates': {'dstate': (dstates['dstate'][0][:, j:j + 1], dstates['dstate'][1][:, j:j + 1])}, 'cv': cv[j:j + 1], 'aws': beam['aws'] + [aw[j:j + 1]], 'lmstate': new_lmstate, 'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None, 'ensmbl_dstate': ensmbl_dstate, 'ensmbl_cv': ensmbl_cv, 'ensmbl_aws': ensmbl_aws})
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps_sorted, end_hyps)
                hyps = new_hyps[:]
                if is_finish:
                    break
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])
            if lm_second is not None:
                self.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')
            if lm_second_bwd is not None:
                self.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_bwd')
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)
            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:][::-1] if self.bwd else end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_att'] * (1 - ctc_weight)))
                    logger.info('log prob (hyp, cp): %.7f' % (end_hyps[k]['score_cp'] * cp_weight))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' % (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-path lm, reverse): %.7f' % (end_hyps[k]['score_lm_second_rev'] * lm_weight_second_bwd))
                    logger.info('-' * 50)
            if self.bwd:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [tensor2np(torch.cat(end_hyps[0]['aws'][1:][::-1], dim=2).squeeze(0))]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [tensor2np(torch.cat(end_hyps[0]['aws'][1:], dim=2).squeeze(0))]
            if length_norm:
                scores += [[(end_hyps[n]['score_att'] / len(end_hyps[n]['hyp'][1:])) for n in range(nbest)]]
            else:
                scores += [[end_hyps[n]['score_att'] for n in range(nbest)]]
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][1:] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][:-1] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
        self.dstates_final = end_hyps[0]['dstates']
        if isinstance(lm, RNNLM):
            self.lmstate_final = end_hyps[0]['lmstate']
        elif trfm_lm:
            if isinstance(lm, TransformerXL):
                self.lmmemory = lm.update_memory(self.lmmemory, end_hyps[0]['lmstate'])
                logging.info('Memory: %d' % self.lmmemory[0].size(1))
            else:
                ys = end_hyps[0]['ys']
                if ys[0, -1].item() == self.eos:
                    ys = ys[:, :-1]
                ys = ys[:, -lm.mem_len:]
            self.lmstate_final = ys
        return nbest_hyps_idx, aws, scores

    def beam_search_chunk_sync(self, eouts_c, params, idx2token, lm=None, ctc_log_probs=None, hyps=False, state_carry_over=False, ignore_eos=False):
        bs, chunk_size, _ = eouts_c.size()
        assert bs == 1
        assert self.attn_type == 'mocha'
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        lp_weight = params['recog_length_penalty']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        eos_threshold = params['recog_eos_threshold']
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        self.score.reset()
        dstates = self.zero_state(1)
        lmstate = None
        ctc_state = None
        self.ctc_prefix_scorer = None
        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)
            if hyps is None:
                self.ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[0], self.blank, self.eos)
            else:
                self.ctc_prefix_scorer.register_new_chunk(ctc_log_probs[0])
            ctc_state = self.ctc_prefix_scorer.initial_state()
        if state_carry_over:
            dstates = self.dstates_final
            if isinstance(lm, RNNLM):
                lmstate = self.lmstate_final
        helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id)
        end_hyps = []
        hyps_nobd = []
        if hyps is None:
            self.n_frames = 0
            self.chunk_size = eouts_c.size(1)
            hyps = [{'hyp': [self.eos], 'score': 0.0, 'score_att': 0.0, 'score_ctc': 0.0, 'score_lm': 0.0, 'dstates': dstates, 'cv': eouts_c.new_zeros(1, 1, self.enc_n_units), 'aws': [None], 'lmstate': lmstate, 'ctc_state': ctc_state, 'no_boundary': False}]
        else:
            for h in hyps:
                h['no_boundary'] = False
        ymax = int(math.floor(eouts_c.size(1) * max_len_ratio)) + 1
        for t in range(ymax):
            if len(hyps) == 0:
                break
            if t > 0 and sum([cand['no_boundary'] for cand in hyps]) == len(hyps):
                break
            new_hyps = []
            hyps_filtered = []
            for j, beam in enumerate(hyps):
                if beam['no_boundary']:
                    new_hyps.append(beam.copy())
                else:
                    hyps_filtered.append(beam.copy())
            if len(hyps_filtered) == 0:
                break
            hyps = hyps_filtered[:]
            y = eouts_c.new_zeros(len(hyps), 1).long()
            for j, beam in enumerate(hyps):
                y[j, 0] = beam['hyp'][-1]
            cv = torch.cat([beam['cv'] for beam in hyps], dim=0)
            aw = torch.cat([beam['aws'][-1] for beam in hyps], dim=0) if t > 0 else None
            hxs = torch.cat([beam['dstates']['dstate'][0] for beam in hyps], dim=1)
            if self.rnn_type == 'lstm':
                cxs = torch.cat([beam['dstates']['dstate'][1] for beam in hyps], dim=1)
            dstates = {'dstate': (hxs, cxs)}
            lmout, lmstate, scores_lm = None, None, None
            if lm is not None or self.lm is not None:
                if beam['lmstate'] is not None:
                    lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                    lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                    lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
                if self.lm is not None:
                    lmout, lmstate, scores_lm = self.lm.predict(y, lmstate)
                elif lm is not None:
                    lmout, lmstate, scores_lm = lm.predict(y, lmstate)
            dstates, cv, aw, attn_v, _ = self.decode_step(eouts_c[0:1].repeat([cv.size(0), 1, 1]), dstates, cv, self.dropout_emb(self.embed(y)), None, aw, lmout, cache=False)
            scores_att = torch.log_softmax(self.output(attn_v).squeeze(1), dim=1)
            for j, beam in enumerate(hyps):
                no_boundary = aw[j].sum().item() == 0
                if no_boundary:
                    beam['aws'][-1] = eouts_c.new_zeros(eouts_c.size(0), 1, 1, eouts_c.size(1))
                    beam['no_boundary'] = True
                    new_hyps.append(beam.copy())
                total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                total_scores = total_scores_att * (1 - ctc_weight)
                total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                if lm is not None:
                    total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                    total_scores_topk += total_scores_lm * lm_weight
                else:
                    total_scores_lm = eouts_c.new_zeros(beam_width)
                total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight
                new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(beam['hyp'], topk_ids, beam['ctc_state'], total_scores_topk, self.ctc_prefix_scorer, new_chunk=t == 0)
                for k in range(beam_width):
                    idx = topk_ids[0, k].item()
                    if no_boundary and idx != self.eos:
                        continue
                    length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                    total_score = total_scores_topk[0, k].item() / length_norm_factor
                    if idx == self.eos:
                        if ignore_eos:
                            beam['aws'][-1] = eouts_c.new_zeros(eouts_c.size(0), 1, 1, eouts_c.size(1))
                            beam['no_boundary'] = True
                            new_hyps.append(beam.copy())
                            continue
                        max_score_no_eos = scores_att[(j), :idx].max(0)[0].item()
                        max_score_no_eos = max(max_score_no_eos, scores_att[(j), idx + 1:].max(0)[0].item())
                        if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                            continue
                    new_hyps.append({'hyp': beam['hyp'] + [idx], 'score': total_score, 'score_att': total_scores_att[0, idx].item(), 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[k].item(), 'dstates': {'dstate': (dstates['dstate'][0][:, j:j + 1], dstates['dstate'][1][:, j:j + 1])}, 'cv': cv[j:j + 1], 'aws': beam['aws'] + [aw[j:j + 1]], 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None, 'ctc_state': new_ctc_states[k] if self.ctc_prefix_scorer is not None else None, 'no_boundary': no_boundary})
            new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)
            hyps_nobd += [hyp for hyp in new_hyps_sorted[beam_width:] if hyp['no_boundary']]
            new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps_sorted[:beam_width], end_hyps)
            hyps = new_hyps[:]
            if is_finish:
                break
        hyps_nobd_sorted = sorted(hyps_nobd, key=lambda x: x['score'], reverse=True)
        hyps = (hyps[:] + hyps_nobd_sorted)[:beam_width]
        if len(end_hyps) > 0:
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)
        merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
        if idx2token is not None:
            logger.info('=' * 200)
            for k in range(len(merged_hyps)):
                logger.info('Hyp: %s' % idx2token(merged_hyps[k]['hyp'][1:]))
                logger.info('log prob (hyp): %.7f' % merged_hyps[k]['score'])
                logger.info('log prob (hyp, att): %.7f' % (merged_hyps[k]['score_att'] * (1 - ctc_weight)))
                if self.ctc_prefix_scorer is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (merged_hyps[k]['score_ctc'] * ctc_weight))
                if lm is not None:
                    logger.info('log prob (hyp, first-path lm): %.7f' % (merged_hyps[k]['score_lm'] * lm_weight))
                logger.info('-' * 50)
        aws = None
        if len(end_hyps) > 0:
            self.dstates_final = end_hyps[0]['dstates']
            self.lmstate_final = end_hyps[0]['lmstate']
        self.n_frames += eouts_c.size(1)
        return end_hyps, hyps, aws


class RNNTransducer(DecoderBase):
    """RNN transducer.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int):
        rnn_type (str): lstm_transducer or gru_transducer
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probability for the RNN layer
        dropout_emb (float): dropout probability for the embedding layer
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        external_lm (RNNLM):
        global_weight (float):
        mtl_per_batch (bool):
        param_init (str): parameter initialization method

    """

    def __init__(self, special_symbols, enc_n_units, rnn_type, n_units, n_projs, n_layers, bottleneck_dim, emb_dim, vocab, dropout=0.0, dropout_emb=0.0, lsm_prob=0.0, ctc_weight=0.0, ctc_lsm_prob=0.0, ctc_fc_list=[], external_lm=None, global_weight=1.0, mtl_per_batch=False, param_init=0.1):
        super(RNNTransducer, self).__init__()
        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm_transducer', 'gru_transducer']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch
        self.prev_spk = ''
        self.lmstate_final = None
        self.state_cache = OrderedDict()
        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos, blank=self.blank, enc_n_units=enc_n_units, vocab=vocab, dropout=dropout, lsm_prob=ctc_lsm_prob, fc_list=ctc_fc_list, param_init=0.1)
        if ctc_weight < global_weight:
            rnn_l = nn.LSTM if rnn_type == 'lstm_transducer' else nn.GRU
            self.rnn = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout)
            if n_projs > 0:
                self.proj = repeat(nn.Linear(n_units, n_projs), n_layers)
            dec_idim = emb_dim
            for l in range(n_layers):
                self.rnn += [rnn_l(dec_idim, n_units, 1, batch_first=True)]
                dec_idim = n_projs if n_projs > 0 else n_units
            self.embed = nn.Embedding(vocab, emb_dim, padding_idx=self.pad)
            self.dropout_emb = nn.Dropout(p=dropout_emb)
            self.w_enc = nn.Linear(enc_n_units, bottleneck_dim)
            self.w_dec = nn.Linear(dec_idim, bottleneck_dim, bias=False)
            self.output = nn.Linear(bottleneck_dim, vocab)
        self.reset_parameters(param_init)
        if external_lm is not None:
            assert external_lm.vocab == vocab
            assert external_lm.n_units == n_units
            assert external_lm.n_projs == n_projs
            assert external_lm.n_layers == n_layers
            param_dict = dict(external_lm.named_parameters())
            for n, p in self.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if 'output' in n:
                        continue
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)

    def start_scheduled_sampling(self):
        self._ss_prob = 0.0

    def forward(self, eouts, elens, ys, task='all', teacher_logits=None, recog_params={}, idx2token=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_transducer': None, 'loss_ctc': None, 'loss_mbr': None}
        loss = eouts.new_zeros((1,))
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc, _ = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_transducer = self.forward_transducer(eouts, elens, ys)
            observation['loss_transducer'] = loss_transducer.item()
            if self.mtl_per_batch:
                loss += loss_transducer
            else:
                loss += loss_transducer * (self.global_weight - self.ctc_weight)
        observation['loss'] = loss.item()
        return loss, observation

    def forward_transducer(self, eouts, elens, ys):
        """Compute RNN-T loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`

        """
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        _ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id) for y in ys]
        ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in _ys], self.pad)
        ys_out = pad_list(_ys, self.blank)
        ys_emb = self.dropout_emb(self.embed(ys_in))
        dout, _ = self.recurrency(ys_emb, None)
        logits = self.joint(eouts, dout)
        log_probs = torch.log_softmax(logits, dim=-1)
        assert log_probs.size(2) == ys_out.size(1) + 1
        if self.device_id >= 0:
            ys_out = ys_out
            elens = elens
            ylens = ylens
            loss = warp_rnnt.rnnt_loss(log_probs, ys_out.int(), elens, ylens, average_frames=False, reduction='mean', gather=False)
        else:
            self.warprnnt_loss = warprnnt_pytorch.RNNTLoss()
            loss = self.warprnnt_loss(log_probs, ys_out.int(), elens, ylens)
        return loss

    def joint(self, eouts, douts):
        """Combine encoder outputs and prediction network outputs.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            douts (FloatTensor): `[B, L, dec_n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`

        """
        eouts = eouts.unsqueeze(2)
        douts = douts.unsqueeze(1)
        out = torch.tanh(self.w_enc(eouts) + self.w_dec(douts))
        out = self.output(out)
        return out

    def recurrency(self, ys_emb, dstate):
        """Update prediction network.

        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`
        Returns:
            dout (FloatTensor): `[B, L, emb_dim]`
            new_dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        if dstate is None:
            dstate = self.zero_state(ys_emb.size(0))
        new_dstate = {'hxs': None, 'cxs': None}
        new_hxs, new_cxs = [], []
        for l in range(self.n_layers):
            if self.rnn_type == 'lstm_transducer':
                ys_emb, (h, c) = self.rnn[l](ys_emb, hx=(dstate['hxs'][l:l + 1], dstate['cxs'][l:l + 1]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru_transducer':
                ys_emb, h = self.rnn[l](ys_emb, hx=dstate['hxs'][l:l + 1])
            new_hxs.append(h)
            ys_emb = self.dropout(ys_emb)
            if self.n_projs > 0:
                ys_emb = torch.tanh(self.proj[l](ys_emb))
        new_dstate['hxs'] = torch.cat(new_hxs, dim=0)
        if self.rnn_type == 'lstm_transducer':
            new_dstate['cxs'] = torch.cat(new_cxs, dim=0)
        return ys_emb, new_dstate

    def zero_state(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int): batch size
        Returns:
            zero_state (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        w = next(self.parameters())
        zero_state = {'hxs': None, 'cxs': None}
        zero_state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
        if self.rnn_type == 'lstm_transducer':
            zero_state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
        return zero_state

    def greedy(self, eouts, elens, max_len_ratio, idx2token, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            hyps (list): length `B`, each of which contains arrays of size `[L]`
            aw: dummy

        """
        bs = eouts.size(0)
        hyps = []
        for b in range(bs):
            hyp_b = []
            y = eouts.new_zeros(1, 1).fill_(self.eos).long()
            y_emb = self.dropout_emb(self.embed(y))
            dout, dstate = self.recurrency(y_emb, None)
            for t in range(elens[b]):
                out = self.joint(eouts[b:b + 1, t:t + 1], dout)
                y = out.squeeze(2).argmax(-1)
                idx = y[0].item()
                if idx != self.blank:
                    hyp_b += [idx]
                    y_emb = self.dropout_emb(self.embed(y))
                    dout, dstate = self.recurrency(y_emb, dstate)
            hyps += [hyp_b]
        for b in range(bs):
            if utt_ids is not None:
                logger.debug('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.debug('Ref: %s' % idx2token(refs_id[b]))
            logger.debug('Hyp: %s' % idx2token(hyps[b]))
        return hyps, None

    def beam_search(self, eouts, elens, params, idx2token=None, lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None, nbest=1, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
            idx2token (): converter from index to token
            lm: firsh path LM
            lm_second: second path LM
            lm_second_bwd: secoding path backward LM
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
        Returns:
            nbest_hyps_idx (list): length `B`, each of which contains list of N hypotheses
            aws: dummy
            scores: dummy

        """
        bs = eouts.size(0)
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        asr_state_carry_over = params['recog_asr_state_carry_over']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        if lm_second is not None:
            assert lm_weight_second > 0
            lm_second.eval()
        if lm_second_bwd is not None:
            assert lm_weight_second_bwd > 0
            lm_second_bwd.eval()
        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)
        nbest_hyps_idx = []
        eos_flags = []
        for b in range(bs):
            y = eouts.new_zeros(bs, 1).fill_(self.eos).long()
            y_emb = self.dropout_emb(self.embed(y))
            dout, dstate = self.recurrency(y_emb, None)
            lmstate = None
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)
            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_carry_over and isinstance(lm, RNNLM):
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]
            helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id)
            end_hyps = []
            hyps = [{'hyp': [self.eos], 'ref_id': [self.eos], 'score': 0.0, 'score_rnnt': 0.0, 'score_lm': 0.0, 'score_ctc': 0.0, 'dout': dout, 'dstate': dstate, 'lmstate': lmstate, 'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None}]
            for t in range(elens[b]):
                douts = torch.cat([beam['dout'] for beam in hyps], dim=0)
                outs = self.joint(eouts[b:b + 1, t:t + 1].repeat([douts.size(0), 1, 1]), douts)
                scores_rnnt = torch.log_softmax(outs.squeeze(2).squeeze(1), dim=-1)
                y = eouts.new_zeros(len(hyps), 1).long()
                for j, beam in enumerate(hyps):
                    y[j, 0] = beam['hyp'][-1]
                lmstate, scores_lm = None, None
                if lm is not None:
                    if hyps[0]['lmstate'] is not None:
                        lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                        lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                        lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
                    lmout, lmstate, scores_lm = lm.predict(y, lmstate)
                new_hyps = []
                for j, beam in enumerate(hyps):
                    dout = douts[j:j + 1]
                    dstate = beam['dstate']
                    lmstate = beam['lmstate']
                    total_scores_rnnt = beam['score_rnnt'] + scores_rnnt[j:j + 1]
                    total_scores = total_scores_rnnt * (1 - ctc_weight)
                    total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=-1, largest=True, sorted=True)
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(beam_width)
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(beam['hyp'], topk_ids, beam['ctc_state'], total_scores_topk, ctc_prefix_scorer)
                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        if idx == self.blank:
                            beam['score'] = total_scores_topk[0, k].item()
                            beam['score_rnnt'] = total_scores_topk[0, k].item()
                            new_hyps.append(beam.copy())
                            continue
                        hyp_id = beam['hyp'] + [idx]
                        hyp_str = ' '.join(list(map(str, hyp_id)))
                        y = eouts.new_zeros(1, 1).fill_(idx).long()
                        y_emb = self.dropout_emb(self.embed(y))
                        dout, new_dstate = self.recurrency(y_emb, dstate)
                        self.state_cache[hyp_str] = {'dout': dout, 'dstate': new_dstate, 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None}
                        new_hyps.append({'hyp': hyp_id, 'score': total_scores_topk[0, k].item(), 'score_rnnt': total_scores_rnnt[0, idx].item(), 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[k].item(), 'dout': dout, 'dstate': new_dstate, 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None, 'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None})
                new_hyps_merged = {}
                for beam in new_hyps:
                    hyp_str = ' '.join(list(map(str, beam['hyp'])))
                    if hyp_str not in new_hyps_merged.keys():
                        new_hyps_merged[hyp_str] = beam
                    elif hyp_str in new_hyps_merged.keys():
                        if beam['score'] > new_hyps_merged[hyp_str]['score']:
                            new_hyps_merged[hyp_str] = beam
                new_hyps = [v for v in new_hyps_merged.values()]
                new_hyps_tmp = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
                new_hyps = []
                for hyp in new_hyps_tmp:
                    new_hyps += [hyp]
                if len(end_hyps) >= beam_width:
                    end_hyps = end_hyps[:beam_width]
                    break
                hyps = new_hyps[:]
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])
            if lm_second is not None:
                self.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')
            if lm_second_bwd is not None:
                self.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_rev')
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)
            self.state_cache = OrderedDict()
            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None and self.vocab == idx2token.vocab:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    if ctc_log_probs is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % end_hyps[k]['score_ctc'])
                    if lm is not None:
                        logger.info('log prob (hyp, lm): %.7f' % end_hyps[k]['score_lm'])
                    logger.info('-' * 50)
            nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])
        return nbest_hyps_idx, None, None


class TransformerDecoder(DecoderBase):
    """Transformer decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of the encoder outputs
        attn_type (str): type of attention mechanism
        n_heads (int): number of attention heads
        n_layers (int): number of self-attention layers
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of the embedding and output layers
        dropout (float): dropout probability for linear layers
        dropout_emb (float): dropout probability for the embedding layer
        dropout_att (float): dropout probability for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        dropout_head (float): HeadDrop probability for attention heads
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        global_weight (float):
        mtl_per_batch (bool):
        param_init (str): parameter initialization method
        memory_transformer (bool): TransformerXL decoder
        mem_len (int):
        mocha_chunk_size (int):
        mocha_n_heads_mono (int):
        mocha_n_heads_chunk (int):
        mocha_init_r (int):
        mocha_eps (float):
        mocha_std (float):
        mocha_no_denominator (bool):
        mocha_1dconv (bool): 1dconv for MoChA
        mocha_quantity_loss_weight (float):
        mocha_head_divergence_loss_weight (float):
        latency_metric (str):
        latency_loss_weight (float):
        mocha_first_layer (int):
        external_lm (RNNLM):
        lm_fusion (str):

    """

    def __init__(self, special_symbols, enc_n_units, attn_type, n_heads, n_layers, d_model, d_ff, pe_type, layer_norm_eps, ffn_activation, vocab, tie_embedding, dropout, dropout_emb, dropout_att, dropout_layer, dropout_head, lsm_prob, ctc_weight, ctc_lsm_prob, ctc_fc_list, backward, global_weight, mtl_per_batch, param_init, memory_transformer, mem_len, mocha_chunk_size, mocha_n_heads_mono, mocha_n_heads_chunk, mocha_init_r, mocha_eps, mocha_std, mocha_no_denominator, mocha_1dconv, mocha_quantity_loss_weight, mocha_head_divergence_loss_weight, latency_metric, latency_loss_weight, mocha_first_layer, external_lm, lm_fusion):
        super(TransformerDecoder, self).__init__()
        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch
        self.prev_spk = ''
        self.lmstate_final = None
        self.memory_transformer = memory_transformer
        self.mem_len = mem_len
        if memory_transformer:
            assert pe_type == 'none'
        self.aws_dict = {}
        self.data_dict = {}
        self.attn_type = attn_type
        self.quantity_loss_weight = mocha_quantity_loss_weight
        self._quantity_loss_weight = 0
        self.headdiv_loss_weight = mocha_head_divergence_loss_weight
        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        self.mocha_first_layer = mocha_first_layer
        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos, blank=self.blank, enc_n_units=enc_n_units, vocab=self.vocab, dropout=dropout, lsm_prob=ctc_lsm_prob, fc_list=ctc_fc_list, param_init=0.1, backward=backward)
        if ctc_weight < global_weight:
            self.embed = nn.Embedding(self.vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type, param_init)
            self.u = None
            self.v = None
            if memory_transformer:
                self.scale = math.sqrt(d_model)
                self.dropout_emb = nn.Dropout(p=dropout_emb)
                self.pos_emb = XLPositionalEmbedding(d_model, dropout_emb)
                if self.mem_len > 0:
                    self.u = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
                    self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
            self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, src_tgt_attention=False if lth < mocha_first_layer - 1 else True, memory_transformer=memory_transformer, mocha_chunk_size=mocha_chunk_size, mocha_n_heads_mono=mocha_n_heads_mono, mocha_n_heads_chunk=mocha_n_heads_chunk, mocha_init_r=mocha_init_r, mocha_eps=mocha_eps, mocha_std=mocha_std, mocha_no_denominator=mocha_no_denominator, mocha_1dconv=mocha_1dconv, dropout_head=dropout_head, lm_fusion=lm_fusion)) for lth in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, self.vocab)
            if tie_embedding:
                self.output.weight = self.embed.weight
            self.lm = external_lm
            if external_lm is not None:
                self.lm_output_proj = nn.Linear(external_lm.output_dim, d_model)
            self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if self.memory_transformer:
            logger.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
            for n, p in self.named_parameters():
                if 'conv' in n:
                    continue
                init_with_normal_dist(n, p, std=0.02)
        elif param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
            nn.init.constant_(self.embed.weight[self.pad], 0.0)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.constant_(self.output.bias, 0.0)

    def init_memory(self):
        """Initialize memory."""
        if self.device_id >= 0:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]
        else:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): List of `[B, L_prev, d_model]`
            hidden_states (list): List of `[B, L, d_model]`
        Returns:
            new_mems (list): List of `[B, mlen, d_model]`

        """
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)
                new_mems.append(cat[:, start_idx:end_idx].detach())
        return new_mems

    def forward(self, eouts, elens, ys, task='all', teacher_logits=None, recog_params={}, idx2token=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None, 'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros((1,))
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc, trigger_points = self.ctc(eouts, elens, ys, forced_align=self.latency_metric and self.training)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
        else:
            trigger_points = None
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_att, acc_att, ppl_att, losses_auxiliary = self.forward_att(eouts, elens, ys, trigger_points=trigger_points)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if 'mocha' in self.attn_type:
                if self._quantity_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_quantity'] * self._quantity_loss_weight
                observation['loss_quantity'] = losses_auxiliary['loss_quantity'].item()
            if self.headdiv_loss_weight > 0:
                loss_att += losses_auxiliary['loss_headdiv'] * self.headdiv_loss_weight
                observation['loss_headdiv'] = losses_auxiliary['loss_headdiv'].item()
            if self.latency_metric:
                observation['loss_latency'] = losses_auxiliary['loss_latency'].item() if self.training else 0
                if self.latency_metric != 'decot' and self.latency_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_latency'] * self.latency_loss_weight
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)
        observation['loss'] = loss.item()
        return loss, observation

    def forward_att(self, eouts, elens, ys, return_logits=False, teacher_logits=None, trigger_points=None):
        """Compute XE loss for the Transformer decoder.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
            trigger_points (IntTensor): `[B, T]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            loss_quantity (FloatTensor): `[1]`
            loss_headdiv (FloatTensor): `[1]`
            loss_latency (FloatTensor): `[1]`

        """
        ys_in, ys_out, ylens = append_sos_eos(eouts, ys, self.eos, self.eos, self.pad, self.bwd)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
        xtime = eouts.size(1)
        bs, ymax = ys_in.size()[:2]
        mlen = 0
        tgt_mask = (ys_out != self.pad).unsqueeze(1).repeat([1, ymax, 1])
        causal_mask = tgt_mask.new_ones(ymax, ymax).byte()
        causal_mask = torch.tril(causal_mask, diagonal=0 + mlen, out=causal_mask).unsqueeze(0)
        tgt_mask = tgt_mask & causal_mask
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1).repeat([1, ymax, 1])
        lmout = None
        if self.lm is not None:
            self.lm.eval()
            with torch.no_grad():
                lmout, lmstate, _ = self.lm.predict(ys_in, None)
            lmout = self.lm_output_proj(lmout)
        out = self.pos_enc(self.embed(ys_in))
        mems = self.init_memory()
        pos_embs = None
        if self.memory_transformer:
            out = self.dropout_emb(out)
            pos_idxs = torch.arange(mlen - 1, -ymax - 1, -1.0, dtype=torch.float)
            pos_embs = self.pos_emb(pos_idxs, self.device_id)
        hidden_states = [out]
        xy_aws_layers = []
        for lth, (mem, layer) in enumerate(zip(mems, self.layers)):
            out, yy_aws, xy_aws, xy_aws_beta, yy_aws_lm = layer(out, tgt_mask, eouts, src_mask, mode='parallel', lmout=lmout, pos_embs=pos_embs, memory=mem, u=self.u, v=self.v)
            if lth < self.n_layers - 1:
                hidden_states.append(out)
            xy_aws_layers.append(xy_aws.clone() if xy_aws is not None else out.new_zeros(bs, yy_aws.size(1), ymax, xtime))
            if not self.training:
                if yy_aws is not None:
                    self.aws_dict['yy_aws_layer%d' % lth] = tensor2np(yy_aws)
                if xy_aws is not None:
                    self.aws_dict['xy_aws_layer%d' % lth] = tensor2np(xy_aws)
                if xy_aws_beta is not None:
                    self.aws_dict['xy_aws_beta_layer%d' % lth] = tensor2np(xy_aws_beta)
                if yy_aws_lm is not None:
                    self.aws_dict['yy_aws_lm_layer%d' % lth] = tensor2np(yy_aws_lm)
        logits = self.output(self.norm_out(out))
        if return_logits:
            return logits
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)
        losses_auxiliary = {}
        if self._quantity_loss_weight > 0 or self.headdiv_loss_weight > 0 or self.latency_loss_weight > 0:
            for lth in range(self.mocha_first_layer - 1, self.n_layers):
                n_heads = xy_aws_layers[lth].size(1)
                xy_aws_layers[lth] = xy_aws_layers[lth].masked_fill_(src_mask.unsqueeze(1).repeat([1, n_heads, 1, 1]) == 0, 0)
                xy_aws_layers[lth] = xy_aws_layers[lth].masked_fill_(tgt_mask[:, :, -1:].unsqueeze(1).repeat([1, n_heads, 1, xtime]) == 0, 0)
        n_heads = xy_aws_layers[-1].size(1)
        losses_auxiliary['loss_quantity'] = 0.0
        if 'mocha' in self.attn_type:
            n_tokens_ref = tgt_mask[:, (-1), :].sum(1).float()
            n_tokens_pred = sum([torch.abs(aws.sum(3).sum(2).sum(1) / aws.size(1)) for aws in xy_aws_layers[self.mocha_first_layer - 1:]])
            n_tokens_pred /= self.n_layers - self.mocha_first_layer + 1
            losses_auxiliary['loss_quantity'] = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))
        losses_auxiliary['loss_headdiv'] = 0.0
        if self.headdiv_loss_weight > 0.0:
            js = torch.arange(xtime, dtype=torch.float)
            js = js.repeat([bs, n_heads, ymax, 1])
            avg_head_pos = sum([(js * aws).sum(3).sum(1) for aws in xy_aws_layers]) / (n_heads * self.n_layers)
            loss_headdiv = sum([(((js * aws).sum(3).sum(1) - avg_head_pos) ** 2) for aws in xy_aws_layers]) / (n_heads * self.n_layers)
            losses_auxiliary['loss_headdiv'] = loss_headdiv.sum() / ylens.sum()
        losses_auxiliary['loss_latency'] = 0.0
        if self.latency_metric == 'interval':
            raise NotImplementedError
        elif trigger_points is not None:
            assert self.latency_loss_weight > 0
            js = torch.arange(xtime, dtype=torch.float)
            js = js.repeat([bs, n_heads, ymax, 1])
            weighted_avg_head_pos = torch.cat([(js * aws).sum(3) for aws in xy_aws_layers], dim=1)
            weighted_avg_head_pos *= torch.softmax(weighted_avg_head_pos.clone(), dim=1)
            trigger_points = trigger_points.float()
            trigger_points = trigger_points.unsqueeze(1)
            if self.latency_metric == 'ctc_sync':
                loss_latency = torch.abs(weighted_avg_head_pos - trigger_points)
            else:
                raise NotImplementedError(self.latency_metric)
            losses_auxiliary['loss_latency'] = loss_latency.sum() / ylens.sum()
        acc = compute_accuracy(logits, ys_out, self.pad)
        return loss, acc, ppl, losses_auxiliary

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        _save_path = mkdir_join(save_path, 'dec_att_weights')
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)
        for k, aw in self.aws_dict.items():
            elens = self.data_dict['elens']
            ylens = self.data_dict['ylens']
            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols * max(1, n_heads // 4)
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp, figsize=(20 * max(1, n_heads // 4), 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                if 'xy' in k:
                    ax.imshow(aw[(-1), (h), :ylens[-1], :elens[-1]], aspect='auto')
                else:
                    ax.imshow(aw[(-1), (h), :ylens[-1], :ylens[-1]], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()

    def greedy(self, eouts, elens, max_len_ratio, idx2token, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, cache_states=True):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            cache_states (bool):
        Returns:
            hyps (list): length `B`, each of which contains arrays of size `[L]`
            aw (list): length `B`, each of which contains arrays of size `[L, T]`

        """
        bs, xtime = eouts.size()[:2]
        ys = eouts.new_zeros(bs, 1).fill_(self.eos).long()
        cache = [None] * self.n_layers
        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        ymax = int(math.floor(xtime * max_len_ratio)) + 1
        for t in range(ymax):
            causal_mask = eouts.new_ones(t + 1, t + 1).byte()
            causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)
            new_cache = [None] * self.n_layers
            out = self.pos_enc(self.embed(ys))
            for lth, layer in enumerate(self.layers):
                out, _, xy_aws, _, _ = layer(out, causal_mask, eouts, None, cache=cache[lth])
                new_cache[lth] = out
            if cache_states:
                cache = new_cache[:]
            y = self.output(self.norm_out(out))[:, -1:].argmax(-1)
            hyps_batch += [y]
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1
            if sum(eos_flags) == bs:
                break
            if t == ymax - 1:
                break
            ys = torch.cat([ys, y], dim=-1)
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        xy_aws = tensor2np(xy_aws.transpose(1, 2).transpose(2, 3))
        if self.bwd:
            hyps = [hyps_batch[(b), :ylens[b]][::-1] for b in range(bs)]
            aws = [xy_aws[(b), :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[(b), :ylens[b]] for b in range(bs)]
            aws = [xy_aws[(b), :, :ylens[b]] for b in range(bs)]
        if exclude_eos:
            if self.bwd:
                hyps = [(hyps[b][1:] if eos_flags[b] else hyps[b]) for b in range(bs)]
            else:
                hyps = [(hyps[b][:-1] if eos_flags[b] else hyps[b]) for b in range(bs)]
        for b in range(bs):
            if utt_ids is not None:
                logger.debug('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.debug('Ref: %s' % idx2token(refs_id[b]))
            if self.bwd:
                logger.debug('Hyp: %s' % idx2token(hyps[b][::-1]))
            else:
                logger.debug('Hyp: %s' % idx2token(hyps[b]))
        return hyps, aws

    def beam_search(self, eouts, elens, params, idx2token=None, lm=None, lm_second=None, lm_bwd=None, ctc_log_probs=None, nbest=1, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[], cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
            idx2token (): converter from index to token
            lm: firsh path LM
            lm_second: second path LM
            lm_bwd: first/secoding path backward LM
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
            cache_states (bool): cache decoder states for fast decoding
        Returns:
            nbest_hyps_idx (list): length `B`, each of which contains list of N hypotheses
            aws (list): length `B`, each of which contains arrays of size `[H, L, T]`
            scores (list):

        """
        bs, xmax, _ = eouts.size()
        n_models = len(ensmbl_decs) + 1
        beam_width = params['recog_beam_width']
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_bwd = params['recog_lm_bwd_weight']
        eos_threshold = params['recog_eos_threshold']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        softmax_smoothing = params['recog_softmax_smoothing']
        eps_wait = params['recog_mma_delay_threshold']
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        if lm_second is not None:
            assert lm_weight_second > 0
            lm_second.eval()
        if lm_bwd is not None:
            assert lm_weight_bwd > 0
            lm_bwd.eval()
        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)
        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            lmstate = None
            ys = eouts.new_zeros(1, 1).fill_(self.eos).long()
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)
            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_carry_over and isinstance(lm, RNNLM):
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]
            helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id)
            end_hyps = []
            ymax = int(math.floor(elens[b] * max_len_ratio)) + 1
            hyps = [{'hyp': [self.eos], 'ys': ys, 'cache': None, 'score': 0.0, 'score_attn': 0.0, 'score_ctc': 0.0, 'score_lm': 0.0, 'aws': [None], 'lmstate': lmstate, 'ensmbl_aws': [[None]] * (n_models - 1), 'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None, 'streamable': True, 'streaming_failed_point': 1000}]
            streamable_global = True
            for t in range(ymax):
                cache = [None] * self.n_layers
                if cache_states and t > 0:
                    for lth in range(self.n_layers):
                        cache[lth] = torch.cat([beam['cache'][lth] for beam in hyps], dim=0)
                ys = eouts.new_zeros(len(hyps), t + 1).long()
                for j, beam in enumerate(hyps):
                    ys[(j), :] = beam['ys']
                if t > 0:
                    xy_aws_prev = torch.cat([beam['aws'][-1] for beam in hyps], dim=0)
                else:
                    xy_aws_prev = None
                lmstate, scores_lm = None, None
                if lm is not None:
                    if hyps[0]['lmstate'] is not None:
                        lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                        lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                        lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
                    y = ys[:, -1:].clone()
                    _, lmstate, scores_lm = lm.predict(y, lmstate)
                causal_mask = eouts.new_ones(t + 1, t + 1).byte()
                causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0).repeat([ys.size(0), 1, 1])
                out = self.pos_enc(self.embed(ys))
                mlen = 0
                if self.memory_transformer:
                    mems = self.init_memory()
                    pos_idxs = torch.arange(mlen - 1, -(t + 1) - 1, -1.0, dtype=torch.float)
                    pos_embs = self.pos_emb(pos_idxs, self.device_id)
                    out = self.dropout_emb(out)
                    hidden_states = [out]
                n_heads_total = 0
                eouts_b = eouts[b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                new_cache = [None] * self.n_layers
                xy_aws_all_layers = []
                xy_aws = None
                boundary_rightmost = None
                lth_s = self.mocha_first_layer - 1
                for lth, layer in enumerate(self.layers):
                    if self.memory_transformer:
                        out, _, xy_aws, _, _ = layer(out, causal_mask, eouts_b, None, cache=cache[lth], pos_embs=pos_embs, memory=mems[lth], u=self.u, v=self.v)
                        hidden_states.append(out)
                    else:
                        out, _, xy_aws, _, _ = layer(out, causal_mask, eouts_b, None, cache=cache[lth], xy_aws_prev=xy_aws_prev[:, (lth - lth_s)] if lth >= lth_s and t > 0 else None, boundary_rightmost=boundary_rightmost, eps_wait=eps_wait)
                        if 'mocha' in self.attn_type:
                            if xy_aws is not None and xy_aws[b].sum() != 0:
                                boundary_rightmost_lth = xy_aws[(b), :, (0)].nonzero()[:, (-1)].max().item()
                                if boundary_rightmost is None:
                                    boundary_rightmost = boundary_rightmost_lth
                                else:
                                    boundary_rightmost = max(boundary_rightmost_lth, boundary_rightmost)
                        if lth >= lth_s:
                            n_heads_total += xy_aws.size(1)
                    new_cache[lth] = out
                    if xy_aws is not None:
                        xy_aws_all_layers.append(xy_aws)
                logits = self.output(self.norm_out(out))
                probs = torch.softmax(logits[:, (-1)] * softmax_smoothing, dim=1)
                xy_aws_all_layers = torch.stack(xy_aws_all_layers, dim=1)
                ensmbl_new_cache = []
                if n_models > 1:
                    for i_e, dec in enumerate(ensmbl_decs):
                        out_e = dec.pos_enc(dec.embed(ys))
                        eouts_e = ensmbl_eouts[i_e][b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                        new_cache_e = [None] * dec.n_layers
                        for l in range(dec.n_layers):
                            out_e, _, xy_aws_e, _, _ = dec.layers[l](out_e, causal_mask, eouts_e, None, cache=cache[lth])
                            new_cache_e[l] = out_e
                        ensmbl_new_cache.append(new_cache_e)
                        logits_e = dec.output(dec.norm_out(out_e))
                        probs += torch.softmax(logits_e[:, (-1)] * softmax_smoothing, dim=1)
                scores_attn = torch.log(probs) / n_models
                new_hyps = []
                for j, beam in enumerate(hyps):
                    total_scores_attn = beam['score_attn'] + scores_attn[j:j + 1]
                    total_scores = total_scores_attn * (1 - ctc_weight)
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j:j + 1, (-1)]
                        total_scores += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(1, self.vocab)
                    total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if lp_weight > 0:
                        total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(beam['hyp'], topk_ids, beam['ctc_state'], total_scores_topk, ctc_prefix_scorer)
                    new_aws = beam['aws'] + [xy_aws_all_layers[j:j + 1, :, :, -1:]]
                    aws_j = torch.cat(new_aws[1:], dim=3)
                    streaming_failed_point = beam['streaming_failed_point']
                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                        total_scores_topk /= length_norm_factor
                        if idx == self.eos:
                            if len(beam['hyp']) - 1 < elens[b] * min_len_ratio:
                                continue
                            max_score_no_eos = scores_attn[(j), :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_attn[(j), idx + 1:].max(0)[0].item())
                            if scores_attn[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue
                        quantity_rate = 1.0
                        if 'mocha' in self.attn_type:
                            n_tokens_hyp_k = t + 1
                            n_quantity_k = aws_j[:, :, :, :n_tokens_hyp_k].int().sum().item()
                            quantity_diff = n_tokens_hyp_k * n_heads_total - n_quantity_k
                            if quantity_diff != 0:
                                if idx == self.eos:
                                    n_tokens_hyp_k -= 1
                                    n_quantity_k = aws_j[:, :, :, :n_tokens_hyp_k].int().sum().item()
                                else:
                                    streamable_global = False
                                if n_tokens_hyp_k * n_heads_total == 0:
                                    quantity_rate = 0
                                else:
                                    quantity_rate = n_quantity_k / (n_tokens_hyp_k * n_heads_total)
                            if beam['streamable'] and not streamable_global:
                                streaming_failed_point = t
                        new_hyps.append({'hyp': beam['hyp'] + [idx], 'ys': torch.cat([beam['ys'], eouts.new_zeros(1, 1).fill_(idx).long()], dim=-1), 'cache': [new_cache_l[j:j + 1] for new_cache_l in new_cache] if cache_states else cache, 'score': total_scores_topk[0, k].item(), 'score_attn': total_scores_attn[0, idx].item(), 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[0, idx].item(), 'aws': new_aws, 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None, 'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None, 'ensmbl_cache': ensmbl_new_cache, 'streamable': streamable_global, 'streaming_failed_point': streaming_failed_point, 'quantity_rate': quantity_rate})
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps_sorted, end_hyps, prune=True)
                hyps = new_hyps[:]
                if is_finish:
                    break
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])
            if lm_second is not None:
                self.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')
            if lm_bwd is not None and lm_weight_bwd > 0:
                self.lm_rescoring(end_hyps, lm_bwd, lm_weight_bwd, tag='second_bwd')
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)
            for j in range(len(end_hyps[0]['aws'][1:])):
                tmp = end_hyps[0]['aws'][j + 1]
                end_hyps[0]['aws'][j + 1] = tmp.view(1, -1, tmp.size(-2), tmp.size(-1))
            self.streamable = end_hyps[0]['streamable']
            self.quantity_rate = end_hyps[0]['quantity_rate']
            self.last_success_frame_ratio = None
            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:][::-1] if self.bwd else end_hyps[k]['hyp'][1:]))
                    logger.info('num tokens (hyp): %d' % len(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_attn'] * (1 - ctc_weight)))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' % (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_bwd is not None:
                        logger.info('log prob (hyp, second-path lm-bwd): %.7f' % (end_hyps[k]['score_lm_second_bwd'] * lm_weight_bwd))
                    if 'mocha' in self.attn_type:
                        logger.info('streamable: %s' % end_hyps[k]['streamable'])
                        logger.info('streaming failed point: %d' % (end_hyps[k]['streaming_failed_point'] + 1))
                        logger.info('quantity rate [%%]: %.2f' % (end_hyps[k]['quantity_rate'] * 100))
                    logger.info('-' * 50)
                if 'mocha' in self.attn_type and end_hyps[0]['streaming_failed_point'] < 1000:
                    assert not self.streamable
                    aws_last_success = end_hyps[0]['aws'][1:][end_hyps[0]['streaming_failed_point'] - 1]
                    rightmost_frame = max(0, aws_last_success[(0), :, (0)].nonzero()[:, (-1)].max().item()) + 1
                    frame_ratio = rightmost_frame * 100 / xmax
                    self.last_success_frame_ratio = frame_ratio
                    logger.info('streaming last success frame ratio: %.2f' % frame_ratio)
            if self.bwd:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [tensor2np(torch.cat(end_hyps[0]['aws'][1:][::-1], dim=2).squeeze(0))]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [tensor2np(torch.cat(end_hyps[0]['aws'][1:], dim=2).squeeze(0))]
            scores += [[end_hyps[n]['score_attn'] for n in range(nbest)]]
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][1:] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][:-1] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
        if len(end_hyps) > 0:
            self.lmstate_final = end_hyps[0]['lmstate']
        return nbest_hyps_idx, aws, scores


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


class EncoderBase(ModelBase):
    """Base class for encoders."""

    def __init__(self):
        super(ModelBase, self).__init__()
        logger.info('Overriding EncoderBase class.')

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def forward(self, xs, xlens, task):
        raise NotImplementedError

    def turn_off_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = False
                    logging.debug('Turn off ceil_mode in %s.' % name)
                else:
                    self.turn_off_ceil_mode(module)


def parse_config(conv_channels, conv_kernel_sizes, conv_strides, conv_poolings):
    channels, kernel_sizes, strides, poolings = [], [], [], []
    if len(conv_channels) > 0:
        channels = [int(c) for c in conv_channels.split('_')]
    if len(conv_kernel_sizes) > 0:
        kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))] for c in conv_kernel_sizes.split('_')]
    if len(conv_strides) > 0:
        strides = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))] for c in conv_strides.split('_')]
    if len(conv_poolings) > 0:
        poolings = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))] for c in conv_poolings.split('_')]
    return channels, kernel_sizes, strides, poolings


class GatedConvEncoder(EncoderBase):
    """Gated convolutional neural netwrok encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channles in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        strides (list): strides in TDS layers
        poolings (list) size of poolings in TDS layers
        dropout (float) probability to drop nodes in hidden-hidden connection
        batch_norm (bool): if True, apply batch normalization
        bottleneck_dim (int): dimension of the bottleneck layer after the last layer
        param_init (float):

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes, dropout, bottleneck_dim=0, param_init=0.1):
        super(GatedConvEncoder, self).__init__()
        channels, kernel_sizes, _, _ = parse_config(channels, kernel_sizes, '', '')
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        layers = OrderedDict()
        for l in range(len(channels)):
            layers['conv%d' % l] = ConvGLUBlock(kernel_sizes[l][0], input_dim, channels[l], weight_norm=True, dropout=0.2)
            input_dim = channels[l]
        self.fc_glu = nn.utils.weight_norm(nn.Linear(input_dim, input_dim * 2), name='weight', dim=0)
        self._odim = int(input_dim)
        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim
        self.layers = nn.Sequential(layers)
        self._factor = 1
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with kaiming_uniform style."""
        logger.info('===== Initialize %s with kaiming_uniform style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() in [2, 4]:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                logger.info('Initialize %s with %s / %.3f' % (n, 'kaiming_uniform', param_init))
            else:
                raise ValueError(n)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', out_ch * feat_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, time, input_dim = xs.size()
        xs = xs.transpose(2, 1).unsqueeze(3)
        xs = self.layers(xs)
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)
        xs = F.glu(self.fc_glu(xs), dim=2)
        if self.bridge is not None:
            xs = self.bridge(xs)
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
        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.factor - 1, -1, -1)], dim=-1) for t in range(xs.size(0)) if (t + 1) % self.factor == 0]
        xs = torch.cat(xs, dim=0).transpose(1, 0)
        xs = torch.relu(self.proj(xs))
        xlens //= self.factor
        return xs, xlens


class Conv1dSubsampler(nn.Module):
    """Subsample by 1d convolution and max-pooling."""

    def __init__(self, factor, n_units, conv_kernel_size=5):
        super(Conv1dSubsampler, self).__init__()
        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.factor = factor
        if factor > 1:
            self.conv1d = nn.Conv1d(in_channels=n_units, out_channels=n_units, kernel_size=conv_kernel_size, stride=1, padding=(conv_kernel_size - 1) // 2)
            self.max_pool = nn.MaxPool1d(1, stride=factor, ceil_mode=True)

    def forward(self, xs, xlens):
        if self.factor == 1:
            return xs, xlens
        xs = torch.relu(self.conv1d(xs.transpose(2, 1)))
        xs = self.max_pool(xs).transpose(2, 1).contiguous()
        xlens //= self.factor
        return xs, xlens


def update_lens(seq_lens, layer, dim=0, device_id=-1):
    """Update lenghts (frequency or time).

    Args:
        seq_lens (list or IntTensor):
        layer (nn.Conv2d or nn.MaxPool2d):
        dim (int):
        device_id (int):
    Returns:
        seq_lens (IntTensor):

    """
    if seq_lens is None:
        return seq_lens
    assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
    if type(layer) == nn.MaxPool2d and layer.ceil_mode:

        def update(seq_len):
            return math.ceil((seq_len + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)
    else:

        def update(seq_len):
            return math.floor((seq_len + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)
    seq_lens = [update(seq_len) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    if device_id >= 0:
        seq_lens = seq_lens
    return seq_lens


class Conv2LBlock(EncoderBase):
    """2-layer CNN block."""

    def __init__(self, input_dim, in_channel, out_channel, kernel_size, stride, pooling, dropout, batch_norm, layer_norm, layer_norm_eps, residual):
        super(Conv2LBlock, self).__init__()
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=tuple(kernel_size), stride=tuple(stride), padding=(1, 1))
        input_dim = update_lens([input_dim], self.conv1, dim=1)[0]
        self.batch_norm1 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm1 = LayerNorm2D(out_channel * input_dim.item(), eps=layer_norm_eps) if layer_norm else lambda x: x
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=tuple(kernel_size), stride=tuple(stride), padding=(1, 1))
        input_dim = update_lens([input_dim], self.conv2, dim=1)[0]
        self.batch_norm2 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm2 = LayerNorm2D(out_channel * input_dim.item(), eps=layer_norm_eps) if layer_norm else lambda x: x
        self.pool = None
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling), stride=tuple(pooling), padding=(0, 0), ceil_mode=True)
            input_dim = update_lens([input_dim], self.pool, dim=1)[0]
        self.input_dim = input_dim

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T', feat_dim]`
            xlens (IntTensor): `[B]`

        """
        residual = xs
        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv1, dim=0)
        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens(xlens, self.conv2, dim=0)
        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens(xlens, self.pool, dim=0)
        return xs, xlens


def init_with_lecun(n, p, param_init):
    if p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() == 2:
        fan_in = p.size(1)
        nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))
        logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
    elif p.dim() == 3:
        fan_in = p.size(1) * p[0][0].numel()
        nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))
        logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
    elif p.dim() == 4:
        fan_in = p.size(1) * p[0][0].numel()
        nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan_in))
        logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
    else:
        raise ValueError(n)


class ConvEncoder(EncoderBase):
    """CNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        in_channel (int): number of channels of input features
        channels (list): number of channles in CNN blocks
        kernel_sizes (list): size of kernels in CNN blocks
        strides (list): strides in CNN blocks
        poolings (list): size of poolings in CNN blocks
        dropout (float): probability to drop nodes in hidden-hidden connection
        batch_norm (bool): apply batch normalization
        layer_norm (bool): apply layer normalization
        residual (bool): add residual connections
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float):
        layer_norm_eps (float):

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes, strides, poolings, dropout, batch_norm, layer_norm, residual, bottleneck_dim, param_init, layer_norm_eps=1e-12):
        super(ConvEncoder, self).__init__()
        channels, kernel_sizes, strides, poolings = parse_config(channels, kernel_sizes, strides, poolings)
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual
        self.bridge = None
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)
        self.layers = nn.ModuleList()
        in_ch = in_channel
        in_freq = self.input_freq
        for l in range(len(channels)):
            block = Conv2LBlock(input_dim=in_freq, in_channel=in_ch, out_channel=channels[l], kernel_size=kernel_sizes[l], stride=strides[l], pooling=poolings[l], dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm, layer_norm_eps=layer_norm_eps, residual=residual)
            self.layers += [block]
            in_freq = block.input_dim
            in_ch = channels[l]
        self._odim = int(in_ch * in_freq)
        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim
        self._factor = 1
        if poolings:
            for p in poolings:
                self._factor *= p[1]
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun(n, p, param_init)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', out_ch * feat_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, time, input_dim = xs.size()
        xs = xs.view(bs, time, self.in_channel, input_dim // self.in_channel).contiguous().transpose(2, 1)
        for block in self.layers:
            xs, xlens = block(xs, xlens)
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)
        if self.bridge is not None:
            xs = self.bridge(xs)
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


class SubsampelBlock(nn.Module):

    def __init__(self, in_channel, out_channel, in_freq, dropout):
        super().__init__()
        self.conv1d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
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
        self.conv2d = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(kernel_size // 2, 0))
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(in_freq * channel, eps=1e-06)
        self.conv1d_1 = nn.Conv2d(in_channels=in_freq * channel, out_channels=in_freq * channel, kernel_size=1, stride=1, padding=0)
        self.dropout2_1 = nn.Dropout(p=dropout)
        self.conv1d_2 = nn.Conv2d(in_channels=in_freq * channel, out_channels=in_freq * channel, kernel_size=1, stride=1, padding=0)
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


class TDSEncoder(EncoderBase):
    """TDS (tim-depth separable convolutional) encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channles in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        strides (list): strides in TDS layers
        poolings (list) size of poolings in TDS layers
        dropout (float) probability to drop nodes in hidden-hidden connection
        batch_norm (bool): if True, apply batch normalization
        bottleneck_dim (int): dimension of the bottleneck layer after the last layer

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes, dropout, bottleneck_dim=0):
        super(TDSEncoder, self).__init__()
        channels, kernel_sizes, _, _ = parse_config(channels, kernel_sizes, '', '')
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        layers = OrderedDict()
        in_ch = in_channel
        in_freq = self.input_freq
        for l in range(len(channels)):
            if in_ch != channels[l]:
                layers['subsample%d' % l] = SubsampelBlock(in_channel=in_ch, out_channel=channels[l], in_freq=in_freq, dropout=dropout)
            layers['tds%d_block%d' % (channels[l], l)] = TDSBlock(channel=channels[l], kernel_size=kernel_sizes[l][0], in_freq=in_freq, dropout=dropout)
            in_ch = channels[l]
        self._odim = int(in_ch * in_freq)
        if bottleneck_dim > 0:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim
        self.layers = nn.Sequential(layers)
        self._factor = 8
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() == 2:
                fan_in = p.size(1)
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            elif p.dim() == 4:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            else:
                raise ValueError(n)

    def forward(self, xs, xlens):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (list): A list of length `[B]`
        Returns:
            xs (FloatTensor): `[B, T', out_ch * feat_dim]`
            xlens (list): A list of length `[B]`

        """
        bs, time, input_dim = xs.size()
        xs = xs.contiguous().view(bs, time, self.in_channel, input_dim // self.in_channel).transpose(2, 1)
        xs = self.layers(xs)
        bs, out_ch, time, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, time, -1)
        if self.bridge is not None:
            xs = self.bridge(xs)
        xlens /= 8
        return xs, xlens


class RNNEncoder(EncoderBase):
    """RNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        rnn_type (str): type of encoder (including pure CNN layers)
        n_units (int): number of units in each layer
        n_projs (int): number of units in each projection layer
        last_proj_dim (int): dimension of the last projection layer
        n_layers (int): number of layers
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probability for hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [False, True, True, False] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop/concat/max_pool
        n_stacks (int): number of frames to stack
        n_splices (int): number of frames to splice
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_layer_norm (bool): apply layer normalization only in the CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and RNN layers
        bidirectional_sum_fwd_bwd (bool): sum up forward and backward outputs for demiension reduction
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (float): parameter initialization method
        chunk_size_left (int): left chunk size for latency-controlled bidirectional encoder
        chunk_size_right (int): right chunk size for latency-controlled bidirectional encoder

    """

    def __init__(self, input_dim, rnn_type, n_units, n_projs, last_proj_dim, n_layers, n_layers_sub1, n_layers_sub2, dropout_in, dropout, subsample, subsample_type, n_stacks, n_splices, conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings, conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, bidirectional_sum_fwd_bwd, task_specific_layer, param_init, chunk_size_left, chunk_size_right):
        super(RNNEncoder, self).__init__()
        if len(subsample) > 0 and len(subsample) != n_layers:
            raise ValueError('subsample must be the same size as n_layers. n_layers: %d, subsample: %s' % (n_layers, subsample))
        if n_layers_sub1 < 0 or n_layers_sub1 > 1 and n_layers < n_layers_sub1:
            raise ValueError('Set n_layers_sub1 between 1 to n_layers. n_layers: %d, n_layers_sub1: %d' % (n_layers, n_layers_sub1))
        if n_layers_sub2 < 0 or n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2:
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1. n_layers_sub1: %d, n_layers_sub2: %d' % (n_layers_sub1, n_layers_sub2))
        self.rnn_type = rnn_type
        self.bidirectional = True if 'blstm' in rnn_type or 'bgru' in rnn_type else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_layers = n_layers
        self.bidir_sum = bidirectional_sum_fwd_bwd
        self.latency_controlled = chunk_size_left > 0 or chunk_size_right > 0
        self.chunk_size_left = chunk_size_left
        self.chunk_size_right = chunk_size_right
        if self.latency_controlled:
            assert n_layers_sub2 == 0
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None
        self.dropout_in = nn.Dropout(p=dropout_in)
        if rnn_type == 'tds':
            self.conv = TDSEncoder(input_dim=input_dim * n_stacks, in_channel=conv_in_channel, channels=conv_channels, kernel_sizes=conv_kernel_sizes, dropout=dropout, bottleneck_dim=last_proj_dim)
        elif rnn_type == 'gated_conv':
            self.conv = GatedConvEncoder(input_dim=input_dim * n_stacks, in_channel=conv_in_channel, channels=conv_channels, kernel_sizes=conv_kernel_sizes, dropout=dropout, bottleneck_dim=last_proj_dim, param_init=param_init)
        elif 'conv' in rnn_type:
            assert n_stacks == 1 and n_splices == 1
            self.conv = ConvEncoder(input_dim, in_channel=conv_in_channel, channels=conv_channels, kernel_sizes=conv_kernel_sizes, strides=conv_strides, poolings=conv_poolings, dropout=0.0, batch_norm=conv_batch_norm, layer_norm=conv_layer_norm, residual=False, bottleneck_dim=conv_bottleneck_dim, param_init=param_init)
        else:
            self.conv = None
        if self.conv is None:
            self._odim = input_dim * n_splices * n_stacks
        else:
            self._odim = self.conv.output_dim
            subsample = [1] * self.n_layers
            logger.warning('Subsampling is automatically ignored because CNN layers are used before RNN layers.')
        self.padding = Padding(bidirectional_sum_fwd_bwd=bidirectional_sum_fwd_bwd)
        if rnn_type not in ['conv', 'tds', 'gated_conv']:
            self.rnn = nn.ModuleList()
            if self.latency_controlled:
                self.rnn_bwd = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout)
            self.proj = None
            if n_projs > 0:
                self.proj = nn.ModuleList()
            self.subsample_layer = None
            if subsample_type == 'max_pool' and np.prod(subsample) > 1:
                self.subsample_layer = nn.ModuleList([MaxpoolSubsampler(subsample[l]) for l in range(n_layers)])
            elif subsample_type == 'concat' and np.prod(subsample) > 1:
                self.subsample_layer = nn.ModuleList([ConcatSubsampler(subsample[l], n_units * self.n_dirs) for l in range(n_layers)])
            elif subsample_type == 'drop' and np.prod(subsample) > 1:
                self.subsample_layer = nn.ModuleList([DropSubsampler(subsample[l]) for l in range(n_layers)])
            elif subsample_type == '1dconv' and np.prod(subsample) > 1:
                self.subsample_layer = nn.ModuleList([Conv1dSubsampler(subsample[l], n_units * self.n_dirs) for l in range(n_layers)])
            for l in range(n_layers):
                if 'lstm' in rnn_type:
                    rnn_i = nn.LSTM
                elif 'gru' in rnn_type:
                    rnn_i = nn.GRU
                else:
                    raise ValueError('rnn_type must be "(conv_)(b)lstm" or "(conv_)(b)gru".')
                if self.latency_controlled:
                    self.rnn += [rnn_i(self._odim, n_units, 1, batch_first=True)]
                    self.rnn_bwd += [rnn_i(self._odim, n_units, 1, batch_first=True)]
                else:
                    self.rnn += [rnn_i(self._odim, n_units, 1, batch_first=True, bidirectional=self.bidirectional)]
                self._odim = n_units if bidirectional_sum_fwd_bwd else n_units * self.n_dirs
                if self.proj is not None:
                    if l != n_layers - 1:
                        self.proj += [nn.Linear(n_units * self.n_dirs, n_projs)]
                        self._odim = n_projs
                if l == n_layers_sub1 - 1 and task_specific_layer:
                    assert not self.latency_controlled
                    self.rnn_sub1 = rnn_i(self._odim, n_units, 1, batch_first=True, bidirectional=self.bidirectional)
                    if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                        self.bridge_sub1 = nn.Linear(n_units, last_proj_dim)
                if l == n_layers_sub2 - 1 and task_specific_layer:
                    assert not self.latency_controlled
                    self.rnn_sub2 = rnn_i(self._odim, n_units, 1, batch_first=True, bidirectional=self.bidirectional)
                    if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                        self.bridge_sub2 = nn.Linear(n_units, last_proj_dim)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge = nn.Linear(self._odim, last_proj_dim)
                self._odim = last_proj_dim
        self._factor = 1
        if self.conv is not None:
            self._factor *= self.conv.subsampling_factor
        self._factor *= np.prod(subsample)
        self.reset_parameters(param_init)
        self.reset_cache()

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'conv' in n or 'tds' in n or 'gated_conv' in n:
                continue
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError(n)

    def reset_cache(self):
        self.fwd_states = [None] * self.n_layers
        logger.debug('Reset cache.')

    def forward(self, xs, xlens, task, use_cache=False, streaming=False):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): A list of length `[B]`
            task (str): all or ys or ys_sub1 or ys_sub2
            use_cache (bool): use the cached forward encoder state in the previous chunk as the initial state
            streaming (bool): streaming encoding
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens (IntTensor): `[B]`
                xs_sub1 (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens_sub1 (IntTensor): `[B]`
                xs_sub2 (FloatTensor): `[B, T // prod(subsample), n_units (*2)]`
                xlens_sub2 (IntTensor): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None}, 'ys_sub1': {'xs': None, 'xlens': None}, 'ys_sub2': {'xs': None, 'xlens': None}}
        if not self.latency_controlled:
            xlens, perm_ids = torch.IntTensor(xlens).sort(0, descending=True)
            xs = xs[perm_ids]
            _, perm_ids_unsort = perm_ids.sort()
        xs = self.dropout_in(xs)
        if self.conv is not None:
            xs, xlens = self.conv(xs, xlens)
            if self.rnn_type in ['conv', 'tds', 'gated_conv']:
                eouts['ys']['xs'] = xs
                eouts['ys']['xlens'] = xlens
                return eouts
        if not use_cache:
            self.reset_cache()
        if self.latency_controlled:
            xs, xlens, xs_sub1 = self._forward_streaming(xs, xlens, streaming)
            xlens_sub1 = xlens.clone()
        else:
            for l in range(self.n_layers):
                self.rnn[l].flatten_parameters()
                xs, self.fwd_states[l] = self.padding(xs, xlens, self.rnn[l], prev_state=self.fwd_states[l])
                xs = self.dropout(xs)
                if l == self.n_layers_sub1 - 1:
                    xs_sub1, xlens_sub1 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub1')
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                        return eouts
                if l == self.n_layers_sub2 - 1:
                    xs_sub2, xlens_sub2 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub2')
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens_sub2
                        return eouts
                if l != self.n_layers - 1:
                    if self.proj is not None:
                        xs = torch.tanh(self.proj[l](xs))
                    if self.subsample_layer is not None:
                        xs, xlens = self.subsample_layer[l](xs, xlens)
        if self.bridge is not None:
            xs = self.bridge(xs)
        if not self.latency_controlled:
            xs = xs[perm_ids_unsort]
            xlens = xlens[perm_ids_unsort]
        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens_sub2
        return eouts

    def _forward_streaming(self, xs, xlens, streaming, task='all'):
        """Streaming encoding for the latency-controlled bidirectional encoder.

        Args:
            xs (FloatTensor): `[B, T, n_units]`
        Returns:
            xs (FloatTensor): `[B, T, n_units]`

        """
        N_l = self.chunk_size_left // self.subsampling_factor
        N_r = self.chunk_size_right // self.subsampling_factor
        xs_sub1 = None
        if N_l < 0:
            for l in range(self.n_layers):
                self.rnn[l].flatten_parameters()
                self.rnn_bwd[l].flatten_parameters()
                xs_bwd = torch.flip(xs, dims=[1])
                xs_bwd, _ = self.rnn_bwd[l](xs_bwd, hx=None)
                xs_bwd = torch.flip(xs_bwd, dims=[1])
                xs_fwd, _ = self.rnn[l](xs, hx=None)
                if self.bidir_sum:
                    xs = xs_fwd + xs_bwd
                else:
                    xs = torch.cat([xs_fwd, xs_bwd], dim=-1)
                xs = self.dropout(xs)
                if l == self.n_layers_sub1 - 1:
                    xs_sub1 = xs.clone()
                    if self.bridge_sub1 is not None:
                        xs_sub1 = self.bridge_sub1(xs_sub1)
                    if task == 'ys_sub1':
                        return None, xlens, xs_sub1
                if l != self.n_layers - 1:
                    if self.proj is not None:
                        xs = torch.tanh(self.proj[l](xs))
                    if self.subsample_layer is not None:
                        xs, xlens = self.subsample_layer[l](xs, xlens)
            return xs, xlens, xs_sub1
        bs, xmax, input_dim = xs.size()
        n_chunks = 1 if streaming else math.ceil(xmax / N_l)
        xlens = torch.IntTensor(bs).fill_(N_l if streaming else xmax)
        xs_chunks = []
        xs_chunks_sub1 = []
        for t in range(0, N_l * n_chunks, N_l):
            xs_chunk = xs[:, t:t + (N_l + N_r)]
            for l in range(self.n_layers):
                self.rnn[l].flatten_parameters()
                self.rnn_bwd[l].flatten_parameters()
                xs_chunk_bwd = torch.flip(xs_chunk, dims=[1])
                xs_chunk_bwd, _ = self.rnn_bwd[l](xs_chunk_bwd, hx=None)
                xs_chunk_bwd = torch.flip(xs_chunk_bwd, dims=[1])
                if xs_chunk.size(1) <= N_l:
                    xs_chunk_fwd, self.fwd_states[l] = self.rnn[l](xs_chunk, hx=self.fwd_states[l])
                else:
                    xs_chunk_fwd1, self.fwd_states[l] = self.rnn[l](xs_chunk[:, :N_l], hx=self.fwd_states[l])
                    xs_chunk_fwd2, _ = self.rnn[l](xs_chunk[:, N_l:], hx=self.fwd_states[l])
                    xs_chunk_fwd = torch.cat([xs_chunk_fwd1, xs_chunk_fwd2], dim=1)
                if self.bidir_sum:
                    xs_chunk = xs_chunk_fwd + xs_chunk_bwd
                else:
                    xs_chunk = torch.cat([xs_chunk_fwd, xs_chunk_bwd], dim=-1)
                xs_chunk = self.dropout(xs_chunk)
                if l == self.n_layers_sub1 - 1:
                    xs_chunk_sub1 = xs_chunk.clone()
                    if self.bridge_sub1 is not None:
                        xs_chunk_sub1 = self.bridge_sub1(xs_chunk_sub1)
                    if task == 'ys_sub1':
                        return None, xlens, xs_chunk_sub1
                if l != self.n_layers - 1:
                    if self.proj is not None:
                        xs_chunk = torch.tanh(self.proj[l](xs_chunk))
                    if self.subsample_layer is not None:
                        xs_chunk, xlens = self.subsample_layer[l](xs_chunk, xlens)
            xs_chunks.append(xs_chunk[:, :N_l])
            if self.n_layers_sub1 > 0:
                xs_chunks_sub1.append(xs_chunk_sub1[:, :N_l])
        xs = torch.cat(xs_chunks, dim=1)
        if self.n_layers_sub1 > 0:
            xs_sub1 = torch.cat(xs_chunks_sub1, dim=1)
        return xs, xlens, xs_sub1

    def sub_module(self, xs, xlens, perm_ids_unsort, module='sub1'):
        if self.task_specific_layer:
            getattr(self, 'rnn_' + module).flatten_parameters()
            xs_sub, _ = self.padding(xs, xlens, getattr(self, 'rnn_' + module))
            xs_sub = self.dropout(xs_sub)
        else:
            xs_sub = xs.clone()[perm_ids_unsort]
        if getattr(self, 'bridge_' + module) is not None:
            xs_sub = getattr(self, 'bridge_' + module)(xs_sub)
        xlens_sub = xlens[perm_ids_unsort]
        return xs_sub, xlens_sub


class NiN(nn.Module):
    """Network in network."""

    def __init__(self, dim):
        super(NiN, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(dim)

    def forward(self, xs):
        xs = xs.contiguous().transpose(2, 1).unsqueeze(3)
        xs = torch.relu(self.batch_norm(self.conv(xs)))
        xs = xs.transpose(2, 1).squeeze(3)
        return xs


class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder
        attn_type (str): type of attention
        n_heads (int): number of heads for multi-head attention
        n_layers (int): number of blocks
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        last_proj_dim (int): dimension of the last projection layer
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in the CNN blocks
        conv_kernel_sizes (list): size of kernels in the CNN blocks
        conv_strides (list): number of strides in the CNN blocks
        conv_poolings (list): size of poolings in the CNN blocks
        conv_batch_norm (bool): apply batch normalization only in the CNN blocks
        conv_layer_norm (bool): apply layer normalization only in the CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers
        conv_param_init (float): only for CNN layers before Transformer layers
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (str): parameter initialization method
        chunk_size_left (int): left chunk size for time-restricted Transformer encoder
        chunk_size_current (int): current chunk size for time-restricted Transformer encoder
        chunk_size_right (int): right chunk size for time-restricted Transformer encoder

    """

    def __init__(self, input_dim, enc_type, attn_type, n_heads, n_layers, n_layers_sub1, n_layers_sub2, d_model, d_ff, last_proj_dim, pe_type, layer_norm_eps, ffn_activation, dropout_in, dropout, dropout_att, dropout_layer, n_stacks, n_splices, conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings, conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, conv_param_init, task_specific_layer, param_init, chunk_size_left, chunk_size_current, chunk_size_right):
        super(TransformerEncoder, self).__init__()
        if n_layers_sub1 < 0 or n_layers_sub1 > 1 and n_layers < n_layers_sub1:
            raise ValueError('Set n_layers_sub1 between 1 to n_layers.')
        if n_layers_sub2 < 0 or n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2:
            raise ValueError('Set n_layers_sub2 between 1 to n_layers_sub1.')
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.N_l = chunk_size_left
        self.N_c = chunk_size_current
        self.N_r = chunk_size_right
        self.latency_controlled = chunk_size_left > 0 or chunk_size_current > 0 or chunk_size_right > 0
        self.memory_transformer = 'transformer_xl' in enc_type
        self.mem_len = chunk_size_left
        self.scale = math.sqrt(d_model)
        if self.memory_transformer:
            assert pe_type == 'none'
            assert chunk_size_left > 0
            assert chunk_size_current > 0
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None
        self.aws_dict = {}
        self.data_dict = {}
        if conv_channels:
            assert n_stacks == 1 and n_splices == 1
            self.conv = ConvEncoder(input_dim, in_channel=conv_in_channel, channels=conv_channels, kernel_sizes=conv_kernel_sizes, strides=conv_strides, poolings=conv_poolings, dropout=0.0, batch_norm=conv_batch_norm, layer_norm=conv_layer_norm, layer_norm_eps=layer_norm_eps, residual=False, bottleneck_dim=d_model, param_init=conv_param_init)
            self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)
        self._factor = 1
        if self.conv is not None:
            self._factor *= self.conv.subsampling_factor
        if self.memory_transformer:
            self.pos_emb = XLPositionalEmbedding(d_model, dropout)
            self.u = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
            self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type, param_init)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderBlock(d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, memory_transformer=self.memory_transformer)) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._odim = d_model
        if n_layers_sub1 > 0:
            if task_specific_layer:
                self.layer_sub1 = TransformerEncoderBlock(d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init)
            self.norm_out_sub1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub1 = nn.Linear(self._odim, last_proj_dim)
        if n_layers_sub2 > 0:
            if task_specific_layer:
                self.layer_sub2 = TransformerEncoderBlock(d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init)
            self.norm_out_sub2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub2 = nn.Linear(self._odim, last_proj_dim)
        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if self.memory_transformer:
            logger.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
            for n, p in self.named_parameters():
                if 'conv' in n:
                    continue
                init_with_normal_dist(n, p, std=0.02)
        elif param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            if self.conv is None:
                nn.init.xavier_uniform_(self.embed.weight)
                nn.init.constant_(self.embed.bias, 0.0)
            if self.bridge is not None:
                nn.init.xavier_uniform_(self.bridge.weight)
                nn.init.constant_(self.bridge.bias, 0.0)
            if self.bridge_sub1 is not None:
                nn.init.xavier_uniform_(self.bridge_sub1.weight)
                nn.init.constant_(self.bridge_sub1.bias, 0.0)
            if self.bridge_sub2 is not None:
                nn.init.xavier_uniform_(self.bridge_sub2.weight)
                nn.init.constant_(self.bridge_sub2.bias, 0.0)

    def init_memory(self):
        """Initialize memory."""
        if self.device_id >= 0:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]
        else:
            return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): length `n_layers`, each of which contains [B, L_prev, d_model]`
            hidden_states (list): length `n_layers`, each of which contains [B, L, d_model]`
        Returns:
            new_mems (list): length `n_layers`, each of which contains `[B, mlen, d_model]`

        """
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.mem_len // self.subsampling_factor)
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)
                new_mems.append(cat[:, start_idx:end_idx].detach())
        return new_mems

    def forward(self, xs, xlens, task, use_cache=False, streaming=False):
        """Forward computation.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): `[B]`
            task (str): not supported now
            use_cache (bool):
            streaming (bool): streaming encoding
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (list): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None}, 'ys_sub1': {'xs': None, 'xlens': None}, 'ys_sub2': {'xs': None, 'xlens': None}}
        if self.latency_controlled:
            bs, xmax, idim = xs.size()
            n_blocks = xmax // self.N_c
            if xmax % self.N_c != 0:
                n_blocks += 1
            xs_tmp = xs.new_zeros(bs, n_blocks, self.N_l + self.N_c + self.N_r, idim)
            xs_pad = torch.cat([xs.new_zeros(bs, self.N_l, idim), xs, xs.new_zeros(bs, self.N_r, idim)], dim=1)
            for blc_id, t in enumerate(range(self.N_l, self.N_l + xmax, self.N_c)):
                xs_chunk = xs_pad[:, t - self.N_l:t + (self.N_c + self.N_r)]
                xs_tmp[:, (blc_id), :xs_chunk.size(1), :] = xs_chunk
            xs = xs_tmp.view(bs * n_blocks, self.N_l + self.N_c + self.N_r, idim)
        if self.conv is None:
            xs = self.embed(xs)
        else:
            xs, xlens = self.conv(xs, xlens)
        if not self.training:
            self.data_dict['elens'] = tensor2np(xlens)
        if self.latency_controlled:
            N_l = max(0, self.N_l // self.subsampling_factor)
            N_c = self.N_c // self.subsampling_factor
            emax = xmax // self.subsampling_factor
            if xmax % self.subsampling_factor != 0:
                emax += 1
            xs = self.pos_enc(xs, scale=True)
            xx_mask = None
            for lth, layer in enumerate(self.layers):
                xs, xx_aws = layer(xs, xx_mask)
                if not self.training:
                    n_heads = xx_aws.size(1)
                    xx_aws = xx_aws[:, :, N_l:N_l + N_c, N_l:N_l + N_c]
                    xx_aws = xx_aws.view(bs, n_blocks, n_heads, N_c, N_c)
                    xx_aws_center = xx_aws.new_zeros(bs, n_heads, emax, emax)
                    for blc_id in range(n_blocks):
                        offset = blc_id * N_c
                        emax_blc = xx_aws_center[:, :, offset:offset + N_c].size(2)
                        xx_aws_chunk = xx_aws[:, (blc_id), :, :emax_blc, :emax_blc]
                        xx_aws_center[:, :, offset:offset + N_c, offset:offset + N_c] = xx_aws_chunk
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws_center)
            xs = xs[:, N_l:N_l + N_c]
            xs = xs.contiguous().view(bs, -1, xs.size(2))
            xs = xs[:, :emax]
        else:
            bs, xmax, idim = xs.size()
            xs = self.pos_enc(xs, scale=True)
            xx_mask = make_pad_mask(xlens, self.device_id).unsqueeze(2).repeat([1, 1, xmax])
            for lth, layer in enumerate(self.layers):
                xs, xx_aws = layer(xs, xx_mask)
                if not self.training:
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws)
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1 = self.layer_sub1(xs, xx_mask)[0] if self.task_specific_layer else xs.clone()
                    xs_sub1 = self.norm_out_sub1(xs_sub1)
                    if self.bridge_sub1 is not None:
                        xs_sub1 = self.bridge_sub1(xs_sub1)
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2 = self.layer_sub2(xs, xx_mask)[0] if self.task_specific_layer else xs.clone()
                    xs_sub2 = self.norm_out_sub2(xs_sub2)
                    if self.bridge_sub2 is not None:
                        xs_sub2 = self.bridge_sub2(xs_sub2)
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens
                        return eouts
        xs = self.norm_out(xs)
        if self.bridge is not None:
            xs = self.bridge(xs)
        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens
        return eouts

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        _save_path = mkdir_join(save_path, 'enc_att_weights')
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)
        for k, aw in self.aws_dict.items():
            elens = self.data_dict['elens']
            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp, figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                ax.imshow(aw[(-1), (h), :elens[-1], :elens[-1]], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()


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

    def __init__(self, input_dim, n_units, n_layers, bottleneck_dim, dropout, param_init=0.1):
        super(SequenceSummaryNetwork, self).__init__()
        self.n_layers = n_layers
        self.ssn = nn.ModuleList()
        self.ssn += [nn.Linear(input_dim, n_units, bias=False)]
        self.ssn += [nn.Dropout(p=dropout)]
        for l in range(1, n_layers - 1):
            self.ssn += [nn.Linear(n_units, bottleneck_dim if l == n_layers - 2 else n_units, bias=False)]
            self.ssn += [nn.Dropout(p=dropout)]
        self.p = nn.Linear(bottleneck_dim, input_dim, bias=False)
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
            elif p.dim() == 2:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
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


class SpecAugment(object):
    """SpecAugment calss.

    Args:
        W (int): parameter for time warping
        F (int): parameter for frequency masking
        T (int): parameter for time masking
        n_freq_masks (int): number of frequency masks
        n_time_masks (int): number of time masks
        p (float): parameter for upperbound of the time mask

    """

    def __init__(self, W=40, F=27, T=70, n_freq_masks=2, n_time_masks=2, p=0.2):
        super(SpecAugment, self).__init__()
        self.W = W
        self.F = F
        self.T = T
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
        self._freq_mask = None
        self._time_mask = None

    @property
    def librispeech_basic(self):
        raise NotImplementedError

    @property
    def librispeech_double(self):
        raise NotImplementedError

    @property
    def switchboard_mild(self):
        raise NotImplementedError

    @property
    def switchboard_strong(self):
        raise NotImplementedError

    @property
    def freq_mask(self):
        return self._freq_mask

    @property
    def time_mask(self):
        return self._time_mask

    def __call__(self, xs):
        """
        Args:
            xs (FloatTensor): `[B, T, F]`
        Returns:
            xs (FloatTensor): `[B, T, F]`

        """
        xs = self.mask_freq_dim(xs)
        xs = self.mask_time_dim(xs)
        return xs

    def time_warp(xs, W=40):
        raise NotImplementedError

    def mask_freq_dim(self, xs, replace_with_zero=False):
        n_bins = xs.size(-1)
        for i in range(0, self.n_freq_masks):
            f = int(np.random.uniform(low=0, high=self.F))
            f_0 = int(np.random.uniform(low=0, high=n_bins - f))
            xs[:, :, f_0:f_0 + f] = 0
            assert f_0 <= f_0 + f
            self._freq_mask = f_0, f_0 + f
        return xs

    def mask_time_dim(self, xs, replace_with_zero=False):
        n_frames = xs.size(1)
        for i in range(self.n_time_masks):
            t = int(np.random.uniform(low=0, high=self.T))
            t = min(t, int(n_frames * self.p))
            t_0 = int(np.random.uniform(low=0, high=n_frames - t))
            xs[:, t_0:t_0 + t] = 0
            assert t_0 <= t_0 + t
            self._time_mask = t_0, t_0 + t
        return xs


def build_decoder(args, special_symbols, enc_n_units, vocab, ctc_weight, ctc_fc_list, global_weight, external_lm=None):
    if args.dec_type in ['transformer', 'transformer_xl']:
        decoder = TransformerDecoder(special_symbols=special_symbols, enc_n_units=enc_n_units, attn_type=args.transformer_attn_type, n_heads=args.transformer_n_heads, n_layers=args.dec_n_layers, d_model=args.transformer_d_model, d_ff=args.transformer_d_ff, layer_norm_eps=args.transformer_layer_norm_eps, ffn_activation=args.transformer_ffn_activation, pe_type=args.transformer_dec_pe_type, vocab=vocab, tie_embedding=args.tie_embedding, dropout=args.dropout_dec, dropout_emb=args.dropout_emb, dropout_att=args.dropout_att, dropout_layer=args.dropout_dec_layer, dropout_head=args.dropout_head, lsm_prob=args.lsm_prob, ctc_weight=ctc_weight, ctc_lsm_prob=args.ctc_lsm_prob, ctc_fc_list=ctc_fc_list, backward=dir == 'bwd', global_weight=global_weight, mtl_per_batch=args.mtl_per_batch, param_init=args.transformer_param_init, memory_transformer=args.dec_type == 'transformer_xl', mem_len=args.mem_len, mocha_chunk_size=args.mocha_chunk_size, mocha_n_heads_mono=args.mocha_n_heads_mono, mocha_n_heads_chunk=args.mocha_n_heads_chunk, mocha_init_r=args.mocha_init_r, mocha_eps=args.mocha_eps, mocha_std=args.mocha_std, mocha_no_denominator=args.mocha_no_denominator, mocha_1dconv=args.mocha_1dconv, mocha_quantity_loss_weight=args.mocha_quantity_loss_weight, mocha_head_divergence_loss_weight=args.mocha_head_divergence_loss_weight, latency_metric=args.mocha_latency_metric, latency_loss_weight=args.mocha_latency_loss_weight, mocha_first_layer=args.mocha_first_layer, external_lm=external_lm, lm_fusion=args.lm_fusion)
    elif args.dec_type in ['lstm_transducer', 'gru_transducer']:
        decoder = RNNTransducer(special_symbols=special_symbols, enc_n_units=enc_n_units, rnn_type=args.dec_type, n_units=args.dec_n_units, n_projs=args.dec_n_projs, n_layers=args.dec_n_layers, bottleneck_dim=args.dec_bottleneck_dim, emb_dim=args.emb_dim, vocab=vocab, dropout=args.dropout_dec, dropout_emb=args.dropout_emb, lsm_prob=args.lsm_prob, ctc_weight=ctc_weight, ctc_lsm_prob=args.ctc_lsm_prob, ctc_fc_list=ctc_fc_list, external_lm=external_lm if args.lm_init else None, global_weight=global_weight, mtl_per_batch=args.mtl_per_batch, param_init=args.param_init)
    else:
        decoder = RNNDecoder(special_symbols=special_symbols, enc_n_units=enc_n_units, rnn_type=args.dec_type, n_units=args.dec_n_units, n_projs=args.dec_n_projs, n_layers=args.dec_n_layers, bottleneck_dim=args.dec_bottleneck_dim, emb_dim=args.emb_dim, vocab=vocab, tie_embedding=args.tie_embedding, attn_type=args.attn_type, attn_dim=args.attn_dim, attn_sharpening_factor=args.attn_sharpening_factor, attn_sigmoid_smoothing=args.attn_sigmoid, attn_conv_out_channels=args.attn_conv_n_channels, attn_conv_kernel_size=args.attn_conv_width, attn_n_heads=args.attn_n_heads, dropout=args.dropout_dec, dropout_emb=args.dropout_emb, dropout_att=args.dropout_att, lsm_prob=args.lsm_prob, ss_prob=args.ss_prob, ss_type=args.ss_type, ctc_weight=ctc_weight, ctc_lsm_prob=args.ctc_lsm_prob, ctc_fc_list=ctc_fc_list, mbr_training=args.mbr_training, mbr_ce_weight=args.mbr_ce_weight, external_lm=external_lm, lm_fusion=args.lm_fusion, lm_init=args.lm_init, backward=dir == 'bwd', global_weight=global_weight, mtl_per_batch=args.mtl_per_batch, param_init=args.param_init, mocha_chunk_size=args.mocha_chunk_size, mocha_n_heads_mono=args.mocha_n_heads_mono, mocha_init_r=args.mocha_init_r, mocha_eps=args.mocha_eps, mocha_std=args.mocha_std, mocha_no_denominator=args.mocha_no_denominator, mocha_1dconv=args.mocha_1dconv, mocha_quantity_loss_weight=args.mocha_quantity_loss_weight, latency_metric=args.mocha_latency_metric, latency_loss_weight=args.mocha_latency_loss_weight, gmm_attn_n_mixtures=args.gmm_attn_n_mixtures, replace_sos=args.replace_sos, distillation_weight=args.distillation_weight, discourse_aware=args.discourse_aware)
    return decoder


def build_encoder(args):
    if args.enc_type == 'tds':
        raise ValueError
        encoder = TDSEncoder(input_dim=args.input_dim * args.n_stacks, in_channel=args.conv_in_channel, channels=args.conv_channels, kernel_sizes=args.conv_kernel_sizes, dropout=args.dropout_enc, bottleneck_dim=args.transformer_d_model if 'transformer' in args.dec_type else args.dec_n_units)
    elif args.enc_type == 'gated_conv':
        raise ValueError
        encoder = GatedConvEncoder(input_dim=args.input_dim * args.n_stacks, in_channel=args.conv_in_channel, channels=args.conv_channels, kernel_sizes=args.conv_kernel_sizes, dropout=args.dropout_enc, bottleneck_dim=args.transformer_d_model if 'transformer' in args.dec_type else args.dec_n_units, param_init=args.param_init)
    elif 'transformer' in args.enc_type:
        encoder = TransformerEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim, enc_type=args.enc_type, attn_type=args.transformer_attn_type, n_heads=args.transformer_n_heads, n_layers=args.enc_n_layers, n_layers_sub1=args.enc_n_layers_sub1, n_layers_sub2=args.enc_n_layers_sub2, d_model=args.transformer_d_model, d_ff=args.transformer_d_ff, last_proj_dim=args.transformer_d_model if 'transformer' in args.dec_type else 0, pe_type=args.transformer_enc_pe_type, layer_norm_eps=args.transformer_layer_norm_eps, ffn_activation=args.transformer_ffn_activation, dropout_in=args.dropout_in, dropout=args.dropout_enc, dropout_att=args.dropout_att, dropout_layer=args.dropout_enc_layer, n_stacks=args.n_stacks, n_splices=args.n_splices, conv_in_channel=args.conv_in_channel, conv_channels=args.conv_channels, conv_kernel_sizes=args.conv_kernel_sizes, conv_strides=args.conv_strides, conv_poolings=args.conv_poolings, conv_batch_norm=args.conv_batch_norm, conv_layer_norm=args.conv_layer_norm, conv_bottleneck_dim=args.conv_bottleneck_dim, conv_param_init=args.param_init, task_specific_layer=args.task_specific_layer, param_init=args.transformer_param_init, chunk_size_left=args.lc_chunk_size_left, chunk_size_current=args.lc_chunk_size_current, chunk_size_right=args.lc_chunk_size_right)
    else:
        subsample = [1] * args.enc_n_layers
        for l, s in enumerate(list(map(int, args.subsample.split('_')[:args.enc_n_layers]))):
            subsample[l] = s
        encoder = RNNEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim, rnn_type=args.enc_type, n_units=args.enc_n_units, n_projs=args.enc_n_projs, last_proj_dim=args.transformer_d_model if 'transformer' in args.dec_type else 0, n_layers=args.enc_n_layers, n_layers_sub1=args.enc_n_layers_sub1, n_layers_sub2=args.enc_n_layers_sub2, dropout_in=args.dropout_in, dropout=args.dropout_enc, subsample=subsample, subsample_type=args.subsample_type, n_stacks=args.n_stacks, n_splices=args.n_splices, conv_in_channel=args.conv_in_channel, conv_channels=args.conv_channels, conv_kernel_sizes=args.conv_kernel_sizes, conv_strides=args.conv_strides, conv_poolings=args.conv_poolings, conv_batch_norm=args.conv_batch_norm, conv_layer_norm=args.conv_layer_norm, conv_bottleneck_dim=args.conv_bottleneck_dim, bidirectional_sum_fwd_bwd=args.bidirectional_sum_fwd_bwd, task_specific_layer=args.task_specific_layer, param_init=args.param_init, chunk_size_left=args.lc_chunk_size_left, chunk_size_right=args.lc_chunk_size_right)
    return encoder


def fwd_bwd_attention(nbest_hyps_fwd, aws_fwd, scores_fwd, nbest_hyps_bwd, aws_bwd, scores_bwd, eos, gnmt_decoding, lp_weight, idx2token, refs_id, flip=False):
    """Decoding with the forward and backward attention-based decoders.

    Args:
        nbest_hyps_fwd (list): A list of length `[B]`, which contains list of n hypotheses
        aws_fwd (list): A list of length `[B]`, which contains arrays of size `[L, T]`
        scores_fwd (list):
        nbest_hyps_bwd (list):
        aws_bwd (list):
        scores_bwd (list):
        eos (int):
        gnmt_decoding (float):
        lp_weight (float):
        idx2token (): converter from index to token
        refs_id ():
        flip (bool): flip the encoder indices
    Returns:

    """
    bs = len(nbest_hyps_fwd)
    nbest = len(nbest_hyps_fwd[0])
    best_hyps = []
    for b in range(bs):
        max_time = len(aws_fwd[b][0])
        merged = []
        for n in range(nbest):
            if len(nbest_hyps_fwd[b][n]) > 1:
                if nbest_hyps_fwd[b][n][-1] == eos:
                    merged.append({'hyp': nbest_hyps_fwd[b][n][:-1], 'score': scores_fwd[b][n][-2]})
                else:
                    merged.append({'hyp': nbest_hyps_fwd[b][n], 'score': scores_fwd[b][n][-1]})
            else:
                logger.info(nbest_hyps_fwd[b][n])
            if len(nbest_hyps_bwd[b][n]) > 1:
                if nbest_hyps_bwd[b][n][0] == eos:
                    merged.append({'hyp': nbest_hyps_bwd[b][n][1:], 'score': scores_bwd[b][n][1]})
                else:
                    merged.append({'hyp': nbest_hyps_bwd[b][n], 'score': scores_bwd[b][n][0]})
            else:
                logger.info(nbest_hyps_bwd[b][n])
        for n_f in range(nbest):
            for n_b in range(nbest):
                for i_f in range(len(aws_fwd[b][n_f]) - 1):
                    for i_b in range(len(aws_bwd[b][n_b]) - 1):
                        if flip:
                            t_prev = max_time - aws_bwd[b][n_b][i_b + 1].argmax(-2)
                            t_curr = aws_fwd[b][n_f][i_f].argmax(-2)
                            t_next = max_time - aws_bwd[b][n_b][i_b - 1].argmax(-2)
                        else:
                            t_prev = aws_bwd[b][n_b][i_b + 1].argmax(-2)
                            t_curr = aws_fwd[b][n_f][i_f].argmax(-2)
                            t_next = aws_bwd[b][n_b][i_b - 1].argmax(-2)
                        if t_curr >= t_prev and t_curr <= t_next and nbest_hyps_fwd[b][n_f][i_f] == nbest_hyps_bwd[b][n_b][i_b]:
                            new_hyp = nbest_hyps_fwd[b][n_f][:i_f + 1].tolist() + nbest_hyps_bwd[b][n_b][i_b + 1:].tolist()
                            score_curr_fwd = scores_fwd[b][n_f][i_f] - scores_fwd[b][n_f][i_f - 1]
                            score_curr_bwd = scores_bwd[b][n_b][i_b] - scores_bwd[b][n_b][i_b + 1]
                            score_curr = max(score_curr_fwd, score_curr_bwd)
                            new_score = scores_fwd[b][n_f][i_f - 1] + scores_bwd[b][n_b][i_b + 1] + score_curr
                            merged.append({'hyp': new_hyp, 'score': new_score})
                            logger.info('time matching')
                            if refs_id is not None:
                                logger.info('Ref: %s' % idx2token(refs_id[b]))
                            logger.info('hyp (fwd): %s' % idx2token(nbest_hyps_fwd[b][n_f]))
                            logger.info('hyp (bwd): %s' % idx2token(nbest_hyps_bwd[b][n_b]))
                            logger.info('hyp (fwd-bwd): %s' % idx2token(new_hyp))
                            logger.info('log prob (fwd): %.3f' % scores_fwd[b][n_f][-1])
                            logger.info('log prob (bwd): %.3f' % scores_bwd[b][n_b][0])
                            logger.info('log prob (fwd-bwd): %.3f' % new_score)
        merged = sorted(merged, key=lambda x: x['score'], reverse=True)
        best_hyps.append(merged[0]['hyp'])
    return best_hyps


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """Load checkpoint.

    Args:
        model (torch.nn.Module):
        checkpoint_path (str): path to the saved model (model..epoch-*)
        optimizer (LRScheduler): optimizer wrapped by LRScheduler class
    Returns:
        topk_list (list): list of (epoch, metric)

    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError('There is no checkpoint')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError('No checkpoint found at %s' % checkpoint_path)
    if 'avg' not in checkpoint_path:
        epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) - 1
        logger.info('=> Loading checkpoint (epoch:%d): %s' % (epoch + 1, checkpoint_path))
    else:
        logger.info('=> Loading checkpoint: %s' % checkpoint_path)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint['state_dict'])
        checkpoint['model_state_dict'] = checkpoint['state_dict']
        checkpoint['optimizer_state_dict'] = checkpoint['optimizer']
        del checkpoint['state_dict']
        del checkpoint['optimizer']
        torch.save(checkpoint, checkpoint_path + '.tmp')
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.optimizer.param_groups[0]['params'] = []
        for param_group in list(model.parameters()):
            optimizer.optimizer.param_groups[0]['params'].append(param_group)
    else:
        logger.warning('Optimizer is not loaded.')
    if 'optimizer_state_dict' in checkpoint.keys() and 'topk_list' in checkpoint['optimizer_state_dict'].keys():
        topk_list = checkpoint['optimizer_state_dict']['topk_list']
    else:
        topk_list = []
    return topk_list


def splice(feat, n_splices=1, n_stacks=1, dtype=np.float32):
    """Splice input data. This is expected to be used for CNN-like encoder.

    Args:
        feat (np.ndarray): A tensor of size
            `[T, input_dim (freq * 3 * n_stacks)]'
        n_splices (int): frames to n_splices. Default is 1 frame.
            ex.) if n_splices == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        n_stacks (int): the number of frames to stack
        dtype ():
    Returns:
        feat_splice (np.ndarray): A tensor of size
            `[T, freq * (n_splices * n_stacks) * 3 (static + Δ + ΔΔ)]`

    """
    assert isinstance(feat, np.ndarray), 'feat should be np.ndarray.'
    assert len(feat.shape) == 2, 'feat must be 2 demension.'
    assert feat.shape[-1] % 3 == 0
    if n_splices == 1:
        return feat
    max_xlen, input_dim = feat.shape
    freq = input_dim // 3 // n_stacks
    feat_splice = np.zeros((max_xlen, freq * (n_splices * n_stacks) * 3), dtype=dtype)
    for i_time in range(max_xlen):
        spliced_frames = np.zeros((n_splices * n_stacks, freq, 3))
        for i_splice in range(0, n_splices, 1):
            if i_time <= n_splices - 1 and i_splice < n_splices - i_time:
                copy_frame = feat[0]
            elif max_xlen - n_splices <= i_time and i_time + (i_splice - n_splices) > max_xlen - 1:
                copy_frame = feat[-1]
            else:
                copy_frame = feat[i_time + (i_splice - n_splices)]
            copy_frame = copy_frame.reshape((freq, 3, n_stacks))
            copy_frame = np.transpose(copy_frame, (2, 0, 1))
            spliced_frames[i_splice:i_splice + n_stacks] = copy_frame
        spliced_frames = np.transpose(spliced_frames, (1, 0, 2))
        feat_splice[i_time] = spliced_frames.reshape(freq * (n_splices * n_stacks) * 3)
    return feat_splice


def stack_frame(feat, n_stacks, n_skips, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        feat (list): `[T, input_dim]`
        n_stacks (int): the number of frames to stack
        n_skips (int): the number of frames to skip
        dtype ():
    Returns:
        stacked_feat (np.ndarray): `[floor(T / n_skips), input_dim * n_stacks]`

    """
    if n_stacks == 1 and n_stacks == 1:
        return feat
    if n_stacks < n_skips:
        raise ValueError('n_skips must be less than n_stacks.')
    n_frames, input_dim = feat.shape
    n_frames_new = (n_frames + 1) // n_skips
    stacked_feat = np.zeros((n_frames_new, input_dim * n_stacks), dtype=dtype)
    stack_count = 0
    stack = []
    for t, frame_t in enumerate(feat):
        if t == len(feat) - 1:
            stack.append(frame_t)
            while stack_count != int(n_frames_new):
                for i in range(len(stack)):
                    stacked_feat[stack_count][input_dim * i:input_dim * (i + 1)] = stack[i]
                stack_count += 1
                for _ in range(n_skips):
                    if len(stack) != 0:
                        stack.pop(0)
        elif len(stack) < n_stacks:
            stack.append(frame_t)
        if len(stack) == n_stacks:
            for i in range(n_stacks):
                stacked_feat[stack_count][input_dim * i:input_dim * (i + 1)] = stack[i]
            stack_count += 1
            for _ in range(n_skips):
                stack.pop(0)
    return stacked_feat


class Speech2Text(ModelBase):
    """Speech to text sequence-to-sequence model."""

    def __init__(self, args, save_path=None, idx2token=None):
        super(ModelBase, self).__init__()
        self.save_path = save_path
        self.input_type = args.input_type
        self.input_dim = args.input_dim
        self.enc_type = args.enc_type
        self.enc_n_units = args.enc_n_units
        if args.enc_type in ['blstm', 'bgru', 'conv_blstm', 'conv_bgru']:
            self.enc_n_units *= 2
        self.dec_type = args.dec_type
        self.enc_n_layers = args.enc_n_layers
        self.enc_n_layers_sub1 = args.enc_n_layers_sub1
        self.subsample = [int(s) for s in args.subsample.split('_')]
        self.vocab = args.vocab
        self.vocab_sub1 = args.vocab_sub1
        self.vocab_sub2 = args.vocab_sub2
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3
        self.main_weight = 1 - args.sub1_weight - args.sub2_weight
        self.sub1_weight = args.sub1_weight
        self.sub2_weight = args.sub2_weight
        self.mtl_per_batch = args.mtl_per_batch
        self.task_specific_layer = args.task_specific_layer
        self.ctc_weight = min(args.ctc_weight, self.main_weight)
        self.ctc_weight_sub1 = min(args.ctc_weight_sub1, self.sub1_weight)
        self.ctc_weight_sub2 = min(args.ctc_weight_sub2, self.sub2_weight)
        self.bwd_weight = min(args.bwd_weight, self.main_weight)
        self.fwd_weight = self.main_weight - self.bwd_weight - self.ctc_weight
        self.fwd_weight_sub1 = self.sub1_weight - self.ctc_weight_sub1
        self.fwd_weight_sub2 = self.sub2_weight - self.ctc_weight_sub2
        self.mbr_training = args.mbr_training
        self.recog_params = vars(args)
        self.idx2token = idx2token
        self.utt_id_prev = None
        self.gaussian_noise = args.gaussian_noise
        self.n_stacks = args.n_stacks
        self.n_skips = args.n_skips
        self.n_splices = args.n_splices
        self.use_specaug = args.n_freq_masks > 0 or args.n_time_masks > 0
        self.specaug = None
        self.flip_time_prob = args.flip_time_prob
        self.flip_freq_prob = args.flip_freq_prob
        self.weight_noise = args.weight_noise
        if self.use_specaug:
            assert args.n_stacks == 1 and args.n_skips == 1
            assert args.n_splices == 1
            self.specaug = SpecAugment(F=args.freq_width, T=args.time_width, n_freq_masks=args.n_freq_masks, n_time_masks=args.n_time_masks, p=args.time_width_upper)
        self.ssn = None
        if args.sequence_summary_network:
            assert args.input_type == 'speech'
            self.ssn = SequenceSummaryNetwork(args.input_dim, n_units=512, n_layers=3, bottleneck_dim=100, dropout=0, param_init=args.param_init)
        self.enc = build_encoder(args)
        if args.freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False
        external_lm = None
        directions = []
        if self.fwd_weight > 0 or self.bwd_weight == 0 and self.ctc_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            if args.external_lm and dir == 'fwd':
                external_lm = RNNLM(args.lm_conf)
                load_checkpoint(external_lm, args.external_lm)
                for n, p in external_lm.named_parameters():
                    p.requires_grad = False
            special_symbols = {'blank': self.blank, 'unk': self.unk, 'eos': self.eos, 'pad': self.pad}
            dec = build_decoder(args, special_symbols, self.enc.output_dim, args.vocab, self.ctc_weight, args.ctc_fc_list, self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight, external_lm)
            setattr(self, 'dec_' + dir, dec)
        for sub in ['sub1', 'sub2']:
            if getattr(self, sub + '_weight') > 0:
                dec_sub = build_decoder(args, special_symbols, self.enc.output_dim, getattr(self, 'vocab_' + sub), getattr(self, 'ctc_weight_' + sub), getattr(args, 'ctc_fc_list_' + sub), getattr(self, sub + '_weight'), external_lm)
                setattr(self, 'dec_fwd_' + sub, dec_sub)
        if args.input_type == 'text':
            if args.vocab == args.vocab_sub1:
                self.embed = dec.embed
            else:
                self.embed = nn.Embedding(args.vocab_sub1, args.emb_dim, padding_idx=self.pad)
                self.dropout_emb = nn.Dropout(p=args.dropout_emb)
        if args.rec_weight_orthogonal:
            self.reset_parameters(args.param_init, dist='orthogonal', keys=['rnn', 'weight'])
        if args.lm_fusion == 'deep' and external_lm is not None:
            for n, p in self.named_parameters():
                if 'output' in n or 'output_bn' in n or 'linear' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def scheduled_sampling_trigger(self):
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).start_scheduled_sampling()
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).start_scheduled_sampling()

    def reset_session(self):
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).reset_session()
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).reset_session()

    def forward(self, batch, task='all', is_eval=False, teacher=None, teacher_lm=None):
        """Forward computation.

        Args:
            batch (dict):
                xs (list): input data of size `[T, input_dim]`
                xlens (list): lengths of each element in xs
                ys (list): reference labels in the main task of size `[L]`
                ys_sub1 (list): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (list): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (list): name of utterances
                speakers (list): name of speakers
            task (str): all/ys*/ys_sub*
            is_eval (bool): evaluation mode
                This should be used in inference model for memory efficiency.
            teacher (Speech2Text): used for knowledge distillation from ASR
            teacher_lm (RNNLM): used for knowledge distillation from LM
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(batch, task)
        else:
            self.train()
            loss, observation = self._forward(batch, task, teacher, teacher_lm)
        return loss, observation

    def _forward(self, batch, task, teacher=None, teacher_lm=None):
        if self.input_type == 'speech':
            if self.mtl_per_batch:
                eout_dict = self.encode(batch['xs'], task)
            else:
                eout_dict = self.encode(batch['xs'], 'all')
        else:
            eout_dict = self.encode(batch['ys_sub1'])
        observation = {}
        loss = torch.zeros((1,), dtype=torch.float32)
        if self.device_id >= 0:
            loss = loss
        if (self.fwd_weight > 0 or self.bwd_weight == 0 and self.ctc_weight > 0 or self.mbr_training) and task in ['all', 'ys', 'ys.ctc', 'ys.mbr']:
            teacher_logits = None
            if teacher is not None:
                teacher.eval()
                teacher_logits = teacher.generate_logits(batch)
            elif teacher_lm is not None:
                teacher_lm.eval()
                teacher_logits = self.generate_lm_logits(batch['ys'], lm=teacher_lm)
            loss_fwd, obs_fwd = self.dec_fwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'], task, teacher_logits, self.recog_params, self.idx2token)
            loss += loss_fwd
            if isinstance(self.dec_fwd, RNNTransducer) or isinstance(self.dec_fwd, TrasformerTransducer):
                observation['loss.transducer'] = obs_fwd['loss_transducer']
            else:
                observation['loss.att'] = obs_fwd['loss_att']
                observation['loss.mbr'] = obs_fwd['loss_mbr']
                if 'loss_quantity' not in obs_fwd.keys():
                    obs_fwd['loss_quantity'] = None
                observation['loss.quantity'] = obs_fwd['loss_quantity']
                if 'loss_headdiv' not in obs_fwd.keys():
                    obs_fwd['loss_headdiv'] = None
                observation['loss.headdiv'] = obs_fwd['loss_headdiv']
                if 'loss_latency' not in obs_fwd.keys():
                    obs_fwd['loss_latency'] = None
                observation['loss.latency'] = obs_fwd['loss_latency']
                observation['acc.att'] = obs_fwd['acc_att']
                observation['ppl.att'] = obs_fwd['ppl_att']
            observation['loss.ctc'] = obs_fwd['loss_ctc']
        if self.bwd_weight > 0 and task in ['all', 'ys.bwd']:
            loss_bwd, obs_bwd = self.dec_bwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'], task)
            loss += loss_bwd
            observation['loss.att-bwd'] = obs_bwd['loss_att']
            observation['acc.att-bwd'] = obs_bwd['acc_att']
            observation['ppl.att-bwd'] = obs_bwd['ppl_att']
            observation['loss.ctc-bwd'] = obs_bwd['loss_ctc']
        for sub in ['sub1', 'sub2']:
            if (getattr(self, 'fwd_weight_' + sub) > 0 or getattr(self, 'ctc_weight_' + sub) > 0) and task in ['all', 'ys_' + sub, 'ys_' + sub + '.ctc']:
                loss_sub, obs_fwd_sub = getattr(self, 'dec_fwd_' + sub)(eout_dict['ys_' + sub]['xs'], eout_dict['ys_' + sub]['xlens'], batch['ys_' + sub], task)
                loss += loss_sub
                if isinstance(getattr(self, 'dec_fwd_' + sub), RNNTransducer):
                    observation['loss.transducer-' + sub] = obs_fwd_sub['loss_transducer']
                else:
                    observation['loss.att-' + sub] = obs_fwd_sub['loss_att']
                    observation['acc.att-' + sub] = obs_fwd_sub['acc_att']
                    observation['ppl.att-' + sub] = obs_fwd_sub['ppl_att']
                observation['loss.ctc-' + sub] = obs_fwd_sub['loss_ctc']
        return loss, observation

    def generate_logits(self, batch, temperature=1.0):
        if self.input_type == 'speech':
            eout_dict = self.encode(batch['xs'], task='ys')
        else:
            eout_dict = self.encode(batch['ys_sub1'], task='ys')
        logits = self.dec_fwd.forward_att(eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'], return_logits=True)
        return logits

    def generate_lm_logits(self, ys, lm, temperature=5.0):
        eos = next(lm.parameters()).new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id) for y in ys]
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        lmout, _ = lm.decode(ys_in, None)
        logits = lm.output(lmout)
        return logits

    def encode(self, xs, task='all', use_cache=False, streaming=False):
        """Encode acoustic or text features.

        Args:
            xs (list): A list of length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all/ys*/ys_sub1*/ys_sub2*
            use_cache (bool): use the cached forward encoder state in the previous chunk as the initial state
            streaming (bool): streaming encoding
        Returns:
            eout_dict (dict):

        """
        if self.input_type == 'speech':
            if self.n_stacks > 1:
                xs = [stack_frame(x, self.n_stacks, self.n_skips) for x in xs]
            if self.n_splices > 1:
                xs = [splice(x, self.n_splices, self.n_stacks) for x in xs]
            xlens = torch.IntTensor([len(x) for x in xs])
            xs = pad_list([np2tensor(x, self.device_id).float() for x in xs], 0.0)
            if self.use_specaug and self.training:
                xs = self.specaug(xs)
                if self.weight_noise:
                    self.add_weight_noise(std=0.075)
            if self.gaussian_noise:
                xs = add_gaussian_noise(xs)
            if self.ssn is not None:
                xs += self.ssn(xs, xlens)
        elif self.input_type == 'text':
            xlens = torch.IntTensor([len(x) for x in xs])
            xs = [np2tensor(np.fromiter(x, dtype=np.int64), self.device_id) for x in xs]
            xs = pad_list(xs, self.pad)
            xs = self.dropout_emb(self.embed(xs))
        eout_dict = self.enc(xs, xlens, task.split('.')[0], use_cache, streaming)
        if self.main_weight < 1 and self.enc_type in ['conv', 'tds', 'gated_conv', 'transformer', 'conv_transformer']:
            for sub in ['sub1', 'sub2']:
                eout_dict['ys_' + sub]['xs'] = eout_dict['ys']['xs'].clone()
                eout_dict['ys_' + sub]['xlens'] = eout_dict['ys']['xlens'][:]
        return eout_dict

    def get_ctc_probs(self, xs, task='ys', temperature=1, topk=None):
        self.eval()
        with torch.no_grad():
            eout_dict = self.encode(xs, task)
            dir = 'fwd' if self.fwd_weight >= self.bwd_weight else 'bwd'
            if task == 'ys_sub1':
                dir += '_sub1'
            elif task == 'ys_sub2':
                dir += '_sub2'
            if task == 'ys':
                assert self.ctc_weight > 0
            elif task == 'ys_sub1':
                assert self.ctc_weight_sub1 > 0
            elif task == 'ys_sub2':
                assert self.ctc_weight_sub2 > 0
            ctc_probs, indices_topk = getattr(self, 'dec_' + dir).ctc_probs_topk(eout_dict[task]['xs'], temperature, topk)
            return tensor2np(ctc_probs), tensor2np(indices_topk), eout_dict[task]['xlens']

    def plot_attention(self):
        if 'transformer' in self.enc_type:
            self.enc._plot_attention(self.save_path)
        if 'transformer' in self.dec_type or 'transducer' not in self.dec_type:
            self.dec_fwd._plot_attention(self.save_path)

    def decode_streaming(self, xs, params, idx2token, exclude_eos=False, task='ys'):
        assert task == 'ys'
        assert self.input_type == 'speech'
        assert self.ctc_weight > 0
        assert self.fwd_weight > 0
        assert len(xs) == 1
        global_params = copy.deepcopy(params)
        global_params['recog_max_len_ratio'] = 1.0
        ctc_vad = params['recog_ctc_vad']
        BLANK_THRESHOLD = params['recog_ctc_vad_blank_threshold']
        SPIKE_THRESHOLD = params['recog_ctc_vad_spike_threshold']
        MAX_N_ACCUM_FRAMES = params['recog_ctc_vad_n_accum_frames']
        N_l = self.enc.lc_chunk_size_left
        N_r = self.enc.lc_chunk_size_right
        if N_l == 0 and N_r == 0:
            N_l = params['recog_lc_chunk_size_left']
        factor = self.enc.subsampling_factor
        BLANK_THRESHOLD /= factor
        x_whole = xs[0]
        if self.enc.conv is not None:
            self.enc.turn_off_ceil_mode(self.enc)
        self.eval()
        with torch.no_grad():
            lm = getattr(self, 'lm_fwd', None)
            lm_second = getattr(self, 'lm_second', None)
            eout_chunks = []
            ctc_probs_chunks = []
            t = 0
            n_blanks = 0
            n_accum_frames = 0
            boundary_offset = -1
            is_reset = True
            hyps = None
            best_hyp_id_stream = []
            while True:
                if self.enc.conv is not None:
                    x_chunk = x_whole[max(0, t - 1):t + (N_l + N_r) + 1]
                else:
                    x_chunk = x_whole[t:t + (N_l + N_r)]
                is_last_chunk = t + N_l >= len(x_whole) - 1
                eout_dict_chunk = self.encode([x_chunk], task, use_cache=not is_reset, streaming=True)
                eout_chunk = eout_dict_chunk[task]['xs']
                boundary_offset = -1
                is_reset = False
                n_accum_frames += eout_chunk.size(1) * factor
                ctc_log_probs_chunk = None
                if ctc_vad:
                    ctc_probs_chunk = self.dec_fwd.ctc_probs(eout_chunk)
                    if params['recog_ctc_weight'] > 0:
                        ctc_log_probs_chunk = torch.log(ctc_probs_chunk)
                    if n_accum_frames >= MAX_N_ACCUM_FRAMES:
                        _, topk_ids_chunk = torch.topk(ctc_probs_chunk, k=1, dim=-1, largest=True, sorted=True)
                        ctc_probs_chunks.append(ctc_probs_chunk)
                        for j in range(ctc_probs_chunk.size(1)):
                            if topk_ids_chunk[0, j, 0] == self.blank:
                                n_blanks += 1
                            elif ctc_probs_chunk[0, j, topk_ids_chunk[0, j, 0]] < SPIKE_THRESHOLD:
                                n_blanks += 1
                            else:
                                n_blanks = 0
                            if not is_reset and n_blanks > BLANK_THRESHOLD:
                                boundary_offset = j
                                is_reset = True
                if is_reset and not is_last_chunk:
                    eout_chunk = eout_chunk[:, :boundary_offset + 1]
                eout_chunks.append(eout_chunk)
                if params['recog_chunk_sync']:
                    end_hyps, hyps, aws_seg = self.dec_fwd.beam_search_chunk_sync(eout_chunk, params, idx2token, lm, lm_second, ctc_log_probs=ctc_log_probs_chunk, hyps=hyps, state_carry_over=False, ignore_eos=self.enc.rnn_type in ['lstm', 'conv_lstm'])
                    merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'], reverse=True)
                    best_hyp_id_prefix = np.array(merged_hyps[0]['hyp'][1:])
                    if len(best_hyp_id_prefix) > 0 and best_hyp_id_prefix[-1] == self.eos:
                        best_hyp_id_prefix = best_hyp_id_prefix[:-1]
                        if not is_reset:
                            boundary_offset = eout_chunk.size(1) - 1
                            is_reset = True
                if is_reset:
                    if not params['recog_chunk_sync']:
                        eout = torch.cat(eout_chunks, dim=1)
                        elens = torch.IntTensor([eout.size(1)])
                        ctc_log_probs = None
                        if params['recog_ctc_weight'] > 0:
                            ctc_log_probs = torch.log(self.dec_fwd.ctc_probs(eout))
                        nbest_hyps_id_offline, _, _ = self.dec_fwd.beam_search(eout, elens, global_params, idx2token, lm, lm_second, ctc_log_probs=ctc_log_probs)
                    eout = torch.cat(eout_chunks, dim=1)
                    elens = torch.IntTensor([eout.size(1)])
                    ctc_log_probs = None
                    nbest_hyps_id_offline, _, _ = self.dec_fwd.beam_search(eout, elens, global_params, idx2token, lm, lm_second, ctc_log_probs=ctc_log_probs)
                    if not params['recog_chunk_sync']:
                        if len(nbest_hyps_id_offline[0][0]) > 0:
                            best_hyp_id_stream.extend(nbest_hyps_id_offline[0][0])
                    elif len(best_hyp_id_prefix) > 0:
                        best_hyp_id_stream.extend(best_hyp_id_prefix)
                    eout_chunks = []
                    ctc_probs_chunks = []
                    n_blanks = 0
                    n_accum_frames = 0
                    hyps = None
                    if not is_last_chunk and 0 <= boundary_offset * factor < N_l - 1:
                        t -= x_chunk[(boundary_offset + 1) * factor:N_l].shape[0]
                        self.dec_fwd.n_frames -= x_chunk[(boundary_offset + 1) * factor:N_l].shape[0] // factor
                t += N_l
                if is_last_chunk:
                    break
            if not params['recog_chunk_sync'] and len(eout_chunks) > 0:
                eout = torch.cat(eout_chunks, dim=1)
                elens = torch.IntTensor([eout.size(1)])
                nbest_hyps_id_offline, _, _ = self.dec_fwd.beam_search(eout, elens, global_params, idx2token, lm, lm_second, None)
                if len(nbest_hyps_id_offline[0][0]) > 0:
                    best_hyp_id_stream.extend(nbest_hyps_id_offline[0][0])
            if not is_reset and params['recog_chunk_sync'] and len(best_hyp_id_prefix) > 0:
                best_hyp_id_stream.extend(best_hyp_id_prefix)
            if len(best_hyp_id_stream) > 0:
                return [np.stack(best_hyp_id_stream, axis=0)], [None]
            else:
                return [[]], [None]

    def decode(self, xs, params, idx2token, exclude_eos=False, refs_id=None, refs=None, utt_ids=None, speakers=None, task='ys', ensemble_models=[]):
        """Decoding in the inference stage.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_dim]`
            params (dict): hyper-parameters for decoding
                beam_width (int): the size of beam
                min_len_ratio (float):
                max_len_ratio (float):
                len_penalty (float): length penalty
                cov_penalty (float): coverage penalty
                cov_threshold (float): threshold for coverage penalty
                lm_weight (float): the weight of RNNLM score
                resolving_unk (bool): not used (to make compatible)
                fwd_bwd_attention (bool):
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from best_hyps_id
            refs_id (list): gold token IDs to compute log likelihood
            refs (list): gold transcriptions
            utt_ids (list):
            speakers (list):
            task (str): ys* or ys_sub1* or ys_sub2*
            ensemble_models (list): list of Speech2Text classes
        Returns:
            best_hyps_id (list): A list of length `[B]`, which contains arrays of size `[L]`
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        if task.split('.')[0] == 'ys':
            dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
        elif task.split('.')[0] == 'ys_sub1':
            dir = 'fwd_sub1'
        elif task.split('.')[0] == 'ys_sub2':
            dir = 'fwd_sub2'
        else:
            raise ValueError(task)
        if self.utt_id_prev != utt_ids[0]:
            self.reset_session()
        self.utt_id_prev = utt_ids[0]
        self.eval()
        with torch.no_grad():
            if self.input_type == 'speech' and self.mtl_per_batch and 'bwd' in dir:
                eout_dict = self.encode(xs, task)
            else:
                eout_dict = self.encode(xs, task)
            if self.fwd_weight == 0 and self.bwd_weight == 0 or self.ctc_weight > 0 and params['recog_ctc_weight'] == 1:
                lm = getattr(self, 'lm_' + dir, None)
                lm_second = getattr(self, 'lm_second', None)
                lm_second_bwd = None
                best_hyps_id = getattr(self, 'dec_' + dir).decode_ctc(eout_dict[task]['xs'], eout_dict[task]['xlens'], params, idx2token, lm, lm_second, lm_second_bwd, 1, refs_id, utt_ids, speakers)
                return best_hyps_id, None
            elif params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
                best_hyps_id, aws = getattr(self, 'dec_' + dir).greedy(eout_dict[task]['xs'], eout_dict[task]['xlens'], params['recog_max_len_ratio'], idx2token, exclude_eos, refs_id, utt_ids, speakers)
            else:
                assert params['recog_batch_size'] == 1
                ctc_log_probs = None
                if params['recog_ctc_weight'] > 0:
                    ctc_log_probs = self.dec_fwd.ctc_log_probs(eout_dict[task]['xs'])
                if params['recog_fwd_bwd_attention']:
                    lm_fwd = getattr(self, 'lm_fwd', None)
                    lm_bwd = getattr(self, 'lm_bwd', None)
                    nbest_hyps_id_fwd, aws_fwd, scores_fwd = self.dec_fwd.beam_search(eout_dict[task]['xs'], eout_dict[task]['xlens'], params, idx2token, lm_fwd, None, lm_bwd, ctc_log_probs, params['recog_beam_width'], False, refs_id, utt_ids, speakers)
                    nbest_hyps_id_bwd, aws_bwd, scores_bwd, _ = self.dec_bwd.beam_search(eout_dict[task]['xs'], eout_dict[task]['xlens'], params, idx2token, lm_bwd, None, lm_fwd, ctc_log_probs, params['recog_beam_width'], False, refs_id, utt_ids, speakers)
                    best_hyps_id = fwd_bwd_attention(nbest_hyps_id_fwd, aws_fwd, scores_fwd, nbest_hyps_id_bwd, aws_bwd, scores_bwd, self.eos, params['recog_gnmt_decoding'], params['recog_length_penalty'], idx2token, refs_id)
                    aws = None
                else:
                    ensmbl_eouts, ensmbl_elens, ensmbl_decs = [], [], []
                    if len(ensemble_models) > 0:
                        for i_e, model in enumerate(ensemble_models):
                            if model.input_type == 'speech' and model.mtl_per_batch and 'bwd' in dir:
                                enc_outs_e = model.encode(xs, task)
                            else:
                                enc_outs_e = model.encode(xs, task)
                            ensmbl_eouts += [enc_outs_e[task]['xs']]
                            ensmbl_elens += [enc_outs_e[task]['xlens']]
                            ensmbl_decs += [getattr(model, 'dec_' + dir)]
                    lm = getattr(self, 'lm_' + dir, None)
                    lm_second = getattr(self, 'lm_second', None)
                    lm_bwd = getattr(self, 'lm_bwd' if dir == 'fwd' else 'lm_bwd', None)
                    nbest_hyps_id, aws, scores = getattr(self, 'dec_' + dir).beam_search(eout_dict[task]['xs'], eout_dict[task]['xlens'], params, idx2token, lm, lm_second, lm_bwd, ctc_log_probs, 1, exclude_eos, refs_id, utt_ids, speakers, ensmbl_eouts, ensmbl_elens, ensmbl_decs)
                    best_hyps_id = [hyp[0] for hyp in nbest_hyps_id]
            return best_hyps_id, aws


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CausalConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConcatSubsampler,
     lambda: ([], {'factor': 4, 'n_units': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1dSubsampler,
     lambda: ([], {'factor': 4, 'n_units': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CustomDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DropSubsampler,
     lambda: ([], {'factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), [4, 4]], {}),
     False),
    (LayerNorm2D,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 2, 2, 2])], {}),
     True),
    (LinearGLUBlock,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxpoolSubsampler,
     lambda: ([], {'factor': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiheadAttentionMechanism,
     lambda: ([], {'kdim': 4, 'qdim': 4, 'adim': 4, 'n_heads': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (NiN,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SubsampelBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'in_freq': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (XLPositionalEmbedding,
     lambda: ([], {'d_model': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4]), 0], {}),
     True),
]

class Test_hirofumi0810_neural_sp(_paritybench_base):
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

