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
args_common = _module
args_lm = _module
asr = _module
ctc_forced_align = _module
eval = _module
plot_attention = _module
plot_ctc = _module
train = _module
eval_utils = _module
lm = _module
plot_cache = _module
train = _module
model_name = _module
plot_utils = _module
train_utils = _module
datasets = _module
alignment = _module
build = _module
dataloader = _module
dataset = _module
sampler = _module
lm = _module
token_converter = _module
character = _module
phone = _module
word = _module
wordpiece = _module
utils = _module
evaluators = _module
accuracy = _module
character = _module
edit_distance = _module
phone = _module
ppl = _module
resolving_unk = _module
word = _module
wordpiece = _module
wordpiece_bleu = _module
models = _module
base = _module
criterion = _module
data_parallel = _module
gated_convlm = _module
lm_base = _module
rnnlm = _module
transformer_xl = _module
transformerlm = _module
modules = _module
attention = _module
causal_conv = _module
cif = _module
conformer_convolution = _module
gelu = _module
glu = _module
gmm_attention = _module
headdrop = _module
initialization = _module
mocha = _module
chunk_energy = _module
hma_test = _module
hma_train = _module
mocha = _module
mocha_test = _module
mocha_train = _module
monotonic_energy = _module
multihead_attention = _module
positional_embedding = _module
positionwise_feed_forward = _module
relative_multihead_attention = _module
softplus = _module
swish = _module
sync_bidir_multihead_attention = _module
transformer = _module
zoneout = _module
__init___ = _module
decoders = _module
beam_search = _module
ctc = _module
decoder_base = _module
fwd_bwd_attention = _module
las = _module
rnn_transducer = _module
transformer = _module
encoders = _module
conformer = _module
conformer_block = _module
conformer_block_v2 = _module
conv = _module
encoder_base = _module
gated_conv = _module
rnn = _module
subsampling = _module
tds = _module
transformer = _module
transformer_block = _module
utils = _module
frontends = _module
frame_stacking = _module
input_noise = _module
sequence_summary = _module
spec_augment = _module
splicing = _module
streaming = _module
speech2text = _module
torch_utils = _module
trainers = _module
lr_scheduler = _module
optimizer = _module
reporter = _module
setup = _module
test = _module
test_las_decoder = _module
test_rnn_transducer_decoder = _module
test_transformer_decoder = _module
test_conformer_encoder = _module
test_conv_encoder = _module
test_rnn_encoder = _module
test_rnn_encoder_streaming_chunkwise = _module
test_tds_encoder = _module
test_transformer_encoder = _module
test_transformer_encoder_streaming_chunkwise = _module
test_utils = _module
test_frame_stacking = _module
test_input_noise = _module
test_sequence_summary = _module
test_specaugment = _module
test_splicing = _module
test_streaming = _module
test_rnnlm = _module
test_transformer_xl_lm = _module
test_transformerlm = _module
test_attention = _module
test_causal_conv = _module
test_cif = _module
test_conformer_convolution = _module
test_gmm_attention = _module
test_mocha = _module
test_multihead_attention = _module
test_pointwise_feed_forward = _module
test_relative_multihead_attention = _module
test_zoneout = _module
compute_oov_rate = _module
concat_ref = _module
make_tsv = _module
map2phone = _module
text2dict = _module
trn2ctm = _module

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


import copy


import logging


import time


import torch


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


import functools


import numpy as np


from torch.utils.data import DataLoader


import pandas as pd


from torch.utils.data import Dataset


import random


import torch.nn as nn


from torch.nn.utils import vector_to_parameters


from torch.nn.utils import parameters_to_vector


import math


import torch.nn.functional as F


from torch.nn import DataParallel


from torch.nn.parallel.scatter_gather import gather


from collections import OrderedDict


import matplotlib


from itertools import groupby


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import warnings


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

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def define_name(dir_name, args):
        raise NotImplementedError

    def reset_parameters(self, param_init):
        raise NotImplementedError

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

    def add_weight_noise(self, std):
        """Add variational Gaussian noise to model parameters.

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

    def scatter(self, inputs, target_gpus, device_ids):
        if len(self.device_ids) <= 1:
            return [inputs], [target_gpus]

        def scatter_map(obj, i):
            if isinstance(obj, list) and len(obj) > 0:
                return [a[i] for a in zip(*([iter(obj)] * len(self.device_ids)))]
        inputs = inputs[0]
        try:
            res = [{k: scatter_map(v, i) for k, v in inputs.items()} for i in range(len(self.device_ids))]
        finally:
            scatter_map = None
        target_gpus = [target_gpus] * len(self.device_ids)
        return res, target_gpus

    def gather(self, outputs, output_device):
        n_returns = len(outputs[0])
        assert n_returns == 2
        n_gpus = len(outputs)
        losses = [output[0] for output in outputs]
        observation_avg = {k: (sum([output[1][k] for output in outputs]) / n_gpus) for k, v in outputs[0][1].items() if v is not None}
        return gather(losses, output_device, dim=self.dim).mean(), observation_avg


class CPUWrapperASR(nn.Module):

    def __init__(self, model):
        super(CPUWrapperASR, self).__init__()
        self.module = model

    def forward(self, batch, task, is_eval=False, teacher=None, teacher_lm=None):
        return self.module(batch, task, is_eval, teacher, teacher_lm)


class CPUWrapperLM(nn.Module):

    def __init__(self, model):
        super(CPUWrapperLM, self).__init__()
        self.module = model

    def forward(self, ys, state=None, is_eval=False, n_caches=0, ylens=[], predict_last=False):
        return self.module(ys, state, is_eval, n_caches, ylens, predict_last)


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
        loss (FloatTensor): `[1]`
        ppl (float): perplexity

    """
    bs, _, vocab = logits.size()
    ys = ys.view(-1)
    logits = logits.view((-1, vocab))
    if lsm_prob == 0 or not training:
        loss = F.cross_entropy(logits, ys, ignore_index=ignore_index, reduction='mean')
        ppl = np.exp(loss.item())
        if not normalize_length:
            loss *= (ys != ignore_index).sum() / float(bs)
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


def np2tensor(array, device=None):
    """Convert form np.ndarray to torch.Tensor.

    Args:
        array (np.ndarray): A tensor of any sizes
    Returns:
        tensor (torch.Tensor):

    """
    tensor = torch.from_numpy(array)
    return tensor


def pad_list(xs, pad_value=0.0, pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which contains Tensors of size `[T, input_size]`
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
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
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
        """Forward pass.

        Args:
            ys (List): length `B`, each of which contains arrays of size `[L]`
            state (tuple or List):
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int): number of cached states
            ylens (List): not used
            predict_last (bool): used for TransformerLM and GatedConvLM
        Returns:
            loss (FloatTensor): `[1]`
            new_state (tuple or List):
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
        ys = [np2tensor(y, self.device) for y in ys]
        ys = pad_list(ys, self.pad)
        ys_in, ys_out = ys[:, :-1], ys[:, 1:]
        logits, out, new_state = self.decode(ys_in, state=state, mems=state)
        if predict_last:
            ys_out = ys_out[:, -1].unsqueeze(1)
            logits = logits[:, -1].unsqueeze(1)
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
                cache_probs[:, :, idx] += cache_attn[:, offset]
            probs = (1 - self.cache_lambda) * probs + self.cache_lambda * cache_probs
            loss = -torch.log(probs[:, :, ys_out[:, -1]])
        elif self.adaptive_softmax is None:
            loss, ppl = cross_entropy_lsm(logits, ys_out.contiguous(), self.lsm_prob, self.pad, self.training, normalize_length=True)
        else:
            loss = self.adaptive_softmax(logits.reshape((-1, logits.size(2))), ys_out.contiguous().view(-1)).loss
            ppl = np.exp(loss.item())
        if n_caches > 0:
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [out]
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(logits.reshape((-1, logits.size(2)))), ys_out, pad=self.pad)
        observation = {'loss.lm': loss.item(), 'acc.lm': acc, 'ppl.lm': ppl}
        return loss, new_state, observation

    def repackage_state(self, state):
        return state

    def reset_length(self, mem_len):
        self.mem_len = mem_len

    def decode(self, ys, state=None, mems=None, incremental=False):
        raise NotImplementedError

    def embed_token_id(self, indices):
        raise NotImplementedError

    def cache_embedding(self, device):
        """Cache token emebdding."""
        if self.embed_cache is None:
            indices = torch.arange(0, self.vocab, 1, dtype=torch.int64)
            self.embed_cache = self.embed_token_id(indices)

    def predict(self, ys, state=None, mems=None, cache=None):
        """Precict function for ASR.

        Args:
            ys (LongTensor): `[B, L]`
            state:
                - RNNLM => (dict):
                    hxs (FloatTensor): `[n_layers, B, n_units]`
                    cxs (FloatTensor): `[n_layers, B, n_units]`
                - TransformerLM => (LongTensor): `[B, L]`
                - TransformerXL => (List): length `n_layers + 1`, each of which contains a tensor`[B, L, d_model]`
            mems (List):
            cache (List):
        Returns:
            lmout (FloatTensor): `[B, L, vocab]`, used for LM integration such as cold fusion
            state:
                - RNNLM (dict):
                    hxs (FloatTensor): `[n_layers, B, n_units]`
                    cxs (FloatTensor): `[n_layers, B, n_units]`
                - TransformerLM => (LongTensor): `[B, L]`
                - TransformerXL => (List): length `n_layers + 1`, each of which contains a tensor`[B, L, d_model]`
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
        idim (int): input and output dimension

    """

    def __init__(self, idim):
        super().__init__()
        self.fc = nn.Linear(idim, idim * 2)

    def forward(self, xs):
        return F.glu(self.fc(xs), dim=-1)


def repeat(module, n_layers):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


class RNNLM(LMBase):
    """RNN language model."""

    def __init__(self, args, save_path=None):
        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)
        self.save_path = save_path
        self.emb_dim = args.emb_dim
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
        self.embed_cache = None
        self.embed = nn.Embedding(self.vocab, args.emb_dim, padding_idx=self.pad)
        self.dropout_emb = nn.Dropout(p=args.dropout_in)
        self.rnn = nn.ModuleList()
        self.dropout = nn.Dropout(p=args.dropout_hidden)
        if args.n_projs > 0:
            self.proj = repeat(nn.Linear(args.n_units, args.n_projs), args.n_layers)
        rnn_idim = args.emb_dim + args.n_units_null_context
        for _ in range(args.n_layers):
            self.rnn += [nn.LSTM(rnn_idim, args.n_units, 1, batch_first=True)]
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

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('RNNLM')
        group.add_argument('--n_units', type=int, default=1024, help='number of units in each layer')
        group.add_argument('--n_projs', type=int, default=0, help='number of units in the projection layer')
        group.add_argument('--residual', type=strtobool, default=False, help='')
        group.add_argument('--use_glu', type=strtobool, default=False, help='use Gated Linear Unit (GLU) for fully-connected layers')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name = args.lm_type
        dir_name += str(args.n_units) + 'H'
        dir_name += str(args.n_projs) + 'P'
        dir_name += str(args.n_layers) + 'L'
        dir_name += '_emb' + str(args.emb_dim)
        if args.tie_embedding:
            dir_name += '_tie'
        if args.adaptive_softmax:
            dir_name += '_adaptiveSM'
        if args.residual:
            dir_name += '_residual'
        if args.use_glu:
            dir_name += '_glu'
        if args.n_units_null_context > 0:
            dir_name += '_nullcv' + str(args.n_units_null_context)
        return dir_name

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

    def embed_token_id(self, indices):
        """Embed token IDs.

        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.dropout_emb(self.embed(indices))
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def decode(self, ys, state, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            cache: dummy interfance for TransformerLM/TransformerXL
            incremental: dummy interfance for TransformerLM/TransformerXL
            cache_emb (bool): precompute token embeddings
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            ys_emb (FloatTensor): `[B, L, n_units]` (for cache)
            new_state (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            new_mems: dummy interfance for TransformerXL

        """
        bs, ymax = ys.size()
        ys_emb = self.embed_token_id(ys)
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
            ys_emb, (h, c) = self.rnn[lth](ys_emb, hx=(state['hxs'][lth:lth + 1], state['cxs'][lth:lth + 1]))
            new_hxs.append(h)
            new_cxs.append(c)
            ys_emb = self.dropout(ys_emb)
            if self.n_projs > 0:
                ys_emb = torch.tanh(self.proj[lth](ys_emb))
            if self.residual and lth > 0:
                ys_emb = ys_emb + residual
            residual = ys_emb
        new_state['hxs'] = torch.cat(new_hxs, dim=0)
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
        state['cxs'] = state['cxs'].detach()
        return state


class ChunkEnergy(nn.Module):
    """Energy function for the chunkwise attention.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of quary
        adim (int): dimension of attention space
        atype (str): type of attention mechanism
        n_heads (int): number of chunkwise attention heads
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method

    """

    def __init__(self, kdim, qdim, adim, atype, n_heads=1, bias=True, param_init=''):
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
            self.v = nn.Linear(adim, n_heads, bias=False)
        elif atype == 'scaled_dot':
            self.w_key = nn.Linear(kdim, adim, bias=bias)
            self.w_query = nn.Linear(qdim, adim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters_xavier_uniform(bias)
        else:
            raise NotImplementedError(atype)

    def reset_parameters_xavier_uniform(self, bias):
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

    def forward(self, key, query, mask, cache=False, boundary_leftmost=0, boundary_rightmost=100000):
        """Compute chunkwise energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
            boundary_leftmost (int): leftmost boundary offset
            boundary_rightmost (int): rightmost boundary offset
        Returns:
            e (FloatTensor): `[B, H_ca, qlen, klen]`

        """
        klen, kdim = key.size()[1:]
        bs, qlen = query.size()[:2]
        if self.key is None or not cache:
            self.key = self.w_key(key).view(-1, klen, self.n_heads, self.d_k)
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                mask_size = bs, qlen, klen, self.n_heads
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None
        k = self.key
        if k.size(0) != bs:
            k = k[0:1].repeat([bs, 1, 1, 1])
        klen = k.size(1)
        q = self.w_query(query).view(-1, qlen, self.n_heads, self.d_k)
        m = self.mask
        if boundary_leftmost > 0 or 0 <= boundary_rightmost < klen:
            k = k[:, boundary_leftmost:boundary_rightmost + 1]
            klen = k.size(1)
            if m is not None:
                m = m[:, :, boundary_leftmost:boundary_rightmost + 1]
        if self.atype == 'scaled_dot':
            e = torch.einsum('bihd,bjhd->bijh', (q, k)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.relu(k[:, None] + q[:, :, None]).view(bs, qlen, klen, -1))
        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(m == 0, NEG_INF)
        e = e.permute(0, 3, 1, 2)
        return e


def init_with_lecun_normal(n, p, param_init):
    """Initialize with Lecun style.

    Args:
        n (str): parameter name
        p (Tensor): parameter
        param_init (float):

    """
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


def init_with_xavier_uniform(n, p):
    """Initialize with Xavier uniform distribution.

    Args:
        n (str): parameter name
        p (Tensor): parameter

    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() in [2, 3, 4]:
        nn.init.xavier_uniform_(p)
        logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
    else:
        raise ValueError(n)


class MonotonicEnergy(nn.Module):
    """Energy function for the monotonic attention.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of quary
        adim (int): dimension of attention space
        atype (str): type of attention mechanism
        n_heads (int): number of monotonic attention heads
        init_r (int): initial value for offset r
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        conv1d (bool): use 1D causal convolution for energy calculation
        conv_kernel_size (int): kernel size for 1D convolution

    """

    def __init__(self, kdim, qdim, adim, atype, n_heads, init_r, bias=True, param_init='', conv1d=False, conv_kernel_size=5):
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
            self.conv1d = nn.Conv1d(kdim, kdim, conv_kernel_size, padding=(conv_kernel_size - 1) // 2)
            for n, p in self.conv1d.named_parameters():
                init_with_lecun_normal(n, p, 0.1)
        if atype == 'add':
            self.v = nn.utils.weight_norm(self.v, name='weight', dim=0)
            self.v.weight_g.data = torch.Tensor([1 / adim]).sqrt()
        elif atype == 'scaled_dot':
            if param_init == 'xavier_uniform':
                self.reset_parameters_xavier_uniform(bias)

    def reset_parameters_xavier_uniform(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.0)
            nn.init.constant_(self.w_query.bias, 0.0)
        if self.conv1d is not None:
            for n, p in self.conv1d.named_parameters():
                init_with_xavier_uniform(n, p)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, query, mask, cache=False, boundary_leftmost=0, boundary_rightmost=100000):
        """Compute monotonic energy.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            cache (bool): cache key and mask
            boundary_leftmost (int): leftmost boundary offset
            boundary_rightmost (int): rightmost boundary offset
        Returns:
            e (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        klen, kdim = key.size()[1:]
        bs, qlen = query.size()[:2]
        if self.key is None or not cache:
            if self.conv1d is not None:
                key = torch.relu(self.conv1d(key.transpose(2, 1))).transpose(2, 1)
            self.key = self.w_key(key)
            self.key = self.key.view(-1, klen, self.n_heads, self.d_k)
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                mask_size = bs, qlen, klen, self.n_heads
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None
        k = self.key
        if k.size(0) != bs:
            k = k[0:1].repeat([bs, 1, 1, 1])
        klen = k.size(1)
        q = self.w_query(query).view(-1, qlen, self.n_heads, self.d_k)
        m = self.mask
        if boundary_leftmost > 0:
            k = k[:, boundary_leftmost:]
            klen = k.size(1)
            if m is not None:
                m = m[:, :, boundary_leftmost:]
        if self.atype == 'scaled_dot':
            e = torch.einsum('bihd,bjhd->bijh', (q, k)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.relu(k[:, None] + q[:, :, None]).view(bs, qlen, klen, -1))
        if self.r is not None:
            e = e + self.r
        if m is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(m == 0, NEG_INF)
        e = e.permute(0, 3, 1, 2)
        return e


def hard_chunkwise_attention(alpha, u, mask, chunk_size, H_ca, sharpening_factor, share_chunkwise_attention):
    """Chunkwise attention in MoChA at test time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        H_ca (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, H_ma, qlen, klen = alpha.size()
    assert u.size(2) == qlen and u.size(3) == klen, (u.size(), alpha.size())
    alpha = alpha.unsqueeze(2)
    u = u.unsqueeze(1)
    if H_ca > 1:
        alpha = alpha.repeat([1, 1, H_ca, 1, 1])
    if H_ma > 1:
        if share_chunkwise_attention:
            u = u.repeat([1, H_ma, 1, 1, 1])
        else:
            u = u.view(bs, H_ma, H_ca, qlen, klen)
    mask = alpha.clone().byte()
    for b in range(bs):
        for h in range(H_ma):
            if alpha[b, h, 0, 0].sum() > 0:
                boundary = alpha[b, h, 0, 0].nonzero()[:, -1].min().item()
                if chunk_size == -1:
                    mask[b, h, :, 0, 0:boundary + 1] = 1
                else:
                    mask[b, h, :, 0, max(0, boundary - chunk_size + 1):boundary + 1] = 1
    NEG_INF = float(np.finfo(torch.tensor(0, dtype=u.dtype).numpy().dtype).min)
    u = u.masked_fill(mask == 0, NEG_INF)
    beta = torch.softmax(u, dim=-1)
    return beta.view(bs, -1, qlen, klen)


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), x.size(1), x.size(2), 1), x[:, :, :, :-1]], dim=-1), dim=-1)


def hard_monotonic_attention(e_ma, aw_prev, eps_wait, p_threshold=0.5):
    """Monotonic attention in MoChA at test time.

    Args:
        e_ma (FloatTensor): `[B, H_ma, qlen, klen]`
        aw_prev (FloatTensor): `[B, H_ma, qlen, klen]`
        eps_wait (int): wait time delay for head-synchronous decoding in MMA
        p_threshold (float): threshold for p_choose during at test time
    Returns:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

    """
    bs, H_ma, qlen, klen = e_ma.size()
    assert qlen == 1
    assert e_ma.size(-1) == aw_prev.size(-1)
    aw_prev = aw_prev[:, :, :, -klen:]
    _p_choose = torch.sigmoid(e_ma[:, :, 0:1])
    p_choose = (_p_choose >= p_threshold).float()
    p_choose *= torch.cumsum(aw_prev[:, :, 0:1, -e_ma.size(3):], dim=-1)
    alpha = p_choose * exclusive_cumprod(1 - p_choose)
    if eps_wait > 0:
        for b in range(bs):
            if alpha[b].sum() == 0:
                continue
            leftmost = alpha[b, :, -1].nonzero()[:, -1].min().item()
            rightmost = alpha[b, :, -1].nonzero()[:, -1].max().item()
            for h in range(H_ma):
                if alpha[b, h, -1].sum().item() == 0:
                    alpha[b, h, -1, min(rightmost, leftmost + eps_wait)] = 1
                    continue
                if alpha[b, h, -1].nonzero()[:, -1].min().item() >= leftmost + eps_wait:
                    alpha[b, h, -1, :] = 0
                    alpha[b, h, -1, leftmost + eps_wait] = 1
    return alpha, _p_choose


def headdrop(aws, n_heads, dropout):
    """HeadDrop regularization.

        Args:
            aws (FloatTensor): `[B, H, qlen, klen]`
            n_heads (int): number of attention heads
            dropout (float): HeadDrop probability
        Returns:
            aws (FloatTensor): `[B, H, qlen, klen]`

    """
    n_effective_heads = n_heads
    head_mask = aws.new_ones(aws.size()).byte()
    for h in range(n_heads):
        if random.random() < dropout:
            head_mask[:, h] = 0
            n_effective_heads -= 1
    aws = aws.masked_fill_(head_mask == 0, 0)
    if n_effective_heads > 0:
        aws = aws * (n_heads / n_effective_heads)
    return aws


def add_gaussian_noise(x, std):
    """Add Gaussian noise to encourage discreteness.

    Args:
        x (FloatTensor): `[B, H_ma, qlen, klen]`
        std (float): standard deviation
    Returns:
        x (FloatTensor): `[B, H_ma, qlen, klen]`

    """
    noise = x.new_zeros(x.size()).normal_(std=std)
    return x + noise


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


def parallel_monotonic_attention(e_ma, aw_prev, trigger_points, eps, noise_std, no_denom, decot, lookahead, stableemit_weight):
    """Efficient monotonic attention in MoChA at training time.

    Args:
        e_ma (FloatTensor): `[B, H_ma, qlen, klen]`
        aw_prev (FloatTensor): `[B, H_ma, qlen, klen]`
        trigger_points (IntTensor): `[B, qlen]`
        eps (float): epsilon parameter to avoid zero division
        noise_std (float): standard deviation for Gaussian noise
        no_denom (bool): set the denominator to 1 in the alpha recurrence
        decot (bool): delay constrainted training (DeCoT)
        lookahead (int): lookahead frames for DeCoT
        stableemit_weight (float): StableEmit weight
    Returns:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

    """
    bs, H_ma, qlen, klen = e_ma.size()
    aw_prev = aw_prev[:, :, :, :klen]
    if decot:
        aw_prev_pad = aw_prev.new_zeros(bs, H_ma, qlen, klen)
        aw_prev_pad[:, :, :, :aw_prev.size(3)] = aw_prev
        aw_prev = aw_prev_pad
    bs, H_ma, qlen, klen = e_ma.size()
    p_choose = torch.sigmoid(add_gaussian_noise(e_ma, noise_std))
    alpha = []
    if stableemit_weight > 0:
        p_choose = (1 - stableemit_weight) * p_choose
    cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=eps)
    for i in range(qlen):
        denom = 1 if no_denom else torch.clamp(cumprod_1mp_choose[:, :, i:i + 1], min=eps, max=1.0)
        cumsum_in = aw_prev / denom
        monotonic = False
        if monotonic and i > 0:
            cumsum_in = torch.cat([denom.new_zeros(bs, H_ma, 1, 1), cumsum_in[:, :, :, 1:]], dim=-1)
        aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :, i:i + 1] * torch.cumsum(cumsum_in, dim=-1)
        if decot:
            assert trigger_points is not None
            for b in range(bs):
                aw_prev[b, :, :, trigger_points[b, i:i + 1] + lookahead + 1:] = 0
        alpha.append(aw_prev)
    alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]
    return alpha, p_choose


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.

    Args:
        x (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`
        back (int): number of lookback frames
        forward (int): number of lookahead frames

    Returns:
        x_sum (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`

    """
    bs, n_heads_mono, n_heads_chunk, qlen, klen = x.size()
    x = x.reshape(-1, klen)
    x_padded = F.pad(x, pad=[back, forward])
    x_padded = x_padded.unsqueeze(1)
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum


def soft_chunkwise_attention(alpha, u, mask, chunk_size, H_ca, sharpening_factor, share_chunkwise_attention):
    """Chunkwise attention in MoChA at training time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        H_ca (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, H_ma, qlen, klen = alpha.size()
    alpha = alpha.unsqueeze(2)
    u = u.unsqueeze(1)
    if H_ca > 1:
        alpha = alpha.repeat([1, 1, H_ca, 1, 1])
    if H_ma > 1 and not share_chunkwise_attention:
        u = u.view(bs, H_ma, H_ca, qlen, klen)
    u -= torch.max(u, dim=-1, keepdim=True)[0]
    softmax_exp = torch.clamp(torch.exp(u), min=1e-05)
    if chunk_size == -1:
        softmax_denominators = torch.cumsum(softmax_exp, dim=-1)
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators, back=0, forward=klen - 1)
    else:
        softmax_denominators = moving_sum(softmax_exp, back=chunk_size - 1, forward=0)
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators, back=0, forward=chunk_size - 1)
    return beta.view(bs, -1, qlen, klen)


class MoChA(nn.Module):
    """Monotonic (multihead) chunkwise attention.

        if chunk_size == 1, this is equivalent to Hard monotonic attention
            "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
                https://arxiv.org/abs/1704.00784
        if chunk_size > 1, this is equivalent to monotonic chunkwise attention (MoChA)
            "Monotonic Chunkwise Attention" (ICLR 2018)
                https://openreview.net/forum?id=Hko85plCW
        if chunk_size == -1, this is equivalent to Monotonic infinite lookback attention (Milk)
            "Monotonic Infinite Lookback Attention for Simultaneous Machine Translation" (ACL 2019)
                https://arxiv.org/abs/1906.05218
        if chunk_size == 1 and n_heads_mono>1, this is equivalent to Monotonic Multihead Attention (MMA)-hard
            "Monotonic Multihead Attention" (ICLR 2020)
                https://openreview.net/forum?id=Hyg96gBKPS
        if chunk_size == -1 and n_heads_mono>1, this is equivalent to Monotonic Multihead Attention (MMA)-Ilk
            "Monotonic Multihead Attention" (ICLR 2020)
                https://openreview.net/forum?id=Hyg96gBKPS

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of the attention layer
        odim: (int) dimension of output
        atype (str): type of attention mechanism
        chunk_size (int): window size for chunkwise attention
        n_heads_mono (int): number of heads for monotonic attention
        n_heads_chunk (int): number of heads for chunkwise attention
        conv1d (bool): apply 1d convolution for energy calculation
        init_r (int): initial value for parameter 'r' used for monotonic attention
        eps (float): epsilon parameter to avoid zero division
        noise_std (float): standard deviation for Gaussian noise
        no_denominator (bool): set the denominator to 1 in the alpha recurrence
        sharpening_factor (float): sharping factor for beta calculation
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        decot (bool): delay constrainted training (DeCoT)
        decot_delta (int): tolerance frames for DeCoT
        share_chunkwise_attention (int): share CA heads among MA heads
        stableemit_weight (float): StableEmit weight for selection probability

    """

    def __init__(self, kdim, qdim, adim, odim, atype, chunk_size, n_heads_mono=1, n_heads_chunk=1, conv1d=False, init_r=-4, eps=1e-06, noise_std=1.0, no_denominator=False, sharpening_factor=1.0, dropout=0.0, dropout_head=0.0, bias=True, param_init='', decot=False, decot_delta=2, share_chunkwise_attention=False, stableemit_weight=0.0):
        super().__init__()
        self.atype = atype
        assert adim % (max(1, n_heads_mono) * n_heads_chunk) == 0
        self.d_k = adim // (max(1, n_heads_mono) * n_heads_chunk)
        self.w = chunk_size
        self.milk = chunk_size == -1
        self.n_heads = n_heads_mono
        self.H_ma = max(1, n_heads_mono)
        self.H_ca = n_heads_chunk
        self.H_total = self.H_ma * self.H_ca
        self.eps = eps
        self.noise_std = noise_std
        self.no_denom = no_denominator
        self.sharpening_factor = sharpening_factor
        self.decot = decot
        self.decot_delta = decot_delta
        self.share_ca = share_chunkwise_attention
        self.stableemit_weight = stableemit_weight
        assert stableemit_weight >= 0
        self._stableemit_weight = 0
        self.p_threshold = 0.5
        logger.info('stableemit_weight: %.3f' % stableemit_weight)
        if n_heads_mono >= 1:
            self.monotonic_energy = MonotonicEnergy(kdim, qdim, adim, atype, n_heads_mono, init_r, bias, param_init, conv1d=conv1d)
        else:
            self.monotonic_energy = None
            logger.info('monotonic attention is disabled.')
        if chunk_size > 1 or self.milk:
            self.chunk_energy = ChunkEnergy(kdim, qdim, adim, atype, n_heads_chunk if self.share_ca else self.H_ma * n_heads_chunk, bias, param_init)
        else:
            self.chunk_energy = None
        if self.H_ma * n_heads_chunk > 1:
            self.w_value = nn.Linear(kdim, adim, bias=bias)
            self.w_out = nn.Linear(adim, odim, bias=bias)
            if param_init == 'xavier_uniform':
                self.reset_parameters_xavier_uniform(bias)
        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head
        self.reset()

    def reset_parameters_xavier_uniform(self, bias):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_value.bias, 0.0)
        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.0)

    def reset(self):
        """Reset when a speaker changes."""
        if self.monotonic_energy is not None:
            self.monotonic_energy.reset()
        if self.chunk_energy is not None:
            self.chunk_energy.reset()
        self.bd_L_prev = 0
        self.key_tail = None

    def register_tail(self, key_tail):
        self.key_tail = key_tail

    def trigger_stableemit(self):
        logger.info('Activate StableEmit')
        self._stableemit_weight = self.stableemit_weight

    def set_p_choose_threshold(self, p):
        self.p_threshold = p

    def forward(self, key, value, query, mask, aw_prev=None, cache=False, mode='hard', trigger_points=None, eps_wait=-1, linear_decoding=False, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, H_ma, 1, klen]`
            cache (bool): cache key and mask
            mode (str): parallel/hard
            trigger_points (IntTensor): `[B, qlen]`
            eps_wait (int): wait time delay for head-synchronous decoding in MMA
            linear_decoding (bool): linear-time decoding mode
            streaming (bool): streaming mode (use self.key_tail)
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            alpha (FloatTensor): `[B, H_ma, qlen, klen]`
            attn_state (dict):
                beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`
                p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

        """
        klen = key.size(1)
        bs, qlen = query.size()[:2]
        tail_len = self.key_tail.size(1) if self.key_tail is not None else 0
        bd_L = self.bd_L_prev
        bd_R = klen - 1
        assert bd_L <= bd_R
        attn_state = {}
        if aw_prev is None:
            aw_prev = key.new_zeros(bs, self.H_ma, 1, klen)
            aw_prev[:, :, :, 0:1] = key.new_ones(bs, self.H_ma, 1, 1)
        e_ma = self.monotonic_energy(key, query, mask, cache, bd_L, bd_R)
        assert e_ma.size(3) + bd_L == klen, (e_ma.size(), self.bd_L_prev, key.size())
        if mode == 'parallel':
            alpha, p_choose = parallel_monotonic_attention(e_ma, aw_prev, trigger_points, self.eps, self.noise_std, self.no_denom, self.decot, self.decot_delta, self._stableemit_weight)
            if self.dropout_head > 0 and self.training:
                alpha_masked = headdrop(alpha.clone(), self.H_ma, self.dropout_head)
            else:
                alpha_masked = alpha.clone()
        elif mode == 'hard':
            aw_prev = aw_prev[:, :, :, -e_ma.size(3):]
            alpha, p_choose = hard_monotonic_attention(e_ma, aw_prev, eps_wait, self.p_threshold)
            alpha_masked = alpha.clone()
        else:
            raise ValueError("mode must be 'parallel' or 'hard'.")
        is_boundary = alpha.sum().item() > 0
        if linear_decoding and mode == 'hard' and is_boundary:
            bd_L = self.bd_L_prev + alpha[:, :, -1].nonzero()[:, -1].min().item()
            bd_R = self.bd_L_prev + alpha[:, :, -1].nonzero()[:, -1].max().item()
        bd_L_ca = max(0, bd_L + 1 - self.w) if not self.milk else 0
        use_tail = streaming and is_boundary and tail_len > 0
        beta = None
        if self.chunk_energy is not None:
            if mode == 'hard':
                if not is_boundary:
                    beta = alpha.new_zeros(bs, self.H_total, qlen, value.size(1))
                else:
                    if use_tail:
                        key = torch.cat([self.key_tail, key], dim=1)
                        bd_L += tail_len
                        bd_R += tail_len
                    bd_L_ca = max(0, bd_L + 1 - self.w) if not self.milk else 0
                    e_ca = self.chunk_energy(key, query, mask, cache, bd_L_ca, bd_R)
                    assert e_ca.size(3) == bd_R - bd_L_ca + 1, (e_ca.size(), bd_L_ca, bd_R, key.size())
                    if alpha_masked.size(3) < klen:
                        alpha_masked = torch.cat([alpha.new_zeros(bs, self.H_ma, qlen, klen - alpha_masked.size(3)), alpha_masked], dim=3)
                    if use_tail:
                        alpha_masked = torch.cat([alpha.new_zeros(bs, self.H_ma, qlen, tail_len), alpha_masked], dim=3)
                        value = torch.cat([self.key_tail[0:1], value[0:1]], dim=1)
                    alpha_masked = alpha_masked[:, :, :, bd_L_ca:bd_R + 1]
                    value = value[:, bd_L_ca:bd_R + 1]
                    beta = hard_chunkwise_attention(alpha_masked, e_ca, mask, self.w, self.H_ca, self.sharpening_factor, self.share_ca)
                    beta = self.dropout_attn(beta)
                    assert beta.size() == (bs, self.H_total, qlen, bd_R - bd_L_ca + 1), (beta.size(), (bs, self.H_total, qlen, bd_L_ca, bd_R))
            else:
                e_ca = self.chunk_energy(key, query, mask, cache, 0, bd_R)
                beta = soft_chunkwise_attention(alpha_masked, e_ca, mask, self.w, self.H_ca, self.sharpening_factor, self.share_ca)
                beta = self.dropout_attn(beta)
                assert beta.size() == (bs, self.H_total, qlen, klen), (beta.size(), (bs, self.H_total, qlen, klen))
        if value.size(0) != bs:
            value = value[0:1].repeat([bs, 1, 1])
        if self.H_total > 1:
            v = self.w_value(value).view(bs, -1, self.H_total, self.d_k)
            v = v.transpose(2, 1).contiguous()
            cv = torch.matmul(alpha_masked if self.w == 1 else beta, v)
            cv = cv.transpose(2, 1).contiguous().view(bs, -1, self.H_total * self.d_k)
            cv = self.w_out(cv)
        else:
            cv = torch.bmm(alpha_masked.squeeze(1) if self.w == 1 else beta.squeeze(1), value)
        if mode == 'hard' and use_tail:
            bd_L -= tail_len
            bd_R -= tail_len
            alpha_masked = alpha_masked[:, :, :, -klen:]
        self.bd_L_prev = bd_L
        if mode == 'hard':
            alpha = alpha.new_zeros(bs, alpha.size(1), qlen, klen)
            if is_boundary:
                alpha[:, :, :, bd_L:bd_R + 1] = alpha_masked[:, :, :, -(bd_R - bd_L + 1):]
        assert alpha.size() == (bs, self.H_ma, qlen, klen), (alpha.size(), (bs, self.H_ma, qlen, klen, bd_L, bd_R))
        attn_state['beta'] = beta
        attn_state['p_choose'] = p_choose
        return cv, alpha, attn_state


class TransformerDecoderBlock(nn.Module):
    """A single layer of the Transformer decoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        atype (str): type of attention mechanism
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention probabilities
        dropout_layer (float): LayerDrop probability
        dropout_head (float): HeadDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        src_tgt_attention (bool): use source-target attention
        memory_transformer (bool): TransformerXL decoder
        mma_chunk_size (int): chunk size for chunkwise attention. -1 means infinite lookback.
        mma_n_heads_mono (int): number of MMA head
        mma_n_heads_chunk (int): number of hard chunkwise attention head
        mma_init_r (int): initial bias value for MMA
        mma_eps (float): epsilon value for MMA
        mma_std (float): standard deviation of Gaussian noise for MMA
        mma_no_denominator (bool): remove denominator in MMA
        mma_1dconv (bool): 1dconv for MMA
        share_chunkwise_attention (bool): share chunkwise attention in the same layer of MMA
        lm_fusion (str): type of LM fusion
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer

    """

    def __init__(self, d_model, d_ff, atype, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, src_tgt_attention=True, memory_transformer=False, mma_chunk_size=0, mma_n_heads_mono=1, mma_n_heads_chunk=1, mma_init_r=2, mma_eps=1e-06, mma_std=1.0, mma_no_denominator=False, mma_1dconv=False, dropout_head=0, share_chunkwise_attention=False, lm_fusion='', ffn_bottleneck_dim=0):
        super().__init__()
        self.atype = atype
        self.n_heads = n_heads
        self.src_tgt_attention = src_tgt_attention
        self.memory_transformer = memory_transformer
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if memory_transformer else MHA
        self.self_attn = mha(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, dropout_head=dropout_head, param_init=param_init, xl_like=memory_transformer)
        if src_tgt_attention:
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            if 'mocha' in atype:
                self.n_heads = mma_n_heads_mono
                self.src_attn = MoChA(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, atype='scaled_dot', chunk_size=mma_chunk_size, n_heads_mono=mma_n_heads_mono, n_heads_chunk=mma_n_heads_chunk, init_r=mma_init_r, eps=mma_eps, noise_std=mma_std, no_denominator=mma_no_denominator, conv1d=mma_1dconv, dropout=dropout_att, dropout_head=dropout_head, param_init=param_init, share_chunkwise_attention=share_chunkwise_attention)
            else:
                self.src_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        else:
            self.src_attn = None
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init, ffn_bottleneck_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_layer = dropout_layer
        self.lm_fusion = lm_fusion
        if lm_fusion:
            self.norm_lm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.linear_lm_feat = nn.Linear(d_model, d_model)
            self.linear_lm_gate = nn.Linear(d_model * 2, d_model)
            self.linear_lm_fusion = nn.Linear(d_model * 2, d_model)
            if 'attention' in lm_fusion:
                self.lm_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.reset_visualization()

    @property
    def yy_aws(self):
        return self._yy_aws

    @property
    def xy_aws(self):
        return self._xy_aws

    @property
    def xy_aws_beta(self):
        return self._xy_aws_beta

    @property
    def xy_aws_p_choose(self):
        return self._xy_aws_p_choose

    @property
    def yy_aws_lm(self):
        return self._yy_aws_lm

    def reset_visualization(self):
        self._yy_aws = None
        self._xy_aws = None
        self._xy_aws_beta = None
        self._xy_aws_p_choose = None
        self._yy_aws_lm = None

    def reset(self):
        if self.src_attn is not None:
            self.src_attn.reset()

    def forward(self, ys, yy_mask, xs=None, xy_mask=None, cache=None, xy_aws_prev=None, mode='hard', eps_wait=-1, lmout=None, pos_embs=None, memory=None, u_bias=None, v_bias=None):
        """Transformer decoder forward pass.

        Args:
            ys (FloatTensor): `[B, L, d_model]`
            yy_mask (ByteTensor): `[B, L (query), L (key)]`
            xs (FloatTensor): encoder outputs. `[B, T, d_model]`
            xy_mask (ByteTensor): `[B, L, T]`
            cache (FloatTensor): `[B, L-1, d_model]`
            xy_aws_prev (FloatTensor): `[B, H, L, T]`
            mode (str): decoding mode for MMA
            eps_wait (int): wait time delay for head-synchronous decoding in MMA
            lmout (FloatTensor): `[B, L, d_model]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            memory (FloatTensor): `[B, L_prev, d_model]`
            u_bias (FloatTensor): global parameter for TransformerXL
            v_bias (FloatTensor): global parameter for TransformerXL
        Returns:
            out (FloatTensor): `[B, L, d_model]`

        """
        self.reset_visualization()
        if self.dropout_layer > 0 and self.training and random.random() < self.dropout_layer:
            return ys
        residual = ys
        if self.memory_transformer:
            if cache is not None:
                pos_embs = pos_embs[-ys.size(1):]
            if memory is not None and memory.dim() > 1:
                cat = self.norm1(torch.cat([memory, ys], dim=1))
                ys = cat[:, memory.size(1):]
            else:
                ys = self.norm1(ys)
                cat = ys
        else:
            ys = self.norm1(ys)
        if cache is not None:
            ys_q = ys[:, -1:]
            residual = residual[:, -1:]
            yy_mask = yy_mask[:, -1:]
        else:
            ys_q = ys
        if self.memory_transformer:
            out, self._yy_aws = self.self_attn(cat, ys_q, pos_embs, yy_mask, u_bias, v_bias)
        else:
            out, self._yy_aws = self.self_attn(ys, ys, ys_q, mask=yy_mask)[:2]
        out = self.dropout(out) + residual
        if self.src_tgt_attention:
            residual = out
            out = self.norm2(out)
            out, self._xy_aws, attn_state = self.src_attn(xs, xs, out, mask=xy_mask, aw_prev=xy_aws_prev, mode=mode, eps_wait=eps_wait)
            out = self.dropout(out) + residual
            if attn_state.get('beta', None) is not None:
                self._xy_aws_beta = attn_state['beta']
            if attn_state.get('p_choose', None) is not None:
                self._xy_aws_p_choose = attn_state['p_choose']
        if self.lm_fusion:
            residual = out
            out = self.norm_lm(out)
            lmout = self.linear_lm_feat(lmout)
            if 'attention' in self.lm_fusion:
                out, self._yy_aws_lm, _ = self.lm_attn(lmout, lmout, out, mask=yy_mask)
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
        return out


class XLPositionalEmbedding(nn.Module):
    """Positional embedding for TransformerXL.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        dropout (float): dropout probability

    """

    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        inv_freq = 1 / 10000 ** (torch.arange(0.0, d_model, 2.0) / d_model)
        self.register_buffer('inv_freq', inv_freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs, scale=False, n_cache=0):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, L, d_model]`
            scale (bool): multiply a scale factor
            n_cache (int): number of state caches
        Returns:
            xs (FloatTensor): `[B, L, d_model]`
            pos_emb (LongTensor): `[L, 1, d_model]`

        """
        if scale:
            xs = xs * self.scale
        pos_idxs = torch.arange(-1, -(xs.size(1) + n_cache) - 1, -1.0, dtype=torch.float, device=xs.device)
        sinusoid_inp_fwd = torch.einsum('i,j->ij', pos_idxs, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp_fwd.sin(), sinusoid_inp_fwd.cos()], dim=-1)
        pos_emb = self.dropout(pos_emb)
        return xs, pos_emb.unsqueeze(1)


def init_like_transformer_xl(n, p, std):
    """Initialize like TransformerXL.
        See https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/train.py

    Args:
        n (str): parameter name
        p (Tensor): parameter
        str (float): standard deviation

    """
    if 'norm' in n and 'weight' in n:
        assert p.dim() == 1
        nn.init.normal_(p, mean=1.0, std=std)
        logger.info('Initialize %s with %s / (1.0, %.3f)' % (n, 'normal', std))
    elif p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() == 2:
        nn.init.normal_(p, mean=0, std=std)
        logger.info('Initialize %s with %s / (0.0, %.3f)' % (n, 'normal', std))
    else:
        raise ValueError(n)


def mkdir_join(path, *dir_name, rank=0):
    """Concatenate root path and 1 or more paths, and make a new directory if the directory does not exist.
    Args:
        path (str): path to a directory
        rank (int): rank of current process group
        dir_name (str): a directory name
    Returns:
        path to the new directory
    """
    p = Path(path)
    if not p.is_dir() and rank == 0:
        p.mkdir()
    for i in range(len(dir_name)):
        if i < len(dir_name) - 1:
            p = p.joinpath(dir_name[i])
            if not p.is_dir() and rank == 0:
                p.mkdir()
        elif '.' not in dir_name[i]:
            p = p.joinpath(dir_name[i])
            if not p.is_dir() and rank == 0:
                p.mkdir()
        else:
            p = p.joinpath(dir_name[i])
    return str(p.absolute())


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.

    Args:
        x (torch.Tensor):
    Returns:
        np.ndarray

    """
    if x is None:
        return x
    return x.cpu().detach().numpy()


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
        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        self.cache_theta = 0.2
        self.cache_lambda = 0.2
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []
        self.embed_cache = None
        self.pos_emb = XLPositionalEmbedding(self.d_model, args.dropout_in)
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.scale = math.sqrt(self.d_model)
        self.dropout_emb = nn.Dropout(p=args.dropout_in)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(self.d_model, args.transformer_d_ff, 'scaled_dot', self.n_heads, args.dropout_hidden, args.dropout_att, args.dropout_layer, args.transformer_layer_norm_eps, args.transformer_ffn_activation, args.transformer_param_init, src_tgt_attention=False, memory_transformer=True)) for lth in range(self.n_layers)])
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

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('Transformer-XL LM')
        group.add_argument('--transformer_d_model', type=int, default=256, help='number of units in the MHA layer')
        group.add_argument('--transformer_d_ff', type=int, default=2048, help='number of units in the FFN layer')
        group.add_argument('--transformer_n_heads', type=int, default=4, help='number of heads in the MHA layer')
        group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12, help='epsilon value for layer normalization')
        group.add_argument('--transformer_ffn_activation', type=str, default='relu', choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'], help='nonlinear activation for the FFN layer')
        group.add_argument('--transformer_param_init', type=str, default='xavier_uniform', choices=['xavier_uniform', 'pytorch'], help='parameter initialization')
        group.add_argument('--dropout_att', type=float, default=0.1, help='dropout probability for the attention weights')
        group.add_argument('--dropout_layer', type=float, default=0.0, help='LayerDrop probability for Transformer layers')
        group.add_argument('--mem_len', type=int, default=0, help='number of tokens for memory in TransformerXL during training')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name = args.lm_type
        dir_name += str(args.transformer_d_model) + 'dmodel'
        dir_name += str(args.transformer_d_ff) + 'dff'
        dir_name += str(args.n_layers) + 'L'
        dir_name += str(args.transformer_n_heads) + 'H'
        if args.tie_embedding:
            dir_name += '_tie'
        if args.adaptive_softmax:
            dir_name += '_adaptiveSM'
        if args.mem_len > 0:
            dir_name += '_mem' + str(args.mem_len)
        return dir_name

    def reset_parameters(self):
        """Initialize parameters with normal distribution."""
        logger.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_like_transformer_xl(n, p, std=0.02)

    def init_memory(self):
        """Initialize memory."""
        return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (List): length `n_layers` (inter-utterance),
                each of which contains a FloatTensor of size `[B, mlen, d_model]`
            hidden_states (List): length `n_layers` (intra-utterance),
                each of which contains a FloatTensor of size `[B, L, d_model]`
        Returns:
            new_mems (List): length `n_layers`,
                each of which contains a FloatTensor of size `[B, mlen, d_model]`

        """
        if memory_prev is None:
            memory_prev = self.init_memory()
        assert len(hidden_states) == len(memory_prev), (len(hidden_states), len(memory_prev))
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

    def embed_token_id(self, indices):
        """Embed token IDs.

        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.dropout_emb(self.embed(indices) * self.scale)
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def decode(self, ys, state=None, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (List): dummy interfance for RNNLM
            mems (List): length `n_layers` (inter-utterance),
                each of which contains a FloatTensor of size `[B, mlen, d_model]`
            cache (List): length `n_layers` (intra-utterance),
                each of which contains a FloatTensor of size `[B, L-1, d_model]`
            incremental (bool): ASR decoding mode
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]`
            new_cache (List): length `n_layers`,
                each of which contains a FloatTensor of size `[B, L, d_model]`

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
        causal_mask = torch.tril(causal_mask, diagonal=mlen).unsqueeze(0)
        causal_mask = causal_mask.repeat([bs, 1, 1])
        out = self.embed_token_id(ys)
        ys, rel_pos_embs = self.pos_emb(ys, n_cache=mlen)
        new_mems = [None] * self.n_layers
        new_cache = [None] * self.n_layers
        hidden_states = [out]
        for lth, (mem, layer) in enumerate(zip(mems, self.layers)):
            if incremental and mlen > 0 and mem.size(0) != bs:
                mem = mem.repeat([bs, 1, 1])
            out = layer(out, causal_mask, cache=cache[lth], pos_embs=rel_pos_embs, memory=mem, u_bias=self.u_bias, v_bias=self.v_bias)
            if incremental:
                new_cache[lth] = out
            elif lth < self.n_layers - 1:
                hidden_states.append(out)
            if not self.training and layer.yy_aws is not None:
                setattr(self, 'yy_aws_layer%d' % lth, tensor2np(layer.yy_aws))
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
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
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
                ax.imshow(yy_aws[-1, h, :, :], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % lth))
            plt.close()


class CausalConv1d(nn.Module):
    """1D dilated causal convolution.

    Args:
        in_channels (int): input channel size
        out_channels (int): output channel size
        kernel_size (int): kernel size
        dilation (int): deletion rate
        param_init (str): parameter initialization method

    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, param_init=''):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, groups=groups)
        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        elif param_init == 'lecun':
            self.reset_parameters_lecun()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def reset_parameters_lecun(self, param_init=0.1):
        """Initialize parameters with lecun style.."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun_normal(n, p, param_init)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, C_in]`
        Returns:
            xs (FloatTensor): `[B, T, C_out]`

        """
        xs = xs.transpose(2, 1)
        xs = self.conv1d(xs)
        if self.padding != 0:
            xs = xs[:, :, :-self.padding]
        xs = xs.transpose(2, 1).contiguous()
        return xs


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Args:
        d_model (int): dimension of MultiheadAttentionMechanism
        dropout (float): dropout probability
        pe_type (str): type of positional encoding
        param_init (str): parameter initialization method
        max_len (int): maximum lenght for sinusoidal positional encoding
        conv_kernel_size (int): window size for 1dconv positional encoding
        layer_norm_eps (float): epsilon value for layer normalization

    """

    def __init__(self, d_model, dropout, pe_type, param_init, max_len=5000, conv_kernel_size=3, layer_norm_eps=1e-12):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type
        self.scale = math.sqrt(d_model)
        if '1dconv' in pe_type:
            causal_conv1d = CausalConv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel_size, param_init=param_init)
            layers = []
            nlayers = int(pe_type.replace('1dconv', '')[0])
            for _ in range(nlayers):
                layers.append(copy.deepcopy(causal_conv1d))
                layers.append(nn.LayerNorm(d_model, eps=layer_norm_eps))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            self.pe = nn.Sequential(*layers)
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

    def forward(self, xs, scale=True, offset=0):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            scale (bool): multiply a scale factor
            offset (int): input offset for streaming inference
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        if scale:
            xs = xs * self.scale
        if self.pe_type == 'none':
            xs = self.dropout(xs)
            return xs
        elif self.pe_type == 'add':
            xs = xs + self.pe[:, offset:xs.size(1) + offset]
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
        self.tie_embedding = args.tie_embedding
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
        self.embed_cache = None
        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.pos_enc = PositionalEncoding(self.d_model, args.dropout_in, args.transformer_pe_type, args.transformer_param_init)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(self.d_model, args.transformer_d_ff, 'scaled_dot', self.n_heads, args.dropout_hidden, args.dropout_att, args.dropout_layer, args.transformer_layer_norm_eps, args.transformer_ffn_activation, args.transformer_param_init, src_tgt_attention=False)) for lth in range(self.n_layers)])
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

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('Transformer LM')
        group.add_argument('--transformer_d_model', type=int, default=256, help='number of units in the MHA layer')
        group.add_argument('--transformer_d_ff', type=int, default=2048, help='number of units in the FFN layer')
        group.add_argument('--transformer_n_heads', type=int, default=4, help='number of heads in the MHA layer')
        group.add_argument('--transformer_pe_type', type=str, default='add', choices=['add', 'concat', 'none', '1dconv3L'], help='type of positional encoding')
        group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12, help='epsilon value for layer normalization')
        group.add_argument('--transformer_ffn_activation', type=str, default='relu', choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'], help='nonlinear activation for the FFN layer')
        group.add_argument('--transformer_param_init', type=str, default='xavier_uniform', choices=['xavier_uniform', 'pytorch'], help='parameter initialization')
        group.add_argument('--dropout_att', type=float, default=0.1, help='dropout probability for the attention weights')
        group.add_argument('--dropout_layer', type=float, default=0.0, help='LayerDrop probability for Transformer layers')
        group.add_argument('--mem_len', type=int, default=0, help='number of tokens for memory in TransformerXL during training')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name = args.lm_type
        dir_name += str(args.transformer_d_model) + 'dmodel'
        dir_name += str(args.transformer_d_ff) + 'dff'
        dir_name += str(args.n_layers) + 'L'
        dir_name += str(args.transformer_n_heads) + 'H'
        dir_name += 'pe' + str(args.transformer_pe_type)
        if args.tie_embedding:
            dir_name += '_tie'
        if args.adaptive_softmax:
            dir_name += '_adaptiveSM'
        return dir_name

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.constant_(self.embed.weight[self.pad], 0)
        if self.output is not None and not self.tie_embedding:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.constant_(self.output.bias, 0.0)

    def embed_token_id(self, indices):
        """Embed token IDs.

        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.embed(indices)
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def decode(self, ys, state=None, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (List): dummy interfance for RNNLM
            mems (List): length `n_layers` (inter-utterance),
                each of which contains a FloatTensor of size `[B, mlen, d_model]`
            cache (List): length `n_layers` (intra-utterance),
                each of which contains a FloatTensor of size `[B, L-1, d_model]`
            incremental (bool): ASR decoding mode
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]`
            new_cache (List): length `n_layers`,
                each of which contains a FloatTensor of size `[B, L, d_model]`

        """
        if cache is None:
            cache = [None] * self.n_layers
        bs, ylen = ys.size()[:2]
        n_hist = 0
        if incremental and cache[0] is not None:
            n_hist = cache[0].size(1)
            ylen += n_hist
        causal_mask = ys.new_ones(ylen, ylen).byte()
        causal_mask = torch.tril(causal_mask).unsqueeze(0)
        causal_mask = causal_mask.repeat([bs, 1, 1])
        out = self.pos_enc(self.embed_token_id(ys), scale=True, offset=max(0, n_hist))
        new_cache = [None] * self.n_layers
        hidden_states = [out]
        for lth, layer in enumerate(self.layers):
            out = layer(out, causal_mask, cache=cache[lth])
            if incremental:
                new_cache[lth] = out
            elif lth < self.n_layers - 1:
                hidden_states.append(out)
            if not self.training and layer.yy_aws is not None:
                setattr(self, 'yy_aws_layer%d' % lth, tensor2np(layer.yy_aws))
        out = self.norm_out(out)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out
        return logits, out, new_cache

    def plot_attention(self, n_cols=4):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
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
                ax.imshow(yy_aws[-1, h, :, :], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % lth))
            plt.close()


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        atype (str): type of attention mechanisms
        adim: (int) dimension of attention space
        sharpening_factor (float): sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): number of channels of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): size of kernel.
            This must be the odd number.
        dropout (float): dropout probability for attention weights
        lookahead (int): lookahead frames for triggered attention

    """

    def __init__(self, kdim, qdim, adim, atype, sharpening_factor=1, sigmoid_smoothing=False, conv_out_channels=10, conv_kernel_size=201, dropout=0.0, lookahead=2):
        super().__init__()
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

    def forward(self, key, value, query, mask=None, aw_prev=None, cache=False, mode='', trigger_points=None, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            klens (IntTensor): `[B]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            cache (bool): cache key and mask
            mode: dummy interface for MoChA/MMA
            trigger_points (IntTensor): `[B]`
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, 1, vdim]`
            aw (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            attn_state (dict): dummy interface

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        attn_state = {}
        if aw_prev is None:
            aw_prev = key.new_zeros(bs, 1, klen)
        else:
            aw_prev = aw_prev.squeeze(1)
        if self.key is None or not cache:
            if self.atype in ['add', 'triggered_attention', 'location', 'dot', 'luong_general']:
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
        NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
        if self.atype == 'triggered_attention':
            assert trigger_points is not None
            for b in range(bs):
                e[b, :, trigger_points[b] + self.lookahead + 1:] = NEG_INF
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        if self.sigmoid_smoothing:
            aw = torch.sigmoid(e) / torch.sigmoid(e).sum(-1).unsqueeze(-1)
        else:
            aw = torch.softmax(e * self.sharpening_factor, dim=-1)
        aw = self.dropout(aw)
        cv = torch.bmm(aw, value)
        return cv, aw.unsqueeze(1), attn_state


def make_pad_mask(seq_lens):
    """Make mask for padding.

    Args:
        seq_lens (IntTensor): `[B]`
    Returns:
        mask (IntTensor): `[B, T]`

    """
    bs = seq_lens.size(0)
    max_time = seq_lens.max()
    seq_range = torch.arange(0, max_time, dtype=torch.int32, device=seq_lens.device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)
    mask = seq_range < seq_lens.unsqueeze(-1)
    return mask


class CIF(nn.Module):
    """Continuous integrate and fire (CIF).

    Args:
        enc_dim (int): dimension of encoder outputs
        window (int): kernel size of 1dconv
        threshold (int): boundary threshold (equivalent to beta in the paper)
        param_init (int): parameter initialization method
        layer_norm_eps (int): epsilon value for layer normalization

    """

    def __init__(self, enc_dim, window, threshold=1.0, param_init='', layer_norm_eps=1e-12):
        super().__init__()
        self.enc_dim = enc_dim
        self.beta = threshold
        assert (window - 1) % 2 == 0, 'window must be the odd number.'
        self.conv1d = nn.Conv1d(in_channels=enc_dim, out_channels=enc_dim, kernel_size=window, stride=1, padding=(window - 1) // 2)
        self.norm = nn.LayerNorm(enc_dim, eps=layer_norm_eps)
        self.proj = nn.Linear(enc_dim, 1)
        if param_init == 'xavier_uniform':
            self.reset_parameters()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def forward(self, eouts, elens, ylens=None, mode='parallel', streaming=False):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, enc_dim]`
            elens (IntTensor): `[B]`
            ylens (IntTensor): `[B]`
            mode (str): parallel/incremental
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, L, enc_dim]`
            aws (FloatTensor): `[B, L, T]`
            attn_state (dict): dummy interface
                alpha (FloatTensor): `[B, T]`

        """
        bs, xmax, enc_dim = eouts.size()
        attn_state = {}
        conv_feat = self.conv1d(eouts.transpose(2, 1)).transpose(2, 1)
        conv_feat = torch.relu(self.norm(conv_feat))
        alpha = torch.sigmoid(self.proj(conv_feat)).squeeze(2)
        if mode == 'parallel':
            assert ylens is not None
            device = eouts.device
            ylens = ylens
            mask = make_pad_mask(elens)
            alpha = alpha.clone().masked_fill_(mask == 0, 0)
            alpha_norm = alpha / alpha.sum(1, keepdim=True) * ylens.float().unsqueeze(1)
            ymax = int(ylens.max().item())
        elif mode == 'incremental':
            alpha_norm = alpha
            ymax = 1
            if bs > 1:
                raise NotImplementedError('Batch mode is not supported.')
        else:
            raise ValueError(mode)
        cv = eouts.new_zeros(bs, ymax + 1, enc_dim)
        aws = eouts.new_zeros(bs, ymax + 1, xmax)
        n_tokens = torch.zeros(bs, dtype=torch.int64)
        state = eouts.new_zeros(bs, self.enc_dim)
        alpha_accum = eouts.new_zeros(bs)
        for j in range(xmax):
            alpha_accum_prev = alpha_accum
            alpha_accum += alpha_norm[:, j]
            if mode == 'parallel' and (alpha_accum >= self.beta).sum() == 0:
                state += alpha_norm[:, j, None] * eouts[:, j]
                aws[:, n_tokens, j] += alpha_norm[:, j]
            else:
                for b in range(bs):
                    if j > elens[b] - 1:
                        continue
                    if mode == 'parallel' and n_tokens[b].item() >= ylens[b]:
                        continue
                    if alpha_accum[b] < self.beta:
                        state[b] += alpha_norm[b, j, None] * eouts[b, j]
                        aws[b, n_tokens[b], j] += alpha_norm[b, j]
                        if mode == 'incremental' and j == elens[b] - 1:
                            if alpha_accum[b] >= 0.5:
                                n_tokens[b] += 1
                                cv[b, n_tokens[b]] = state[b]
                            break
                    else:
                        ak1 = 1 - alpha_accum_prev[b]
                        ak2 = alpha_norm[b, j] - ak1
                        cv[b, n_tokens[b]] = state[b] + ak1 * eouts[b, j]
                        aws[b, n_tokens[b], j] += ak1
                        n_tokens[b] += 1
                        state[b] = ak2 * eouts[b, j]
                        alpha_accum[b] = ak2
                        aws[b, n_tokens[b], j] += ak2
                        if mode == 'incremental':
                            break
                if mode == 'incremental' and n_tokens[0] >= 1:
                    break
        cv = cv[:, :ymax]
        aws = aws[:, :ymax]
        attn_state['alpha'] = alpha
        return cv, aws, attn_state


class Swish(torch.nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerConvBlock(nn.Module):
    """A single convolution block for the Conformer encoder.

    Args:
        d_model (int): input/output dimension
        kernel_size (int): kernel size in depthwise convolution
        param_init (str): parameter initialization method
        normalization (str): batch_norm/group_norm/layer_norm
        causal (bool): causal mode for streaming infernece

    """

    def __init__(self, d_model, kernel_size, param_init, normalization='batch_norm', causal=False):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, 'kernel_size must be the odd number.'
        assert kernel_size >= 3, 'kernel_size must be larger than 3.'
        self.kernel_size = kernel_size
        self.causal = causal
        if causal:
            self.padding = kernel_size - 1
        else:
            self.padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, stride=1, padding=self.padding, groups=d_model, bias=True)
        if normalization == 'batch_norm':
            self.norm = nn.BatchNorm1d(d_model)
        elif normalization == 'group_norm':
            num_groups = 2
            self.norm = nn.GroupNorm(num_groups=max(1, d_model // num_groups), num_channels=d_model)
        elif normalization == 'layer_norm':
            self.norm = nn.LayerNorm(d_model, eps=1e-12)
        else:
            raise NotImplementedError(normalization)
        logger.info('normalization: %s' % normalization)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        elif param_init == 'lecun':
            self.reset_parameters_lecun()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for conv_layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
            for n, p in conv_layer.named_parameters():
                init_with_xavier_uniform(n, p)

    def reset_parameters_lecun(self, param_init=0.1):
        """Initialize parameters with lecun style.."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for conv_layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
            for n, p in conv_layer.named_parameters():
                init_with_lecun_normal(n, p, param_init)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        bs, xmax, dim = xs.size()
        xs = xs.transpose(2, 1).contiguous()
        xs = self.pointwise_conv1(xs)
        xs = F.glu(xs, dim=1)
        xs = self.depthwise_conv(xs)
        if self.causal:
            xs = xs[:, :, :-self.padding]
        xs = xs.transpose(2, 1)
        if isinstance(self.norm, nn.LayerNorm):
            xs = self.activation(self.norm(xs))
        else:
            xs = xs.contiguous().view(bs * xmax, -1, 1)
            xs = self.activation(self.norm(xs))
            xs = xs.view(bs, xmax, -1)
        xs = xs.transpose(2, 1)
        xs = self.pointwise_conv2(xs)
        xs = xs.transpose(2, 1).contiguous()
        return xs


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
        """Forward pass.

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


def softplus(x):
    if hasattr(torch.nn.functional, 'softplus'):
        return torch.nn.functional.softplus(x.float()).type_as(x)
    else:
        raise NotImplementedError


class GMMAttention(nn.Module):
    """GMM attention.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        n_mixtures (int): number of mixtures
        dropout (float): dropout probability for attention weights
        param_init (str): parameter initialization method
        vfloor (float): parameter for numerical stability
        nonlinear (torch.function): exp or softplus

    """

    def __init__(self, kdim, qdim, adim, n_mixtures, dropout=0.0, param_init='', vfloor=1e-08, nonlinear='exp'):
        super().__init__()
        self.n_mix = n_mixtures
        self.n_heads = 1
        self.vfloor = vfloor
        self.reset()
        self.dropout = nn.Dropout(p=dropout)
        self.w_mixture = nn.Linear(qdim, n_mixtures)
        self.w_var = nn.Linear(qdim, n_mixtures)
        self.w_myu = nn.Linear(qdim, n_mixtures)
        if nonlinear == 'exp':
            self.nonlinear = torch.exp
        elif nonlinear == 'softplus':
            self.nonlinear = softplus
        else:
            raise NotImplementedError
        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def reset(self):
        pass

    def forward(self, key, value, query, mask=None, aw_prev=None, cache=False, mode='', trigger_points=None, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qmax, klen]`
            aw_prev (FloatTensor): `[B, klen, 1]`
            cache (bool): cache key and mask
            mode: dummy interface for MoChA/MMA
            trigger_points: dummy interface for MoChA/MMA
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, 1, vdim]`
            aw (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            attn_state (dict):
                myu (FloatTensor): `[B, 1 (qlen), n_mix]`

        """
        bs, klen = key.size()[:2]
        attn_state = {}
        if aw_prev is None:
            myu_prev = query.new_zeros(bs, 1, self.n_mix)
        else:
            myu_prev = aw_prev
        if mask is not None:
            assert mask.size() == (bs, 1, klen), (mask.size(), (bs, 1, klen))
        w_mix = torch.softmax(self.w_mixture(query), dim=-1)
        var = self.nonlinear(self.w_var(query))
        myu = self.nonlinear(self.w_myu(query))
        myu = myu + myu_prev + myu_prev
        attn_state['myu'] = myu
        js = torch.arange(klen, dtype=torch.float, device=query.device)
        js = js.unsqueeze(0).unsqueeze(2).repeat([bs, 1, self.n_mix])
        numerator = torch.exp(-torch.pow(js - myu, 2) / (2 * var + self.vfloor))
        denominator = torch.pow(2 * math.pi * var + self.vfloor, 0.5)
        aw = w_mix * numerator / denominator
        aw = aw.sum(2).unsqueeze(1)
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=myu.dtype).numpy().dtype).min)
            aw = aw.masked_fill_(mask == 0, NEG_INF)
        aw = self.dropout(aw)
        cv = torch.bmm(aw, value)
        return cv, aw.unsqueeze(1), attn_state


class MultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention (MHA) layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        atype (str): type of attention mechanism
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        xl_like: dummy argument for compatibility with relative MHA
        clamp_len: dummy

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, dropout_head=0.0, atype='scaled_dot', bias=True, param_init='', xl_like=False, clamp_len=-1):
        super().__init__()
        self.atype = atype
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.reset()
        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head
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
        self.w_out = nn.Linear(adim, odim, bias=bias)
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

    def forward(self, key, value, query, mask, aw_prev=None, aw_lower=None, cache=False, mode='', trigger_points=None, eps_wait=-1, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, qlen, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev: dummy interface
            cache (bool): cache key, value, and mask
            mode: dummy interface for MoChA/MMA
            trigger_points: dummy interface for MoChA/MMA
            eps_wait: dummy interface for MMA
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, klen]`
            attn_state (dict): dummy interface

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        attn_state = {}
        if self.key is None or not cache:
            self.key = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
            self.value = self.w_value(value).view(bs, -1, self.n_heads, self.d_k)
            if mask is not None:
                self.mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
                mask_size = bs, qlen, klen, self.n_heads
                assert self.mask.size() == mask_size, (self.mask.size(), mask_size)
            else:
                self.mask = None
        key = self.key
        query = self.w_query(query).view(bs, -1, self.n_heads, self.d_k)
        if self.atype == 'scaled_dot':
            e = torch.einsum('bihd,bjhd->bijh', (query, key)) / self.scale
        elif self.atype == 'add':
            e = self.v(torch.tanh(key[:, None] + query[:, :, None]).view(bs, qlen, klen, -1))
        if self.mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        aw = torch.softmax(e, dim=2)
        aw = self.dropout_attn(aw)
        aw_masked = aw.clone()
        if self.dropout_head > 0 and self.training:
            aw_masked = aw_masked.permute(0, 3, 1, 2)
            aw_masked = headdrop(aw_masked, self.n_heads, self.dropout_head)
            aw_masked = aw_masked.permute(0, 2, 3, 1)
        cv = torch.einsum('bijh,bjhd->bihd', (aw_masked, self.value))
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)
        return cv, aw, attn_state


def gelu(x):
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_accurate(x):
    if not hasattr(gelu_accurate, '_a'):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """Positionwise fully-connected feed-forward neural network (FFN) layer.

    Args:
        d_model (int): input and output dimension
        d_ff (int): hidden dimension
        dropout (float): dropout probability
        activation (str): non-linear activation function
        param_init (str): parameter initialization method
        bottleneck_dim (int): bottleneck dimension for low-rank FFN

    """

    def __init__(self, d_model, d_ff, dropout, activation, param_init, bottleneck_dim=0):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim > 0:
            self.w_1_e = nn.Linear(d_model, bottleneck_dim)
            self.w_1_d = nn.Linear(bottleneck_dim, d_ff)
            self.w_2_e = nn.Linear(d_ff, bottleneck_dim)
            self.w_2_d = nn.Linear(bottleneck_dim, d_model)
        else:
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
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError(activation)
        logger.info('FFN activation: %s' % activation)
        if param_init == 'xavier_uniform':
            self.reset_parameters()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        if self.bottleneck_dim > 0:
            return self.w_2_d(self.w_2_e(self.dropout(self.activation(self.w_1_d(self.w_1_e(xs))))))
        else:
            return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class RelativeMultiheadAttentionMechanism(nn.Module):
    """Relative multi-head attention layer for TransformerXL.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability for attention weights
        dropout_head (float): HeadDrop probability
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        xl_like (bool): use TransformerXL like relative positional encoding.
            Otherwise, use relative positional encoding like Shaw et al. 2018
        clamp_len (int): maximum relative distance from each position

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, dropout_head=0.0, bias=False, param_init='', xl_like=False, clamp_len=-1):
        super().__init__()
        assert adim % n_heads == 0
        self.d_k = adim // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)
        self.xl_like = xl_like
        self.clamp_len = clamp_len
        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_head = dropout_head
        assert kdim == qdim
        self.w_key = nn.Linear(kdim, adim, bias=bias)
        self.w_value = nn.Linear(kdim, adim, bias=bias)
        self.w_query = nn.Linear(qdim, adim, bias=bias)
        self.w_out = nn.Linear(adim, odim, bias=bias)
        if xl_like:
            self.w_pos = nn.Linear(qdim, adim, bias=bias)
        if param_init == 'xavier_uniform':
            self.reset_parameters(bias)
        else:
            logger.info('Parameter initialization is skipped.')

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
        if self.xl_like:
            nn.init.xavier_uniform_(self.w_pos.weight)
            if bias:
                nn.init.constant_(self.w_pos.bias, 0.0)

    def _rel_shift_legacy(self, xs):
        """Calculate relative positional attention efficiently (old version).

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

    def _rel_shift(self, xs):
        """Calculate relative positional attention efficiently.

        Args:
            xs (FloatTensor): `[B, qlen, klen, H]`
        Returns:
            xs_shifted (FloatTensor): `[B, qlen, klen, H]`

        """
        bs, qlen, klen, n_heads = xs.size()
        xs = xs.permute(0, 3, 2, 1)
        idx = torch.arange(klen, device=xs.device)
        k_idx, q_idx = idx.unsqueeze(0), idx.unsqueeze(1)
        rel_pos_idx = torch.abs(k_idx - q_idx)
        if klen != qlen:
            rel_pos_idx = rel_pos_idx[:, :qlen]
            mask = xs.new_ones(qlen, klen, dtype=torch.bool if torch_12_plus else torch.uint8)
            mask = torch.tril(mask, diagonal=0).transpose(1, 0)
            rel_pos_idx[mask] *= -1
            rel_pos_idx = klen - qlen - rel_pos_idx
            rel_pos_idx[rel_pos_idx < 0] *= -1
        if self.clamp_len > 0:
            rel_pos_idx.clamp_(max=self.clamp_len)
        rel_pos_idx = rel_pos_idx.expand_as(xs)
        x_shift = torch.gather(xs, dim=2, index=rel_pos_idx)
        x_shift = x_shift.permute(0, 3, 2, 1)
        return x_shift

    def forward(self, key, query, pos_embs, mask, u_bias=None, v_bias=None):
        """Forward pass.

        Args:
            cat (FloatTensor): `[B, mlen+qlen, kdim]`
            mask (ByteTensor): `[B, qlen, mlen+qlen]`
            pos_embs (LongTensor): `[mlen+qlen, 1, d_model]`
            u_bias (nn.Parameter): `[H, d_k]`
            v_bias (nn.Parameter): `[H, d_k]`
        Returns:
            cv (FloatTensor): `[B, qlen, vdim]`
            aw (FloatTensor): `[B, H, qlen, mlen+qlen]`

        """
        bs, qlen = query.size()[:2]
        mlen = key.size(1) - qlen
        if mask is not None:
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + qlen, self.n_heads), (mask.size(), (bs, qlen, mlen + qlen, self.n_heads))
        k = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)
        v = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)
        q = self.w_query(key[:, -qlen:]).view(bs, -1, self.n_heads, self.d_k)
        if self.xl_like:
            _pos_embs = self.w_pos(pos_embs)
        else:
            _pos_embs = self.w_value(pos_embs)
        _pos_embs = _pos_embs.view(-1, self.n_heads, self.d_k)
        if u_bias is not None:
            assert self.xl_like
            AC = torch.einsum('bihd,bjhd->bijh', (q + u_bias[None, None], k))
        else:
            AC = torch.einsum('bihd,bjhd->bijh', (q, k))
        if v_bias is not None:
            assert self.xl_like
            BD = torch.einsum('bihd,jhd->bijh', (q + v_bias[None, None], _pos_embs))
        else:
            BD = torch.einsum('bihd,jhd->bijh', (q, _pos_embs))
        BD = self._rel_shift(BD)
        e = (AC + BD) / self.scale
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(mask == 0, NEG_INF)
        aw = torch.softmax(e, dim=2)
        aw = self.dropout_attn(aw)
        aw_masked = aw.clone()
        if self.dropout_head > 0 and self.training:
            aw_masked = aw_masked.permute(0, 3, 1, 2)
            aw_masked = headdrop(aw_masked, self.n_heads, self.dropout_head)
            aw_masked = aw_masked.permute(0, 2, 3, 1)
        cv = torch.einsum('bijh,bjhd->bihd', (aw, v))
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)
        return cv, aw


class SyncBidirMultiheadAttentionMechanism(nn.Module):
    """Multi-headed attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        adim: (int) dimension of attention space
        odim: (int) dimension of output
        n_heads (int): number of heads
        dropout (float): dropout probability
        atype (str): type of attention mechanisms
        bias (bool): use bias term in linear layers
        param_init (str): parameter initialization method
        future_weight (float):

    """

    def __init__(self, kdim, qdim, adim, odim, n_heads, dropout, atype='scaled_dot', bias=True, param_init='', future_weight=0.1):
        super().__init__()
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
        self.w_out = nn.Linear(adim, odim, bias=bias)
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

    def forward(self, key_fwd, value_fwd, query_fwd, key_bwd, value_bwd, query_bwd, tgt_mask, identity_mask, mode='', cache=True, trigger_points=None):
        """Forward pass.

        Args:
            key_fwd (FloatTensor): `[B, klen, kdim]`
            value_fwd (FloatTensor): `[B, klen, vdim]`
            query_fwd (FloatTensor): `[B, qlen, qdim]`
            key_bwd (FloatTensor): `[B, klen, kdim]`
            value_bwd (FloatTensor): `[B, klen, vdim]`
            query_bwd (FloatTensor): `[B, qlen, qdim]`
            tgt_mask (ByteTensor): `[B, qlen, klen]`
            identity_mask (ByteTensor): `[B, qlen, klen]`
            mode: dummy interface for MoChA/MMA
            cache (bool): cache key, value, and tgt_mask
            trigger_points (IntTensor): dummy
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
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e_fwd_h.dtype).numpy().dtype).min)
            e_fwd_h = e_fwd_h.masked_fill_(self.tgt_mask == 0, NEG_INF)
            e_bwd_h = e_bwd_h.masked_fill_(self.tgt_mask == 0, NEG_INF)
        if self.identity_mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e_fwd_f.dtype).numpy().dtype).min)
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


class SyncBidirTransformerDecoderBlock(nn.Module):
    """A single layer of the synchronous bidirectional Transformer decoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention probabilities
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method

    """

    def __init__(self, d_model, d_ff, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = SyncBidirMHA(kdim=d_model, qdim=d_model, adim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.src_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_visualization()

    def reset_visualization(self):
        self._yy_aws_h, self.yy_aws_f = None, None
        self._yy_aws_bwd_h, self._yy_aws_bwd_f = None, None
        self._xy_aws, self._xy_aws_bwd = None, None

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

        """
        self.reset_visualization()
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
        out, out_bwd, self._yy_aws_h, self.yy_aws_f, self._yy_aws_bwd_h, self._yy_aws_bwd_f = self.self_attn(ys, ys, ys_q, ys_bwd, ys_bwd, ys_bwd_q, tgt_mask=yy_mask, identity_mask=identity_mask)
        out = self.dropout(out) + residual
        out_bwd = self.dropout(out_bwd) + residual_bwd
        residual = out
        out = self.norm2(out)
        out, self._xy_aws, _ = self.src_attn(xs, xs, out, mask=xy_mask)
        out = self.dropout(out) + residual
        residual_bwd = out_bwd
        out_bwd = self.norm2(out_bwd)
        out_bwd, self._xy_aws_bwd, _ = self.src_attn(xs, xs, out_bwd, mask=xy_mask)
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
        return out, out_bwd


class ZoneoutCell(nn.Module):

    def __init__(self, cell, zoneout_prob_h, zoneout_prob_c):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        if not isinstance(cell, nn.RNNCellBase):
            raise TypeError('The cell is not a LSTMCell or GRUCell!')
        if isinstance(cell, nn.LSTMCell):
            self.prob = zoneout_prob_h, zoneout_prob_c
        else:
            self.prob = zoneout_prob_h

    def forward(self, inputs, state):
        """Forward pass.

        Args:
            inputs (FloatTensor): `[B, input_dim]'
            state (tuple or FloatTensor):
        Returns:
            state (tuple or FloatTensor):

        """
        return self.zoneout(state, self.cell(inputs, state), self.prob)

    def zoneout(self, state, next_state, prob):
        if isinstance(state, tuple):
            return self.zoneout(state[0], next_state[0], prob[0]), self.zoneout(state[1], next_state[1], prob[1])
        mask = state.new(state.size()).bernoulli_(prob)
        if self.training:
            return mask * next_state + (1 - mask) * state
        else:
            return prob * next_state + (1 - prob) * state


class ConformerEncoderBlock(nn.Module):
    """A single layer of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        pe_type (str): type of positional encoding
        clamp_len (int): maximum relative distance from each position
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        unidirectional (bool): pad right context for unidirectional encoding
        normalization (str): batch_norm/group_norm/layer_norm

    """

    def __init__(self, d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, pe_type, clamp_len, ffn_bottleneck_dim, unidirectional, normalization='layer_norm'):
        super(ConformerEncoderBlock, self).__init__()
        self.n_heads = n_heads
        self.fc_factor = 0.5
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward_macaron = FFN(d_model, d_ff, dropout, ffn_activation, param_init, ffn_bottleneck_dim)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = RelMHA(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init, xl_like=pe_type == 'relative_xl', clamp_len=clamp_len)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(d_model, kernel_size, param_init, normalization, causal=unidirectional)
        self.conv_context = kernel_size
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init, ffn_bottleneck_dim)
        self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer
        logger.info('Stochastic depth prob: %.3f' % dropout_layer)
        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, cache=None, pos_embs=None, rel_bias=(None, None)):
        """Conformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T (query), d_model]`
            xx_mask (ByteTensor): `[B, T (query), T (key)]`
            cache (dict):
                input_san: `[B, n_cache, d_model]`
                input_conv: `[B, n_cache, d_model]`
            pos_embs (LongTensor): `[T (query), 1, d_model]`
            rel_bias (tuple):
                u_bias (FloatTensor): global parameter for relative positional encoding
                v_bias (FloatTensor): global parameter for relative positional encoding
        Returns:
            xs (FloatTensor): `[B, T (query), d_model]`
            new_cache (dict):
                input_san: `[B, n_cache+T, d_model]`
                input_conv: `[B, n_cache+T, d_model]`

        """
        self.reset_visualization()
        new_cache = {}
        qlen = xs.size(1)
        u_bias, v_bias = rel_bias
        if self.dropout_layer > 0:
            if self.training and random.random() < self.dropout_layer:
                return xs, new_cache
            else:
                xs = xs / (1 - self.dropout_layer)
        residual = xs
        xs = self.norm1(xs)
        xs = self.feed_forward_macaron(xs)
        xs = self.fc_factor * self.dropout(xs) + residual
        residual = xs
        xs = self.norm2(xs)
        if cache is not None:
            xs = torch.cat([cache['input_san'], xs], dim=1)
        new_cache['input_san'] = xs
        xs_kv = xs
        if cache is not None:
            xs = xs[:, -qlen:]
            residual = residual[:, -qlen:]
            xx_mask = xx_mask[:, -qlen:]
        xs, self._xx_aws = self.self_attn(xs_kv, xs, pos_embs, xx_mask, u_bias, v_bias)
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm3(xs)
        if cache is not None:
            xs = torch.cat([cache['input_conv'], xs], dim=1)
            xs = xs[:, -(self.conv_context + qlen - 1):]
        new_cache['input_conv'] = xs
        xs = self.conv(xs)
        if cache is not None:
            xs = xs[:, -qlen:]
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm4(xs)
        xs = self.feed_forward(xs)
        xs = self.fc_factor * self.dropout(xs) + residual
        xs = self.norm5(xs)
        return xs, new_cache


class ConformerEncoderBlock_v2(nn.Module):
    """A single layer of the Conformer encoder (version 2, flip conv and self-attention,
       relative positional encoding is not used).

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        pe_type: dummy
        clamp_len: dummy
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        unidirectional (bool): pad right context for unidirectional encoding
        normalization (str): batch_norm/group_norm/layer_norm

    """

    def __init__(self, d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, pe_type, clamp_len, ffn_bottleneck_dim, unidirectional, normalization='layer_norm'):
        super(ConformerEncoderBlock_v2, self).__init__()
        self.n_heads = n_heads
        self.fc_factor = 0.5
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward_macaron = FFN(d_model, d_ff, dropout, ffn_activation, param_init, ffn_bottleneck_dim)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(d_model, kernel_size, param_init, normalization, causal=unidirectional)
        self.conv_context = kernel_size
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = MHA(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init, ffn_bottleneck_dim)
        self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer
        logger.info('Stochastic depth prob: %.3f' % dropout_layer)
        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, cache=None, pos_embs=None, rel_bias=(None, None)):
        """Conformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T (query), d_model]`
            xx_mask (ByteTensor): `[B, T (query), T (key)]`
            cache (dict):
                input_san: `[B, n_cache, d_model]`
                input_conv: `[B, n_cache, d_model]`
            pos_embs (LongTensor): not used
            rel_bias (tuple):
                u_bias (FloatTensor): not used
                v_bias (FloatTensor): not used
        Returns:
            xs (FloatTensor): `[B, T (query), d_model]`
            new_cache (dict):
                input_san: `[B, n_cache+T, d_model]`
                input_conv: `[B, n_cache+T, d_model]`

        """
        self.reset_visualization()
        new_cache = {}
        qlen = xs.size(1)
        u_bias, v_bias = rel_bias
        assert u_bias is None and v_bias is None
        if self.dropout_layer > 0:
            if self.training and random.random() < self.dropout_layer:
                return xs, new_cache
            else:
                xs = xs / (1 - self.dropout_layer)
        residual = xs
        xs = self.norm1(xs)
        xs = self.feed_forward_macaron(xs)
        xs = self.fc_factor * self.dropout(xs) + residual
        residual = xs
        xs = self.norm2(xs)
        if cache is not None:
            xs = torch.cat([cache['input_conv'], xs], dim=1)
            xs = xs[:, -(self.conv_context + qlen - 1):]
        new_cache['input_conv'] = xs
        xs = self.conv(xs)
        if cache is not None:
            xs = xs[:, -qlen:]
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm3(xs)
        if cache is not None:
            xs = torch.cat([cache['input_san'], xs], dim=1)
        new_cache['input_san'] = xs
        xs_kv = xs
        if cache is not None:
            xs = xs[:, -qlen:]
            residual = residual[:, -qlen:]
            xx_mask = xx_mask[:, -qlen:]
        xs, self._xx_aws = self.self_attn(xs_kv, xs_kv, xs, mask=xx_mask)[:2]
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm4(xs)
        xs = self.feed_forward(xs)
        xs = self.fc_factor * self.dropout(xs) + residual
        xs = self.norm5(xs)
        return xs, new_cache


class LayerNorm2D(nn.Module):
    """Layer normalization for CNN outputs."""

    def __init__(self, channel, idim, eps=1e-12):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm([channel, idim], eps=eps)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`

        """
        B, C, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous()
        xs = self.norm(xs)
        xs = xs.transpose(2, 1)
        return xs


class EncoderBase(ModelBase):
    """Base class for encoders."""

    def __init__(self):
        super(ModelBase, self).__init__()
        logger.info('Overriding EncoderBase class.')

    @property
    def output_dim(self):
        return self._odim

    @property
    def output_dim_sub1(self):
        return getattr(self, '_odim_sub1', self._odim)

    @property
    def output_dim_sub2(self):
        return getattr(self, '_odim_sub2', self._odim)

    @property
    def subsampling_factor(self):
        return self._factor

    @property
    def subsampling_factor_sub1(self):
        return self._factor_sub1

    @property
    def subsampling_factor_sub2(self):
        return self._factor_sub2

    def forward(self, xs, xlens, task):
        raise NotImplementedError

    def reset_cache(self):
        raise NotImplementedError

    def turn_on_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = True
                    logging.debug('Turn ON ceil_mode in %s.' % name)
                else:
                    self.turn_on_ceil_mode(module)

    def turn_off_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = False
                    logging.debug('Turn OFF ceil_mode in %s.' % name)
                else:
                    self.turn_off_ceil_mode(module)

    def _plot_attention(self, save_path=None, n_cols=2):
        """Plot attention for each head in all encoder layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        if not hasattr(self, 'aws_dict'):
            return
        for k, aw in self.aws_dict.items():
            if aw is None:
                continue
            lth = k.split('_')[-1].replace('layer', '')
            elens_l = self.data_dict['elens' + lth]
            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp, figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                ax.imshow(aw[-1, h, :elens_l[-1], :elens_l[-1]], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, '%s.png' % k))
            plt.close()


def parse_cnn_config(channels, kernel_sizes, strides, poolings):
    _channels, _kernel_sizes, _strides, _poolings = [], [], [], []
    is_1dconv = '(' not in kernel_sizes
    if len(channels) > 0:
        _channels = [int(c) for c in channels.split('_')]
    if len(kernel_sizes) > 0:
        if is_1dconv:
            _kernel_sizes = [int(c) for c in kernel_sizes.split('_')]
        else:
            _kernel_sizes = [[int(c.split(',')[0].replace('(', '')), int(c.split(',')[1].replace(')', ''))] for c in kernel_sizes.split('_')]
    if len(strides) > 0:
        if is_1dconv:
            assert '(' not in _strides and ')' not in _strides
            _strides = [int(s) for s in strides.split('_')]
        else:
            _strides = [[int(s.split(',')[0].replace('(', '')), int(s.split(',')[1].replace(')', ''))] for s in strides.split('_')]
    if len(poolings) > 0:
        if is_1dconv:
            assert '(' not in poolings and ')' not in poolings
            _poolings = [int(p) for p in poolings.split('_')]
        else:
            _poolings = [[int(p.split(',')[0].replace('(', '')), int(p.split(',')[1].replace(')', ''))] for p in poolings.split('_')]
    return (_channels, _kernel_sizes, _strides, _poolings), is_1dconv


class GatedConvEncoder(EncoderBase):
    """Gated convolutional encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channels in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        dropout (float) dropout probability
        batch_norm (bool): apply batch normalization
        last_proj_dim (int): dimension of the last projection layer
        param_init (float): model initialization parameter

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes, dropout, last_proj_dim, param_init):
        super(GatedConvEncoder, self).__init__()
        (channels, kernel_sizes, _, _), _ = parse_cnn_config(channels, kernel_sizes, '', '')
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        layers = OrderedDict()
        for lth in range(len(channels)):
            layers['conv%d' % lth] = ConvGLUBlock(kernel_sizes[lth][0], input_dim, channels[lth], weight_norm=True, dropout=0.2)
            input_dim = channels[lth]
        self.fc_glu = nn.utils.weight_norm(nn.Linear(input_dim, input_dim * 2), name='weight', dim=0)
        self._odim = int(input_dim)
        if last_proj_dim > 0:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim
        self.layers = nn.Sequential(layers)
        self._factor = 1
        self.reset_parameters(param_init)

    @staticmethod
    def define_name(dir_name, args):
        return dir_name

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

    def forward(self, xs, xlens, task, streaming=False, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]`
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T', C_o * F]`
                xlens (IntTensor): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None}, 'ys_sub1': {'xs': None, 'xlens': None}, 'ys_sub2': {'xs': None, 'xlens': None}}
        bs, xmax, input_dim = xs.size()
        xs = xs.transpose(2, 1).unsqueeze(3)
        xs = self.layers(xs)
        bs, out_ch, xmax, freq = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(bs, xmax, -1)
        xs = F.glu(self.fc_glu(xs), dim=2)
        if self.bridge is not None:
            xs = self.bridge(xs)
        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        else:
            raise NotImplementedError
        return eouts


class AddSubsampler(nn.Module):
    """Subsample by summing input frames."""

    def __init__(self, subsampling_factor):
        super(AddSubsampler, self).__init__()
        self.factor = subsampling_factor
        assert subsampling_factor <= 2

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens
        if batch_first:
            bs, xmax, idim = xs.size()
            xs_even = xs[:, ::self.factor]
            if xmax % 2 == 0:
                xs_odd = xs[:, 1::self.factor]
            else:
                xs_odd = torch.cat([xs, xs.new_zeros(bs, 1, idim)], dim=1)[:, 1::self.factor]
        else:
            xmax, bs, idim = xs.size()
            xs_even = xs[::self.factor]
            if xmax % 2 == 0:
                xs_odd = xs[1::self.factor]
            else:
                xs_odd = torch.cat([xs, xs.new_zeros(1, bs, idim)], dim=0)[1::self.factor]
        xs = xs_odd + xs_even
        xlens = [max(1, math.ceil(i.item() / self.factor)) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class ConcatSubsampler(nn.Module):
    """Subsample by concatenating successive input frames."""

    def __init__(self, subsampling_factor, n_units):
        super(ConcatSubsampler, self).__init__()
        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.proj = nn.Linear(n_units * subsampling_factor, n_units)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens
        if batch_first:
            xs = xs.transpose(1, 0).contiguous()
        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.factor - 1, -1, -1)], dim=-1) for t in range(xs.size(0)) if (t + 1) % self.factor == 0]
        xs = torch.cat(xs, dim=0)
        xs = torch.relu(self.proj(xs))
        if batch_first:
            xs = xs.transpose(1, 0)
        xlens = [max(1, i.item() // self.factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


def _update_1d(seq_len, layer):
    if type(layer) == nn.MaxPool1d and layer.ceil_mode:
        return math.ceil((seq_len + 1 + 2 * layer.padding - (layer.kernel_size - 1) - 1) // layer.stride + 1)
    else:
        return math.floor((seq_len + 2 * layer.padding[0] - (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)


def update_lens_1d(seq_lens, layer):
    """Update lengths (frequency or time).

    Args:
        seq_lens (IntTensor): `[B]`
        layer (nn.Conv1d or nn.MaxPool1d):
    Returns:
        seq_lens (IntTensor): `[B]`

    """
    if seq_lens is None:
        return seq_lens
    assert isinstance(seq_lens, torch.IntTensor)
    assert type(layer) in [nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d]
    seq_lens = [_update_1d(seq_len, layer) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    return seq_lens


class Conv1dSubsampler(nn.Module):
    """Subsample by stride in 1d convolution."""

    def __init__(self, subsampling_factor, n_units, kernel_size=3):
        super(Conv1dSubsampler, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.conv1d = nn.Conv1d(in_channels=n_units, out_channels=n_units, kernel_size=kernel_size, stride=subsampling_factor, padding=(kernel_size - 1) // 2)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens
        if batch_first:
            xs = self.conv1d(xs.transpose(2, 1))
            xs = xs.transpose(2, 1).contiguous()
        else:
            xs = self.conv1d(xs.permute(1, 2, 0))
            xs = xs.permute(2, 0, 1).contiguous()
        xs = torch.relu(xs)
        xlens = update_lens_1d(xlens, self.conv1d)
        return xs, xlens


class Conv1dBlock(EncoderBase):
    """1d-CNN block."""

    def __init__(self, in_channel, out_channel, kernel_size, stride, pooling, dropout, normalization, residual):
        super(Conv1dBlock, self).__init__()
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=1)
        self._odim = update_lens_1d(torch.IntTensor([in_channel]), self.conv1)[0].item()
        if normalization == 'batch_norm':
            self.norm1 = nn.BatchNorm1d(out_channel)
        elif normalization == 'layer_norm':
            self.norm1 = nn.LayerNorm(out_channel, eps=1e-12)
        else:
            self.norm1 = None
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1)
        self._odim = update_lens_1d(torch.IntTensor([self._odim]), self.conv2)[0].item()
        if normalization == 'batch_norm':
            self.norm2 = nn.BatchNorm1d(out_channel)
        elif normalization == 'layer_norm':
            self.norm2 = nn.LayerNorm(out_channel, eps=1e-12)
        else:
            self.norm2 = None
        self.pool = None
        if pooling > 1:
            self.pool = nn.MaxPool1d(kernel_size=pooling, stride=pooling, padding=0, ceil_mode=True)
            self._odim = update_lens_1d(torch.IntTensor([self._odim]), self.pool)[0].item()
            if self._odim % 2 != 0:
                self._odim = self._odim // 2 * 2

    def forward(self, xs, xlens, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]` (on CPU)
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        residual = xs
        xs = self.conv1(xs.transpose(2, 1)).transpose(2, 1)
        if self.norm1 is not None:
            xs = self.norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv1)
        xs = self.conv2(xs.transpose(2, 1)).transpose(2, 1)
        if self.norm2 is not None:
            xs = self.norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv2)
        if self.pool is not None:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1)
            xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens


def _update_2d(seq_len, layer, dim):
    if type(layer) == nn.MaxPool2d and layer.ceil_mode:
        return math.ceil((seq_len + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) // layer.stride[dim] + 1)
    else:
        return math.floor((seq_len + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) // layer.stride[dim] + 1)


def update_lens_2d(seq_lens, layer, dim=0):
    """Update lengths (frequency or time).

    Args:
        seq_lens (IntTensor): `[B]`
        layer (nn.Conv2d or nn.MaxPool2d):
        dim (int):
    Returns:
        seq_lens (IntTensor): `[B]`

    """
    if seq_lens is None:
        return seq_lens
    assert isinstance(seq_lens, torch.IntTensor)
    assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
    seq_lens = [_update_2d(seq_len, layer, dim) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    return seq_lens


class Conv2dBlock(EncoderBase):
    """2d-CNN block."""

    def __init__(self, input_dim, in_channel, out_channel, kernel_size, stride, pooling, dropout, normalization, residual):
        super(Conv2dBlock, self).__init__()
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.time_axis = 0
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=tuple(kernel_size), stride=(1, 1), padding=(1, 1))
        self._odim = update_lens_2d(torch.IntTensor([input_dim]), self.conv1, dim=1)[0].item()
        if normalization == 'batch_norm':
            self.norm1 = nn.BatchNorm2d(out_channel)
        elif normalization == 'layer_norm':
            self.norm1 = LayerNorm2D(out_channel, self._odim, eps=1e-12)
        else:
            self.norm1 = None
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=tuple(kernel_size), stride=tuple(stride), padding=(1, 1))
        self._odim = update_lens_2d(torch.IntTensor([self._odim]), self.conv2, dim=1)[0].item()
        if normalization == 'batch_norm':
            self.norm2 = nn.BatchNorm2d(out_channel)
        elif normalization == 'layer_norm':
            self.norm2 = LayerNorm2D(out_channel, self._odim, eps=1e-12)
        else:
            self.norm2 = None
        self.pool = None
        self._factor = 1
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling), stride=tuple(pooling), padding=(0, 0), ceil_mode=True)
            self._odim = update_lens_2d(torch.IntTensor([self._odim]), self.pool, dim=1)[0].item()
            if self._odim % 2 != 0:
                self._odim = self._odim // 2 * 2
            self._factor *= pooling[0]

    def forward(self, xs, xlens, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (IntTensor): `[B]` (on CPU)
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            xs (FloatTensor): `[B, C_o, T', F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        residual = xs
        xs = self.conv1(xs)
        if self.norm1 is not None:
            xs = self.norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv1, dim=0)
        stride = self.conv1.stride[self.time_axis]
        if lookback and xs.size(2) > stride:
            xs = xs[:, :, stride:]
            xlens -= stride
        if lookahead and xs.size(2) > stride:
            xs = xs[:, :, :xs.size(2) - stride]
            xlens -= stride
        xs = self.conv2(xs)
        if self.norm2 is not None:
            xs = self.norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv2, dim=0)
        stride = self.conv2.stride[self.time_axis]
        if lookback and xs.size(2) > stride:
            xs = xs[:, :, stride:]
            xlens -= stride
        if lookahead and xs.size(2) > stride:
            xs = xs[:, :, :xs.size(2) - stride]
            xlens -= stride
        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens_2d(xlens, self.pool, dim=0)
        return xs, xlens


class ConvEncoder(EncoderBase):
    """CNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        in_channel (int): number of channels of input features
        channels (str): number of channles in CNN blocks
        kernel_sizes (str): size of kernels in CNN blocks
        strides (str): strides in CNN blocks
        poolings (str): size of poolings in CNN blocks
        dropout (float): probability to drop nodes in hidden-hidden connection
        normalization (str): normalization in CNN blocks
        residual (bool): apply residual connections
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float): mean of uniform distribution for parameter initialization

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes, strides, poolings, dropout, normalization, residual, bottleneck_dim, param_init):
        super(ConvEncoder, self).__init__()
        assert channels
        (channels, kernel_sizes, strides, poolings), is_1dconv = parse_cnn_config(channels, kernel_sizes, strides, poolings)
        self.is_1dconv = is_1dconv
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)
        self.layers = nn.ModuleList()
        C_i = input_dim if is_1dconv else in_channel
        in_freq = self.input_freq
        for lth in range(len(channels)):
            if is_1dconv:
                block = Conv1dBlock(in_channel=C_i, out_channel=channels[lth], kernel_size=kernel_sizes[lth], stride=strides[lth], pooling=poolings[lth], dropout=dropout, normalization=normalization, residual=residual)
            else:
                block = Conv2dBlock(input_dim=in_freq, in_channel=C_i, out_channel=channels[lth], kernel_size=kernel_sizes[lth], stride=strides[lth], pooling=poolings[lth], dropout=dropout, normalization=normalization, residual=residual)
            self.layers += [block]
            in_freq = block.output_dim
            C_i = channels[lth]
        self._odim = C_i if is_1dconv else int(C_i * in_freq)
        self.bridge = None
        if bottleneck_dim > 0 and bottleneck_dim != self._odim:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim
        self._factor = 1
        if strides:
            for s in strides:
                self._factor *= s if is_1dconv else s[0]
        if poolings:
            for p in poolings:
                self._factor *= p if is_1dconv else p[0]
        self.calculate_context_size(kernel_sizes, strides, poolings)
        self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('CNN encoder')
        group.add_argument('--conv_in_channel', type=int, default=1, help='input dimension of the first CNN block')
        group.add_argument('--conv_channels', type=str, default='', help='delimited list of channles in each CNN block')
        group.add_argument('--conv_kernel_sizes', type=str, default='', help='delimited list of kernel sizes in each CNN block')
        group.add_argument('--conv_strides', type=str, default='', help='delimited list of strides in each CNN block')
        group.add_argument('--conv_poolings', type=str, default='', help='delimited list of poolings in each CNN block')
        group.add_argument('--conv_normalization', type=str, default='', choices=['', 'layer_norm', 'batch_norm'], help='normalization in each CNN block')
        group.add_argument('--conv_bottleneck_dim', type=int, default=0, help='dimension of the bottleneck layer between CNN and the subsequent RNN/Transformer layers')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        assert 'conv' in args.enc_type
        dir_name = args.enc_type.replace('conv_', '')
        if args.conv_channels and len(args.conv_channels.split('_')) > 0:
            tmp = dir_name
            dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
            if args.conv_normalization:
                dir_name += args.conv_normalization
            dir_name += tmp
        return dir_name

    @property
    def context_size(self):
        return self._context_size

    def calculate_context_size(self, kernel_sizes, strides, poolings):
        self._context_size = 0
        context_size_bottom = 0
        factor = 1
        for lth in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[lth] if self.is_1dconv else kernel_sizes[lth][0]
            stride = strides[lth] if self.is_1dconv else strides[lth][0]
            pooling = poolings[lth] if self.is_1dconv else poolings[lth][0]
            lookahead = (kernel_size - 1) // 2
            lookahead *= 2
            if factor == 1:
                self._context_size += lookahead
                context_size_bottom = self._context_size
            else:
                self._context_size += context_size_bottom * lookahead
                context_size_bottom *= stride * pooling
            factor *= stride * pooling

    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun_normal(n, p, param_init)

    def forward(self, xs, xlens, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTenfor): `[B]` (on CPU)
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTenfor): `[B]` (on CPU)

        """
        B, T, F = xs.size()
        C_i = self.in_channel
        if not self.is_1dconv:
            xs = xs.view(B, T, C_i, F // C_i).contiguous().transpose(2, 1)
        for block in self.layers:
            xs, xlens = block(xs, xlens, lookback=lookback, lookahead=lookahead)
        if not self.is_1dconv:
            B, C_o, T, F = xs.size()
            xs = xs.transpose(2, 1).contiguous().view(B, T, -1)
        if self.bridge is not None:
            xs = self.bridge(xs)
        return xs, xlens


class DropSubsampler(nn.Module):
    """Subsample by dropping input frames."""

    def __init__(self, subsampling_factor):
        super(DropSubsampler, self).__init__()
        self.factor = subsampling_factor

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens
        if batch_first:
            xs = xs[:, ::self.factor]
        else:
            xs = xs[::self.factor]
        xlens = [max(1, math.ceil(i.item() / self.factor)) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class MaxPoolSubsampler(nn.Module):
    """Subsample by max-pooling input frames."""

    def __init__(self, subsampling_factor):
        super(MaxPoolSubsampler, self).__init__()
        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.pool = nn.MaxPool1d(kernel_size=subsampling_factor, stride=subsampling_factor, padding=0, ceil_mode=True)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens
        if batch_first:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()
        else:
            xs = self.pool(xs.permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens


class MeanPoolSubsampler(nn.Module):
    """Subsample by mean-pooling input frames."""

    def __init__(self, subsampling_factor):
        super(MeanPoolSubsampler, self).__init__()
        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=subsampling_factor, stride=subsampling_factor, padding=0, ceil_mode=True)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens
        if batch_first:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()
        else:
            xs = self.pool(xs.permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens


class Padding(nn.Module):
    """Padding variable length of sequences."""

    def __init__(self, bidir_sum_fwd_bwd):
        super(Padding, self).__init__()
        self.bidir_sum = bidir_sum_fwd_bwd

    def forward(self, xs, xlens, rnn, prev_state=None, streaming=False):
        if not streaming and xlens is not None:
            xs = pack_padded_sequence(xs, xlens.tolist(), batch_first=True)
            xs, state = rnn(xs, hx=prev_state)
            xs = pad_packed_sequence(xs, batch_first=True)[0]
        else:
            xs, state = rnn(xs, hx=prev_state)
        if self.bidir_sum:
            assert rnn.bidirectional
            half = xs.size(-1) // 2
            xs = xs[:, :, :half] + xs[:, :, half:]
        return xs, state


def chunkwise(xs, N_l, N_c, N_r, padding=True):
    """Slice input frames chunk by chunk and regard each chunk (with left and
        right contexts) as a single utterance for efficient training of
        latency-controlled bidirectional encoder.

    Args:
        xs (FloatTensor): `[B, T, input_dim]`
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        N_r (int): number of frames for right context
    Returns:
        xs (FloatTensor): `[B * n_chunks, N_l + N_c + N_r, input_dim]`
            where n_chunks = ceil(T / N_c)

    """
    bs, xmax, idim = xs.size()
    n_chunks = math.ceil(xmax / N_c) if padding else xmax // (N_l + N_c + N_r)
    xs_tmp = xs.new_zeros(bs, n_chunks, N_l + N_c + N_r, idim)
    if padding:
        xs = torch.cat([xs.new_zeros(bs, N_l, idim), xs, xs.new_zeros(bs, N_r, idim)], dim=1)
    t = N_l
    for chunk_idx in range(n_chunks):
        xs_chunk = xs[:, t - N_l:t + (N_c + N_r)]
        xs_tmp[:, chunk_idx, :xs_chunk.size(1), :] = xs_chunk
        t += N_c
    xs = xs_tmp.view(bs * n_chunks, N_l + N_c + N_r, idim)
    return xs


def init_with_uniform(n, p, param_init):
    """Initialize with uniform distribution.

    Args:
        n (str): parameter name
        p (Tensor): parameter
        param_init (float):

    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.0)
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.0))
    elif p.dim() in [2, 3, 4]:
        nn.init.uniform_(p, a=-param_init, b=param_init)
        logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
    else:
        raise ValueError(n)


class RNNEncoder(EncoderBase):
    """RNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder (including pure CNN layers)
        n_units (int): number of units in each layer
        n_projs (int): number of units in each projection layer
        last_proj_dim (int): dimension of the last projection layer
        n_layers (int): number of layers
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probability for hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [1, 2, 2, 1] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): subsampling type in intermediate layers
        n_stacks (int): number of frames to stack
        n_splices (int): number of frames to splice
        frontend_conv (nn.Module): frontend CNN module
        bidir_sum_fwd_bwd (bool): sum up forward and backward outputs for dimension reduction
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (float): model initialization parameter
        chunk_size_current (str): current chunk size for latency-controlled bidirectional encoder
        chunk_size_right (str): right chunk size for latency-controlled bidirectional encoder
        cnn_lookahead (bool): enable lookahead for frontend CNN layers for LC-BLSTM
        rsp_prob (float): probability of Random State Passing (RSP)

    """

    def __init__(self, input_dim, enc_type, n_units, n_projs, last_proj_dim, n_layers, n_layers_sub1, n_layers_sub2, dropout_in, dropout, subsample, subsample_type, n_stacks, n_splices, frontend_conv, bidir_sum_fwd_bwd, task_specific_layer, param_init, chunk_size_current, chunk_size_right, cnn_lookahead, rsp_prob):
        super(RNNEncoder, self).__init__()
        subsamples = [1] * n_layers
        for lth, s in enumerate(list(map(int, subsample.split('_')[:n_layers]))):
            subsamples[lth] = s
        if len(subsamples) > 0 and len(subsamples) != n_layers:
            raise ValueError('subsample must be the same size as n_layers. n_layers: %d, subsample: %s' % (n_layers, subsamples))
        if n_layers_sub1 < 0 or n_layers_sub1 > 1 and n_layers < n_layers_sub1:
            raise Warning('Set n_layers_sub1 between 1 to n_layers. n_layers: %d, n_layers_sub1: %d' % (n_layers, n_layers_sub1))
        if n_layers_sub2 < 0 or n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2:
            raise Warning('Set n_layers_sub2 between 1 to n_layers_sub1. n_layers_sub1: %d, n_layers_sub2: %d' % (n_layers_sub1, n_layers_sub2))
        self.enc_type = enc_type
        self.bidirectional = True if 'blstm' in enc_type else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_layers = n_layers
        self.bidir_sum = bidir_sum_fwd_bwd
        chunk_size_current = str(chunk_size_current)
        chunk_size_right = str(chunk_size_right)
        self.N_c = int(chunk_size_current.split('_')[0]) // n_stacks
        self.N_r = int(chunk_size_right.split('_')[0]) // n_stacks
        self.lc_bidir = (self.N_c > 0 or self.N_r > 0) and self.bidirectional
        if self.lc_bidir:
            assert enc_type not in ['lstm', 'conv_lstm']
            assert n_layers_sub2 == 0
        self.rsp_prob = rsp_prob
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None
        self.dropout_in = nn.Dropout(p=dropout_in)
        self.conv = frontend_conv
        if self.conv is not None:
            self._odim = self.conv.output_dim
        else:
            self._odim = input_dim * n_splices * n_stacks
        self.cnn_lookahead = cnn_lookahead
        if not cnn_lookahead:
            assert self.N_c > 0
            assert self.lc_bidir
        if enc_type != 'conv':
            self.rnn = nn.ModuleList()
            if self.lc_bidir:
                self.rnn_bwd = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout)
            self.proj = nn.ModuleList() if n_projs > 0 else None
            self.subsample = nn.ModuleList() if np.prod(subsamples) > 1 else None
            self.padding = Padding(bidir_sum_fwd_bwd=bidir_sum_fwd_bwd if not self.lc_bidir else False)
            for lth in range(n_layers):
                if self.lc_bidir:
                    self.rnn += [nn.LSTM(self._odim, n_units, 1, batch_first=True)]
                    self.rnn_bwd += [nn.LSTM(self._odim, n_units, 1, batch_first=True)]
                else:
                    self.rnn += [nn.LSTM(self._odim, n_units, 1, batch_first=True, bidirectional=self.bidirectional)]
                self._odim = n_units if bidir_sum_fwd_bwd else n_units * self.n_dirs
                if lth == n_layers_sub1 - 1 and task_specific_layer:
                    self.layer_sub1 = nn.Linear(self._odim, n_units)
                    self._odim_sub1 = n_units
                    if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                        self.bridge_sub1 = nn.Linear(n_units, last_proj_dim)
                        self._odim_sub1 = last_proj_dim
                if lth == n_layers_sub2 - 1 and task_specific_layer:
                    assert not self.lc_bidir
                    self.layer_sub2 = nn.Linear(self._odim, n_units)
                    self._odim_sub2 = n_units
                    if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                        self.bridge_sub2 = nn.Linear(n_units, last_proj_dim)
                        self._odim_sub2 = last_proj_dim
                if self.proj is not None:
                    if lth != n_layers - 1:
                        self.proj += [nn.Linear(self._odim, n_projs)]
                        self._odim = n_projs
                if np.prod(subsamples) > 1:
                    if subsample_type == 'max_pool':
                        self.subsample += [MaxPoolSubsampler(subsamples[lth])]
                    elif subsample_type == 'mean_pool':
                        self.subsample += [MeanPoolSubsampler(subsamples[lth])]
                    elif subsample_type == 'concat':
                        self.subsample += [ConcatSubsampler(subsamples[lth], self._odim)]
                    elif subsample_type == 'drop':
                        self.subsample += [DropSubsampler(subsamples[lth])]
                    elif subsample_type == 'conv1d':
                        self.subsample += [Conv1dSubsampler(subsamples[lth], self._odim)]
                    elif subsample_type == 'add':
                        self.subsample += [AddSubsampler(subsamples[lth])]
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge = nn.Linear(self._odim, last_proj_dim)
                self._odim = last_proj_dim
        self.conv_factor = self.conv.subsampling_factor if self.conv is not None else 1
        self._factor = self.conv_factor
        self._factor_sub1 = self.conv_factor
        self._factor_sub2 = self.conv_factor
        if n_layers_sub1 > 1:
            self._factor_sub1 *= np.prod(subsamples[:n_layers_sub1 - 1])
        if n_layers_sub2 > 1:
            self._factor_sub1 *= np.prod(subsamples[:n_layers_sub2 - 1])
        self._factor *= np.prod(subsamples)
        if self.N_c > 0:
            assert self.N_c % self._factor == 0
        if self.N_r > 0:
            assert self.N_r % self._factor == 0
        self.reset_parameters(param_init)
        self.reset_cache()

    @staticmethod
    def add_args(parser, args):
        group = parser.add_argument_group('RNN encoder')
        parser = ConvEncoder.add_args(parser, args)
        group.add_argument('--enc_n_units', type=int, default=512, help='number of units in each encoder RNN layer')
        group.add_argument('--enc_n_projs', type=int, default=0, help='number of units in the projection layer after each encoder RNN layer')
        group.add_argument('--bidirectional_sum_fwd_bwd', type=strtobool, default=False, help='sum forward and backward RNN outputs for dimension reduction')
        group.add_argument('--lc_chunk_size_left', type=str, default='-1', help='current chunk size for latency-controlled RNN encoder')
        group.add_argument('--lc_chunk_size_right', type=str, default='0', help='right chunk size for latency-controlled RNN encoder')
        group.add_argument('--cnn_lookahead', type=strtobool, default=True, help='disable lookahead frames in CNN layers')
        group.add_argument('--rsp_prob_enc', type=float, default=0.0, help='probability for Random State Passing (RSP)')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        if 'conv' in args.enc_type:
            dir_name = ConvEncoder.define_name(dir_name, args)
        dir_name += str(args.enc_n_units) + 'H'
        if args.enc_n_projs > 0:
            dir_name += str(args.enc_n_projs) + 'P'
        dir_name += str(args.enc_n_layers) + 'L'
        if args.bidirectional_sum_fwd_bwd:
            dir_name += '_sumfwdbwd'
        if int(str(args.lc_chunk_size_left).split('_')[0]) > 0 or int(str(args.lc_chunk_size_right).split('_')[0]) > 0:
            dir_name += '_chunkL' + str(args.lc_chunk_size_left) + 'R' + str(args.lc_chunk_size_right)
            if not args.cnn_lookahead:
                dir_name += '_blockwise'
        if args.rsp_prob_enc > 0:
            dir_name += '_RSP' + str(args.rsp_prob_enc)
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'conv' in n:
                continue
            init_with_uniform(n, p, param_init)

    def reset_cache(self):
        self.hx_fwd = [None] * self.n_layers
        logger.debug('Reset cache.')

    def forward(self, xs, xlens, task, streaming=False, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (list): A list of length `[B]`
            task (str): all or ys or ys_sub1 or ys_sub2
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
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
        perm_ids_unsort = None
        if not self.lc_bidir:
            xlens, perm_ids = torch.IntTensor(xlens).sort(0, descending=True)
            xs = xs[perm_ids]
            _, perm_ids_unsort = perm_ids.sort()
        xs = self.dropout_in(xs)
        bs, xmax, idim = xs.size()
        N_c, N_r = self.N_c, self.N_r
        if self.lc_bidir and not self.cnn_lookahead:
            xs = chunkwise(xs, 0, N_c, 0)
            xs = xs.contiguous().view(bs, -1, xs.size(2))
            xs = xs[:, :xlens.max()]
        if self.conv is not None:
            xs, xlens = self.conv(xs, xlens, lookback=lookback, lookahead=lookahead)
            if self.enc_type == 'conv':
                eouts['ys']['xs'] = xs
                eouts['ys']['xlens'] = xlens
                return eouts
            if self.lc_bidir:
                N_c = N_c // self.conv_factor
                N_r = N_r // self.conv_factor
        carry_over = self.rsp_prob > 0 and self.training and random.random() < self.rsp_prob
        carry_over = carry_over and bs == (self.hx_fwd[0][0].size(0) if self.hx_fwd[0] is not None else 0)
        if not streaming and not carry_over:
            self.reset_cache()
        if self.lc_bidir:
            if self.N_c <= 0:
                xs, xlens, xs_sub1, xlens_sub1 = self._forward_full_context(xs, xlens)
            else:
                xs, xlens, xs_sub1, xlens_sub1 = self._forward_latency_controlled(xs, xlens, N_c, N_r, streaming)
            if task == 'ys_sub1':
                eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                return eouts
        else:
            for lth in range(self.n_layers):
                self.rnn[lth].flatten_parameters()
                xs, state = self.padding(xs, xlens, self.rnn[lth], prev_state=self.hx_fwd[lth], streaming=streaming)
                self.hx_fwd[lth] = state
                xs = self.dropout(xs)
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1, xlens_sub1 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub1')
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2, xlens_sub2 = self.sub_module(xs, xlens, perm_ids_unsort, 'sub2')
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens_sub2
                        return eouts
                if self.proj is not None and lth != self.n_layers - 1:
                    xs = torch.relu(self.proj[lth](xs))
                if self.subsample is not None:
                    xs, xlens = self.subsample[lth](xs, xlens)
        if self.bridge is not None:
            xs = self.bridge(xs)
        xs = xs[:, :xlens.max()]
        if task in ['all', 'ys']:
            if perm_ids_unsort is not None:
                xs = xs[perm_ids_unsort]
                xlens = xlens[perm_ids_unsort]
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens_sub2
        return eouts

    def _forward_full_context(self, xs, xlens, task='all'):
        """Full context BPTT encoding.
           This is used for pre-training latency-controlled bidirectional encoder.

        Args:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            task (str):
        Returns:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            xs_sub1 (FloatTensor): `[B, T, n_units]`
            xlens_sub1 (IntTensor): `[B]`

        """
        xs_sub1, xlens_sub1 = None, None
        for lth in range(self.n_layers):
            self.rnn[lth].flatten_parameters()
            self.rnn_bwd[lth].flatten_parameters()
            xs_bwd = torch.flip(self.rnn_bwd[lth](torch.flip(xs, dims=[1]))[0], dims=[1])
            xs_fwd, self.hx_fwd[lth] = self.rnn[lth](xs, hx=self.hx_fwd[lth])
            if self.bidir_sum:
                xs = xs_fwd + xs_bwd
            else:
                xs = torch.cat([xs_fwd, xs_bwd], dim=-1)
            xs = self.dropout(xs)
            if lth == self.n_layers_sub1 - 1:
                xs_sub1, xlens_sub1 = self.sub_module(xs, xlens, None, 'sub1')
                if task == 'ys_sub1':
                    return None, None, xs_sub1, xlens_sub1
            if self.proj is not None and lth != self.n_layers - 1:
                xs = torch.relu(self.proj[lth](xs))
            if self.subsample is not None:
                xs, xlens = self.subsample[lth](xs, xlens)
        return xs, xlens, xs_sub1, xlens_sub1

    def _forward_latency_controlled(self, xs, xlens, N_c, N_r, streaming, task='all'):
        """Streaming encoding for the latency-controlled bidirectional encoder.

        Args:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            N_c (int):
            N_r (int):
            streaming (bool):
            task (str):
        Returns:
            xs (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`
            xs_sub1 (FloatTensor): `[B, T, n_units]`
            xlens (IntTensor): `[B]`

        """
        bs, xmax, _ = xs.size()
        n_chunks = math.ceil(xmax / N_c)
        if streaming:
            xlens = torch.IntTensor(bs).fill_(min(xmax, N_c))
        xlens_sub1 = xlens.clone() if self.n_layers_sub1 > 0 else None
        xs_chunks = []
        xs_chunks_sub1 = []
        for chunk_idx, t in enumerate(range(0, N_c * n_chunks, N_c)):
            xs_chunk = xs[:, t:t + (N_c + N_r)]
            _N_c = N_c
            for lth in range(self.n_layers):
                self.rnn[lth].flatten_parameters()
                self.rnn_bwd[lth].flatten_parameters()
                xs_chunk_bwd = torch.flip(self.rnn_bwd[lth](torch.flip(xs_chunk, dims=[1]))[0], dims=[1])
                if xs_chunk.size(1) <= _N_c:
                    xs_chunk_fwd, self.hx_fwd[lth] = self.rnn[lth](xs_chunk, hx=self.hx_fwd[lth])
                else:
                    xs_chunk_fwd1, self.hx_fwd[lth] = self.rnn[lth](xs_chunk[:, :_N_c], hx=self.hx_fwd[lth])
                    xs_chunk_fwd2, _ = self.rnn[lth](xs_chunk[:, _N_c:], hx=self.hx_fwd[lth])
                    xs_chunk_fwd = torch.cat([xs_chunk_fwd1, xs_chunk_fwd2], dim=1)
                if self.bidir_sum:
                    xs_chunk = xs_chunk_fwd + xs_chunk_bwd
                else:
                    xs_chunk = torch.cat([xs_chunk_fwd, xs_chunk_bwd], dim=-1)
                xs_chunk = self.dropout(xs_chunk)
                if lth == self.n_layers_sub1 - 1:
                    xs_chunks_sub1.append(xs_chunk.clone()[:, :_N_c])
                    if chunk_idx == 0:
                        xlens_sub1 = xlens.clone()
                if self.proj is not None and lth != self.n_layers - 1:
                    xs_chunk = torch.relu(self.proj[lth](xs_chunk))
                if self.subsample is not None:
                    xs_chunk, xlens_tmp = self.subsample[lth](xs_chunk, xlens)
                    if chunk_idx == 0:
                        xlens = xlens_tmp
                    _N_c = _N_c // self.subsample[lth].factor
            xs_chunks.append(xs_chunk[:, :_N_c])
            if streaming:
                break
        xs = torch.cat(xs_chunks, dim=1)
        if self.n_layers_sub1 > 0:
            xs_sub1 = torch.cat(xs_chunks_sub1, dim=1)
            xs_sub1, xlens_sub1 = self.sub_module(xs_sub1, xlens_sub1, None, 'sub1')
        else:
            xs_sub1 = None
        return xs, xlens, xs_sub1, xlens_sub1

    def sub_module(self, xs, xlens, perm_ids_unsort, module='sub1'):
        if self.task_specific_layer:
            xs_sub = self.dropout(torch.relu(getattr(self, 'layer_' + module)(xs)))
        else:
            xs_sub = xs.clone()
        if getattr(self, 'bridge_' + module) is not None:
            xs_sub = getattr(self, 'bridge_' + module)(xs_sub)
        if perm_ids_unsort is not None:
            xs_sub = xs_sub[perm_ids_unsort]
            xlens_sub = xlens[perm_ids_unsort]
        else:
            xlens_sub = xlens.clone()
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


class SubsampleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, in_freq, dropout, layer_norm_eps=1e-12):
        super().__init__()
        self.C_in = in_channel
        self.C_out = out_channel
        self.in_freq = in_freq
        self.conv1d = nn.Conv1d(in_channels=in_freq * in_channel, out_channels=in_freq * out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=in_freq)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm2D(out_channel, in_freq, eps=layer_norm_eps)

    def forward(self, xs, xlens):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, C_o, T, F]`
            xlens (IntTensor): `[B]`

        """
        B, C, T, F = xs.size()
        xs = xs.transpose(3, 2).view(B, C * F, T)
        xs = self.dropout(torch.relu(self.conv1d(xs)))
        xs = xs.view(B, self.C_out, F, -1).transpose(3, 2)
        xs = self.norm(xs)
        xlens = update_lens_1d(xlens, self.conv1d)
        return xs, xlens


class TDSBlock(nn.Module):
    """TDS block.

    Args:
        channel (int): input/output channels size
        kernel_size (int): kernel size
        in_freq (int): frequency width
        dropout (float): dropout probability

    """

    def __init__(self, channel, kernel_size, in_freq, dropout, layer_norm_eps=1e-12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv1d = nn.Conv1d(in_channels=in_freq * channel, out_channels=in_freq * channel, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=in_freq)
        self.norm1 = LayerNorm2D(channel, in_freq, eps=layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(in_channels=in_freq * channel, out_channels=in_freq * channel, kernel_size=1, stride=1, padding=0)
        self.pointwise_conv2 = nn.Conv1d(in_channels=in_freq * channel, out_channels=in_freq * channel, kernel_size=1, stride=1, padding=0)
        self.norm2 = LayerNorm2D(channel, in_freq, eps=layer_norm_eps)

    def forward(self, xs, xlens):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`
            xlens (IntTensor): `[B]`

        """
        B, C, T, F = xs.size()
        residual = xs
        xs = xs.transpose(3, 2).view(B, C * F, T)
        xs = self.dropout(torch.relu(self.conv1d(xs)))
        xs = xs.view(B, -1, F, T).transpose(3, 2)
        xs = xs + residual
        xs = self.norm1(xs)
        B, C, T, F = xs.size()
        residual = xs
        xs = xs.transpose(3, 2).view(B, C * F, T)
        xs = self.dropout(torch.relu(self.pointwise_conv1(xs)))
        xs = self.dropout(self.pointwise_conv2(xs))
        xs = xs.view(B, -1, F, T).transpose(3, 2)
        xs = xs + residual
        xs = self.norm2(xs)
        return xs, xlens


class TDSEncoder(EncoderBase):
    """Time-depth separable convolution (TDS) encoder.

    Args:
        input_dim (int) dimension of input features (freq * channel)
        in_channel (int) number of channels of input features
        channels (list) number of channels in TDS layers
        kernel_sizes (list) size of kernels in TDS layers
        dropout (float) dropout probability
        last_proj_dim (int): dimension of the last projection layer
        layer_norm_eps (float): epsilon value for layer normalization

    """

    def __init__(self, input_dim, in_channel, channels, kernel_sizes, dropout, last_proj_dim, layer_norm_eps=1e-12):
        super(TDSEncoder, self).__init__()
        (channels, kernel_sizes, _, _), _ = parse_cnn_config(channels, kernel_sizes, '', '')
        self.C_in = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.bridge = None
        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes)
        C_i = in_channel
        in_freq = self.input_freq
        n_subsampling = 0
        self.layers = nn.ModuleList()
        for lth in range(len(channels)):
            if C_i != channels[lth]:
                self.layers += [SubsampleBlock(in_channel=C_i, out_channel=channels[lth], kernel_size=kernel_sizes[lth][0], stride=2 if n_subsampling < 3 else 1, in_freq=in_freq, dropout=dropout)]
                n_subsampling += 1
            self.layers += [TDSBlock(channel=channels[lth], kernel_size=kernel_sizes[lth][0], in_freq=in_freq, dropout=dropout, layer_norm_eps=layer_norm_eps)]
            C_i = channels[lth]
        self._odim = int(C_i * in_freq)
        if last_proj_dim > 0:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim
        self._factor = 8
        self.reset_parameters()

    @staticmethod
    def add_args(parser, args):
        parser = ConvEncoder.add_args(parser, args)
        return parser

    @staticmethod
    def define_name(dir_name, args):
        return dir_name

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
            elif p.dim() == 3:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            elif p.dim() == 4:
                fan_in = p.size(1) * p[0][0].numel()
                nn.init.uniform_(p, a=-math.sqrt(4 / fan_in), b=math.sqrt(4 / fan_in))
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', math.sqrt(4 / fan_in)))
            else:
                raise ValueError(n)

    def forward(self, xs, xlens, task, streaming=False, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]`
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T', C_o * F]`
                xlens (IntTensor): `[B]`

        """
        eouts = {'ys': {'xs': None, 'xlens': None}, 'ys_sub1': {'xs': None, 'xlens': None}, 'ys_sub2': {'xs': None, 'xlens': None}}
        B, T, F = xs.size()
        xs = xs.contiguous().view(B, T, self.C_in, F // self.C_in).transpose(2, 1)
        for layer in self.layers:
            xs, xlens = layer(xs, xlens)
        B, C_o, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous().view(B, T, -1)
        if self.bridge is not None:
            xs = self.bridge(xs)
        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        else:
            raise NotImplementedError
        return eouts


class TransformerEncoderBlock(nn.Module):
    """A single layer of the Transformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        pe_type (str): type of positional encoding
        clamp_len (int): maximum relative distance from each position
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer

    """

    def __init__(self, d_model, d_ff, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, pe_type, clamp_len, ffn_bottleneck_dim):
        super(TransformerEncoderBlock, self).__init__()
        self.n_heads = n_heads
        self.rel_attn = pe_type in ['relaive', 'relative_xl']
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if self.rel_attn else MHA
        self.self_attn = mha(kdim=d_model, qdim=d_model, adim=d_model, odim=d_model, n_heads=n_heads, dropout=dropout_att, param_init=param_init, xl_like=pe_type == 'relative_xl', clamp_len=clamp_len)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init, ffn_bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer
        logger.info('Stochastic depth prob: %.3f' % dropout_layer)
        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, cache=None, pos_embs=None, rel_bias=(None, None)):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T (query), d_model]`
            xx_mask (ByteTensor): `[B, T (query), T (key)]`
            cache (dict):
                input_san: `[B, n_cache, d_model]`
            pos_embs (LongTensor): `[T (query), 1, d_model]`
            rel_bias (tuple):
                u_bias (FloatTensor): global parameter for relative positional encoding
                v_bias (FloatTensor): global parameter for relative positional encoding
        Returns:
            xs (FloatTensor): `[B, T (query), d_model]`
            new_cache (dict):
                input_san: `[B, n_cache+T, d_model]`

        """
        self.reset_visualization()
        new_cache = {}
        qlen = xs.size(1)
        u_bias, v_bias = rel_bias
        if self.dropout_layer > 0:
            if self.training and random.random() < self.dropout_layer:
                return xs, new_cache
            else:
                xs = xs / (1 - self.dropout_layer)
        residual = xs
        xs = self.norm1(xs)
        if cache is not None:
            xs = torch.cat([cache['input_san'], xs], dim=1)
        new_cache['input_san'] = xs
        xs_kv = xs
        if cache is not None:
            xs = xs[:, -qlen:]
            residual = residual[:, -qlen:]
            xx_mask = xx_mask[:, -qlen:]
        if self.rel_attn:
            xs, self._xx_aws = self.self_attn(xs_kv, xs, pos_embs, xx_mask, u_bias, v_bias)
        else:
            xs, self._xx_aws = self.self_attn(xs_kv, xs_kv, xs, mask=xx_mask)[:2]
        xs = self.dropout(xs) + residual
        residual = xs
        xs = self.norm2(xs)
        xs = self.feed_forward(xs)
        xs = self.dropout(xs) + residual
        return xs, new_cache


def causal(xx_mask, lookahead):
    """Causal masking.

    Args:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`
        lookahead (int): lookahead frame
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    causal_mask = xx_mask.new_ones(xx_mask.size(1), xx_mask.size(1), dtype=xx_mask.dtype)
    causal_mask = torch.tril(causal_mask, diagonal=lookahead, out=causal_mask).unsqueeze(0)
    xx_mask = xx_mask & causal_mask
    return xx_mask


def make_san_mask(xs, xlens, unidirectional=False, lookahead=0):
    """Mask self-attention mask.

    Args:
        xs (FloatTensor): `[B, T, d_model]`
        xlens (InteTensor): `[B]` (on CPU)
        unidirectional (bool): pad future context
        lookahead (int): lookahead frame
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    xx_mask = make_pad_mask(xlens)
    xx_mask = xx_mask.unsqueeze(1).repeat([1, xlens.max(), 1])
    if unidirectional:
        xx_mask = causal(xx_mask, lookahead)
    return xx_mask


def make_chunkwise_san_mask(xs, xlens, N_l, N_c, n_chunks):
    """Mask self-attention mask for chunkwise processing.

    Args:
        xs (FloatTensor): `[B, T, d_model]`
        xlens (InteTensor): `[B]` (on CPU)
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        n_chunks (int): number of chunks
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    xx_mask = make_san_mask(xs, xlens)
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * N_c
        xx_mask[:, offset:offset + N_c, :max(0, offset - N_l)] = 0
        xx_mask[:, offset:offset + N_c, offset + N_c:] = 0
    return xx_mask


class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder
        n_heads (int): number of heads for multi-head attention
        n_layers (int): number of blocks
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        last_proj_dim (int): dimension of the last projection layer
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        subsample (List): subsample in the corresponding Transformer layers
            ex.) [1, 2, 2, 1] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): subsampling type in intermediate layers
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        frontend_conv (nn.Module): frontend CNN module
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (str): parameter initialization method
        clamp_len (int): maximum length for relative positional encoding
        lookahead (int): lookahead frames per layer for unidirectional Transformer encoder
        chunk_size_left (int): left chunk size for latency-controlled Transformer encoder
        chunk_size_current (int): current chunk size for latency-controlled Transformer encoder
        chunk_size_right (int): right chunk size for latency-controlled Transformer encoder
        streaming_type (str): implementation methods of latency-controlled Transformer encoder

    """

    def __init__(self, input_dim, enc_type, n_heads, n_layers, n_layers_sub1, n_layers_sub2, d_model, d_ff, ffn_bottleneck_dim, ffn_activation, pe_type, layer_norm_eps, last_proj_dim, dropout_in, dropout, dropout_att, dropout_layer, subsample, subsample_type, n_stacks, n_splices, frontend_conv, task_specific_layer, param_init, clamp_len, lookahead, chunk_size_left, chunk_size_current, chunk_size_right, streaming_type):
        super(TransformerEncoder, self).__init__()
        self.subsample_factors = [1] * n_layers
        for lth, s in enumerate(list(map(int, subsample.split('_')[:n_layers]))):
            self.subsample_factors[lth] = s
        lookaheads = [0] * n_layers
        for lth, s in enumerate(list(map(int, lookahead.split('_')[:n_layers]))):
            lookaheads[lth] = s
        if n_layers_sub1 < 0 or n_layers_sub1 > 1 and n_layers < n_layers_sub1:
            raise Warning('Set n_layers_sub1 between 1 to n_layers. n_layers: %d, n_layers_sub1: %d' % (n_layers, n_layers_sub1))
        if n_layers_sub2 < 0 or n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2:
            raise Warning('Set n_layers_sub2 between 1 to n_layers_sub1. n_layers_sub1: %d, n_layers_sub2: %d' % (n_layers_sub1, n_layers_sub2))
        self.enc_type = enc_type
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.scale = math.sqrt(d_model)
        chunk_size_left = str(chunk_size_left)
        chunk_size_current = str(chunk_size_current)
        chunk_size_right = str(chunk_size_right)
        self.unidir = 'uni' in enc_type
        self.lookaheads = lookaheads
        if sum(lookaheads) > 0:
            assert self.unidir
        self.N_l = int(chunk_size_left.split('_')[-1]) // n_stacks
        self.N_c = int(chunk_size_current.split('_')[-1]) // n_stacks
        self.N_r = int(chunk_size_right.split('_')[-1]) // n_stacks
        self.lc_bidir = self.N_c > 0 and enc_type != 'conv' and 'uni' not in enc_type
        self.cnn_lookahead = self.unidir or enc_type == 'conv'
        self.streaming_type = streaming_type if self.lc_bidir else ''
        self.causal = self.unidir or self.streaming_type == 'mask'
        if self.unidir:
            assert self.N_l == self.N_c == self.N_r == 0
        if self.streaming_type == 'mask':
            assert self.N_r == 0
            assert self.N_l % self.N_c == 0
        if self.lc_bidir:
            assert n_layers_sub1 == 0
            assert n_layers_sub2 == 0
            assert not self.unidir
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None
        self.aws_dict = {}
        self.data_dict = {}
        self.conv = frontend_conv
        if self.conv is not None:
            self._odim = self.conv.output_dim
        else:
            self._odim = input_dim * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)
        self._factor = 1
        self.conv_factor = self.conv.subsampling_factor if self.conv is not None else 1
        self._factor *= self.conv_factor
        self.subsample_layers = None
        if np.prod(self.subsample_factors) > 1:
            self._factor *= np.prod(self.subsample_factors)
            if subsample_type == 'max_pool':
                self.subsample_layers = nn.ModuleList([MaxPoolSubsampler(factor) for factor in self.subsample_factors])
            elif subsample_type == 'mean_pool':
                self.subsample_layers = nn.ModuleList([MeanPoolSubsampler(factor) for factor in self.subsample_factors])
            elif subsample_type == 'concat':
                self.subsample_layers = nn.ModuleList([ConcatSubsampler(factor, self._odim) for factor in self.subsample_factors])
            elif subsample_type == 'drop':
                self.subsample_layers = nn.ModuleList([DropSubsampler(factor) for factor in self.subsample_factors])
            elif subsample_type == 'conv1d':
                assert not self.causal
                self.subsample_layers = nn.ModuleList([Conv1dSubsampler(factor, self._odim) for factor in self.subsample_factors])
            elif subsample_type == 'add':
                self.subsample_layers = nn.ModuleList([AddSubsampler(factor) for factor in self.subsample_factors])
            else:
                raise NotImplementedError(subsample_type)
        assert self.N_l % self._factor == 0
        assert self.N_c % self._factor == 0
        assert self.N_r % self._factor == 0
        self.pos_enc, self.pos_emb = None, None
        self.u_bias, self.v_bias = None, None
        if pe_type in ['relative', 'relative_xl']:
            self.pos_emb = XLPositionalEmbedding(d_model, dropout)
            if pe_type == 'relative_xl':
                self.u_bias = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
                self.v_bias = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
        else:
            self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type, param_init)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderBlock(d_model, d_ff, n_heads, dropout, dropout_att, dropout_layer * (lth + 1) / n_layers, layer_norm_eps, ffn_activation, param_init, pe_type, clamp_len, ffn_bottleneck_dim)) for lth in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._odim = d_model
        if n_layers_sub1 > 0:
            if task_specific_layer:
                self.layer_sub1 = TransformerEncoderBlock(d_model, d_ff, n_heads, dropout, dropout_att, dropout_layer * n_layers_sub1 / n_layers, layer_norm_eps, ffn_activation, param_init, pe_type, clamp_len, ffn_bottleneck_dim)
            odim_sub1 = d_model
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub1 = nn.Linear(self._odim, last_proj_dim)
                odim_sub1 = last_proj_dim
            if n_layers_sub1 == n_layers:
                self.norm_out_sub1 = None
            else:
                self.norm_out_sub1 = nn.LayerNorm(odim_sub1, eps=layer_norm_eps)
        if n_layers_sub2 > 0:
            if task_specific_layer:
                self.layer_sub2 = TransformerEncoderBlock(d_model, d_ff, n_heads, dropout, dropout_att, dropout_layer * n_layers_sub2 / n_layers, layer_norm_eps, ffn_activation, param_init, pe_type, clamp_len, ffn_bottleneck_dim)
            odim_sub2 = d_model
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub2 = nn.Linear(self._odim, last_proj_dim)
                odim_sub2 = last_proj_dim
            if n_layers_sub2 == n_layers:
                self.norm_out_sub2 = None
            else:
                self.norm_out_sub2 = nn.LayerNorm(odim_sub2, eps=layer_norm_eps)
        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim
        self.reset_parameters(param_init)
        self.reset_cache()
        self.cache_sizes = self.calculate_cache_size()

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('Transformer encoder')
        if 'conv' in args.enc_type:
            parser = ConvEncoder.add_args(parser, args)
        if not hasattr(args, 'transformer_layer_norm_eps'):
            group.add_argument('--transformer_ffn_bottleneck_dim', type=int, default=0, help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_input_bottleneck_dim', type=int, default=0, help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12, help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu', choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'], help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform', choices=['xavier_uniform', 'pytorch'], help='parameter initialization')
        group.add_argument('--transformer_enc_d_model', type=int, default=256, help='number of units in the MHA layer for Transformer encoder')
        group.add_argument('--transformer_enc_d_ff', type=int, default=2048, help='number of units in the FFN layer for Transformer encoder')
        group.add_argument('--transformer_enc_n_heads', type=int, default=4, help='number of heads in the MHA layer for Transformer encoder')
        group.add_argument('--transformer_enc_pe_type', type=str, default='add', choices=['add', 'none', 'relative', 'relative_xl'], help='type of positional encoding for Transformer encoder')
        group.add_argument('--dropout_enc_layer', type=float, default=0.0, help='LayerDrop probability for Transformer encoder layers')
        group.add_argument('--transformer_enc_clamp_len', type=int, default=-1, help='maximum length for relative positional encoding. -1 means infinite length.')
        group.add_argument('--transformer_enc_lookaheads', type=str, default='0_0_0_0_0_0_0_0_0_0_0_0', help='lookahead frames per layer for unidirectional Transformer encoder')
        group.add_argument('--lc_chunk_size_left', type=str, default='0', help='left chunk size for latency-controlled Transformer encoder')
        group.add_argument('--lc_chunk_size_current', type=str, default='0', help='current chunk size (and hop size) for latency-controlled Transformer encoder')
        group.add_argument('--lc_chunk_size_right', type=str, default='0', help='right chunk size for latency-controlled Transformer encoder')
        group.add_argument('--lc_type', type=str, default='reshape', choices=['reshape', 'mask'], help='implementation methods of latency-controlled Transformer encoder')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        if 'conv' in args.enc_type:
            dir_name = ConvEncoder.define_name(dir_name, args)
        dir_name += str(args.transformer_enc_d_model) + 'dmodel'
        dir_name += str(args.transformer_enc_d_ff) + 'dff'
        if args.transformer_ffn_bottleneck_dim > 0:
            dir_name += str(args.transformer_ffn_bottleneck_dim) + 'bn'
        dir_name += str(args.enc_n_layers) + 'L'
        dir_name += str(args.transformer_enc_n_heads) + 'H'
        dir_name += 'pe' + str(args.transformer_enc_pe_type)
        if args.transformer_enc_clamp_len > 0:
            dir_name += '_clamp' + str(args.transformer_enc_clamp_len)
        if args.dropout_enc_layer > 0:
            dir_name += '_LD' + str(args.dropout_enc_layer)
        if int(str(args.lc_chunk_size_left).split('_')[-1]) > 0 or int(str(args.lc_chunk_size_current).split('_')[-1]) > 0 or int(str(args.lc_chunk_size_right).split('_')[-1]) > 0:
            dir_name += '_chunkL' + str(args.lc_chunk_size_left) + 'C' + str(args.lc_chunk_size_current) + 'R' + str(args.lc_chunk_size_right)
            dir_name += '_' + args.lc_type
        elif sum(list(map(int, args.transformer_enc_lookaheads.split('_')))) > 0:
            dir_name += '_LA' + str(sum(list(map(int, args.transformer_enc_lookaheads.split('_')))))
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if param_init == 'xavier_uniform':
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
            if self.pe_type == 'relative_xl':
                nn.init.xavier_uniform_(self.u_bias)
                nn.init.xavier_uniform_(self.v_bias)

    def reset_cache(self):
        """Reset state cache for streaming infernece."""
        self.cache = [None] * self.n_layers
        self.offset = 0
        logger.debug('Reset cache.')

    def truncate_cache(self, cache):
        """Truncate cache (left context) for streaming inference.

        Args:
            cache (List[FloatTensor]): list of `[B, cache_size+T, d_model]`
        Returns:
            cache (List[FloatTensor]): list of `[B, cache_size, d_model]`

        """
        if cache[0] is not None:
            for lth in range(self.n_layers):
                cache_size = self.cache_sizes[lth]
                if cache[lth]['input_san'].size(1) > cache_size:
                    cache[lth]['input_san'] = cache[lth]['input_san'][:, -cache_size:]
        return cache

    def calculate_cache_size(self):
        """Calculate the maximum cache size per layer."""
        cache_size = self._total_chunk_size_left()
        N_l = self.N_l // self.conv_factor
        cache_sizes = []
        for lth in range(self.n_layers):
            cache_sizes.append(cache_size)
            if self.lc_bidir:
                cache_size = max(0, cache_size - N_l)
                N_l //= self.subsample_factors[lth]
            cache_size //= self.subsample_factors[lth]
        return cache_sizes

    def _total_chunk_size_left(self):
        """Calculate the total left context size accumulated by layer depth.
           This corresponds to the frame length after CNN subsampling.
        """
        if self.streaming_type == 'reshape':
            return self.N_l // self.conv_factor
        elif self.streaming_type == 'mask':
            return self.N_l // self.conv_factor * self.n_layers
        elif self.unidir:
            return 10000 // self.conv_factor
        else:
            return 10000 // self.conv_factor

    def forward(self, xs, xlens, task, streaming=False, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (InteTensor): `[B]` (on CPU)
            task (str): ys/ys_sub1/ys_sub2
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (InteTensor): `[B]` (on CPU)

        """
        eouts = {'ys': {'xs': None, 'xlens': None}, 'ys_sub1': {'xs': None, 'xlens': None}, 'ys_sub2': {'xs': None, 'xlens': None}}
        bs, xmax = xs.size()[:2]
        n_chunks = 0
        unidir = self.unidir
        lc_bidir = self.lc_bidir
        N_l, N_c, N_r = self.N_l, self.N_c, self.N_r
        if streaming and self.streaming_type == 'mask':
            assert xmax <= N_c
        elif streaming and self.streaming_type == 'reshape':
            assert xmax <= N_l + N_c + N_r
        if lc_bidir:
            if self.streaming_type == 'mask' and not streaming:
                xs = chunkwise(xs, 0, N_c, 0, padding=True)
            elif self.streaming_type == 'reshape':
                xs = chunkwise(xs, N_l, N_c, N_r, padding=not streaming)
            n_chunks = xs.size(0) // bs
            assert bs * n_chunks == xs.size(0)
            if streaming:
                assert n_chunks == 1, xs.size()
        if self.conv is None:
            xs = self.embed(xs)
        else:
            xs, xlens = self.conv(xs, xlens, lookback=False if lc_bidir else lookback, lookahead=False if lc_bidir else lookahead)
            N_l = max(0, N_l // self.conv_factor)
            N_c = N_c // self.conv_factor
            N_r = N_r // self.conv_factor
        emax = xs.size(1)
        if streaming and self.streaming_type != 'reshape':
            xs = xs[:, :xlens.max()]
            xlens.clamp_(max=xs.size(1))
        elif not streaming and self.streaming_type == 'mask':
            xs = xs.contiguous().view(bs, -1, xs.size(2))[:, :xlens.max()]
        if self.enc_type == 'conv':
            eouts['ys']['xs'] = xs
            eouts['ys']['xlens'] = xlens
            return eouts
        if streaming:
            self.cache = self.truncate_cache(self.cache)
        else:
            self.reset_cache()
        n_cache = self.cache[0]['input_san'].size(1) if streaming and self.cache[0] is not None else 0
        if 'relative' in self.pe_type:
            xs, rel_pos_embs = self.pos_emb(xs, scale=True, n_cache=n_cache)
        else:
            xs = self.pos_enc(xs, scale=True, offset=self.offset)
            rel_pos_embs = None
        new_cache = [None] * self.n_layers
        if lc_bidir:
            if self.streaming_type == 'mask':
                if streaming:
                    n_chunks = math.ceil((xlens.max().item() + n_cache) / N_c)
                xx_mask = make_chunkwise_san_mask(xs, xlens + n_cache, N_l, N_c, n_chunks)
            else:
                xx_mask = None
            for lth, layer in enumerate(self.layers):
                xs, cache = layer(xs, xx_mask, cache=self.cache[lth], pos_embs=rel_pos_embs, rel_bias=(self.u_bias, self.v_bias))
                if self.streaming_type == 'mask':
                    new_cache[lth] = cache
                if not self.training and not streaming:
                    if self.streaming_type == 'reshape':
                        n_heads = layer.xx_aws.size(1)
                        xx_aws = layer.xx_aws[:, :, N_l:N_l + N_c, N_l:N_l + N_c]
                        xx_aws = xx_aws.view(bs, n_chunks, n_heads, N_c, N_c)
                        emax = xlens.max().item()
                        xx_aws_center = xx_aws.new_zeros(bs, n_heads, emax, emax)
                        for chunk_idx in range(n_chunks):
                            offset = chunk_idx * N_c
                            emax_chunk = xx_aws_center[:, :, offset:offset + N_c].size(2)
                            xx_aws_chunk = xx_aws[:, chunk_idx, :, :emax_chunk, :emax_chunk]
                            xx_aws_center[:, :, offset:offset + N_c, offset:offset + N_c] = xx_aws_chunk
                        self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws_center)
                    elif self.streaming_type == 'mask':
                        self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(layer.xx_aws)
                    self.data_dict['elens%d' % lth] = tensor2np(xlens)
                if lth < len(self.layers) - 1:
                    if self.subsample_factors[lth] > 1:
                        xs, xlens = self.subsample_layers[lth](xs, xlens)
                        N_l = max(0, N_l // self.subsample_factors[lth])
                        N_c //= self.subsample_factors[lth]
                        N_r //= self.subsample_factors[lth]
                    if streaming:
                        n_cache = self.cache[lth + 1]['input_san'].size(1) if self.cache[lth + 1] is not None else 0
                        if 'relative' in self.pe_type:
                            xs, rel_pos_embs = self.pos_emb(xs, n_cache=n_cache)
                        if self.streaming_type == 'mask':
                            xx_mask = make_chunkwise_san_mask(xs, xlens + n_cache, N_l, N_c, n_chunks)
                    elif self.subsample_factors[lth] > 1:
                        if 'relative' in self.pe_type:
                            xs, rel_pos_embs = self.pos_emb(xs)
                        if self.streaming_type == 'mask':
                            xx_mask = make_chunkwise_san_mask(xs, xlens, N_l, N_c, n_chunks)
            if self.streaming_type == 'reshape':
                xs = xs[:, N_l:N_l + N_c]
                xs = xs.contiguous().view(bs, -1, xs.size(2))
                xs = xs[:, :xlens.max()]
        else:
            xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[0])
            for lth, layer in enumerate(self.layers):
                xs, cache = layer(xs, xx_mask, cache=self.cache[lth], pos_embs=rel_pos_embs, rel_bias=(self.u_bias, self.v_bias))
                new_cache[lth] = cache
                if not self.training and not streaming:
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(layer.xx_aws)
                    self.data_dict['elens%d' % lth] = tensor2np(xlens)
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1 = self.sub_module(xs, xx_mask, lth, rel_pos_embs, 'sub1')
                    xlens_sub1 = xlens.clone()
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2 = self.sub_module(xs, xx_mask, lth, rel_pos_embs, 'sub2')
                    xlens_sub2 = xlens.clone()
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens_sub2
                        return eouts
                if lth < len(self.layers) - 1:
                    if self.subsample_factors[lth] > 1:
                        xs, xlens = self.subsample_layers[lth](xs, xlens)
                    if streaming:
                        n_cache = self.cache[lth + 1]['input_san'].size(1) if streaming and self.cache[lth + 1] is not None else 0
                        if 'relative' in self.pe_type:
                            xs, rel_pos_embs = self.pos_emb(xs, n_cache=n_cache)
                        xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[lth + 1])
                    elif self.subsample_factors[lth] > 1:
                        if 'relative' in self.pe_type:
                            xs, rel_pos_embs = self.pos_emb(xs)
                        xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[lth + 1])
                    elif self.lookaheads[lth] != self.lookaheads[lth + 1]:
                        xx_mask = make_san_mask(xs, xlens + n_cache, unidir, self.lookaheads[lth + 1])
        xs = self.norm_out(xs)
        if streaming:
            self.cache = new_cache
            if self.streaming_type != 'reshape':
                self.offset += emax
        if self.bridge is not None:
            xs = self.bridge(xs)
        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens_sub2
        return eouts

    def sub_module(self, xs, xx_mask, lth, pos_embs=None, module='sub1'):
        if self.task_specific_layer:
            xs_sub, cache = getattr(self, 'layer_' + module)(xs, xx_mask, pos_embs=pos_embs)
            if not self.training:
                self.aws_dict['xx_aws_%s_layer%d' % (module, lth)] = tensor2np(getattr(self, 'layer_' + module).xx_aws)
        else:
            xs_sub = xs.clone()
        if getattr(self, 'bridge_' + module) is not None:
            xs_sub = getattr(self, 'bridge_' + module)(xs_sub)
        if getattr(self, 'norm_out_' + module) is not None:
            xs_sub = getattr(self, 'norm_out_' + module)(xs_sub)
        return xs_sub


class SequenceSummaryNetwork(nn.Module):
    """Sequence summary network.

    Args:
        input_dim (int): dimension of input features
        n_units (int):
        n_layers (int):
        bottleneck_dim (int): dimension of the last bottleneck layer
        dropout (float): dropout probability
        param_init (str): parameter initialization method

    """

    def __init__(self, input_dim, n_units, n_layers, bottleneck_dim, dropout, param_init=0.1):
        super(SequenceSummaryNetwork, self).__init__()
        self.n_layers = n_layers
        layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        odim = input_dim
        for lth in range(n_layers - 1):
            layers += [nn.Linear(odim, n_units)]
            layers += [nn.Tanh()]
            layers += [nn.Dropout(p=dropout)]
            odim = n_units
        layers += [nn.Linear(odim, bottleneck_dim if bottleneck_dim > 0 else n_units)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=dropout)]
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Linear(bottleneck_dim if bottleneck_dim > 0 else n_units, input_dim)
        self.reset_parameters(param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_uniform(n, p, param_init)

    def forward(self, xs, xlens):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, input_dim (+Δ, ΔΔ)]`
            xlens (IntTensor): `[B]`
        Returns:
            xs (FloatTensor): `[B, T, input_dim]`

        """
        residual = xs
        xs = self.layers(xs)
        xlens = xlens
        mask = make_pad_mask(xlens).unsqueeze(2)
        xs = xs.clone().masked_fill_(mask == 0, 0)
        denom = xlens.float().unsqueeze(1)
        xs = xs.sum(1) / denom
        xs = residual + self.proj(xs).unsqueeze(1)
        return xs


class BeamSearch(object):

    def __init__(self, beam_width, eos, ctc_weight, lm_weight, device, beam_width_bwd=0):
        super(BeamSearch, self).__init__()
        self.beam_width = beam_width
        self.beam_width_bwd = beam_width_bwd
        self.eos = eos
        self.device = device
        self.ctc_weight = ctc_weight
        self.lm_weight = lm_weight

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
        total_scores_topk += total_scores_ctc * self.ctc_weight
        total_scores_topk, joint_ids_topk = torch.topk(total_scores_topk, k=beam_width, dim=1, largest=True, sorted=True)
        topk_ids = topk_ids[:, joint_ids_topk[0]]
        new_ctc_states = new_ctc_states[joint_ids_topk[0].cpu().numpy()]
        return new_ctc_states, total_scores_ctc, total_scores_topk

    def add_lm_score(self, after_topk=True):
        raise NotImplementedError

    @staticmethod
    def update_rnnlm_state(lm, hyp, y):
        """Update RNNLM state for a single utterance.

        Args:
            lm (RNNLM): RNNLM
            hyp (dict): beam candiate
            y (LongTensor): `[1, 1]`
        Returns:
            lmout (FloatTensor): `[1, 1, lm_n_units]`
            lmstate (dict):
                hxs (FloatTensor): `[n_layers, 1, n_units]`
                cxs (FloatTensor): `[n_layers, 1, n_units]`
            scores_lm (FloatTensor): `[1, 1, vocab]`

        """
        lmout, lmstate, scores_lm = None, None, None
        if lm is not None:
            lmout, lmstate, scores_lm = lm.predict(y, hyp['lmstate'])
        return lmout, lmstate, scores_lm

    @staticmethod
    def update_rnnlm_state_batch(lm, hyps, y):
        """Update RNNLM state in batch-mode.

        Args:
            lm (RNNLM): RNNLM
            hyps (List[dict]): beam candidates
            y (LongTensor): `[B, 1]`
        Returns:
            lmout (FloatTensor): `[B, 1, lm_n_units]`
            lmstate (dict):
                hxs (FloatTensor): `[n_layers, B, n_units]`
                cxs (FloatTensor): `[n_layers, B, n_units]`
            scores_lm (FloatTensor): `[B, 1, vocab]`

        """
        lmout, lmstate, scores_lm = None, None, None
        if lm is not None:
            if hyps[0]['lmstate'] is not None:
                lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
            lmout, lmstate, scores_lm = lm.predict(y, lmstate)
        return lmout, lmstate, scores_lm

    @staticmethod
    def lm_rescoring(hyps, lm, lm_weight, reverse=False, length_norm=False, tag=''):
        if lm is None:
            return hyps
        for i in range(len(hyps)):
            ys = hyps[i]['hyp']
            if reverse:
                ys = ys[::-1]
            ys = [np2tensor(np.fromiter(ys, dtype=np.int64), lm.device)]
            ys_in = pad_list([y[:-1] for y in ys], -1)
            ys_out = pad_list([y[1:] for y in ys], -1)
            if ys_in.size(1) > 0:
                _, _, scores_lm = lm.predict(ys_in, None)
                score_lm = sum([scores_lm[0, t, ys_out[0, t]] for t in range(ys_out.size(1))])
                if length_norm:
                    score_lm /= ys_out.size(1)
            else:
                score_lm = 0
            hyps[i]['score'] += score_lm * lm_weight
            hyps[i]['score_lm_' + tag] = score_lm
        return hyps

    @staticmethod
    def verify_lm_eval_mode(lm, lm_weight, cache_emb=True):
        if lm is not None:
            assert lm_weight > 0
            lm.eval()
            if cache_emb:
                lm.cache_embedding(lm.device)
        return lm

    @staticmethod
    def merge_ctc_path(hyps, merge_prob=False):
        """Merge multiple alignment paths corresponding to the same token IDs for CTC.

        Args:
            hyps (List): length of `[beam_width]`
        Returns:
            hyps (List): length of `[less than beam_width]`

        """
        hyps_merged = {}
        for beam in hyps:
            hyp_ids_str = beam['hyp_ids_str']
            if hyp_ids_str not in hyps_merged.keys():
                hyps_merged[hyp_ids_str] = beam
            elif merge_prob:
                for k in ['score', 'score_ctc']:
                    hyps_merged[hyp_ids_str][k] = np.logaddexp(hyps_merged[hyp_ids_str][k], beam[k])
            elif beam['score'] > hyps_merged[hyp_ids_str]['score']:
                hyps_merged[hyp_ids_str] = beam
        hyps = [v for v in hyps_merged.values()]
        return hyps

    @staticmethod
    def merge_rnnt_path(hyps, merge_prob=False):
        """Merge multiple alignment paths corresponding to the same token IDs for RNN-T.

        Args:
            hyps (List): length of `[beam_width]`
        Returns:
            hyps (List): length of `[less than beam_width]`

        """
        hyps_merged = {}
        for beam in hyps:
            hyp_ids_str = beam['hyp_ids_str']
            if hyp_ids_str not in hyps_merged.keys():
                hyps_merged[hyp_ids_str] = beam
            elif merge_prob:
                for k in ['score', 'score_rnnt']:
                    hyps_merged[hyp_ids_str][k] = np.logaddexp(hyps_merged[hyp_ids_str][k], beam[k])
            elif beam['score'] > hyps_merged[hyp_ids_str]['score']:
                hyps_merged[hyp_ids_str] = beam
        hyps = [v for v in hyps_merged.values()]
        return hyps


LOG_0 = float(np.finfo(np.float32).min)


LOG_1 = 0


def _computes_transition(seq_log_prob, same_transition, outside, cum_log_prob, log_prob_yt, skip_accum=False):
    bs, max_path_len = seq_log_prob.size()
    mat = seq_log_prob.new_zeros(3, bs, max_path_len).fill_(LOG_0)
    mat[0, :, :] = seq_log_prob
    mat[1, :, 1:] = seq_log_prob[:, :-1]
    mat[2, :, 2:] = seq_log_prob[:, :-2]
    mat[2, :, 2:][same_transition] = LOG_0
    seq_log_prob = torch.logsumexp(mat, dim=0)
    seq_log_prob[outside] = LOG_0
    if not skip_accum:
        cum_log_prob += seq_log_prob
    seq_log_prob += log_prob_yt
    return seq_log_prob


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
    rotate = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    return torch.flip(log_probs[rotate[:, :, None], torch.arange(bs, dtype=torch.int64)[None, :, None], torch.arange(vocab, dtype=torch.int64)[None, None, :]], dims=[0])


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
    rotate = (torch.arange(max_path_len) + path_lens[:, None]) % max_path_len
    return torch.flip(path[torch.arange(bs, dtype=torch.int64)[:, None], rotate], dims=[1])


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
    rotate_input = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    rotate_label = (torch.arange(max_path_len, dtype=torch.int64) + path_lens[:, None]) % max_path_len
    return torch.flip(cum_log_prob[rotate_input[:, :, None], torch.arange(bs, dtype=torch.int64)[None, :, None], rotate_label], dims=[0, 2])


def _label_to_path(labels, blank):
    path = labels.new_zeros(labels.size(0), labels.size(1) * 2 + 1).fill_(blank).long()
    path[:, 1::2] = labels
    return path


class CTCForcedAligner(object):

    def __init__(self, blank=0):
        self.blank = blank

    def __call__(self, logits, elens, ys, ylens):
        """Forced alignment with references.

        Args:
            logits (FloatTensor): `[B, T, vocab]`
            elens (List): length `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            ylens (List): length `[B]`
        Returns:
            trigger_points (IntTensor): `[B, L]`

        """
        with torch.no_grad():
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), logits.device) for y in ys]
            ys_in_pad = pad_list(ys, 0)
            mask = make_pad_mask(elens)
            mask = mask.unsqueeze(2).expand_as(logits)
            logits = logits.masked_fill_(mask == 0, LOG_0)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            trigger_points = self.align(log_probs, elens, ys_in_pad, ylens)
        return trigger_points

    def align(self, log_probs, elens, ys, ylens, add_eos=True):
        """Calculte the best CTC alignment with the forward-backward algorithm.
        Args:
            log_probs (FloatTensor): `[T, B, vocab]`
            elens (FloatTensor): `[B]`
            ys (FloatTensor): `[B, L]`
            ylens (FloatTensor): `[B]`
            add_eos (bool): Use the last time index as a boundary corresponding to <eos>
        Returns:
            trigger_points (IntTensor): `[B, L]`

        """
        xmax, bs, vocab = log_probs.size()
        path = _label_to_path(ys, self.blank)
        path_lens = 2 * ylens.long() + 1
        ymax = ys.size(1)
        max_path_len = path.size(1)
        assert ys.size() == (bs, ymax), ys.size()
        assert path.size() == (bs, ymax * 2 + 1)
        alpha = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
        alpha[:, 0] = LOG_1
        beta = alpha.clone()
        gamma = alpha.clone()
        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        frame_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[frame_index, batch_index, path]
        same_transition = path[:, :-2] == path[:, 2:]
        outside = torch.arange(max_path_len, dtype=torch.int64) >= path_lens.unsqueeze(1)
        log_probs_gold = log_probs[:, batch_index, path]
        for t in range(xmax):
            alpha = _computes_transition(alpha, same_transition, outside, log_probs_fwd_bwd[t], log_probs_gold[t])
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(log_probs, elens.long())
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        r_same_transition = r_path[:, :-2] == r_path[:, 2:]
        log_probs_inv_gold = log_probs_inv[:, batch_index, r_path]
        for t in range(xmax):
            beta = _computes_transition(beta, r_same_transition, outside, log_probs_fwd_bwd[t], log_probs_inv_gold[t])
        best_aligns = log_probs.new_zeros((bs, xmax), dtype=torch.int64)
        log_probs_fwd_bwd = _flip_path_probability(log_probs_fwd_bwd, elens.long(), path_lens)
        for t in range(xmax):
            gamma = _computes_transition(gamma, same_transition, outside, log_probs_fwd_bwd[t], log_probs_gold[t], skip_accum=True)
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(gamma == LOG_0, LOG_0)
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_aligns[b, t] = token_idx
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
            for b in range(bs):
                gamma[b, offsets[b]] = LOG_1
        trigger_aligns = torch.zeros((bs, xmax), dtype=torch.int64)
        trigger_points = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)
        for b in range(bs):
            n_triggers = 0
            if add_eos:
                trigger_points[b, ylens[b]] = elens[b] - 1
            for t in range(elens[b]):
                token_idx = best_aligns[b, t]
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_aligns[b, t - 1]):
                    continue
                trigger_aligns[b, t] = token_idx
                trigger_points[b, n_triggers] = t
                n_triggers += 1
        assert ylens.sum() == (trigger_aligns != 0).sum()
        return trigger_points


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):
        super(ModelBase, self).__init__()
        logger.info('Overriding DecoderBase class.')

    def reset_session(self):
        self._new_session = True

    def trigger_scheduled_sampling(self):
        logger.info('Activate scheduled sampling')
        self._ss_prob = getattr(self, 'ss_prob', 0)

    def trigger_quantity_loss(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            logger.info('Activate quantity loss')
            self._quantity_loss_weight = getattr(self, 'quantity_loss_weight', 0)

    def trigger_latency_loss(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            logger.info('Activate latency loss')
            self._latency_loss_weight = getattr(self, 'latency_loss_weight', 0)

    def trigger_stableemit(self):
        if getattr(self, 'attn_type', '') == 'mocha':
            if hasattr(self, 'score'):
                self.score.trigger_stableemit()
            elif hasattr(self, 'layers'):
                pass

    def greedy(self, eouts, elens, max_len_ratio):
        raise NotImplementedError

    def embed_token_id(self, indices):
        raise NotImplementedError

    def cache_embedding(self, device):
        raise NotImplementedError

    def initialize_beam(self, hyp, lmstate):
        raise NotImplementedError

    def beam_search(self, eouts, elens, params, idx2token):
        raise NotImplementedError

    def _plot_attention(self, save_path=None, n_cols=2):
        """Plot attention for each head in all decoder layers."""
        if len(getattr(self, 'aws_dict', {}).keys()) == 0:
            return
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        elens = self.data_dict['elens']
        ylens = self.data_dict['ylens']
        aws_dict = self.aws_dict
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        for k, aw in aws_dict.items():
            if aw is None:
                continue
            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols * max(1, n_heads // 4)
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp, figsize=(20 * max(1, n_heads // 4), 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                if 'yy' in k:
                    ax.imshow(aw[-1, h, :ylens[-1], :ylens[-1]], aspect='auto')
                else:
                    ax.imshow(aw[-1, h, :ylens[-1], :elens[-1]], aspect='auto')
                ax.grid(False)
                ax.set_xlabel('Input (head%d)' % h)
                ax.set_ylabel('Output (head%d)' % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, '%s.png' % k))
            plt.close()

    def _plot_ctc(self, save_path=None, topk=10):
        """Plot CTC posterior probabilities."""
        if self.ctc_weight == 0:
            return
        if len(self.ctc.prob_dict.keys()) == 0:
            return
        from matplotlib import pyplot as plt
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        elen = self.ctc.data_dict['elens'][-1]
        probs = self.ctc.prob_dict['probs'][-1, :elen]
        topk_ids = np.argsort(probs, axis=1)
        plt.clf()
        n_frames = probs.shape[0]
        times_probs = np.arange(n_frames)
        plt.figure(figsize=(20, 8))
        for idx in set(topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(times_probs, probs[:, 0], ':', label='<blank>', color='grey')
            else:
                plt.plot(times_probs, probs[:, idx])
        plt.xlabel(u'Time [frame]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xticks(list(range(0, int(n_frames) + 1, 10)))
        plt.yticks(list(range(0, 2, 1)))
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'prob.png'))
        plt.close()


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
    loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean


class CTC(DecoderBase):
    """Connectionist temporal classification (CTC).

    Args:
        eos (int): index for <eos> (shared with <sos>)
        blank (int): index for <blank>
        enc_n_units (int):
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probability for the RNN layer
        lsm_prob (float): label smoothing probability
        fc_list (List):
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
        self.prev_spk = ''
        self.lmstate_final = None
        self.prob_dict = {}
        self.data_dict = {}
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
        self.use_warpctc = LooseVersion(torch.__version__) < LooseVersion('1.4.0')
        if self.use_warpctc:
            self.ctc_loss = warpctc_pytorch.CTCLoss(size_average=True)
        elif LooseVersion(torch.__version__) < LooseVersion('1.7.0'):
            self.ctc_loss = nn.CTCLoss(reduction='sum')
        else:
            self.ctc_loss = nn.CTCLoss(reduction='sum', zero_infinity=True)
        self.forced_aligner = CTCForcedAligner()

    def forward(self, eouts, elens, ys, forced_align=False):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (List): length `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            trigger_points (IntTensor): `[B, L]`

        """
        ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
        ys_ctc = torch.cat([np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int32)) for y in ys], dim=0)
        logits = self.output(eouts)
        loss = self.loss_fn(logits.transpose(1, 0), ys_ctc, elens, ylens)
        if self.lsm_prob > 0:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits, elens) * self.lsm_prob
        trigger_points = self.forced_aligner(logits.clone(), elens, ys, ylens) if forced_align else None
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.prob_dict['probs'] = tensor2np(torch.softmax(logits, dim=-1))
        return loss, trigger_points

    def loss_fn(self, logits, ys_ctc, elens, ylens):
        if self.use_warpctc:
            loss = self.ctc_loss(logits, ys_ctc, elens.cpu(), ylens)
        else:
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(logits.log_softmax(2), ys_ctc, elens, ylens) / logits.size(1)
        return loss

    def trigger_points(self, eouts, elens):
        """Extract trigger points for inference.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
        Returns:
            trigger_points_pred (IntTensor): `[B, L]`

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
        trigger_points_pred = log_probs.new_zeros((bs, ymax + 1), dtype=torch.int32)
        for b in range(bs):
            n_triggers = 0
            for t in range(elens[b]):
                token_idx = best_paths[b, t]
                if token_idx == self.blank:
                    continue
                if not (t == 0 or token_idx != best_paths[b, t - 1]):
                    continue
                trigger_points_pred[b, n_triggers] = t
                n_triggers += 1
        return trigger_points_pred

    def probs(self, eouts, temperature=1.0):
        """Get CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.softmax(self.output(eouts) / temperature, dim=-1)

    def scores(self, eouts, temperature=1.0):
        """Get log-scale CTC probabilities.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            log_probs (FloatTensor): `[B, T, vocab]`

        """
        return torch.log_softmax(self.output(eouts) / temperature, dim=-1)

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
            hyps.append([best_hyp])
        return hyps

    def initialize_beam(self, hyp, lmstate):
        """Initialize beam."""
        hyps = [{'hyp': hyp, 'hyp_ids_str': '', 'p_b': LOG_1, 'p_nb': LOG_0, 'score_lm': LOG_1, 'lmstate': lmstate, 'update_lm': True}]
        return hyps

    def beam_search(self, eouts, elens, params, idx2token, lm=None, lm_second=None, lm_second_bwd=None, nbest=1, refs_id=None, utt_ids=None, speakers=None):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (List): length `[B]`
            params (dict): decoding hyperparameters
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh-pass LM
            lm_second (torch.nn.module): second-pass LM
            lm_second_bwd (torch.nn.module): second-pass backward LM
            nbest (int): number of N-best list
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
        Returns:
            nbest_hyps_idx (List[List[List]]): Best path hypothesis

        """
        bs = eouts.size(0)
        beam_width = params.get('recog_beam_width')
        lp_weight = params.get('recog_length_penalty')
        cache_emb = params.get('recog_cache_embedding')
        lm_weight = params.get('recog_lm_weight')
        lm_weight_second = params.get('recog_lm_second_weight')
        lm_weight_second_bwd = params.get('recog_lm_bwd_weight')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')
        helper = BeamSearch(beam_width, self.eos, 1.0, lm_weight, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight, cache_emb)
        if lm is not None:
            assert isinstance(lm, RNNLM)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second, cache_emb)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd, cache_emb)
        log_probs = torch.log_softmax(self.output(eouts) * softmax_smoothing, dim=-1)
        nbest_hyps_idx = []
        for b in range(bs):
            lmstate = {'hxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units), 'cxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units)} if lm is not None else None
            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_CO:
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]
            hyps = self.initialize_beam([self.eos], lmstate)
            self.state_cache = OrderedDict()
            hyps, new_hyps = self._beam_search(hyps, helper, log_probs[b], lm, lp_weight)
            end_hyps = hyps[:]
            if len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(new_hyps[:nbest - len(end_hyps)])
            end_hyps = helper.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')
            end_hyps = helper.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, tag='second_bwd')
            end_hyps = sorted(end_hyps, key=lambda x: x['score'] / max(len(x['hyp'][1:]), 1), reverse=True)
            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, ctc): %.7f' % end_hyps[k]['score_ctc'])
                    logger.info('log prob (hyp, lp): %.7f' % (end_hyps[k]['score_lp'] * lp_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-pass lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-pass lm): %.7f' % (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-pass lm, reverse): %.7f' % (end_hyps[k]['score_lm_second_bwd'] * lm_weight_second_bwd))
                    logger.info('-' * 50)
            nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
        if bs == 1:
            self.lmstate_final = end_hyps[0]['lmstate']
        return nbest_hyps_idx

    def _beam_search(self, hyps, helper, scores_ctc, lm, lp_weight):
        beam_width = helper.beam_width
        lm_weight = helper.lm_weight
        merge_prob = True
        for t in range(scores_ctc.size(0)):
            _, topk_ids = torch.topk(scores_ctc[t, 1:], k=min(beam_width, self.vocab), dim=-1, largest=True, sorted=True)
            topk_ids += 1
            batch_hyps = [beam for beam in hyps if beam['update_lm']]
            if len(batch_hyps) > 0:
                ys = scores_ctc.new_zeros((len(batch_hyps), 1), dtype=torch.int64)
                for i, beam in enumerate(batch_hyps):
                    ys[i] = beam['hyp'][-1]
                _, lmstates, scores_lm = helper.update_rnnlm_state_batch(lm, batch_hyps, ys)
                hyp_ids_strs = [beam['hyp_ids_str'] for beam in hyps]
                for i, beam in enumerate(batch_hyps):
                    lmstate = {'hxs': lmstates['hxs'][:, i:i + 1], 'cxs': lmstates['cxs'][:, i:i + 1]} if lmstates is not None else None
                    index = hyp_ids_strs.index(beam['hyp_ids_str'])
                    hyps[index]['lmstate'] = lmstate
                    if lm is not None:
                        hyps[index]['next_scores_lm'] = scores_lm[i:i + 1]
                    else:
                        hyps[index]['next_scores_lm'] = None
                    assert hyps[index]['update_lm']
                    hyps[index]['update_lm'] = False
                    self.state_cache[beam['hyp_ids_str']] = {'next_scores_lm': hyps[index]['next_scores_lm'], 'lmstate': lmstate}
            new_hyps = []
            for j, beam in enumerate(hyps):
                p_b = beam['p_b']
                p_nb = beam['p_nb']
                total_score_lm = beam['score_lm']
                new_p_b = np.logaddexp(p_b + scores_ctc[t, self.blank].item(), p_nb + scores_ctc[t, self.blank].item())
                if len(beam['hyp'][1:]) > 0:
                    new_p_nb = p_nb + scores_ctc[t, beam['hyp'][-1]].item()
                else:
                    new_p_nb = LOG_0
                total_score_ctc = np.logaddexp(new_p_b, new_p_nb)
                total_score_lp = len(beam['hyp'][1:]) * lp_weight
                total_score = total_score_ctc + total_score_lp + total_score_lm * lm_weight
                new_hyps.append({'hyp': beam['hyp'][:], 'hyp_ids_str': beam['hyp_ids_str'], 'score': total_score, 'p_b': new_p_b, 'p_nb': new_p_nb, 'score_ctc': total_score_ctc, 'score_lm': total_score_lm, 'score_lp': total_score_lp, 'next_scores_lm': beam['next_scores_lm'], 'lmstate': beam['lmstate'], 'update_lm': False})
                new_p_b = LOG_0
                for k in range(beam_width):
                    idx = topk_ids[k].item()
                    p_t = scores_ctc[t, idx].item()
                    c_prev = beam['hyp'][-1] if len(beam['hyp']) > 1 else None
                    if idx == c_prev:
                        new_p_nb = p_b + p_t
                    else:
                        new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                        if idx == self.space:
                            pass
                    total_score_ctc = np.logaddexp(new_p_b, new_p_nb)
                    total_score_lp = (len(beam['hyp'][1:]) + 1) * lp_weight
                    total_score = total_score_ctc + total_score_lp
                    if lm is not None:
                        total_score_lm += beam['next_scores_lm'][0, 0, idx].item()
                    total_score += total_score_lm * lm_weight
                    hyp_ids = beam['hyp'] + [idx]
                    hyp_ids_str = ' '.join(list(map(str, hyp_ids)))
                    exist_cache = hyp_ids_str in self.state_cache.keys()
                    if exist_cache:
                        scores_lm = self.state_cache[hyp_ids_str]['next_scores_lm']
                        lmstate = self.state_cache[hyp_ids_str]['lmstate']
                    else:
                        scores_lm = None
                        lmstate = beam['lmstate']
                    new_hyps.append({'hyp': hyp_ids, 'hyp_ids_str': hyp_ids_str, 'score': total_score, 'p_b': new_p_b, 'p_nb': new_p_nb, 'score_ctc': total_score_ctc, 'score_lm': total_score_lm, 'score_lp': total_score_lp, 'next_scores_lm': scores_lm, 'lmstate': lmstate, 'update_lm': not exist_cache})
            new_hyps = sorted(new_hyps, key=lambda x: x['score'], reverse=True)
            new_hyps = helper.merge_ctc_path(new_hyps, merge_prob)
            hyps = new_hyps[:beam_width]
        return hyps, new_hyps

    def beam_search_block_sync(self, eouts, params, helper, idx2token, hyps, lm):
        assert eouts.size(0) == 1
        beam_width = params.get('recog_beam_width')
        lp_weight = params.get('recog_length_penalty')
        lm_weight = params.get('recog_lm_weight')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')
        end_hyps = []
        if hyps is None:
            if lm_state_CO:
                lmstate = self.lmstate_final
            else:
                lmstate = {'hxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units), 'cxs': eouts.new_zeros(lm.n_layers, 1, lm.n_units)} if lm is not None else None
            self.n_frames = 0
            hyps = self.initialize_beam([self.eos], lmstate)
            self.state_cache = OrderedDict()
        log_probs = torch.log_softmax(self.output(eouts) * softmax_smoothing, dim=-1)
        hyps, _ = self._beam_search(hyps, helper, log_probs[0], lm, lp_weight)
        merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
        if idx2token is not None:
            logger.info('=' * 200)
            for k in range(len(merged_hyps)):
                logger.info('Hyp: %s' % idx2token(merged_hyps[k]['hyp'][1:]))
                if len(merged_hyps[k]['hyp']) > 1:
                    logger.info('num tokens (hyp): %d' % len(merged_hyps[k]['hyp'][1:]))
                logger.info('log prob (hyp): %.7f' % merged_hyps[k]['score'])
                logger.info('log prob (hyp, ctc): %.7f' % merged_hyps[k]['score_ctc'])
                if lm is not None:
                    logger.info('log prob (hyp, first-pass lm): %.7f' % (merged_hyps[k]['score_lm'] * lm_weight))
                logger.info('-' * 50)
        if len(merged_hyps) > 0:
            self.lmstate_final = merged_hyps[0]['lmstate']
        self.n_frames += eouts.size(1)
        return end_hyps, hyps


class CTCPrefixScore(object):
    """Compute CTC label sequence scores.

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probabilities of multiple labels
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
            hyp (List): prefix label sequence
            cs (np.ndarray): array of next labels. A tensor of size `[beam_width]`
            r_prev (np.ndarray): previous CTC state `[T, 2]`
        Returns:
            ctc_scores (np.ndarray): `[beam_width]`
            ctc_states (np.ndarray): `[beam_width, T, 2]`

        """
        beam_width = len(cs)
        ylen = len(hyp) - 1
        r = np.ndarray((self.xlen, 2, beam_width), dtype=np.float32)
        xs = self.log_probs[:, cs]
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
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])
        last = hyp[-1]
        if ylen > 0 and last in cs:
            log_phi = np.ndarray((self.xlen, beam_width), dtype=np.float32)
            for k in range(beam_width):
                log_phi[:, k] = r_sum if cs[k] != last else r_prev[:, 1]
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
    def forward(ctx, log_probs, hyps, exp_risk, grad_input):
        """Forward pass.

        Args:
            log_probs (FloatTensor): `[B * nbest, L, vocab]`
            hyps (LongTensor): `[B * nbest, L]`
            exp_risk (FloatTensor): `[1]` (for forward)
            grad_input (FloatTensor): `[1]` (for backward)
        Returns:
            exp_risk (FloatTensor): `[1]`

        """
        ctx.save_for_backward(grad_input)
        return exp_risk

    @staticmethod
    def backward(ctx, grad_output):
        grads, = ctx.saved_tensors
        return grads, None, None, None


def append_sos_eos(ys, sos, eos, pad, device, bwd=False, replace_sos=False):
    """Append <sos> and <eos> and return padded sequences.

    Args:
        ys (list): A list of length `[B]`, which contains a list of size `[L]`
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>

        bwd (bool): reverse ys for backward reference
        replace_sos (bool): replace <sos> with the special token
    Returns:
        ys_in (LongTensor): `[B, L]`
        ys_out (LongTensor): `[B, L]`
        ylens (IntTensor): `[B]`

    """
    _eos = torch.zeros(1, dtype=torch.int64, device=device).fill_(eos)
    ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64), device) for y in ys]
    if replace_sos:
        ylens = np2tensor(np.fromiter([(y[1:].size(0) + 1) for y in ys], dtype=np.int32))
        ys_in = pad_list([y for y in ys], pad)
        ys_out = pad_list([torch.cat([y[1:], _eos], dim=0) for y in ys], pad)
    else:
        _sos = torch.zeros(1, dtype=torch.int64, device=device).fill_(sos)
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


def distillation(logits_student, logits_teacher, ylens, temperature=1.0):
    """Compute cross entropy loss for knowledge distillation of sequence-to-sequence models.

    Args:
        logits_student (FloatTensor): `[B, T, vocab]`
        logits_teacher (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
        temperature (float):
    Returns:
        loss_mean (FloatTensor): `[1]`

    """
    bs = logits_student.size(0)
    log_probs_student = torch.log_softmax(logits_student, dim=-1)
    probs_teacher = torch.softmax(logits_teacher / temperature, dim=-1).data
    loss = -torch.mul(probs_teacher, log_probs_student)
    loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean


def tensor2scalar(x):
    """Convert torch.Tensor to a scalar value.

    Args:
        x (torch.Tensor):
    Returns:
        scaler

    """
    if isinstance(x, float):
        return x
    return x.cpu().detach().item()


class RNNDecoder(DecoderBase):
    """RNN decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of encoder outputs
        attn_type (str): type of attention mechanism
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of bottleneck layer before softmax layer for label generation
        emb_dim (int): dimension of embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of embedding and output layers
        attn_dim (int): dimension of attention space
        attn_sharpening_factor (float): sharpening factor in softmax for attention
        attn_sigmoid_smoothing (bool): replace softmax with sigmoid for attention calculation
        attn_conv_out_channels (int): channel size of convolution in location-aware attention
        attn_conv_kernel_size (int): kernel size of convolution in location-aware attention
        attn_n_heads (int): number of attention heads
        dropout (float): dropout probability for RNN layer
        dropout_emb (float): dropout probability for embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        ss_prob (float): scheduled sampling probability
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (List): fully-connected layer configuration before CTC softmax
        mbr_training (bool): MBR training
        mbr_ce_weight (float): CE loss weight for regularization during MBR training
        external_lm (RNNLM): external RNNLM for LM fusion/initialization
        lm_fusion (str): type of LM fusion
        lm_init (bool): initialize decoder with pre-trained LM
        backward (bool): decode in backward order
        global_weight (float): global loss weight for multi-task learning
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (float): parameter initialization
        mocha_chunk_size (int): chunk size for MoChA
        mocha_n_heads_mono (int): number of monotonic head for MoChA
        mocha_init_r (int): initial bias value for MoChA
        mocha_eps (float): epsilon value for MoChA
        mocha_std (float): standard deviation of Gaussian noise for MoChA
        mocha_no_denominator (bool): remove denominator in MoChA
        mocha_1dconv (bool): 1dconv for MoChA
        mocha_decot_lookahead (int): lookahead frames of DeCoT for MoChA
        quantity_loss_weight (float): quantity loss weight for MoChA
        latency_metric (str): latency metric for MoChA
        latency_loss_weight (float): latency loss weight for MoChA
        mocha_stableemit_weight (float): StableEmit weight for MoChA
        gmm_attn_n_mixtures (int): number of mixtures for GMM attention
        replace_sos (bool): replace <sos> with special tokens
        distil_weight (float): soft label weight for knowledge distillation
        discourse_aware (str): state_carry_over

    """

    def __init__(self, special_symbols, enc_n_units, attn_type, n_units, n_projs, n_layers, bottleneck_dim, emb_dim, vocab, tie_embedding, attn_dim, attn_sharpening_factor, attn_sigmoid_smoothing, attn_conv_out_channels, attn_conv_kernel_size, attn_n_heads, dropout, dropout_emb, dropout_att, lsm_prob, ss_prob, ctc_weight, ctc_lsm_prob, ctc_fc_list, mbr_training, mbr_ce_weight, external_lm, lm_fusion, lm_init, backward, global_weight, mtl_per_batch, param_init, mocha_chunk_size, mocha_n_heads_mono, mocha_init_r, mocha_eps, mocha_std, mocha_no_denominator, mocha_1dconv, mocha_decot_lookahead, quantity_loss_weight, latency_metric, latency_loss_weight, mocha_stableemit_weight, gmm_attn_n_mixtures, replace_sos, distillation_weight, discourse_aware):
        super(RNNDecoder, self).__init__()
        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.attn_type = attn_type
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ss_prob = ss_prob
        self._ss_prob = 0
        if mbr_training and ss_prob > 0:
            self.ss_prob = 0
            logging.warning('scheduled sampling is turned off for MBR training.')
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
        self.lm_fusion = lm_fusion
        self.bwd = backward
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos
        self.distil_weight = distillation_weight
        logger.info('Attention weight: %.3f' % self.att_weight)
        logger.info('CTC weight: %.3f' % self.ctc_weight)
        self.quantity_loss_weight = quantity_loss_weight
        self._quantity_loss_weight = 0
        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        self._latency_loss_weight = 0
        if 'ctc_sync' in latency_metric or attn_type == 'triggered_attention':
            assert 0 < self.ctc_weight < 1
        self.mbr_ce_weight = mbr_ce_weight
        self.mbr = MBR.apply if mbr_training else None
        self.discourse_aware = discourse_aware
        self.dstate_prev = None
        self._new_session = False
        self.prev_spk = ''
        self.dstates_final = None
        self.asrstate_final = None
        self.lmstate_final = None
        self.trflm_mem = None
        self.embed_cache = None
        self.key_tail = None
        self.aws_dict = {}
        self.data_dict = {}
        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos, blank=self.blank, enc_n_units=enc_n_units, vocab=vocab, dropout=dropout, lsm_prob=ctc_lsm_prob, fc_list=ctc_fc_list, param_init=param_init)
        else:
            self.ctc = None
        if self.att_weight > 0:
            qdim = n_units if n_projs == 0 else n_projs
            if attn_type == 'mocha':
                assert attn_n_heads == 1
                self.score = MoChA(enc_n_units, qdim, attn_dim, enc_n_units, atype='add', chunk_size=mocha_chunk_size, n_heads_mono=mocha_n_heads_mono, init_r=mocha_init_r, eps=mocha_eps, noise_std=mocha_std, no_denominator=mocha_no_denominator, conv1d=mocha_1dconv, sharpening_factor=attn_sharpening_factor, decot='decot' in latency_metric, decot_delta=mocha_decot_lookahead, stableemit_weight=mocha_stableemit_weight)
            elif attn_type == 'gmm':
                self.score = GMMAttention(enc_n_units, qdim, attn_dim, n_mixtures=gmm_attn_n_mixtures)
            elif attn_n_heads > 1:
                assert attn_type == 'add'
                self.score = MultiheadAttentionMechanism(enc_n_units, qdim, attn_dim, enc_n_units, n_heads=attn_n_heads, dropout=dropout_att, atype='add')
            else:
                self.score = AttentionMechanism(enc_n_units, qdim, attn_dim, attn_type, sharpening_factor=attn_sharpening_factor, sigmoid_smoothing=attn_sigmoid_smoothing, conv_out_channels=attn_conv_out_channels, conv_kernel_size=attn_conv_kernel_size, dropout=dropout_att, lookahead=2)
            self.rnn = nn.ModuleList()
            dec_odim = enc_n_units + emb_dim
            self.proj = repeat(nn.Linear(n_units, n_projs), n_layers) if n_projs > 0 else None
            self.dropout = nn.Dropout(p=dropout)
            for _ in range(n_layers):
                self.rnn += [nn.LSTMCell(dec_odim, n_units)]
                dec_odim = n_projs if n_projs > 0 else n_units
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
                    raise ValueError('When using tied flag, n_units must be equal to emb_dim.')
                self.output.weight = self.embed.weight
        self.reset_parameters(param_init)
        self.lm = external_lm if lm_fusion else None
        if lm_init:
            assert self.att_weight > 0
            assert external_lm is not None
            assert external_lm.vocab == vocab, 'vocab'
            assert external_lm.n_units == n_units, 'n_units'
            assert external_lm.emb_dim == emb_dim, 'emb_dim'
            logger.info('===== Initialize decoder with pre-trained RNNLM')
            assert external_lm.n_projs == 0
            assert external_lm.n_units_cv == enc_n_units, 'enc_n_units'
            for lth in range(external_lm.n_layers):
                for n, p in external_lm.rnn[lth].named_parameters():
                    n = '_'.join(n.split('_')[:2])
                    assert getattr(self.rnn[lth], n).size() == p.size()
                    getattr(self.rnn[lth], n).data = p.data
                    logger.info('Overwrite %s' % n)
            assert self.embed.weight.size() == external_lm.embed.weight.size()
            self.embed.weight.data = external_lm.embed.weight.data
            logger.info('Overwrite %s' % 'embed.weight')

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('LAS decoder')
        if not hasattr(args, 'dec_n_units'):
            group.add_argument('--dec_n_units', type=int, default=512, help='number of units in each decoder RNN layer')
            group.add_argument('--dec_n_projs', type=int, default=0, help='number of units in projection layer after each decoder RNN layer')
            group.add_argument('--dec_bottleneck_dim', type=int, default=1024, help='number of dimensions of bottleneck layer before softmax layer')
            group.add_argument('--emb_dim', type=int, default=512, help='number of dimensions in embedding layer')
        group.add_argument('--attn_type', type=str, default='location', choices=['no', 'location', 'add', 'dot', 'luong_dot', 'luong_general', 'luong_concat', 'mocha', 'gmm', 'cif', 'triggered_attention'], help='type of attention mechanism for RNN decoder')
        group.add_argument('--attn_dim', type=int, default=128, help='dimension of attention layer')
        group.add_argument('--attn_n_heads', type=int, default=1, help='number of heads in attention layer')
        group.add_argument('--attn_sharpening_factor', type=float, default=1.0, help='sharpening factor')
        group.add_argument('--attn_conv_n_channels', type=int, default=10, help='')
        group.add_argument('--attn_conv_width', type=int, default=201, help='')
        group.add_argument('--attn_sigmoid', type=strtobool, default=False, nargs='?', help='')
        group.add_argument('--gmm_attn_n_mixtures', type=int, default=5, help='number of mixtures for GMM attention')
        parser.add_argument('--ss_prob', type=float, default=0.0, help='probability of scheduled sampling')
        parser.add_argument('--ss_start_epoch', type=int, default=0, help='epoch to turn on scheduled sampling')
        parser.add_argument('--mocha_n_heads_mono', type=int, default=1, help='number of heads for monotonic attention')
        parser.add_argument('--mocha_n_heads_chunk', type=int, default=1, help='number of heads for chunkwise attention')
        parser.add_argument('--mocha_chunk_size', type=int, default=1, help='chunk size for MoChA. -1 means infinite lookback.')
        parser.add_argument('--mocha_init_r', type=float, default=-4, help='initialization of bias parameter for monotonic attention')
        parser.add_argument('--mocha_eps', type=float, default=1e-06, help='epsilon value to avoid numerical instability for MoChA')
        parser.add_argument('--mocha_std', type=float, default=1.0, help='standard deviation of Gaussian noise for MoChA during training')
        parser.add_argument('--mocha_no_denominator', type=strtobool, default=False, help='remove denominator (set to 1) in alpha recurrence in MoChA')
        parser.add_argument('--mocha_1dconv', type=strtobool, default=False, help='1dconv for MoChA')
        parser.add_argument('--mocha_quantity_loss_weight', type=float, default=0.0, help='quantity loss weight for MoChA')
        parser.add_argument('--mocha_quantity_loss_start_epoch', type=int, default=0, help='epoch to turn on quantity loss')
        parser.add_argument('--mocha_latency_metric', type=str, default='', choices=['', 'decot', 'minlt', 'ctc_sync', 'decot_ctc_sync', 'interval'], help='latency metric for MoChA')
        parser.add_argument('--mocha_latency_loss_weight', type=float, default=0.0, help='latency loss weight for MoChA')
        parser.add_argument('--mocha_decot_lookahead', type=int, default=0, help='buffer frames in DeCoT')
        parser.add_argument('--mocha_stableemit_weight', type=float, default=0.0, help='StableEmit weight for MoChA')
        parser.add_argument('--mocha_stableemit_start_epoch', type=int, default=0, help='epoch to turn on StableEmit')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name += '_' + args.dec_type
        dir_name += str(args.dec_n_units) + 'H'
        if args.dec_n_projs > 0:
            dir_name += str(args.dec_n_projs) + 'P'
        dir_name += str(args.dec_n_layers) + 'L'
        dir_name += '_' + args.attn_type
        if args.attn_sigmoid:
            dir_name += '_sig'
        if 'mocha' in args.attn_type:
            dir_name += '_w' + str(args.mocha_chunk_size)
            if args.mocha_n_heads_mono > 1:
                dir_name += '_ma' + str(args.mocha_n_heads_mono) + 'H'
            if args.mocha_no_denominator:
                dir_name += '_denom1'
            if args.mocha_1dconv:
                dir_name += '_1dconv'
        elif args.attn_type in ['gmm']:
            dir_name += '_mix' + str(args.gmm_attn_n_mixtures)
        if args.attn_sharpening_factor > 1:
            dir_name += '_temp' + str(args.attn_sharpening_factor)
        if args.mocha_quantity_loss_weight > 0:
            dir_name += '_qua' + str(args.mocha_quantity_loss_weight)
        if args.mocha_latency_metric:
            dir_name += '_' + args.mocha_latency_metric
            if 'decot' in args.mocha_latency_metric:
                dir_name += str(args.mocha_decot_lookahead)
            else:
                dir_name += str(args.mocha_latency_loss_weight)
        if args.mocha_stableemit_weight != 0:
            dir_name += '_stableemit' + str(args.mocha_stableemit_weight)
        if args.attn_n_heads > 1:
            dir_name += '_head' + str(args.attn_n_heads)
        if args.tie_embedding:
            dir_name += '_tieemb'
        if args.ctc_weight < 1 and args.ss_prob > 0:
            dir_name += '_ss' + str(args.ss_prob)
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'score.monotonic_energy.v.weight_g' in n or 'score.monotonic_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'score.monotonic_energy.conv1d' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'score.chunk_energy.v.weight_g' in n or 'score.chunk_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'linear_lm_gate.fc.bias' in n and p.dim() == 1:
                nn.init.constant_(p, -1.0)
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', -1.0))
                continue
            init_with_uniform(n, p, param_init)

    def forward(self, eouts, elens, ys, task='all', teacher_logits=None, recog_params={}, idx2token=None, trigger_points=None):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List[List]): length `[B]`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): decoding hyperparameters for N-best generation in MBR training
            idx2token ():
            trigger_points (np.ndarray): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None, 'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros(1)
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task) and self.mbr is None:
            ctc_forced_align = 'ctc_sync' in self.latency_metric and self.training or self.attn_type == 'triggered_attention'
            loss_ctc, ctc_trigger_points = self.ctc(eouts, elens, ys, forced_align=ctc_forced_align)
            observation['loss_ctc'] = tensor2scalar(loss_ctc)
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
            if self.latency_metric in ['minlt', 'decot', 'decot_ctc_sync'] and trigger_points is not None:
                trigger_points = np2tensor(trigger_points, eouts.device)
        else:
            ctc_trigger_points = None
        if self.att_weight > 0 and (task == 'all' or 'ctc' not in task) and self.mbr is None:
            loss_att, acc_att, ppl_att, loss_quantity, loss_latency = self.forward_att(eouts, elens, ys, teacher_logits=teacher_logits, ctc_trigger_points=ctc_trigger_points, forced_trigger_points=trigger_points)
            observation['loss_att'] = tensor2scalar(loss_att)
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.attn_type == 'mocha':
                if self._quantity_loss_weight > 0:
                    loss_att += loss_quantity * self._quantity_loss_weight
                observation['loss_quantity'] = tensor2scalar(loss_quantity)
            if self.latency_metric:
                if self._latency_loss_weight > 0:
                    loss_att += loss_latency * self._latency_loss_weight
                observation['loss_latency'] = tensor2scalar(loss_latency) if self.training else 0
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * self.att_weight
        if self.mbr is not None and (task == 'all' or 'mbr' in task):
            loss_mbr, loss_ce = self.forward_mbr(eouts, elens, ys, recog_params, idx2token)
            loss = loss_mbr + loss_ce * self.mbr_ce_weight
            observation['loss_mbr'] = tensor2scalar(loss_mbr)
            observation['loss_att'] = tensor2scalar(loss_ce)
        observation['loss'] = tensor2scalar(loss)
        return loss, observation

    def forward_mbr(self, eouts, elens, ys_ref, recog_params, idx2token):
        """Compute XE loss for attention-based decoder.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys_ref (List[List]): length `[B]`, each of which contains a list of size `[L]`
            recog_params (dict): decoding hyperparameters for N-best generation in MBR training
            idx2token:
        Returns:
            loss_mbr (FloatTensor): `[1]`
            loss_ce (FloatTensor): `[1]`

        """
        bs, xmax, xdim = eouts.size()
        nbest = recog_params.get('recog_beam_width')
        assert nbest >= 2
        assert idx2token is not None
        scaling_factor = 1.0
        training = self.training
        self.eval()
        with torch.no_grad():
            nbest_hyps_id, _, scores = self.beam_search(eouts, elens, params=recog_params, nbest=nbest, exclude_eos=True)
        exp_wer = 0
        nbest_hyps_id_batch = []
        grad_list = []
        for b in range(bs):
            nbest_hyps_id_b = [np.fromiter(y, dtype=np.int64) for y in nbest_hyps_id[b]]
            nbest_hyps_id_batch += nbest_hyps_id_b
            scores_b = np2tensor(np.array(scores[b], dtype=np.float32), eouts.device)
            probs_b_norm = torch.softmax(scaling_factor * scores_b, dim=-1)
            wers_b = np2tensor(np.array([(compute_wer(ref=idx2token(ys_ref[b]).split(' '), hyp=idx2token(nbest_hyps_id_b[n]).split(' '))[0] / 100) for n in range(nbest)], dtype=np.float32), eouts.device)
            exp_wer_b = (probs_b_norm * wers_b).sum()
            grad_list += [(probs_b_norm * (wers_b - exp_wer_b)).sum()]
            exp_wer += exp_wer_b
        exp_wer /= bs
        if training:
            self.train()
        eouts_expand = eouts.unsqueeze(1).expand(-1, nbest, -1, -1).contiguous().view(bs * nbest, xmax, xdim)
        elens_expand = elens.unsqueeze(1).expand(-1, nbest).contiguous().view(bs * nbest)
        ys_in, ys_out, ylens = append_sos_eos(nbest_hyps_id_batch, self.eos, self.eos, self.pad, eouts.device)
        dstates = self.zero_state(bs * nbest)
        cv = eouts.new_zeros(bs * nbest, 1, self.enc_n_units)
        self.score.reset()
        aw = None
        lmout, lmstate = None, None
        ys_emb = self.embed_token_id(ys_in)
        src_mask = make_pad_mask(elens_expand).unsqueeze(1)
        logits = []
        for i in range(ys_in.size(1)):
            if self.lm is not None:
                lmout, lmstate, _ = self.lm.predict(ys_in[:, i:i + 1], lmstate)
            dstates, cv, aw, attn_state, attn_v = self.decode_step(eouts_expand, dstates, cv, ys_emb[:, i:i + 1], src_mask, aw, lmout, mode='parallel')
            logits.append(attn_v)
            if self.attn_type in ['gmm', 'sagmm']:
                aw = attn_state['myu']
        logits = self.output(torch.cat(logits, dim=1))
        log_probs = torch.log_softmax(logits, dim=-1)
        eos = ys_in.new_zeros((1,)).fill_(self.eos)
        nbest_hyps_id_batch_pad = pad_list([torch.cat([np2tensor(y, eouts.device), eos], dim=0) for y in nbest_hyps_id_batch], self.pad)
        grad = eouts.new_zeros(bs * nbest, nbest_hyps_id_batch_pad.size(1), self.vocab)
        for b in range(bs):
            onehot = torch.eye(self.vocab)[nbest_hyps_id_batch_pad[b * nbest:(b + 1) * nbest]]
            grad[b * nbest:(b + 1) * nbest] = grad_list[b] * onehot
        grad = grad.masked_fill_((nbest_hyps_id_batch_pad == self.pad).unsqueeze(2), 0)
        loss_mbr = self.mbr(log_probs, nbest_hyps_id_batch_pad, exp_wer, grad)
        loss_ce = torch.zeros((1,), dtype=torch.float32, device=eouts.device)
        if self.mbr_ce_weight > 0:
            loss_ce = self.forward_att(eouts, elens, ys_ref)[0]
            loss_ce = loss_ce.unsqueeze(0)
        return loss_mbr, loss_ce

    def forward_att(self, eouts, elens, ys, return_logits=False, teacher_logits=None, ctc_trigger_points=None, forced_trigger_points=None):
        """Compute XE loss for attention-based decoder.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List[List]): length `[B]`, each of which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
            ctc_trigger_points (IntTensor): `[B, L]` (used for latency loss)
            forced_trigger_points (IntTensor): `[B, L]` (used for alignment path restriction)
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            loss_quantity (FloatTensor): `[1]`
            loss_latency (FloatTensor): `[1]`

        """
        bs, xmax = eouts.size()[:2]
        device = eouts.device
        ys_in, ys_out, ylens = append_sos_eos(ys, self.eos, self.eos, self.pad, eouts.device, self.bwd)
        ymax = ys_in.size(1)
        if forced_trigger_points is not None:
            for b in range(bs):
                forced_trigger_points[b, ylens[b] - 1] = elens[b] - 1
        dstates = self.zero_state(bs)
        if self.training:
            if self.discourse_aware and not self._new_session:
                dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
            self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
            self._new_session = False
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw, aws = None, []
        betas, p_chooses = [], []
        lmout, lmstate = None, None
        ys_emb = self.embed_token_id(ys_in)
        src_mask = make_pad_mask(elens).unsqueeze(1)
        tgt_mask = (ys_out != self.pad).unsqueeze(2)
        logits = []
        for i in range(ymax):
            is_sample = i > 0 and self._ss_prob > 0 and random.random() < self._ss_prob
            if self.lm is not None:
                self.lm.eval()
                with torch.no_grad():
                    y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in[:, i:i + 1]
                    lmout, lmstate, _ = self.lm.predict(y_lm, lmstate)
            y_emb = self.embed_token_id(self.output(logits[-1]).detach().argmax(-1)) if is_sample else ys_emb[:, i:i + 1]
            dstates, cv, aw, attn_state, attn_v = self.decode_step(eouts, dstates, cv, y_emb, src_mask, aw, lmout, mode='parallel', trigger_points=forced_trigger_points[:, i:i + 1] if forced_trigger_points is not None else None)
            logits.append(attn_v)
            aws.append(aw)
            if attn_state.get('beta', None) is not None:
                betas.append(attn_state['beta'])
            if attn_state.get('p_choose', None) is not None:
                p_chooses.append(attn_state['p_choose'])
            if self.attn_type in ['gmm', 'sagmm']:
                aw = attn_state['myu']
            if self.training and self.discourse_aware:
                for b in [b for b, ylen in enumerate(ylens.tolist()) if i == ylen - 1]:
                    self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1].detach()
                    self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1].detach()
        if self.training and self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = torch.cat(self.dstate_prev['hxs'], dim=1)
                self.dstate_prev['cxs'] = torch.cat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs'][0]
                self.dstate_prev['cxs'] = self.dstate_prev['cxs'][0]
        logits = self.output(torch.cat(logits, dim=1))
        if return_logits:
            return logits
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)
        acc = compute_accuracy(logits, ys_out, self.pad)
        aws = torch.cat(aws, dim=2)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.aws_dict['xy_aws'] = tensor2np(aws)
            if len(betas) > 0:
                self.aws_dict['xy_aws_beta'] = tensor2np(torch.cat(betas, dim=2))
            if len(p_chooses) > 0:
                self.aws_dict['xy_p_choose'] = tensor2np(torch.cat(p_chooses, dim=2))
        if self.attn_type == 'mocha' or (ctc_trigger_points is not None or forced_trigger_points is not None):
            aws = aws.masked_fill_(tgt_mask.unsqueeze(1).expand_as(aws) == 0, 0)
        loss_quantity = 0.0
        if self.attn_type == 'mocha':
            n_tokens_pred = aws.sum(3).sum(2).sum(1) / aws.size(1)
            n_tokens_ref = tgt_mask.squeeze(2).sum(1).float()
            loss_quantity = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))
        loss_latency = 0.0
        if self.latency_metric == 'interval':
            assert ctc_trigger_points is None
            assert aws.size(1) == 1
            aws_prev = torch.cat([aws.new_zeros(aws.size())[:, :, -1:], aws.clone()[:, :, :-1]], dim=2)
            aws_mat = aws_prev.unsqueeze(3) * aws.unsqueeze(4)
            delay_mat = aws.new_ones(xmax, xmax).float()
            delay_mat = torch.tril(delay_mat, diagonal=-1, out=delay_mat)
            delay_mat = torch.cumsum(delay_mat, dim=-2).unsqueeze(0)
            delay_mat = delay_mat.unsqueeze(1).unsqueeze(2).expand_as(aws_mat)
            loss_latency = torch.pow((aws_mat * delay_mat).sum(-1), 2).sum(-1)
            loss_latency = torch.mean(loss_latency.squeeze(1))
        elif ctc_trigger_points is not None or 'ctc_sync' not in self.latency_metric and forced_trigger_points is not None:
            if 'ctc_sync' in self.latency_metric:
                trigger_points = ctc_trigger_points
            else:
                trigger_points = forced_trigger_points
            js = torch.arange(xmax, dtype=torch.float, device=device).expand_as(aws)
            exp_trigger_points = (js * aws).sum(3)
            trigger_points = trigger_points.float().unsqueeze(1)
            loss_latency = torch.abs(exp_trigger_points - trigger_points)
            loss_latency = loss_latency.sum() / ylens.sum()
        if teacher_logits is not None:
            kl_loss = distillation(logits, teacher_logits, ylens, temperature=5.0)
            loss = loss * (1 - self.distil_weight) + kl_loss * self.distil_weight
        return loss, acc, ppl, loss_quantity, loss_latency

    def decode_step(self, eouts, dstates, cv, y_emb, mask, aw, lmout, cache=True, mode='hard', trigger_points=None, streaming=False, internal_lm=False):
        dstates = self.recurrency(torch.cat([y_emb, cv], dim=-1), dstates['dstate'])
        if internal_lm:
            attn_state = None
        else:
            cv, aw, attn_state = self.score(eouts, eouts, dstates['dout_score'], mask, aw, cache=cache, mode=mode, trigger_points=trigger_points, streaming=streaming)
        attn_v = self.generate(cv, dstates['dout_gen'], lmout)
        return dstates, cv, aw, attn_state, attn_v

    def zero_state(self, bs):
        """Initialize decoder state.

        Args:
            bs (int): batch size
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        dstates = {'dstate': None}
        w = next(self.parameters())
        hxs = w.new_zeros(self.n_layers, bs, self.dec_n_units)
        cxs = w.new_zeros(self.n_layers, bs, self.dec_n_units)
        dstates['dstate'] = hxs, cxs
        return dstates

    def recurrency(self, inputs, dstate):
        """Recurrency function.

        Args:
            inputs (FloatTensor): `[B, 1, emb_dim + enc_n_units]`
            dstate (tuple): (hxs, cxs)
        Returns:
            new_dstates (dict):
                dout_score (FloatTensor): `[B, 1, dec_n_units]`
                dout_gen (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        hxs, cxs = dstate
        dout = inputs.squeeze(1)
        new_dstates = {'dout_score': None, 'dout_gen': None, 'dstate': None}
        new_hxs, new_cxs = [], []
        for lth in range(self.n_layers):
            h, c = self.rnn[lth](dout, (hxs[lth], cxs[lth]))
            new_hxs.append(h)
            new_cxs.append(c)
            dout = self.dropout(h)
            if self.proj is not None:
                dout = torch.relu(self.proj[lth](dout))
            if lth == 0:
                new_dstates['dout_score'] = dout.unsqueeze(1)
        new_hxs = torch.stack(new_hxs, dim=0)
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

    def greedy(self, eouts, elens, max_len_ratio, idx2token, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, trigger_points=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            trigger_points (IntTensor): `[B, T]`
        Returns:
            hyps (List[np.array]): length `[B]`, each of which contains arrays of size `[L]`
            aws (List[np.array]): length `[B]`, each of which contains arrays of size `[H, L, T]`

        """
        bs, xmax = eouts.size()[:2]
        dstates = self.zero_state(bs)
        if self.discourse_aware and not self._new_session:
            dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
        self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
        self._new_session = False
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw = None
        lmout, lmstate = None, None
        y = eouts.new_zeros((bs, 1), dtype=torch.int64).fill_(refs_id[0][0] if self.replace_sos else self.eos)
        src_mask = make_pad_mask(elens).unsqueeze(1)
        if self.attn_type == 'triggered_attention':
            assert trigger_points is not None
        hyps_batch, aws_batch = [], []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        ymax = math.ceil(xmax * max_len_ratio)
        for i in range(ymax):
            if self.lm is not None:
                lmout, lmstate, _ = self.lm.predict(y, lmstate)
            dstates, cv, aw, attn_state, attn_v = self.decode_step(eouts, dstates, cv, self.embed_token_id(y), src_mask, aw, lmout, trigger_points=trigger_points[:, i:i + 1] if trigger_points is not None else None)
            aws_batch += [aw]
            if self.attn_type in ['gmm', 'sagmm']:
                aw = attn_state['myu']
            y = self.output(attn_v).argmax(-1)
            hyps_batch += [y]
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                        if self.discourse_aware:
                            self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1]
                            self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1]
                    ylens[b] += 1
            if sum(eos_flags) == bs:
                break
            if i == ymax - 1:
                break
        if self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = torch.cat(self.dstate_prev['hxs'], dim=1)
                self.dstate_prev['cxs'] = torch.cat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs']
                self.dstate_prev['cxs'] = self.dstate_prev['cxs']
        self.lmstate_final = lmstate
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        aws_batch = tensor2np(torch.cat(aws_batch, dim=2))
        if self.bwd:
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]] for b in range(bs)]
        if exclude_eos:
            if self.bwd:
                hyps = [(hyps[b][1:] if eos_flags[b] else hyps[b]) for b in range(bs)]
                aws = [(aws[b][:, 1:] if eos_flags[b] else aws[b]) for b in range(bs)]
            else:
                hyps = [(hyps[b][:-1] if eos_flags[b] else hyps[b]) for b in range(bs)]
                aws = [(aws[b][:, :-1] if eos_flags[b] else aws[b]) for b in range(bs)]
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
                logger.debug('=' * 200)
        return hyps, aws

    def embed_token_id(self, indices):
        """Embed token IDs.
        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.dropout_emb(self.embed(indices))
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def cache_embedding(self, device):
        """Cache token emebdding."""
        if self.embed_cache is None:
            indices = torch.arange(0, self.vocab, 1, dtype=torch.int64)
            self.embed_cache = self.embed_token_id(indices)

    def initialize_beam(self, hyp, dstates, cv, lmstate, ctc_state, ensmbl_decs=[], ilm_dstates=None):
        """Initialize beam."""
        ensmbl_dstate, ensmbl_cv = [], []
        for dec in ensmbl_decs:
            dec.score.reset()
            ensmbl_dstate += [dec.zero_state(1)]
            ensmbl_cv += [cv.new_zeros(1, 1, dec.enc_n_units)]
        hyps = [{'hyp': hyp, 'score': 0.0, 'score_att': 0.0, 'score_ctc': 0.0, 'score_lm': 0.0, 'score_ilm': 0.0, 'dstates': dstates, 'ilm_dstates': ilm_dstates, 'cv': cv, 'aws': [None], 'myu': None, 'lmstate': lmstate, 'ensmbl_dstate': ensmbl_dstate, 'ensmbl_cv': ensmbl_cv, 'ensmbl_aws': [[None]] * len(ensmbl_dstate), 'ctc_state': ctc_state, 'quantity_rate': 1.0, 'streamable': True, 'streaming_failed_point': 1000, 'boundary': [], 'no_boundary': False}]
        return hyps

    def beam_search(self, eouts, elens, params, idx2token=None, lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None, nbest=1, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, ensmbl_eouts=[], ensmbl_elens=[], ensmbl_decs=[], cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict): decoding hyperparameters
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh-pass LM
            lm_second (torch.nn.module): second-pass LM
            lm_second_bwd (torch.nn.module): second-pass backward LM
            ctc_log_probs (FloatTensor): `[B, T, vocab]`
            nbest (int): number of N-best list
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            ensmbl_eouts (List[FloatTensor]): encoder outputs for ensemble models
            ensmbl_elens (List[IntTensor]) encoder outputs for ensemble models
            ensmbl_decs (List[torch.nn.Module): decoders for ensemble models
            cache_states (bool): cache TransformerLM/TransformerXL states
        Returns:
            nbest_hyps_idx (List[List[np.array]]): length `[B]`, each of which contains a list of hypotheses of size `[nbest]`,
                each of which containts a list of arrays of size `[L]`
            aws (List[List[[np.array]]]): length `[B]`, each of which contains a list of attention weights of size `[nbest]`,
                each of which containts a list of arrays of size `[H, L, T]`
            scores (List[List[np.array]]): sequence-level scores

        """
        bs, xmax, _ = eouts.size()
        beam_width = params.get('recog_beam_width')
        assert 1 <= nbest <= beam_width
        ctc_weight = params.get('recog_ctc_weight')
        max_len_ratio = params.get('recog_max_len_ratio')
        min_len_ratio = params.get('recog_min_len_ratio')
        lp_weight = params.get('recog_length_penalty')
        cp_weight = params.get('recog_coverage_penalty')
        cp_threshold = params.get('recog_coverage_threshold')
        length_norm = params.get('recog_length_norm')
        cache_emb = params.get('recog_cache_embedding')
        lm_weight = params.get('recog_lm_weight')
        ilm_weight = params.get('recog_ilm_weight')
        lm_weight_second = params.get('recog_lm_second_weight')
        lm_weight_second_bwd = params.get('recog_lm_bwd_weight')
        gnmt_decoding = params.get('recog_gnmt_decoding')
        eos_threshold = params.get('recog_eos_threshold')
        asr_state_CO = params.get('recog_asr_state_carry_over')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')
        if self.attn_type == 'mocha':
            self.score.set_p_choose_threshold(params.get('recog_mocha_p_choose_threshold', 0.5))
        helper = BeamSearch(beam_width, self.eos, ctc_weight, lm_weight, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight, cache_emb)
        if lm is not None:
            assert isinstance(lm, RNNLM)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second, cache_emb)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd, cache_emb)
        if cache_emb:
            self.cache_embedding(eouts.device)
        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)
        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            self.score.reset()
            cv = eouts.new_zeros(1, 1, self.enc_n_units)
            dstates = self.zero_state(1)
            lmstate = None
            ctc_state = None
            ilm_dstates = self.zero_state(1)
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)
                ctc_state = ctc_prefix_scorer.initial_state()
            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if asr_state_CO:
                        dstates = self.dstates_final
                    if lm_state_CO:
                        lmstate = self.lmstate_final
                else:
                    self.dstates_final = None
                    self.lmstate_final = None
                    self.trflm_mem = None
                self.prev_spk = speakers[b]
            end_hyps = []
            hyps = self.initialize_beam([self.eos], dstates, cv, lmstate, ctc_state, ensmbl_decs, ilm_dstates)
            streamable_global = True
            ymax = math.ceil(elens[b] * max_len_ratio)
            for i in range(ymax):
                y = eouts.new_zeros((len(hyps), 1), dtype=torch.int64)
                for j, beam in enumerate(hyps):
                    if self.replace_sos and i == 0:
                        prev_idx = refs_id[0][0]
                    else:
                        prev_idx = beam['hyp'][-1]
                    y[j, 0] = prev_idx
                cv = torch.cat([beam['cv'] for beam in hyps], dim=0)
                eouts_b_i = eouts[b:b + 1, :elens[b]].repeat([cv.size(0), 1, 1])
                if self.attn_type in ['gmm', 'sagmm']:
                    aw = torch.cat([beam['myu'] for beam in hyps], dim=0) if i > 0 else None
                else:
                    aw = torch.cat([beam['aws'][-1] for beam in hyps], dim=0) if i > 0 else None
                hxs = torch.cat([beam['dstates']['dstate'][0] for beam in hyps], dim=1)
                cxs = torch.cat([beam['dstates']['dstate'][1] for beam in hyps], dim=1)
                dstates = {'dstate': (hxs, cxs)}
                if ilm_weight > 0:
                    ilm_hxs = torch.cat([beam['ilm_dstates']['dstate'][0] for beam in hyps], dim=1)
                    ilm_cxs = torch.cat([beam['ilm_dstates']['dstate'][1] for beam in hyps], dim=1)
                    ilm_dstates = {'dstate': (ilm_hxs, ilm_cxs)}
                lmout, lmstate, scores_lm = None, None, None
                if lm is not None or self.lm is not None:
                    if i > 0:
                        lmstate = {'hxs': torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1), 'cxs': torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)}
                    if self.lm is not None:
                        lmout, lmstate, scores_lm = self.lm.predict(y, lmstate)
                    elif lm is not None:
                        lmout, lmstate, scores_lm = lm.predict(y, lmstate)
                y_emb = self.embed_token_id(y)
                dstates, cv, aw, attn_state, attn_v = self.decode_step(eouts_b_i, dstates, cv, y_emb, None, aw, lmout)
                probs = torch.softmax(self.output(attn_v).squeeze(1) * softmax_smoothing, dim=1)
                if ilm_weight > 0:
                    ilm_dstates, _, _, _, ilm_attn_v = self.decode_step(eouts.new_zeros(eouts_b_i.size()), ilm_dstates, cv.new_zeros(cv.size()), y_emb, None, None, lmout, internal_lm=True)
                    scores_ilm = torch.log_softmax(self.output(ilm_attn_v).squeeze(1) * softmax_smoothing, dim=1)
                ensmbl_dstate, ensmbl_cv, ensmbl_aws = [], [], []
                for i_e, dec in enumerate(ensmbl_decs):
                    cv_e = torch.cat([beam['ensmbl_cv'][i_e] for beam in hyps], dim=0)
                    aw_e = torch.cat([beam['ensmbl_aws'][i_e][-1] for beam in hyps], dim=0) if i > 0 else None
                    hxs_e = torch.cat([beam['ensmbl_dstate'][i_e]['dstate'][0] for beam in hyps], dim=1)
                    cxs_e = torch.cat([beam['ensmbl_dstate'][i_e]['dstate'][1] for beam in hyps], dim=1)
                    dstates_e = {'dstate': (hxs_e, cxs_e)}
                    dstates_e, cv_e, aw_e, _, attn_v_e = dec.decode_step(ensmbl_eouts[i_e][b:b + 1, :ensmbl_elens[i_e][b]].repeat([cv_e.size(0), 1, 1]), dstates_e, cv_e, dec.embed_token_id(y), None, aw_e, lmout)
                    probs += torch.softmax(dec.output(attn_v_e).squeeze(1) * softmax_smoothing, dim=1)
                    ensmbl_dstate += [dstates_e]
                    ensmbl_cv += [cv_e]
                    ensmbl_aws += [aw_e]
                scores_att = torch.log(probs / (len(ensmbl_decs) + 1))
                new_hyps = []
                for j, beam in enumerate(hyps):
                    ensmbl_dstate_j, ensmbl_cv_j, ensmbl_aws_j = [], [], []
                    if len(ensmbl_decs) > 0:
                        for i_e in range(len(ensmbl_decs)):
                            ensmbl_dstate_j += [{'dstate': (ensmbl_dstate[i_e]['dstate'][0][:, j:j + 1], ensmbl_dstate[i_e]['dstate'][1][:, j:j + 1])}]
                            ensmbl_cv_j += [ensmbl_cv[i_e][j:j + 1]]
                            ensmbl_aws_j += [beam['ensmbl_aws'][i_e] + [ensmbl_aws[i_e][j:j + 1]]]
                    total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                    if ilm_weight > 0:
                        total_scores_ilm = beam['score_ilm'] + scores_ilm[j:j + 1]
                    else:
                        total_scores_ilm = eouts.new_zeros(1, self.vocab)
                    total_scores = total_scores_att * (1 - ctc_weight)
                    total_scores -= total_scores_ilm * ilm_weight * (1 - ctc_weight)
                    total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if lm is not None or self.lm is not None:
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
                        aw_mat = aw_mat[:, 0, :, :]
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
                        length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                        total_score = total_scores_topk[0, k].item() / length_norm_factor
                        if idx == self.eos:
                            if len(beam['hyp'][1:]) < elens[b] * min_len_ratio:
                                continue
                            max_score_no_eos = scores_att[j, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_att[j, idx + 1:].max(0)[0].item())
                            if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue
                        streaming_failed_point = beam['streaming_failed_point']
                        quantity_rate = 1.0
                        if self.attn_type == 'mocha':
                            n_heads_total = 1
                            n_quantity_k = aw[j:j + 1, :, 0].int().sum().item()
                            quantity_diff = n_heads_total - n_quantity_k
                            if quantity_diff != 0:
                                if idx == self.eos:
                                    quantity_rate = 1
                                else:
                                    streamable_global = False
                                    quantity_rate = n_quantity_k / n_heads_total
                            if beam['streamable'] and not streamable_global:
                                streaming_failed_point = i
                        new_lmstate = None
                        if lmstate is not None:
                            new_lmstate = {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]}
                        new_hyps.append({'hyp': beam['hyp'] + [idx], 'score': total_score, 'score_att': total_scores_att[0, idx].item(), 'score_ilm': total_scores_ilm[0, idx].item(), 'score_cp': cp, 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[k].item(), 'dstates': {'dstate': (dstates['dstate'][0][:, j:j + 1], dstates['dstate'][1][:, j:j + 1])}, 'ilm_dstates': {'dstate': (ilm_dstates['dstate'][0][:, j:j + 1], ilm_dstates['dstate'][1][:, j:j + 1])} if ilm_weight > 0 else None, 'cv': cv[j:j + 1], 'aws': beam['aws'] + [aw[j:j + 1]], 'myu': attn_state['myu'][j:j + 1] if self.attn_type in ['gmm', 'sagmm'] else None, 'lmstate': new_lmstate, 'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None, 'ensmbl_dstate': ensmbl_dstate_j, 'ensmbl_cv': ensmbl_cv_j, 'ensmbl_aws': ensmbl_aws_j, 'streamable': streamable_global, 'streaming_failed_point': streaming_failed_point, 'quantity_rate': quantity_rate})
                new_hyps = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
                hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps, end_hyps)
                if is_finish:
                    break
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < beam_width:
                end_hyps.extend(hyps[:beam_width - len(end_hyps)])
            end_hyps = helper.lm_rescoring(end_hyps, lm_second, lm_weight_second, length_norm=length_norm, tag='second')
            end_hyps = helper.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, length_norm=length_norm, tag='second_bwd')
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)
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
                    if len(end_hyps[k]['hyp']) > 1:
                        logger.info('num tokens (hyp): %d' % len(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_att'] * (1 - ctc_weight)))
                    logger.info('log prob (hyp, ilm): %.7f' % (end_hyps[k]['score_ilm'] * (1 - ctc_weight) * ilm_weight))
                    logger.info('log prob (hyp, cp): %.7f' % (end_hyps[k]['score_cp'] * cp_weight))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None or self.lm is not None:
                        logger.info('log prob (hyp, first-pass lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-pass lm): %.7f' % (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-pass lm, reverse): %.7f' % (end_hyps[k]['score_lm_second_bwd'] * lm_weight_second_bwd))
                    if self.attn_type == 'mocha':
                        logger.info('streamable: %s' % end_hyps[k]['streamable'])
                        logger.info('streaming failed point: %d' % (end_hyps[k]['streaming_failed_point'] + 1))
                        logger.info('quantity rate [%%]: %.2f' % (end_hyps[k]['quantity_rate'] * 100))
                    logger.info('-' * 50)
                if self.attn_type == 'mocha' and end_hyps[0]['streaming_failed_point'] < 1000:
                    assert not self.streamable
                    aws_last_success = end_hyps[0]['aws'][1:][end_hyps[0]['streaming_failed_point'] - 1]
                    rightmost_frame = max(0, aws_last_success[0, :, 0].nonzero()[:, -1].max().item()) + 1
                    frame_ratio = rightmost_frame * 100 / xmax
                    self.last_success_frame_ratio = frame_ratio
                    logger.info('streaming last success frame ratio: %.2f' % frame_ratio)
            if self.bwd:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [[tensor2np(torch.cat(end_hyps[n]['aws'][1:][::-1], dim=2).squeeze(0)) for n in range(nbest)]]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [[tensor2np(torch.cat(end_hyps[n]['aws'][1:], dim=2).squeeze(0)) for n in range(nbest)]]
            if length_norm:
                scores += [[(end_hyps[n]['score_att'] / len(end_hyps[n]['hyp'][1:])) for n in range(nbest)]]
            else:
                scores += [[end_hyps[n]['score_att'] for n in range(nbest)]]
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][1:] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
                aws = [[(aws[b][n][:, 1:] if eos_flags[b][n] else aws[b][n]) for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][:-1] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
                aws = [[(aws[b][n][:, :-1] if eos_flags[b][n] else aws[b][n]) for n in range(nbest)] for b in range(bs)]
        if bs == 1:
            self.dstates_final = end_hyps[0]['dstates']
            self.lmstate_final = end_hyps[0]['lmstate']
        return nbest_hyps_idx, aws, scores

    def batchfy_beam(self, hyps, i, ilm_weight):
        """Batchfy all the active hypetheses in an utternace for efficient matrix multiplication."""
        y = torch.zeros((len(hyps), 1), dtype=torch.int64, device=self.device)
        for j, beam in enumerate(hyps):
            y[j, 0] = beam['hyp'][-1]
        cv = torch.cat([beam['cv'] for beam in hyps], dim=0)
        if self.attn_type in ['gmm', 'sagmm']:
            aw = torch.cat([beam['myu'] for beam in hyps], dim=0) if i > 0 else None
        else:
            aw = torch.cat([beam['aws'][-1] for beam in hyps], dim=0) if i > 0 else None
        hxs = torch.cat([beam['dstates']['dstate'][0] for beam in hyps], dim=1)
        cxs = torch.cat([beam['dstates']['dstate'][1] for beam in hyps], dim=1)
        dstates = {'dstate': (hxs, cxs)}
        if ilm_weight > 0:
            ilm_hxs = torch.cat([beam['ilm_dstates']['dstate'][0] for beam in hyps], dim=1)
            ilm_cxs = torch.cat([beam['ilm_dstates']['dstate'][1] for beam in hyps], dim=1)
            ilm_dstates = {'dstate': (ilm_hxs, ilm_cxs)}
        else:
            ilm_dstates = None
        return y, cv, aw, dstates, ilm_dstates

    def beam_search_block_sync(self, eouts, params, helper, idx2token, hyps, hyps_nobd, lm, ctc_log_probs=None, speaker=None, ignore_eos=False, dualhyp=True):
        assert eouts.size(0) == 1
        assert self.attn_type == 'mocha'
        beam_width = params.get('recog_beam_width')
        ctc_weight = params.get('recog_ctc_weight')
        max_len_ratio = params.get('recog_max_len_ratio')
        lp_weight = params.get('recog_length_penalty')
        length_norm = params.get('recog_length_norm')
        lm_weight = params.get('recog_lm_weight')
        ilm_weight = params.get('recog_ilm_weight')
        eos_threshold = params.get('recog_eos_threshold')
        lm_state_CO = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')
        end_hyps = []
        if hyps is None:
            self.score.reset()
            cv = eouts.new_zeros(1, 1, self.enc_n_units)
            dstates = self.zero_state(1)
            lmstate = None
            ctc_state = None
            ilm_dstates = self.zero_state(1) if ilm_weight > 0 else None
            if speaker is not None:
                if lm_state_CO and speaker == self.prev_spk:
                    lmstate = self.lmstate_final
                self.prev_spk = speaker
            self.lmstate_final = None
            self.ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                assert ctc_weight > 0
                ctc_log_probs = tensor2np(ctc_log_probs)
                if hyps is None:
                    self.ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[0], self.blank, self.eos)
                    ctc_state = self.ctc_prefix_scorer.initial_state()
                else:
                    self.ctc_prefix_scorer.register_new_chunk(ctc_log_probs[0])
            self.n_frames = 0
            self.key_tail = None
            hyps = self.initialize_beam([self.eos], dstates, cv, lmstate, ctc_state, ilm_dstates=ilm_dstates)
        else:
            hyps += hyps_nobd
            hyps_nobd = []
            for beam in hyps:
                beam['no_boundary'] = False
            self.score.reset()
            self.score.register_tail(self.key_tail)
        ymax = math.ceil(eouts.size(1) * max_len_ratio)
        for i in range(ymax):
            if len(hyps) == 0:
                break
            y, cv, aw, dstates, ilm_dstates = self.batchfy_beam(hyps, i, ilm_weight)
            lmout, lmstate, scores_lm = helper.update_rnnlm_state_batch(self.lm if self.lm is not None else lm, hyps, y)
            y_emb = self.embed_token_id(y)
            dstates, cv, aw, _, attn_v = self.decode_step(eouts, dstates, cv, y_emb, None, aw, lmout, streaming=True)
            scores_att = torch.log_softmax(self.output(attn_v).squeeze(1) * softmax_smoothing, dim=1)
            if ilm_weight > 0:
                ilm_dstates, _, _, _, ilm_attn_v = self.decode_step(eouts.new_zeros(eouts.size()), ilm_dstates, cv.new_zeros(cv.size()), y_emb, None, None, lmout, streaming=True, internal_lm=True)
                scores_ilm = torch.log_softmax(self.output(ilm_attn_v).squeeze(1) * softmax_smoothing, dim=1)
            new_hyps = []
            for j, beam in enumerate(hyps):
                no_boundary = aw[j].sum().item() == 0
                if no_boundary:
                    hyps_nobd.append(beam.copy())
                    hyps_nobd[-1]['no_boundary'] = True
                total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                if ilm_weight > 0:
                    total_scores_ilm = beam['score_ilm'] + scores_ilm[j:j + 1]
                else:
                    total_scores_ilm = eouts.new_zeros(1, self.vocab)
                total_scores = total_scores_att * (1 - ctc_weight)
                total_scores -= total_scores_ilm * ilm_weight * (1 - ctc_weight)
                total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                if lm is not None or self.lm is not None:
                    total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                    total_scores_topk += total_scores_lm * lm_weight
                else:
                    total_scores_lm = eouts.new_zeros(beam_width)
                total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight
                bd = self.n_frames
                if not no_boundary:
                    boundary_list_j = np.where(tensor2np(aw[j].sum(1).sum(0)) != 0)[0]
                    bd += int(boundary_list_j[0])
                    if len(beam['boundary']) > 0:
                        assert bd >= beam['boundary'][-1], (bd, beam['boundary'])
                new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(beam['hyp'], topk_ids, beam['ctc_state'], total_scores_topk, self.ctc_prefix_scorer, new_chunk=i == 0)
                for k in range(beam_width):
                    idx = topk_ids[0, k].item()
                    if ignore_eos and idx == self.eos:
                        continue
                    if no_boundary and idx != self.eos:
                        continue
                    length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                    total_score = total_scores_topk[0, k].item() / length_norm_factor
                    if idx == self.eos:
                        max_score_no_eos = scores_att[j, :idx].max(0)[0].item()
                        max_score_no_eos = max(max_score_no_eos, scores_att[j, idx + 1:].max(0)[0].item())
                        if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                            continue
                    new_hyps.append({'hyp': beam['hyp'] + [idx], 'score': total_score, 'score_att': total_scores_att[0, idx].item(), 'score_ilm': total_scores_ilm[0, idx].item(), 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[k].item(), 'dstates': {'dstate': (dstates['dstate'][0][:, j:j + 1], dstates['dstate'][1][:, j:j + 1])}, 'ilm_dstates': {'dstate': (ilm_dstates['dstate'][0][:, j:j + 1], ilm_dstates['dstate'][1][:, j:j + 1])} if ilm_weight > 0 else None, 'cv': cv[j:j + 1], 'aws': beam['aws'] + [aw[j:j + 1]], 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None, 'ctc_state': new_ctc_states[k] if self.ctc_prefix_scorer is not None else None, 'boundary': beam['boundary'] + [bd] if not no_boundary else beam['boundary'], 'no_boundary': no_boundary})
            if not dualhyp:
                new_hyps += hyps_nobd
            new_hyps = sorted(new_hyps, key=lambda x: x['score'], reverse=True)
            new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps, end_hyps)
            if dualhyp:
                hyps = new_hyps[:]
            else:
                hyps_nobd = [beam for beam in new_hyps if beam['no_boundary']]
                hyps = [beam for beam in new_hyps if not beam['no_boundary']]
            if is_finish:
                break
        if len(end_hyps) > 0:
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)
        merged_hyps = sorted(end_hyps + hyps + hyps_nobd, key=lambda x: x['score'], reverse=True)[:beam_width]
        if idx2token is not None:
            logger.info('=' * 200)
            for k in range(len(merged_hyps)):
                logger.info('Hyp: %s' % idx2token(merged_hyps[k]['hyp'][1:]))
                if len(merged_hyps[k]['hyp']) > 1:
                    logger.info('num tokens (hyp): %d' % len(merged_hyps[k]['hyp'][1:]))
                if len(merged_hyps[k]['boundary']) > 0:
                    logger.info('boundary: %s' % ' '.join(list(map(str, merged_hyps[k]['boundary']))))
                logger.info('no boundary: %s' % merged_hyps[k]['no_boundary'])
                logger.info('log prob (hyp): %.7f' % merged_hyps[k]['score'])
                logger.info('log prob (hyp, att): %.7f' % (merged_hyps[k]['score_att'] * (1 - ctc_weight)))
                logger.info('log prob (hyp, ilm): %.7f' % (merged_hyps[k]['score_ilm'] * (1 - ctc_weight) * ilm_weight))
                if self.ctc_prefix_scorer is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (merged_hyps[k]['score_ctc'] * ctc_weight))
                if lm is not None or self.lm is not None:
                    logger.info('log prob (hyp, first-pass lm): %.7f' % (merged_hyps[k]['score_lm'] * lm_weight))
                logger.info('-' * 50)
        if len(merged_hyps) > 0:
            self.asrstate_final = merged_hyps[0]
            self.lmstate_final = merged_hyps[0]['lmstate']
        self.n_frames += eouts.size(1)
        self.score.reset()
        if eouts.size(1) < self.score.w - 1 and self.key_tail is not None:
            self.key_tail = torch.cat([self.key_tail, eouts], dim=1)[:, -(self.score.w - 1):]
        else:
            self.key_tail = eouts[:, -(self.score.w - 1):]
        return end_hyps, hyps, hyps_nobd


class SpecAugment(object):
    """SpecAugment class.

    Args:
        F (int): parameter for frequency masking
        T (int): parameter for time masking
        n_freq_masks (int): number of frequency masks
        n_time_masks (int): number of time masks
        W (int): parameter for time warping
        p (float): parameter for upperbound of the time mask
        adaptive_number_ratio (float): adaptive multiplicity ratio for time masking
        adaptive_size_ratio (float): adaptive size ratio for time masking
        max_n_time_masks (int): maximum number of time masking

    """

    def __init__(self, F, T, n_freq_masks, n_time_masks, p=1.0, W=40, adaptive_number_ratio=0, adaptive_size_ratio=0, max_n_time_masks=20):
        super(SpecAugment, self).__init__()
        self.W = W
        self.F = F
        self.T = T
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
        self.adaptive_number_ratio = adaptive_number_ratio
        self.adaptive_size_ratio = adaptive_size_ratio
        self.max_n_time_masks = max_n_time_masks
        if adaptive_number_ratio > 0:
            self.n_time_masks = 0
            logger.info('n_time_masks is set ot zero for adaptive SpecAugment.')
        if adaptive_size_ratio > 0:
            self.T = 0
            logger.info('T is set to zero for adaptive SpecAugment.')
        self._freq_mask = None
        self._time_mask = None

    def librispeech_basic(self):
        self.W = 80
        self.F = 27
        self.T = 100
        self.n_freq_masks = 1
        self.n_time_masks = 1
        self.p = 1.0

    def librispeech_double(self):
        self.W = 80
        self.F = 27
        self.T = 100
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 1.0

    def switchboard_mild(self):
        self.W = 40
        self.F = 15
        self.T = 70
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 0.2

    def switchboard_strong(self):
        self.W = 40
        self.F = 27
        self.T = 70
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 0.2

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
        xs = self.mask_freq(xs)
        xs = self.mask_time(xs)
        return xs

    def time_warp(xs, W=40):
        raise NotImplementedError

    def mask_freq(self, xs, replace_with_zero=False):
        n_bins = xs.size(-1)
        for i in range(0, self.n_freq_masks):
            f = int(np.random.uniform(low=0, high=self.F))
            f_0 = int(np.random.uniform(low=0, high=n_bins - f))
            xs[:, :, f_0:f_0 + f] = 0
            assert f_0 <= f_0 + f
            self._freq_mask = f_0, f_0 + f
        return xs

    def mask_time(self, xs, replace_with_zero=False):
        n_frames = xs.size(1)
        if self.adaptive_number_ratio > 0:
            n_masks = int(n_frames * self.adaptive_number_ratio)
            n_masks = min(n_masks, self.max_n_time_masks)
        else:
            n_masks = self.n_time_masks
        if self.adaptive_size_ratio > 0:
            T = self.adaptive_size_ratio * n_frames
        else:
            T = self.T
        for i in range(n_masks):
            t = int(np.random.uniform(low=0, high=T))
            t = min(t, int(n_frames * self.p))
            t_0 = int(np.random.uniform(low=0, high=n_frames - t))
            xs[:, t_0:t_0 + t] = 0
            assert t_0 <= t_0 + t
            self._time_mask = t_0, t_0 + t
        return xs


class Streaming(object):
    """Streaming encoding interface."""

    def __init__(self, x_whole, params, encoder, idx2token=None):
        """
        Args:
            x_whole (FloatTensor): `[T, input_dim]`
            params (dict): decoding hyperparameters
            encoder (torch.nn.module): encoder module
            idx2token (): converter from index to token

        """
        super(Streaming, self).__init__()
        self.x_whole = x_whole
        self.xmax_whole = len(x_whole)
        self.input_dim = x_whole.shape[1]
        self.enc_type = encoder.enc_type
        self.idx2token = idx2token
        if self.enc_type in ['lstm', 'conv_lstm', 'conv_uni_transformer', 'conv_uni_conformer', 'conv_uni_conformer_v2']:
            self.streaming_type = 'unidir'
        elif 'lstm' in self.enc_type or 'gru' in self.enc_type:
            self.streaming_type = 'lc_bidir'
        else:
            assert hasattr(encoder, 'streaming_type')
            self.streaming_type = getattr(encoder, 'streaming_type', '')
        self._factor = encoder.subsampling_factor
        self.N_l = getattr(encoder, 'N_l', 0)
        self.N_c = encoder.N_c
        self.N_r = encoder.N_r
        if self.streaming_type == 'mask':
            self.N_l = 0
        if self.N_c <= 0 and self.N_r <= 0:
            self.N_c = params.get('recog_block_sync_size')
            assert self.N_c % self._factor == 0
        self.blank_id = 0
        self.enable_ctc_reset_point_detection = params.get('recog_ctc_vad')
        self.BLANK_THRESHOLD = params.get('recog_ctc_vad_blank_threshold')
        self.SPIKE_THRESHOLD = params.get('recog_ctc_vad_spike_threshold')
        self.MAX_N_ACCUM_FRAMES = params.get('recog_ctc_vad_n_accum_frames')
        assert self.BLANK_THRESHOLD % self._factor == 0
        assert self.MAX_N_ACCUM_FRAMES % self._factor == 0
        self._offset = 0
        self._n_blanks = 0
        self._n_accum_frames = 0
        self.conv_context = encoder.conv.context_size if encoder.conv is not None else 0
        if not getattr(encoder, 'cnn_lookahead', True):
            self.conv_context = 0
        self._eout_blocks = []

    @property
    def offset(self):
        return self._offset

    @property
    def n_blanks(self):
        return self._n_blanks

    @property
    def n_accum_frames(self):
        return self._n_accum_frames

    @property
    def n_cache_block(self):
        return len(self._eout_blocks)

    @property
    def safeguard_reset(self):
        return self._n_accum_frames < self.MAX_N_ACCUM_FRAMES

    def reset(self):
        self._eout_blocks = []
        self._n_blanks = 0
        self._n_accum_frames = 0

    def cache_eout(self, eout_block):
        self._eout_blocks.append(eout_block)

    def pop_eouts(self):
        return torch.cat(self._eout_blocks, dim=1)

    def next_block(self):
        self._offset += self.N_c

    def extract_feat(self):
        """Slice acoustic features.

        Returns:
            x_block (np.array): `[T_block, input_dim]`
            is_last_block (bool): flag for the last input block
            cnn_lookback (bool): use lookback frames in CNN
            cnn_lookahead (bool): use lookahead frames in CNN
            xlen_block (int): input length of the cernter region in a block (for the last block)

        """
        j = self._offset
        N_l, N_c, N_r = self.N_l, self.N_c, self.N_r
        start = j - (self.conv_context + N_l)
        end = j + (N_c + N_r + self.conv_context)
        x_block = self.x_whole[max(0, start):end]
        is_last_block = j + N_c >= self.xmax_whole
        cnn_lookback = self.streaming_type != 'reshape' and start >= 0
        cnn_lookahead = self.streaming_type != 'reshape' and end < self.xmax_whole
        N_conv = self.conv_context if j == 0 or is_last_block else self.conv_context * 2
        if self.streaming_type in ['reshape', 'mask']:
            xlen_block = min(self.xmax_whole - j, N_c)
        elif self.streaming_type == 'lc_bidir':
            xlen_block = min(self.xmax_whole - j + N_conv, N_c + N_conv)
        else:
            xlen_block = len(x_block)
        if self.streaming_type == 'reshape':
            if start < 0:
                zero_pad = np.zeros((-start, self.input_dim)).astype(np.float32)
                x_block = np.concatenate([zero_pad, x_block], axis=0)
            if len(x_block) < N_l + N_c + N_r:
                zero_pad = np.zeros((N_l + N_c + N_r - len(x_block), self.input_dim)).astype(np.float32)
                x_block = np.concatenate([x_block, zero_pad], axis=0)
        self._n_accum_frames += min(self.N_c, xlen_block)
        xlen_block = max(xlen_block, self._factor)
        return x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block

    def ctc_reset_point_detection(self, ctc_probs_block, stdout=False):
        """Reset point detection with CTC posterior probabilities.

        Args:
            ctc_probs_block (FloatTensor): `[1, T_block, vocab]`
        Returns:
            is_reset (bool): reset encoder/decoder states if successive blank
                labels are generated above the pre-defined threshold (BLANK_THRESHOLD)

        """
        is_reset = False
        if self.safeguard_reset:
            return is_reset
        assert ctc_probs_block is not None
        topk_ids_block = ctc_probs_block[0].argmax(-1)
        bs, xmax_block, vocab = ctc_probs_block.size()
        if (topk_ids_block == self.blank_id).sum() == xmax_block:
            self._n_blanks += xmax_block
            if stdout:
                for j in range(xmax_block):
                    None
                None
            if self._n_blanks * self._factor >= self.BLANK_THRESHOLD:
                is_reset = True
            return is_reset
        n_blanks_tmp = self._n_blanks
        for j in range(xmax_block):
            if topk_ids_block[j] == self.blank_id:
                self._n_blanks += 1
                if stdout:
                    None
            else:
                if ctc_probs_block[0, j, topk_ids_block[j]] < self.SPIKE_THRESHOLD:
                    self._n_blanks += 1
                else:
                    self._n_blanks = 0
                if stdout and self.idx2token is not None:
                    None
            if self._n_blanks * self._factor >= self.BLANK_THRESHOLD:
                is_reset = True
                n_blanks_tmp = self._n_blanks
        if stdout and is_reset:
            None
        return is_reset


class TransformerDecoder(DecoderBase):
    """Transformer decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of encoder outputs
        attn_type (str): type of attention mechanism
        n_heads (int): number of attention heads
        n_layers (int): number of self-attention layers
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        ffn_bottleneck_dim (int): bottleneck dimension for light-weight FFN layer
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of embedding and output layers
        dropout (float): dropout probability for linear layers
        dropout_emb (float): dropout probability for embedding layer
        dropout_att (float): dropout probability for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        dropout_head (float): HeadDrop probability for attention heads
        lsm_prob (float): label smoothing probability
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (List): fully-connected layer configuration before the CTC softmax
        backward (bool): decode in the backward order
        global_weight (float): global loss weight for multi-task learning
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (str): parameter initialization method
        mma_chunk_size (int): chunk size for chunkwise attention. -1 means infinite lookback.
        mma_n_heads_mono (int): number of MMA head
        mma_n_heads_chunk (int): number of hard chunkwise attention head
        mma_init_r (int): initial bias value for MMA
        mma_eps (float): epsilon value for MMA
        mma_std (float): standard deviation of Gaussian noise for MMA
        mma_no_denominator (bool): remove demominator in MMA
        mma_1dconv (bool): 1dconv for MMA
        mma_quantity_loss_weight (float): quantity loss weight for MMA
        mma_headdiv_loss_weight (float): head divergence loss for MMA
        latency_metric (str): latency metric
        latency_loss_weight (float): latency loss weight for MMA
        mma_first_layer (int): first layer to enable source-target attention (start from idx:1)
        share_chunkwise_attention (bool): share chunkwise attention in the same layer of MMA
        external_lm (RNNLM): external RNNLM for LM fusion
        lm_fusion (str): type of LM fusion

    """

    def __init__(self, special_symbols, enc_n_units, attn_type, n_heads, n_layers, d_model, d_ff, ffn_bottleneck_dim, pe_type, layer_norm_eps, ffn_activation, vocab, tie_embedding, dropout, dropout_emb, dropout_att, dropout_layer, dropout_head, lsm_prob, ctc_weight, ctc_lsm_prob, ctc_fc_list, backward, global_weight, mtl_per_batch, param_init, mma_chunk_size, mma_n_heads_mono, mma_n_heads_chunk, mma_init_r, mma_eps, mma_std, mma_no_denominator, mma_1dconv, mma_quantity_loss_weight, mma_headdiv_loss_weight, latency_metric, latency_loss_weight, mma_first_layer, share_chunkwise_attention, external_lm, lm_fusion):
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
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.mtl_per_batch = mtl_per_batch
        self.prev_spk = ''
        self.lmstate_final = None
        self.embed_cache = None
        self.aws_dict = {}
        self.data_dict = {}
        self.attn_type = attn_type
        self.quantity_loss_weight = mma_quantity_loss_weight
        self._quantity_loss_weight = mma_quantity_loss_weight
        self.mma_first_layer = max(1, mma_first_layer)
        self.headdiv_loss_weight = mma_headdiv_loss_weight
        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        self.ctc_trigger = self.latency_metric in ['ctc_sync']
        if self.ctc_trigger:
            assert 0 < self.ctc_weight < 1
        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos, blank=self.blank, enc_n_units=enc_n_units, vocab=vocab, dropout=dropout, lsm_prob=ctc_lsm_prob, fc_list=ctc_fc_list, param_init=0.1, backward=backward)
        if self.att_weight > 0:
            self.embed = nn.Embedding(self.vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type, param_init)
            self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer, layer_norm_eps, ffn_activation, param_init, src_tgt_attention=False if lth < mma_first_layer - 1 else True, mma_chunk_size=mma_chunk_size, mma_n_heads_mono=mma_n_heads_mono, mma_n_heads_chunk=mma_n_heads_chunk, mma_init_r=mma_init_r, mma_eps=mma_eps, mma_std=mma_std, mma_no_denominator=mma_no_denominator, mma_1dconv=mma_1dconv, dropout_head=dropout_head, lm_fusion=lm_fusion, ffn_bottleneck_dim=ffn_bottleneck_dim, share_chunkwise_attention=share_chunkwise_attention)) for lth in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, self.vocab)
            if tie_embedding:
                self.output.weight = self.embed.weight
            self.lm = external_lm
            if external_lm is not None:
                self.lm_output_proj = nn.Linear(external_lm.output_dim, d_model)
            self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group('Transformer decoder')
        if not hasattr(args, 'transformer_layer_norm_eps'):
            group.add_argument('--transformer_ffn_bottleneck_dim', type=int, default=0, help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12, help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu', choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'], help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform', choices=['xavier_uniform', 'pytorch'], help='parameter initialization')
        group.add_argument('--transformer_dec_d_model', type=int, default=256, help='number of units in the MHA layer for Transformer decoder')
        group.add_argument('--transformer_dec_d_ff', type=int, default=2048, help='number of units in the FFN layer for Transformer decoder')
        group.add_argument('--transformer_dec_n_heads', type=int, default=4, help='number of heads in the MHA layer for Transformer decoder')
        group.add_argument('--transformer_dec_attn_type', type=str, default='scaled_dot', choices=['scaled_dot', 'mocha'], help='type of attention mechasnism for Transformer decoder')
        group.add_argument('--transformer_dec_pe_type', type=str, default='add', choices=['add', 'none', '1dconv3L'], help='type of positional encoding for the Transformer decoder')
        group.add_argument('--dropout_dec_layer', type=float, default=0.0, help='LayerDrop probability for Transformer decoder layers')
        group.add_argument('--dropout_head', type=float, default=0.0, help='HeadDrop probability for masking out a head in the Transformer decoder')
        parser.add_argument('--mocha_n_heads_mono', type=int, default=1, help='number of heads for monotonic attention')
        parser.add_argument('--mocha_n_heads_chunk', type=int, default=1, help='number of heads for chunkwise attention')
        parser.add_argument('--mocha_chunk_size', type=int, default=1, help='chunk size for MMA. -1 means infinite lookback.')
        parser.add_argument('--mocha_init_r', type=float, default=-4, help='initialization of bias parameter for monotonic attention')
        parser.add_argument('--mocha_eps', type=float, default=1e-06, help='epsilon value to avoid numerical instability for MMA')
        parser.add_argument('--mocha_std', type=float, default=1.0, help='standard deviation of Gaussian noise for MMA during training')
        parser.add_argument('--mocha_no_denominator', type=strtobool, default=False, help='remove denominator (set to 1) in the alpha recurrence in MMA')
        parser.add_argument('--mocha_1dconv', type=strtobool, default=False, help='1dconv for MMA')
        parser.add_argument('--mocha_quantity_loss_weight', type=float, default=0.0, help='quantity loss weight for MMA')
        parser.add_argument('--mocha_latency_metric', type=str, default='', choices=['', 'ctc_sync'], help='differentiable latency metric for MMA')
        parser.add_argument('--mocha_latency_loss_weight', type=float, default=0.0, help='latency loss weight for MMA')
        group.add_argument('--mocha_first_layer', type=int, default=1, help='the initial layer to have a MMA function')
        group.add_argument('--mocha_head_divergence_loss_weight', type=float, default=0.0, help='head divergence loss weight for MMA')
        group.add_argument('--share_chunkwise_attention', type=strtobool, default=False, help='share chunkwise attention heads among monotonic attention heads in the same layer')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name += '_' + args.dec_type
        dir_name += str(args.transformer_dec_d_model) + 'dmodel'
        dir_name += str(args.transformer_dec_d_ff) + 'dff'
        if args.transformer_ffn_bottleneck_dim > 0:
            dir_name += str(args.transformer_ffn_bottleneck_dim) + 'bn'
        dir_name += str(args.dec_n_layers) + 'L'
        dir_name += str(args.transformer_dec_n_heads) + 'H'
        dir_name += 'pe' + str(args.transformer_dec_pe_type)
        dir_name += args.transformer_dec_attn_type
        if args.transformer_dec_attn_type == 'mocha':
            dir_name += '_ma' + str(args.mocha_n_heads_mono) + 'H'
            dir_name += '_ca' + str(args.mocha_n_heads_chunk) + 'H'
            dir_name += '_w' + str(args.mocha_chunk_size)
            dir_name += '_bias' + str(args.mocha_init_r)
            if args.mocha_no_denominator:
                dir_name += '_denom1'
            if args.mocha_1dconv:
                dir_name += '_1dconv'
            if args.mocha_quantity_loss_weight > 0:
                dir_name += '_qua' + str(args.mocha_quantity_loss_weight)
            if args.mocha_head_divergence_loss_weight != 0:
                dir_name += '_headdiv' + str(args.mocha_head_divergence_loss_weight)
            if args.mocha_latency_metric:
                dir_name += '_' + args.mocha_latency_metric
                dir_name += str(args.mocha_latency_loss_weight)
            if args.share_chunkwise_attention:
                dir_name += '_share'
            if args.mocha_first_layer > 1:
                dir_name += '_from' + str(args.mocha_first_layer) + 'L'
        if args.dropout_dec_layer > 0:
            dir_name += '_LD' + str(args.dropout_dec_layer)
        if args.dropout_head > 0:
            dir_name += '_HD' + str(args.dropout_head)
        if args.tie_embedding:
            dir_name += '_tieemb'
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
            nn.init.constant_(self.embed.weight[self.pad], 0.0)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.constant_(self.output.bias, 0.0)

    def forward(self, eouts, elens, ys, task='all', teacher_logits=None, recog_params={}, idx2token=None, trigger_points=None):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
            trigger_points (np.ndarray): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None, 'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros(1)
        trigger_points = None
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            ctc_forced_align = self.ctc_trigger and self.training or self.attn_type == 'triggered_attention'
            loss_ctc, trigger_points = self.ctc(eouts, elens, ys, forced_align=ctc_forced_align)
            observation['loss_ctc'] = tensor2scalar(loss_ctc)
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
        if self.att_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_att, acc_att, ppl_att, losses_auxiliary = self.forward_att(eouts, elens, ys, trigger_points=trigger_points)
            observation['loss_att'] = tensor2scalar(loss_att)
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.attn_type == 'mocha':
                if self._quantity_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_quantity'] * self._quantity_loss_weight
                observation['loss_quantity'] = tensor2scalar(losses_auxiliary['loss_quantity'])
            if self.headdiv_loss_weight > 0:
                loss_att += losses_auxiliary['loss_headdiv'] * self.headdiv_loss_weight
                observation['loss_headdiv'] = tensor2scalar(losses_auxiliary['loss_headdiv'])
            if self.latency_metric:
                observation['loss_latency'] = tensor2scalar(losses_auxiliary['loss_latency']) if self.training else 0
                if self.latency_metric != 'decot' and self.latency_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_latency'] * self.latency_loss_weight
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * self.att_weight
        observation['loss'] = tensor2scalar(loss)
        return loss, observation

    def forward_att(self, eouts, elens, ys, trigger_points=None):
        """Compute XE loss for the Transformer decoder.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            trigger_points (IntTensor): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            losses_auxiliary (dict):

        """
        losses_auxiliary = {}
        ys_in, ys_out, ylens = append_sos_eos(ys, self.eos, self.eos, self.pad, self.device, self.bwd)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
        bs, ymax = ys_in.size()[:2]
        tgt_mask = (ys_out != self.pad).unsqueeze(1).repeat([1, ymax, 1])
        causal_mask = tgt_mask.new_ones(ymax, ymax, dtype=tgt_mask.dtype)
        causal_mask = torch.tril(causal_mask).unsqueeze(0)
        tgt_mask = tgt_mask & causal_mask
        src_mask = make_pad_mask(elens).unsqueeze(1).repeat([1, ymax, 1])
        if self.attn_type == 'mocha':
            attn_mask = (ys_out != self.pad).unsqueeze(1).unsqueeze(3)
        else:
            attn_mask = None
        lmout = None
        if self.lm is not None:
            self.lm.eval()
            with torch.no_grad():
                lmout, lmstate, _ = self.lm.predict(ys_in, None)
            lmout = self.lm_output_proj(lmout)
        out = self.pos_enc(self.embed_token_id(ys_in), scale=True)
        xy_aws_layers = []
        xy_aws = None
        for lth, layer in enumerate(self.layers):
            out = layer(out, tgt_mask, eouts, src_mask, mode='parallel', lmout=lmout)
            xy_aws = layer.xy_aws
            if xy_aws is not None and self.attn_type == 'mocha':
                xy_aws_masked = xy_aws.masked_fill_(attn_mask.expand_as(xy_aws) == 0, 0)
                xy_aws_layers.append(xy_aws_masked.clone())
            if not self.training:
                self.aws_dict['yy_aws_layer%d' % lth] = tensor2np(layer.yy_aws)
                self.aws_dict['xy_aws_layer%d' % lth] = tensor2np(layer.xy_aws)
                self.aws_dict['xy_aws_beta_layer%d' % lth] = tensor2np(layer.xy_aws_beta)
                self.aws_dict['xy_aws_p_choose%d' % lth] = tensor2np(layer.xy_aws_p_choose)
                self.aws_dict['yy_aws_lm_layer%d' % lth] = tensor2np(layer.yy_aws_lm)
        logits = self.output(self.norm_out(out))
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)
        losses_auxiliary['loss_quantity'] = 0.0
        if self.attn_type == 'mocha':
            n_tokens_ref = tgt_mask[:, -1, :].sum(1).float()
            n_tokens_pred = sum([torch.abs(aws.sum(3).sum(2).sum(1) / aws.size(1)) for aws in xy_aws_layers])
            n_tokens_pred /= len(xy_aws_layers)
            losses_auxiliary['loss_quantity'] = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))
        acc = compute_accuracy(logits, ys_out, self.pad)
        return loss, acc, ppl, losses_auxiliary

    def greedy(self, eouts, elens, max_len_ratio, idx2token, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, cache_states=True):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            cache_states (bool): cache decoder states for fast decoding
        Returns:
            hyps (List): length `[B]`, each of which contains arrays of size `[L]`
            aws (List): length `[B]`, each of which contains arrays of size `[H * n_layers, L, T]`

        """
        bs, xmax = eouts.size()[:2]
        ys = eouts.new_zeros((bs, 1), dtype=torch.int64).fill_(self.eos)
        for layer in self.layers:
            layer.reset()
        cache = [None] * self.n_layers
        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        xy_aws_layers_steps = []
        ymax = math.ceil(xmax * max_len_ratio)
        for i in range(ymax):
            causal_mask = eouts.new_ones(i + 1, i + 1, dtype=torch.uint8)
            causal_mask = torch.tril(causal_mask).unsqueeze(0).repeat([bs, 1, 1])
            new_cache = [None] * self.n_layers
            xy_aws_layers = []
            out = self.pos_enc(self.embed_token_id(ys), scale=True)
            for lth, layer in enumerate(self.layers):
                out = layer(out, causal_mask, eouts, None, cache=cache[lth])
                new_cache[lth] = out
                if layer.xy_aws is not None:
                    xy_aws_layers.append(layer.xy_aws[:, :, -1:])
            if cache_states:
                cache = new_cache[:]
            y = self.output(self.norm_out(out))[:, -1:].argmax(-1)
            hyps_batch += [y]
            xy_aws_layers = torch.stack(xy_aws_layers, dim=2)
            xy_aws_layers_steps.append(xy_aws_layers)
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1
            if sum(eos_flags) == bs:
                break
            if i == ymax - 1:
                break
            ys = torch.cat([ys, y], dim=-1)
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        xy_aws_layers_steps = torch.cat(xy_aws_layers_steps, dim=-2)
        xy_aws_layers_steps = xy_aws_layers_steps.reshape(bs, self.n_heads * self.n_layers, ys.size(1), xmax)
        xy_aws = tensor2np(xy_aws_layers_steps)
        if self.bwd:
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [xy_aws[b, :, :ylens[b], :][:, ::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [xy_aws[b, :, :ylens[b], :] for b in range(bs)]
        if exclude_eos:
            if self.bwd:
                hyps = [(hyps[b][1:] if eos_flags[b] else hyps[b]) for b in range(bs)]
                aws = [(aws[b][:, 1:] if eos_flags[b] else aws[b]) for b in range(bs)]
            else:
                hyps = [(hyps[b][:-1] if eos_flags[b] else hyps[b]) for b in range(bs)]
                aws = [(aws[b][:, :-1] if eos_flags[b] else aws[b]) for b in range(bs)]
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
                logger.info('=' * 200)
        return hyps, aws

    def embed_token_id(self, indices):
        """Embed token IDs.
        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.embed(indices)
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def cache_embedding(self, device):
        """Cache token emebdding."""
        if self.embed_cache is None:
            indices = torch.arange(0, self.vocab, 1, dtype=torch.int64)
            self.embed_cache = self.embed_token_id(indices)

    def beam_search(self, eouts, elens, params, idx2token=None, lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None, nbest=1, exclude_eos=False, refs_id=None, utt_ids=None, speakers=None, ensmbl_eouts=[], ensmbl_elens=[], ensmbl_decs=[], cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            params (dict): decoding hyperparameters
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh-pass LM
            lm_second (torch.nn.module): second-pass LM
            lm_second_bwd (torch.nn.module): secoding-pass backward LM
            ctc_log_probs (FloatTensor):
            nbest (int): number of N-best list
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            ensmbl_eouts (List[FloatTensor]): encoder outputs for ensemble models
            ensmbl_elens (List[IntTensor]) encoder outputs for ensemble models
            ensmbl_decs (List[torch.nn.Module): decoders for ensemble models
            cache_states (bool): cache decoder states for fast decoding
        Returns:
            nbest_hyps_idx (List): length `[B]`, each of which contains list of N hypotheses
            aws (List): length `[B]`, each of which contains arrays of size `[H, L, T]`
            scores (List):

        """
        bs, xmax, _ = eouts.size()
        n_models = len(ensmbl_decs) + 1
        beam_width = params.get('recog_beam_width')
        assert 1 <= nbest <= beam_width
        ctc_weight = params.get('recog_ctc_weight')
        max_len_ratio = params.get('recog_max_len_ratio')
        min_len_ratio = params.get('recog_min_len_ratio')
        lp_weight = params.get('recog_length_penalty')
        length_norm = params.get('recog_length_norm')
        cache_emb = params.get('recog_cache_embedding')
        lm_weight = params.get('recog_lm_weight')
        lm_weight_second = params.get('recog_lm_second_weight')
        lm_weight_second_bwd = params.get('recog_lm_bwd_weight')
        eos_threshold = params.get('recog_eos_threshold')
        lm_state_carry_over = params.get('recog_lm_state_carry_over')
        softmax_smoothing = params.get('recog_softmax_smoothing')
        eps_wait = params.get('recog_mma_delay_threshold')
        helper = BeamSearch(beam_width, self.eos, ctc_weight, lm_weight, self.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight, cache_emb)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second, cache_emb)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd, cache_emb)
        if cache_emb:
            self.cache_embedding(eouts.device)
        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)
        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            lmstate = None
            ys = eouts.new_zeros((1, 1), dtype=torch.int64).fill_(self.eos)
            for layer in self.layers:
                layer.reset()
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
            end_hyps = []
            hyps = [{'hyp': [self.eos], 'ys': ys, 'cache': None, 'score': 0.0, 'score_att': 0.0, 'score_ctc': 0.0, 'score_lm': 0.0, 'aws': [None], 'lmstate': lmstate, 'ensmbl_cache': [([None] * dec.n_layers) for dec in ensmbl_decs] if n_models > 1 else None, 'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None, 'quantity_rate': 1.0, 'streamable': True, 'streaming_failed_point': 1000}]
            streamable_global = True
            ymax = math.ceil(elens[b] * max_len_ratio)
            for i in range(ymax):
                cache = [None] * self.n_layers
                if cache_states and i > 0:
                    for lth in range(self.n_layers):
                        cache[lth] = torch.cat([beam['cache'][lth] for beam in hyps], dim=0)
                ys = eouts.new_zeros((len(hyps), i + 1), dtype=torch.int64)
                for j, beam in enumerate(hyps):
                    ys[j, :] = beam['ys']
                if i > 0:
                    xy_aws_prev = torch.cat([beam['aws'][-1] for beam in hyps], dim=0)
                else:
                    xy_aws_prev = None
                y_lm = ys[:, -1:].clone()
                _, lmstate, scores_lm = helper.update_rnnlm_state_batch(lm, hyps, y_lm)
                causal_mask = eouts.new_ones(i + 1, i + 1, dtype=torch.uint8)
                causal_mask = torch.tril(causal_mask).unsqueeze(0).repeat([ys.size(0), 1, 1])
                out = self.pos_enc(self.embed_token_id(ys), scale=True)
                n_heads_total = 0
                eouts_b = eouts[b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                new_cache = [None] * self.n_layers
                xy_aws_layers = []
                xy_aws = None
                lth_s = self.mma_first_layer - 1
                for lth, layer in enumerate(self.layers):
                    out = layer(out, causal_mask, eouts_b, None, cache=cache[lth], xy_aws_prev=xy_aws_prev[:, lth - lth_s] if lth >= lth_s and i > 0 else None, eps_wait=eps_wait)
                    xy_aws = layer.xy_aws
                    new_cache[lth] = out
                    if xy_aws is not None:
                        xy_aws_layers.append(xy_aws)
                logits = self.output(self.norm_out(out[:, -1]))
                probs = torch.softmax(logits * softmax_smoothing, dim=1)
                xy_aws_layers = torch.stack(xy_aws_layers, dim=1)
                ensmbl_cache = [([None] * dec.n_layers) for dec in ensmbl_decs]
                if n_models > 1 and cache_states and i > 0:
                    for i_e, dec in enumerate(ensmbl_decs):
                        for lth in range(dec.n_layers):
                            ensmbl_cache[i_e][lth] = torch.cat([beam['ensmbl_cache'][i_e][lth] for beam in hyps], dim=0)
                ensmbl_new_cache = [([None] * dec.n_layers) for dec in ensmbl_decs]
                for i_e, dec in enumerate(ensmbl_decs):
                    out_e = dec.pos_enc(dec.embed(ys))
                    eouts_e = ensmbl_eouts[i_e][b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                    for lth in range(dec.n_layers):
                        out_e = dec.layers[lth](out_e, causal_mask, eouts_e, None, cache=ensmbl_cache[i_e][lth])
                        ensmbl_new_cache[i_e][lth] = out_e
                    logits_e = dec.output(dec.norm_out(out_e[:, -1]))
                    probs += torch.softmax(logits_e * softmax_smoothing, dim=1)
                scores_att = torch.log(probs / n_models)
                new_hyps = []
                for j, beam in enumerate(hyps):
                    total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                    total_scores = total_scores_att * (1 - ctc_weight)
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j:j + 1, -1]
                        total_scores += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(1, self.vocab)
                    total_scores_topk, topk_ids = torch.topk(total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if lp_weight > 0:
                        total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(beam['hyp'], topk_ids, beam['ctc_state'], total_scores_topk, ctc_prefix_scorer)
                    new_aws = beam['aws'] + [xy_aws_layers[j:j + 1, :, :, -1:]]
                    aws_j = torch.cat(new_aws[1:], dim=3)
                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                        total_score = total_scores_topk[0, k].item() / length_norm_factor
                        if idx == self.eos:
                            if len(beam['hyp'][1:]) < elens[b] * min_len_ratio:
                                continue
                            max_score_no_eos = scores_att[j, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_att[j, idx + 1:].max(0)[0].item())
                            if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue
                        streaming_failed_point = beam['streaming_failed_point']
                        quantity_rate = 1.0
                        if self.attn_type == 'mocha':
                            n_tokens_hyp_k = i + 1
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
                                streaming_failed_point = i
                        new_hyps.append({'hyp': beam['hyp'] + [idx], 'ys': torch.cat([beam['ys'], eouts.new_zeros((1, 1), dtype=torch.int64).fill_(idx)], dim=-1), 'cache': [new_cache_l[j:j + 1] for new_cache_l in new_cache] if cache_states else cache, 'score': total_score, 'score_att': total_scores_att[0, idx].item(), 'score_ctc': total_scores_ctc[k].item(), 'score_lm': total_scores_lm[0, idx].item(), 'aws': new_aws, 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None, 'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None, 'ensmbl_cache': [[new_cache_e_l[j:j + 1] for new_cache_e_l in new_cache_e] for new_cache_e in ensmbl_new_cache] if cache_states else None, 'streamable': streamable_global, 'streaming_failed_point': streaming_failed_point, 'quantity_rate': quantity_rate})
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps_sorted, end_hyps, prune=True)
                hyps = new_hyps[:]
                if is_finish:
                    break
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])
            end_hyps = helper.lm_rescoring(end_hyps, lm_second, lm_weight_second, length_norm=length_norm, tag='second')
            end_hyps = helper.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd, length_norm=length_norm, tag='second_bwd')
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
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_att'] * (1 - ctc_weight)))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-pass lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-pass lm): %.7f' % (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-pass lm, reverse): %.7f' % (end_hyps[k]['score_lm_second_bwd'] * lm_weight_second_bwd))
                    if self.attn_type == 'mocha':
                        logger.info('streamable: %s' % end_hyps[k]['streamable'])
                        logger.info('streaming failed point: %d' % (end_hyps[k]['streaming_failed_point'] + 1))
                        logger.info('quantity rate [%%]: %.2f' % (end_hyps[k]['quantity_rate'] * 100))
                    logger.info('-' * 50)
                if self.attn_type == 'mocha' and end_hyps[0]['streaming_failed_point'] < 1000:
                    assert not self.streamable
                    aws_last_success = end_hyps[0]['aws'][1:][end_hyps[0]['streaming_failed_point'] - 1]
                    rightmost_frame = max(0, aws_last_success[0, :, 0].nonzero()[:, -1].max().item()) + 1
                    frame_ratio = rightmost_frame * 100 / xmax
                    self.last_success_frame_ratio = frame_ratio
                    logger.info('streaming last success frame ratio: %.2f' % frame_ratio)
            if self.bwd:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [[tensor2np(torch.cat(end_hyps[n]['aws'][1:][::-1], dim=2).squeeze(0)) for n in range(nbest)]]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [[tensor2np(torch.cat(end_hyps[n]['aws'][1:], dim=2).squeeze(0)) for n in range(nbest)]]
            scores += [[end_hyps[n]['score_att'] for n in range(nbest)]]
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][1:] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
                aws = [[(aws[b][n][:, 1:] if eos_flags[b][n] else aws[b][n]) for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[(nbest_hyps_idx[b][n][:-1] if eos_flags[b][n] else nbest_hyps_idx[b][n]) for n in range(nbest)] for b in range(bs)]
                aws = [[(aws[b][n][:, :-1] if eos_flags[b][n] else aws[b][n]) for n in range(nbest)] for b in range(bs)]
        if bs == 1:
            self.lmstate_final = end_hyps[0]['lmstate']
        return nbest_hyps_idx, aws, scores


def add_input_noise(xs, std):
    noise = torch.normal(xs.new_zeros(xs.shape[-1]), std)
    xs.data += noise
    return xs


def build_decoder(args, special_symbols, enc_n_units, vocab, ctc_weight, global_weight, external_lm=None):
    if not hasattr(args, 'transformer_dec_d_model') and hasattr(args, 'transformer_d_model'):
        args.transformer_dec_d_model = args.transformer_d_model
    if not hasattr(args, 'transformer_dec_d_ff') and hasattr(args, 'transformer_d_ff'):
        args.transformer_dec_d_ff = args.transformer_d_ff
    if not hasattr(args, 'transformer_dec_n_heads') and hasattr(args, 'transformer_n_heads'):
        args.transformer_dec_n_heads = args.transformer_n_heads
    if not hasattr(args, 'transformer_dec_attn_type') and hasattr(args, 'transformer_attn_type'):
        args.transformer_dec_attn_type = args.transformer_attn_type
    if args.dec_type in ['transformer', 'transformer_xl']:
        decoder = TransformerDecoder(special_symbols=special_symbols, enc_n_units=enc_n_units, attn_type=args.transformer_dec_attn_type, n_heads=args.transformer_dec_n_heads, n_layers=args.dec_n_layers, d_model=args.transformer_dec_d_model, d_ff=args.transformer_dec_d_ff, ffn_bottleneck_dim=args.transformer_ffn_bottleneck_dim, pe_type=args.transformer_dec_pe_type, layer_norm_eps=args.transformer_layer_norm_eps, ffn_activation=args.transformer_ffn_activation, vocab=vocab, tie_embedding=args.tie_embedding, dropout=args.dropout_dec, dropout_emb=args.dropout_emb, dropout_att=args.dropout_att, dropout_layer=args.dropout_dec_layer, dropout_head=args.dropout_head, lsm_prob=args.lsm_prob, ctc_weight=ctc_weight, ctc_lsm_prob=args.ctc_lsm_prob, ctc_fc_list=args.ctc_fc_list, backward=dir == 'bwd', global_weight=global_weight, mtl_per_batch=args.mtl_per_batch, param_init=args.transformer_param_init, mma_chunk_size=args.mocha_chunk_size, mma_n_heads_mono=args.mocha_n_heads_mono, mma_n_heads_chunk=args.mocha_n_heads_chunk, mma_init_r=args.mocha_init_r, mma_eps=args.mocha_eps, mma_std=args.mocha_std, mma_no_denominator=args.mocha_no_denominator, mma_1dconv=args.mocha_1dconv, mma_quantity_loss_weight=args.mocha_quantity_loss_weight, mma_headdiv_loss_weight=args.mocha_head_divergence_loss_weight, latency_metric=str(args.mocha_latency_metric), latency_loss_weight=args.mocha_latency_loss_weight, mma_first_layer=args.mocha_first_layer, share_chunkwise_attention=args.share_chunkwise_attention, external_lm=external_lm, lm_fusion=args.lm_fusion)
    elif args.dec_type in ['lstm_transducer', 'gru_transducer']:
        decoder = RNNTransducer(special_symbols=special_symbols, enc_n_units=enc_n_units, n_units=args.dec_n_units, n_projs=args.dec_n_projs, n_layers=args.dec_n_layers, bottleneck_dim=args.dec_bottleneck_dim, emb_dim=args.emb_dim, vocab=vocab, dropout=args.dropout_dec, dropout_emb=args.dropout_emb, ctc_weight=ctc_weight, ctc_lsm_prob=args.ctc_lsm_prob, ctc_fc_list=args.ctc_fc_list, external_lm=external_lm if args.lm_init else None, global_weight=global_weight, mtl_per_batch=args.mtl_per_batch, param_init=args.param_init)
    else:
        decoder = RNNDecoder(special_symbols=special_symbols, enc_n_units=enc_n_units, n_units=args.dec_n_units, n_projs=args.dec_n_projs, n_layers=args.dec_n_layers, bottleneck_dim=args.dec_bottleneck_dim, emb_dim=args.emb_dim, vocab=vocab, tie_embedding=args.tie_embedding, attn_type=args.attn_type, attn_dim=args.attn_dim, attn_sharpening_factor=args.attn_sharpening_factor, attn_sigmoid_smoothing=args.attn_sigmoid, attn_conv_out_channels=args.attn_conv_n_channels, attn_conv_kernel_size=args.attn_conv_width, attn_n_heads=args.attn_n_heads, dropout=args.dropout_dec, dropout_emb=args.dropout_emb, dropout_att=args.dropout_att, lsm_prob=args.lsm_prob, ss_prob=args.ss_prob, ctc_weight=ctc_weight, ctc_lsm_prob=args.ctc_lsm_prob, ctc_fc_list=args.ctc_fc_list, mbr_training=args.mbr_training, mbr_ce_weight=args.mbr_ce_weight, external_lm=external_lm, lm_fusion=args.lm_fusion, lm_init=args.lm_init, backward=dir == 'bwd', global_weight=global_weight, mtl_per_batch=args.mtl_per_batch, param_init=args.param_init, mocha_chunk_size=args.mocha_chunk_size, mocha_n_heads_mono=args.mocha_n_heads_mono, mocha_init_r=args.mocha_init_r, mocha_eps=args.mocha_eps, mocha_std=args.mocha_std, mocha_no_denominator=args.mocha_no_denominator, mocha_1dconv=args.mocha_1dconv, mocha_decot_lookahead=args.mocha_decot_lookahead, quantity_loss_weight=args.mocha_quantity_loss_weight, latency_metric=str(args.mocha_latency_metric), latency_loss_weight=args.mocha_latency_loss_weight, mocha_stableemit_weight=args.mocha_stableemit_weight, gmm_attn_n_mixtures=args.gmm_attn_n_mixtures, replace_sos=args.replace_sos, distillation_weight=args.distillation_weight, discourse_aware=args.discourse_aware)
    return decoder


def build_encoder(args):
    if 'conv' in args.enc_type:
        assert args.n_stacks == 1 and args.n_splices == 1
        conv = ConvEncoder(args.input_dim, in_channel=args.conv_in_channel, channels=args.conv_channels, kernel_sizes=args.conv_kernel_sizes, strides=args.conv_strides, poolings=args.conv_poolings, dropout=0.0, normalization=args.conv_normalization, residual=False, bottleneck_dim=args.transformer_enc_d_model if 'former' in args.enc_type else args.conv_bottleneck_dim, param_init=args.param_init)
    else:
        conv = None
    if not hasattr(args, 'transformer_enc_d_model') and hasattr(args, 'transformer_d_model'):
        args.transformer_enc_d_model = args.transformer_d_model
        args.transformer_dec_d_model = args.transformer_d_model
    if not hasattr(args, 'transformer_enc_d_ff') and hasattr(args, 'transformer_d_ff'):
        args.transformer_enc_d_ff = args.transformer_d_ff
    if not hasattr(args, 'transformer_enc_n_heads') and hasattr(args, 'transformer_n_heads'):
        args.transformer_enc_n_heads = args.transformer_n_heads
    if args.enc_type == 'tds':
        encoder = TDSEncoder(input_dim=args.input_dim * args.n_stacks, in_channel=args.conv_in_channel, channels=args.conv_channels, kernel_sizes=args.conv_kernel_sizes, dropout=args.dropout_enc, last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else args.dec_n_units)
    elif args.enc_type == 'gated_conv':
        encoder = GatedConvEncoder(input_dim=args.input_dim * args.n_stacks, in_channel=args.conv_in_channel, channels=args.conv_channels, kernel_sizes=args.conv_kernel_sizes, dropout=args.dropout_enc, last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else args.dec_n_units, param_init=args.param_init)
    elif 'transformer' in args.enc_type:
        encoder = TransformerEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim, enc_type=args.enc_type, n_heads=args.transformer_enc_n_heads, n_layers=args.enc_n_layers, n_layers_sub1=args.enc_n_layers_sub1, n_layers_sub2=args.enc_n_layers_sub2, d_model=args.transformer_enc_d_model, d_ff=args.transformer_enc_d_ff, ffn_bottleneck_dim=args.transformer_ffn_bottleneck_dim, ffn_activation=args.transformer_ffn_activation, pe_type=args.transformer_enc_pe_type, layer_norm_eps=args.transformer_layer_norm_eps, last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else 0, dropout_in=args.dropout_in, dropout=args.dropout_enc, dropout_att=args.dropout_att, dropout_layer=args.dropout_enc_layer, subsample=args.subsample, subsample_type=args.subsample_type, n_stacks=args.n_stacks, n_splices=args.n_splices, frontend_conv=conv, task_specific_layer=args.task_specific_layer, param_init=args.transformer_param_init, clamp_len=args.transformer_enc_clamp_len, lookahead=args.transformer_enc_lookaheads, chunk_size_left=args.lc_chunk_size_left, chunk_size_current=args.lc_chunk_size_current, chunk_size_right=args.lc_chunk_size_right, streaming_type=args.lc_type)
    elif 'conformer' in args.enc_type:
        encoder = ConformerEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim, enc_type=args.enc_type, n_heads=args.transformer_enc_n_heads, kernel_size=args.conformer_kernel_size, normalization=args.conformer_normalization, n_layers=args.enc_n_layers, n_layers_sub1=args.enc_n_layers_sub1, n_layers_sub2=args.enc_n_layers_sub2, d_model=args.transformer_enc_d_model, d_ff=args.transformer_enc_d_ff, ffn_bottleneck_dim=args.transformer_ffn_bottleneck_dim, ffn_activation='swish', pe_type=args.transformer_enc_pe_type, layer_norm_eps=args.transformer_layer_norm_eps, last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else 0, dropout_in=args.dropout_in, dropout=args.dropout_enc, dropout_att=args.dropout_att, dropout_layer=args.dropout_enc_layer, subsample=args.subsample, subsample_type=args.subsample_type, n_stacks=args.n_stacks, n_splices=args.n_splices, frontend_conv=conv, task_specific_layer=args.task_specific_layer, param_init=args.transformer_param_init, clamp_len=args.transformer_enc_clamp_len, lookahead=args.transformer_enc_lookaheads, chunk_size_left=args.lc_chunk_size_left, chunk_size_current=args.lc_chunk_size_current, chunk_size_right=args.lc_chunk_size_right, streaming_type=args.lc_type)
    else:
        encoder = RNNEncoder(input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim, enc_type=args.enc_type, n_units=args.enc_n_units, n_projs=args.enc_n_projs, last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else 0, n_layers=args.enc_n_layers, n_layers_sub1=args.enc_n_layers_sub1, n_layers_sub2=args.enc_n_layers_sub2, dropout_in=args.dropout_in, dropout=args.dropout_enc, subsample=args.subsample, subsample_type=args.subsample_type, n_stacks=args.n_stacks, n_splices=args.n_splices, frontend_conv=conv, bidir_sum_fwd_bwd=args.bidirectional_sum_fwd_bwd, task_specific_layer=args.task_specific_layer, param_init=args.param_init, chunk_size_current=args.lc_chunk_size_left, chunk_size_right=args.lc_chunk_size_right, cnn_lookahead=args.cnn_lookahead, rsp_prob=args.rsp_prob_enc)
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


def load_checkpoint(checkpoint_path, model=None, scheduler=None, amp=None):
    """Load checkpoint.

    Args:
        checkpoint_path (str): path to the saved model (model.epoch-*)
        model (torch.nn.Module):
        scheduler (LRScheduler): optimizer wrapped by LRScheduler class
        amp ():
    Returns:
        topk_list (List): (epoch, metric)

    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        raise ValueError('No checkpoint found at %s' % checkpoint_path)
    if 'avg' not in checkpoint_path:
        epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) - 1
        logger.info('=> Loading checkpoint (epoch:%d): %s' % (epoch + 1, checkpoint_path))
    else:
        logger.info('=> Loading checkpoint: %s' % checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.optimizer.param_groups[0]['params'] = []
        for param_group in list(model.parameters()):
            scheduler.optimizer.param_groups[0]['params'].append(param_group)
    else:
        logger.warning('Scheduler/Optimizer is not loaded.')
    if amp is not None:
        amp.load_state_dict(checkpoint['amp_state_dict'])
    else:
        logger.warning('amp is not loaded.')
    if 'optimizer_state_dict' in checkpoint.keys() and 'topk_list' in checkpoint['optimizer_state_dict'].keys():
        topk_list = checkpoint['optimizer_state_dict']['topk_list']
    else:
        topk_list = []
    return topk_list


def splice(x, n_splices=1, n_stacks=1, dtype=np.float32):
    """Splice input data. This is expected to be used for CNN-like encoder.

    Args:
        x (np.ndarray): A tensor of size
            `[T, input_dim (F * 3 * n_stacks)]'
        n_splices (int): frames to n_splices. Default is 1 frame.
            ex.) if n_splices == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        n_stacks (int): the number of stacked frames in frame stacking
        dtype ():
    Returns:
        feat_splice (np.ndarray): A tensor of size
            `[T, F * (n_splices * n_stacks) * 3 (static + Δ + ΔΔ)]`

    """
    if n_splices == 1:
        return x
    assert isinstance(x, np.ndarray), 'x should be np.ndarray.'
    assert len(x.shape) == 2, 'x must be 2 dimension.'
    is_delta = x.shape[-1] // n_stacks % 3 == 0
    n_delta = 3 if is_delta else 1
    None
    T, input_dim = x.shape
    F = input_dim // n_delta // n_stacks
    feat_splice = np.zeros((T, F * (n_splices * n_stacks) * n_delta), dtype=dtype)
    for i_time in range(T):
        spliced_frames = np.zeros((n_splices * n_stacks, F, n_delta))
        for i_splice in range(0, n_splices, 1):
            if i_time <= n_splices - 1 and i_splice < n_splices - i_time:
                copy_frame = x[0]
            elif T - n_splices <= i_time and i_time + (i_splice - n_splices) > T - 1:
                copy_frame = x[-1]
            else:
                copy_frame = x[i_time + (i_splice - n_splices)]
            copy_frame = copy_frame.reshape((F, n_delta, n_stacks))
            copy_frame = np.transpose(copy_frame, (2, 0, 1))
            spliced_frames[i_splice:i_splice + n_stacks] = copy_frame
        spliced_frames = np.transpose(spliced_frames, (1, 0, 2))
        feat_splice[i_time] = spliced_frames.reshape(F * (n_splices * n_stacks) * n_delta)
    return feat_splice


def stack_frame(x, n_stacks, n_skips, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        x (np.ndarray): `[T, input_dim]`
        n_stacks (int): the number of frames to stack
        n_skips (int): the number of frames to skip
        dtype ():
    Returns:
        stacked_feat (np.ndarray): `[floor(T / n_skips), input_dim * n_stacks]`

    """
    if n_stacks == 1 and n_skips == 1:
        return x
    if n_stacks < n_skips:
        raise ValueError('n_skips must be less than n_stacks.')
    assert isinstance(x, np.ndarray), 'x should be np.ndarray.'
    T, input_dim = x.shape
    T_new = T // n_skips if T % n_stacks == 0 else T // n_skips + 1
    stacked_feat = np.zeros((T_new, input_dim * n_stacks), dtype=dtype)
    stack_count = 0
    stack = []
    for t, frame_t in enumerate(x):
        if t == len(x) - 1:
            stack.append(frame_t)
            while stack_count != int(T_new):
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
        self.main_weight = args.total_weight - args.sub1_weight - args.sub2_weight
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
        self.input_noise_std = args.input_noise_std
        self.n_stacks = args.n_stacks
        self.n_skips = args.n_skips
        self.n_splices = args.n_splices
        self.weight_noise_std = args.weight_noise_std
        self.specaug = None
        if args.n_freq_masks > 0 or args.n_time_masks > 0:
            assert args.n_stacks == 1 and args.n_skips == 1
            assert args.n_splices == 1
            self.specaug = SpecAugment(F=args.freq_width, T=args.time_width, n_freq_masks=args.n_freq_masks, n_time_masks=args.n_time_masks, p=args.time_width_upper, adaptive_number_ratio=args.adaptive_number_ratio, adaptive_size_ratio=args.adaptive_size_ratio, max_n_time_masks=args.max_n_time_masks)
        self.ssn = None
        if args.sequence_summary_network:
            assert args.input_type == 'speech'
            self.ssn = SequenceSummaryNetwork(args.input_dim, n_units=512, n_layers=3, bottleneck_dim=100, dropout=0, param_init=args.param_init)
        self.enc = build_encoder(args)
        if args.freeze_encoder:
            for n, p in self.enc.named_parameters():
                if 'bridge' in n or 'sub1' in n:
                    continue
                p.requires_grad = False
                logger.info('freeze %s' % n)
        special_symbols = {'blank': self.blank, 'unk': self.unk, 'eos': self.eos, 'pad': self.pad}
        external_lm = None
        directions = []
        if self.fwd_weight > 0 or self.bwd_weight == 0 and self.ctc_weight > 0:
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')
        for dir in directions:
            if args.external_lm and dir == 'fwd':
                external_lm = RNNLM(args.lm_conf)
                load_checkpoint(args.external_lm, external_lm)
                for n, p in external_lm.named_parameters():
                    p.requires_grad = False
            dec = build_decoder(args, special_symbols, self.enc.output_dim, args.vocab, self.ctc_weight, self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight, external_lm)
            setattr(self, 'dec_' + dir, dec)
        for sub in ['sub1', 'sub2']:
            if getattr(self, sub + '_weight') > 0:
                args_sub = copy.deepcopy(args)
                if hasattr(args, 'dec_config_' + sub):
                    for k, v in getattr(args, 'dec_config_' + sub).items():
                        setattr(args_sub, k, v)
                dec_sub = build_decoder(args_sub, special_symbols, getattr(self.enc, 'output_dim_' + sub), getattr(self, 'vocab_' + sub), getattr(self, 'ctc_weight_' + sub), getattr(self, sub + '_weight'), external_lm)
                setattr(self, 'dec_fwd_' + sub, dec_sub)
        if args.input_type == 'text':
            if args.vocab == args.vocab_sub1:
                self.embed = dec.embed
            else:
                self.embed = nn.Embedding(args.vocab_sub1, args.emb_dim, padding_idx=self.pad)
                self.dropout_emb = nn.Dropout(p=args.dropout_emb)
        if args.lm_fusion == 'deep' and external_lm is not None:
            for n, p in self.named_parameters():
                if 'output' in n or 'output_bn' in n or 'linear' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def trigger_scheduled_sampling(self):
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).trigger_scheduled_sampling()
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).trigger_scheduled_sampling()

    def trigger_quantity_loss(self):
        if hasattr(self, 'dec_fwd'):
            getattr(self, 'dec_fwd').trigger_quantity_loss()
            getattr(self, 'dec_fwd').trigger_latency_loss()

    def trigger_stableemit(self):
        if hasattr(self, 'dec_fwd'):
            getattr(self, 'dec_fwd').trigger_stableemit()

    def reset_session(self):
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).reset_session()
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).reset_session()

    def forward(self, batch, task, is_eval=False, teacher=None, teacher_lm=None):
        """Forward pass.

        Args:
            batch (dict):
                xs (List): input data of size `[T, input_dim]`
                xlens (List): lengths of each element in xs
                ys (List): reference labels in the main task of size `[L]`
                ys_sub1 (List): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (List): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (List): name of utterances
                speakers (List): name of speakers
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
        loss = torch.zeros((1,), dtype=torch.float32, device=self.device)
        if (self.fwd_weight > 0 or self.bwd_weight == 0 and self.ctc_weight > 0 or self.mbr_training) and task in ['all', 'ys', 'ys.ctc', 'ys.mbr']:
            teacher_logits = None
            if teacher is not None:
                teacher.eval()
                teacher_logits = teacher.generate_logits(batch)
            elif teacher_lm is not None:
                teacher_lm.eval()
                teacher_logits = self.generate_lm_logits(batch['ys'], lm=teacher_lm)
            loss_fwd, obs_fwd = self.dec_fwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'], task, teacher_logits, self.recog_params, self.idx2token, batch['trigger_points'])
            loss += loss_fwd
            if isinstance(self.dec_fwd, RNNT):
                observation['loss.transducer'] = obs_fwd['loss_transducer']
            else:
                observation['acc.att'] = obs_fwd['acc_att']
                observation['ppl.att'] = obs_fwd['ppl_att']
                observation['loss.att'] = obs_fwd['loss_att']
                observation['loss.mbr'] = obs_fwd['loss_mbr']
                if 'loss_quantity' not in obs_fwd.keys():
                    obs_fwd['loss_quantity'] = None
                observation['loss.quantity'] = obs_fwd['loss_quantity']
                if 'loss_latency' not in obs_fwd.keys():
                    obs_fwd['loss_latency'] = None
                observation['loss.latency'] = obs_fwd['loss_latency']
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
                if len(batch['ys_' + sub]) == 0:
                    continue
                loss_sub, obs_fwd_sub = getattr(self, 'dec_fwd_' + sub)(eout_dict['ys_' + sub]['xs'], eout_dict['ys_' + sub]['xlens'], batch['ys_' + sub], task)
                loss += loss_sub
                if isinstance(getattr(self, 'dec_fwd_' + sub), RNNT):
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
        ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device) for y in ys]
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        lmout, _ = lm.decode(ys_in, None)
        logits = lm.output(lmout)
        return logits

    def encode(self, xs, task='all', streaming=False, cnn_lookback=False, cnn_lookahead=False, xlen_block=-1):
        """Encode acoustic or text features.

        Args:
            xs (List): length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all/ys*/ys_sub1*/ys_sub2*
            streaming (bool): streaming encoding
            cnn_lookback (bool): truncate leftmost frames for lookback in CNN context
            cnn_lookahead (bool): truncate rightmost frames for lookahead in CNN context
            xlen_block (int): input length in a block in the streaming mode
        Returns:
            eout_dict (dict):

        """
        if self.input_type == 'speech':
            if self.n_stacks > 1:
                xs = [stack_frame(x, self.n_stacks, self.n_skips) for x in xs]
            if self.n_splices > 1:
                xs = [splice(x, self.n_splices, self.n_stacks) for x in xs]
            if streaming:
                xlens = torch.IntTensor([xlen_block])
            else:
                xlens = torch.IntTensor([len(x) for x in xs])
            xs = pad_list([np2tensor(x, self.device).float() for x in xs], 0.0)
            if self.specaug is not None and self.training:
                xs = self.specaug(xs)
            if self.weight_noise_std > 0 and self.training:
                self.add_weight_noise(std=self.weight_noise_std)
            if self.input_noise_std > 0 and self.training:
                xs = add_input_noise(xs, std=self.input_noise_std)
            if self.ssn is not None:
                xs = self.ssn(xs, xlens)
        elif self.input_type == 'text':
            xlens = torch.IntTensor([len(x) for x in xs])
            xs = [np2tensor(np.fromiter(x, dtype=np.int64), self.device) for x in xs]
            xs = pad_list(xs, self.pad)
            xs = self.dropout_emb(self.embed(xs))
        eout_dict = self.enc(xs, xlens, task.split('.')[0], streaming, cnn_lookback, cnn_lookahead)
        if self.main_weight < 1 and self.enc_type in ['conv', 'tds', 'gated_conv']:
            for sub in ['sub1', 'sub2']:
                eout_dict['ys_' + sub]['xs'] = eout_dict['ys']['xs'].clone()
                eout_dict['ys_' + sub]['xlens'] = eout_dict['ys']['xlens'][:]
        return eout_dict

    def get_ctc_probs(self, xs, task='ys', temperature=1, topk=None):
        """Get CTC top-K probabilities.

        Args:
            xs (FloatTensor): `[B, T, idim]`
            task (str): task to evaluate
            temperature (float): softmax temperature
            topk (int): top-K classes to sample
        Returns:
            probs (np.ndarray): `[B, T, vocab]`
            topk_ids (np.ndarray): `[B, T, topk]`
            elens (IntTensor): `[B]`

        """
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
            probs = getattr(self, 'dec_' + dir).ctc.probs(eout_dict[task]['xs'])
            if topk is None:
                topk = probs.size(-1)
            _, topk_ids = torch.topk(probs, k=topk, dim=-1, largest=True, sorted=True)
            return tensor2np(probs), tensor2np(topk_ids), eout_dict[task]['xlens']

    def ctc_forced_align(self, xs, ys, task='ys'):
        """CTC-based forced alignment.

        Args:
            xs (FloatTensor): `[B, T, idim]`
            ys (List): length `B`, each of which contains a list of size `[L]`
        Returns:
            trigger_points (np.ndarray): `[B, L]`

        """
        forced_aligner = CTCForcedAligner()
        self.eval()
        with torch.no_grad():
            eout_dict = self.encode(xs, 'ys')
            ctc = getattr(self, 'dec_fwd').ctc
            logits = ctc.output(eout_dict[task]['xs'])
            ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
            trigger_points = forced_aligner(logits, eout_dict[task]['xlens'], ys, ylens)
        return tensor2np(trigger_points)

    def plot_attention(self):
        """Plot attention weights during training."""
        self.enc._plot_attention(mkdir_join(self.save_path, 'enc_att_weights'))
        self.dec_fwd._plot_attention(mkdir_join(self.save_path, 'dec_att_weights'))
        if getattr(self, 'dec_fwd_sub1', None) is not None:
            self.dec_fwd_sub1._plot_attention(mkdir_join(self.save_path, 'dec_att_weights_sub1'))
        if getattr(self, 'dec_fwd_sub2', None) is not None:
            self.dec_fwd_sub2._plot_attention(mkdir_join(self.save_path, 'dec_att_weights_sub2'))

    def plot_ctc(self):
        """Plot CTC posteriors during training."""
        self.dec_fwd._plot_ctc(mkdir_join(self.save_path, 'ctc'))
        if getattr(self, 'dec_fwd_sub1', None) is not None:
            self.dec_fwd_sub1._plot_ctc(mkdir_join(self.save_path, 'ctc_sub1'))
        if getattr(self, 'dec_fwd_sub2', None) is not None:
            self.dec_fwd_sub2._plot_ctc(mkdir_join(self.save_path, 'ctc_sub2'))

    def encode_streaming(self, xs, params, task='ys'):
        """Simulate streaming encoding. Decoding is performed in the offline mode.
        Args:
            xs (FloatTensor): `[B, T, idim]`
            params (dict): hyper-parameters for decoding
            task (str): task to evaluate
        Returns:
            eout (FloatTensor): `[B, T, idim]`
            elens (IntTensor): `[B]`

        """
        assert task == 'ys'
        assert self.input_type == 'speech'
        assert self.fwd_weight > 0
        assert len(xs) == 1
        streaming = Streaming(xs[0], params, self.enc)
        self.enc.reset_cache()
        while True:
            x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block = streaming.extract_feat()
            eout_block_dict = self.encode([x_block], 'all', streaming=True, cnn_lookback=cnn_lookback, cnn_lookahead=cnn_lookahead, xlen_block=xlen_block)
            eout_block = eout_block_dict[task]['xs']
            streaming.cache_eout(eout_block)
            streaming.next_block()
            if is_last_block:
                break
        eout = streaming.pop_eouts()
        elens = torch.IntTensor([eout.size(1)])
        return eout, elens

    @torch.no_grad()
    def decode_streaming(self, xs, params, idx2token, exclude_eos=False, speaker=None, task='ys'):
        """Simulate streaming encoding+decoding. Both encoding and decoding are performed in the online mode."""
        self.eval()
        block_size = params.get('recog_block_sync_size')
        cache_emb = params.get('recog_cache_embedding')
        ctc_weight = params.get('recog_ctc_weight')
        backoff = True
        assert task == 'ys'
        assert self.input_type == 'speech'
        assert self.ctc_weight > 0
        assert self.fwd_weight > 0 or self.ctc_weight == 1.0
        assert len(xs) == 1
        assert params.get('recog_block_sync')
        streaming = Streaming(xs[0], params, self.enc, idx2token)
        factor = self.enc.subsampling_factor
        block_size //= factor
        assert block_size >= 1, 'block_size is too small.'
        is_transformer_enc = 'former' in self.enc.enc_type
        hyps = None
        hyps_nobd = []
        best_hyp_id_session = []
        is_reset = False
        helper = BeamSearch(params.get('recog_beam_width'), self.eos, params.get('recog_ctc_weight'), params.get('recog_lm_weight'), self.device)
        lm = getattr(self, 'lm_fwd', None)
        lm_second = getattr(self, 'lm_second', None)
        lm = helper.verify_lm_eval_mode(lm, params.get('recog_lm_weight'), cache_emb)
        if lm is not None:
            assert isinstance(lm, RNNLM)
        lm_second = helper.verify_lm_eval_mode(lm_second, params.get('recog_lm_second_weight'), cache_emb)
        if cache_emb and self.fwd_weight > 0:
            self.dec_fwd.cache_embedding(self.device)
        self.enc.reset_cache()
        eout_block_tail = None
        x_block_prev, xlen_block_prev = None, None
        while True:
            x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block = streaming.extract_feat()
            if not is_transformer_enc and is_reset:
                self.enc.reset_cache()
                if backoff:
                    self.encode([x_block_prev], 'all', streaming=True, cnn_lookback=cnn_lookback, cnn_lookahead=cnn_lookahead, xlen_block=xlen_block_prev)
            x_block_prev = x_block
            xlen_block_prev = xlen_block
            eout_block_dict = self.encode([x_block], 'all', streaming=True, cnn_lookback=cnn_lookback, cnn_lookahead=cnn_lookahead, xlen_block=xlen_block)
            eout_block = eout_block_dict[task]['xs']
            if eout_block_tail is not None:
                eout_block = torch.cat([eout_block_tail, eout_block], dim=1)
                eout_block_tail = None
            if eout_block.size(1) > 0:
                streaming.cache_eout(eout_block)
                if ctc_weight == 1 or self.ctc_weight == 1:
                    end_hyps, hyps = self.dec_fwd.ctc.beam_search_block_sync(eout_block, params, helper, idx2token, hyps, lm)
                elif isinstance(self.dec_fwd, RNNT):
                    end_hyps, hyps = self.dec_fwd.beam_search_block_sync(eout_block, params, helper, idx2token, hyps, lm)
                elif isinstance(self.dec_fwd, RNNDecoder):
                    n_frames = getattr(self.dec_fwd, 'n_frames', 0)
                    for i in range(math.ceil(eout_block.size(1) / block_size)):
                        eout_block_i = eout_block[:, i * block_size:(i + 1) * block_size]
                        end_hyps, hyps, hyps_nobd = self.dec_fwd.beam_search_block_sync(eout_block_i, params, helper, idx2token, hyps, hyps_nobd, lm, speaker=speaker)
                elif isinstance(self.dec_fwd, TransformerDecoder):
                    raise NotImplementedError
                else:
                    raise NotImplementedError(self.dec_fwd)
                is_reset = False
                if streaming.enable_ctc_reset_point_detection:
                    if self.ctc_weight_sub1 > 0:
                        ctc_probs_block = self.dec_fwd_sub1.ctc.probs(eout_block_dict['ys_sub1']['xs'])
                    else:
                        ctc_probs_block = self.dec_fwd.ctc.probs(eout_block)
                    is_reset = streaming.ctc_reset_point_detection(ctc_probs_block)
                merged_hyps = sorted(end_hyps + hyps + hyps_nobd, key=lambda x: x['score'], reverse=True)
                if len(merged_hyps) > 0:
                    best_hyp_id_prefix = np.array(merged_hyps[0]['hyp'][1:])
                    best_hyp_id_prefix_viz = np.array(merged_hyps[0]['hyp'][1:])
                    if len(best_hyp_id_prefix) > 0 and best_hyp_id_prefix[-1] == self.eos:
                        best_hyp_id_prefix = best_hyp_id_prefix[:-1]
                        if not is_reset and not streaming.safeguard_reset:
                            is_reset = True
                    if len(best_hyp_id_prefix_viz) > 0:
                        n_frames = self.dec_fwd.ctc.n_frames if ctc_weight == 1 or self.ctc_weight == 1 else self.dec_fwd.n_frames
                        None
            if is_reset:
                if len(best_hyp_id_prefix) > 0:
                    best_hyp_id_session.extend(best_hyp_id_prefix)
                streaming.reset()
                hyps = None
                hyps_nobd = []
            streaming.next_block()
            if is_last_block:
                break
        if not is_reset and len(best_hyp_id_prefix) > 0:
            best_hyp_id_session.extend(best_hyp_id_prefix)
        if len(best_hyp_id_session) > 0:
            return [[np.stack(best_hyp_id_session, axis=0)]], [None]
        else:
            return [[[]]], [None]

    def streamable(self):
        return getattr(self.dec_fwd, 'streamable', False)

    def quantity_rate(self):
        return getattr(self.dec_fwd, 'quantity_rate', 1.0)

    def last_success_frame_ratio(self):
        return getattr(self.dec_fwd, 'last_success_frame_ratio', 0)

    @torch.no_grad()
    def decode(self, xs, params, idx2token, exclude_eos=False, refs_id=None, refs=None, utt_ids=None, speakers=None, task='ys', ensemble_models=[], trigger_points=None, teacher_force=False):
        """Decode in the inference stage.

        Args:
            xs (List): length `[B]`, which contains arrays of size `[T, input_dim]`
            params (dict): hyper-parameters for decoding
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from best_hyps_id
            refs_id (List): gold token IDs to compute log likelihood
            refs (List): gold transcriptions
            utt_ids (List): utterance id list
            speakers (List): speaker list
            task (str): ys* or ys_sub1* or ys_sub2*
            ensemble_models (List): Speech2Text classes
            trigger_points (np.ndarray): `[B, L]`
            teacher_force (bool): conduct teacher-forcing
        Returns:
            nbest_hyps_id (List[List[np.ndarray]]): length `[B]`, which contains a list of length `[n_best]` which contains arrays of size `[L]`
            aws (List[np.ndarray]): length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        self.eval()
        if task.split('.')[0] == 'ys':
            dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
        elif task.split('.')[0] == 'ys_sub1':
            dir = 'fwd_sub1'
        elif task.split('.')[0] == 'ys_sub2':
            dir = 'fwd_sub2'
        else:
            raise ValueError(task)
        if utt_ids is not None:
            if self.utt_id_prev != utt_ids[0]:
                self.reset_session()
            self.utt_id_prev = utt_ids[0]
        if params['recog_streaming_encoding']:
            eouts, elens = self.encode_streaming(xs, params, task)
        else:
            eout_dict = self.encode(xs, task)
            eouts = eout_dict[task]['xs']
            elens = eout_dict[task]['xlens']
        if self.fwd_weight == 0 and self.bwd_weight == 0 or self.ctc_weight > 0 and params['recog_ctc_weight'] == 1:
            lm = getattr(self, 'lm_' + dir, None)
            lm_second = getattr(self, 'lm_second', None)
            lm_second_bwd = None
            if params.get('recog_beam_width') == 1:
                nbest_hyps_id = getattr(self, 'dec_' + dir).ctc.greedy(eouts, elens)
            else:
                nbest_hyps_id = getattr(self, 'dec_' + dir).ctc.beam_search(eouts, elens, params, idx2token, lm, lm_second, lm_second_bwd, 1, refs_id, utt_ids, speakers)
            return nbest_hyps_id, None
        elif params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
            best_hyps_id, aws = getattr(self, 'dec_' + dir).greedy(eouts, elens, params['recog_max_len_ratio'], idx2token, exclude_eos, refs_id, utt_ids, speakers)
            nbest_hyps_id = [[hyp] for hyp in best_hyps_id]
        else:
            assert params['recog_batch_size'] == 1
            scores_ctc = None
            if params['recog_ctc_weight'] > 0:
                scores_ctc = self.dec_fwd.ctc.scores(eouts)
            if params['recog_fwd_bwd_attention']:
                lm = getattr(self, 'lm_fwd', None)
                lm_bwd = getattr(self, 'lm_bwd', None)
                nbest_hyps_id_fwd, aws_fwd, scores_fwd = self.dec_fwd.beam_search(eouts, elens, params, idx2token, lm, None, lm_bwd, scores_ctc, params['recog_beam_width'], False, refs_id, utt_ids, speakers)
                nbest_hyps_id_bwd, aws_bwd, scores_bwd, _ = self.dec_bwd.beam_search(eouts, elens, params, idx2token, lm_bwd, None, lm, scores_ctc, params['recog_beam_width'], False, refs_id, utt_ids, speakers)
                best_hyps_id = fwd_bwd_attention(nbest_hyps_id_fwd, aws_fwd, scores_fwd, nbest_hyps_id_bwd, aws_bwd, scores_bwd, self.eos, params['recog_gnmt_decoding'], params['recog_length_penalty'], idx2token, refs_id)
                nbest_hyps_id = [[hyp] for hyp in best_hyps_id]
                aws = None
            else:
                ensmbl_eouts, ensmbl_elens, ensmbl_decs = [], [], []
                if len(ensemble_models) > 0:
                    for i_e, model in enumerate(ensemble_models):
                        enc_outs_e = model.encode(xs, task)
                        ensmbl_eouts += [enc_outs_e[task]['xs']]
                        ensmbl_elens += [enc_outs_e[task]['xlens']]
                        ensmbl_decs += [getattr(model, 'dec_' + dir)]
                lm = getattr(self, 'lm_' + dir, None)
                lm_second = getattr(self, 'lm_second', None)
                lm_bwd = getattr(self, 'lm_bwd' if dir == 'fwd' else 'lm_bwd', None)
                nbest_hyps_id, aws, scores = getattr(self, 'dec_' + dir).beam_search(eouts, elens, params, idx2token, lm, lm_second, lm_bwd, scores_ctc, params['recog_beam_width'], exclude_eos, refs_id, utt_ids, speakers, ensmbl_eouts, ensmbl_elens, ensmbl_decs)
        return nbest_hyps_id, aws


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CausalConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConformerConvBlock,
     lambda: ([], {'d_model': 4, 'kernel_size': 3, 'param_init': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GMMAttention,
     lambda: ([], {'kdim': 4, 'qdim': 4, 'adim': 4, 'n_mixtures': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm2D,
     lambda: ([], {'channel': 4, 'idim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearGLUBlock,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttentionMechanism,
     lambda: ([], {'kdim': 4, 'qdim': 4, 'adim': 4, 'odim': 4, 'n_heads': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (NiN,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (XLPositionalEmbedding,
     lambda: ([], {'d_model': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
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

