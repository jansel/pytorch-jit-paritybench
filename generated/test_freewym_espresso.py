import sys
_module = sys.modules[__name__]
del sys
conf = _module
espresso = _module
criterions = _module
cross_entropy_v2 = _module
label_smoothed_cross_entropy_v2 = _module
lf_mmi_loss = _module
subsampled_cross_entropy_with_accuracy = _module
data = _module
asr_chain_dataset = _module
asr_dataset = _module
asr_dictionary = _module
asr_xent_dataset = _module
encoders = _module
characters_asr = _module
feat_text_dataset = _module
dump_posteriors = _module
models = _module
external_language_model = _module
lstm_lm = _module
speech_fconv = _module
speech_lstm = _module
speech_lstm_encoder_model = _module
speech_tdnn = _module
speech_transformer = _module
tensorized_lookahead_language_model = _module
modules = _module
speech_attention = _module
optim = _module
lr_scheduler = _module
reduce_lr_on_plateau_v2 = _module
speech_recognize = _module
speech_train = _module
tasks = _module
language_modeling_for_asr = _module
speech_recognition = _module
speech_recognition_hybrid = _module
tools = _module
asr_prep_json = _module
compute_wer = _module
estimate_initial_state_prior_from_alignments = _module
generate_log_probs_for_decoding = _module
lexical_prefix_tree = _module
scheduled_sampling_rate_scheduler = _module
simple_greedy_decoder = _module
specaug_interpolate = _module
tensorized_prefix_tree = _module
text2token = _module
text2vocabulary = _module
utils = _module
wer = _module
eval_lm = _module
examples = _module
prepare_ctm = _module
deduplicate_lines = _module
extract_bt_data = _module
get_bitext = _module
gru_transformer = _module
detok = _module
noisychannel = _module
rerank = _module
rerank_generate = _module
rerank_options = _module
rerank_score_bw = _module
rerank_score_lm = _module
rerank_tune = _module
rerank_utils = _module
paraphrase = _module
commonsense_qa = _module
commonsense_qa_task = _module
multiprocessing_bpe_encoder = _module
preprocess_RACE = _module
wsc = _module
wsc_criterion = _module
wsc_task = _module
wsc_utils = _module
simultaneous_translation = _module
label_smoothed_cross_entropy_latency_augmented = _module
eval = _module
agents = _module
agent = _module
simul_trans_agent = _module
simul_trans_text_agent = _module
word_splitter = _module
client = _module
eval_latency = _module
evaluate = _module
scorers = _module
scorer = _module
text_scorer = _module
server = _module
transformer_monotonic_attention = _module
monotonic_multihead_attention = _module
monotonic_transformer_layer = _module
functions = _module
latency = _module
ASG_loss = _module
CTC_loss = _module
cross_entropy_acc = _module
collaters = _module
data_utils = _module
replabels = _module
infer = _module
vggtransformer = _module
w2l_conv_glu_enc = _module
wer_utils = _module
w2l_decoder = _module
score = _module
src = _module
logsumexp_moe = _module
mean_pool_gating_network = _module
translation_moe = _module
wav2vec_featurize = _module
wav2vec_manifest = _module
fairseq = _module
benchmark = _module
dummy_lm = _module
dummy_masked_lm = _module
dummy_model = _module
binarizer = _module
bleu = _module
checkpoint_utils = _module
adaptive_loss = _module
binary_cross_entropy = _module
composite_loss = _module
cross_entropy = _module
fairseq_criterion = _module
label_smoothed_cross_entropy = _module
label_smoothed_cross_entropy_with_alignment = _module
legacy_masked_lm = _module
masked_lm = _module
nat_loss = _module
sentence_prediction = _module
sentence_ranking = _module
append_token_dataset = _module
audio = _module
raw_audio_dataset = _module
backtranslation_dataset = _module
base_wrapper_dataset = _module
colorize_dataset = _module
concat_dataset = _module
concat_sentences_dataset = _module
denoising_dataset = _module
dictionary = _module
byte_bpe = _module
byte_utils = _module
bytes = _module
characters = _module
fastbpe = _module
gpt2_bpe = _module
gpt2_bpe_utils = _module
hf_bert_bpe = _module
hf_byte_bpe = _module
moses_tokenizer = _module
nltk_tokenizer = _module
sentencepiece_bpe = _module
space_tokenizer = _module
subword_nmt_bpe = _module
fairseq_dataset = _module
id_dataset = _module
indexed_dataset = _module
iterators = _module
language_pair_dataset = _module
legacy = _module
block_pair_dataset = _module
masked_lm_dataset = _module
masked_lm_dictionary = _module
list_dataset = _module
lm_context_window_dataset = _module
lru_cache_dataset = _module
mask_tokens_dataset = _module
monolingual_dataset = _module
multi_corpus_sampled_dataset = _module
nested_dictionary_dataset = _module
noising = _module
num_samples_dataset = _module
numel_dataset = _module
offset_tokens_dataset = _module
pad_dataset = _module
plasma_utils = _module
prepend_dataset = _module
prepend_token_dataset = _module
raw_label_dataset = _module
replace_dataset = _module
resampling_dataset = _module
roll_dataset = _module
round_robin_zip_datasets = _module
sort_dataset = _module
strip_token_dataset = _module
subsample_dataset = _module
token_block_dataset = _module
transform_eos_dataset = _module
transform_eos_lang_pair_dataset = _module
truncate_dataset = _module
distributed_utils = _module
file_io = _module
file_utils = _module
hub_utils = _module
incremental_decoding_utils = _module
iterative_refinement_generator = _module
legacy_distributed_data_parallel = _module
logging = _module
meters = _module
metrics = _module
progress_bar = _module
model_parallel = _module
vocab_parallel_cross_entropy = _module
megatron_trainer = _module
transformer = _module
transformer_lm = _module
multihead_attention = _module
transformer_layer = _module
bart = _module
hub_interface = _module
model = _module
composite_encoder = _module
distributed_fairseq_model = _module
fairseq_decoder = _module
fairseq_encoder = _module
fairseq_incremental_decoder = _module
fairseq_model = _module
fconv = _module
fconv_lm = _module
fconv_self_att = _module
huggingface = _module
hf_gpt2 = _module
lightconv = _module
lightconv_lm = _module
lstm = _module
masked_lm = _module
model_utils = _module
multilingual_transformer = _module
nat = _module
cmlm_transformer = _module
fairseq_nat_model = _module
insertion_transformer = _module
iterative_nonautoregressive_transformer = _module
levenshtein_transformer = _module
levenshtein_utils = _module
nat_crf_transformer = _module
nonautoregressive_ensembles = _module
nonautoregressive_transformer = _module
roberta = _module
alignment_utils = _module
hub_interface = _module
model = _module
model_camembert = _module
model_xlmr = _module
transformer = _module
transformer_align = _module
transformer_from_pretrained_xlm = _module
wav2vec = _module
adaptive_input = _module
adaptive_softmax = _module
beamable_mm = _module
character_token_embedder = _module
conv_tbc = _module
cross_entropy = _module
downsampled_multihead_attention = _module
dynamic_convolution = _module
dynamic_crf_layer = _module
dynamicconv_layer = _module
cuda_function_gen = _module
dynamicconv_layer = _module
setup = _module
fp32_group_norm = _module
gelu = _module
grad_multiply = _module
gumbel_vector_quantizer = _module
kmeans_vector_quantizer = _module
layer_drop = _module
layer_norm = _module
learned_positional_embedding = _module
lightconv_layer = _module
lightconv_layer = _module
lightweight_convolution = _module
linearized_convolution = _module
multihead_attention = _module
positional_embedding = _module
quant_noise = _module
quantization = _module
pq = _module
em = _module
qconv = _module
qemb = _module
qlinear = _module
utils = _module
quantization_options = _module
scalar = _module
qact = _module
qconv = _module
qemb = _module
qlinear = _module
ops = _module
utils = _module
scalar_bias = _module
sinusoidal_positional_embedding = _module
sparse_multihead_attention = _module
sparse_transformer_sentence_encoder = _module
sparse_transformer_sentence_encoder_layer = _module
transformer_layer = _module
transformer_sentence_encoder = _module
transformer_sentence_encoder_layer = _module
unfold = _module
vggblock = _module
nan_detector = _module
adadelta = _module
adafactor = _module
adagrad = _module
adam = _module
adamax = _module
bmuf = _module
fairseq_optimizer = _module
fp16_optimizer = _module
fused_adam = _module
fused_lamb = _module
cosine_lr_scheduler = _module
fairseq_lr_scheduler = _module
fixed_schedule = _module
inverse_square_root_schedule = _module
polynomial_decay_schedule = _module
reduce_lr_on_plateau = _module
tri_stage_lr_scheduler = _module
triangular_lr_scheduler = _module
nag = _module
sgd = _module
options = _module
pdb = _module
quantization_utils = _module
registry = _module
search = _module
sequence_generator = _module
sequence_scorer = _module
audio_pretraining = _module
cross_lingual_lm = _module
denoising = _module
fairseq_task = _module
language_modeling = _module
multilingual_denoising = _module
multilingual_masked_lm = _module
multilingual_translation = _module
semisupervised_translation = _module
translation = _module
translation_from_pretrained_bart = _module
translation_from_pretrained_xlm = _module
translation_lev = _module
tokenizer = _module
trainer = _module
utils = _module
fairseq_cli = _module
generate = _module
interactive = _module
preprocess = _module
train = _module
validate = _module
hubconf = _module
scripts = _module
average_checkpoints = _module
build_sym_alignment = _module
compare_namespaces = _module
count_docs = _module
read_binarized = _module
rm_pt = _module
shard_docs = _module
split_train_valid_docs = _module
spm_decode = _module
spm_encode = _module
spm_train = _module
tests = _module
test_asr_dataset = _module
test_speech_utils = _module
asr_test_base = _module
test_collaters = _module
test_cross_entropy = _module
test_vggtransformer = _module
test_average_checkpoints = _module
test_backtranslation_dataset = _module
test_binaries = _module
test_bmuf = _module
test_character_token_embedder = _module
test_concat_dataset = _module
test_convtbc = _module
test_dictionary = _module
test_export = _module
test_file_io = _module
test_iterators = _module
test_label_smoothing = _module
test_lstm_jitable = _module
test_memory_efficient_fp16 = _module
test_metrics = _module
test_multi_corpus_sampled_dataset = _module
test_multihead_attention = _module
test_noising = _module
test_reproducibility = _module
test_resampling_dataset = _module
test_sequence_generator = _module
test_sequence_scorer = _module
test_sparse_multihead_attention = _module
test_token_block_dataset = _module
test_train = _module
test_utils = _module
utils = _module

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


import logging


import numpy as np


import torch.nn.functional as F


import torch


from typing import List


from typing import Optional


import math


import torch.nn as nn


from typing import Dict


from torch import Tensor


from torch import nn


from torch.nn import Parameter


from itertools import groupby


from collections.abc import Iterable


import inspect


from typing import Any


from torch.nn.modules.loss import _Loss


import copy


from typing import Iterator


from typing import Tuple


import uuid


from torch.autograd import Variable


from typing import NamedTuple


import functools


from torch.nn.modules.utils import _single


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import re


import torch.distributed as dist


from torch.nn.modules.conv import _ConvNd


import torch.onnx.operators


import random


from itertools import repeat


from itertools import chain


import warnings


from collections import defaultdict


from itertools import accumulate


from typing import Callable


import collections


def Convolution2d(in_channels, out_channels, kernel_size, stride):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != 2:
            assert len(kernel_size) == 1
            kernel_size = kernel_size[0], kernel_size[0]
    else:
        assert isinstance(kernel_size, int)
        kernel_size = kernel_size, kernel_size
    if isinstance(stride, (list, tuple)):
        if len(stride) != 2:
            assert len(stride) == 1
            stride = stride[0], stride[0]
    else:
        assert isinstance(stride, int)
        stride = stride, stride
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    padding = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
        padding=padding)
    return m


class ConvBNReLU(nn.Module):
    """Sequence of convolution-BatchNorm-ReLU layers."""

    def __init__(self, out_channels, kernel_sizes, strides, in_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.in_channels = in_channels
        num_layers = len(out_channels)
        assert num_layers == len(kernel_sizes) and num_layers == len(strides)
        self.convolutions = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for i in range(num_layers):
            self.convolutions.append(Convolution2d(self.in_channels if i ==
                0 else self.out_channels[i - 1], self.out_channels[i], self
                .kernel_sizes[i], self.strides[i]))
            self.batchnorms.append(nn.BatchNorm2d(out_channels[i]))

    def output_lengths(self, in_lengths):
        out_lengths = in_lengths
        for stride in self.strides:
            if isinstance(stride, (list, tuple)):
                assert len(stride) > 0
                s = stride[0]
            else:
                assert isinstance(stride, int)
                s = stride
            out_lengths = (out_lengths + s - 1) // s
        return out_lengths

    def forward(self, src, src_lengths):
        x = src.view(src.size(0), src.size(1), self.in_channels, src.size(2
            ) // self.in_channels).transpose(1, 2)
        for conv, bn in zip(self.convolutions, self.batchnorms):
            x = F.relu(bn(conv(x)))
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x, x_lengths, padding_mask


class TdnnBNReLU(nn.Module):
    """A block of Tdnn-BatchNorm-ReLU layers."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.tdnn = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def output_lengths(self, in_lengths):
        out_lengths = (in_lengths + 2 * self.padding - self.dilation * (
            self.kernel_size - 1) + self.stride - 1) // self.stride
        return out_lengths

    def forward(self, src, src_lengths):
        x = src.transpose(1, 2).contiguous()
        x = F.relu(self.bn(self.tdnn(x)))
        x = x.transpose(2, 1).contiguous()
        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x, x_lengths, padding_mask


class BaseAttention(nn.Module):
    """Base class for attention layers."""

    def __init__(self, query_dim, value_dim, embed_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        pass

    def forward(self, query, value, key_padding_mask=None, state=None):
        raise NotImplementedError


def safe_cumprod(tensor, dim: int, eps: float=1e-10):
    """
    An implementation of cumprod to prevent precision issue.
    cumprod(x)
    = [x1, x1x2, x1x2x3, ....]
    = [exp(log(x1)), exp(log(x1) + log(x2)), exp(log(x1) + log(x2) + log(x3)), ...]
    = exp(cumsum(log(x)))
    """
    if (tensor + eps < 0).any().item():
        raise RuntimeError(
            'Safe cumprod can only take non-negative tensors as input.Consider use torch.cumprod if you want to calculate negative values.'
            )
    log_tensor = torch.log(tensor + eps)
    cumsum_log_tensor = torch.cumsum(log_tensor, dim)
    exp_cumsum_log_tensor = torch.exp(cumsum_log_tensor)
    return exp_cumsum_log_tensor


def exclusive_cumprod(tensor, dim: int, eps: float=1e-10):
    """
    Implementing exclusive cumprod.
    There is cumprod in pytorch, however there is no exclusive mode.
    cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
    exclusive means cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    tensor_size = list(tensor.size())
    tensor_size[dim] = 1
    return_tensor = safe_cumprod(torch.cat([torch.ones(tensor_size).type_as
        (tensor), tensor], dim=dim), dim=dim, eps=eps)
    if dim == 0:
        return return_tensor[:-1]
    elif dim == 1:
        return return_tensor[:, :-1]
    elif dim == 2:
        return return_tensor[:, :, :-1]
    else:
        raise RuntimeError('Cumprod on dimension 3 and more is not implemented'
            )


class FairseqIncrementalState(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) ->str:
        return '{}.{}'.format(self._incremental_state_id, key)

    def get_incremental_state(self, incremental_state: Optional[Dict[str,
        Dict[str, Optional[Tensor]]]], key: str) ->Optional[Dict[str,
        Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(self, incremental_state: Optional[Dict[str,
        Dict[str, Optional[Tensor]]]], key: str, value: Dict[str, Optional[
        Tensor]]) ->Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.
        __bases__ if b != FairseqIncrementalState)
    return cls


@with_incremental_state
class MonotonicAttention(nn.Module):
    """
    Abstract class of monotonic attentions
    """

    def __init__(self, args):
        self.eps = args.attention_eps
        self.mass_preservation = args.mass_preservation
        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var
        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = nn.Parameter(self.energy_bias_init * torch.ones([1])
            ) if args.energy_bias is True else 0

    @staticmethod
    def add_args(parser):
        parser.add_argument('--no-mass-preservation', action='store_false',
            dest='mass_preservation', help=
            'Do not stay on the last token when decoding')
        parser.add_argument('--mass-preservation', action='store_true',
            dest='mass_preservation', help=
            'Stay on the last token when decoding')
        parser.set_defaults(mass_preservation=True)
        parser.add_argument('--noise-var', type=float, default=1.0, help=
            'Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0, help=
            'Mean of discretness noise')
        parser.add_argument('--energy-bias', action='store_true', default=
            False, help='Bias for energy')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0,
            help='Initial value of the bias for energy')
        parser.add_argument('--attention-eps', type=float, default=1e-06,
            help='Epsilon when calculating expected attention')

    def p_choose(self, *args):
        raise NotImplementedError

    def input_projections(self, *args):
        raise NotImplementedError

    def attn_energy(self, q_proj, k_proj, key_padding_mask=None):
        """
        Calculating monotonic energies

        ============================================================
        Expected input size
        q_proj: bsz * num_heads, tgt_len, self.head_dim
        k_proj: bsz * num_heads, src_len, self.head_dim
        key_padding_mask: bsz, src_len
        attn_mask: tgt_len, src_len
        """
        bsz, tgt_len, embed_dim = q_proj.size()
        bsz = bsz // self.num_heads
        src_len = k_proj.size(1)
        attn_energy = torch.bmm(q_proj, k_proj.transpose(1, 2)
            ) + self.energy_bias
        attn_energy = attn_energy.view(bsz, self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None:
            attn_energy = attn_energy.masked_fill(key_padding_mask.
                unsqueeze(1).unsqueeze(2).bool(), float('-inf'))
        return attn_energy

    def expected_alignment_train(self, p_choose, key_padding_mask):
        """
        Calculating expected alignment for MMA
        Mask is not need because p_choose will be 0 if masked

        q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
        a_ij = p_ij q_ij

        parellel solution:
        ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        """
        bsz_num_heads, tgt_len, src_len = p_choose.size()
        cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=self.eps)
        cumprod_1mp_clamp = torch.clamp(cumprod_1mp, self.eps, 1.0)
        init_attention = p_choose.new_zeros([bsz_num_heads, 1, src_len])
        init_attention[:, :, (0)] = 1.0
        previous_attn = [init_attention]
        for i in range(tgt_len):
            alpha_i = (p_choose[:, (i)] * cumprod_1mp[:, (i)] * torch.
                cumsum(previous_attn[i][:, (0)] / cumprod_1mp_clamp[:, (i)],
                dim=1)).clamp(0, 1.0)
            previous_attn.append(alpha_i.unsqueeze(1))
        alpha = torch.cat(previous_attn[1:], dim=1)
        if self.mass_preservation:
            alpha[:, :, (-1)] = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0.0, 1.0
                )
        assert not torch.isnan(alpha).any(), 'NaN detected in alpha.'
        return alpha

    def expected_alignment_infer(self, p_choose, key_padding_mask,
        incremental_state):
        """
        Calculating mo alignment for MMA during inference time

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        key_padding_mask: bsz * src_len
        incremental_state: dict
        """
        bsz_num_heads, tgt_len, src_len = p_choose.size()
        assert tgt_len == 1
        p_choose = p_choose[:, (0), :]
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        bsz = bsz_num_heads // self.num_heads
        prev_monotonic_step = monotonic_cache.get('step', p_choose.
            new_zeros([bsz, self.num_heads]).long())
        bsz, num_heads = prev_monotonic_step.size()
        assert num_heads == self.num_heads
        assert bsz * num_heads == bsz_num_heads
        p_choose = p_choose.view(bsz, num_heads, src_len)
        if key_padding_mask is not None:
            src_lengths = src_len - key_padding_mask.sum(dim=1, keepdim=True
                ).long()
        else:
            src_lengths = prev_monotonic_step.new_ones(bsz, 1) * src_len
        src_lengths = src_lengths.expand_as(prev_monotonic_step)
        new_monotonic_step = prev_monotonic_step
        step_offset = 0
        if key_padding_mask is not None:
            if key_padding_mask[:, (0)].any():
                step_offset = key_padding_mask.sum(dim=-1, keepdim=True)
        max_steps = src_lengths - 1 if self.mass_preservation else src_lengths
        finish_read = new_monotonic_step.eq(max_steps)
        while finish_read.sum().item() < bsz * self.num_heads:
            p_choose_i = p_choose.gather(2, (step_offset +
                new_monotonic_step).unsqueeze(2).clamp(0, src_len - 1)
                ).squeeze(2)
            action = (p_choose_i < 0.5).type_as(prev_monotonic_step
                ).masked_fill(finish_read, 0)
            new_monotonic_step += action
            finish_read = new_monotonic_step.eq(max_steps) | (action == 0)
        monotonic_cache['step'] = new_monotonic_step
        alpha = p_choose.new_zeros([bsz * self.num_heads, src_len]).scatter(
            1, (step_offset + new_monotonic_step).view(bsz * self.num_heads,
            1).clamp(0, src_len - 1), 1)
        if not self.mass_preservation:
            alpha = alpha.masked_fill((new_monotonic_step == max_steps).
                view(bsz * self.num_heads, 1), 0)
        alpha = alpha.unsqueeze(1)
        self._set_monotonic_buffer(incremental_state, monotonic_cache)
        return alpha

    def v_proj_output(self, value):
        raise NotImplementedError

    def forward(self, query, key, value, key_padding_mask=None,
        incremental_state=None, *args, **kwargs):
        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)
        p_choose = self.p_choose(query, key, key_padding_mask)
        if incremental_state is not None:
            alpha = self.expected_alignment_infer(p_choose,
                key_padding_mask, incremental_state)
        else:
            alpha = self.expected_alignment_train(p_choose, key_padding_mask)
        beta = self.expected_attention(alpha, query, key, value,
            key_padding_mask, incremental_state)
        attn_weights = beta
        v_proj = self.v_proj_output(value)
        attn = torch.bmm(attn_weights.type_as(v_proj), v_proj)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)
        return attn, {'alpha': alpha, 'beta': beta, 'p_choose': p_choose}

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(incremental_state, new_order)
        input_buffer = self._get_monotonic_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_monotonic_buffer(incremental_state, input_buffer)

    def _get_monotonic_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'monotonic'
            ) or {}

    def _set_monotonic_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(self, incremental_state, 'monotonic',
            buffer)

    def get_pointer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'monotonic'
            ) or {}

    def get_fastest_pointer(self, incremental_state):
        return self.get_pointer(incremental_state)['step'].max(0)[0]

    def set_pointer(self, incremental_state, p_choose):
        curr_pointer = self.get_pointer(incremental_state)
        if len(curr_pointer) == 0:
            buffer = torch.zeros_like(p_choose)
        else:
            buffer = self.get_pointer(incremental_state)['step']
        buffer += (p_choose < 0.5).type_as(buffer)
        utils.set_incremental_state(self, incremental_state, 'monotonic', {
            'step': buffer})


class MeanPoolGatingNetwork(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.

    This module applies mean pooling over an encoder's output and returns
    reponsibilities for each expert. The encoder format is expected to match
    :class:`fairseq.models.transformer.TransformerEncoder`.
    """

    def __init__(self, embed_dim, num_experts, dropout=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout
            ) if dropout is not None else None
        self.fc2 = torch.nn.Linear(embed_dim, num_experts)

    def forward(self, encoder_out):
        if not (hasattr(encoder_out, 'encoder_out') and hasattr(encoder_out,
            'encoder_padding_mask') and encoder_out.encoder_out.size(2) ==
            self.embed_dim):
            raise ValueError('Unexpected format for encoder_out')
        encoder_padding_mask = encoder_out.encoder_padding_mask
        encoder_out = encoder_out.encoder_out.transpose(0, 1)
        if encoder_padding_mask is not None:
            encoder_out = encoder_out.clone()
            encoder_out[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(encoder_out, dim=1) / ntokens.type_as(encoder_out)
        else:
            x = torch.mean(encoder_out, dim=1)
        x = torch.tanh(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)


ARCH_CONFIG_REGISTRY = {}


ARCH_MODEL_INV_REGISTRY = {}


MODEL_REGISTRY = {}


ARCH_MODEL_REGISTRY = {}


def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
        def lstm_luong_wmt_en_de(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                'Cannot register model architecture for unknown model type ({})'
                .format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                'Cannot register duplicate model architecture ({})'.format(
                arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable ({})'.
                format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn
    return register_model_arch_fn


class FairseqCriterion(_Loss):

    def __init__(self, task):
        super().__init__()
        self.task = task
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (p.kind == p.POSITIONAL_ONLY or p.kind == p.VAR_POSITIONAL or
                p.kind == p.VAR_KEYWORD):
                raise NotImplementedError('{} not supported'.format(p.kind))
            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
            if p.name == 'task':
                init_args['task'] = task
            elif hasattr(args, p.name):
                init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass
            else:
                raise NotImplementedError(
                    'Unable to infer Criterion arguments, please implement {}.build_criterion'
                    .format(cls.__name__))
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs: List[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'The aggregate_logging_outputs API is deprecated. Please use the reduce_metrics API instead.'
            )
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) ->None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'Criterions should implement the reduce_metrics API. Falling back to deprecated aggregate_logging_outputs API.'
            )
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


logger = logging.getLogger(__name__)


class GeneratorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, args, task, models):
        super().__init__()
        self.args = args
        self.task = task
        self.models = nn.ModuleList(models)
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary
        for model in self.models:
            model.make_generation_fast_(beamable_mm_beam_size=None if
                getattr(args, 'no_beamable_mm', False) else getattr(args,
                'beam', 5), need_attn=getattr(args, 'print_alignment', False))
        self.align_dict = utils.load_align_dict(getattr(args, 'replace_unk',
            None))
        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)
        self.max_positions = utils.resolve_max_positions(self.task.
            max_positions(), *[model.max_positions() for model in models])
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch
            .float))

    @property
    def device(self):
        return self._float_tensor.device

    def translate(self, sentences: List[str], beam: int=5, verbose: bool=
        False, **kwargs) ->List[str]:
        return self.sample(sentences, beam, verbose, **kwargs)

    def sample(self, sentences: List[str], beam: int=1, verbose: bool=False,
        **kwargs) ->List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **
                kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose,
            **kwargs)
        return [self.decode(hypos[0]['tokens']) for hypos in batched_hypos]

    def score(self, sentences: List[str], **kwargs):
        if isinstance(sentences, str):
            return self.score([sentences], **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        return [hypos[0] for hypos in self.generate(tokenized_sentences,
            score_reference=True, **kwargs)]

    def generate(self, tokenized_sentences: List[torch.LongTensor], beam:
        int=5, verbose: bool=False, skip_invalid_size_inputs=False,
        inference_step_args=None, **kwargs) ->List[List[Dict[str, torch.
        Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim(
            ) == 1:
            return self.generate(tokenized_sentences.unsqueeze(0), beam=
                beam, verbose=verbose, **kwargs)[0]
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)
        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences,
            skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(generator, self.models,
                batch, **inference_step_args)
            for id, hypos in zip(batch['id'].tolist(), translations):
                results.append((id, hypos))
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.args, name,
                    default))
            for source_tokens, target_hypotheses in zip(tokenized_sentences,
                outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info('S\t{}'.format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo['tokens'])
                    logger.info('H\t{}\t{}'.format(hypo['score'], hypo_str))
                    logger.info('P\t{}'.format(' '.join(map(lambda x:
                        '{:.4f}'.format(x), hypo['positional_scores'].
                        tolist()))))
                    if hypo['alignment'] is not None and getarg(
                        'print_alignment', False):
                        logger.info('A\t{}'.format(' '.join(map(lambda x:
                            str(utils.item(x)), hypo['alignment'].int().cpu
                            ()))))
        return outputs

    def encode(self, sentence: str) ->torch.LongTensor:
        sentence = self.tokenize(sentence)
        sentence = self.apply_bpe(sentence)
        return self.binarize(sentence)

    def decode(self, tokens: torch.LongTensor) ->str:
        sentence = self.string(tokens)
        sentence = self.remove_bpe(sentence)
        return self.detokenize(sentence)

    def tokenize(self, sentence: str) ->str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.encode(sentence)
        return sentence

    def detokenize(self, sentence: str) ->str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.decode(sentence)
        return sentence

    def apply_bpe(self, sentence: str) ->str:
        if self.bpe is not None:
            sentence = self.bpe.encode(sentence)
        return sentence

    def remove_bpe(self, sentence: str) ->str:
        if self.bpe is not None:
            sentence = self.bpe.decode(sentence)
        return sentence

    def binarize(self, sentence: str) ->torch.LongTensor:
        return self.src_dict.encode_line(sentence, add_if_not_exist=False
            ).long()

    def string(self, tokens: torch.LongTensor) ->str:
        return self.tgt_dict.string(tokens)

    def _build_batches(self, tokens: List[List[int]],
        skip_invalid_size_inputs: bool) ->Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(dataset=self.task.
            build_dataset_for_inference(tokens, lengths), max_tokens=self.
            args.max_tokens, max_sentences=self.args.max_sentences,
            max_positions=self.max_positions, ignore_invalid_inputs=
            skip_invalid_size_inputs).next_epoch_itr(shuffle=False)
        return batch_iterator


@with_incremental_state
class ModelParallelMultiheadAttention(nn.Module):
    """Model parallel Multi-headed attention.
    This performs the Multi-headed attention over multiple gpus.

    See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=
        0.0, bias=True, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        if not has_megatron_submodule:
            raise ImportError(
                """

Please install the megatron submodule:

  git submodule update --init fairseq/model_parallel/megatron"""
                )
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.model_parallel_size = get_model_parallel_world_size()
        self.num_heads_partition = num_heads // self.model_parallel_size
        assert self.num_heads_partition * self.model_parallel_size == num_heads, 'Number of heads must be divisble by model parallel size'
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        self.k_proj = ColumnParallelLinear(self.kdim, embed_dim, bias=bias,
            gather_output=False)
        self.v_proj = ColumnParallelLinear(self.vdim, embed_dim, bias=bias,
            gather_output=False)
        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=bias,
            gather_output=False)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=bias,
            input_is_parallel=True)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
        key_padding_mask: Optional[Tensor]=None, incremental_state:
        Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None, static_kv:
        bool=False, attn_mask: Optional[Tensor]=None, **unused_kwargs) ->Tuple[
        Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads_partition,
            self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads_partition,
                self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads_partition,
                self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads_partition, -
                    1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.
                    num_heads_partition, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = (ModelParallelMultiheadAttention.
                _append_prev_key_padding_mask(key_padding_mask=
                key_padding_mask, prev_key_padding_mask=
                prev_key_padding_mask, batch_size=bsz, src_len=k.size(1),
                static_kv=static_kv))
            saved_state['prev_key'] = k.view(bsz, self.num_heads_partition,
                -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.
                num_heads_partition, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state,
                saved_state)
        assert k is not None
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads_partition,
            tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads_partition,
                tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.
                unsqueeze(1).unsqueeze(2).to(torch.bool), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads_partition,
                tgt_len, src_len)
        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        with get_cuda_rng_tracker().fork():
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights),
                p=self.dropout, training=self.training)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads_partition,
            tgt_len, self.head_dim]
        embed_dim_partition = embed_dim // self.model_parallel_size
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
            embed_dim_partition)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor], batch_size: int, src_len:
        int, static_kv: bool) ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(),
                key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len -
                prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(),
                filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1)
                )
            if key_padding_mask.is_cuda:
                filler = filler
            new_key_padding_mask = torch.cat([filler.float(),
                key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[
        str, Optional[Tensor]]], new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order
                        )
            incremental_state = self._set_input_buffer(incremental_state,
                input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[
        str, Optional[Tensor]]]]) ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str,
        Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, 'attn_state',
            buffer)


class BARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        self.max_positions = min(utils.resolve_max_positions(self.task.
            max_positions(), self.model.max_positions()))
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch
            .float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences, no_separator=True
        ) ->torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += ' </s>' if not no_separator else ''
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence,
            append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for
            s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        dataset = self.task.build_dataset_for_inference(src_tokens, [x.
            numel() for x in src_tokens])
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device
            ), sample)
        return sample

    def sample(self, sentences: List[str], beam: int=1, verbose: bool=False,
        **kwargs) ->str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate(input, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def generate(self, tokens: List[torch.LongTensor], beam: int=5, verbose:
        bool=False, **kwargs) ->torch.LongTensor:
        sample = self._build_sample(tokens)
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(generator, [self.model],
            sample, prefix_tokens=sample['net_input']['src_tokens'].
            new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.
            bos()))
        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens:
        bool=False) ->torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.
                format(tokens.size(-1), self.model.max_positions()))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()
        prev_output_tokens[:, (0)] = tokens.gather(1, (tokens.ne(self.task.
            source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1)).squeeze()
        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(src_tokens=tokens, src_lengths=None,
            prev_output_tokens=prev_output_tokens, features_only=True,
            return_all_hiddens=return_all_hiddens)
        if return_all_hiddens:
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states
                ]
        else:
            return features

    def register_classification_head(self, name: str, num_classes: int=None,
        embedding_size: int=None, **kwargs):
        self.model.register_classification_head(name, num_classes=
            num_classes, embedding_size=embedding_size, **kwargs)

    def predict(self, head: str, tokens: torch.LongTensor, return_logits:
        bool=False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[(tokens.eq(self.task.
            source_dictionary.eos())), :].view(features.size(0), -1,
            features.size(-1))[:, (-1), :]
        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn,
        pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=
            encoder_out, **kwargs)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(self, net_output: Tuple[Tensor, Optional[Dict[
        str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[
        Dict[str, Tensor]]=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'adaptive_softmax'
            ) and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=
                target)
            return out.exp_() if not log_probs else out
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace
                )
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1000000.0

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


class FairseqEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(src_tokens=net_input['src_tokens'],
                src_lengths=net_input['src_lengths'])
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {k: v for k, v in net_input.items() if k !=
            'prev_output_tokens'}
        return self.forward(**encoder_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1000000.0

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


def prune_state_dict(state_dict, args):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    if not args or args.arch == 'ptt_transformer':
        return state_dict
    encoder_layers_to_keep = (args.encoder_layers_to_keep if 
        'encoder_layers_to_keep' in vars(args) else None)
    decoder_layers_to_keep = (args.decoder_layers_to_keep if 
        'decoder_layers_to_keep' in vars(args) else None)
    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict
    logger.info(
        'Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop'
        )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted([int(layer_string) for layer_string in
            layers_to_keep.split(',')])
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)
        regex = re.compile('^{layer}.*\\.layers\\.(\\d+)'.format(layer=
            layer_name))
        return {'substitution_regex': regex, 'mapping_dict': mapping_dict}
    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep,
            'encoder'))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep,
            'decoder'))
    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search('\\.layers\\.(\\d+)\\.', layer_name)
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue
        original_layer_number = match.group(1)
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass['mapping_dict'
                ] and pruning_pass['substitution_regex'].search(layer_name):
                new_layer_number = pruning_pass['mapping_dict'][
                    original_layer_number]
                substitution_match = pruning_pass['substitution_regex'].search(
                    layer_name)
                new_state_key = layer_name[:substitution_match.start(1)
                    ] + new_layer_number + layer_name[substitution_match.
                    end(1):]
                new_state_dict[new_state_key] = state_dict[layer_name]
    if 'encoder_layers_to_keep' in vars(args):
        args.encoder_layers_to_keep = None
    if 'decoder_layers_to_keep' in vars(args):
        args.decoder_layers_to_keep = None
    return new_state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class AttentionLayer(nn.Module):

    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        self.out_projection = Linear(embed_dim, conv_channels)
        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])
        if encoder_padding_mask is not None:
            x = x.float().masked_fill(encoder_padding_mask.unsqueeze(1),
                float('-inf')).type_as(x)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x
        x = self.bmm(x, encoder_out[1])
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(dim=1, keepdim=True)
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, export=
    False):
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class SelfAttention(nn.Module):

    def __init__(self, out_channels, embed_dim, num_heads, project_input=
        False, gated=False, downsample=False):
        super().__init__()
        self.attention = DownsampledMultiHeadAttention(out_channels,
            embed_dim, num_heads, dropout=0, bias=True, project_input=
            project_input, gated=gated, downsample=downsample)
        self.in_proj_q = Linear(out_channels, embed_dim)
        self.in_proj_k = Linear(out_channels, embed_dim)
        self.in_proj_v = Linear(out_channels, embed_dim)
        self.ln = LayerNorm(out_channels)

    def forward(self, x):
        residual = x
        query = self.in_proj_q(x)
        key = self.in_proj_k(x)
        value = self.in_proj_v(x)
        x, _ = self.attention(query, key, value, mask_future_timesteps=True,
            use_scalar_bias=True)
        return self.ln(x + residual)


def DynamicConv(input_size, kernel_size=1, padding_l=None, num_heads=1,
    weight_dropout=0.0, weight_softmax=False, renorm_padding=False, bias=
    False, conv_bias=False, query_size=None, in_proj=False):
    if torch.cuda.is_available():
        try:
            from fairseq.modules.dynamicconv_layer import DynamicconvLayer
            return DynamicconvLayer(input_size, kernel_size=kernel_size,
                padding_l=padding_l, num_heads=num_heads, weight_dropout=
                weight_dropout, weight_softmax=weight_softmax, bias=bias)
        except ImportError as e:
            print(e)
    return DynamicConv1dTBC(input_size, kernel_size=kernel_size, padding_l=
        padding_l, num_heads=num_heads, weight_dropout=weight_dropout,
        weight_softmax=weight_softmax, bias=bias)


def LightweightConv(input_size, kernel_size=1, padding_l=None, num_heads=1,
    weight_dropout=0.0, weight_softmax=False, bias=False):
    if torch.cuda.is_available():
        try:
            from fairseq.modules.lightconv_layer import LightconvLayer
            return LightconvLayer(input_size, kernel_size=kernel_size,
                padding_l=padding_l, num_heads=num_heads, weight_dropout=
                weight_dropout, weight_softmax=weight_softmax, bias=bias)
        except ImportError as e:
            print(e)
    return LightweightConv1dTBC(input_size, kernel_size=kernel_size,
        padding_l=padding_l, num_heads=num_heads, weight_dropout=
        weight_dropout, weight_softmax=weight_softmax, bias=bias)


class LightConvEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, kernel_size=0):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.conv_dim = args.encoder_conv_dim
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((
            kernel_size - 1) // 2, kernel_size // 2)
        if args.encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.encoder_conv_type == 'lightweight':
            self.conv = LightweightConv(self.conv_dim, kernel_size,
                padding_l=padding_l, weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads, weight_dropout=args
                .weight_dropout)
        elif args.encoder_conv_type == 'dynamic':
            self.conv = DynamicConv(self.conv_dim, kernel_size, padding_l=
                padding_l, weight_softmax=args.weight_softmax, num_heads=
                args.encoder_attention_heads, weight_dropout=args.
                weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.input_dropout = args.input_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in
            range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).
                unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def extra_repr(self):
        return (
            'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'
            .format(self.dropout, self.relu_dropout, self.input_dropout,
            self.normalize_before))


class LightConvDecoderLayer(nn.Module):
    """Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, no_encoder_attn=False, kernel_size=0):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.conv_dim = args.decoder_conv_dim
        if args.decoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.decoder_conv_type == 'lightweight':
            self.conv = LightweightConv(self.conv_dim, kernel_size,
                padding_l=kernel_size - 1, weight_softmax=args.
                weight_softmax, num_heads=args.decoder_attention_heads,
                weight_dropout=args.weight_dropout)
        elif args.decoder_conv_type == 'dynamic':
            self.conv = DynamicConv(self.conv_dim, kernel_size, padding_l=
                kernel_size - 1, weight_softmax=args.weight_softmax,
                num_heads=args.decoder_attention_heads, weight_dropout=args
                .weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.input_dropout = args.input_dropout
        self.normalize_before = args.decoder_normalize_before
        self.conv_layer_norm = LayerNorm(self.embed_dim)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(self.embed_dim, args.
                decoder_attention_heads, dropout=args.attention_dropout,
                encoder_decoder_attention=True)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, x, encoder_out, encoder_padding_mask,
        incremental_state, prev_conv_state=None, prev_attn_state=None,
        conv_mask=None, conv_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
        if prev_conv_state is not None:
            if incremental_state is None:
                incremental_state = {}
            self.conv._set_input_buffer(incremental_state, prev_conv_state)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.conv(x, incremental_state=incremental_state)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)
        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x,
                before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
                self.encoder_attn._set_input_buffer(incremental_state,
                    saved_state)
            x, attn = self.encoder_attn(query=x, key=encoder_out, value=
                encoder_out, key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state, static_kv=True,
                need_weights=not self.training and self.need_attn)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x,
                after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return (
            'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'
            .format(self.dropout, self.relu_dropout, self.input_dropout,
            self.normalize_before))


class AttentionLayer(nn.Module):

    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim,
        bias=False):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim,
            output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        x = self.input_proj(input)
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(encoder_padding_mask
                , float('-inf')).type_as(attn_scores)
        attn_scores = F.softmax(attn_scores, dim=0)
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class _EnsembleModelEncoder(object):

    def __init__(self, models):
        self.models = models

    def reorder_encoder_out(self, encoder_outs, new_order):
        encoder_outs = [model.encoder.reorder_encoder_out(encoder_out,
            new_order) for model, encoder_out in zip(self.models, encoder_outs)
            ]
        return encoder_outs


class BasicEnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.bos = self.models[0].decoder.dictionary.bos()
        self.eos = self.models[0].decoder.dictionary.eos()
        self.pad = self.models[0].decoder.dictionary.pad()
        self.unk = self.models[0].decoder.dictionary.unk()
        self.encoder = _EnsembleModelEncoder(self.models)

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.forward_encoder(encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, *inputs):
        raise NotImplementedError

    def initialize_output_tokens(self, *inputs):
        raise NotImplementedError


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[(masked_tokens), :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn,
        pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(nn.Linear(inner_dim, num_classes
            ), q_noise, qn_block_size)

    def forward(self, features, **kwargs):
        x = features[:, (0), :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(TransposeLast(), Fp32LayerNorm(dim,
            elementwise_affine=affine), TransposeLast())
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)
    return mod


class ConvFeatureExtractionModel(nn.Module):

    def __init__(self, conv_layers, dropout, log_compression,
        skip_connections, residual_scale, non_affine_group_norm, activation):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(nn.Conv1d(n_in, n_out, k, stride=stride,
                bias=False), nn.Dropout(p=dropout), norm_block(
                is_layer_norm=False, dim=n_out, affine=not
                non_affine_group_norm), activation)
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[(...), ::r_tsz // tsz][(...), :tsz]
                x = (x + residual) * self.residual_scale
        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()
        return x


class ZeroPad1d(nn.Module):

    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class ConvAggegator(nn.Module):

    def __init__(self, conv_layers, embed, dropout, skip_connections,
        residual_scale, non_affine_group_norm, conv_bias, zero_pad, activation
        ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka
            pad = ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((
                ka + kb, 0))
            return nn.Sequential(pad, nn.Conv1d(n_in, n_out, k, stride=
                stride, bias=conv_bias), nn.Dropout(p=dropout), norm_block(
                False, n_out, affine=not non_affine_group_norm), activation)
        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Wav2VecPredictionsModel(nn.Module):

    def __init__(self, in_dim, out_dim, prediction_steps, n_negatives,
        cross_sample_negatives, sample_distance, dropout, offset,
        balanced_classes, infonce):
        super().__init__()
        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(in_dim, out_dim, (1,
            prediction_steps))
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape
        y = y.transpose(0, 1)
        y = y.contiguous().view(fsz, -1)
        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.
            sample_distance)
        assert high > 1
        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.
            n_negatives * tsz))
        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = buffered_arange(tsz).unsqueeze(-1).expand(-1, self.
                    n_negatives).flatten()
                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, 
                    self.n_negatives * tsz))
                neg_idxs[neg_idxs >= tszs] += 1
            if self.cross_sample_negatives > 0:
                tszs = buffered_arange(tsz).unsqueeze(-1).expand(-1, self.
                    cross_sample_negatives).flatten()
                cross_neg_idxs = torch.randint(low=0, high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz))
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs
        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(fsz, bsz, self.n_negatives + self.
            cross_sample_negatives, tsz).permute(2, 1, 0, 3)
        return negs

    def forward(self, x, y):
        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)
        x = self.dropout(x)
        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)
        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)
        predictions = x.new(bsz * copies * (tsz - self.offset + 1) * steps -
            (steps + 1) * steps // 2 * copies * bsz)
        if self.infonce:
            labels = predictions.new_full((predictions.shape[0] // copies,),
                0, dtype=torch.long)
        else:
            labels = torch.zeros_like(predictions)
        weights = torch.full_like(labels, 1 / self.n_negatives
            ) if self.balanced_classes and not self.infonce else None
        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum('bct,nbct->tbn', x[(
                    ...), :-offset, (i)], targets[(...), offset:]).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum('bct,nbct->nbt', x[(
                    ...), :-offset, (i)], targets[(...), offset:]).flatten()
                labels[start:start + pos_num] = 1.0
                if weights is not None:
                    weights[start:start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), '{} != {}'.format(end,
            predictions.numel())
        if self.infonce:
            predictions = predictions.view(-1, copies)
        elif weights is not None:
            labels = labels, weights
        return predictions, labels


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """
    if p <= 0:
        return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv:
        assert module.weight.size(1
            ) % block_size == 0, 'Input features must be a multiple of block sizes'
    elif module.kernel_size == (1, 1):
        assert module.in_channels % block_size == 0, 'Input channels must be a multiple of block sizes'
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        assert k % block_size == 0, 'Kernel size must be a multiple of block size'

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features,
                    device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1,
                    in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size *
                        out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1,
                        in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1),
                        device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.
                        kernel_size[0], mod.kernel_size[1])
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class AdaptiveInput(nn.Module):

    def __init__(self, vocab_size: int, padding_idx: int, initial_dim: int,
        factor: float, output_dim: int, cutoff: List[int], q_noise: float=0,
        qn_block_size: int=8):
        super().__init__()
        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[-1
                ], 'cannot specify cutoff larger than vocab size'
        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx
        self.embeddings = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // factor ** i)
            seq = nn.Sequential(nn.Embedding(size, dim, self.padding_idx),
                quant_noise(nn.Linear(dim, output_dim, bias=False), q_noise,
                qn_block_size))
            self.embeddings.append(seq)
            self.padding_idx = None
        self.padding_idx = padding_idx

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5
                    )
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input: torch.Tensor):
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)
        return result


class TiedLinear(nn.Module):

    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.
            weight)


class TiedHeadModule(nn.Module):

    def __init__(self, weights, input_dim, num_classes, q_noise, qn_block_size
        ):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()
        self.word_proj = quant_noise(TiedLinear(tied_emb, transpose=False),
            q_noise, qn_block_size)
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(quant_noise(nn.Linear(input_dim,
                emb_dim, bias=False), q_noise, qn_block_size), self.word_proj)
        self.class_proj = quant_noise(nn.Linear(input_dim, num_classes,
            bias=False), q_noise, qn_block_size)
        self.out_dim = self.num_words + num_classes
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, input):
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, :self.num_words] = self.word_proj(input.view(inp_sz, -1))
        out[:, self.num_words:] = self.class_proj(input.view(inp_sz, -1))
        return out


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, vocab_size, input_dim, cutoff, dropout, factor=4.0,
        adaptive_inputs=None, tie_proj=False, q_noise=0, qn_block_size=8):
        super().__init__()
        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[-1
                ], 'cannot specify cutoff larger than vocab size'
        output_dim = cutoff[0] + len(cutoff) - 1
        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim
        self.factor = factor
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.lsm = nn.LogSoftmax(dim=1)
        if adaptive_inputs is not None:
            self.head = TiedHeadModule(adaptive_inputs.weights_for_band(0),
                input_dim, len(cutoff) - 1, self.q_noise, self.qn_block_size)
        else:
            self.head = quant_noise(nn.Linear(input_dim, output_dim, bias=
                False), self.q_noise, self.qn_block_size)
        self._make_tail(adaptive_inputs, tie_proj)

        def init_weights(m):
            if hasattr(m, 'weight') and not isinstance(m, TiedLinear
                ) and not isinstance(m, TiedHeadModule):
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)
        self.register_buffer('version', torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))
            tied_emb, tied_proj = adaptive_inputs.weights_for_band(i + 1
                ) if adaptive_inputs is not None else (None, None)
            if tied_proj is not None:
                if tie_proj:
                    proj = quant_noise(TiedLinear(tied_proj, transpose=True
                        ), self.q_noise, self.qn_block_size)
                else:
                    proj = quant_noise(nn.Linear(tied_proj.size(0),
                        tied_proj.size(1), bias=False), self.q_noise, self.
                        qn_block_size)
            else:
                proj = quant_noise(nn.Linear(self.input_dim, dim, bias=
                    False), self.q_noise, self.qn_block_size)
            if tied_emb is None:
                out_proj = nn.Linear(dim, self.cutoff[i + 1] - self.cutoff[
                    i], bias=False)
            else:
                out_proj = TiedLinear(tied_emb, transpose=False)
            m = nn.Sequential(proj, nn.Dropout(self.dropout), quant_noise(
                out_proj, self.q_noise, self.qn_block_size))
            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + '.version'
        if version_name not in state_dict:
            raise Exception('This version of the model is no longer supported')

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """
        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i
            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)
        return new_target, target_idxs

    def forward(self, input, target):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """
        input = input.contiguous().view(-1, input.size(-1))
        input = F.dropout(input, p=self.dropout, training=self.training)
        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]
        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](input.index_select(0,
                    target_idxs[i])))
            else:
                output.append(None)
        return output, new_target

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """
        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)
        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None
        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)
        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0]:head_sz].clone()
        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]
            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(tail_priors
                    [:, (i), (None)])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[(idxs), start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[(idxs), start:end] = self.lsm(tail_out).add_(
                    tail_priors[idxs, i, None])
        log_probs = log_probs.view(bsz, length, -1)
        return log_probs


class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """

    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if not self.training and self.beam_size is not None and input1.dim(
            ) == 3 and input1.size(1) == 1:
            bsz, beam = input1.size(0), self.beam_size
            input1 = input1[:, (0), :].unfold(0, beam, beam).transpose(2, 1)
            input2 = input2.unfold(0, beam, beam)[:, :, :, (0)]
            if input1.size(0) == 1:
                output = torch.mm(input1[(0), :, :], input2[(0), :, :])
            else:
                output = input1.bmm(input2)
            return output.view(bsz, 1, -1)
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size


SPACE_NORMALIZER = re.compile('\\s+')


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(' ', line)
    line = line.strip()
    return line.split()


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(path: str, mode: str='r', buffering: int=-1, encoding:
        Optional[str]=None, errors: Optional[str]=None, newline: Optional[
        str]=None):
        if FVCorePathManager:
            return FVCorePathManager.open(path=path, mode=mode, buffering=
                buffering, encoding=encoding, errors=errors, newline=newline)
        return open(path, mode=mode, buffering=buffering, encoding=encoding,
            errors=errors, newline=newline)

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool=False) ->bool:
        if FVCorePathManager:
            return FVCorePathManager.copy(src_path=src_path, dst_path=
                dst_path, overwrite=overwrite)
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) ->str:
        if FVCorePathManager:
            return FVCorePathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: str) ->bool:
        if FVCorePathManager:
            return FVCorePathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) ->bool:
        if FVCorePathManager:
            return FVCorePathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def ls(path: str) ->List[str]:
        if FVCorePathManager:
            return FVCorePathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) ->None:
        if FVCorePathManager:
            return FVCorePathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) ->None:
        if FVCorePathManager:
            return FVCorePathManager.rm(path)
        os.remove(path)

    @staticmethod
    def register_handler(handler) ->None:
        if FVCorePathManager:
            return FVCorePathManager.register_handler(handler=handler)


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, *, pad='<pad>', eos='</s>', unk='<unk>', bos='<s>',
        extra_special_symbols=None):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False,
        extra_symbols_to_ignore=None, unk_string=None):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk,
                extra_symbols_to_ignore) for t in tensor)
        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]
        if hasattr(self, 'bos_index'):
            extra_symbols_to_ignore.add(self.bos())
        sent = ' '.join(token_string(i) for i in tensor if utils.item(i) not in
            extra_symbols_to_ignore)
        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)
        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.
            nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]
        c = Counter(dict(sorted(zip(self.symbols[self.nspecial:], self.
            count[self.nspecial:]))))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break
        assert len(new_symbols) == len(new_indices)
        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices
        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with PathManager.open(f, 'r', encoding='utf-8') as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    'Incorrect encoding detected in {}, please rebuild the dataset'
                    .format(f))
            return
        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(' ', 1)
                if field == '#fairseq:overwrite':
                    overwrite = True
                    line, field = line.rsplit(' ', 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. Duplicate words can overwrite earlier ones by adding the #fairseq:overwrite flag at the end of the corresponding row in the dictionary file. If using the Camembert model, please download an updated copy of the model file."
                        .format(word))
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                    )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols[self.nspecial:], ex_vals +
            self.count[self.nspecial:]))

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(self, line, line_tokenizer=tokenize_line,
        add_if_not_exist=True, consumer=None, append_eos=True,
        reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(filename, tokenize, eos_word,
        worker_id=0, num_workers=1):
        counter = Counter()
        with open(PathManager.get_local_path(filename), 'r', encoding='utf-8'
            ) as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):

        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(Dictionary.
                    _add_file_to_dictionary_single_worker, (filename,
                    tokenize, dict.eos_word, worker_id, num_workers)))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Dictionary._add_file_to_dictionary_single_worker(
                filename, tokenize, dict.eos_word))


CHAR_EOS_IDX = 257


CHAR_PAD_IDX = 0


class CharacterTokenEmbedder(torch.nn.Module):

    def __init__(self, vocab: Dictionary, filters: List[Tuple[int, int]],
        char_embed_dim: int, word_embed_dim: int, highway_layers: int,
        max_char_len: int=50, char_inputs: bool=False):
        super(CharacterTokenEmbedder, self).__init__()
        self.onnx_trace = False
        self.embedding_dim = word_embed_dim
        self.max_char_len = max_char_len
        self.char_embeddings = nn.Embedding(257, char_embed_dim, padding_idx=0)
        self.symbol_embeddings = nn.Parameter(torch.FloatTensor(2,
            word_embed_dim))
        self.eos_idx, self.unk_idx = 0, 1
        self.char_inputs = char_inputs
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(nn.Conv1d(char_embed_dim, out_c,
                kernel_size=width))
        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers
            ) if highway_layers > 0 else None
        self.projection = nn.Linear(last_dim, word_embed_dim)
        assert vocab is not None or char_inputs, 'vocab must be set if not using char inputs'
        self.vocab = None
        if vocab is not None:
            self.set_vocab(vocab, max_char_len)
        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def set_vocab(self, vocab, max_char_len):
        word_to_char = torch.LongTensor(len(vocab), max_char_len)
        truncated = 0
        for i in range(len(vocab)):
            if i < vocab.nspecial:
                char_idxs = [0] * max_char_len
            else:
                chars = vocab[i].encode()
                char_idxs = [(c + 1) for c in chars] + [0] * (max_char_len -
                    len(chars))
            if len(char_idxs) > max_char_len:
                truncated += 1
                char_idxs = char_idxs[:max_char_len]
            word_to_char[i] = torch.LongTensor(char_idxs)
        if truncated > 0:
            logger.info('truncated {} words longer than {} characters'.
                format(truncated, max_char_len))
        self.vocab = vocab
        self.word_to_char = word_to_char

    @property
    def padding_idx(self):
        return Dictionary().pad() if self.vocab is None else self.vocab.pad()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.char_embeddings.weight)
        nn.init.xavier_normal_(self.symbol_embeddings)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.char_embeddings.weight[self.char_embeddings.
            padding_idx], 0.0)
        nn.init.constant_(self.projection.bias, 0.0)

    def forward(self, input: torch.Tensor):
        if self.char_inputs:
            chars = input.view(-1, self.max_char_len)
            pads = chars[:, (0)].eq(CHAR_PAD_IDX)
            eos = chars[:, (0)].eq(CHAR_EOS_IDX)
            if eos.any():
                if self.onnx_trace:
                    chars = torch.where(eos.unsqueeze(1), chars.new_zeros(1
                        ), chars)
                else:
                    chars[eos] = 0
            unk = None
        else:
            flat_words = input.view(-1)
            chars = self.word_to_char[flat_words.type_as(self.word_to_char)
                ].type_as(input)
            pads = flat_words.eq(self.vocab.pad())
            eos = flat_words.eq(self.vocab.eos())
            unk = flat_words.eq(self.vocab.unk())
        word_embs = self._convolve(chars)
        if self.onnx_trace:
            if pads.any():
                word_embs = torch.where(pads.unsqueeze(1), word_embs.
                    new_zeros(1), word_embs)
            if eos.any():
                word_embs = torch.where(eos.unsqueeze(1), self.
                    symbol_embeddings[self.eos_idx], word_embs)
            if unk is not None and unk.any():
                word_embs = torch.where(unk.unsqueeze(1), self.
                    symbol_embeddings[self.unk_idx], word_embs)
        else:
            if pads.any():
                word_embs[pads] = 0
            if eos.any():
                word_embs[eos] = self.symbol_embeddings[self.eos_idx]
            if unk is not None and unk.any():
                word_embs[unk] = self.symbol_embeddings[self.unk_idx]
        return word_embs.view(input.size()[:2] + (-1,))

    def _convolve(self, char_idxs: torch.Tensor):
        char_embs = self.char_embeddings(char_idxs)
        char_embs = char_embs.transpose(1, 2)
        conv_result = []
        for conv in self.convolutions:
            x = conv(char_embs)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)
        x = torch.cat(conv_result, dim=-1)
        if self.highway is not None:
            x = self.highway(x)
        x = self.projection(x)
        return x


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """

    def __init__(self, input_dim: int, num_layers: int=1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for
            _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x


class ConvTBC(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.weight = torch.nn.Parameter(torch.Tensor(self.kernel_size[0],
            in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias,
            self.padding[0])

    def __repr__(self):
        s = (
            '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, padding={padding}'
            )
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def GatedLinear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(Linear(in_features, out_features * 4, dropout,
        bias), nn.GLU(), Linear(out_features * 2, out_features * 2, dropout,
        bias), nn.GLU(), Linear(out_features, out_features, dropout, bias))


class ScalarBias(torch.autograd.Function):
    """
    Adds a vector of scalars, used in self-attention mechanism to allow
    the model to optionally attend to this vector instead of the past
    """

    @staticmethod
    def forward(ctx, input, dim, bias_init):
        size = list(input.size())
        size[dim] += 1
        output = input.new(*size).fill_(bias_init)
        output.narrow(dim, 1, size[dim] - 1).copy_(input)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad):
        return grad.narrow(ctx.dim, 1, grad.size(ctx.dim) - 1), None, None


def scalar_bias(input, dim, bias_init=0):
    return ScalarBias.apply(input, dim, bias_init)


class SingleHeadAttention(nn.Module):
    """
    Single-head attention that supports Gating and Downsampling
    """

    def __init__(self, out_channels, embed_dim, head_dim, head_index,
        dropout=0.0, bias=True, project_input=True, gated=False, downsample
        =False, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.head_index = head_index
        self.head_dim = head_dim
        self.project_input = project_input
        self.gated = gated
        self.downsample = downsample
        self.num_heads = num_heads
        self.projection = None
        k_layers = []
        v_layers = []
        if self.downsample:
            k_layers.append(Downsample(self.head_index))
            v_layers.append(Downsample(self.head_index))
            out_proj_size = self.head_dim
        else:
            out_proj_size = self.head_dim * self.num_heads
        if self.gated:
            k_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias
                =bias))
            self.in_proj_q = GatedLinear(self.embed_dim, out_proj_size,
                bias=bias)
            v_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias
                =bias))
        else:
            k_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = Linear(self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
        self.in_proj_k = nn.Sequential(*k_layers)
        self.in_proj_v = nn.Sequential(*v_layers)
        if self.downsample:
            self.out_proj = Linear(out_proj_size, self.head_dim, bias=bias)
        else:
            self.out_proj = Linear(out_proj_size, out_channels, bias=bias)
        self.scaling = self.head_dim ** -0.5

    def forward(self, query, key, value, mask_future_timesteps=False,
        key_padding_mask=None, use_scalar_bias=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        src_len, bsz, out_channels = key.size()
        tgt_len = query.size(0)
        assert list(query.size()) == [tgt_len, bsz, out_channels]
        assert key.size() == value.size()
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.downsample:
            size = bsz
        else:
            size = bsz * self.num_heads
        k = key
        v = value
        q = query
        if self.project_input:
            q = self.in_proj_q(q)
            k = self.in_proj_k(k)
            v = self.in_proj_v(v)
            src_len = k.size()[0]
        q *= self.scaling
        if not self.downsample:
            q = q.view(tgt_len, size, self.head_dim)
            k = k.view(src_len, size, self.head_dim)
            v = v.view(src_len, size, self.head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if mask_future_timesteps:
            assert query.size() == key.size(
                ), 'mask_future_timesteps only applies to self-attention'
            attn_weights *= torch.tril(attn_weights.data.new([1]).expand(
                tgt_len, tgt_len).clone(), diagonal=-1)[:, ::self.
                head_index + 1 if self.downsample else 1].unsqueeze(0)
            attn_weights += torch.triu(attn_weights.data.new([-math.inf]).
                expand(tgt_len, tgt_len).clone(), diagonal=0)[:, ::self.
                head_index + 1 if self.downsample else 1].unsqueeze(0)
        tgt_size = tgt_len
        if use_scalar_bias:
            attn_weights = scalar_bias(attn_weights, 2)
            v = scalar_bias(v, 1)
            tgt_size += 1
        if key_padding_mask is not None:
            if key_padding_mask.max() > 0:
                if self.downsample:
                    attn_weights = attn_weights.view(bsz, 1, tgt_len, src_len)
                else:
                    attn_weights = attn_weights.view(size, self.num_heads,
                        tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(key_padding_mask.
                    unsqueeze(1).unsqueeze(2), -math.inf)
                attn_weights = attn_weights.view(size, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=
            self.training)
        attn = torch.bmm(attn_weights, v)
        if self.downsample:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
                self.head_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
                self.embed_dim)
        attn = self.out_proj(attn)
        return attn, attn_weights


class DownsampledMultiHeadAttention(nn.ModuleList):
    """
    Multi-headed attention with Gating and Downsampling
    """

    def __init__(self, out_channels, embed_dim, num_heads, dropout=0.0,
        bias=True, project_input=True, gated=False, downsample=False):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.downsample = downsample
        self.gated = gated
        self.project_input = project_input
        assert self.head_dim * num_heads == embed_dim
        if self.downsample:
            attention_heads = []
            for index in range(self.num_heads):
                attention_heads.append(SingleHeadAttention(out_channels,
                    self.embed_dim, self.head_dim, index, self.dropout,
                    bias, self.project_input, self.gated, self.downsample,
                    self.num_heads))
            super().__init__(modules=attention_heads)
            self.out_proj = Linear(embed_dim, out_channels, bias=bias)
        else:
            super().__init__()
            self.attention_module = SingleHeadAttention(out_channels, self.
                embed_dim, self.head_dim, 1, self.dropout, bias, self.
                project_input, self.gated, self.downsample, self.num_heads)

    def forward(self, query, key, value, mask_future_timesteps=False,
        key_padding_mask=None, use_scalar_bias=False):
        src_len, bsz, embed_dim = key.size()
        tgt_len = query.size(0)
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        tgt_size = tgt_len
        if use_scalar_bias:
            tgt_size += 1
        attn = []
        attn_weights = []
        if self.downsample:
            for attention_head_number in range(self.num_heads):
                _attn, _attn_weight = self[attention_head_number](query,
                    key, value, mask_future_timesteps, key_padding_mask,
                    use_scalar_bias)
                attn.append(_attn)
                attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn = self.out_proj(full_attn)
            return full_attn, attn_weights[0].clone()
        else:
            _attn, _attn_weight = self.attention_module(query, key, value,
                mask_future_timesteps, key_padding_mask, use_scalar_bias)
            attn.append(_attn)
            attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn_weights = torch.cat(attn_weights)
            full_attn_weights = full_attn_weights.view(bsz, self.num_heads,
                tgt_size, src_len)
            full_attn_weights = full_attn_weights.sum(dim=1) / self.num_heads
            return full_attn, full_attn_weights


class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[::self.index + 1]


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    """unfold T x B x C to T x B x C x K"""
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l),
            value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x


@with_incremental_state
class DynamicConv1dTBC(nn.Module):
    """Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads
        =1, weight_dropout=0.0, weight_softmax=False, renorm_padding=False,
        bias=False, conv_bias=False, query_size=None, in_proj=False):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding
        if in_proj:
            self.weight_linear = Linear(self.input_size, self.input_size + 
                num_heads * kernel_size * 1)
        else:
            self.weight_linear = Linear(self.query_size, num_heads *
                kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return (self.weight_linear.out_features == self.input_size + self.
            num_heads * self.kernel_size)

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.0)

    def forward(self, x, incremental_state=None, query=None, unfold=None):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        """
        unfold = x.size(0) > 512 if unfold is None else unfold
        unfold = unfold or incremental_state is not None
        assert query is None or not self.in_proj
        if query is None:
            query = x
        if unfold:
            output = self._forward_unfolded(x, incremental_state, query)
        else:
            output = self._forward_expanded(x, incremental_state, query)
        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        return output

    def _forward_unfolded(self, x, incremental_state, query):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H * K).contiguous().view(
                T * B * H, -1)
        else:
            weight = self.weight_linear(query).view(T * B * H, -1)
        assert not self.renorm_padding or incremental_state is not None
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :,
                    -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            padding_l = self.padding_l
            if K > T and padding_l == K - 1:
                weight = weight.narrow(1, K - T, T)
                K, padding_l = T, T - 1
            x_unfold = unfold1d(x, K, padding_l, 0)
            x_unfold = x_unfold.view(T * B * H, R, K)
        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)
        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = F.dropout(weight, self.weight_dropout, training=self.
            training, inplace=False)
        output = torch.bmm(x_unfold, weight.unsqueeze(2))
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_stat, query):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H * K).contiguous().view(
                T * B * H, -1)
        else:
            weight = self.weight_linear(query).view(T * B * H, -1)
        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout, training=self.
                training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)
        x = x.view(T, B * H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            weight_expanded = weight.new(B * H, T, T + K - 1).fill_(float(
                '-inf'))
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T +
                K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(weight_expanded, self.
                weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            if K > T and P == K - 1:
                weight = weight.narrow(2, K - T, T)
                K, P = T, T - 1
            weight_expanded = weight.new_zeros(B * H, T, T + K - 1,
                requires_grad=False)
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T +
                K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state,
            'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state,
            'input_buffer', new_buffer)

    def extra_repr(self):
        s = (
            '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, conv_bias={}, renorm_padding={}, in_proj={}'
            .format(self.input_size, self.kernel_size, self.padding_l, self
            .num_heads, self.weight_softmax, self.conv_bias is not None,
            self.renorm_padding, self.in_proj))
        if self.query_size != self.input_size:
            s += ', query_size={}'.format(self.query_size)
        if self.weight_dropout > 0.0:
            s += ', weight_dropout={}'.format(self.weight_dropout)
        return s


def logsumexp(x, dim=1):
    return torch.logsumexp(x.float(), dim=dim).type_as(x)


class DynamicCRF(nn.Module):
    """Dynamic CRF layer is used to approximate the traditional
       Conditional Random Fields (CRF)
       $P(y | x) = 1/Z(x) exp(sum_i s(y_i, x) + sum_i t(y_{i-1}, y_i, x))$

       where in this function, we assume the emition scores (s) are given,
       and the transition score is a |V| x |V| matrix $M$

       in the following two aspects:
        (1) it used a low-rank approximation for the transition matrix:
            $M = E_1 E_2^T$
        (2) it used a beam to estimate the normalizing factor Z(x)
    """

    def __init__(self, num_embedding, low_rank=32, beam_size=64):
        super().__init__()
        self.E1 = nn.Embedding(num_embedding, low_rank)
        self.E2 = nn.Embedding(num_embedding, low_rank)
        self.vocb = num_embedding
        self.rank = low_rank
        self.beam = beam_size

    def extra_repr(self):
        return 'vocab_size={}, low_rank={}, beam_size={}'.format(self.vocb,
            self.rank, self.beam)

    def forward(self, emissions, targets, masks, beam=None):
        """
        Compute the conditional log-likelihood of a sequence of target tokens given emission scores

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            targets (`~torch.LongTensor`): Sequence of target token indices
                ``(batch_size, seq_len)
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.Tensor`: approximated log-likelihood
        """
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalizer(emissions, targets, masks, beam)
        return numerator - denominator

    def forward_decoder(self, emissions, masks=None, beam=None):
        """
        Find the most likely output sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.LongTensor`: decoded sequence from the CRF model
        """
        return self._viterbi_decode(emissions, masks, beam)

    def _compute_score(self, emissions, targets, masks=None):
        batch_size, seq_len = targets.size()
        emission_scores = emissions.gather(2, targets[:, :, (None)])[:, :, (0)]
        transition_scores = (self.E1(targets[:, :-1]) * self.E2(targets[:, 1:])
            ).sum(2)
        scores = emission_scores
        scores[:, 1:] += transition_scores
        if masks is not None:
            scores = scores * masks.type_as(scores)
        return scores.sum(-1)

    def _compute_normalizer(self, emissions, targets=None, masks=None, beam
        =None):
        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        if targets is not None:
            _emissions = emissions.scatter(2, targets[:, :, (None)], np.
                float('inf'))
            beam_targets = _emissions.topk(beam, 2)[1]
            beam_emission_scores = emissions.gather(2, beam_targets)
        else:
            beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])
        beam_transition_score2 = self.E2(beam_targets[:, 1:])
        beam_transition_matrix = torch.bmm(beam_transition_score1.view(-1,
            beam, self.rank), beam_transition_score2.view(-1, beam, self.
            rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1,
            beam, beam)
        score = beam_emission_scores[:, (0)]
        for i in range(1, seq_len):
            next_score = score[:, :, (None)] + beam_transition_matrix[:, (i -
                1)]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:,
                (i)]
            if masks is not None:
                score = torch.where(masks[:, i:i + 1], next_score, score)
            else:
                score = next_score
        return logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, masks=None, beam=None):
        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])
        beam_transition_score2 = self.E2(beam_targets[:, 1:])
        beam_transition_matrix = torch.bmm(beam_transition_score1.view(-1,
            beam, self.rank), beam_transition_score2.view(-1, beam, self.
            rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1,
            beam, beam)
        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []
        score = beam_emission_scores[:, (0)]
        dummy = torch.arange(beam, device=score.device).expand(*score.size()
            ).contiguous()
        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, (None)] + beam_transition_matrix[:, (i - 1)]
            _score, _index = _score.max(dim=1)
            _score = _score + beam_emission_scores[:, (i)]
            if masks is not None:
                score = torch.where(masks[:, i:i + 1], _score, score)
                index = torch.where(masks[:, i:i + 1], _index, dummy)
            else:
                score, index = _score, _index
            traj_tokens.append(index)
        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, (None)])
        finalized_scores.append(best_score[:, (None)])
        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))
        finalized_tokens.reverse()
        finalized_tokens = torch.cat(finalized_tokens, 1)
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, (
            None)])[:, :, (0)]
        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:,
            :-1]
        return finalized_scores, finalized_tokens


class dynamicconvFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = dynamicconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = dynamicconv_cuda.backward(grad_output.contiguous(), ctx.
            padding_l, *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


@with_incremental_state
class DynamicconvLayer(nn.Module):

    def __init__(self, input_size, kernel_size=1, padding_l=None,
        weight_softmax=False, num_heads=1, weight_dropout=0.0, bias=False,
        renorm_padding=False, conv_bias=False, query_size=None):
        super(DynamicconvLayer, self).__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.renorm_padding = renorm_padding
        self.bias = bias
        self.weight_linear = nn.Linear(input_size, num_heads * kernel_size,
            bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight)
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.0)
            nn.init.constant_(self.weight_linaer.bias, 0.0)

    def forward(self, x, incremental_state=None, query=None, unfold=None):
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        if incremental_state is not None:
            unfold = x.size(0) > 512 if unfold is None else unfold
            unfold = unfold or incremental_state is not None
            assert query is None
            if query is None:
                query = x
            if unfold:
                output = self._forward_unfolded(x, incremental_state, query)
            else:
                output = self._forward_expanded(x, incremental_state, query)
            if self.conv_bias is not None:
                output = output + self.conv_bias.view(1, 1, -1)
            return output
        else:
            weight = self.weight_linear(x).view(T, B, H, K)
            if self.weight_softmax:
                weight = F.softmax(weight, dim=-1)
            if self.weight_dropout:
                weight = F.dropout(weight, self.weight_dropout, training=
                    self.training)
            weight = weight.permute(1, 2, 3, 0).contiguous()
            self.filters = weight
            x = x.permute(1, 2, 0).contiguous()
            output = dynamicconvFunction.apply(x, weight, self.padding_l
                ).permute(2, 0, 1)
            if self.conv_bias is not None:
                output = output + self.conv_bias.view(1, 1, -1)
            return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state,
            'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state,
            'input_buffer', new_buffer)

    def _forward_unfolded(self, x, incremental_state, query):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight_linear(query).view(T * B * H, -1)
        assert not self.renorm_padding or incremental_state is not None
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :,
                    -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            padding_l = self.padding_l
            if K > T and padding_l == K - 1:
                weight = weight.narrow(1, K - T, T)
                K, padding_l = T, T - 1
            x_unfold = unfold1d(x, K, padding_l, 0)
            x_unfold = x_unfold.view(T * B * H, R, K)
        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)
        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = F.dropout(weight, self.weight_dropout, training=self.
            training, inplace=False)
        output = torch.bmm(x_unfold, weight.unsqueeze(2))
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_stat, query):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight_linear(query).view(T * B * H, -1)
        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout, training=self.
                training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)
        x = x.view(T, B * H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            weight_expanded = weight.new(B * H, T, T + K - 1).fill_(float(
                '-inf'))
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T +
                K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(weight_expanded, self.
                weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            if K > T and P == K - 1:
                weight = weight.narrow(2, K - T, T)
                K, P = T, T - 1
            weight_expanded = weight.new_zeros(B * H, T, T + K - 1,
                requires_grad=False)
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T +
                K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output


class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(input.float(), self.num_groups, self.weight.
            float() if self.weight is not None else None, self.bias.float() if
            self.bias is not None else None, self.eps)
        return output.type_as(input)


class KmeansVectorQuantizer(nn.Module):

    def __init__(self, dim, num_vars, groups, combine_groups, vq_dim,
        time_first, gamma=0.25):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

                Args:
                    dim: input dimension (channels)
                    num_vars: number of quantized vectors per group
                    groups: number of groups for vector quantization
                    combine_groups: whether to use the vectors for all groups
                    vq_dim: dimensionality of the resulting quantized vector
                    time_first: if true, expect input in BxTxC format, otherwise in BxCxT
                    gamma: commitment loss coefficient
                """
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first
        assert vq_dim % groups == 0, f'dim {vq_dim} must be divisible by groups {groups} for concatenation'
        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.embedding = nn.Parameter(0.01 * torch.randn(num_vars,
            num_groups, self.var_dim))
        self.projection = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1,
            groups=groups, bias=False), Fp32GroupNorm(groups, dim))
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction='mean')

    def _pass_grad(self, x, y):
        """ Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """
        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.
                var_dim)
        return self.embedding

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res['x'], res['targets']

    def forward(self, x, produce_targets=False):
        result = {'num_vars': self.num_vars}
        if self.time_first:
            x = x.transpose(1, 2)
        bsz, fsz, tsz = x.shape
        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1)
            ).view(self.num_vars, bsz, tsz, self.groups, -1).norm(dim=-1, p=2)
        idx = d.argmin(dim=0)
        zq = torch.stack([self.expand_embedding[idx[..., group], group] for
            group in range(self.groups)], dim=-2).view(bsz, tsz, self.
            groups * self.var_dim).permute(0, 2, 1)
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)
        hard_x = idx.new_zeros(bsz * tsz * self.groups, self.num_vars
            ).scatter_(-1, idx.view(-1, 1), 1.0).view(bsz * tsz, self.
            groups, -1)
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result['code_perplexity'] = torch.exp(-torch.sum(hard_probs * torch
            .log(hard_probs + 1e-07), dim=-1)).sum()
        if produce_targets:
            result['targets'] = idx
        if self.time_first:
            x = x.transpose(1, 2)
        result['x'] = x
        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())
        result['kmeans_loss'] = latent_loss + self.gamma * commitment_loss
        return result


class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or dropout_probs[i] > self.p:
                yield m


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(input.float(), self.normalized_shape, self.
            weight.float() if self.weight is not None else None, self.bias.
            float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx:
        int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(self, input: Tensor, incremental_state: Optional[Dict[str,
        Dict[str, Optional[Tensor]]]]=None, positions: Optional[Tensor]=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert positions is None or self.padding_idx is None, 'If positions is pre-computed then padding_idx should not be set.'
        if positions is None:
            if incremental_state is not None:
                positions = torch.zeros((1, 1), device=input.device, dtype=
                    input.dtype).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(input, self.padding_idx,
                    onnx_trace=self.onnx_trace)
        return F.embedding(positions, self.weight, self.padding_idx, self.
            max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class lightconvFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = lightconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = lightconv_cuda.backward(grad_output.contiguous(), ctx.
            padding_l, *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


@with_incremental_state
class LightconvLayer(nn.Module):

    def __init__(self, input_size, kernel_size=1, padding_l=None,
        weight_softmax=False, num_heads=1, weight_dropout=0.0, bias=False):
        super(LightconvLayer, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.weight = nn.Parameter(torch.Tensor(num_heads, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.reset_parameters()

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        for k, v in state_dict.items():
            if k.endswith(prefix + 'weight'):
                if v.dim() == 3 and v.size(1) == 1:
                    state_dict[k] = v.squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            T, B, C = x.size()
            K, H = self.kernel_size, self.num_heads
            R = C // H
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :,
                    -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
            weight = self.weight
            if self.weight_softmax:
                weight = F.softmax(weight.float(), dim=1).type_as(weight)
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
            weight = weight.view(1, H, K).expand(T * B, H, K).contiguous(
                ).view(T * B * H, K, 1)
            weight = F.dropout(weight, self.weight_dropout, training=self.
                training)
            output = torch.bmm(x_unfold, weight)
            output = output.view(T, B, C)
            return output
        else:
            x = x.permute(1, 2, 0).contiguous()
            weight = self.weight
            if self.weight_softmax:
                weight = F.softmax(self.weight, -1)
            if self.weight_dropout:
                weight = F.dropout(weight, self.weight_dropout, training=
                    self.training)
            return lightconvFunction.apply(x, weight, self.padding_l).permute(
                2, 0, 1)

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state,
            'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state,
            'input_buffer', new_buffer)

    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)


class LightweightConv1d(nn.Module):
    """Lightweight Convolution assuming the input is BxCxT
    This is just an example that explains LightConv clearer than the TBC version.
    We don't use this module in the model.

    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution

    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding=0, num_heads=1,
        weight_softmax=False, bias=False, weight_dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout = weight_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads
        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.
            num_heads)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


@with_incremental_state
class LightweightConv1dTBC(nn.Module):
    """Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads
        =1, weight_dropout=0.0, weight_softmax=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.reset_parameters()
        self.onnx_trace = False

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, incremental_state=None, unfold=False):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
        """
        unfold = unfold or incremental_state is not None
        if unfold:
            output = self._forward_unfolded(x, incremental_state)
        else:
            output = self._forward_expanded(x, incremental_state)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)
        return output

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _forward_unfolded(self, x, incremental_state):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight.view(H, K)
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :,
                    -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
            x_unfold = x_unfold.view(T * B * H, R, K)
        if self.weight_softmax:
            weight = utils.softmax(weight, dim=1, onnx_trace=self.onnx_trace
                ).type_as(weight)
        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous().view(
            T * B * H, K, 1)
        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        output = torch.bmm(x_unfold, weight)
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_state):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight.view(H, K)
        if self.weight_softmax:
            weight = utils.softmax(weight, dim=1, onnx_trace=self.onnx_trace
                ).type_as(weight)
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)
        x = x.view(T, B * H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K - 1:
            weight = weight.narrow(2, K - T, T)
            K, P = T, T - 1
        weight_expanded = weight.new_zeros(B * H, T, T + K - 1,
            requires_grad=False)
        weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)
            ).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T)
        weight_expanded = F.dropout(weight_expanded, self.weight_dropout,
            training=self.training)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state,
            'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state,
            'input_buffer', new_buffer)

    def extra_repr(self):
        s = (
            '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}'
            .format(self.input_size, self.kernel_size, self.padding_l, self
            .num_heads, self.weight_softmax, self.bias is not None))
        if self.weight_dropout > 0.0:
            s += ', weight_dropout={}'.format(self.weight_dropout)
        return s


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=
        0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
        self_attention=False, encoder_decoder_attention=False, q_noise=0.0,
        qn_block_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias
            ), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias
            ), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias
            ), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=
            bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
        key_padding_mask: Optional[Tensor]=None, incremental_state:
        Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None, need_weights:
        bool=True, static_kv: bool=False, attn_mask: Optional[Tensor]=None,
        before_softmax: bool=False, need_head_weights: bool=False) ->Tuple[
        Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if (not self.onnx_trace and not self.tpu and incremental_state is
            None and not static_kv and not torch.jit.is_scripting()):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(query, key, value, self.
                embed_dim, self.num_heads, torch.empty([0]), torch.cat((
                self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k, self.bias_v, self.add_zero_attn, self.dropout,
                self.out_proj.weight, self.out_proj.bias, self.training,
                key_padding_mask, need_weights, attn_mask,
                use_separate_proj_weight=True, q_proj_weight=self.q_proj.
                weight, k_proj_weight=self.k_proj.weight, v_proj_weight=
                self.v_proj.weight)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(
                    attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask,
                    key_padding_mask.new_zeros(key_padding_mask.size(0), 1)
                    ], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim
            ).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim
                ).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim
                ).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.
                    head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1,
                    self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = (MultiheadAttention.
                _append_prev_key_padding_mask(key_padding_mask=
                key_padding_mask, prev_key_padding_mask=
                prev_key_padding_mask, batch_size=bsz, src_len=k.size(1),
                static_kv=static_kv))
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.
                head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1,
                self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state,
                saved_state)
        assert k is not None
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])],
                dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])],
                dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(
                    attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros
                    (key_padding_mask.size(0), 1).type_as(key_padding_mask)
                    ], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights,
            tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len,
            src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.
                    unsqueeze(1).unsqueeze(2).to(torch.bool), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask,
                    float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace
            =self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.
            training)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.
            head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
                embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads,
                tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor], batch_size: int, src_len:
        int, static_kv: bool) ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(),
                key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len -
                prev_key_padding_mask.size(1)), device=
                prev_key_padding_mask.device)
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(),
                filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - key_padding_mask.
                size(1)), device=key_padding_mask.device)
            new_key_padding_mask = torch.cat([filler.float(),
                key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[
        str, Optional[Tensor]]], new_order: Tensor):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0
                        ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state,
                input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[
        str, Optional[Tensor]]]]) ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str,
        Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, 'attn_state',
            buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + 'in_proj_weight'):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + 'q_proj.weight'] = state_dict[k][:dim]
                items_to_add[prefix + 'k_proj.weight'] = state_dict[k][dim:
                    2 * dim]
                items_to_add[prefix + 'v_proj.weight'] = state_dict[k][2 * dim:
                    ]
                keys_to_remove.append(k)
                k_bias = prefix + 'in_proj_bias'
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + 'q_proj.bias'] = state_dict[k_bias][:
                        dim]
                    items_to_add[prefix + 'k_proj.bias'] = state_dict[k_bias][
                        dim:2 * dim]
                    items_to_add[prefix + 'v_proj.bias'] = state_dict[k_bias][
                        2 * dim:]
                    keys_to_remove.append(prefix + 'in_proj_bias')
        for k in keys_to_remove:
            del state_dict[k]
        for key, value in items_to_add.items():
            state_dict[key] = value


class PQConv2d(nn.Module):
    """
    Quantized counterpart of nn.Conv2d module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiated at each forward
    pass and autograd automatically computes the gradients with respect to the
    centroids.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_channels x n_blocks
        - bias: the non-quantized bias, must be either torch.Tensor or None

    Remarks:
        - We refer the reader to the official documentation of the nn.Conv2d module
          for the other arguments and the behavior of the module.
        - Performance tests on GPU show that this implementation is 10% slower than
          the non-quantized nn.Conv2d module for a standard training loop.
        - During the backward, the gradients are averaged by cluster and not summed.
          This explains the hook registered to the centroids.
    """

    def __init__(self, centroids, assignments, bias, in_channels,
        out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=
        1, padding_mode='zeros'):
        super(PQConv2d, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        if in_channels // groups * np.prod(self.kernel_size
            ) % self.block_size != 0:
            raise ValueError('Wrong PQ sizes')
        if len(assignments) % out_channels != 0:
            raise ValueError('Wrong PQ sizes')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer('assignments', assignments)
        self.register_buffer('counts', torch.bincount(assignments).type_as(
            centroids))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        self.centroids.register_hook(lambda x: x / self.counts[:, (None)])

    @property
    def weight(self):
        return self.centroids[self.assignments].reshape(-1, self.
            out_channels, self.block_size).permute(1, 0, 2).reshape(self.
            out_channels, self.in_channels // self.groups, *self.kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)

    def extra_repr(self):
        s = (
            '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
            )
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', n_centroids={n_centroids}, block_size={block_size}'
        return s.format(**self.__dict__)


class PQEmbedding(nn.Module):
    """
    Quantized counterpart of nn.Embedding module. Stores the centroids and
    the assignments. The full weight is re-instantiated at each forward
    pass.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Embedding module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 10% slower than
          the non-quantized nn.Embedding module for a standard training loop.
    """

    def __init__(self, centroids, assignments, num_embeddings,
        embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
        scale_grad_by_freq=False, sparse=False, _weight=None):
        super(PQEmbedding, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if self.embedding_dim % self.block_size != 0:
            raise ValueError('Wrong PQ sizes')
        if len(assignments) % self.num_embeddings != 0:
            raise ValueError('Wrong PQ sizes')
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer('assignments', assignments)
        self.register_buffer('counts', torch.bincount(assignments).type_as(
            centroids))

    @property
    def weight(self):
        return self.centroids[self.assignments].reshape(-1, self.
            num_embeddings, self.block_size).permute(1, 0, 2).flatten(1, 2)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.
            max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ', n_centroids={n_centroids}, block_size={block_size}'
        return s.format(**self.__dict__)


class PQLinear(nn.Module):
    """
    Quantized counterpart of nn.Linear module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiated at each forward
    pass.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Linear module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 15% slower than
          the non-quantized nn.Linear module for a standard training loop.
    """

    def __init__(self, centroids, assignments, bias, in_features, out_features
        ):
        super(PQLinear, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.in_features = in_features
        self.out_features = out_features
        if self.in_features % self.block_size != 0:
            raise ValueError('Wrong PQ sizes')
        if len(assignments) % self.out_features != 0:
            raise ValueError('Wrong PQ sizes')
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer('assignments', assignments)
        self.register_buffer('counts', torch.bincount(assignments).type_as(
            centroids))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    @property
    def weight(self):
        return self.centroids[self.assignments].reshape(-1, self.
            out_features, self.block_size).permute(1, 0, 2).flatten(1, 2)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features},                 out_features={self.out_features},                 n_centroids={self.n_centroids},                 block_size={self.block_size},                 bias={self.bias is not None}'
            )


def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f'emulate_int{bits}_{method}']
    return q(w, scale=scale, zero_point=zero_point)


class IntConv2d(_ConvNd):
    """
    Quantized counterpart of the nn.Conv2d module that applies QuantNoise during training.

    Args:
        - standard nn.Conv2d parameters
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-thgourh estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', p
        =0, bits=8, method='histogram', update_step=1000):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(IntConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, False, _pair(0), groups,
            bias, padding_mode)
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode
                =self.padding_mode), weight, self.bias, self.stride, _pair(
                0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)

    def forward(self, input):
        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(self.
            weight.detach(), bits=self.bits, method=self.method, scale=self
            .scale, zero_point=self.zero_point)
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(self.weight, clamp_low.item(), clamp_high.item()
            ) + noise.detach()
        output = self._conv_forward(input, weight)
        return output

    def extra_repr(self):
        return (
            'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}, quant_noise={}, bits={}, method={}'
            .format(self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups, self.
            bias is not None, self.p, self.bits, self.method))


class IntEmbedding(nn.Module):
    """
    Quantized counterpart of the nn.Embedding module that applies QuantNoise during training.

    Args:
        - num_embeddings: number of tokens
        - embedding_dim: embedding dimension
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
        max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=
        False, _weight=None, p=0, update_step=1000, bits=8, method='histogram'
        ):
        super(IntEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings,
                embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim
                ], 'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(self.
            weight.detach(), bits=self.bits, method=self.method, scale=self
            .scale, zero_point=self.zero_point)
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(self.weight, clamp_low.item(), clamp_high.item()
            ) + noise.detach()
        output = F.embedding(input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return output

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += 'quant_noise={p}, bits={bits}, method={method}'
        return s.format(**self.__dict__)


class IntLinear(nn.Module):
    """
    Quantized counterpart of the nn.Linear module that applies QuantNoise during training.

    Args:
        - in_features: input features
        - out_features: output features
        - bias: bias or not
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick.
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, in_features, out_features, bias=True, p=0,
        update_step=3000, bits=8, method='histogram'):
        super(IntLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features,
            in_features))
        self.chosen_bias = bias
        if self.chosen_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.chosen_bias:
            nn.init.constant_(self.bias, 0.0)
        return

    def forward(self, input):
        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(self.
            weight.detach(), bits=self.bits, method=self.method, scale=self
            .scale, zero_point=self.zero_point)
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(self.weight, clamp_low.item(), clamp_high.item()
            ) + noise.detach()
        output = F.linear(input, weight, self.bias)
        return output

    def extra_repr(self):
        return (
            'in_features={}, out_features={}, bias={}, quant_noise={}, bits={}, method={}'
            .format(self.in_features, self.out_features, self.bias is not
            None, self.p, self.bits, self.method))


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size,
            embedding_dim, padding_idx)
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
        self.max_positions = int(100000.0)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx:
        Optional[int]=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1
            ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[(padding_idx), :] = 0
        return emb

    def forward(self, input, incremental_state: Optional[Any]=None,
        timestep: Optional[Tensor]=None, positions: Optional[Any]=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos,
                self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights.index_select(index=self.padding_idx +
                    pos, dim=0).unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[(self.padding_idx + pos), :].expand(bsz, 1, -1)
        positions = utils.make_positions(input, self.padding_idx,
            onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0,
                positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1),
                torch.tensor([-1], dtype=torch.long)))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz,
            seq_len, -1).detach()


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args,
            'quant_noise_pq_block_size', 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(activation=getattr(
            args, 'activation_fn', 'relu'))
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(self.embed_dim, args.
            encoder_ffn_embed_dim, self.quant_noise, self.
            quant_noise_block_size)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.
            embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise,
            block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise,
            block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True, q_noise=
            self.quant_noise, qn_block_size=self.quant_noise_block_size)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {'0': 'self_attn_layer_norm', '1': 'final_layer_norm'}
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict['{}.{}.{}'.format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor]=None
        ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -
                100000000.0)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=
            encoder_padding_mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.
            training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False,
        add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args,
            'quant_noise_pq_block_size', 8)
        self.cross_self_attention = getattr(args, 'cross_self_attention', False
            )
        self.self_attn = self.build_self_attention(self.embed_dim, args,
            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.activation_fn = utils.get_activation_fn(activation=getattr(
            args, 'activation_fn', 'relu'))
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim,
                args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export
                =export)
        self.fc1 = self.build_fc1(self.embed_dim, args.
            decoder_ffn_embed_dim, self.quant_noise, self.
            quant_noise_block_size)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.
            embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True
        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise,
            qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise,
            qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False,
        add_zero_attn=False):
        return MultiheadAttention(embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout, add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn, self_attention=not getattr(args,
            'cross_self_attention', False), q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size)

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(embed_dim, args.decoder_attention_heads,
            kdim=getattr(args, 'encoder_embed_dim', None), vdim=getattr(
            args, 'encoder_embed_dim', None), dropout=args.
            attention_dropout, encoder_decoder_attention=True, q_noise=self
            .quant_noise, qn_block_size=self.quant_noise_block_size)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out: Optional[torch.Tensor]=None,
        encoder_padding_mask: Optional[torch.Tensor]=None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
        =None, prev_self_attn_state: Optional[List[torch.Tensor]]=None,
        prev_attn_state: Optional[List[torch.Tensor]]=None, self_attn_mask:
        Optional[torch.Tensor]=None, self_attn_padding_mask: Optional[torch
        .Tensor]=None, need_attn: bool=False, need_head_weights: bool=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {'prev_key':
                prev_key, 'prev_value': prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state['prev_key_padding_mask'] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(
            incremental_state)
        if self.cross_self_attention and not (incremental_state is not None and
            _self_attn_input_buffer is not None and 'prev_key' in
            _self_attn_input_buffer):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat((x.new_zeros(x.size(0),
                    encoder_out.size(0)), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0))
                self_attn_padding_mask = torch.cat((encoder_padding_mask,
                    self_attn_padding_mask), dim=1)
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        x, attn = self.self_attn(query=x, key=y, value=y, key_padding_mask=
            self_attn_padding_mask, incremental_state=incremental_state,
            need_weights=False, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {'prev_key':
                    prev_key, 'prev_value': prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state['prev_key_padding_mask'] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state,
                    saved_state)
            x, attn = self.encoder_attn(query=x, key=encoder_out, value=
                encoder_out, key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state, static_kv=True,
                need_weights=need_attn or not self.training and self.
                need_attn, need_head_weights=need_head_weights)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.
            training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [saved_state['prev_key'], saved_state[
                    'prev_value'], saved_state['prev_key_padding_mask']]
            else:
                self_attn_state = [saved_state['prev_key'], saved_state[
                    'prev_value']]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool=False, **kwargs):
        self.need_attn = need_attn

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[
        str, Optional[Tensor]]], new_order: Tensor):
        """Scriptable reorder incremental state in transformer layers."""
        self.self_attn.reorder_incremental_state(incremental_state, new_order)
        if self.encoder_attn is not None:
            self.encoder_attn.reorder_incremental_state(incremental_state,
                new_order)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


def PositionalEmbedding(num_embeddings: int, embedding_dim: int,
    padding_idx: int, learned: bool=False):
    if learned:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim,
            padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx,
            init_size=num_embeddings + padding_idx + 1)
    return m


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(self, padding_idx: int, vocab_size: int,
        num_encoder_layers: int=6, embedding_dim: int=768,
        ffn_embedding_dim: int=3072, num_attention_heads: int=8, dropout:
        float=0.1, attention_dropout: float=0.1, activation_dropout: float=
        0.1, layerdrop: float=0.0, max_seq_len: int=256, num_segments: int=
        2, use_position_embeddings: bool=True, offset_positions_by_padding:
        bool=True, encoder_normalize_before: bool=False, apply_bert_init:
        bool=False, activation_fn: str='relu', learned_pos_embedding: bool=
        True, embed_scale: float=None, freeze_embeddings: bool=False,
        n_trans_layers_to_freeze: int=0, export: bool=False, traceable:
        bool=False, q_noise: float=0.0, qn_block_size: int=8) ->None:
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False
        self.embed_tokens = nn.Embedding(self.vocab_size, self.
            embedding_dim, self.padding_idx)
        self.embed_scale = embed_scale
        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(nn.Linear(self.
                embedding_dim, self.embedding_dim, bias=False), q_noise,
                qn_block_size)
        else:
            self.quant_noise = None
        self.segment_embeddings = nn.Embedding(self.num_segments, self.
            embedding_dim, padding_idx=None) if self.num_segments > 0 else None
        self.embed_positions = PositionalEmbedding(self.max_seq_len, self.
            embedding_dim, padding_idx=self.padding_idx if
            offset_positions_by_padding else None, learned=self.
            learned_pos_embedding) if self.use_position_embeddings else None
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([TransformerSentenceEncoderLayer(embedding_dim=
            self.embedding_dim, ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads, dropout=self.dropout,
            attention_dropout=attention_dropout, activation_dropout=
            activation_dropout, activation_fn=activation_fn, export=export,
            q_noise=q_noise, qn_block_size=qn_block_size) for _ in range(
            num_encoder_layers)])
        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(self, tokens: torch.Tensor, segment_labels: torch.Tensor=
        None, last_state_only: bool=False, positions: Optional[torch.Tensor
        ]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(tokens)
        if self.embed_scale is not None:
            x *= self.embed_scale
        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)
        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)
        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)
        sentence_rep = x[(0), :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embedding_dim: int=768, ffn_embedding_dim: int=3072,
        num_attention_heads: int=8, dropout: float=0.1, attention_dropout:
        float=0.1, activation_dropout: float=0.1, activation_fn: str='relu',
        export: bool=False, q_noise: float=0.0, qn_block_size: int=8) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(self.embedding_dim,
            num_attention_heads, dropout=attention_dropout, self_attention=
            True, q_noise=q_noise, qn_block_size=qn_block_size)
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export
            )
        self.fc1 = quant_noise(nn.Linear(self.embedding_dim,
            ffn_embedding_dim), q_noise, qn_block_size)
        self.fc2 = quant_noise(nn.Linear(ffn_embedding_dim, self.
            embedding_dim), q_noise, qn_block_size)
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(self, x: torch.Tensor, self_attn_mask: Optional[torch.
        Tensor]=None, self_attn_padding_mask: Optional[torch.Tensor]=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=
            self_attn_padding_mask, need_weights=False, attn_mask=
            self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


def infer_conv_output_dim(conv_op, input_dim, sample_inchannel):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, sample_inchannel, sample_seq_len, input_dim)
    x = conv_op(x)
    x = x.transpose(1, 2)
    bsz, seq = x.size()[:2]
    per_channel_dim = x.size()[3]
    return x.contiguous().view(bsz, seq, -1).size(-1), per_channel_dim


class VGGBlock(torch.nn.Module):
    """
    VGG motibated cnn module https://arxiv.org/pdf/1409.1556.pdf

    Args:
        in_channels: (int) number of input channels (typically 1)
        out_channels: (int) number of output channels
        conv_kernel_size: convolution channels
        pooling_kernel_size: the size of the pooling window to take a max over
        num_conv_layers: (int) number of convolution layers
        input_dim: (int) input dimension
        conv_stride: the stride of the convolving kernel.
            Can be a single number or a tuple (sH, sW)  Default: 1
        padding: implicit paddings on both sides of the input.
            Can be a single number or a tuple (padH, padW). Default: None
        layer_norm: (bool) if layer norm is going to be applied. Default: False

    Shape:
        Input: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
        Output: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size,
        pooling_kernel_size, num_conv_layers, input_dim, conv_stride=1,
        padding=None, layer_norm=False):
        assert input_dim is not None, 'Need input_dim for LayerNorm and infer_conv_output_dim'
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = _pair(conv_kernel_size)
        self.pooling_kernel_size = _pair(pooling_kernel_size)
        self.num_conv_layers = num_conv_layers
        self.padding = tuple(e // 2 for e in self.conv_kernel_size
            ) if padding is None else _pair(padding)
        self.conv_stride = _pair(conv_stride)
        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(in_channels if layer == 0 else out_channels,
                out_channels, self.conv_kernel_size, stride=self.
                conv_stride, padding=self.padding)
            self.layers.append(conv_op)
            if layer_norm:
                conv_output_dim, per_channel_dim = infer_conv_output_dim(
                    conv_op, input_dim, in_channels if layer == 0 else
                    out_channels)
                self.layers.append(nn.LayerNorm(per_channel_dim))
                input_dim = per_channel_dim
            self.layers.append(nn.ReLU())
        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size,
                ceil_mode=True)
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(
                pool_op, input_dim, out_channels)

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x


class Search(nn.Module):

    def __init__(self, tgt_dict):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.src_lengths = torch.tensor(-1)

    def step(self, step, lprobs, scores):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


EncoderOut = NamedTuple('EncoderOut', [('encoder_out', Tensor), (
    'encoder_padding_mask', Tensor), ('encoder_embedding', Tensor), (
    'encoder_states', Optional[List[Tensor]]), ('src_tokens', Optional[
    Tensor]), ('src_lengths', Optional[Tensor])])


@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None,
        incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, encoder_out=None,
        incremental_state=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[
        str, Optional[Tensor]]], new_order: Tensor):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        seen: Dict[int, Optional[Tensor]] = {}
        for _, module in self.named_modules():
            if hasattr(module, 'reorder_incremental_state'):
                if id(module) not in seen and module is not self:
                    seen[id(module)] = None
                    result = module.reorder_incremental_state(incremental_state
                        , new_order)
                    if result is not None:
                        incremental_state = result

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size'
                    ) and module not in seen:
                    seen.add(module)
                    module.set_beam_size(beam_size)
            self.apply(apply_set_beam_size)
            self._beam_size = beam_size


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""
    incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]]

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        self.single_model = models[0]
        self.models = nn.ModuleList(models)
        self.incremental_states = torch.jit.annotate(List[Dict[str, Dict[
            str, Optional[Tensor]]]], [torch.jit.annotate(Dict[str, Dict[
            str, Optional[Tensor]]], {}) for i in range(self.models_size)])
        self.has_incremental: bool = False
        if all(hasattr(m, 'decoder') and isinstance(m.decoder,
            FairseqIncrementalDecoder) for m in models):
            self.has_incremental = True

    def forward(self):
        pass

    def reset_incremental_state(self):
        if self.has_incremental_states():
            self.incremental_states = torch.jit.annotate(List[Dict[str,
                Dict[str, Optional[Tensor]]]], [torch.jit.annotate(Dict[str,
                Dict[str, Optional[Tensor]]], {}) for i in range(self.
                models_size)])
        return

    def has_encoder(self):
        return hasattr(self.single_model, 'encoder')

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in
            self.models]

    @torch.jit.export
    def forward_decoder(self, tokens, encoder_outs: List[EncoderOut],
        temperature: float=1.0):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(tokens, encoder_out=
                    encoder_out, incremental_state=self.incremental_states[i])
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=
                    encoder_out)
            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]['attn']
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, (-1), :]
            decoder_out_tuple = decoder_out[0][:, -1:, :].div_(temperature
                ), None if decoder_len <= 1 else decoder_out[1]
            probs = model.get_normalized_probs(decoder_out_tuple, log_probs
                =True, sample=None)
            probs = probs[:, (-1), :]
            if self.models_size == 1:
                return probs, attn
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0
            ) - math.log(self.models_size)
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[EncoderOut]],
        new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[EncoderOut] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(model.encoder.reorder_encoder_out(encoder_outs[
                i], new_order))
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(self, new_order):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state(self.incremental_states
                [i], new_order)


class ModelWithSharedParameter(nn.Module):

    def __init__(self):
        super(ModelWithSharedParameter, self).__init__()
        self.embedding = nn.Embedding(1000, 200)
        self.FC1 = nn.Linear(200, 200)
        self.FC2 = nn.Linear(200, 200)
        self.FC2.weight = nn.Parameter(self.FC1.weight)
        self.FC2.bias = nn.Parameter(self.FC1.bias)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.FC2(self.ReLU(self.FC1(input))) + self.FC1(input)


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_freewym_espresso(_paritybench_base):
    pass

    def test_000(self):
        self._check(TransposeLast(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ZeroPad1d(*[], **{'pad_left': 4, 'pad_right': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(AdaptiveSoftmax(*[], **{'vocab_size': 4, 'input_dim': 4, 'cutoff': [4, 4], 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BeamableMM(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})
    @_fails_compile()

    def test_004(self):
        self._check(Highway(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Downsample(*[], **{'index': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Fp32GroupNorm(*[], **{'num_groups': 1, 'num_channels': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_007(self):
        self._check(KmeansVectorQuantizer(*[], **{'dim': 4, 'num_vars': 4, 'groups': 1, 'combine_groups': 1, 'vq_dim': 4, 'time_first': 4}), [torch.rand([4, 4, 4])], {})

    def test_008(self):
        self._check(Fp32LayerNorm(*[], **{'normalized_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(LightweightConv1d(*[], **{'input_size': 4}), [torch.rand([4, 4, 4])], {})
    @_fails_compile()

    def test_010(self):
        self._check(MultiheadAttention(*[], **{'embed_dim': 4, 'num_heads': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_011(self):
        self._check(VGGBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'conv_kernel_size': 4, 'pooling_kernel_size': 4, 'num_conv_layers': 1, 'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(Model(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})
