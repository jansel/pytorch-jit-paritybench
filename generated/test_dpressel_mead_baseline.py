import sys
_module = sys.modules[__name__]
del sys
addons = _module
paired = _module
reader_conllcased = _module
reader_pandas = _module
reader_parallel_classify = _module
reporting_xpctl = _module
vec_text = _module
analyze_calibration = _module
bio_to_iobes = _module
convert_huggingface_checkpoints = _module
iob_to_bio = _module
iob_to_iobes = _module
iobes_result_to_bio = _module
transformer_utils = _module
baseline = _module
bleu = _module
confusion = _module
conlleval = _module
data = _module
embeddings = _module
mime_type = _module
model = _module
progress = _module
pytorch = _module
classify = _module
model = _module
train = _module
lm = _module
model = _module
train = _module
optz = _module
remote = _module
seq2seq = _module
decoders = _module
encoders = _module
model = _module
train = _module
tagger = _module
model = _module
train = _module
torchy = _module
transformer = _module
reader = _module
reporting = _module
services = _module
tensorflow_serving = _module
apis = _module
classification_pb2 = _module
classification_pb2_grpc = _module
get_model_metadata_pb2 = _module
get_model_metadata_pb2_grpc = _module
inference_pb2 = _module
inference_pb2_grpc = _module
input_pb2 = _module
input_pb2_grpc = _module
model_management_pb2 = _module
model_management_pb2_grpc = _module
model_pb2 = _module
model_pb2_grpc = _module
model_service_pb2 = _module
model_service_pb2_grpc = _module
predict_pb2 = _module
predict_pb2_grpc = _module
prediction_service_pb2 = _module
prediction_service_pb2_grpc = _module
regression_pb2 = _module
regression_pb2_grpc = _module
tf = _module
training = _module
datasets = _module
distributed = _module
eager = _module
feed = _module
utils = _module
v1 = _module
v2 = _module
tfy = _module
vectorizers = _module
version = _module
w2v = _module
eight_mile = _module
calibration = _module
metrics = _module
calibration_error = _module
plot = _module
confidence_histogram = _module
reliability_diagram = _module
embeddings = _module
layers = _module
optz = _module
serialize = _module
setup = _module
mead = _module
clean = _module
eval = _module
export = _module
exporters = _module
preprocessors = _module
exporters = _module
tasks = _module
preproc_exporters = _module
trainer = _module
bump = _module
compare_calibrations = _module
download_all = _module
lr_compare = _module
lr_find = _module
lr_visualize = _module
speed_test = _module
report = _module
run = _module
speed_tests = _module
test_bump = _module
test_beam_pytorch = _module
test_beam_tensorflow = _module
test_bleu = _module
test_calc_feats = _module
test_cm = _module
test_conll = _module
test_crf_pytorch = _module
test_crf_tensorflow = _module
test_decay = _module
test_decoders_pytorch = _module
test_decoders_tensorflow = _module
test_embeddings = _module
test_hash_utils = _module
test_iobes = _module
test_label_first_data_utils = _module
test_layers_pytorch = _module
test_layers_tf1 = _module
test_layers_tf2 = _module
test_lr_sched_tf1 = _module
test_lr_sched_tf2 = _module
test_mead_tasks = _module
test_mead_utils = _module
test_model = _module
test_parallel_conv = _module
test_parse_extra_args = _module
test_pytorch_masks = _module
test_pytorch_transformer = _module
test_pytorch_variational_dropout = _module
test_pytorch_weight_sharing = _module
test_read_files = _module
test_readers = _module
test_reporting_hooks = _module
test_rnn_dropout = _module
test_sample = _module
test_tf_ema = _module
test_tf_transformer = _module
test_tf_weight_sharing = _module
test_tlm_serialization = _module
test_torchy = _module
test_transition_masks = _module
test_utils = _module
test_vectorizers = _module

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


import torch


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import logging


import numpy as np


from collections import Counter


from collections import namedtuple


from torch.nn.parallel import DistributedDataParallel


import random


from torch.utils.data.dataset import IterableDataset


from torch.utils.data.dataset import TensorDataset


import torch.backends.cudnn as cudnn


import torch.autograd


import math


from functools import partial


from torch.autograd import Variable


import copy


import torch.nn as nn


from collections import OrderedDict


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch.jit as jit


class TripletLoss(nn.Module):
    """Provide a Triplet Loss using the reversed batch for negatives"""

    def __init__(self, model):
        super().__init__()
        self.score = nn.CosineSimilarity(dim=1)
        self.model = model

    def forward(self, inputs, targets):
        neg = targets.flip(0)
        query = self.model.encode_query(inputs)
        response = self.model.encode_response(targets)
        neg_response = self.model.encode_response(neg)
        pos_score = self.score(query, response)
        neg_score = self.score(query, neg_response)
        score = neg_score - pos_score
        score = score.masked_fill(score < 0.0, 0.0).sum(0)
        return score


def vec_log_sum_exp(vec: torch.Tensor, dim: int) ->torch.Tensor:
    """Vectorized version of log-sum-exp

    :param vec: Vector
    :param dim: What dimension to operate on
    :return:
    """
    max_scores, idx = torch.max(vec, dim, keepdim=True)
    max_scores_broadcast = max_scores.expand_as(vec)
    return max_scores + torch.log(torch.sum(torch.exp(vec -
        max_scores_broadcast), dim, keepdim=True))


class AllLoss(nn.Module):

    def __init__(self, model, warmup_steps=10000):
        """Loss from here https://arxiv.org/pdf/1705.00652.pdf see section 4

        We want to minimize the negative log prob of y given x

        -log P(y|x)

        P(y|x) P(x) = P(x, y)                             Chain Rule of Probability
        P(y|x) = P(x, y) / P(x)                           Algebra
        P(y|x) = P(x, y) / \\sum_\\hat(y) P(x, y = \\hat(y)) Marginalize over all possible ys to get the probability of x
        P_approx(y|x) = P(x, y) / \\sum_i^k P(x, y_k)      Approximate the Marginalization by just using the ys in the batch

        S(x, y) is the score (cosine similarity between x and y in this case) from our neural network
        P(x, y) = e^S(x, y)

        P(y|x) = e^S(x, y) / \\sum_i^k e^S(x, y_k)
        log P(y|x) = log( e^S(x, y) / \\sum_i^k e^S(x, y_k))
        log P(y|x) = S(x, y) - log \\sum_i^k e^S(x, y_k)
        -log P(y|x) = -(S(x, y) - log \\sum_i^k e^S(x, y_k))
        """
        super().__init__()
        self.score = nn.CosineSimilarity(dim=-1)
        self.model = model
        self.max_scale = math.sqrt(self.model.embedding_layers.get_dsz())
        self.steps = 0
        self.warmup_steps = warmup_steps

    def forward(self, inputs, targets):
        fract = min(self.steps / self.warmup_steps, 1)
        c = (self.max_scale - 1) * fract + 1
        self.steps += 1
        query = self.model.encode_query(inputs).unsqueeze(1)
        response = self.model.encode_response(targets).unsqueeze(0)
        all_score = c * self.score(query, response)
        pos_score = torch.diag(all_score)
        loss = pos_score - vec_log_sum_exp(all_score, -1).squeeze()
        loss = torch.sum(loss)
        return -loss


class TwoHeadConcat(nn.Module):
    """Use two parallel SingleHeadReduction, and concatenate the outputs. It is used in the conveRT
    paper (https://arxiv.org/pdf/1911.03688.pdf)"""

    def __init__(self, d_model, dropout, scale=False, d_k=None):
        """Two parallel 1-head self-attention, then concatenate the output
        :param d_model: dim of the self-attention
        :param dropout: dropout of the self-attention
        :param scale: scale fo the self-attention
        :param d_k: d_k of the self-attention
        :return: concatenation of the two 1-head attention
        """
        super().__init__()
        self.reduction1 = SingleHeadReduction(d_model, dropout, scale=scale,
            d_k=d_k)
        self.reduction2 = SingleHeadReduction(d_model, dropout, scale=scale,
            d_k=d_k)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        encoding1 = self.reduction1(x)
        encoding2 = self.reduction2(x)
        x = torch.cat([encoding1, encoding2], dim=-1)
        return x


def parameterize(func):
    """Allow as decorator to be called with arguments, returns a new decorator that should be called with the function to be wrapped."""

    @wraps(func)
    def decorator(*args, **kwargs):
        return lambda x: func(x, *args, **kwargs)
    return decorator


def pytorch_linear(in_sz: int, out_sz: int, unif: float=0, initializer: str
    =None, bias: bool=True):
    """Utility function that wraps a linear (AKA dense) layer creation, with options for weight init and bias"""
    l = nn.Linear(in_sz, out_sz, bias=bias)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    elif initializer == 'ortho':
        nn.init.orthogonal(l.weight)
    elif initializer == 'he' or initializer == 'kaiming':
        nn.init.kaiming_uniform(l.weight)
    else:
        nn.init.xavier_uniform_(l.weight)
    if bias:
        l.bias.data.zero_()
    return l


def optional_params(func):
    """Allow a decorator to be called without parentheses if no kwargs are given.

    parameterize is a decorator, function is also a decorator.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        """If a decorator is called with only the wrapping function just execute the real decorator.
           Otherwise return a lambda that has the args and kwargs partially applied and read to take a function as an argument.

        *args, **kwargs are the arguments that the decorator we are parameterizing is called with.

        the first argument of *args is the actual function that will be wrapped
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)
    return wrapped


class ArcPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, hsz, beam_width=1):
        pass


def repeat_batch(t, K, dim=0):
    """Repeat a tensor while keeping the concept of a batch.

    :param t: `torch.Tensor`: The tensor to repeat.
    :param K: `int`: The number of times to repeat the tensor.
    :param dim: `int`: The dimension to repeat in. This should be the
        batch dimension.

    :returns: `torch.Tensor`: The repeated tensor. The new shape will be
        batch size * K at dim, the rest of the shapes will be the same.

    Example::

        >>> a = torch.arange(10).view(2, -1)
        >>> a
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> a.repeat(2, 1)
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> repeat_batch(a, 2)
	tensor([[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[5, 6, 7, 8, 9]])
    """
    shape = t.shape
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = t.unsqueeze(dim + 1).repeat(tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim + 1:])
    return tiled.view(new_shape)


TransformerEncoderOutput = namedtuple('TransformerEncoderOutput', ('output',
    'src_mask'))


RNNEncoderOutput = namedtuple('RNNEncoderOutput', ('output', 'hidden',
    'src_mask'))


def _cat_dir(h: torch.Tensor) ->torch.Tensor:
    """Concat forward and backword state vectors.

    The shape of the hidden is `[#layers * #dirs, B, H]`. The docs say you can
    separate directions with `h.view(#l, #dirs, B, H)` with the forward dir being
    index 0 and backwards dir being 1.

    This means that before separating with the view the forward dir are the even
    indices in the first dim while the backwards dirs are the odd ones. Here we select
    the even and odd values and concatenate them

    :param h: The hidden shape as it comes back from PyTorch modules
    """
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=-1)


def concat_state_dirs(state):
    """Convert the bidirectional out of an RNN so the forward and backward values are a single vector."""
    if isinstance(state, tuple):
        return tuple(_cat_dir(h) for h in state)
    return _cat_dir(state)


def sequence_mask(lengths: torch.Tensor, max_len: int=-1) ->torch.Tensor:
    """Generate a sequence mask of shape `BxT` based on the given lengths

    :param lengths: A `B` tensor containing the lengths of each example
    :param max_len: The maximum width (length) allowed in this mask (default to None)
    :return: A mask
    """
    lens = lengths.cpu()
    if max_len < 0:
        max_len_v = torch.max(lens)
    else:
        max_len_v = max_len
    row = torch.arange(0, max_len_v).type_as(lens).view(1, -1)
    col = lens.view(-1, 1)
    mask = row < col
    return mask


BASELINE_SEQ2SEQ_ENCODERS = {}


def unsort_batch(batch: torch.Tensor, perm_idx: torch.Tensor) ->torch.Tensor:
    """Undo the sort on a batch of tensors done for packing the data in the RNN.

    :param batch: The batch of data batch first `[B, ...]`
    :param perm_idx: The permutation index returned from the torch.sort.

    :returns: The batch in the original order.
    """
    perm_idx = perm_idx.to(batch.device)
    diff = len(batch.shape) - len(perm_idx.shape)
    extra_dims = [1] * diff
    perm_idx = perm_idx.view([-1] + extra_dims)
    return batch.scatter_(0, perm_idx.expand_as(batch), batch)


TensorDef = torch.Tensor


class SequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss, avg='token'):
        super(SequenceCriterion, self).__init__()
        if avg == 'token':
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=True)
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=False)
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)


class PyTorchEmbeddings(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    @property
    def output_dim(self):
        return self.get_dsz()

    def encode(self, x):
        return self(x)


class PositionalMixin(nn.Module):
    """A Mixin that provides functionality to generate positional embeddings to be added to the normal embeddings.

    Note, mixins need to be before the base case when used, i.e.
        `Embedding(Mixin, BaseEmbed)` NOT `Embedding(BaseEmbed, Mixin)`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def positional(self, length):
        pass

    def extra_repr(self):
        return f'mxlen={self.mxlen}'


class VariationalDropout(nn.Module):
    """Inverted dropout that applies the same mask at each time step."""

    def __init__(self, pdrop: float=0.5, batch_first: bool=False):
        """Variational Dropout

        :param pdrop: the percentage to drop
        """
        super().__init__()
        self.pdrop = pdrop
        self.batch_first = batch_first

    def extra_repr(self):
        return 'p=%.1f' % self.pdrop

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if not self.training:
            return input
        if self.batch_first:
            dim0 = input.size(0)
            dim1 = 1
        else:
            dim0 = 1
            dim1 = input.size(1)
        mask = torch.zeros(dim0, dim1, input.size(2)).bernoulli_(1 - self.pdrop
            ).to(input.device)
        mask = mask / self.pdrop
        return mask * input


class SequenceLoss(nn.Module):
    """Computes the loss over a sequence"""

    def __init__(self, LossFn: nn.Module=nn.NLLLoss, avg: str='token'):
        """A class that applies a Loss function to sequence via the folding trick.

        :param LossFn: A loss function to apply (defaults to `nn.NLLLoss`)
        :param avg: A divisor to apply, valid values are `token` and `batch`
        """
        super().__init__()
        self.avg = avg
        if avg == 'token':
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction='mean')
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction='sum')
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor
        ) ->torch.Tensor:
        """Evaluate some loss over a sequence.
        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.
        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)

    def extra_repr(self):
        return f'reduction={self.avg}'


def tensor_and_lengths(inputs) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return either the unpacked inputs (2), or a `Tuple` of the input with None

    TODO: this function should probably be changed to always return the lengths second.
    To do this, we just need a sentinel value, e.g. <PAD> (0).  The problem with doing this is
    that it might be possible to generate <PAD> in the middle of the tensor which would make that
    length invalid.

    :param inputs: Either a sequence of the `(tensor, length)` or just the `tensor`
    :return: A `Tuple` of `(tensor, length)` or `(tensor, None)`
    """
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None
    return in_tensor, lengths


class MeanPool1D(nn.Module):
    """Do a mean pool while accounting for the length of a sequence
    """

    def __init__(self, outsz, batch_first=True):
        """Set up pooling module

        :param outsz: The output dim, for dowstream access
        :param batch_first: Is this module batch first or time first?
        """
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz
        self.requires_length = True

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]
        ) ->torch.Tensor:
        """Apply mean pooling on the valid inputs

        :param inputs: A tuple of `(input, lengths)`
        :return: Pooled output
        """
        tensor, lengths = tensor_and_lengths(inputs)
        return torch.sum(tensor, self.reduction_dim, keepdim=False
            ) / torch.unsqueeze(lengths, -1).to(tensor.dtype).to(tensor.device)

    def extra_repr(self):
        return f'batch_first={self.batch_first}'


MASK_FALSE = False


def bth2tbh(t: torch.Tensor) ->torch.Tensor:
    """Transpose the first 2 dims"""
    return t.transpose(0, 1).contiguous()


class MaxPool1D(nn.Module):
    """Do a max-pooling operation with or without a length given
    """

    def __init__(self, outsz, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch
        .Tensor]]) ->torch.Tensor:
        """If we are given a tuple as input, we will use the length, otherwise we will do an operation without masking

        :param inputs: either a tuple of `(input, lengths)` or a tensor `input`
        :return: A pooled tensor
        """
        tensor, lengths = tensor_and_lengths(inputs)
        if lengths is not None:
            mask = sequence_mask(lengths).to(tensor.device)
            mask = mask if self.batch_first else bth2tbh(mask)
            tensor = tensor.masked_fill(mask.unsqueeze(-1) == MASK_FALSE, -
                10000.0)
        dmax, _ = torch.max(tensor, self.reduction_dim, keepdim=False)
        return dmax

    def extra_repr(self) ->str:
        return f'batch_first={self.batch_first}'


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Conv1DSame(nn.Module):
    """Perform a 1D convolution with output size same as input size

    To make this operation work as expected, we cannot just use `padding=kernel_size//2` inside
    of the convolution operation.  Instead, we zeropad the input using the `ConstantPad1d` module

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size:
        int, bias: bool=True):
        """Create a 1D conv to produce the same output size as input

        :param in_channels: The number of input feature maps
        :param out_channels: The number of output feature maps
        :param kernel_size: The kernel size
        :param bias: Is bias on?
        """
        super().__init__()
        start_pad = kernel_size // 2
        end_pad = start_pad - 1 if kernel_size % 2 == 0 else start_pad
        self.conv = nn.Sequential(nn.ConstantPad1d((start_pad, end_pad), 
            0.0), nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Do convolution1d on an input tensor, `[B, C, T]`

        :param x: The input tensor of shape `[B, C, T]`
        :return: The output tensor of shape `[B, H, T]`
        """
        return self.conv(x)


def get_activation(name: str='relu') ->nn.Module:
    """Get back an `nn.Module` by string name of the activation operator

    :param name: A string name of the operation
    :return: A module associated with that string
    """
    if name is None or name == 'ident':
        return nn.Identity()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'gelu':
        return GeLU()
    if name == 'hardtanh':
        return nn.Hardtanh()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'log_sigmoid':
        return nn.LogSigmoid()
    if name == 'log_softmax':
        return nn.LogSoftmax(dim=-1)
    if name == 'softmax':
        return nn.Softmax(dim=-1)
    return nn.ReLU()


class ConvEncoder(nn.Module):
    """1D Convolutional layer encoder with given activation function, optional dropout

    This module takes in a temporal signal of either shape `[B, C, T]` or `[B, T, C]`, depending on the constructor
    and produces an output signal of the same orientation (`[B, H, T]` or `[B, T, H]`, respectively).  We default
    to `[B, T, H]` orientation to make it more convenient for typical layout, but this requires transposing the last
    2 dims before and after the convolution operation.

    """

    def __init__(self, insz: int, outsz: int, filtsz: int, pdrop: float=0.0,
        activation: str='relu', hidden_last=True):
        """Construct the encoder with optional dropout, given activation, and orientation

        :param insz: The number of input feature maps
        :param outsz: The number of output feature maps (or hidden size)
        :param filtsz: The kernel size
        :param pdrop: The amount of dropout to apply, this defaults to 0
        :param activation: The activation function by name, defaults to `relu`
        :param hidden_last: PyTorch only! If `True` the orientatiation is `[B, T, H]`, o.w. `[B, H, T]` expected
        """
        super().__init__()
        self.output_dim = outsz
        conv = Conv1DSame(insz, outsz, filtsz)
        act = get_activation(activation)
        dropout = nn.Dropout(pdrop)
        if hidden_last:
            self.conv = nn.Sequential(BTH2BHT(), conv, act, dropout, BHT2BTH())
        else:
            self.conv = nn.Sequential(conv, act, dropout)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return self.conv(input)


class ConvEncoderStack(nn.Module):
    """Create a stack of convolutional encoders with residual connections between, using the `ConvEncoder` underneath

    This creates an encoder stack of convolutions, finally returning the last temporal output.  Each layer uses zero-padding
    which causes the output of the convolution at each layer to be the same length.

    As in the `ConvEncoder` we support input tensor shapes of `[B, C, T]` or `[B, T, C]` depending on the constructor
    initialization, and transpose underneath the input and output of the stack if the orientation is defaulted to
    `[B, T, C]`
    """

    def __init__(self, insz: int, outsz: int, filtsz: int, nlayers: int=1,
        pdrop: float=0.0, activation: str='relu', hidden_last=True):
        """Construct the encoder stack

        :param insz: The input number of feature maps
        :param outsz: The output number of feature maps
        :param filtsz: The kernel size
        :param nlayers: The number of layers in the stack (defaults to a single layer)
        :param pdrop: The amount of dropout to apply (defaults to `0`)
        :param activation: The activation function to use as a string, defaults to `relu`
        :param hidden_last: PyTorch only! If `True` the orientatiation is `[B, T, H]`, o.w. `[B, H, T]` expected
        """
        super().__init__()
        if hidden_last:
            first_layer = nn.Sequential(BTH2BHT(), ConvEncoder(insz, outsz,
                filtsz, pdrop, activation, hidden_last=False))
        else:
            first_layer = ConvEncoder(insz, outsz, filtsz, pdrop,
                activation, hidden_last=False)
        subsequent_layer = ResidualBlock(ConvEncoder(outsz, outsz, filtsz,
            pdrop, activation, hidden_last=False))
        self.layers = nn.ModuleList([first_layer] + [copy.deepcopy(
            subsequent_layer) for _ in range(nlayers - 1)])
        if hidden_last:
            self.layers.append(BHT2BTH())
        self.output_dim = outsz

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Apply a stack of 1D convolutions with residual connections between them

        :param input: A tensor of shape `[B, T, C]` or `[B, C, T]` depending on value of `hidden_last`
        :return: A tensor of shape `[B, T, H]` or `[B, H, T]` depending on the value of `hidden_last`
        """
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


def bth2bht(t: torch.Tensor) ->torch.Tensor:
    """Transpose the 2nd and 3rd dim of a tensor"""
    return t.transpose(1, 2).contiguous()


class BTH2BHT(nn.Module):
    """Utility layer to convert from `[B, T, H]` to `[B, H, T]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return bth2bht(t)


def tbh2bht(t: torch.Tensor) ->torch.Tensor:
    """Permute the dimensions, first goes to third, second goes to first, last moves to second"""
    return t.permute(1, 2, 0).contiguous()


class TBH2BHT(nn.Module):
    """Utility layer to convert from `[T, B, H]` to `[B, H, T]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return tbh2bht(t)


def tbh2bth(t: torch.Tensor) ->torch.Tensor:
    """Transpose the first 2 dims"""
    return t.transpose(0, 1).contiguous()


class TBH2BTH(nn.Module):
    """Utility layer to convert from `[T, B, H]` to `[B, T, H]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return tbh2bth(t)


class BTH2TBH(nn.Module):
    """Utility layer to convert from `[B, T, H]` to `[T, B, H]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return bth2tbh(t)


def bht2bth(t: torch.Tensor) ->torch.Tensor:
    return t.transpose(1, 2).contiguous()


class BHT2BTH(nn.Module):
    """Utility layer to convert from `[B, H, T]` to `[B, T, H]`
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) ->torch.Tensor:
        return bht2bth(t)


class ParallelConv(nn.Module):
    """Layer of parallel convolutions with varying filter sizes followed by max over time pooling

    This module takes an input tensor of any orientation based on its constructor, and pools its
    output to shape `[B, H]`, where `H` is `outsz * len(filtsz)`
    """

    def __init__(self, insz: int, outsz: int, filtsz: List[int], activation:
        str='relu', input_fmt: str='bth'):
        """
        Constructor for a parallel convolution from any orientation tensor input

        :param insz: The number of input feature maps
        :param outsz: The number of output feature maps
        :param filtsz: The kernel size as a list of parallel filters to apply, e.g. `[3, 4, 5]`
        :param activation: An activation function by name to apply
        :param input_fmt: A string for the orientation.  Valid values are `bth` or `btc` meaning hidden units last,
        `bht` or `bct` meaning the temporal dim last or `tbh` or `tbc` meaning the hidden units last and the temporal dim
        first
        """
        super().__init__()
        self.requires_length = False
        convs = []
        outsz_filts = outsz
        self.input_fmt = input_fmt.lower()
        if type(outsz) == int:
            outsz_filts = len(filtsz) * [outsz]
        self.output_dim = sum(outsz_filts)
        for i, fsz in enumerate(filtsz):
            pad = fsz // 2
            conv = nn.Sequential(nn.Conv1d(insz, outsz_filts[i], fsz,
                padding=pad), get_activation(activation))
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def transform_input(self, t: torch.Tensor) ->torch.Tensor:
        if self.input_fmt == 'bth' or self.input_fmt == 'btc':
            return bth2bht(t)
        elif self.input_fmt == 'tbh' or self.input_fmt == 'tbc':
            return tbh2bht(t)
        else:
            return t

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        """Transform the input to `[B, C, T]` from any orientation and perform parallel 1D convs and max over time pool

        :param inputs: An input tensor of any format specified in the constructor
        :return: A `[B, H]` tensor representing the pooled outputs
        """
        mots = []
        input_bct = self.transform_input(inputs)
        for conv in self.convs:
            conv_out = conv(input_bct)
            mot, _ = conv_out.max(2)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return mots


class Highway(nn.Module):
    """Highway layer as defined in https://arxiv.org/abs/1505.00387

    """

    def __init__(self, input_size: int, **kwargs):
        """Highway layer constructor

        :param input_size: The input hidden size
        :param kwargs:
        """
        super().__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)
        self.output_dim = input_size

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Take a tensor in and produce the highway layer output

        :param input: Input tensor
        :return: output tensor
        """
        proj_result = torch.relu(self.proj(input))
        proj_gate = torch.sigmoid(self.transform(input))
        gated = proj_gate * proj_result + (1 - proj_gate) * input
        return gated


class StackedLSTMCell(nn.Module):
    """A stacked LSTM cells applied at a timestep
    """

    def __init__(self, num_layers: int, input_size: int, rnn_size: int,
        dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size=input_size,
                hidden_size=rnn_size, bias=False))
            input_size = rnn_size

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        """Apply a stack of LSTMs

        :param input: The input to the first LSTM `[B, H]`
        :param hidden: The previous `(h, c)` where `h=(h_0, h_1,..)`, `c=(c_0, c_1,..)`
        :return: The output and hidden `(h, c)` where `h=(h_0, h_1,..)`, `c=(c_0, c_1,..)`
        """
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs.append(h_i)
            cs.append(c_i)
        hs = torch.stack(hs)
        cs = torch.stack(cs)
        return input, (hs, cs)


class StackedGRUCell(nn.Module):
    """A stacked GRU cells applied at a timestep
    """

    def __init__(self, num_layers: int, input_size: int, rnn_size: int,
        dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size=input_size,
                hidden_size=rnn_size))
            input_size = rnn_size

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) ->Tuple[
        torch.Tensor, torch.Tensor]:
        """Apply a stack of GRUs

        :param input: The input to the first LSTM `[B, H]`
        :param hidden: The previous `h` where `h=(h_0, h_1,..)`
        :return: The output and hidden `h` where `h=(h_0, h_1,..)`
        """
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            h_i = layer(input, h_0[i])
            input = h_i
            if i != self.num_layers:
                input = self.dropout(input)
            hs.append(h_i)
        hs = torch.stack(hs)
        return input, hs


class Dense(nn.Module):
    """Dense (Linear) layer with optional activation given

    This module is the equivalent of the tf.keras.layer.Dense, module with optional activations applied
    """

    def __init__(self, insz: int, outsz: int, activation: Optional[str]=
        None, unif: float=0, initializer: Optional[str]=None):
        """Constructor for "dense" or "linear" layer, with optional activation applied

        :param insz: The number of hidden units in the input
        :param outsz: The number of hidden units in the output
        :param activation: The activation function by name, defaults to `None`, meaning no activation is applied
        :param unif: An optional initialization value which can set the linear weights.  If given, biases will init to 0
        :param initializer: An initialization scheme by string name: `ortho`, `kaiming` or `he`, `xavier` or `glorot`
        """
        super().__init__()
        self.layer = pytorch_linear(insz, outsz, unif, initializer)
        self.activation = get_activation(activation)
        self.output_dim = outsz

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Run a linear projection over the input, followed by an optional activation given by constructor

        :param input: the input tensor
        :return: the transformed output
        """
        return self.activation(self.layer(input))


class WeightTieDense(nn.Module):
    """Do weight tying from the input parameter
    """

    def __init__(self, tie: nn.Module, bias=False):
        super().__init__()
        self.tie = tie
        self.weight, self.transform = self._get_weight(tie)
        if bias:
            bias = torch.nn.Parameter(torch.zeros(self.transform(self.
                weight).shape[0]))
        else:
            bias = None
        self.register_parameter('bias', bias)

    def _get_weight(self, tie: nn.Module):
        emb = getattr(tie, 'embeddings', None)
        if emb is not None:
            return getattr(emb, 'weight'), self._identity
        return getattr(tie, 'weight'), self._transpose

    def _identity(self, x: torch.Tensor) ->torch.Tensor:
        return x

    def _transpose(self, x: torch.Tensor) ->torch.Tensor:
        return x.transpose(0, 1).contiguous()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return F.linear(input, self.transform(self.weight), self.bias)


class ResidualBlock(nn.Module):
    """Create a residual block by wrapping an layer with a residual connection"""

    def __init__(self, layer: Optional[nn.Module]=None, **kwargs):
        """Wrap an layer with a residual connection

        :param layer: This layer will be applied to the input and added to the input
        :param kwargs:
        """
        super().__init__()
        self.layer = layer
        if self.layer is not None and hasattr(layer, 'output_dim'):
            self.output_dim = layer.output_dim

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """Apply a residual block

        :param input: A tensor to use as input and to add to output
        :return: The residual connection output
        """
        return input + self.layer(input)


class LSTMEncoderBase(nn.Module):
    """The LSTM encoder is a base for a set of encoders producing various outputs.

    All LSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `LSTMEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `LSTMEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0,
        requires_length: bool=True, batch_first: bool=False, unif: float=0,
        initializer: str=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: PyTorch only! Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop,
            bidirectional=False, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: Tuple[torch.Tensor, torch.Tensor]
        ) ->List[torch.Tensor]:
        """Get a view of the top state of shape [B, H]`

        :param state:
        :return:
        """
        top = []
        for s in state:
            top.append(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0])
        return top


class LSTMEncoderWithState(nn.Module):
    """LSTM encoder producing the hidden state and the output, where the input doesnt require any padding

    PyTorch note: This type of encoder doesnt inherit the `LSTMEncoderWithState` base
    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0,
        batch_first: bool=False, unif: float=0, initializer: str=None, **kwargs
        ):
        """
        :param insz: The size of the input
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param batch_first: PyTorch only! do batch first or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN

        """
        super().__init__()
        self.requires_length = False
        self.requires_state = True
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop,
            bidirectional=False, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def forward(self, input_and_prev_h: Tuple[torch.Tensor, torch.Tensor]
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        :param input_and_prev_h: The input at this timestep and the previous hidden unit or `None`
        :return: Raw `torch.nn.LSTM` output
        """
        inputs, hidden = input_and_prev_h
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden


class BiLSTMEncoderBase(nn.Module):
    """BiLSTM encoder base for a set of encoders producing various outputs.

    All BiLSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `BiLSTMEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `BiLSTMEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0,
        requires_length: bool=True, batch_first: bool=False, unif: float=0,
        initializer: str=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz // 2, nlayers, dropout=pdrop,
            bidirectional=True, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state):
        return tuple(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0] for
            s in state)


class GRUEncoderBase(nn.Module):
    """The GRU encoder is a base for a set of encoders producing various outputs.

    All GRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `GRUEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `GRUEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0,
        requires_length: bool=True, batch_first: bool=False, unif: float=0,
        initializer: str=None, **kwargs):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: PyTorch only! Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=pdrop,
            bidirectional=False, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal_(self.rnn.weight_ih_l0)
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.rnn.weight_ih_l0)
            nn.init.kaiming_uniform_(self.rnn.weight_hh_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: torch.Tensor) ->torch.Tensor:
        return state[-1]


class BiGRUEncoderBase(nn.Module):
    """BiGRU encoder base for a set of encoders producing various outputs.

    All BiGRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.

    *PyTorch Note*: In PyTorch, its more common for the input shape to be temporal length first (`[T, B, H]`) and this
    is the PyTorch default.  There is an extra parameter in all of these models called `batch_first` which controls this.
    Currently, the default is time first (`batch_first=False`), which differs from TensorFlow.  To match the TF impl,
    set `batch_first=True`.

    *PyTorch Note*:
    Most `BiGRUEncoder` variants just define the `forward`.  This module cannot provide the same utility as the
    TensorFlow `BiGRUEncoder` base right now, because because the JIT isnt handling subclassing of forward properly.

    """

    def __init__(self, insz: int, hsz: int, nlayers: int, pdrop: float=0.0,
        requires_length: bool=True, batch_first: bool=False, unif: float=0,
        initializer: str=None, **kwargs):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input
        :param hsz: The number of hidden units per BiGRU (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiGRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param batch_first: Should we do batch first input or time-first input? Defaults to `False` (differs from TF!)
        :param unif: PyTorch only! Initialization parameters for RNN
        :param initializer: PyTorch only! A string describing optional initialization type for RNN
        """
        super().__init__()
        self.requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.GRU(insz, hsz // 2, nlayers, dropout=pdrop,
            bidirectional=True, batch_first=batch_first)
        if initializer == 'ortho':
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == 'he' or initializer == 'kaiming':
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_dim = hsz

    def extract_top_state(self, state: torch.Tensor) ->torch.Tensor:
        return state[-1]


class Reduction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]) ->torch.Tensor:
        pass


class SumLayerNormReduction(Reduction):

    def __init__(self, output_dims: List[int], layer_norm_eps: float=1e-12):
        super().__init__()
        self.output_dim = output_dims[0]
        self.ln = nn.LayerNorm(self.output_dim, eps=layer_norm_eps)

    def forward(self, inputs: List[torch.Tensor]) ->torch.Tensor:
        output = sum(inputs)
        return self.ln(output)


class SumReduction(Reduction):

    def __init__(self, output_dims: List[int]):
        super().__init__()
        self.output_dim = output_dims[0]

    def forward(self, inputs: List[torch.Tensor]) ->torch.Tensor:
        return sum(inputs)


class EmbeddingsStack(nn.Module):

    def __init__(self, embeddings_dict: Dict[str, nn.Embedding],
        dropout_rate: float=0.0, requires_length: bool=False, reduction:
        Optional[Union[str, nn.Module]]='concat', **kwargs):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings
        :param embeddings_dict: dictionary of each feature embedding
        :param dropout_rate: The dropout rate (0.0 means no dropout, 1.0 means complete)
        """
        super().__init__()
        self._keys: List[str] = []
        embeddings_list = []
        output_dims = []
        for k, embedding in embeddings_dict.items():
            embeddings_list.append(embedding)
            self._keys.append(k)
            output_dims += [embedding.get_dsz()]
        self.embeddings: nn.ModuleList = nn.ModuleList(embeddings_list)
        if isinstance(reduction, str):
            if reduction == 'sum':
                self.reduction = SumReduction(output_dims)
            elif reduction == 'sum-layer-norm':
                self.reduction = SumLayerNormReduction(output_dims,
                    layer_norm_eps=kwargs.get('layer_norm_eps', 1e-12))
            else:
                self.reduction = ConcatReduction(output_dims)
        else:
            self.reduction = reduction
        self.dsz = self.reduction.output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.requires_length = requires_length

    def __getitem__(self, item: str) ->nn.Module:
        idx = self._keys.index(item)
        if idx < 0:
            raise Exception(f'Invalid item ({item})')
        return self.embeddings[idx]

    def forward(self, inputs: Dict[str, torch.Tensor]) ->torch.Tensor:
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings
        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        i = 0
        for embedding in self.embeddings:
            k = self._keys[i]
            x = inputs[k]
            embeddings_out = embedding(x)
            all_embeddings_out.append(embeddings_out)
            i += 1
        word_embeddings = self.reduction(all_embeddings_out)
        return self.dropout(word_embeddings)

    def keys(self):
        return self._keys

    @property
    def output_dim(self):
        return self.dsz

    def items(self):
        for k, v in zip(self.keys(), self.embeddings):
            yield k, v


def TRAIN_FLAG():
    """Create a global training flag on first use"""
    global BASELINE_TF_TRAIN_FLAG
    if BASELINE_TF_TRAIN_FLAG is not None:
        return BASELINE_TF_TRAIN_FLAG
    BASELINE_TF_TRAIN_FLAG = tf.compat.v1.placeholder_with_default(False,
        shape=(), name='TRAIN_FLAG')
    return BASELINE_TF_TRAIN_FLAG


class VectorSequenceAttention(nn.Module):

    def __init__(self, hsz: int):
        super().__init__()
        self.hsz = hsz
        self.W_c = nn.Linear(2 * self.hsz, hsz, bias=False)

    def forward(self, query_t, keys_bth, values_bth, keys_mask=None):
        a = self._attention(query_t, keys_bth, keys_mask)
        attended = self._update(a, query_t, values_bth)
        return attended

    def _attention(self, query_t, keys_bth, keys_mask):
        pass

    def _update(self, a, query_t, values_bth):
        a = a.view(a.size(0), 1, a.size(1))
        c_t = torch.bmm(a, values_bth).squeeze(1)
        attended = torch.cat([c_t, query_t], -1)
        attended = torch.tanh(self.W_c(attended))
        return attended


class FineTuneModel(nn.Module):

    def __init__(self, nc, embeddings, stack_model=None):
        super().__init__()
        if isinstance(embeddings, dict):
            self.finetuned = EmbeddingsStack(embeddings)
        else:
            self.finetuned = embeddings
        self.stack_model = stack_model
        output_dim = (self.finetuned.output_dim if stack_model is None else
            stack_model.output_dim)
        self.output_layer = Dense(output_dim, nc, activation='log_softmax')

    def forward(self, inputs):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers
            ) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)


class CompositePooling(nn.Module):
    """Composite pooling allows for multiple sub-modules during pooling to be used in parallel
    """

    def __init__(self, models):
        """
        Note, this currently requires that each submodel is an eight_mile model with an `output_dim` attr
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.output_dim = sum(m.output_dim for m in self.models)
        self.requires_length = any(getattr(m, 'requires_length', False) for
            m in self.models)

    def forward(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        pooled = []
        for sub_model in self.models:
            if getattr(sub_model, 'requires_length', False):
                pooled.append(sub_model((inputs, lengths)))
            else:
                pooled.append(sub_model(inputs))
        return torch.cat(pooled, -1)


class EmbedPoolStackModel(nn.Module):
    """This provides an idiom for classification consisting of multiple phases

    In the first phase, we embed the input tensors, and subsequently pool them to
    a fixed width representation.  Finally, we allow multiple hidden "stacking"
    layers, ultimately ending in a projection to the output space

    """

    def __init__(self, nc: int, embeddings: nn.Module, pool_model: nn.
        Module, stack_model: Optional[nn.Module]=None, output_model:
        Optional[nn.Module]=None):
        super().__init__()
        self.embed_model = embeddings
        self.pool_model = pool_model
        self.stack_model = stack_model if stack_model else nn.Identity()
        output_dim = (self.pool_model.output_dim if stack_model is None else
            stack_model.output_dim)
        self.output_layer = Dense(output_dim, nc, activation='log_softmax'
            ) if output_model is None else output_model

    def forward(self, inputs: Dict[str, torch.Tensor]):
        lengths = inputs['lengths']
        embedded = self.embed_model(inputs)
        embedded = embedded, lengths
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


class PassThru(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        return inputs


class WithoutLength(nn.Module):
    """Wrapper layer to remove lengths from the input
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.output_dim = self.layer.output_dim if hasattr(self.layer,
            'output_dim') else 0

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]
        ) ->torch.Tensor:
        return self.layer(inputs[0])


class WithDropout(nn.Module):
    """Wrapper for any layer that surrounds it with dropout"""

    def __init__(self, layer: nn.Module, pdrop: float=0.5, variational=False):
        """Create a dropout wrapper around the given layer

        :param layer: Some sort of layer
        :param pdrop: A dropout value
        """
        super().__init__()
        self.layer = layer
        self.dropout = VariationalDropout(pdrop
            ) if variational else nn.Dropout(pdrop)
        self.output_dim = self.layer.output_dim if hasattr(self.layer,
            'output_dim') else 0

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        """Apply the layer followed by dropout

        :param inputs: input tensor
        :return: output transformed by the held layer and subsequent dropout
        """
        return self.dropout(self.layer(inputs))


class WithDropoutOnFirst(nn.Module):
    """Wrapper for any layer that surrounds it with dropout

    This exists primarily for the LSTMEncoderWithState to allow dropout on the output while
    passing back the hidden state
    """

    def __init__(self, layer: nn.Module, pdrop: float=0.5, variational=False):
        """Create a dropout wrapper around the given layer

        :param layer: Some sort of layer
        :param pdrop: A dropout value
        """
        super().__init__()
        self.layer = layer
        self.dropout = VariationalDropout(pdrop
            ) if variational else nn.Dropout(pdrop)
        self.output_dim = self.layer.output_dim if hasattr(self.layer,
            'output_dim') else 0

    def forward(self, inputs: Tuple[torch.Tensor]) ->torch.Tensor:
        """Apply the layer followed by dropout

        :param inputs: input tensor
        :return: output transformed by the held layer and subsequent dropout
        """
        outputs = self.layer(inputs)
        return self.dropout(outputs[0]), outputs[1]


@torch.jit.script
def script_viterbi(unary: torch.Tensor, trans: torch.Tensor, start_idx: int,
    end_idx: int) ->Tuple[torch.Tensor, torch.Tensor]:
    seq_len: int = unary.size(0)
    num_tags: int = unary.size(1)
    fill_value: float = -10000.0
    alphas = torch.full((num_tags,), fill_value, dtype=unary.dtype, device=
        unary.device)
    broadcast_idx = torch.full((num_tags,), start_idx, dtype=torch.long)
    alphas.scatter_(0, broadcast_idx, torch.zeros((num_tags,)))
    alphas.unsqueeze_(0)
    backpointers: torch.Tensor = torch.zeros(num_tags, dtype=torch.long
        ).unsqueeze(0)
    for i in range(seq_len):
        unary_t = unary[(i), :]
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 1)
        backpointers = torch.cat([backpointers, best_tag_ids.unsqueeze(0)], 0)
        alphas = (viterbi + unary_t).unsqueeze(0)
    terminal_vars = alphas.squeeze(0) + trans[(end_idx), :]
    path_score, best_tag_id = torch.max(terminal_vars, 0)
    best_path = best_tag_id.unsqueeze(0)
    for i in range(unary.size(0)):
        t = seq_len - i - 1
        best_tag_id = backpointers[t + 1, best_tag_id]
        best_path = torch.cat([best_path, best_tag_id.unsqueeze(0)], -1)
    new_path_vec = best_path.flip(0)
    return new_path_vec[1:], path_score


class ViterbiBatchSize1(nn.Module):

    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, _: torch.Tensor
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        unary = unary.squeeze(1)
        trans = trans.squeeze(0)
        path, score = script_viterbi(unary, trans, self.start_idx, self.end_idx
            )
        return path.unsqueeze(1), score


class Viterbi(nn.Module):

    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, lengths:
        torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param trans: torch.FloatTensor: [1, N, N]
        :param norm: Callable: This function should take the initial and a dim to
            normalize along.

        :return: torch.LongTensor: [T, B] the padded paths
        :return: torch.FloatTensor: [B] the path scores
        """
        seq_len, batch_size, tag_size = unary.size()
        min_length = torch.min(lengths)
        backpointers = []
        alphas = torch.full((batch_size, 1, tag_size), -10000.0, device=
            unary.device)
        alphas[:, (0), (self.start_idx)] = 0
        for i, unary_t in enumerate(unary):
            next_tag_var = alphas + trans
            viterbi, best_tag_ids = torch.max(next_tag_var, 2)
            backpointers.append(best_tag_ids)
            new_alphas = viterbi + unary_t
            new_alphas.unsqueeze_(1)
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(
                    mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas
        terminal_var = alphas.squeeze(1) + trans[:, (self.end_idx), :]
        path_score, best_tag_id = torch.max(terminal_var, 1)
        rev_len = seq_len - lengths - 1
        best_path = [best_tag_id]
        for i in range(len(backpointers)):
            t = len(backpointers) - i - 1
            backpointer_t = backpointers[t]
            new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)
                ).squeeze(1)
            mask = i > rev_len
            best_tag_id = best_tag_id.masked_fill(mask, 0
                ) + new_best_tag_id.masked_fill(mask == MASK_FALSE, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        seq_mask = sequence_mask(lengths, seq_len).to(best_path.device
            ).transpose(0, 1)
        best_path = best_path.masked_fill(seq_mask == MASK_FALSE, 0)
        return best_path, path_score


def ident(x):
    return x


class ViterbiLogSoftmaxNorm(Viterbi):

    def forward(self, unary: torch.Tensor, trans: torch.Tensor, lengths:
        torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param trans: torch.FloatTensor: [1, N, N]
        :param norm: Callable: This function should take the initial and a dim to
            normalize along.

        :return: torch.LongTensor: [T, B] the padded paths
        :return: torch.FloatTensor: [B] the path scores
        """
        seq_len, batch_size, tag_size = unary.size()
        min_length = torch.min(lengths)
        backpointers = []
        alphas = torch.full((batch_size, 1, tag_size), -10000.0, device=
            unary.device)
        alphas[:, (0), (self.start_idx)] = 0
        alphas = F.log_softmax(alphas, dim=-1)
        for i, unary_t in enumerate(unary):
            next_tag_var = alphas + trans
            viterbi, best_tag_ids = torch.max(next_tag_var, 2)
            backpointers.append(best_tag_ids)
            new_alphas = viterbi + unary_t
            new_alphas.unsqueeze_(1)
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(
                    mask == MASK_FALSE, 0)
            else:
                alphas = new_alphas
        terminal_var = alphas.squeeze(1) + trans[:, (self.end_idx), :]
        path_score, best_tag_id = torch.max(terminal_var, 1)
        rev_len = seq_len - lengths - 1
        best_path = [best_tag_id]
        for i in range(len(backpointers)):
            t = len(backpointers) - i - 1
            backpointer_t = backpointers[t]
            new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)
                ).squeeze(1)
            mask = i > rev_len
            best_tag_id = best_tag_id.masked_fill(mask, 0
                ) + new_best_tag_id.masked_fill(mask == MASK_FALSE, 0)
            best_path.append(best_tag_id)
        _ = best_path.pop()
        best_path.reverse()
        best_path = torch.stack(best_path)
        seq_mask = sequence_mask(lengths).to(best_path.device).transpose(0, 1)
        best_path = best_path.masked_fill(seq_mask == MASK_FALSE, 0)
        return best_path, path_score


class TaggerGreedyDecoder(nn.Module):

    def __init__(self, num_tags: int, constraint_mask: Optional[torch.
        Tensor]=None, batch_first: bool=True, reduction: str='batch'):
        """A Greedy decoder and loss module for taggers.

        :param num_tags: `int` The number of output classes
        :param constraint_mask: `Tensor[1, N, N]` A mask with valid transitions as 1 and invalid as 0
        :param batch_first: `bool` Should the batch dimensions be first?
        :param reduction: `str` Should the loss be calculated at the token level or batch level
        """
        super().__init__()
        self.num_tags = num_tags
        if constraint_mask is not None:
            constraint_mask = F.log_softmax(torch.zeros(constraint_mask.
                shape).masked_fill(constraint_mask, -10000.0), dim=1)
            self.register_buffer('constraint_mask', constraint_mask)
        else:
            self.constraint_mask = None
        self.to_batch_first = ident if batch_first else tbh2bth
        self.to_time_first = bth2tbh if batch_first else ident
        self.batch_first = batch_first
        self.loss = SequenceLoss(LossFn=nn.CrossEntropyLoss, avg=reduction)
        self.viterbi = ViterbiLogSoftmaxNorm(Offsets.GO, Offsets.EOS)

    @property
    def transitions(self):
        return self.constraint_mask

    def neg_log_loss(self, inputs, tags, lengths):
        unaries = self.to_batch_first(inputs)
        tags = self.to_batch_first(tags)
        return self.loss(unaries, tags)

    def forward(self, inputs) ->torch.Tensor:
        unaries, lengths = tensor_and_lengths(inputs)
        if self.constraint_mask is not None:
            probv = self.to_time_first(unaries)
            probv = F.log_softmax(probv, dim=-1)
            preds, scores = self.viterbi(probv, self.constraint_mask, lengths)
            if self.batch_first:
                return tbh2bth(preds)
            else:
                return preds
        else:
            _, preds = torch.max(unaries, -1)
            mask = sequence_mask(lengths, unaries.shape[1]).to(preds.device)
            mask = mask if self.batch_first else mask.transpose(0, 1)
            preds = preds.masked_fill(mask == MASK_FALSE, 0)
        return preds

    def extra_repr(self) ->str:
        str_ = f'n_tags={self.num_tags}, batch_first={self.batch_first}'
        if self.constraint_mask is not None:
            str_ += ', constrained=True'
        return str_


class SequenceModel(nn.Module):

    def __init__(self, nc: int, embeddings: nn.Module, transducer: nn.
        Module, decoder: Optional[nn.Module]=None):
        super().__init__()
        self.embed_model = embeddings
        self.transducer_model = transducer
        if transducer.output_dim != nc:
            self.proj_layer = Dense(transducer.output_dim, nc)
        else:
            self.proj_layer = nn.Identity()
        self.decoder_model = decoder

    def transduce(self, inputs: Dict[str, torch.Tensor]) ->torch.Tensor:
        lengths = inputs['lengths']
        embedded = self.embed_model(inputs)
        embedded = embedded, lengths
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced: torch.Tensor, lengths: torch.Tensor
        ) ->torch.Tensor:
        return self.decoder_model((transduced, lengths))

    def forward(self, inputs: Dict[str, torch.Tensor]) ->torch.Tensor:
        pass


class LangSequenceModel(nn.Module):

    def __init__(self, nc: int, embeddings: nn.Module, transducer: nn.
        Module, decoder: Optional[nn.Module]=None, name: Optional[str]=None):
        super().__init__()
        self.embed_model = embeddings
        self.transducer_model = transducer
        if hasattr(transducer, 'requires_state') and transducer.requires_state:
            self._call = self._call_with_state
            self.requires_state = True
        else:
            self._call = self._call_without_state
            self.requires_state = False
        self.output_layer = nn.Linear(self.transducer_model.output_dim, nc)
        self.decoder_model = decoder

    def forward(self, inputs: Dict[str, torch.Tensor]) ->Tuple[torch.Tensor,
        Optional[torch.Tensor]]:
        return self._call(inputs)

    def _call_with_state(self, inputs: Dict[str, torch.Tensor]) ->Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        h = inputs['h']
        embedded = self.embed_model(inputs)
        transduced, hidden = self.transducer_model((embedded, h))
        transduced = self.output_layer(transduced)
        return transduced, hidden

    def _call_without_state(self, inputs: Dict[str, torch.Tensor]) ->Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        embedded = self.embed_model(inputs)
        transduced = self.transducer_model((embedded, None))
        transduced = self.output_layer(transduced)
        return transduced, None


class SequenceSequenceAttention(nn.Module):

    def __init__(self, hsz: int=None, pdrop: float=0.1, **kwargs):
        super().__init__()
        self.hsz = hsz
        self.dropout = nn.Dropout(pdrop)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor]) ->torch.Tensor:
        query, key, value, mask = qkvm
        a = self._attention(query, key, mask)
        self.attn = a
        a = self.dropout(a)
        return self._update(a, value)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask:
        Optional[torch.Tensor]=None) ->torch.Tensor:
        pass

    def _update(self, a: torch.Tensor, value: torch.Tensor) ->torch.Tensor:
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T, T]
        :param values: The values [B, H, T, D]
        :returns: A tensor of shape [B, H, T, D]
        """
        return torch.matmul(a, value)


class SequenceSequenceRelativeAttention(nn.Module):
    """This form of attention is specified in Shaw et al 2018: https://www.aclweb.org/anthology/N18-2074.pdf

    """

    def __init__(self, hsz: int=None, pdrop: float=0.1, **kwargs):
        super().__init__()
        self.hsz = hsz
        self.dropout = nn.Dropout(pdrop)
        self.attn = None

    def forward(self, q_k_v_ek_ev_m: Tuple[torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ) ->torch.Tensor:
        """Take in a tuple of tensors corresponding to the query, key, value, edges_key, edges_value and mask variables

        :param q_k_v_ek_ev_m: A tuple consisting of query, key, value, `edges_key`, `edges_value` and `mask` respectively
        :return: An updated value Tensor
        """
        query, key, value, edges_key, edges_value, mask = q_k_v_ek_ev_m
        a = self._attention(query, key, edges_key, mask)
        self.attn = a
        a = self.dropout(a)
        return self._update(a, value, edges_value)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, edges_key:
        torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        pass

    def _update(self, a: torch.Tensor, value: torch.Tensor, edges_value:
        torch.Tensor) ->torch.Tensor:
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T, T]
        :param value: The values [B, H, T, D]
        :param edge_value: The edge values [T, T, D]
        :returns: A tensor of shape [B, H, T, D]
        """
        B, H, T, D = value.shape
        updated_values = torch.matmul(a, value)
        a = a.view(B * H, T, T).transpose(0, 1)
        t = torch.matmul(a, edges_value)
        update_edge_values = t.transpose(0, 1).view(B, H, T, D)
        return updated_values + update_edge_values


class SeqDotProductAttention(SequenceSequenceAttention):

    def __init__(self, pdrop: float=0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask:
        Optional[torch.Tensor]=None) ->torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1000000000.0)
        return F.softmax(scores, dim=-1)


class SeqScaledDotProductAttention(SequenceSequenceAttention):

    def __init__(self, pdrop: float=0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, mask:
        Optional[torch.Tensor]=None) ->torch.Tensor:
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: A tensor that is (BxHxTxT)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1000000000.0)
        return F.softmax(scores, dim=-1)


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """

    def __init__(self, num_heads: int, d_model: int, dropout: float=0.1,
        scale: bool=False, d_k: Optional[int]=None):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: Should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(
                    f'd_model ({d_model}) must be evenly divisible by num_heads ({num_heads})'
                    )
        else:
            self.d_k = d_k
        self.h = num_heads
        self.w_Q = Dense(d_model, self.d_k * self.h)
        self.w_K = Dense(d_model, self.d_k * self.h)
        self.w_V = Dense(d_model, self.d_k * self.h)
        self.w_O = Dense(self.d_k * self.h, d_model)
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor]) ->torch.Tensor:
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(
            1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(
            1, 2)
        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn
        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)


class SeqScaledDotProductRelativeAttention(SequenceSequenceRelativeAttention):

    def __init__(self, pdrop: float=0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, edges_key:
        torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param edges_key: a matrix of relative embeddings between each word in a sequence [TxTxD]
        :return: A tensor that is (BxHxTxT)
        """
        B, H, T, d_k = query.shape
        scores_qk = torch.matmul(query, key.transpose(-2, -1))
        tbhd = query.reshape(B * H, T, d_k).transpose(0, 1)
        scores_qek = torch.matmul(tbhd, edges_key.transpose(-2, -1))
        scores_qek = scores_qek.transpose(0, 1).view(B, H, T, T)
        scores = (scores_qk + scores_qek) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1000000000.0)
        return F.softmax(scores, dim=-1)


class SeqDotProductRelativeAttention(SequenceSequenceRelativeAttention):

    def __init__(self, pdrop: float=0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query: torch.Tensor, key: torch.Tensor, edges_key:
        torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        B, H, T, d_k = query.shape
        scores_qk = torch.matmul(query, key.transpose(-2, -1))
        tbhd = query.reshape(B * H, T, d_k).transpose(0, 1)
        scores_qek = torch.matmul(tbhd, edges_key.transpose(-2, -1))
        scores_qek = scores_qek.transpose(0, 1).view(B, H, T, T)
        scores = scores_qk + scores_qek
        if mask is not None:
            scores = scores.masked_fill(mask == MASK_FALSE, -1000000000.0)
        return F.softmax(scores, dim=-1)


class MultiHeadedRelativeAttention(nn.Module):
    """
    Multi-headed relative attention from Shaw et al 2018 (https://www.aclweb.org/anthology/N18-2074.pdf)

    This method follows the same approach of MultiHeadedAttention, but it computes Relative Position Representations (RPR)
    which are used as part of the attention computations.  To facilitate this, the model has its own internal
    embeddings lookup table, and it has an updated computation for both the attention weights and the application
    of those weights to follow them.

    """

    def __init__(self, num_heads: int, d_model: int, rpr_k: int, dropout:
        float=0.1, scale: bool=False, d_k: Optional[int]=None):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: Should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(
                    f'd_model ({d_model}) must be evenly divisible by num_heads ({num_heads})'
                    )
        else:
            self.d_k = d_k
        self.rpr_k = rpr_k
        self.rpr_key = nn.Embedding(2 * rpr_k + 1, self.d_k)
        self.rpr_value = nn.Embedding(2 * rpr_k + 1, self.d_k)
        self.h = num_heads
        self.w_Q = Dense(d_model, self.d_k * self.h)
        self.w_K = Dense(d_model, self.d_k * self.h)
        self.w_V = Dense(d_model, self.d_k * self.h)
        self.w_O = Dense(self.d_k * self.h, d_model)
        if scale:
            self.attn_fn = SeqScaledDotProductRelativeAttention(dropout)
        else:
            self.attn_fn = SeqDotProductRelativeAttention(dropout)
        self.attn = None

    def make_rpr(self, seq_len, device) ->Tuple[torch.Tensor, torch.Tensor]:
        """Create a matrix shifted by self.rpr_k and bounded between 0 and 2*self.rpr_k to provide 0-based indexing for embedding
        """
        seq = torch.arange(seq_len).to(device)
        window_len = 2 * self.rpr_k
        edges = seq.view(1, -1) - seq.view(-1, 1) + self.rpr_k
        edges = torch.clamp(edges, 0, window_len)
        return self.rpr_key(edges), self.rpr_value(edges)

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor]) ->torch.Tensor:
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        seq_len = query.size(1)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(
            1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(
            1, 2)
        rpr_key, rpr_value = self.make_rpr(seq_len, query.device)
        x = self.attn_fn((query, key, value, rpr_key, rpr_value, mask))
        self.attn = self.attn_fn.attn
        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)


class TransformerEncoder(nn.Module):

    def __init__(self, num_heads: int, d_model: int, pdrop: float, scale:
        bool=True, activation_type: str='relu', d_ff: Optional[int]=None,
        d_k: Optional[int]=None, rpr_k: Optional[int]=None, ffn_pdrop:
        Optional[float]=0.0, layer_norms_after: bool=False, layer_norm_eps:
        float=1e-06):
        super().__init__()
        self.layer_norms_after = layer_norms_after
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads,
                d_model, rpr_k, pdrop, scale, d_k=d_k)
        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop,
                scale=scale, d_k=d_k)
        self.ffn = nn.Sequential(Dense(self.d_model, self.d_ff),
            get_activation(activation_type), nn.Dropout(ffn_pdrop), Dense(
            self.d_ff, self.d_model))
        self.ln1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]
        ) ->torch.Tensor:
        """
        :param inputs: `(x, mask)`
        :return: The output tensor
        """
        x, mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        h = self.self_attn((x, x, x, mask))
        x = x + self.dropout(h)
        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, num_heads: int, d_model: int, pdrop: float, scale:
        bool=True, activation_type: str='relu', d_ff: Optional[int]=None,
        d_k: Optional[int]=None, rpr_k: Optional[int]=None, ffn_pdrop:
        Optional[float]=0.0, layer_norms_after: bool=False, layer_norm_eps:
        float=1e-06):
        super().__init__()
        self.d_model = d_model
        self.layer_norms_after = layer_norms_after
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads,
                d_model, rpr_k, pdrop, scale, d_k=d_k)
            self.src_attn = MultiHeadedRelativeAttention(num_heads, d_model,
                rpr_k, pdrop, scale, d_k=d_k)
        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop,
                scale, d_k=d_k)
            self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop,
                scale, d_k=d_k)
        self.ffn = nn.Sequential(Dense(self.d_model, self.d_ff), nn.Dropout
            (ffn_pdrop), get_activation(activation_type), Dense(self.d_ff,
            self.d_model))
        self.ln1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.ln3 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.
        Tensor, torch.Tensor]) ->torch.Tensor:
        x, memory, src_mask, tgt_mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x, tgt_mask)))
        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory, src_mask)))
        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x))
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerDecoderStack(nn.Module):

    def __init__(self, num_heads: int, d_model: int, pdrop: float, scale:
        bool=True, layers: int=1, activation_type: str='relu', d_ff:
        Optional[int]=None, d_k: Optional[int]=None, rpr_k: Optional[Union[
        int, List[int]]]=None, ffn_pdrop: Optional[float]=0.0,
        layer_norms_after: bool=False, layer_norm_eps: float=1e-06):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.ln = nn.Identity() if layer_norms_after else nn.LayerNorm(d_model,
            eps=layer_norm_eps)
        if not is_sequence(rpr_k):
            rpr_k = [rpr_k] * layers
        for i in range(layers):
            self.decoders.append(TransformerDecoder(num_heads, d_model,
                pdrop, scale, activation_type, d_ff, d_k=d_k, rpr_k=rpr_k[i
                ], ffn_pdrop=ffn_pdrop, layer_norms_after=layer_norms_after,
                layer_norm_eps=layer_norm_eps))

    def forward(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        for layer in self.decoders:
            x = layer((x, memory, src_mask, tgt_mask))
        return self.ln(x)


class SingleHeadReduction(nn.Module):
    """
    Implementation of the "self_attention_head" layer from the conveRT paper (https://arxiv.org/pdf/1911.03688.pdf)
    """

    def __init__(self, d_model: int, dropout: float=0.0, scale: bool=True,
        d_k: Optional[int]=None):
        """
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        self.d_model = d_model
        if d_k is None:
            self.d_k = d_model
        else:
            self.d_k = d_k
        self.w_Q = Dense(d_model, self.d_k)
        self.w_K = Dense(d_model, self.d_k)
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor]) ->torch.Tensor:
        """According to conveRT model's graph, they project token encodings to lower-dimensional query and key in single
        head, use them to calculate the attention score matrix that has dim [B, T, T], then sum over the query dim to
        get a tensor with [B, 1, T] (meaning the amount of attentions each token gets from all other tokens), scale it
        by sqrt of sequence lengths, then use it as the weight to weighted sum the token encoding to get the sentence
        encoding. we implement it in an equivalent way that can best make use of the eight_mile codes: do the matrix
        multiply with value first, then sum over the query dimension.
        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: sentence-level encoding with dim [B, d_model]
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        seq_mask = mask.squeeze()
        seq_lengths = seq_mask.sum(dim=1)
        query = self.w_Q(query).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        value = value.view(batchsz, -1, 1, self.d_model).transpose(1, 2)
        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn
        x = x.squeeze(1)
        x = x * seq_mask.unsqueeze(-1)
        x = x.sum(dim=1)
        x = x * seq_lengths.float().sqrt().unsqueeze(-1)
        return x


class TiedWeights(nn.Module):

    def __init__(self):
        super().__init__()
        self.tgt_embeddings = nn.Embedding(100, 10)
        self.preds = pytorch_linear(10, 100)
        self.preds.weight = self.tgt_embeddings.weight

    def forward(self, input_vec):
        return self.preds(self.tgt_embeddings(input_vec))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dpressel_mead_baseline(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ArcPolicy(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BHT2BTH(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BTH2BHT(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BTH2TBH(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Conv1DSame(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})

    def test_005(self):
        self._check(ConvEncoder(*[], **{'insz': 4, 'outsz': 4, 'filtsz': 4}), [torch.rand([4, 4, 4])], {})

    def test_006(self):
        self._check(ConvEncoderStack(*[], **{'insz': 4, 'outsz': 4, 'filtsz': 4}), [torch.rand([4, 4, 4])], {})

    def test_007(self):
        self._check(Dense(*[], **{'insz': 4, 'outsz': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(GeLU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(Highway(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(MaxPool1D(*[], **{'outsz': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(MultiHeadedAttention(*[], **{'num_heads': 4, 'd_model': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(MultiHeadedRelativeAttention(*[], **{'num_heads': 4, 'd_model': 4, 'rpr_k': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(ParallelConv(*[], **{'insz': 4, 'outsz': [4, 4], 'filtsz': [4, 4]}), [torch.rand([4, 4, 4])], {})

    def test_014(self):
        self._check(PassThru(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(Reduction(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_016(self):
        self._check(SeqDotProductAttention(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_017(self):
        self._check(SeqScaledDotProductAttention(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_018(self):
        self._check(SingleHeadReduction(*[], **{'d_model': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(SumLayerNormReduction(*[], **{'output_dims': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_020(self):
        self._check(SumReduction(*[], **{'output_dims': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(TBH2BTH(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_022(self):
        self._check(TiedWeights(*[], **{}), [torch.zeros([4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_023(self):
        self._check(TwoHeadConcat(*[], **{'d_model': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(VariationalDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

