import sys
_module = sys.modules[__name__]
del sys
build_readme = _module
conf = _module
example = _module
helpers = _module
performance = _module
plot = _module
movielens_sequence = _module
setup = _module
spotlight = _module
cross_validation = _module
datasets = _module
_transport = _module
amazon = _module
goodbooks = _module
movielens = _module
synthetic = _module
evaluation = _module
factorization = _module
_components = _module
explicit = _module
implicit = _module
representations = _module
interactions = _module
layers = _module
losses = _module
sampling = _module
sequence = _module
implicit = _module
representations = _module
torch_utils = _module
test_api = _module
test_explicit = _module
test_implicit = _module
test_sequence_implicit = _module
test_cross_validation = _module
test_datasets = _module
test_evaluation_metrics = _module
test_interactions = _module
test_layers = _module
test_serialization = _module

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


import time


import numpy as np


import torch


import torch.optim as optim


import torch.nn as nn


from sklearn.utils import murmurhash3_32


import torch.nn.functional as F


from torch.backends import cudnn


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, user_embedding_layer=None, item_embedding_layer=None, sparse=False):
        super(BilinearNet, self).__init__()
        self.embedding_dim = embedding_dim
        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        dot = (user_embedding * item_embedding).sum(1)
        return dot + user_bias + item_bias


class ScaledEmbeddingBag(nn.EmbeddingBag):
    """
    EmbeddingBag layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)


SEEDS = [179424941, 179425457, 179425907, 179426369, 179424977, 179425517, 179425943, 179426407, 179424989, 179425529, 179425993, 179426447, 179425003, 179425537, 179426003, 179426453, 179425019, 179425559, 179426029, 179426491, 179425027, 179425579, 179426081, 179426549]


class BloomEmbedding(nn.Module):
    """
    An embedding layer that compresses the number of embedding
    parameters required by using bloom filter-like hashing.

    Parameters
    ----------

    num_embeddings: int
        Number of entities to be represented.
    embedding_dim: int
        Latent dimension of the embedding.
    compression_ratio: float, optional
        The underlying number of rows in the embedding layer
        after compression. Numbers below 1.0 will use more
        and more compression, reducing the number of parameters
        in the layer.
    num_hash_functions: int, optional
        Number of hash functions used to compute the bloom filter indices.
    bag: bool, optional
        Whether to use the ``EmbeddingBag`` layer for the underlying embedding.
        This should be faster in principle, but currently seems to perform
        very poorly.

    Notes
    -----

    Large embedding layers are a performance problem for fitting models:
    even though the gradients are sparse (only a handful of user and item
    vectors need parameter updates in every minibatch), PyTorch updates
    the entire embedding layer at every backward pass. Computation time
    is then wasted on applying zero gradient steps to whole embedding matrix.

    To alleviate this problem, we can use a smaller underlying embedding layer,
    and probabilistically hash users and items into that smaller space. With
    good hash functions, collisions should be rare, and we should observe
    fitting speedups without a decrease in accuracy.

    The idea follows the RecSys 2017 "Getting recommenders fit"[1]_
    paper. The authors use a bloom-filter-like approach to hashing. Their approach
    uses one-hot encoded inputs followed by fully connected layers as
    well as softmax layers for the output, and their hashing reduces the
    size of the fully connected layers rather than embedding layers as
    implemented here; mathematically, however, the two formulations are
    identical.

    The hash function used is murmurhash3, hashing the indices with a different
    seed for every hash function, modulo the size of the compressed embedding layer.
    The hash mapping is computed once at the start of training, and indexed
    into for every minibatch.

    References
    ----------

    .. [1] Serra, Joan, and Alexandros Karatzoglou.
       "Getting deep recommenders fit: Bloom embeddings
       for sparse binary input/output networks."
       arXiv preprint arXiv:1706.03993 (2017).
    """

    def __init__(self, num_embeddings, embedding_dim, compression_ratio=0.2, num_hash_functions=4, bag=False, padding_idx=0):
        super(BloomEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.compressed_num_embeddings = int(compression_ratio * num_embeddings)
        self.num_hash_functions = num_hash_functions
        self.padding_idx = padding_idx
        self._bag = bag
        if num_hash_functions > len(SEEDS):
            raise ValueError('Can use at most {} hash functions ({} requested)'.format(len(SEEDS), num_hash_functions))
        self._masks = SEEDS[:self.num_hash_functions]
        if self._bag:
            self.embeddings = ScaledEmbeddingBag(self.compressed_num_embeddings, self.embedding_dim, mode='sum')
        else:
            self.embeddings = ScaledEmbedding(self.compressed_num_embeddings, self.embedding_dim, padding_idx=self.padding_idx)
        self._hashes = None
        self._offsets = None

    def __repr__(self):
        return '<BloomEmbedding (compression_ratio: {}): {}>'.format(self.compression_ratio, repr(self.embeddings))

    def _get_hashed_indices(self, original_indices):

        def _hash(x, seed):
            result = murmurhash3_32(x, seed=seed)
            result[self.padding_idx] = 0
            return result % self.compressed_num_embeddings
        if self._hashes is None:
            indices = np.arange(self.num_embeddings, dtype=np.int32)
            hashes = np.stack([_hash(indices, seed) for seed in self._masks], axis=1).astype(np.int64)
            assert hashes[self.padding_idx].sum() == 0
            self._hashes = torch.from_numpy(hashes)
            if original_indices.is_cuda:
                self._hashes = self._hashes
        hashed_indices = torch.index_select(self._hashes, 0, original_indices.squeeze())
        return hashed_indices

    def forward(self, indices):
        """
        Retrieve embeddings corresponding to indices.

        See documentation on PyTorch ``nn.Embedding`` for details.
        """
        if indices.dim() == 2:
            batch_size, seq_size = indices.size()
        else:
            batch_size, seq_size = indices.size(0), 1
        if not indices.is_contiguous():
            indices = indices.contiguous()
        indices = indices.data.view(batch_size * seq_size, 1)
        if self._bag:
            if self._offsets is None or self._offsets.size(0) != batch_size * seq_size:
                self._offsets = torch.arange(0, indices.numel(), indices.size(1)).long()
                if indices.is_cuda:
                    self._offsets = self._offsets
            hashed_indices = self._get_hashed_indices(indices)
            embedding = self.embeddings(hashed_indices.view(-1), self._offsets)
            embedding = embedding.view(batch_size, seq_size, -1)
        else:
            hashed_indices = self._get_hashed_indices(indices)
            embedding = self.embeddings(hashed_indices)
            embedding = embedding.sum(1)
            embedding = embedding.view(batch_size, seq_size, -1)
        return embedding


PADDING_IDX = 0


class PoolNet(nn.Module):
    """
    Module representing users through averaging the representations of items
    they have interacted with, a'la [1]_.

    To represent a sequence, it simply averages the representations of all
    the items that occur in the sequence up to that point.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [1] Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for
       youtube recommendations." Proceedings of the 10th ACM Conference
       on Recommender Systems. ACM, 2016.

    """

    def __init__(self, num_items, embedding_dim=32, item_embedding_layer=None, sparse=False):
        super(PoolNet, self).__init__()
        self.embedding_dim = embedding_dim
        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, padding_idx=PADDING_IDX, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse, padding_idx=PADDING_IDX)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """
        sequence_embeddings = self.item_embeddings(item_sequences).permute(0, 2, 1)
        sequence_embeddings = sequence_embeddings.unsqueeze(3)
        sequence_embeddings = F.pad(sequence_embeddings, (0, 0, 1, 0))
        sequence_embedding_sum = torch.cumsum(sequence_embeddings, 2)
        non_padding_entries = torch.cumsum((sequence_embeddings != 0.0).float(), 2).expand_as(sequence_embedding_sum)
        user_representations = (sequence_embedding_sum / (non_padding_entries + 1)).squeeze(3)
        return user_representations[:, :, :-1], user_representations[:, :, (-1)]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            Minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """
        target_embedding = self.item_embeddings(targets).permute(0, 2, 1).squeeze()
        target_bias = self.item_biases(targets).squeeze()
        dot = (user_representations * target_embedding).sum(1)
        return target_bias + dot


class LSTMNet(nn.Module):
    """
    Module representing users through running a recurrent neural network
    over the sequence, using the hidden state at each timestep as the
    sequence representation, a'la [2]_

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [2] Hidasi, Balazs, et al. "Session-based recommendations with
       recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
    """

    def __init__(self, num_items, embedding_dim=32, item_embedding_layer=None, sparse=False):
        super(LSTMNet, self).__init__()
        self.embedding_dim = embedding_dim
        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, padding_idx=PADDING_IDX, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse, padding_idx=PADDING_IDX)
        self.lstm = nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=embedding_dim)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """
        sequence_embeddings = self.item_embeddings(item_sequences).permute(0, 2, 1)
        sequence_embeddings = sequence_embeddings.unsqueeze(3)
        sequence_embeddings = F.pad(sequence_embeddings, (0, 0, 1, 0)).squeeze(3)
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)
        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        return user_representations[:, :, :-1], user_representations[:, :, (-1)]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """
        target_embedding = self.item_embeddings(targets).permute(0, 2, 1).squeeze()
        target_bias = self.item_biases(targets).squeeze()
        dot = (user_representations * target_embedding).sum(1).squeeze()
        return target_bias + dot


def _to_iterable(val, num):
    try:
        iter(val)
        return val
    except TypeError:
        return (val,) * num


class CNNNet(nn.Module):
    """
    Module representing users through stacked causal atrous convolutions ([3]_, [4]_).

    To represent a sequence, it runs a 1D convolution over the input sequence,
    from left to right. At each timestep, the output of the convolution is
    the representation of the sequence up to that point. The convolution is causal
    because future states are never part of the convolution's receptive field;
    this is achieved by left-padding the sequence.

    In order to increase the receptive field (and the capacity to encode states
    further back in the sequence), one can increase the kernel width, stack
    more layers, or increase the dilation factor.
    Input dimensionality is preserved from layer to layer.

    Residual connections can be added between all layers.

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and across time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of filters
        in each convolutional layer.
    kernel_width: tuple or int, optional
        The kernel width of the convolutional layers. If tuple, should contain
        the kernel widths for all convolutional layers. If int, it will be
        expanded into a tuple to match the number of layers.
    dilation: tuple or int, optional
        The dilation factor for atrous convolutions. Setting this to a number
        greater than 1 inserts gaps into the convolutional layers, increasing
        their receptive field without increasing the number of parameters.
        If tuple, should contain the dilation factors for all convolutional
        layers. If int, it will be expanded into a tuple to match the number
        of layers.
    num_layers: int, optional
        Number of stacked convolutional layers.
    nonlinearity: string, optional
        One of ('tanh', 'relu'). Denotes the type of non-linearity to apply
        after each convolutional layer.
    residual_connections: boolean, optional
        Whether to use residual connections between convolutional layers.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [3] Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio."
       arXiv preprint arXiv:1609.03499 (2016).
    .. [4] Kalchbrenner, Nal, et al. "Neural machine translation in linear time."
       arXiv preprint arXiv:1610.10099 (2016).
    """

    def __init__(self, num_items, embedding_dim=32, kernel_width=3, dilation=1, num_layers=1, nonlinearity='tanh', residual_connections=True, sparse=False, benchmark=True, item_embedding_layer=None):
        super(CNNNet, self).__init__()
        cudnn.benchmark = benchmark
        self.embedding_dim = embedding_dim
        self.kernel_width = _to_iterable(kernel_width, num_layers)
        self.dilation = _to_iterable(dilation, num_layers)
        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        else:
            raise ValueError('Nonlinearity must be one of (tanh, relu)')
        self.residual_connections = residual_connections
        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, padding_idx=PADDING_IDX, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse, padding_idx=PADDING_IDX)
        self.cnn_layers = [nn.Conv2d(embedding_dim, embedding_dim, (_kernel_width, 1), dilation=(_dilation, 1)) for _kernel_width, _dilation in zip(self.kernel_width, self.dilation)]
        for i, layer in enumerate(self.cnn_layers):
            self.add_module('cnn_{}'.format(i), layer)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """
        sequence_embeddings = self.item_embeddings(item_sequences).permute(0, 2, 1)
        sequence_embeddings = sequence_embeddings.unsqueeze(3)
        receptive_field_width = self.kernel_width[0] + (self.kernel_width[0] - 1) * (self.dilation[0] - 1)
        x = F.pad(sequence_embeddings, (0, 0, receptive_field_width, 0))
        x = self.nonlinearity(self.cnn_layers[0](x))
        if self.residual_connections:
            residual = F.pad(sequence_embeddings, (0, 0, 1, 0))
            x = x + residual
        for cnn_layer, kernel_width, dilation in zip(self.cnn_layers[1:], self.kernel_width[1:], self.dilation[1:]):
            receptive_field_width = kernel_width + (kernel_width - 1) * (dilation - 1)
            residual = x
            x = F.pad(x, (0, 0, receptive_field_width - 1, 0))
            x = self.nonlinearity(cnn_layer(x))
            if self.residual_connections:
                x = x + residual
        x = x.squeeze(3)
        return x[:, :, :-1], x[:, :, (-1)]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            Minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            Of shape (minibatch_size, sequence_length).
        """
        target_embedding = self.item_embeddings(targets).permute(0, 2, 1).squeeze()
        target_bias = self.item_biases(targets).squeeze()
        dot = (user_representations * target_embedding).sum(1).squeeze()
        return target_bias + dot


class MixtureLSTMNet(nn.Module):
    """
    A representation that models users as mixtures-of-tastes.

    This is accomplished via an LSTM with a layer on top that
    projects the last hidden state taste vectors and
    taste attention vectors that match items with the taste
    vectors that are best for evaluating them.

    For a full description of the model, see [5]_.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    num_mixtures: int, optional
        Number of mixture components (distinct user tastes) that
        the network should model.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [5] Kula, Maciej. "Mixture-of-tastes Models for Representing
       Users with Diverse Interests" https://github.com/maciejkula/mixture (2017)
    """

    def __init__(self, num_items, embedding_dim=32, num_mixtures=4, item_embedding_layer=None, sparse=False):
        super(MixtureLSTMNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures
        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim, padding_idx=PADDING_IDX, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse, padding_idx=PADDING_IDX)
        self.lstm = nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim, embedding_dim * self.num_mixtures * 2, kernel_size=1)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """
        batch_size, sequence_length = item_sequences.size()
        sequence_embeddings = self.item_embeddings(item_sequences).permute(0, 2, 1)
        sequence_embeddings = sequence_embeddings.unsqueeze(3)
        sequence_embeddings = F.pad(sequence_embeddings, (0, 0, 1, 0)).squeeze(3)
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)
        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.view(batch_size, self.num_mixtures * 2, self.embedding_dim, sequence_length + 1)
        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """
        user_components = user_representations[:, :self.num_mixtures, :, :]
        mixture_vectors = user_representations[:, self.num_mixtures:, :, :]
        target_embedding = self.item_embeddings(targets).permute(0, 2, 1)
        target_bias = self.item_biases(targets).squeeze()
        mixture_weights = mixture_vectors * target_embedding.unsqueeze(1).expand_as(user_components)
        mixture_weights = F.softmax(mixture_weights.sum(2), 1).unsqueeze(2).expand_as(user_components)
        weighted_user_representations = (mixture_weights * user_components).sum(1)
        dot = (weighted_user_representations * target_embedding).sum(1).squeeze()
        return target_bias + dot


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearNet,
     lambda: ([], {'num_users': 4, 'num_items': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (BloomEmbedding,
     lambda: ([], {'num_embeddings': 18, 'embedding_dim': 64}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (ScaledEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (ZeroEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
]

class Test_maciejkula_spotlight(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

