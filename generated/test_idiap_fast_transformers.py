import sys
_module = sys.modules[__name__]
del sys
fast_transformers = _module
aggregate = _module
attention = _module
aft_attention = _module
attention_layer = _module
causal_linear_attention = _module
clustered_attention = _module
conditional_full_attention = _module
exact_topk_attention = _module
full_attention = _module
improved_clustered_attention = _module
improved_clustered_causal_attention = _module
linear_attention = _module
local_attention = _module
reformer_attention = _module
attention_registry = _module
registry = _module
spec = _module
builders = _module
attention_builders = _module
base = _module
transformer_builders = _module
causal_product = _module
clustering = _module
hamming = _module
events = _module
event = _module
event_dispatcher = _module
filters = _module
feature_maps = _module
base = _module
fourier_features = _module
hashing = _module
local_product = _module
masking = _module
recurrent = _module
_utils = _module
cross_attention = _module
attention_layer = _module
full_attention = _module
linear_attention = _module
self_attention = _module
attention_layer = _module
full_attention = _module
linear_attention = _module
transformers = _module
sparse_product = _module
transformers = _module
utils = _module
weight_mapper = _module
setup = _module
tests = _module
test_aggregate_cpu = _module
test_aggregate_gpu = _module
test_clustered_aggregate_cpu = _module
test_clustered_aggregate_gpu = _module
test_clustered_broadcast_cpu = _module
test_clustered_broadcast_gpu = _module
test_aft_attention = _module
test_attention_layer = _module
test_causal_linear_attention = _module
test_clustered_transformer = _module
test_clustered_transformer_gpu = _module
test_full_attention = _module
test_improved_clustered_transformer_gpu = _module
test_linear_attention = _module
test_local_attention = _module
test_causal_product = _module
test_causal_product_cpu = _module
test_causal_product_gpu = _module
test_cluster_cpu = _module
test_cluster_gpu = _module
test_python_api_gpu = _module
time_python_api_gpu = _module
test_event_dispatcher = _module
test_event_filters = _module
test_events = _module
test_fourier_features = _module
test_hash_cpu = _module
test_hash_gpu = _module
test_local_product_cpu = _module
test_local_product_cuda = _module
test_attention_layer = _module
test_full_attention = _module
test_linear_attention = _module
test_attention_layer = _module
test_full_attention = _module
test_linear_attention = _module
test_transformer_decoder = _module
test_transformer_encoder = _module
test_clustered_sparse_product_backward_cpu = _module
test_clustered_sparse_product_backward_cpu_v2 = _module
test_clustered_sparse_product_backward_gpu = _module
test_clustered_sparse_product_cpu = _module
test_clustered_sparse_product_cpu_v2 = _module
test_clustered_sparse_product_gpu = _module
test_clustered_sparse_weighted_average_cpu = _module
test_clustered_sparse_weighted_average_cpu_v2 = _module
test_clustered_sparse_weighted_average_gpu = _module
test_sparse_product_backward_cpu = _module
test_sparse_product_backward_gpu = _module
test_sparse_product_cpu = _module
test_sparse_product_gpu = _module
test_sparse_weighted_average_cpu = _module
test_sparse_weighted_average_gpu = _module
test_builders = _module
test_masking = _module
test_transformer_decoder = _module
test_transformer_encoder = _module
test_weight_mapper = _module
tools = _module

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


from torch.nn import Module


from torch.nn import Linear


from math import sqrt


import torch.autograd


from torch.nn import Dropout


from torch.nn.init import normal_


from torch.nn import functional as F


from torch.nn import LayerNorm


import numpy as np


from functools import partial


from math import log


import warnings


from torch.nn import ModuleList


import torch.nn.functional as F


import re


from functools import lru_cache


from itertools import dropwhile


import time


import torch.nn as nn


class Event(object):
    """The Event is the base class for all events that are dispatched from any
    transformer module.

    This class defines only the basic attributes of an event without any
    payload.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
    """

    def __init__(self, source):
        self.source = source


class EventFilter(object):
    """EventFilter instances are predicates (ie functions that return True or
    False) to be used with an event dispatcher for filtering event
    instances.

    The main benefit from using raw functions is that an EventFilter composes
    very easily using operators such as &, |, ~.

    Example
    --------

        event_filter = AttentionEvent | layer_name_contains("layers.1")
        event_filter = from_layer(transformer.layers[2].attention)
        event_filter = (
            AttentionEvent &
            lambda ev: torch.isnan(ev.attention_matrix).any()
        )
    """

    def __call__(self, event):
        raise NotImplementedError()

    def _to_event_filter(self, other):
        if isinstance(other, EventFilter):
            return other
        if isinstance(other, type) and issubclass(other, Event):
            return event_class(other)
        if callable(other):
            return CallableEventFilter(other)
        return NotImplemented

    def __and__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: self(ev) and other(ev))

    def __rand__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: other(ev) and self(ev))

    def __or__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: self(ev) or other(ev))

    def __ror__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: other(ev) or self(ev))

    def __invert__(self):
        return CallableEventFilter(lambda ev: not self(ev))


class CallableEventFilter(EventFilter):
    """Wrap a function with an EventFilter object."""

    def __init__(self, event_filter):
        self._event_filter = event_filter

    def __call__(self, event):
        return self._event_filter(event)


def event_class(klass):
    """Select events that are instances of `klass`.

    Arguments
    ---------
        klass: A class to check the event instance against

    Returns
    -------
        An instance of EventFilter
    """
    return CallableEventFilter(lambda ev: isinstance(ev, klass))


class EventDispatcher(object):
    """An EventDispatcher is a simple way to implement an observer pattern for
    loose coupling of components. In our case it is used so that the internals
    of large neural networks can communicate with the outside world in an
    agnostic and efficient way.

    Example usage
    -------------

        from fast_transformers.events import EventDispatcher, AttentionEvent
        from fast_transformers.events.filters import             layer_name_contains

        def attention_event_handler(event):
            print(event.attention_matrix)

        ed = EventDispatcher()
        ed.listen(AttentionEvent, attention_event_handler)
        ed.listen(
            AttentionEvent & layer_name_contains("layers.12"),
            attention_event_handler
        )
    """
    _dispatchers = {}

    def __init__(self):
        self._listeners = OrderedDict()

    def listen(self, event_filter, event_handler):
        """Add an event handler for the events that pass the event filter.

        Arguments
        ---------
            event_filter: callable or Event class to define for which events
                          this handler will be called
            event_handler: callable that accepts an instance of Event
        """
        if isinstance(event_filter, type) and issubclass(event_filter, Event):
            event_filter = event_class(event_filter)
        self._listeners[event_handler] = event_filter

    def remove(self, event_handler):
        """Remove the event_handler from the listeners so that no more events
        are dispatched to this handler."""
        self._listeners.pop(event_handler, None)

    def clear(self):
        """Remove all listeners from the event dispatcher."""
        self._listeners.clear()

    def dispatch(self, event):
        """Dispatch an event to the listeners.

        Arguments
        ---------
            event: Event instance
        """
        for event_handler, event_filter in self._listeners.items():
            if event_filter(event):
                event_handler(event)

    @classmethod
    def get(cls, key=''):
        """Factory method for creating global event dispatchers for loosely
        coupling parts of a larger codebase.

        Since global objects are a complete antipattern, we suggest that this
        is only used to set a default value for an event dispatcher passed as
        an argument.

        Argument
        --------
            key: A key to uniquely identify a dispatcher or an instance of a
                 dispatcher to be returned as is
        """
        if isinstance(key, cls):
            return key
        if key not in cls._dispatchers:
            cls._dispatchers[key] = cls()
        return cls._dispatchers[key]


class AFTFullAttention(Module):
    """Implement the "full" attention proposed 'An Attention Free Transformer'.

    AFT attention uses only element wise operations to generate the new values
    given the queries, keys and values. The full AFT is computed as follows:

        v' = sigmoid(q) * (softmax(K + w_q, dim=1) * V).sum(dim=1)

    where q is a single query, K and V are the key and value matrices and w_q
    is a learnable vector for the given query position.

    Arguments
    ---------
        max_sequence_length: int, it defines the maximum acceptable sequence
                             length in order to allocate the learnable
                             parameters
        aft_parameterization: int, defines the dimensionality of the low rank
                              parameterization for the position bias
                              (default: 64)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, max_sequence_length=1024, aft_parameterization=64, event_dispatcher=''):
        super().__init__()
        self.u = torch.nn.Parameter(torch.randn(max_sequence_length, aft_parameterization) * 0.01)
        self.v = torch.nn.Parameter(torch.randn(max_sequence_length, aft_parameterization) * 0.01)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        if E != D:
            raise ValueError('AFT requires that queries, keys and values have the same dimensionality')
        Q = torch.sigmoid(queries)
        K = keys.permute(0, 2, 3, 1).contiguous()
        V = values.permute(0, 2, 3, 1).contiguous()
        K = K + key_lengths.additive_matrix[:, None, None, :]
        w = self.u[:L].mm(self.v[:S].t()) + attn_mask.additive_matrix
        K = K[:, None, :, :, :] * w[None, :, None, None, :]
        K = torch.softmax(K, dim=-1)
        V = Q * torch.einsum('nlhds,nhds->nlhd', K, V)
        return V


class AFTSimpleAttention(Module):
    """Implement the "simple" attention proposed 'An Attention Free Transformer'.

    AFT attention uses only element wise operations to generate the new values
    given the queries, keys and values. For the simple case that has no
    learnable parameters the new values are computed as follows:

        V' = sigmoid(Q) * (softmax(K, dim=1) * V).sum(dim=1)

    Arguments
    ---------
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, event_dispatcher=''):
        super().__init__()
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        if E != D:
            raise ValueError('AFT requires that queries, keys and values have the same dimensionality')
        Q = torch.sigmoid(queries)
        K = keys + key_lengths.additive_matrix[:, :, None, None]
        if attn_mask.lower_triangular:
            M, _ = K.max(dim=1, keepdim=True)
            Kexp = torch.exp(K - M)
            Kexpsum = Kexp.cumsum(dim=1)
            K = Kexp / Kexpsum
            V = Q * (K * values).cumsum(dim=1)
        elif attn_mask.all_ones:
            K = torch.softmax(K, dim=1)
            V = Q * (K * values).sum(dim=1, keepdim=True)
        else:
            raise ValueError('You cannot use general attention masks with AFTSimpleAttention because it would be quadratic in time. Use AFTFullAttention instead.')
        return V


class QKVEvent(Event):
    """An event containing the queries, keys and values projected in their
    multiple heads.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
        queries: torch.tensor containing the queries in shape NLHE
        keys: torch.tensor containing the keys in shape NSHE
        values: torch.tensor containing the values in shape NSHD
    """

    def __init__(self, source, queries, keys, values):
        super(QKVEvent, self).__init__(source)
        self.queries = queries
        self.keys = keys
        self.values = values


class AttentionLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, d_model_keys=None, event_dispatcher=''):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        d_model_keys = d_model_keys or d_model
        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))
        new_values = self.inner_attention(queries, keys, values, attn_mask, query_lengths, key_lengths).view(N, L, -1)
        return self.out_projection(new_values)


def causal_linear(Q, K, V):
    Q = Q.permute(0, 2, 1, 3).contiguous()
    K = K.permute(0, 2, 1, 3).contiguous()
    V = V.permute(0, 2, 1, 3).contiguous()
    V_new = causal_dot_product(Q, K, V)
    return V_new.permute(0, 2, 1, 3).contiguous()


class FeatureMap(Module):
    """Define the FeatureMap interface."""

    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """

        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""

    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(lambda x: torch.nn.functional.elu(x) + 1)


class CausalLinearAttention(Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, query_dimensions, feature_map=None, eps=1e-06, event_dispatcher=''):
        super(CausalLinearAttention, self).__init__()
        self.feature_map = feature_map(query_dimensions) if feature_map else elu_feature_map(query_dimensions)
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def _make_sizes_compatible(self, Q, K):
        """Either slice or pad K in case that the sizes do not match between Q
        and K."""
        N, L, H, E = Q.shape
        _, S, _, _ = K.shape
        if L == S:
            return Q, K
        if L < S:
            return Q, K[:, :L, :, :]
        if L > S:
            return Q, torch.cat([K, K.new_zeros(N, L - S, H, E)], dim=1)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)
        if not attn_mask.lower_triangular:
            raise RuntimeError('CausalLinearAttention only supports full lower triangular masks')
        K = K * key_lengths.float_matrix[:, :, None, None]
        Q, K = self._make_sizes_compatible(Q, K)
        Z = 1 / (torch.einsum('nlhi,nlhi->nlh', Q, K.cumsum(1)) + self.eps)
        V = causal_linear(Q, K, values)
        return V * Z[:, :, :, None]


def clustered_aggregate(X, G, F, lengths, Y=None):
    device = X.device
    if Y is None:
        Y = torch.zeros(F.shape + (X.shape[-1],), device=device, dtype=X.dtype)
    else:
        Y.zero_()
    if device.type == 'cpu':
        aggregate_cpu(X, G, F, Y)
    else:
        clustered_aggregate_gpu(X, G, F, lengths, Y)
    return Y


def set_group(C, E):
    C_per_block = int(192 * 64 / (E + 1))
    G_min = (C + C_per_block - 1) // C_per_block
    for G in range(G_min, C + 1):
        if C % G == 0:
            return G


def clustered_broadcast(Y, groups, counts, factors, X=None):
    device = Y.device
    if X is None:
        X = torch.zeros(groups.shape + (Y.shape[-1],), device=device, dtype=Y.dtype)
    if device.type == 'cpu':
        broadcast_cpu(Y, groups, factors, X)
    else:
        N, H, C, E = Y.shape
        _, _, L, _ = X.shape
        with torch.no_grad():
            threads = 256
            G = set_group(C, E)
            group_counts = counts.view(N, H, G, -1).sum(-1)
            block_counts = (group_counts + threads - 1) // threads
            total_blocks = block_counts.sum().item()
            indx_maps = torch.ones((total_blocks, 5), device=X.device, dtype=torch.int32)
        clustered_broadcast_gpu(Y, groups, factors, X, block_counts.int(), group_counts.int(), threads, G, total_blocks, indx_maps)
    return X


class _BroadcastValues(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v_grouped, clusters, counts, lengths):
        factors = torch.ones_like(counts, dtype=v_grouped.dtype)
        V = clustered_broadcast(v_grouped, clusters, counts, factors)
        ctx.save_for_backward(clusters, counts, factors, lengths)
        return V

    @staticmethod
    def backward(ctx, grad_v):
        clusters, counts, factors, lengths = ctx.saved_tensors
        grad_v_grouped = clustered_aggregate(grad_v, clusters, factors, lengths)
        return grad_v_grouped, None, None, None, None


class _GroupQueries(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, clusters, counts, lengths):
        factors = 1.0 / counts.float()
        q_grouped = clustered_aggregate(Q, clusters, factors, lengths)
        ctx.save_for_backward(clusters, counts, factors)
        return q_grouped

    @staticmethod
    def backward(ctx, grad_q_grouped):
        clusters, counts, factors = ctx.saved_tensors
        grad_q = clustered_broadcast(grad_q_grouped, clusters, counts, factors)
        return grad_q, None, None, None


def cluster(hashes, lengths, groups=None, counts=None, centroids=None, distances=None, bitcounts=None, clusters=30, iterations=10, bits=32):
    """Cluster hashes using a few iterations of K-Means with hamming distance.

    All the tensors default initialized to None are optional buffers to avoid
    memory allocations. distances and bitcounts are only used by the CUDA
    version of this call. clusters will be ignored if centroids is provided.

    Arguments
    ---------
        hashes: A long tensor of shape (N, H, L) containing a hashcode for each
                query.
        lengths: An int tensor of shape (N,) containing the sequence length for
                 each sequence in hashes.
        groups: An int tensor buffer of shape (N, H, L) contaning the cluster
                in which the corresponding hash belongs to.
        counts: An int tensor buffer of shape (N, H, K) containing the number
                of elements in each cluster.
        centroids: A long tensor buffer of shape (N, H, K) containing the
                   centroid for each cluster.
        distances: An int tensor of shape (N, H, L) containing the distance to
                   the closest centroid for each hash.
        bitcounts: An int tensor of shape (N, H, K, bits) containing the number
                   of elements that have 1 for a given bit.
        clusters: The number of clusters to use for each sequence. It is
                  ignored if centroids is not None.
        iterations: How many k-means iterations to perform.
        bits: How many of the least-significant bits in hashes to consider.

    Returns
    -------
        groups and counts as defined above.
    """
    device = hashes.device
    N, H, L = hashes.shape
    if device.type == 'cpu':
        if groups is None:
            groups = torch.empty((N, H, L), dtype=torch.int32)
        if centroids is None:
            centroids = torch.empty((N, H, clusters), dtype=torch.int64)
            centroids = hashes[:, :, np.random.choice(L, size=[clusters], replace=False)]
        K = centroids.shape[2]
        if counts is None:
            counts = torch.empty((N, H, K), dtype=torch.int32)
        cluster_cpu(hashes, lengths, centroids, groups, counts, iterations, bits)
        return groups, counts
    else:
        if groups is None:
            groups = torch.empty((N, H, L), dtype=torch.int32, device=device)
        if centroids is None:
            centroids = torch.empty((N, H, clusters), dtype=torch.int64, device=device)
            centroids = hashes[:, :, np.random.choice(L, size=[clusters], replace=False)]
        K = centroids.numel() // N // H
        if counts is None:
            counts = torch.empty((N, H, K), dtype=torch.int32, device=device)
        if distances is None:
            distances = torch.empty((N, H, L), dtype=torch.int32, device=device)
        if bitcounts is None:
            bitcounts = torch.empty((N, H, K, bits), dtype=torch.int32, device=device)
        groups = groups.view(N, H, L)
        counts = counts.view(N, H, K)
        centroids = centroids.view(N, H, K)
        distances = distances.view(N, H, L)
        bitcounts = bitcounts.view(N, H, K, -1)
        cluster_gpu(hashes, lengths, centroids, distances, bitcounts, groups, counts, iterations, bits)
        return groups, counts


def compute_hashes(X, A, H=None):
    device = X.device
    if H is None:
        H = torch.zeros(len(X), dtype=torch.int64, device=device)
    else:
        H.zero_()
    if A.shape[1] != X.shape[1] + 1:
        raise ValueError('The hash requires a bias')
    if device.type == 'cpu':
        compute_hashes_cpu(X, A, H)
    else:
        compute_hashes_cuda(X, A, H)
    return H


class ClusteredAttention(Module):
    """Use LSH and clustering in the resulting Hamming space to group queries
    that will have minimal L2 distance from each other.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to compute the attention using:

        V'_c = softmax(Q_c.mm(K.t()), dim=-1).mm(V).

    Now the computed values V'_c are "broadcasted" back to the query members
    of the corresponding cluster.

    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, clusters, iterations=10, bits=32, hash_bias=True, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(ClusteredAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def _create_query_groups(self, Q, query_lengths):
        N, H, L, E = Q.shape
        planes = Q.new_empty((self.bits, E + 1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N * H * L, E), planes).view(N, H, L)
        clusters, counts = cluster(hashes, query_lengths._lengths.int(), clusters=self.clusters, iterations=self.iterations, bits=self.bits)
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _group_queries(self, Q, groups, lengths):
        """Aggregate the Qs based on the index of cluster they belong to. Make
        sure to allow for gradient propagation backwards from the grouped
        queries to each query."""
        q_grouped = _GroupQueries.apply(Q, *groups, lengths)
        return q_grouped

    def _broadcast_values(self, V, groups, lengths):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), *groups, lengths)
        return V_new

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        assert attn_mask.all_ones, 'Clustered attention cannot use an arbitrary attention mask.'
        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        N, H, L, E = queries.shape
        _, _, S, D = values.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)
        q_offset = torch.arange(N * H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N * H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
        Q_grouped = self._group_queries(s_queries, groups, query_lengths._lengths.int())
        QK = torch.einsum('nhle,nhse->nhls', Q_grouped, keys)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum('nhls,nhsd->nhld', A, values)
        V_broadcast = self._broadcast_values(V, groups, query_lengths._lengths.int())
        rev_indx = torch.argsort(sorted_indx, dim=-1)
        q_rev_flat = (rev_indx.view(N * H, -1) + q_offset).reshape(-1)
        V_new = V_broadcast.reshape(-1, D).index_select(0, q_rev_flat).view(N, H, L, D)
        V_new = V_new.permute(0, 2, 1, 3).contiguous()
        return V_new


class AttentionEvent(Event):
    """An event containing an attention matrix.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
        attention_matrix: torch.tensor of the multihead attention matrix
                          computed in the corresponding attention layer
    """

    def __init__(self, source, attention_matrix):
        super(AttentionEvent, self).__init__(source)
        self.attention_matrix = attention_matrix


class FullAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        queries = queries * softmax_temp
        QK = torch.einsum('nlhe,nshe->nhls', queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]
        A = self.dropout(torch.softmax(QK, dim=-1))
        V = torch.einsum('nhls,nshd->nlhd', A, values)
        self.event_dispatcher.dispatch(AttentionEvent(self, A))
        return V.contiguous()


class ConditionalFullAttention(Module):
    """"Delegate to full attention if the input sequence is short.

    Arguments
    ---------
        other_attention: Use the passed attention module if the sequence is
                         longer than 'length_limit'.
        length_limit: An integer denoting the maximum sequence length to
                      consider.
        softmax_temp: See fast_transformers.attention.full_attention.
        attention_dropout: See fast_transformers.attention.full_attention.
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, other_attention, length_limit=512, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(ConditionalFullAttention, self).__init__()
        self.full_attention = FullAttention(softmax_temp, attention_dropout)
        self.other_attention = other_attention
        self.length_limit = length_limit
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        L = queries.shape[1]
        S = values.shape[1]
        if L > self.length_limit or S > self.length_limit:
            return self.other_attention(queries, keys, values, attn_mask, query_lengths, key_lengths)
        else:
            return self.full_attention(queries, keys, values, attn_mask, query_lengths, key_lengths)


class ExactTopKAttention(Module):
    """Implement the oracle top-k softmax attention.

    Arguments
    ---------
        top-k: The top k keys to attend to  (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, topk=32, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(ExactTopKAttention, self).__init__()
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        QK = torch.einsum('nlhe,nshe->nhls', queries, keys)
        topk = min(self.topk, S)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]
        topk_values, topk_idx = torch.topk(QK, topk, sorted=False, dim=-1)
        mask = QK.new_ones(QK.shape) * float('-inf')
        mask[torch.arange(N, device=QK.device).view(N, 1, 1, 1), torch.arange(H, device=QK.device).view(1, H, 1, 1), torch.arange(L, device=QK.device).view(1, 1, L, 1), topk_idx] = 0.0
        QK = QK + mask
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum('nhls,nshd->nlhd', A, values)
        return V.contiguous()


class ImprovedClusteredAttention(Module):
    """
    Immproved clustered attention approximation by recompution attention
    for each query with the top-k keys for the corresponding cluster.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to identify the top-k keys with highest
    dot products.

    Subsequently, for each query we compute the sparse dot product with
    the corresponding top-k keys to improve the attention approximation.

    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
        topk: Number of top-k keys to for improved approximation (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, clusters, iterations=10, bits=32, hash_bias=True, topk=32, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(ImprovedClusteredAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def _create_query_groups(self, Q, query_lengths):
        N, H, L, E = Q.shape
        planes = Q.new_empty((self.bits, E + 1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N * H * L, E), planes).view(N, H, L)
        clusters, counts = cluster(hashes, query_lengths._lengths.int(), clusters=self.clusters, iterations=self.iterations, bits=self.bits)
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _topk_attention(self, Q, K, V, clusters, counts, topk, topk_values, A_bottomk, softmax_temp, query_lengths):
        """Return the attention with just the topk heads."""
        N, H, L, E = Q.shape
        _, _, S, _ = K.shape
        _, _, C, k = topk.shape
        QK = clustered_sparse_dot_product(Q, K, topk, clusters, counts, query_lengths._lengths.int())
        QK = QK.masked_fill(torch.isinf(topk_values[:, 0, 0, :]).view(N, 1, 1, k), float('-inf'))
        A = torch.softmax(softmax_temp * QK, dim=-1)
        assert A_bottomk.is_contiguous()
        A_bottomk = clustered_broadcast(A_bottomk.unsqueeze(3), clusters, counts, torch.ones_like(counts, dtype=torch.float32))
        A = A * (1.0 - A_bottomk)
        A = self.dropout(A)
        assert A.is_contiguous()
        V_new = clustered_sparse_weighted_average(A, V, topk, clusters, counts)
        return V_new

    def _broadcast_values(self, V, clusters, counts, lengths):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), clusters, counts, lengths)
        return V_new

    def _bottomk_attention(self, QK, V, clusters, counts, query_lengths, topk, softmax_temp):
        """Return the attention with just the bottomk keys."""
        N, H, C, S = QK.shape
        A = torch.softmax(softmax_temp * QK, dim=-1)
        mask = QK.new_ones(QK.shape)
        mask[torch.arange(N, device=QK.device).view(N, 1, 1, 1), torch.arange(H, device=QK.device).view(1, H, 1, 1), torch.arange(C, device=QK.device).view(1, 1, C, 1), topk] = 0
        A = A * mask
        A_bottomk = A.sum(-1)
        A = self.dropout(A)
        V_new = torch.einsum('nhls,nhse->nhle', A, V)
        V_new = self._broadcast_values(V_new, clusters, counts, query_lengths._lengths.int())
        return V_new, A_bottomk

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        assert attn_mask.all_ones, 'Improved-clustered attention cannot use an arbitrary attention mask.'
        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        N, H, L, E = queries.shape
        _, _, S, D = values.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)
        clusters, counts = groups
        q_offset = torch.arange(N * H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N * H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
        Q_grouped = _GroupQueries.apply(s_queries, *groups, query_lengths.lengths.int())
        QK = torch.einsum('nhle,nhse->nhls', Q_grouped, keys)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        topk_values, topk = torch.topk(QK, min(self.topk, S), sorted=False, dim=-1)
        assert topk.is_contiguous()
        V_bottomk, A_bottomk = self._bottomk_attention(QK, values, clusters, counts, query_lengths, topk, softmax_temp)
        V_topk = self._topk_attention(s_queries, keys, values, clusters, counts, topk, topk_values, A_bottomk, softmax_temp, query_lengths)
        V_sorted_new = V_topk + V_bottomk
        sorted_rev_indx = torch.argsort(sorted_indx, dim=-1)
        q_rev_flat = (sorted_rev_indx.view(N * H, -1) + q_offset).reshape(-1)
        V_new = V_sorted_new.reshape(-1, D).index_select(0, q_rev_flat).view(N, H, L, D)
        return V_new.permute(0, 2, 1, 3).contiguous()


class ImprovedClusteredCausalAttention(Module):
    """
    Immproved clustered causal attention approximation by recomputing attention
    for each query with the top-k keys for the corresponding cluster.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to identify the top-k keys with highest
    dot products.

    Subsequently, for each query we compute the sparse dot product with
    the corresponding top-k keys to improve the attention approximation.

    Key difference with improved clustered attention is that we only use
    top-k keys with causal mask, we do not compute attention on the
    bottom-k keys.

    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
        topk: Number of top-k keys to for improved approximation (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, clusters, iterations=10, bits=32, hash_bias=True, topk=32, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(ImprovedClusteredCausalAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def _create_query_groups(self, Q, query_lengths):
        N, H, L, E = Q.shape
        planes = Q.new_empty((self.bits, E + 1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N * H * L, E), planes).view(N, H, L)
        clusters, counts = cluster(hashes, query_lengths.lengths.int(), clusters=self.clusters, iterations=self.iterations, bits=self.bits)
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _topk_attention(self, Q, K, V, q_flat, q_rev_flat, clusters, counts, topk, topk_values, softmax_temp, query_lengths):
        """Return the attention with just the topk heads."""
        N, H, L, E = Q.shape
        _, _, S, _ = K.shape
        _, _, C, k = topk.shape
        QK = clustered_sparse_dot_product(Q, K, topk, clusters, counts, query_lengths.lengths.int())
        assert topk.is_contiguous()
        topk_broadcast = clustered_broadcast(topk.float(), clusters, counts, torch.ones_like(counts, dtype=torch.float32))
        seq_ids = torch.arange(L, device=QK.device).view(1, 1, L, 1).repeat(N, H, 1, 1)
        s_seq_ids = seq_ids.reshape(-1, 1).index_select(0, q_flat).view(N, H, L, 1)
        future_mask = topk_broadcast.long() > s_seq_ids
        QK = QK.masked_fill(future_mask, float('-1e7'))
        A = torch.softmax(softmax_temp * QK, dim=-1)
        A = A * (1.0 - future_mask.float())
        A = self.dropout(A)
        assert A.is_contiguous()
        V_new = clustered_sparse_weighted_average(A, V, topk, clusters, counts)
        return V_new

    def _broadcast_values(self, V, clusters, counts, lengths):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), clusters, counts, lengths)
        return V_new

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        if not attn_mask.lower_triangular:
            raise RuntimeError('ImprovedClusteredCausalAttention only supports lower triangular masks')
        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        N, H, L, E = queries.shape
        _, _, S, D = values.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)
        clusters, counts = groups
        q_offset = torch.arange(N * H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N * H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
        Q_grouped = _GroupQueries.apply(s_queries, *groups, query_lengths.lengths.int())
        QK = torch.einsum('nhle,nhse->nhls', Q_grouped, keys)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        cur_topk = min(self.topk, min(key_lengths.lengths).item())
        topk_values, topk = torch.topk(QK, cur_topk, sorted=False, dim=-1)
        assert topk.is_contiguous()
        sorted_rev_indx = torch.argsort(sorted_indx, dim=-1)
        q_rev_flat = (sorted_rev_indx.view(N * H, -1) + q_offset).reshape(-1)
        V_topk = self._topk_attention(s_queries, keys, values, q_flat, q_rev_flat, clusters, counts, topk, topk_values, softmax_temp, query_lengths)
        V_sorted_new = V_topk
        V_new = V_sorted_new.reshape(-1, D).index_select(0, q_rev_flat).view(N, H, L, D)
        return V_new.permute(0, 2, 1, 3).contiguous()


class LinearAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, query_dimensions, feature_map=None, eps=1e-06, event_dispatcher=''):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map(query_dimensions) if feature_map else elu_feature_map(query_dimensions)
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)
        if not attn_mask.all_ones:
            raise RuntimeError('LinearAttention does not support arbitrary attention masks')
        K = K * key_lengths.float_matrix[:, :, None, None]
        KV = torch.einsum('nshd,nshm->nhmd', K, values)
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum('nlhd,nhmd,nlh->nlhm', Q, KV, Z)
        return V.contiguous()


class LocalAttention(Module):
    """Implement fast local attention where a query can only attend to
    neighboring keys.

    In this attention module the query Q_i can only attend to a key K_j if
    |i-j| < local_context/2.

    Arguments
    ---------
        local_context: The neighborhood to consider for local attention.
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, local_context, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(LocalAttention, self).__init__()
        self.local_context = local_context
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        """Implements the local attention.

        The attn_mask can be anything but the only values that will be
        considered will be the ones in the neighborhood of each query.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        context = self.local_context
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        QK = local_dot_product(queries, keys, attn_mask.additive_matrix_finite, key_lengths.lengths, self.local_context)
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V_new = local_weighted_average(A, values)
        return V_new.permute(0, 2, 1, 3).contiguous()


class ReformerAttention(Module):
    """Implement the attention module of the paper "Reformer the efficient
    transformer"

    Arguments
    ---------
        chunk_size  : Chunk size for each block (default: 32)
        bits        : Number of bits for hashing (default: 8)
        rounds      : Number of rounds of attention computation (default: 4)
        masked      : If true, the query does not attend to itsself (default: False)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, chunk_size=32, bits=8, rounds=4, masked=False, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(ReformerAttention, self).__init__()
        self.chunk_size = chunk_size
        self.bits = bits
        self.rounds = rounds
        self.masked = masked
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def _normalize(self, x):
        norms = torch.sqrt(torch.einsum('nlhe,nlhe->nlh', x, x))
        x_normed = x / norms.unsqueeze(-1)
        return x_normed

    def _look_back(self, x):
        xshape = x.shape
        return torch.cat([x.new_zeros((xshape[0], 1) + xshape[2:]), torch.repeat_interleave(x, 2, dim=1)[:, :-1]], dim=1).view(xshape[0], xshape[1], 2 * xshape[2], *xshape[3:])

    def _reformer_round(self, Q, K, V, mask, softmax_temp):
        N, L, H, E = Q.shape
        planes = Q.new_empty(self.bits, E)
        normal_(planes)
        projected = torch.einsum('nlhe,be->nlhb', K, planes)
        hashes = torch.argmax(torch.cat([projected, -projected], dim=-1), dim=-1)
        group = torch.argsort(hashes, dim=1)
        invert_group = torch.empty_like(group)
        batch_indices = torch.arange(N, device=hashes.device).view(N, 1, 1)
        sequence_indices = torch.arange(L, device=hashes.device).view(1, L, 1)
        head_indices = torch.arange(H, device=hashes.device).view(1, 1, H)
        invert_group[batch_indices, group, head_indices] = sequence_indices
        group = group.view(N, -1, self.chunk_size, H)
        invert_group = invert_group.view(N, -1, self.chunk_size, H)
        batch_indices = batch_indices.unsqueeze(1)
        head_indices = head_indices.unsqueeze(0)
        Q_grouped = Q[batch_indices, group, head_indices]
        K_grouped = K[batch_indices, group, head_indices]
        V_grouped = V[batch_indices, group, head_indices]
        mask_grouped = mask[batch_indices.unsqueeze(1), group.unsqueeze(3), self._look_back(group).unsqueeze(2)]
        mask_grouped[:, 0, :, :Q_grouped.shape[2]] = float('-inf')
        infmask = torch.isinf(mask_grouped)
        infmask = torch.all(infmask, dim=3, keepdims=True)
        mask_grouped = mask_grouped.masked_fill(infmask, 0.0)
        K_grouped = self._look_back(K_grouped)
        QQ = torch.einsum('nblhe,nbshe->nbhls', Q_grouped, K_grouped)
        QQ = QQ + mask_grouped.permute(0, 1, 4, 2, 3)
        A = torch.softmax(softmax_temp * QQ, dim=-1)
        A = self.dropout(A)
        V_grouped = self._look_back(V_grouped)
        V_new = torch.einsum('nbhls,nbshe->nblhe', A, V_grouped)
        V_new = V_new.contiguous().view(N, -1, H, E)
        V_new = V_new[batch_indices, invert_group, head_indices]
        V_new = V_new.contiguous().view(N, L, H, E)
        return V_new

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        N, L, H, E = queries.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        mask = key_lengths.additive_matrix.unsqueeze(1).expand(N, L, L)
        if self.masked:
            mask = mask + torch.eye(L, device=queries.device).unsqueeze(0) * float(-1000000000.0)
        if not attn_mask.all_ones:
            mask = mask + attn_mask.additive_matrix.unsqueeze(0)
        K = self._normalize(queries)
        K = K * key_lengths.float_matrix.view(N, L, 1, 1)
        V_new = 0
        factor = 1 / self.rounds
        for i in range(self.rounds):
            V_new = V_new + factor * self._reformer_round(queries, K, values, mask, softmax_temp)
        return V_new


def orthogonal_random_matrix_(w):
    """Initialize the matrix w in-place to compute orthogonal random features.

    The matrix is initialized such that its columns are orthogonal to each
    other (in groups of size `rows`) and their norms is drawn from the
    chi-square distribution with `rows` degrees of freedom (namely the norm of
    a `rows`-dimensional vector distributed as N(0, I)).

    Arguments
    ---------
        w: float tensor of size (rows, columns)
    """
    rows, columns = w.shape
    start = 0
    while start < columns:
        end = min(start + rows, columns)
        block = torch.randn(rows, rows, device=w.device)
        norms = torch.sqrt(torch.einsum('ab,ab->a', block, block))
        Q, _ = torch.qr(block)
        w[:, start:end] = Q[:, :end - start] * norms[None, :end - start]
        start += rows


class RandomFourierFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].

    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """

    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None, orthogonal=False, redraw=1, deterministic_eval=False):
        super(RandomFourierFeatures, self).__init__(query_dimensions)
        self.n_dims = n_dims or query_dimensions
        self.query_dimensions = query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = 1 / sqrt(query_dimensions) if softmax_temp is None else softmax_temp
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval
        self.register_buffer('omega', torch.zeros(self.query_dimensions, self.n_dims // 2))
        self._calls = -1

    def new_feature_map(self, device):
        if self.deterministic_eval and not self.training:
            return
        self._calls += 1
        if self._calls % self.redraw != 0:
            return
        omega = torch.zeros(self.query_dimensions, self.n_dims // 2, device=device)
        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()
        self.register_buffer('omega', omega)

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2 / self.n_dims)


class SmoothedRandomFourierFeatures(RandomFourierFeatures):
    """Simply add a constant value to the dot product in order to avoid
    possible numerical instabilities when the feature map is slightly
    negative.

    Implements K(x, y) = exp(-|x-y|^2) + s.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        smoothing: float, The smoothing parameter to add to the dot product.
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """

    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None, orthogonal=False, smoothing=1.0, redraw=1, deterministic_eval=False):
        super(SmoothedRandomFourierFeatures, self).__init__(query_dimensions, n_dims=query_dimensions - 1 if n_dims is None else n_dims - 1, softmax_temp=softmax_temp, orthogonal=orthogonal, redraw=redraw, deterministic_eval=deterministic_eval)
        self.smoothing = smoothing

    def forward(self, x):
        y = super().forward(x)
        smoothing = torch.full(y.shape[:-1] + (1,), self.smoothing, dtype=y.dtype, device=y.device)
        return torch.cat([y, smoothing], dim=-1)


class Favor(RandomFourierFeatures):
    """Positive orthogonal random features that approximate the softmax kernel.

    Basically implementation of Lemma 1 from "Rethinking Attention with
    Performers".

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """

    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None, orthogonal=True, stabilize=False, redraw=1, deterministic_eval=False):
        super(Favor, self).__init__(query_dimensions, n_dims=n_dims, softmax_temp=softmax_temp, orthogonal=orthogonal, redraw=redraw, deterministic_eval=deterministic_eval)
        self.stabilize = stabilize

    def _check_sequence_length(self, x):
        """Check that the 2nd dimension is larger than the 3rd as a heuristic
        that the sequence length will be larger than the number of heads. If
        not simply warn of a possible bug."""
        if len(x.shape) != 4:
            warnings.warn('Favor.stabilize is set to True but the input feature does not have the shape (N, L, H, D) which may result in unexpected behaviour')
        if x.shape[1] < x.shape[2]:
            warnings.warn('Favor.stabilize is set to True but the 2nd dimension of the input is smaller than the 3rd which could indicate that the sequence length and the heads are flipped. This may result in incorrect behaviour. The shape of the input is {!r}.'.format(x.shape))

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum('...d,...d->...', x, x).unsqueeze(-1)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        offset = norm_x_squared * 0.5 + 0.5 * log(self.n_dims)
        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]
        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)
        return phi


class GeneralizedRandomFeatures(RandomFourierFeatures):
    """Implements the generalized random Fourier features from Performers.

    It computes φ(χ) = [f(ω_1 χ), f(ω_2 χ), ..., f(ω_n χ)] where f(.) is the
    passed in `kernel_fn`.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (default: query_dimensions)
        softmax_temp: float, A normalizer for the dot products that is
                     multiplied to the input features before the feature map
                     application (default: 1.0)
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        kernel_fn: callable, defines the f used for the feature map.
                   (default: relu)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """

    def __init__(self, query_dimensions, n_dims=None, softmax_temp=1.0, orthogonal=True, kernel_fn=torch.relu, redraw=1, deterministic_eval=False):
        super(GeneralizedRandomFeatures, self).__init__(query_dimensions, n_dims=2 * query_dimensions if n_dims is None else 2 * n_dims, softmax_temp=softmax_temp, orthogonal=orthogonal, redraw=redraw, deterministic_eval=deterministic_eval)
        self.kernel_fn = kernel_fn

    def forward(self, x):
        if self.softmax_temp != 1.0:
            x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        return self.kernel_fn(u)


class RecurrentCrossAttentionLayer(Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer .

    The differences with the aforementioned module as well as the
    RecurrentAttentionLayer are that this module projects the query every time
    and the keys and values only the first time they are provided.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, d_model_keys=None, event_dispatcher=''):
        super(RecurrentCrossAttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        d_model_keys = d_model_keys or d_model
        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, keys, values, key_lengths, state=None):
        """Attend to the keys and values based on the passed in query.

        In the argument description we make use of the following sizes

            - N: the batch size
            - S: the sequence length of the keys and values
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Argument
        --------
            query: (N, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            key_lengths: A fast_transformers.masking.BaseMask implementation
                         that defines the length of each key/value sequence
            state: The state varies depending on the inner attention
                   implementation, but if it is not None then the keys and
                   values are ignored
        """
        N, _ = query.shape
        H = self.n_heads
        query = self.query_projection(query).view(N, H, -1)
        if state is None:
            _, S, _ = keys.shape
            keys = self.key_projection(keys).view(N, S, H, -1)
            values = self.value_projection(values).view(N, S, H, -1)
        else:
            keys = None
            values = None
        new_value, state = self.inner_attention(query, keys, values, key_lengths, state=state)
        new_value = new_value.view(N, -1)
        return self.out_projection(new_value), state


class RecurrentCrossFullAttention(Module):
    """Implement autoregressive softmax cross attention as a recurrent
    module.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(RecurrentCrossFullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, keys, values, key_lengths, state=None):
        N, H, E = query.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        if state is not None:
            keys, values = state
        QK = torch.einsum('nhe,nshe->nsh', query, keys)
        QK = QK + key_lengths.additive_matrix[:, :, None]
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=1))
        V = torch.einsum('nsh,nshd->nhd', A, values)
        self.event_dispatcher.dispatch(AttentionEvent(self, A))
        return V.contiguous(), [keys, values]


class RecurrentCrossLinearAttention(Module):
    """Implement autoregressive linear cross attention as a recurrent
    module.

    See fast_transformers.attention.linear_attention.LinearAttention .

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, query_dimensions, feature_map=None, eps=1e-06, event_dispatcher=''):
        super(RecurrentCrossLinearAttention, self).__init__()
        self.feature_map = feature_map(query_dimensions) if feature_map else elu_feature_map(query_dimensions)
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, keys, values, key_lengths, state=None):
        if state is None:
            self.feature_map.new_feature_map(query.device)
        Q = self.feature_map.forward_queries(query)
        if state is None:
            K = self.feature_map.forward_keys(keys)
            K = K * key_lengths.float_matrix[:, :, None, None]
            S = torch.einsum('nshd,nshm->nhmd', K, values)
            Z = K.sum(dim=1)
        else:
            S, Z = state
        QZ = 1 / (torch.einsum('nhd,nhd->nh', Q, Z) + self.eps)
        V = torch.einsum('nhd,nhmd,nh->nhm', Q, S, QZ)
        return V.contiguous(), [S, Z]


def check_state(state=None, memory=None):
    if memory is not None:
        warnings.warn("'memory' is deprecated for recurrent transformers  and will be removed in the future, use 'state' instead", DeprecationWarning)
    if state is None:
        state = memory
    return state


class RecurrentAttentionLayer(Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer.

    The only difference with the corresponding module is that this projects
    only one input and then calls the inner attention with the provided
    previous state.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, d_model_keys=None, event_dispatcher=''):
        super(RecurrentAttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        d_model_keys = d_model_keys or d_model
        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, state=None, memory=None):
        """Apply attention to the passed in query/key/value after projecting
        them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            query: (N, D) The tensor containing the queries
            key: (N, D) The tensor containing the keys
            value: (N, D) The tensor containing the values
            state: The state varies depending on the inner attention implementation
            memory: **Deprecated** and replaced by state

        Returns
        -------
            The new value for each query as a tensor of shape (N, D).
        """
        state = check_state(state, memory)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        N, D = query.shape
        H = self.n_heads
        new_value, state = self.inner_attention(query.view(N, H, -1), key.view(N, H, -1), value.view(N, H, -1), state)
        new_value = new_value.view(N, -1)
        return self.out_projection(new_value), state


class RecurrentFullAttention(Module):
    """Implement the full softmax attention as a recurrent module.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, softmax_temp=None, attention_dropout=0.1, event_dispatcher=''):
        super(RecurrentFullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, state=None, memory=None):
        state = check_state(state, memory)
        N, H, E = query.shape
        _, _, D = value.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)
        if state is not None:
            keys, values = state
            keys = torch.cat([keys, key[:, :, None]], dim=2)
            values = torch.cat([values, value[:, :, None]], dim=2)
        else:
            keys = key[:, :, None]
            values = value[:, :, None]
        QK = torch.einsum('nhe,nhse->nhs', query, keys)
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum('nhs,nhsd->nhd', A, values).contiguous()
        self.event_dispatcher.dispatch(AttentionEvent(self, A))
        return V, [keys, values]


class RecurrentLinearAttention(Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, query_dimensions, feature_map=None, eps=1e-06, event_dispatcher=''):
        super(RecurrentLinearAttention, self).__init__()
        self.feature_map = feature_map(query_dimensions) if feature_map else elu_feature_map(query_dimensions)
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, state=None, memory=None):
        state = check_state(state, memory)
        if state is None:
            self.feature_map.new_feature_map(query.device)
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)
        N, H, D = Q.shape
        _, _, M = value.shape
        if state is None:
            Si = query.new_zeros((N, H, D, M))
            Zi = query.new_zeros((N, H, D))
        else:
            Si, Zi = state
        if len(Si) != N:
            raise ValueError('The batch size changed during iteration')
        if K.grad_fn is not None or value.grad_fn is not None:
            Zi = Zi + K
            Si = Si + torch.einsum('nhd,nhm->nhdm', K, value)
        else:
            Zi += K
            Si += torch.einsum('nhd,nhm->nhdm', K, value)
        Z = 1.0 / (torch.einsum('nhd,nhd->nh', Q, Zi) + self.eps)
        V = torch.einsum('nhd,nhdm,nh->nhm', Q, Si, Z)
        return V, [Si, Zi]


class RecurrentTransformerEncoderLayer(Module):
    """Attention to the previous inputs and feed forward with skip connections.

    This transformer encoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerEncoderLayer . The results should
    be identical given the same inputs and a lower triangular mask.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu', event_dispatcher=''):
        super(RecurrentTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, state=None, memory=None):
        """Apply the transformer encoder to the input x using the provided
        memory.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            state: The state can vary depending on the attention implementation
            memory: **Deprecated** name for the state argument
        """
        state = check_state(state, memory)
        x2, state = self.attention(x, x, x, state)
        x = x + self.dropout(x2)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), state


class IntermediateOutput(Event):
    """Used by the TransformerEncoder and the TransformerDecoder to provide the
    intermediate outputs to interested callers.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
        x: torch.tensor containing the intermediate features in shape NLD
    """

    def __init__(self, source, x):
        super().__init__(source)
        self.x = x


class RecurrentTransformerEncoder(Module):
    """RecurrentTransformerEncoder is a sequence of
    RecurrentTransformerEncoderLayer instances.

    RecurrentTransformerEncoder keeps a separate state per
    RecurrentTransformerEncoderLayer.

    Arguments
    ---------
        layers: list, RecurrentTransformerEncoderLayer instances or instances
                that implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, layers, norm_layer=None, event_dispatcher=''):
        super(RecurrentTransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, state=None, memory=None):
        """Apply all recurrent transformer layers to the input x using the
        provided state.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor of each recurrent
               transformer encoder layer
            state: A list of objects to be passed to each recurrent
                   transformer encoder layer
            memory: **Deprecated** name for the state argument
        """
        state = check_state(state, memory)
        if state is None:
            state = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state[i])
            state[i] = s
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))
        if self.norm is not None:
            x = self.norm(x)
        return x, state


class BaseMask(object):

    @property
    def bool_matrix(self):
        """Return a bool (uint8) matrix with 1s to all places that should be
        kept."""
        raise NotImplementedError()

    @property
    def float_matrix(self):
        """Return the bool matrix as a float to be used as a multiplicative
        mask for non softmax attentions."""
        if not hasattr(self, '_float_matrix'):
            with torch.no_grad():
                self._float_matrix = self.bool_matrix.float()
        return self._float_matrix

    @property
    def lengths(self):
        """If the matrix is of the following form

            1 1 1 0 0 0 0
            1 0 0 0 0 0 0
            1 1 0 0 0 0 0

        then return it as a vector of integers

            3 1 2.
        """
        if not hasattr(self, '_lengths'):
            with torch.no_grad():
                lengths = self.bool_matrix.long().sum(dim=-1)
                m = self.bool_matrix.view(-1, self.shape[-1])
                for i, l in enumerate(lengths.view(-1)):
                    if not torch.all(m[i, :l]):
                        raise ValueError('The mask is not a length mask')
                self._lengths = lengths
        return self._lengths

    @property
    def shape(self):
        """Return the shape of the boolean mask."""
        return self.bool_matrix.shape

    @property
    def additive_matrix(self):
        """Return a float matrix to be added to an attention matrix before
        softmax."""
        if not hasattr(self, '_additive_matrix'):
            with torch.no_grad():
                self._additive_matrix = torch.log(self.bool_matrix.float())
        return self._additive_matrix

    @property
    def additive_matrix_finite(self):
        """Same as additive_matrix but with -1e24 instead of infinity."""
        if not hasattr(self, '_additive_matrix_finite'):
            with torch.no_grad():
                self._additive_matrix_finite = (~self.bool_matrix).float() * -1e+24
        return self._additive_matrix_finite

    @property
    def all_ones(self):
        """Return true if the mask is all ones."""
        if not hasattr(self, '_all_ones'):
            with torch.no_grad():
                self._all_ones = torch.all(self.bool_matrix)
        return self._all_ones

    @property
    def lower_triangular(self):
        """Return true if the attention is a triangular causal mask."""
        if not hasattr(self, '_lower_triangular'):
            self._lower_triangular = False
            with torch.no_grad():
                try:
                    lengths = self.lengths
                    if len(lengths.shape) == 1:
                        target = torch.arange(1, len(lengths) + 1, device=lengths.device)
                        self._lower_triangular = torch.all(lengths == target)
                except ValueError:
                    pass
        return self._lower_triangular


class LengthMask(BaseMask):
    """Provide a BaseMask interface for lengths. Mostly to be used with
    sequences of different lengths.

    Arguments
    ---------
        lengths: The lengths as a PyTorch long tensor
        max_len: The maximum length for the mask (defaults to lengths.max())
        device: The device to be used for creating the masks (defaults to
                lengths.device)
    """

    def __init__(self, lengths, max_len=None, device=None):
        self._device = device or lengths.device
        with torch.no_grad():
            self._lengths = lengths.clone()
        self._max_len = max_len or self._lengths.max()
        self._bool_matrix = None
        self._all_ones = torch.all(self._lengths == self._max_len).item()

    @property
    def bool_matrix(self):
        if self._bool_matrix is None:
            with torch.no_grad():
                indices = torch.arange(self._max_len, device=self._device)
                self._bool_matrix = indices.view(1, -1) < self._lengths.view(-1, 1)
        return self._bool_matrix


class RecurrentTransformerDecoderLayer(Module):
    """Attention to the previous inputs and a preprocessed memory.

    This transformer decoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerDecoderLayer . The results should
    be identical given the same inputs and a lower triangular mask for x_mask.

    Arguments
    ---------
        self_attention: The attention implementation to use for self attention
                        given as a nn.Module
        cross_attention: The attention implementation to use for cross
                         attention given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu', event_dispatcher=''):
        super(RecurrentTransformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, memory_length_mask=None, state=None):
        """Apply the transformer decoder to the input x and also attend to
        memory.

        Note the memory mask is assumed to be a full mask.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            memory: A sequence of features (N, S, E) that the input will attend
                    to. S is the sequence length and E is the same as for x.
            memory_length_mask: An implementation of a BaseMask that encodes
                                how many elements each memory sequence in the
                                batch consists of.
            state: The state varies depending on the attention implementations
                   but it allows for recurrent implementation.
        """
        N = x.shape[0]
        L = memory.shape[1]
        memory_length_mask = memory_length_mask or LengthMask(x.new_full((N,), L, dtype=torch.int64))
        self_state, cross_state = state or [None, None]
        x2, self_state = self.self_attention(x, x, x, state=self_state)
        x = self.norm1(x + self.dropout(x2))
        x2, cross_state = self.cross_attention(x, memory, memory, memory_length_mask, state=cross_state)
        x = self.norm2(x + self.dropout(x2))
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm3(x + y), [self_state, cross_state]


class RecurrentTransformerDecoder(Module):
    """RecurrentTransformerDecoder is little more than a sequence of
    RecurrentTransformerDecoderLayer instances.

    Simlar to the recurrent encoder a separate state is kept per decoder layer.

    Arguments
    ---------
        layers: list, RecurrentTransformerDecoderLayer instances or instances
                that implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, layers, norm_layer=None, event_dispatcher=''):
        super(RecurrentTransformerDecoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, memory_length_mask=None, state=None):
        """Apply all recurrent transformer layers to the input x using the
        provided state.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            memory: A sequence of features (N, S, E) that the input will attend
                    to. S is the sequence length and E is the same as for x.
            memory_length_mask: An implementation of a BaseMask that encodes
                                how many elements each memory sequence in the
                                batch consists of
            state: A list of objects to be passed to each recurrent
                   transformer decoder layer
        """
        if state is None:
            state = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, s = layer(x, memory, memory_length_mask=memory_length_mask, state=state[i])
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))
            state[i] = s
        if self.norm is not None:
            x = self.norm(x)
        return x, state


class FullMask(BaseMask):
    """Thin wrapper over a pytorch tensor that provides the BaseMask
    interface.

    The arguments can be given both by keyword arguments and positional
    arguments. To imitate function overloading, the constructor checks the type
    of the first argument and if it is a tensor it treats it as the mask.
    otherwise it assumes that it was the N argument.

    Arguments
    ---------
        mask: The mask as a PyTorch tensor.
        N: The rows of the all True mask to be created if the mask argument is
           not provided.
        M: The columns of the all True mask to be created if the mask argument
           is not provided. If N is given M defaults to N.
        device: The device to create the mask in (defaults to cpu)
    """

    def __init__(self, mask=None, N=None, M=None, device='cpu'):
        if mask is not None and isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                raise ValueError('FullMask expects the mask to be bool')
            with torch.no_grad():
                self._mask = mask.clone()
            return
        if mask is not None and M is None and isinstance(mask, int):
            M = N
            N = mask
        if N is not None:
            M = M or N
            with torch.no_grad():
                self._mask = torch.ones(N, M, dtype=torch.bool, device=device)
            self._all_ones = True
            return
        raise ValueError('Either mask or N should be provided')

    @property
    def bool_matrix(self):
        return self._mask


class TransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu', event_dispatcher=''):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or LengthMask(x.new_full((N,), L, dtype=torch.int64))
        x = x + self.dropout(self.attention(x, x, x, attn_mask=attn_mask, query_lengths=length_mask, key_lengths=length_mask))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y)


class TransformerEncoder(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, layers, norm_layer=None, event_dispatcher=''):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or LengthMask(x.new_full((N,), L, dtype=torch.int64))
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoderLayer(Module):
    """The decoder layer from "Attention Is All You Need".

    Similar to the encoder layer, this layer implements the decoder that
    PyTorch implements but can be used with any attention implementation
    because it receives the attention layers as constructor arguments.

    Arguments
    ---------
        self_attention: The attention implementation to use for self attention
                        given as a nn.Module
        cross_attention: The attention implementation to use for cross
                         attention given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu', event_dispatcher=''):
        super(TransformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None, memory_mask=None, memory_length_mask=None):
        """Apply the transformer decoder to the input x using the memory
        `memory`.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E should be the same as
               the d_model passed in the constructor.
            memory: The memory features of shape (N, L', E) where N is the
                    batch size, L' is the memory's sequence length (padded) and
                    E should be the same as the d_model.
            x_mask: An implementation of fast_transformers.masking.BaseMask
                    that encodes where each element of x can attend to in x.
                    Namely the self attention mask.
            x_length_mask: An implementation of a BaseMask that encodes how
                           many elements each sequence in the batch consists
                           of.
            memory_mask: An implementation of BaseMask that encodes where each
                         element of x can attend to in the memory. Namely the
                         cross attention mask.
            memory_length_mask: An implementation of a BaseMask that encodes how
                                many elements each memory sequence in the batch
                                consists of.
        """
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask or LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, query_lengths=x_length_mask, key_lengths=x_length_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, memory, memory, attn_mask=memory_mask, query_lengths=x_length_mask, key_lengths=memory_length_mask))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm3(x + y)


class TransformerDecoder(Module):
    """TransformerDecoder is little more than a sequence of transformer decoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ----------
        layers: list, TransformerDecoderLayer instances or instances that
                implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, layers, norm_layer=None, event_dispatcher=''):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None, memory_mask=None, memory_length_mask=None):
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask or LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))
        for layer in self.layers:
            x = layer(x, memory, x_mask=x_mask, x_length_mask=x_length_mask, memory_mask=memory_mask, memory_length_mask=memory_length_mask)
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))
        if self.norm is not None:
            x = self.norm(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActivationFunctionFeatureMap,
     lambda: ([], {'query_dims': 4, 'activation_function': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Favor,
     lambda: ([], {'query_dimensions': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedRandomFeatures,
     lambda: ([], {'query_dimensions': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RandomFourierFeatures,
     lambda: ([], {'query_dimensions': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmoothedRandomFourierFeatures,
     lambda: ([], {'query_dimensions': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_idiap_fast_transformers(_paritybench_base):
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

