import sys
_module = sys.modules[__name__]
del sys
benchmark_averaging = _module
benchmark_dht = _module
benchmark_optimizer = _module
benchmark_tensor_compression = _module
benchmark_throughput = _module
conf = _module
arguments = _module
run_trainer = _module
run_training_monitor = _module
tokenize_wikitext103 = _module
utils = _module
hivemind = _module
averaging = _module
allreduce = _module
averager = _module
control = _module
group_info = _module
key_manager = _module
load_balancing = _module
matchmaking = _module
partition = _module
compression = _module
adaptive = _module
base = _module
floating = _module
quantization = _module
serialization = _module
dht = _module
crypto = _module
node = _module
protocol = _module
routing = _module
schema = _module
storage = _module
traverse = _module
validation = _module
hivemind_cli = _module
run_dht = _module
run_server = _module
moe = _module
client = _module
beam_search = _module
expert = _module
moe = _module
remote_expert_worker = _module
switch_moe = _module
expert_uid = _module
server = _module
checkpoints = _module
connection_handler = _module
dht_handler = _module
layers = _module
common = _module
custom_experts = _module
dropout = _module
lr_schedule = _module
optim = _module
module_backend = _module
runtime = _module
server = _module
task_pool = _module
grad_averager = _module
grad_scaler = _module
optimizer = _module
power_sgd_averager = _module
progress_tracker = _module
state_averager = _module
training_averager = _module
p2p = _module
p2p_daemon = _module
p2p_daemon_bindings = _module
datastructures = _module
p2pclient = _module
servicer = _module
asyncio = _module
auth = _module
limits = _module
logging = _module
math = _module
mpfuture = _module
nested = _module
networking = _module
performance_ema = _module
serializer = _module
streaming = _module
tensor_descr = _module
timed_storage = _module
setup = _module
conftest = _module
test_allreduce = _module
test_allreduce_fault_tolerance = _module
test_auth = _module
test_averaging = _module
test_cli_scripts = _module
test_compression = _module
test_connection_handler = _module
test_custom_experts = _module
test_dht = _module
test_dht_crypto = _module
test_dht_experts = _module
test_dht_node = _module
test_dht_protocol = _module
test_dht_schema = _module
test_dht_storage = _module
test_dht_validation = _module
test_expert_backend = _module
test_moe = _module
test_optimizer = _module
test_p2p_daemon = _module
test_p2p_daemon_bindings = _module
test_p2p_servicer = _module
test_routing = _module
test_start_server = _module
test_training = _module
test_util_modules = _module
custom_networks = _module
dht_swarms = _module

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


import math


import time


import torch


import random


from functools import partial


from typing import Callable


import torchvision


from torch import nn as nn


from torch.nn import functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import Optional


from enum import Enum


from typing import AsyncIterator


from typing import Sequence


from typing import Set


from typing import Tuple


from typing import Type


from typing import Any


from typing import Dict


from typing import Union


import numpy as np


from collections import deque


from typing import AsyncIterable


from typing import TypeVar


from abc import ABC


from abc import abstractmethod


from typing import Mapping


import warnings


from enum import auto


from typing import Iterable


from typing import List


import torch.nn as nn


from torch.autograd.function import once_differentiable


from queue import Empty


from queue import Queue


import torch.autograd


from torch.optim.lr_scheduler import LambdaLR


from torch import nn


from collections import defaultdict


from itertools import chain


from queue import SimpleQueue


from time import time


from typing import NamedTuple


from abc import ABCMeta


from collections import namedtuple


from typing import Generator


from typing import Iterator


from copy import deepcopy


from torch.cuda.amp import GradScaler as TorchGradScaler


from torch.cuda.amp.grad_scaler import OptState


from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state


from torch.optim import Optimizer as TorchOptimizer


import logging


import torch.nn.functional as F


import uuid


from typing import Generic


from torch.nn import Linear


from sklearn.datasets import load_digits


DUMMY = torch.empty(0, requires_grad=True)


ExpertUID, ExpertPrefix, Coordinate, Score = str, str, int, float


ENABLE_INLINING = True


IDENTITY_MULTIHASH_CODE = 0


MAX_INLINE_KEY_LENGTH = 42


def sha256_digest(data: Union[str, bytes]) ->bytes:
    if isinstance(data, str):
        data = data.encode('utf8')
    return hashlib.sha256(data).digest()


class PeerID:

    def __init__(self, peer_id_bytes: bytes) ->None:
        self._bytes = peer_id_bytes
        self._xor_id = int(sha256_digest(self._bytes).hex(), 16)
        self._b58_str = base58.b58encode(self._bytes).decode()

    @property
    def xor_id(self) ->int:
        return self._xor_id

    def to_bytes(self) ->bytes:
        return self._bytes

    def to_base58(self) ->str:
        return self._b58_str

    def __repr__(self) ->str:
        return f'<libp2p.peer.id.ID ({self.to_base58()})>'

    def __str__(self):
        return self.to_base58()

    def pretty(self):
        return self.to_base58()

    def to_string(self):
        return self.to_base58()

    def __eq__(self, other: object) ->bool:
        if isinstance(other, str):
            return self.to_base58() == other
        elif isinstance(other, bytes):
            return self._bytes == other
        elif isinstance(other, PeerID):
            return self._bytes == other._bytes
        else:
            return False

    def __lt__(self, other: object) ->bool:
        if not isinstance(other, PeerID):
            raise TypeError(f"'<' not supported between instances of 'PeerID' and '{type(other)}'")
        return self.to_base58() < other.to_base58()

    def __hash__(self) ->int:
        return hash(self._bytes)

    @classmethod
    def from_base58(cls, base58_id: str) ->'PeerID':
        peer_id_bytes = base58.b58decode(base58_id)
        return cls(peer_id_bytes)

    @classmethod
    def from_identity(cls, data: bytes) ->'PeerID':
        """
        See [1] for the specification of how this conversion should happen.

        [1] https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md#peer-ids
        """
        key_data = crypto_pb2.PrivateKey.FromString(data).data
        private_key = serialization.load_der_private_key(key_data, password=None)
        encoded_public_key = private_key.public_key().public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        encoded_public_key = crypto_pb2.PublicKey(key_type=crypto_pb2.RSA, data=encoded_public_key).SerializeToString()
        algo = multihash.Func.sha2_256
        if ENABLE_INLINING and len(encoded_public_key) <= MAX_INLINE_KEY_LENGTH:
            algo = IDENTITY_MULTIHASH_CODE
        encoded_digest = multihash.digest(encoded_public_key, algo).encode()
        return cls(encoded_digest)


ExpertInfo = NamedTuple('ExpertInfo', [('uid', ExpertUID), ('peer_id', PeerID)])


class SerializerBase(ABC):

    @staticmethod
    @abstractmethod
    def dumps(obj: object) ->bytes:
        pass

    @staticmethod
    @abstractmethod
    def loads(buf: bytes) ->object:
        pass


_default_handler = None


def _enable_default_handler(name: str) ->None:
    logger = get_logger(name)
    logger.addHandler(_default_handler)
    logger.propagate = False
    logger.setLevel(loglevel)


def _initialize_if_necessary():
    global _current_mode, _default_handler
    with _init_lock:
        if _default_handler is not None:
            return
        formatter = CustomFormatter(fmt='{asctime}.{msecs:03.0f} [{bold}{levelcolor}{levelname}{reset}]{caller_block} {message}', style='{', datefmt='%b %d %H:%M:%S')
        _default_handler = logging.StreamHandler()
        _default_handler.setFormatter(formatter)
        _enable_default_handler('hivemind')


def get_logger(name: Optional[str]=None) ->logging.Logger:
    """
    Same as ``logging.getLogger()`` but ensures that the default hivemind log handler is initialized.

    :note: By default, the hivemind log handler (that reads the ``HIVEMIND_LOGLEVEL`` env variable and uses
           the colored log formatter) is only applied to messages logged inside the hivemind package.
           If you want to extend this handler to other loggers in your application, call
           ``use_hivemind_log_handler("in_root_logger")``.
    """
    _initialize_if_necessary()
    return logging.getLogger(name)


class MSGPackSerializer(SerializerBase):
    _ext_types: Dict[Any, int] = {}
    _ext_type_codes: Dict[int, Any] = {}
    _TUPLE_EXT_TYPE_CODE = 64

    @classmethod
    def ext_serializable(cls, type_code: int):
        assert isinstance(type_code, int), 'Please specify a (unique) int type code'

        def wrap(wrapped_type: type):
            assert callable(getattr(wrapped_type, 'packb', None)) and callable(getattr(wrapped_type, 'unpackb', None)), f'Every ext_type must have 2 methods: packb(self) -> bytes and classmethod unpackb(cls, bytes)'
            if type_code in cls._ext_type_codes:
                logger.warning(f'{cls.__name__}: type {type_code} is already registered, overwriting')
            cls._ext_type_codes[type_code], cls._ext_types[wrapped_type] = wrapped_type, type_code
            return wrapped_type
        return wrap

    @classmethod
    def _encode_ext_types(cls, obj):
        type_code = cls._ext_types.get(type(obj))
        if type_code is not None:
            return msgpack.ExtType(type_code, obj.packb())
        elif isinstance(obj, tuple):
            data = msgpack.packb(list(obj), strict_types=True, use_bin_type=True, default=cls._encode_ext_types)
            return msgpack.ExtType(cls._TUPLE_EXT_TYPE_CODE, data)
        return obj

    @classmethod
    def _decode_ext_types(cls, type_code: int, data: bytes):
        if type_code in cls._ext_type_codes:
            return cls._ext_type_codes[type_code].unpackb(data)
        elif type_code == cls._TUPLE_EXT_TYPE_CODE:
            return tuple(msgpack.unpackb(data, ext_hook=cls._decode_ext_types, raw=False))
        logger.warning(f'Unknown ExtType code: {type_code}, leaving it as is')
        return data

    @classmethod
    def dumps(cls, obj: object) ->bytes:
        return msgpack.dumps(obj, use_bin_type=True, default=cls._encode_ext_types, strict_types=True)

    @classmethod
    def loads(cls, buf: bytes) ->object:
        return msgpack.loads(buf, ext_hook=cls._decode_ext_types, raw=False)


class P2PDaemonError(Exception):
    """
    Raised if daemon failed to handle request
    """


class ControlFailure(P2PDaemonError):
    pass


DEFAULT_MAX_MSG_SIZE = 4 * 1024 ** 2


P2PD_FILENAME = 'p2pd'


class P2PHandlerError(Exception):
    """
    Raised if remote handled a request with an exception
    """


class PublicKey(ABC):

    @abstractmethod
    def verify(self, data: bytes, signature: bytes) ->bool:
        ...

    @abstractmethod
    def to_bytes(self) ->bytes:
        ...

    @classmethod
    @abstractmethod
    def from_bytes(cls, key: bytes) ->bytes:
        ...


class PrivateKey(ABC):

    @abstractmethod
    def sign(self, data: bytes) ->bytes:
        ...

    @abstractmethod
    def get_public_key(self) ->PublicKey:
        ...


T = TypeVar('T')


async def as_aiter(*args: T) ->AsyncIterator[T]:
    """create an asynchronous iterator from a sequence of values"""
    for arg in args:
        yield arg


async def asingle(aiter: AsyncIterable[T]) ->T:
    """If ``aiter`` has exactly one item, returns this item. Otherwise, raises ``ValueError``."""
    count = 0
    async for item in aiter:
        count += 1
        if count == 2:
            raise ValueError('asingle() expected an iterable with exactly one item, but got two or more items')
    if count == 0:
        raise ValueError('asingle() expected an iterable with exactly one item, but got an empty iterable')
    return item


def golog_level_to_python(level: str) ->int:
    level = level.upper()
    if level in ['DPANIC', 'PANIC', 'FATAL']:
        return logging.CRITICAL
    level = logging.getLevelName(level)
    if not isinstance(level, int):
        raise ValueError(f'Unknown go-log level: {level}')
    return level


def python_level_to_golog(level: str) ->str:
    if not isinstance(level, str):
        raise ValueError('`level` is expected to be a Python log level in the string form')
    if level == 'CRITICAL':
        return 'FATAL'
    if level == 'WARNING':
        return 'WARN'
    return level


MAX_UNARY_PAYLOAD_SIZE = DEFAULT_MAX_MSG_SIZE // 2


async def azip(*iterables: AsyncIterable[T]) ->AsyncIterator[Tuple[T, ...]]:
    """equivalent of zip for asynchronous iterables"""
    iterators = [iterable.__aiter__() for iterable in iterables]
    while True:
        try:
            yield tuple(await asyncio.gather(*(itr.__anext__() for itr in iterators)))
        except StopAsyncIteration:
            break


async def iter_as_aiter(iterable: Iterable[T]) ->AsyncIterator[T]:
    """create an asynchronous iterator from single iterable"""
    for elem in iterable:
        yield elem


STREAMING_CHUNK_SIZE_BYTES = 2 ** 16


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


Key = Any


def _safe_check_pinned(tensor: torch.Tensor) ->bool:
    """Check whether or not a tensor is pinned. If torch cannot initialize cuda, returns False instead of error."""
    try:
        return torch.cuda.is_available() and tensor.is_pinned()
    except RuntimeError:
        return False


class TensorRole(Enum):
    ACTIVATION = auto()
    PARAMETER = auto()
    GRADIENT = auto()
    OPTIMIZER = auto()
    UNSPECIFIED = auto()


class _RemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

    @staticmethod
    def forward(ctx, dummy: torch.Tensor, uid: str, stub: 'ConnectionHandlerStub', info: Dict[str, Any], *inputs: torch.Tensor) ->Tuple[torch.Tensor, ...]:
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.uid, ctx.stub, ctx.info = uid, stub, info
        ctx.save_for_backward(*inputs)
        serialized_tensors = (serialize_torch_tensor(tensor, proto.compression) for tensor, proto in zip(inputs, nested_flatten(info['forward_schema'])))
        deserialized_outputs = RemoteExpertWorker.run_coroutine(expert_forward(uid, inputs, serialized_tensors, stub))
        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) ->Tuple[Optional[torch.Tensor], ...]:
        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info['forward_schema'], ctx.info['outputs_schema'])))
        serialized_tensors = (serialize_torch_tensor(tensor, proto.compression) for tensor, proto in zip(inputs_and_grad_outputs, backward_schema))
        deserialized_grad_inputs = RemoteExpertWorker.run_coroutine(expert_backward(ctx.uid, inputs_and_grad_outputs, serialized_tensors, ctx.stub))
        return DUMMY, None, None, None, *deserialized_grad_inputs


def nested_compare(t, u):
    """
    Return whether nested structure of t1 and t2 matches.
    """
    if isinstance(t, (list, tuple)):
        if not isinstance(u, type(t)):
            return False
        if len(t) != len(u):
            return False
        for a, b in zip(t, u):
            if not nested_compare(a, b):
                return False
        return True
    if isinstance(t, dict):
        if not isinstance(u, dict):
            return False
        if set(t.keys()) != set(u.keys()):
            return False
        for k in t:
            if not nested_compare(t[k], u[k]):
                return False
        return True
    else:
        return True


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
    elif isinstance(structure, (list, tuple)):
        return type(structure)(_nested_pack(flat_iter, x) for x in structure)
    elif isinstance(structure, dict):
        return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
    else:
        return next(flat_iter)


def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)


DHTExpiration = float


DHTKey = Subkey = DHTValue = Any


KeyType = TypeVar('KeyType')


ROOT = 0


ValueType = TypeVar('ValueType')


class Blacklist:
    """
    A temporary blacklist of non-responding peers with exponential backoff policy
    :param base_time: peers are suspended for this many seconds by default
    :param backoff_rate: suspension time increases by this factor after each successive failure
    """

    def __init__(self, base_time: float, backoff_rate: float, **kwargs):
        self.base_time, self.backoff = base_time, backoff_rate
        self.banned_peers = TimedStorage[PeerID, int](**kwargs)
        self.ban_counter = Counter()

    def register_failure(self, peer: PeerID):
        """peer failed to respond, add him to blacklist or increase his downtime"""
        if peer not in self.banned_peers and self.base_time > 0:
            ban_duration = self.base_time * self.backoff ** self.ban_counter[peer]
            self.banned_peers.store(peer, self.ban_counter[peer], expiration_time=get_dht_time() + ban_duration)
            self.ban_counter[peer] += 1

    def register_success(self, peer):
        """peer responded successfully, remove him from blacklist and reset his ban time"""
        del self.banned_peers[peer], self.ban_counter[peer]

    def __contains__(self, peer: PeerID) ->bool:
        return peer in self.banned_peers

    def __repr__(self):
        return f'{self.__class__.__name__}(base_time={self.base_time}, backoff={self.backoff}, banned_peers={len(self.banned_peers)})'

    def clear(self):
        self.banned_peers.clear()
        self.ban_counter.clear()


class AuthRole(Enum):
    CLIENT = 0
    SERVICER = 1


BinaryDHTID = BinaryDHTValue = bytes


MAX_DHT_TIME_DISCREPANCY_SECONDS = 3


class ValidationError(Exception):
    """This exception is thrown if DHT node didn't pass validation by other nodes."""


ResultType = TypeVar('ResultType')


class UpdateType(Enum):
    RESULT = auto()
    EXCEPTION = auto()
    CANCEL = auto()


ReturnType = TypeVar('ReturnType')


FLAT_EXPERT = -1


PREFIX_PATTERN = re.compile('^(([^.])+)([.](?:[0]|([1-9]([0-9]*))))*[.]$')


UID_DELIMITER = '.'


def is_valid_prefix(maybe_prefix: str) ->bool:
    """An uid prefix must contain a string expert type, followed by optional numeric indices and a trailing period"""
    return bool(PREFIX_PATTERN.fullmatch(maybe_prefix))


UID_PATTERN = re.compile('^(([^.])+)([.](?:[0]|([1-9]([0-9]*))))+$')


def is_valid_uid(maybe_uid: str) ->bool:
    """An uid must contain a string expert type, followed by one or more .-separated numeric indices"""
    return bool(UID_PATTERN.fullmatch(maybe_uid))


def nested_map(fn, *t):
    if not t:
        raise ValueError('Expected 2+ arguments, got 1')
    for i in range(1, len(t)):
        if not nested_compare(t[0], t[i]):
            msg = 'Nested structure of %r and %r differs'
            raise ValueError(msg % (t[0], t[i]))
    flat = map(nested_flatten, t)
    return nested_pack(map(fn, *flat), t[0])


ffn_sample_input = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@torch.jit.script
def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


name_to_block = {}


name_to_input = {}


def register_expert_class(name: str, sample_input: Callable[[int, int], torch.tensor]):
    """
    Adds a custom user expert to hivemind server.
    :param name: the name of the expert. It shouldn't coincide with existing modules        ('ffn', 'transformer', 'nop', 'det_dropout')
    :param sample_input: a function which gets batch_size and hid_dim and outputs a         sample of an input in the module
    :unchanged module
    """

    def _register_expert_class(custom_class: Type[nn.Module]):
        if name in name_to_block or name in name_to_input:
            raise RuntimeError('The class might already exist or be added twice')
        name_to_block[name] = custom_class
        name_to_input[name] = sample_input
        return custom_class
    return _register_expert_class


class FeedforwardBlock(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        self.ffn = nn.Linear(hid_dim, 4 * hid_dim)
        self.ffn_output = nn.Linear(4 * hid_dim, hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-12)

    def forward(self, x):
        ffn_output = self.ffn(x)
        ffn_output = gelu_fast(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return self.layer_norm(x + ffn_output)


class TransformerEncoderLayer(nn.Module):
    """
    A slight modification of torch.nn.TransformerEncoderLayer which allows for torch.jit scripting
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = gelu_fast

    def forward(self, src, src_key_padding_mask=None):
        src = src.transpose(0, 1)
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1)
        return src


transformer_sample_input = lambda batch_size, hid_dim: (torch.empty((batch_size, 128, hid_dim)), torch.empty((batch_size, 128), dtype=torch.bool))


class TunedTransformer(TransformerEncoderLayer):

    def __init__(self, hid_dim):
        super().__init__(hid_dim, dim_feedforward=4 * hid_dim, nhead=16)


nop_sample_input = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


class NopExpert(nn.Sequential):

    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(0), requires_grad=True)

    def forward(self, x):
        return x.clone()


class DelayedNopExpert(nn.Sequential):

    def __init__(self, hid_dim, delay=0.5):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(0), requires_grad=True)
        self.delay = delay

    def forward(self, x):
        time.sleep(self.delay)
        return x.clone()


class DeterministicDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, keep_prob, mask):
        ctx.keep_prob = keep_prob
        ctx.save_for_backward(mask)
        return x * mask / keep_prob

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.saved_tensors[0] * grad_output / ctx.keep_prob, None, None


class DeterministicDropout(nn.Module):
    """
    Custom dropout layer which accepts dropout mask as an input (drop_prob is only used for scaling input activations).
    Can be used with RemoteExpert/ModuleBackend to ensure that dropout mask is the same at forward and backward steps
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.keep_prob = 1 - drop_prob

    def forward(self, x, mask):
        if self.training:
            return DeterministicDropoutFunction.apply(x, self.keep_prob, mask)
        else:
            return x


dropout_sample_input = lambda batch_size, hid_dim: (torch.empty((batch_size, hid_dim)), torch.randint(0, 1, (batch_size, hid_dim)))


class DeterministicDropoutNetwork(nn.Module):

    def __init__(self, hid_dim, dropout_prob=0.2):
        super().__init__()
        self.linear_in = nn.Linear(hid_dim, 2 * hid_dim)
        self.activation = nn.ReLU()
        self.dropout = DeterministicDropout(dropout_prob)
        self.linear_out = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, x, mask):
        x = self.linear_in(self.dropout(x, mask))
        return self.linear_out(self.activation(x))


class SwitchNetwork(nn.Module):

    def __init__(self, dht, in_features, num_classes, num_experts):
        super().__init__()
        self.moe = RemoteSwitchMixtureOfExperts(in_features=in_features, grid_size=(num_experts,), dht=dht, jitter_eps=0, uid_prefix='expert.', k_best=1, k_min=1)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        moe_output, balancing_loss = self.moe(x)
        return self.linear(moe_output), balancing_loss


sample_input = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


class MultilayerPerceptron(nn.Module):

    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.layer2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.layer3 = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


multihead_sample_input = lambda batch_size, hidden_dim: (torch.empty((batch_size, hidden_dim)), torch.empty((batch_size, 2 * hidden_dim)), torch.empty((batch_size, 3 * hidden_dim)))


class MultiheadNetwork(nn.Module):

    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, num_classes)
        self.layer2 = nn.Linear(2 * hidden_dim, num_classes)
        self.layer3 = nn.Linear(3 * hidden_dim, num_classes)

    def forward(self, x1, x2, x3):
        x = self.layer1(x1) + self.layer2(x2) + self.layer3(x3)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeterministicDropout,
     lambda: ([], {'drop_prob': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeterministicDropoutNetwork,
     lambda: ([], {'hid_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedforwardBlock,
     lambda: ([], {'hid_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultilayerPerceptron,
     lambda: ([], {'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NopExpert,
     lambda: ([], {'hid_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_learning_at_home_hivemind(_paritybench_base):
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

