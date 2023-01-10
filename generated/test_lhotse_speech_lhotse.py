import sys
_module = sys.modules[__name__]
del sys
conf = _module
lhotse = _module
array = _module
audio = _module
augmentation = _module
common = _module
torchaudio = _module
transform = _module
utils = _module
wpe = _module
bin = _module
modes = _module
cli_base = _module
cut = _module
features = _module
install_tools = _module
kaldi = _module
manipulation = _module
recipes = _module
adept = _module
aidatatang_200zh = _module
aishell = _module
aishell2 = _module
aishell4 = _module
ali_meeting = _module
ami = _module
aspire = _module
babel = _module
broadcast_news = _module
bvcc = _module
callhome_egyptian = _module
callhome_english = _module
chime6 = _module
cmu_arctic = _module
cmu_indic = _module
cmu_kids = _module
commonvoice = _module
csj = _module
cslu_kids = _module
daily_talk = _module
dihard3 = _module
dipco = _module
earnings21 = _module
earnings22 = _module
eval2000 = _module
fisher_english = _module
fisher_spanish = _module
gale_arabic = _module
gale_mandarin = _module
gigaspeech = _module
heroico = _module
hifitts = _module
icsi = _module
l2_arctic = _module
libricss = _module
librimix = _module
librispeech = _module
libritts = _module
ljspeech = _module
magicdata = _module
mgb2 = _module
mls = _module
mtedx = _module
musan = _module
nsc = _module
peoples_speech = _module
primewords = _module
rir_noise = _module
spgispeech = _module
stcmds = _module
switchboard = _module
tal_asr = _module
tal_csasr = _module
tedlium = _module
thchs_30 = _module
timit = _module
vctk = _module
voxceleb = _module
wenet_speech = _module
xbmu_amdo31 = _module
yesno = _module
validate = _module
workflows = _module
caching = _module
base = _module
data = _module
mixed = _module
mono = _module
multi = _module
padding = _module
set = _module
dataset = _module
collation = _module
cut_transforms = _module
concatenate = _module
extra_padding = _module
mix = _module
perturb_speed = _module
perturb_tempo = _module
perturb_volume = _module
reverberate = _module
dataloading = _module
diarization = _module
input_strategies = _module
iterable_dataset = _module
sampling = _module
base = _module
bucketing = _module
cut_pairs = _module
data_source = _module
dynamic = _module
dynamic_bucketing = _module
round_robin = _module
simple = _module
zip = _module
signal_transforms = _module
source_separation = _module
speech_recognition = _module
speech_synthesis = _module
unsupervised = _module
vad = _module
vis = _module
webdataset = _module
base = _module
compression = _module
fbank = _module
io = _module
extractors = _module
layers = _module
kaldifeat = _module
librosa_fbank = _module
mfcc = _module
mixer = _module
opensmile = _module
spectrogram = _module
ssl = _module
lazy = _module
manifest = _module
parallel = _module
qa = _module
mobvoihotwords = _module
serialization = _module
shar = _module
readers = _module
datapipes = _module
lazy = _module
tar = _module
utils = _module
writers = _module
audio = _module
supervision = _module
testing = _module
dummies = _module
fixtures = _module
tools = _module
env = _module
sph2pipe = _module
utils = _module
workarounds = _module
forced_alignment = _module
whisper = _module
setup = _module
test = _module
test_torchaudio = _module
conftest = _module
test_custom_attrs = _module
test_custom_attrs_randomized = _module
test_cut = _module
test_cut_augmentation = _module
test_cut_drop_attributes = _module
test_cut_extend_by = _module
test_cut_fill_supervision = _module
test_cut_merge_supervisions = _module
test_cut_mixing = _module
test_cut_ops_preserve_id = _module
test_cut_set = _module
test_cut_set_mix = _module
test_cut_trim_to_supervisions = _module
test_cut_truncate = _module
test_cut_with_in_memory_data = _module
test_feature_extraction = _module
test_invariants_randomized = _module
test_masks = _module
test_multi_cut_augmentation = _module
test_padding_cut = _module
test_dynamic_bucketing = _module
test_sampler_pickling = _module
test_sampler_restoring = _module
test_sampling = _module
test_batch_io = _module
test_collation = _module
test_cut_transforms = _module
test_diarization = _module
test_iterable_dataset = _module
test_signal_transforms = _module
test_speech_recognition_dataset = _module
test_speech_recognition_dataset_randomized = _module
test_speech_synthesis_dataset = _module
test_unsupervised_dataset = _module
test_vad_dataset = _module
test_webdataset = _module
test_webdataset_ddp = _module
test_array = _module
test_chunky_writer = _module
test_copy_feats = _module
test_kaldi_features = _module
test_kaldi_layers = _module
test_kaldifeat_features = _module
test_librosa_fbank = _module
test_opensmile = _module
test_s3prl = _module
test_temporal_array = _module
test_torchaudio_features = _module
test_writer_append = _module
known_issues = _module
test_augment_with_executor = _module
test_cut_consistency = _module
test_lazy_cuts_issues = _module
test_mixed_cut_num_frames = _module
test_mixing_zero_energy_cuts = _module
test_dataloading = _module
test_missing_values = _module
test_read_datapipe = _module
test_read_lazy = _module
test_write = _module
test_audio_reads = _module
test_feature_set = _module
test_kaldi_dirs = _module
test_lazy = _module
test_manipulation = _module
test_multipexing_iterables = _module
test_parallel = _module
test_qa = _module
test_recording_set = _module
test_resample_randomized = _module
test_serialization = _module
test_supervision_set = _module
test_utils = _module

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


import functools


import logging


import random


import re


import warnings


from functools import lru_cache


from functools import partial


from itertools import islice


from math import ceil


from math import sqrt


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import List


from typing import Mapping


from typing import NamedTuple


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import torch


from typing import Set


from abc import ABCMeta


from abc import abstractmethod


from math import isclose


import itertools


from collections import Counter


from collections import defaultdict


from functools import reduce


from itertools import chain


from typing import FrozenSet


from typing import Sequence


from typing import Type


from typing import TypeVar


from torch.nn import CrossEntropyLoss


from torch.utils.data import Dataset


from copy import deepcopy


from torch import distributed as dist


from torch.utils.data import Sampler


import math


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataloader import default_collate


from typing import Generator


from torch import nn


from abc import ABC


import types


import torchaudio


import torchaudio.backend.no_backend


import inspect


import uuid


from typing import Iterator


from numpy.testing import assert_array_almost_equal


from torch.utils.data import DataLoader


import torch.testing


import torch.utils.data


from torch import tensor


import torch.distributed


import torch.multiprocessing as mp


def is_module_available(*modules: str) ->bool:
    """Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).

    Note: "borrowed" from torchaudio:
    https://github.com/pytorch/audio/blob/6bad3a66a7a1c7cc05755e9ee5931b7391d2b94c/torchaudio/_internal/module_utils.py#L9
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


class Dillable:
    """
    Mix-in that will leverage ``dill`` instead of ``pickle``
    when pickling an object.

    It is useful when the user can't avoid ``pickle`` (e.g. in multiprocessing),
    but needs to use unpicklable objects such as lambdas.

    If ``dill`` is not installed, it defers to what ``pickle`` does by default.
    """

    def __getstate__(self):
        if is_module_available('dill'):
            return dill.dumps(self.__dict__)
        else:
            return self.__dict__

    def __setstate__(self, state):
        if is_module_available('dill'):
            self.__dict__ = dill.loads(state)
        else:
            self.__dict__ = state


class ImitatesDict(Dillable):
    """
    Helper base class for lazy iterators defined below.
    It exists to make them drop-in replacements for data-holding dicts
    in Lhotse's CutSet, RecordingSet, etc. classes.
    """

    def __iter__(self):
        raise NotImplemented

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)


class LazyIteratorChain(ImitatesDict):
    """
    A thin wrapper over multiple iterators that enables to combine lazy manifests
    in Lhotse. It iterates all underlying iterables sequentially.

    .. note:: if any of the input iterables is a dict, we'll iterate only its values.
    """

    def __init__(self, *iterators: Iterable) ->None:
        self.iterators = []
        for it in iterators:
            if isinstance(it, LazyIteratorChain):
                for sub_it in it.iterators:
                    self.iterators.append(sub_it)
            else:
                self.iterators.append(it)

    def __iter__(self):
        for it in self.iterators:
            if isinstance(it, dict):
                it = it.values()
            yield from it

    def __len__(self) ->int:
        return sum(len(it) for it in self.iterators)

    def __add__(self, other) ->'LazyIteratorChain':
        return LazyIteratorChain(self, other)


T = TypeVar('T')


def fastcopy(dataclass_obj: T, **kwargs) ->T:
    """
    Returns a new object with the same member values.
    Selected members can be overwritten with kwargs.
    It's supposed to work only with dataclasses.
    It's 10X faster than the other methods I've tried...

    Example:
        >>> ts1 = TimeSpan(start=5, end=10)
        >>> ts2 = fastcopy(ts1, end=12)
    """
    return type(dataclass_obj)(**{**dataclass_obj.__dict__, **kwargs})


Seconds = float


def seconds_to_frames(duration: Seconds, frame_shift: Seconds, max_index: Optional[int]=None) ->int:
    """
    Convert time quantity in seconds to a frame index.
    It takes the shape of the array into account and limits
    the possible indices values to be compatible with the shape.
    """
    assert duration >= 0
    index = int(decimal.Decimal(round(duration / frame_shift, ndigits=8)).quantize(0, rounding=decimal.ROUND_HALF_UP))
    if max_index is not None:
        return min(index, max_index)
    return index


def asdict_nonull(dclass) ->Dict[str, Any]:
    """
    Recursively convert a dataclass into a dict, removing all the fields with `None` value.
    Intended to use in place of dataclasses.asdict(), when the null values are not desired in the serialized document.
    """

    def non_null_dict_factory(collection):
        d = dict(collection)
        remove_keys = []
        for key, val in d.items():
            if val is None:
                remove_keys.append(key)
        for k in remove_keys:
            del d[k]
        return d
    return asdict(dclass, dict_factory=non_null_dict_factory)


def compute_num_frames(duration: Seconds, frame_shift: Seconds, sampling_rate: int) ->int:
    """
    Compute the number of frames from duration and frame_shift in a safe way.
    """
    num_samples = round(duration * sampling_rate)
    window_hop = round(frame_shift * sampling_rate)
    num_frames = int((num_samples + window_hop // 2) // window_hop)
    return num_frames


class FeaturesReader(metaclass=ABCMeta):
    """
    ``FeaturesReader`` defines the interface of how to load numpy arrays from a particular storage backend.
    This backend could either be:

    - separate files on a local filesystem;
    - a single file with multiple arrays;
    - cloud storage;
    - etc.

    Each class inheriting from ``FeaturesReader`` must define:

    - the ``read()`` method, which defines the loading operation
        (accepts the ``key`` to locate the array in the storage and return it).
        The read method should support selecting only a subset of the feature matrix,
        with the bounds expressed as arguments ``left_offset_frames`` and ``right_offset_frames``.
        It's up to the Reader implementation to load only the required part or trim it to that
        range only after loading. It is assumed that the time dimension is always the first one.
    - the ``name()`` property that is unique to this particular storage mechanism -
        it is stored in the features manifests (metadata) and used to automatically deduce
        the backend when loading the features.

    The features writing must be defined separately in a class inheriting from ``FeaturesWriter``.
    """

    @property
    @abstractmethod
    def name(self) ->str:
        ...

    @abstractmethod
    def read(self, key: str, left_offset_frames: int=0, right_offset_frames: Optional[int]=None) ->np.ndarray:
        ...


READER_BACKENDS = {}


def get_reader(name: str) ->Type[FeaturesReader]:
    """
    Find a ``FeaturesReader`` sub-class that corresponds to the provided ``name`` and return its type.

    Example:

        reader_type = get_reader("lilcom_files")
        reader = reader_type("/storage/features/")
    """
    return READER_BACKENDS.get(name)


def ifnone(item: Optional[Any], alt_item: Any) ->Any:
    """Return ``alt_item`` if ``item is None``, otherwise ``item``."""
    return alt_item if item is None else item


class StdStreamWrapper:

    def __init__(self, stream):
        self.stream = stream

    def close(self):
        pass

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getattr__(self, item: str):
        if item == 'close':
            return self.close
        return getattr(self.stream, item)


def gzip_open_robust(filename, mode='rb', compresslevel=9, encoding=None, errors=None, newline=None):
    """Open a gzip-compressed file in binary or text mode.

    The filename argument can be an actual filename (a str or bytes object), or
    an existing file object to read from or write to.

    The mode argument can be "r", "rb", "w", "wb", "x", "xb", "a" or "ab" for
    binary mode, or "rt", "wt", "xt" or "at" for text mode. The default mode is
    "rb", and the default compresslevel is 9.

    For binary mode, this function is equivalent to the GzipFile constructor:
    GzipFile(filename, mode, compresslevel). In this case, the encoding, errors
    and newline arguments must not be provided.

    For text mode, a GzipFile object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error handling
    behavior, and line ending(s).

    Note: This method is copied from Python's 3.7 stdlib, and patched to handle
    "trailing garbage" in gzip files. We could monkey-patch the stdlib version,
    but we imagine that some users prefer third-party libraries like Lhotse
    not to do such things.
    """
    if 't' in mode:
        if 'b' in mode:
            raise ValueError('Invalid mode: %r' % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")
    gz_mode = mode.replace('t', '')
    if isinstance(filename, (str, bytes, os.PathLike)):
        binary_file = AltGzipFile(filename, gz_mode, compresslevel)
    elif hasattr(filename, 'read') or hasattr(filename, 'write'):
        binary_file = AltGzipFile(None, gz_mode, compresslevel, filename)
    else:
        raise TypeError('filename must be a str or bytes object, or a file')
    if 't' in mode:
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


def open_pipe(cmd: str, mode: str):
    """
    Runs the command and redirects stdin/stdout depending on the mode.
    Returns a file-like object that can be read from or written to.
    """
    return Pipe(cmd, mode=mode, shell=True, bufsize=8092)


class NotALhotseManifest(Exception):
    pass


def resolve_manifest_set_class(item):
    """Returns the right *Set class for a manifest, e.g. Recording -> RecordingSet."""
    if isinstance(item, Recording):
        return RecordingSet
    if isinstance(item, SupervisionSegment):
        return SupervisionSet
    if isinstance(item, Cut):
        return CutSet
    if isinstance(item, Features):
        return FeatureSet
    raise NotALhotseManifest(f"No corresponding 'Set' class is known for item of type: {type(item)}")


class InvalidPathExtension(ValueError):
    pass


def deserialize_item(data: dict) ->Any:
    if 'shape' in data or 'array' in data:
        return deserialize_array(data)
    if 'sources' in data:
        return Recording.from_dict(data)
    if 'num_features' in data:
        return Features.from_dict(data)
    if 'type' not in data:
        return SupervisionSegment.from_dict(data)
    cut_type = data.pop('type')
    if cut_type == 'MonoCut':
        return MonoCut.from_dict(data)
    if cut_type == 'MultiCut':
        return MultiCut.from_dict(data)
    if cut_type == 'Cut':
        warnings.warn('Your manifest was created with Lhotse version earlier than v0.8, when MonoCut was called Cut. Please re-generate it with Lhotse v0.8 as it might stop working in a future version (using manifest.from_file() and then manifest.to_file() should be sufficient).')
        return MonoCut.from_dict(data)
    if cut_type == 'MixedCut':
        return MixedCut.from_dict(data)
    raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")


def exactly_one_not_null(*args) ->bool:
    not_null = [(arg is not None) for arg in args]
    return sum(not_null) == 1


def split_sequence(seq: Sequence[Any], num_splits: int, shuffle: bool=False, drop_last: bool=False) ->List[List[Any]]:
    """
    Split a sequence into ``num_splits`` equal parts. The element order can be randomized.
    Raises a ``ValueError`` if ``num_splits`` is larger than ``len(seq)``.

    :param seq: an input iterable (can be a Lhotse manifest).
    :param num_splits: how many output splits should be created.
    :param shuffle: optionally shuffle the sequence before splitting.
    :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
        by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
        When ``True``, it may discard the last element in some splits to ensure they are
        equally long.
    :return: a list of length ``num_splits`` containing smaller lists (the splits).
    """
    seq = list(seq)
    num_items = len(seq)
    if num_splits > num_items:
        raise ValueError(f'Cannot split iterable into more chunks ({num_splits}) than its number of items {num_items}')
    if shuffle:
        random.shuffle(seq)
    chunk_size = num_items // num_splits
    num_shifts = num_items % num_splits
    if drop_last:
        end_shifts = [0] * num_splits
        begin_shifts = [0] * num_splits
    else:
        end_shifts = list(range(1, num_shifts + 1)) + [num_shifts] * (num_splits - num_shifts)
        begin_shifts = [0] + end_shifts[:-1]
    split_indices = [[i * chunk_size + begin_shift, (i + 1) * chunk_size + end_shift] for i, begin_shift, end_shift in zip(range(num_splits), begin_shifts, end_shifts)]
    splits = [seq[begin:end] for begin, end in split_indices]
    return splits


Channels = Union[int, List[int]]


class DurationMismatchError(Exception):
    pass


FileObject = Any


LHOTSE_CACHING_ENABLED = False


def is_caching_enabled() ->bool:
    return LHOTSE_CACHING_ENABLED


def dynamic_lru_cache(method: Callable) ->Callable:
    """
    Least-recently-used cache decorator.

    It enhances Python's built-in ``lru_cache`` with a dynamic
    lookup of whether to apply the cached, or noncached variant
    of the decorated function.

    To disable/enable caching globally in Lhotse, call::

        >>> from lhotse import set_caching_enabled
        >>> set_caching_enabled(True)   # enable
        >>> set_caching_enabled(False)  # disable

    Currently it hard-codes the cache size at 512 items
    (regardless of the array size).

    Check :meth:`functools.lru_cache` for more details.
    """
    global LHOTSE_CACHED_METHOD_REGISTRY
    name = method.__qualname__
    if name in LHOTSE_CACHED_METHOD_REGISTRY['cached']:
        raise ValueError(f"Method '{name}' is already cached. We don't support caching different methods which have the same __qualname__ attribute (i.e., class name + method name).")
    LHOTSE_CACHED_METHOD_REGISTRY['noncached'][name] = method
    LHOTSE_CACHED_METHOD_REGISTRY['cached'][name] = lru_cache(maxsize=512)(method)

    @wraps(method)
    def wrapper(*args, **kwargs) ->Any:
        if is_caching_enabled():
            m = LHOTSE_CACHED_METHOD_REGISTRY['cached'][name]
        else:
            m = LHOTSE_CACHED_METHOD_REGISTRY['noncached'][name]
        return m(*args, **kwargs)
    return wrapper


@lru_cache(maxsize=1)
def _available_audioread_backends():
    """
    Reduces the overhead of ``audioread.audio_open()`` when called repeatedly
    by caching the results of scanning for FFMPEG etc.
    """
    backends = audioread.available_backends()
    logging.info(f'Using audioread. Available backends: {backends}')
    return backends


def _buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    This function is based on librosa:
    https://github.com/librosa/librosa/blob/main/librosa/util/utils.py#L1312

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)
    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """
    scale = 1.0 / float(1 << 8 * n_bytes - 1)
    fmt = '<i{:d}'.format(n_bytes)
    return scale * np.frombuffer(x, fmt).astype(dtype)


class AudioLoadingError(Exception):
    pass


def verbose_audio_loading_exceptions() ->bool:
    return os.environ.get('LHOTSE_AUDIO_LOADING_EXCEPTION_VERBOSE') == '1'


def parse_channel_from_ffmpeg_output(ffmpeg_stderr: bytes) ->str:
    pattern = re.compile('^\\s*Stream #0:0.*: Audio: pcm_f32le.+(mono|stereo).+\\s*$')
    for line in ffmpeg_stderr.splitlines():
        try:
            line = line.decode()
        except UnicodeDecodeError:
            continue
        match = pattern.match(line)
        if match is not None:
            return match.group(1)
    raise ValueError(f'Could not determine the number of channels for OPUS file from the following ffmpeg output (shown as bytestring due to avoid possible encoding issues):\n{str(ffmpeg_stderr)}')


class LibsndfileCompatibleAudioInfo(NamedTuple):
    channels: int
    frames: int
    samplerate: int
    duration: float


@lru_cache(maxsize=1)
def torchaudio_supports_ffmpeg() ->bool:
    """
    Returns ``True`` when torchaudio version is at least 0.12.0, which
    has support for FFMPEG streamer API.
    """
    import torchaudio
    return version.parse(torchaudio.__version__) >= version.parse('0.12.0')


@lru_cache(maxsize=1)
def get_default_audio_backend():
    """
    Return a backend that can be used to read all audio formats supported by Lhotse.

    It first looks for special cases that need very specific handling
    (such as: opus, sphere/shorten, in-memory buffers)
    and tries to match them against relevant audio backends.

    Then, it tries to use several audio loading libraries (torchaudio, soundfile, audioread).
    In case the first fails, it tries the next one, and so on.
    """
    return CompositeAudioBackend([FfmpegSubprocessOpusBackend(), Sph2pipeSubprocessBackend(), LibsndfileBackend(), TorchaudioDefaultBackend(), AudioreadBackend()])


class AudioTransform:
    """
    Base class for all audio transforms that are going to be lazily applied on
    ``Recording`` during loading the audio into memory.

    Any ``AudioTransform`` can be used like a Python function, that expects two arguments:
    a numpy array of samples, and a sampling rate. E.g.:

        >>> fn = AudioTransform.from_dict(...)
        >>> new_audio = fn(audio, sampling_rate)

    Since we often use cuts of the original recording, they will refer to the timestamps
    of the augmented audio (which might be speed perturbed and of different duration).
    Each transform provides a helper method to recover the original audio timestamps:

        >>> # When fn does speed perturbation:
        >>> fn.reverse_timestamps(offset=5.055555, duration=10.1111111, sampling_rate=16000)
        ... (5.0, 10.0)

    Furthermore, ``AudioTransform`` can be easily (de)serialized to/from dict
    that contains its name and parameters.
    This enables storing recording and cut manifests with the transform info
    inside, avoiding the need to store the augmented recording version on disk.

    All audio transforms derived from this class are "automagically" registered,
    so that ``AudioTransform.from_dict()`` can "find" the right type given its name
    to instantiate a specific transform object.
    All child classes are expected to be decorated with a ``@dataclass`` decorator.
    """
    KNOWN_TRANSFORMS = {}

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in AudioTransform.KNOWN_TRANSFORMS:
            AudioTransform.KNOWN_TRANSFORMS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    def to_dict(self) ->dict:
        data = asdict(self)
        return {'name': type(self).__name__, 'kwargs': data}

    @staticmethod
    def from_dict(data: dict) ->'AudioTransform':
        assert data['name'] in AudioTransform.KNOWN_TRANSFORMS, f"Unknown transform type: {data['name']}"
        return AudioTransform.KNOWN_TRANSFORMS[data['name']](**data['kwargs'])

    def __call__(self, samples: np.ndarray, sampling_rate: int) ->np.ndarray:
        """
        Apply transform.

        To be implemented in derived classes.
        """
        raise NotImplementedError

    def reverse_timestamps(self, offset: Seconds, duration: Optional[Seconds], sampling_rate: int) ->Tuple[Seconds, Optional[Seconds]]:
        """
        Convert ``offset`` and ``duration`` timestamps to be adequate for the audio before the transform.
        Useful for on-the-fly augmentation when a particular chunk of audio needs to be read from disk.

        To be implemented in derived classes.
        """
        raise NotImplementedError


def during_docs_build() ->bool:
    return bool(os.environ.get('READTHEDOCS'))


def check_torchaudio_version():
    import torchaudio
    if not during_docs_build() and _version(torchaudio.__version__) < _version('0.7'):
        warnings.warn('Torchaudio SoX effects chains are only introduced in version 0.7 - please upgrade your PyTorch to 1.7.1 and torchaudio to 0.7.2 (or higher) to use them.')


def get_or_create_resampler(source_sampling_rate: int, target_sampling_rate: int) ->torch.nn.Module:
    global _precompiled_resamplers
    tpl = source_sampling_rate, target_sampling_rate
    if tpl not in _precompiled_resamplers:
        check_torchaudio_version()
        import torchaudio
        _precompiled_resamplers[tpl] = torchaudio.transforms.Resample(source_sampling_rate, target_sampling_rate)
    return _precompiled_resamplers[tpl]


_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.
    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass
    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def convolve1d(signal: torch.Tensor, kernel: torch.Tensor) ->torch.Tensor:
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    Both signal and kernel must be 1-dimensional.
    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: torch.Tensor Convolution of signal with kernel. Returns the full convolution, i.e.,
        the output tensor will have size m + n - 1, where m is the length of the
        signal and n is the length of the kernel.
    """
    assert signal.ndim == 1 and kernel.ndim == 1, 'signal and kernel must be 1-dimensional'
    m = signal.size(-1)
    n = kernel.size(-1)
    padded_size = m + n - 1
    fast_ftt_size = next_fast_len(padded_size)
    f_signal = rfft(signal, n=fast_ftt_size)
    f_kernel = rfft(kernel, n=fast_ftt_size)
    f_result = f_signal * f_kernel
    result = irfft(f_result, n=fast_ftt_size)
    return result[:padded_size]


class SetContainingAnything:

    def __contains__(self, item):
        return True

    def intersection(self, iterable):
        return True


def perturb_num_samples(num_samples: int, factor: float) ->int:
    """Mimicks the behavior of the speed perturbation on the number of samples."""
    rounding = ROUND_HALF_UP if factor >= 1.0 else ROUND_HALF_DOWN
    return int(Decimal(round(num_samples / factor, ndigits=8)).quantize(0, rounding=rounding))


def rich_exception_info(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise type(e)(f'{e}\n[extra info] When calling: {fn.__qualname__}(args={args} kwargs={kwargs})')
    return wrapper


@lru_cache(maxsize=1)
def torchaudio_soundfile_supports_format() ->bool:
    """
    Returns ``True`` when torchaudio version is at least 0.9.0, which
    has support for ``format`` keyword arg in ``torchaudio.save()``.
    """
    import torchaudio
    return version.parse(torchaudio.__version__) >= version.parse('0.9.0')


def index_by_id_and_check(manifests: Iterable[T]) ->Dict[str, T]:
    id2man = {}
    for m in manifests:
        assert m.id not in id2man, f'Duplicated manifest ID: {m.id}'
        id2man[m.id] = m
    return id2man


def add_durations(*durs: Seconds, sampling_rate: int) ->Seconds:
    """
    Adds two durations in a way that avoids floating point precision issues.
    The durations in seconds are first converted to audio sample counts,
    then added, and finally converted back to floating point seconds.
    """
    tot_num_samples = sum(compute_num_samples(d, sampling_rate=sampling_rate) for d in durs)
    return tot_num_samples / sampling_rate


class AlignmentItem(NamedTuple):
    """
    This class contains an alignment item, for example a word, along with its
    start time (w.r.t. the start of recording) and duration. It can potentially
    be used to store other kinds of alignment items, such as subwords, pdfid's etc.
    """
    symbol: str
    start: Seconds
    duration: Seconds
    score: Optional[float] = None

    @staticmethod
    def deserialize(data: Union[List, Dict]) ->'AlignmentItem':
        if isinstance(data, dict):
            return AlignmentItem(*list(data.values()))
        return AlignmentItem(*data)

    def serialize(self) ->list:
        return list(self)

    @property
    def end(self) ->Seconds:
        return round(self.start + self.duration, ndigits=8)

    def with_offset(self, offset: Seconds) ->'AlignmentItem':
        """Return an identical ``AlignmentItem``, but with the ``offset`` added to the ``start`` field."""
        return AlignmentItem(start=add_durations(self.start, offset, sampling_rate=48000), duration=self.duration, symbol=self.symbol, score=self.score)

    def perturb_speed(self, factor: float, sampling_rate: int) ->'AlignmentItem':
        """
        Return an ``AlignmentItem`` that has time boundaries matching the
        recording/cut perturbed with the same factor. See :meth:`SupervisionSegment.perturb_speed`
        for details.
        """
        start_sample = compute_num_samples(self.start, sampling_rate)
        num_samples = compute_num_samples(self.duration, sampling_rate)
        new_start = perturb_num_samples(start_sample, factor) / sampling_rate
        new_duration = perturb_num_samples(num_samples, factor) / sampling_rate
        return AlignmentItem(symbol=self.symbol, start=new_start, duration=new_duration, score=self.score)

    def trim(self, end: Seconds, start: Seconds=0) ->'AlignmentItem':
        """
        See :meth:`SupervisionSegment.trim`.
        """
        assert start >= 0
        start_exceeds_by = abs(min(0, self.start - start))
        end_exceeds_by = max(0, self.end - end)
        return AlignmentItem(symbol=self.symbol, start=max(start, self.start), duration=add_durations(self.duration, -end_exceeds_by, -start_exceeds_by, sampling_rate=48000))

    def transform(self, transform_fn: Callable[[str], str]) ->'AlignmentItem':
        """
        Perform specified transformation on the alignment content.
        """
        return AlignmentItem(symbol=transform_fn(self.symbol), start=self.start, duration=self.duration, score=self.score)


def to_list(item: Union[Any, Sequence[Any]]) ->List[Any]:
    """Convert ``item`` to a list if it is not already a list."""
    return item if isinstance(item, list) else [item]


def is_equal_or_contains(value: Union[T, Sequence[T]], other: Union[T, Sequence[T]]) ->bool:
    value = to_list(value)
    other = to_list(other)
    return set(other).issubset(set(value))


def overspans(spanning: Any, spanned: Any) ->bool:
    """Indicates whether the left-hand-side time-span/segment covers the whole right-hand-side time-span/segment."""
    return spanning.start <= spanned.start <= spanned.end <= spanning.end


class LazyFilter(ImitatesDict):
    """
    A wrapper over an iterable that enables lazy filtering.
    It works like Python's `filter` built-in by applying the filter predicate
    to each element and yielding it further if predicate returned ``True``.
    """

    def __init__(self, iterator: Iterable, predicate: Callable[[Any], bool]) ->None:
        self.iterator = iterator
        self.predicate = predicate
        assert callable(self.predicate), f"LazyFilter: 'predicate' arg must be callable (got {predicate})."
        if isinstance(self.predicate, types.LambdaType) and self.predicate.__name__ == '<lambda>' and not is_module_available('dill'):
            warnings.warn('A lambda was passed to LazyFilter: it may prevent you from forking this process. If you experience issues with num_workers > 0 in torch.utils.data.DataLoader, try passing a regular function instead.')

    def __iter__(self):
        return filter(self.predicate, self.iterator)

    def __add__(self, other) ->'LazyIteratorChain':
        return LazyIteratorChain(self, other)

    def __len__(self) ->int:
        raise NotImplementedError('LazyFilter does not support __len__ because it would require iterating over the whole iterator, which is not possible in a lazy fashion. If you really need to know the length, convert to eager mode first using `.to_eager()`. Note that this will require loading the whole iterator into memory.')


class LazyIteratorMultiplexer(ImitatesDict):
    """
    A wrapper over multiple iterators that enables to combine lazy manifests in Lhotse.
    During iteration, unlike :class:`.LazyIteratorChain`, :class:`.LazyIteratorMultiplexer`
    at each step randomly selects the iterable used to yield an item.

    Since the iterables might be of different length, we provide a ``weights`` parameter
    to let the user decide which iterables should be sampled more frequently than others.
    When an iterable is exhausted, we will keep sampling from the other iterables, until
    we exhaust them all, unless ``stop_early`` is set to ``True``.
    """

    def __init__(self, *iterators: Iterable, stop_early: bool=False, weights: Optional[List[Union[int, float]]]=None, seed: int=0) ->None:
        self.iterators = list(iterators)
        self.stop_early = stop_early
        self.seed = seed
        assert len(self.iterators) > 1, 'There have to be at least two iterables to multiplex.'
        if weights is None:
            self.weights = [1] * len(self.iterators)
        else:
            self.weights = weights
        assert len(self.iterators) == len(self.weights)

    def __iter__(self):
        rng = random.Random(self.seed)
        iters = [iter(it) for it in self.iterators]
        exhausted = [(False) for _ in range(len(iters))]

        def should_continue():
            if self.stop_early:
                return not any(exhausted)
            else:
                return not all(exhausted)
        while should_continue():
            active_indexes, active_weights = zip(*[(i, w) for i, (is_exhausted, w) in enumerate(zip(exhausted, self.weights)) if not is_exhausted])
            idx = rng.choices(active_indexes, weights=active_weights, k=1)[0]
            selected = iters[idx]
            try:
                item = next(selected)
                yield item
            except StopIteration:
                exhausted[idx] = True
                continue

    def __len__(self) ->int:
        return sum(len(it) for it in self.iterators)

    def __add__(self, other) ->'LazyIteratorChain':
        return LazyIteratorChain(self, other)


class LazyMapper(ImitatesDict):
    """
    A wrapper over an iterable that enables lazy function evaluation on each item.
    It works like Python's `map` built-in by applying a callable ``fn``
    to each element ``x`` and yielding the result of ``fn(x)`` further.
    """

    def __init__(self, iterator: Iterable, fn: Callable[[Any], Any]) ->None:
        self.iterator = iterator
        self.fn = fn
        assert callable(self.fn), f"LazyMapper: 'fn' arg must be callable (got {fn})."
        if isinstance(self.fn, types.LambdaType) and self.fn.__name__ == '<lambda>' and not is_module_available('dill'):
            warnings.warn('A lambda was passed to LazyMapper: it may prevent you from forking this process. If you experience issues with num_workers > 0 in torch.utils.data.DataLoader, try passing a regular function instead.')

    def __iter__(self):
        return map(self.fn, self.iterator)

    def __len__(self) ->int:
        return len(self.iterator)

    def __add__(self, other) ->'LazyIteratorChain':
        return LazyIteratorChain(self, other)


def attach_repeat_idx_to_id(item: Any, idx: int) ->Any:
    if not hasattr(item, 'id'):
        return item
    return fastcopy(item, id=f'{item.id}_repeat{idx}')


class LazyRepeater(ImitatesDict):
    """
    A wrapper over an iterable that enables to repeat it N times or infinitely (default).
    """

    def __init__(self, iterator: Iterable, times: Optional[int]=None, preserve_id: bool=False) ->None:
        self.iterator = iterator
        self.times = times
        self.preserve_id = preserve_id
        assert self.times is None or self.times > 0

    def __iter__(self):
        epoch = 0
        while self.times is None or epoch < self.times:
            if self.preserve_id:
                iterator = self.iterator
            else:
                iterator = LazyMapper(self.iterator, partial(attach_repeat_idx_to_id, idx=epoch))
            yield from iterator
            epoch += 1

    def __len__(self) ->int:
        if self.times is None:
            raise AttributeError()
        return len(self.iterator) * self.times

    def __add__(self, other) ->'LazyIteratorChain':
        return LazyIteratorChain(self, other)


def streaming_shuffle(data: Iterator[T], bufsize: int=10000, rng: Optional[random.Random]=None) ->Generator[T, None, None]:
    """
    Shuffle the data in the stream.

    This uses a buffer of size ``bufsize``. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    This code is mostly borrowed from WebDataset; note that we use much larger default
    buffer size because Cuts are very lightweight and fast to read.
    https://github.com/webdataset/webdataset/blob/master/webdataset/iterators.py#L145

    .. warning: The order of the elements is expected to be much less random than
        if the whole sequence was shuffled before-hand with standard methods like
        ``random.shuffle``.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :param rng: either random module or random.Random instance
    :return: a generator of cuts, shuffled on-the-fly.
    """
    if rng is None:
        rng = random
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            try:
                buf.append(next(data))
            except StopIteration:
                pass
        if len(buf) > 0:
            k = rng.randint(0, len(buf) - 1)
            sample, buf[k] = buf[k], sample
        if startup and len(buf) < bufsize:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample


class LazyShuffler(ImitatesDict):
    """
    A wrapper over an iterable that enables lazy shuffling.
    The shuffling algorithm is reservoir-sampling based.
    See :func:`lhotse.utils.streaming_shuffle` for details.
    """

    def __init__(self, iterator: Iterable, buffer_size: int=10000, rng: Optional[random.Random]=None) ->None:
        self.iterator = iterator
        self.buffer_size = buffer_size
        self.rng = rng

    def __iter__(self):
        return iter(streaming_shuffle(iter(self.iterator), bufsize=self.buffer_size, rng=self.rng))

    def __len__(self) ->int:
        return len(self.iterator)

    def __add__(self, other) ->'LazyIteratorChain':
        return LazyIteratorChain(self, other)


AugmentFn = Callable[[np.ndarray, int], np.ndarray]


Decibels = float


FEATURE_EXTRACTORS = {}


def get_extractor_type(name: str) ->Type:
    """
    Return the feature extractor type corresponding to the given name.

    :param name: specifies which feature extractor should be used.
    :return: A feature extractors type.
    """
    return FEATURE_EXTRACTORS[name]


def uuid4():
    """
    Generates uuid4's exactly like Python's uuid.uuid4() function.
    When ``fix_random_seed()`` is called, it will instead generate deterministic IDs.
    """
    if _lhotse_uuid is not None:
        return _lhotse_uuid()
    return uuid.uuid4()


def compute_num_windows(sig_len: Seconds, win_len: Seconds, hop: Seconds) ->int:
    """
    Return a number of windows obtained from signal of length equal to ``sig_len``
    with windows of ``win_len`` and ``hop`` denoting shift between windows.
    Examples:
    ```
      (sig_len,win_len,hop) -> num_windows # list of windows times
      (1, 6.1, 3) -> 1  # 0-1
      (3, 1, 6.1) -> 1  # 0-1
      (3, 6.1, 1) -> 1  # 0-3
      (5.9, 1, 3) -> 2  # 0-1, 3-4
      (5.9, 3, 1) -> 4  # 0-3, 1-4, 2-5, 3-5.9
      (6.1, 1, 3) -> 3  # 0-1, 3-4, 6-6.1
      (6.1, 3, 1) -> 5  # 0-3, 1-4, 2-5, 3-6, 4-6.1
      (5.9, 3, 3) -> 2  # 0-3, 3-5.9
      (6.1, 3, 3) -> 3  # 0-3, 3-6, 6-6.1
      (0.0, 3, 3) -> 0
    ```
    :param sig_len: Signal length in seconds.
    :param win_len: Window length in seconds
    :param hop: Shift between windows in seconds.
    :return: Number of windows in signal.
    """
    n = ceil(max(sig_len - win_len, 0) / hop)
    b = sig_len - n * hop > 0
    return (sig_len > 0) * (n + int(b))


def overlaps(lhs: Any, rhs: Any) ->bool:
    """Indicates whether two time-spans/segments are overlapping or not."""
    return lhs.start < rhs.end and rhs.start < lhs.end and not isclose(lhs.start, rhs.end) and not isclose(rhs.start, lhs.end)


def to_hashable(item: Any) ->Any:
    """Convert ``item`` to a hashable type if it is not already hashable."""
    return tuple(item) if isinstance(item, list) else item


EPSILON = 1e-10


LOG_EPSILON = math.log(EPSILON)


def measure_overlap(lhs: Any, rhs: Any) ->float:
    """
    Given two objects with "start" and "end" attributes, return the % of their overlapped time
    with regard to the shorter of the two spans.
    ."""
    lhs, rhs = sorted([lhs, rhs], key=lambda item: item.start)
    overlapped_area = lhs.end - rhs.start
    if overlapped_area <= 0:
        return 0.0
    dur = min(lhs.end - lhs.start, rhs.end - rhs.start)
    return overlapped_area / dur


CHUNKY_FORMAT_CHUNK_SIZE = 500


WRITER_BACKENDS = {}


def register_writer(cls):
    """
    Decorator used to add a new ``FeaturesWriter`` to Lhotse's registry.

    Example::

        @register_writer
        class MyFeatureWriter(FeatureWriter):
            ...
    """
    WRITER_BACKENDS[cls.name] = cls
    return cls


class NonPositiveEnergyError(ValueError):
    pass


def audio_energy(audio: np.ndarray) ->float:
    return float(np.average(audio ** 2))


class AudioMixer:
    """
    Utility class to mix multiple waveforms into a single one.
    It should be instantiated separately for each mixing session (i.e. each ``MixedCut``
    will create a separate ``AudioMixer`` to mix its tracks).
    It is initialized with a numpy array of audio samples (typically float32 in [-1, 1] range)
    that represents the "reference" signal for the mix.
    Other signals can be mixed to it with different time offsets and SNRs using the
    ``add_to_mix`` method.
    The time offset is relative to the start of the reference signal
    (only positive values are supported).
    The SNR is relative to the energy of the signal used to initialize the ``AudioMixer``.

    .. note:: Both single-channel and multi-channel signals are supported as reference
        and added signals. The only requirement is that the when mixing 2 multi-channel
        signals, they must have the same number of channels.

    .. note:: When the AudioMixer contains multi-channel tracks, 2 types of mixed signals
        can be generated:
        - `mixed_audio` mixes each channel independently, and returns a multi-channel signal.
          If there is a mono track, it is added to all the channels.
        - `mixed_mono_audio` mixes all channels together, and returns a single-channel signal.
    """

    def __init__(self, base_audio: np.ndarray, sampling_rate: int, reference_energy: Optional[float]=None):
        """
        AudioMixer's constructor.

        :param base_audio: A numpy array with the audio samples for the base signal
            (all the other signals will be mixed to it).
        :param sampling_rate: Sampling rate of the audio.
        :param reference_energy: Optionally pass a reference energy value to compute SNRs against.
            This might be required when ``base_audio`` corresponds to zero-padding.
        """
        self.tracks = [base_audio]
        self.offsets = [0]
        self.sampling_rate = sampling_rate
        self.num_channels = base_audio.shape[0]
        self.dtype = self.tracks[0].dtype
        if reference_energy is None:
            self.reference_energy = audio_energy(base_audio)
        else:
            self.reference_energy = reference_energy
        if self.reference_energy <= 0.0:
            raise NonPositiveEnergyError(f'To perform mix, energy must be non-zero and non-negative (got {self.reference_energy})')

    def _pad_track(self, audio: np.ndarray, offset: int, total: Optional[int]=None) ->np.ndarray:
        assert audio.ndim == 2, f'audio.ndim={audio.ndim}'
        if total is None:
            total = audio.shape[1] + offset
        assert audio.shape[1] + offset <= total, f'{audio.shape[1]} + {offset} <= {total}'
        return np.pad(audio, pad_width=((0, 0), (offset, total - audio.shape[1] - offset)))

    @property
    def num_samples_total(self) ->int:
        longest = 0
        for offset, audio in zip(self.offsets, self.tracks):
            longest = max(longest, offset + audio.shape[1])
        return longest

    @property
    def unmixed_audio(self) ->List[np.ndarray]:
        """
        Return a list of numpy arrays with the shape (C, num_samples), where each track is
        zero padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        total = self.num_samples_total
        return [self._pad_track(track, offset=offset, total=total) for offset, track in zip(self.offsets, self.tracks)]

    @property
    def mixed_audio(self) ->np.ndarray:
        """
        Return a numpy ndarray with the shape (num_channels, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_samples_total
        mixed = np.zeros((self.num_channels, total), dtype=self.dtype)
        for offset, track in zip(self.offsets, self.tracks):
            if track.shape[0] == 1 and self.num_channels > 1:
                track = np.tile(track, (self.num_channels, 1))
            mixed[:, offset:offset + track.shape[1]] += track
        return mixed

    @property
    def mixed_mono_audio(self) ->np.ndarray:
        """
        Return a numpy ndarray with the shape (1, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_samples_total
        mixed = np.zeros((1, total), dtype=self.dtype)
        for offset, track in zip(self.offsets, self.tracks):
            if track.shape[0] > 1:
                track = np.sum(track, axis=0, keepdims=True)
            mixed[:, offset:offset + track.shape[1]] += track
        return mixed

    def add_to_mix(self, audio: np.ndarray, snr: Optional[Decibels]=None, offset: Seconds=0.0):
        """
        Add audio of a new track into the mix.
        :param audio: An array of audio samples to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `audio` represents noise (positive SNR - lower `audio` energy,
        negative SNR - higher `audio` energy)
        :param offset: How many seconds to shift `audio` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :return:
        """
        if audio.size == 0:
            return
        assert offset >= 0.0, 'Negative offset in mixing is not supported.'
        num_samples_offset = compute_num_samples(offset, self.sampling_rate)
        gain = 1.0
        if snr is not None:
            added_audio_energy = audio_energy(audio)
            if added_audio_energy <= 0.0:
                raise NonPositiveEnergyError(f'To perform mix, energy must be non-zero and non-negative (got {added_audio_energy}). ')
            target_energy = self.reference_energy * 10.0 ** (-snr / 10)
            gain = sqrt(target_energy / added_audio_energy)
        self.tracks.append(gain * audio)
        self.offsets.append(num_samples_offset)
        if audio.shape[0] != self.num_channels and self.num_channels != 1 and audio.shape[0] != 1:
            raise ValueError(f'Cannot mix audios with {audio.shape[0]} and {self.num_channels} channels.')
        self.num_channels = max(self.num_channels, audio.shape[0])


DEFAULT_PADDING_VALUE = 0


def create_default_feature_extractor(name: str) ->'Optional[FeatureExtractor]':
    """
    Create a feature extractor object with a default configuration.

    :param name: specifies which feature extractor should be used.
    :return: A new feature extractor instance.
    """
    return get_extractor_type(name)()


def hash_str_to_int(s: str, max_value: Optional[int]=None) ->int:
    """Hash a string to an integer in the range [0, max_value)."""
    if max_value is None:
        max_value = sys.maxsize
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % max_value


def merge_items_with_delimiter(values: Iterable[str], prefix: str='cat', delimiter: str='#') ->Optional[str]:
    values = list(values)
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return delimiter.join(chain([prefix], values))


class StatsAccumulator:

    def __init__(self, feature_dim: int):
        self.total_sum = np.zeros((feature_dim,), dtype=np.float64)
        self.total_unnorm_var = np.zeros((feature_dim,), dtype=np.float64)
        self.total_frames = 0

    def update(self, arr: np.ndarray) ->None:
        with np.errstate(divide='ignore', invalid='ignore'):
            arr = arr.astype(np.float64)
            curr_sum = arr.sum(axis=0)
            updated_total_sum = self.total_sum + curr_sum
            curr_frames = arr.shape[0]
            updated_total_frames = self.total_frames + curr_frames
            total_over_curr_frames = self.total_frames / curr_frames
            curr_unnorm_var = np.var(arr, axis=0) * curr_frames
            if self.total_frames > 0:
                self.total_unnorm_var = self.total_unnorm_var + curr_unnorm_var + total_over_curr_frames / updated_total_frames * (self.total_sum / total_over_curr_frames - curr_sum) ** 2
            else:
                self.total_unnorm_var = curr_unnorm_var
            self.total_sum = updated_total_sum
            self.total_frames = updated_total_frames

    @property
    def norm_means(self) ->np.ndarray:
        return self.total_sum / self.total_frames

    @property
    def norm_stds(self) ->np.ndarray:
        return np.sqrt(self.total_unnorm_var / self.total_frames)

    def get(self) ->Dict[str, np.ndarray]:
        return {'norm_means': self.norm_means, 'norm_stds': self.norm_stds}


def _add_features_path_prefix_single(cut, path):
    return cut.with_features_path_prefix(path)


def _add_recording_path_prefix_single(cut, path):
    return cut.with_recording_path_prefix(path)


def _takewhile(iterable: Iterable[T], predicate: Callable[[T], bool]) ->Tuple[List[T], Iterable[T]]:
    """
    Collects items from ``iterable`` as long as they satisfy the ``predicate``.
    Returns a tuple of ``(collected_items, iterable)``, where ``iterable`` may
    continue yielding items starting from the first one that did not satisfy
    ``predicate`` (unlike ``itertools.takewhile``).
    """
    collected = []
    try:
        while True:
            item = next(iterable)
            if predicate(item):
                collected.append(item)
            else:
                iterable = chain([item], iterable)
                break
    except StopIteration:
        pass
    return collected, iterable


class suppress_and_warn:
    """Context manager to suppress specified exceptions that logs the error message.

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         >>> with suppress_and_warn(FileNotFoundError):
         ...     os.remove(somefile)
         >>> # Execution still resumes here if the file was already removed
    """

    def __init__(self, *exceptions, enabled: bool=True):
        self._enabled = enabled
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        if not self._enabled:
            return
        should_suppress = exctype is not None and issubclass(exctype, self._exceptions)
        if should_suppress:
            logging.warning(f'[Suppressed {exctype.__qualname__}] Error message: {excinst}')
        return should_suppress


def null_result_on_audio_loading_error(func: Callable) ->Callable:
    """
    This is a decorator that makes a function return None when reading audio with Lhotse failed.

    Example::

        >>> @null_result_on_audio_loading_error
        ... def func_loading_audio(rec):
        ...     audio = rec.load_audio()  # if this fails, will return None instead
        ...     return other_func(audio)

    Another example::

        >>> # crashes on loading audio
        >>> audio = load_audio(cut)
        >>> # does not crash on loading audio, return None instead
        >>> maybe_audio: Optional = null_result_on_audio_loading_error(load_audio)(cut)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) ->Optional:
        with suppress_audio_loading_errors():
            return func(*args, **kwargs)
    return wrapper


def random_mask_along_batch_axis(tensor: torch.Tensor, p: float=0.5) ->torch.Tensor:
    """
    For a given tensor with shape (N, d1, d2, d3, ...), returns a mask with shape (N, 1, 1, 1, ...),
    that randomly masks the samples in a batch.

    E.g. for a 2D input matrix it looks like:

        >>> [[0., 0., 0., ...],
        ...  [1., 1., 1., ...],
        ...  [0., 0., 0., ...]]

    :param tensor: the input tensor.
    :param p: the probability of masking an element.
    """
    mask_shape = (tensor.shape[0],) + tuple(1 for _ in tensor.shape[1:])
    mask = torch.rand(mask_shape) > p
    return mask


def schedule_value_for_step(schedule: Sequence[Tuple[int, T]], step: int) ->T:
    milestones, values = zip(*schedule)
    assert milestones[0] <= step, f'Cannot determine the scheduled value for step {step} with schedule: {schedule}. Did you forget to add the first part of the schedule for steps below {milestones[0]}?'
    idx = bisect.bisect_right(milestones, step) - 1
    return values[idx]


class RandomizedSmoothing(torch.nn.Module):
    """
    Randomized smoothing - gaussian noise added to an input waveform, or a batch of waveforms.
    The summed audio is clipped to ``[-1.0, 1.0]`` before returning.
    """

    def __init__(self, sigma: Union[float, Sequence[Tuple[int, float]]]=0.1, sample_sigma: bool=True, p: float=0.3):
        """
        RandomizedSmoothing's constructor.

        :param sigma: standard deviation of the gaussian noise. Either a constant float, or a schedule,
            i.e. a list of tuples that specify which value to use from which step.
            For example, ``[(0, 0.01), (1000, 0.1)]`` means that from steps 0-999, the sigma value
            will be 0.01, and from step 1000 onwards, it will be 0.1.
        :param sample_sigma: when ``False``, then sigma is used as the standard deviation in each forward step.
            When ``True``, the standard deviation is sampled from a uniform distribution of
            ``[-sigma, sigma]`` for each forward step.
        :param p: the probability of applying this transform.
        """
        super().__init__()
        self.sigma = sigma
        self.sample_sigma = sample_sigma
        self.p = p
        self.step = 0

    def forward(self, audio: torch.Tensor, *args, **kwargs) ->torch.Tensor:
        if isinstance(self.sigma, float):
            sigma = self.sigma
        else:
            sigma = schedule_value_for_step(self.sigma, self.step)
            self.step += 1
        if self.sample_sigma:
            mask_shape = (audio.shape[0],) + tuple(1 for _ in audio.shape[1:])
            sigma = sigma * (2 * torch.rand(mask_shape) - 1)
        noise = sigma * torch.randn_like(audio)
        noise_mask = random_mask_along_batch_axis(noise, p=1.0 - self.p)
        noise = noise * noise_mask
        return torch.clip(audio + noise, min=-1.0, max=1.0)


def mask_along_axis_optimized(features: torch.Tensor, mask_size: int, mask_times: int, mask_value: float, axis: int) ->torch.Tensor:
    """
    Apply Frequency and Time masking along axis.
    Frequency and Time masking as described in the SpecAugment paper.

    :param features: input tensor of shape ``(T, F)``
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    :mask_value: Value to assign to the masked regions.
    :axis: Axis to apply masking on (1 -> time, 2 -> frequency)
    """
    if axis not in [1, 2]:
        raise ValueError('Only Frequency and Time masking are supported!')
    features = features.unsqueeze(0)
    features = features.reshape([-1] + list(features.size()[-2:]))
    values = torch.randint(int(0), int(mask_size), (1, mask_times))
    min_values = torch.rand(1, mask_times) * (features.size(axis) - values)
    mask_starts = min_values.long().squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()
    if axis == 1:
        if mask_times == 1:
            features[:, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for mask_start, mask_end in zip(mask_starts, mask_ends):
            features[:, mask_start:mask_end] = mask_value
    else:
        if mask_times == 1:
            features[:, :, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for mask_start, mask_end in zip(mask_starts, mask_ends):
            features[:, :, mask_start:mask_end] = mask_value
    features = features.squeeze(0)
    return features


def time_warp(features: torch.Tensor, factor: int) ->torch.Tensor:
    """
    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(T, F)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(T, F)``
    """
    t = features.size(0)
    if t - factor <= factor + 1:
        return features
    center = np.random.randint(factor + 1, t - factor)
    warped = np.random.randint(center - factor, center + factor + 1)
    if warped == center:
        return features
    features = features.unsqueeze(0).unsqueeze(0)
    left = torch.nn.functional.interpolate(features[:, :, :center, :], size=(warped, features.size(3)), mode='bicubic', align_corners=False)
    right = torch.nn.functional.interpolate(features[:, :, center:, :], size=(t - warped, features.size(3)), mode='bicubic', align_corners=False)
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0)


class SpecAugment(torch.nn.Module):
    """
    SpecAugment performs three augmentations:
    - time warping of the feature matrix
    - masking of ranges of features (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(self, time_warp_factor: Optional[int]=80, num_feature_masks: int=2, features_mask_size: int=27, num_frame_masks: int=10, frames_mask_size: int=100, max_frames_mask_fraction: float=0.15, p=0.9):
        """
        SpecAugment's constructor.

        :param time_warp_factor: parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
        :param num_feature_masks: how many feature masks should be applied. Set to ``0`` to disable.
        :param features_mask_size: the width of the feature mask (expressed in the number of masked feature bins).
            This is the ``F`` parameter from the SpecAugment paper.
        :param num_frame_masks: the number of masking regions for utterances. Set to ``0`` to disable.
        :param frames_mask_size: the width of the frame (temporal) masks (expressed in the number of masked frames).
            This is the ``T`` parameter from the SpecAugment paper.
        :param max_frames_mask_fraction: limits the size of the frame (temporal) mask to this value times the length
            of the utterance (or supervision segment).
            This is the parameter denoted by ``p`` in the SpecAugment paper.
        :param p: the probability of applying this transform.
            It is different from ``p`` in the SpecAugment paper!
        """
        super().__init__()
        assert 0 <= p <= 1
        assert num_feature_masks >= 0
        assert num_frame_masks >= 0
        assert features_mask_size > 0
        assert frames_mask_size > 0
        self.time_warp_factor = time_warp_factor
        self.num_feature_masks = num_feature_masks
        self.features_mask_size = features_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.max_frames_mask_fraction = max_frames_mask_fraction
        self.p = p

    def forward(self, features: torch.Tensor, supervision_segments: Optional[torch.IntTensor]=None, *args, **kwargs) ->torch.Tensor:
        """
        Computes SpecAugment for a batch of feature matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``supervision_segments`` tensor that will be used to apply SpecAugment
        only to selected areas of the input. The format of this input is described below.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param supervision_segments: an int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features`` -- there may be either
            less or more than the batch size.
            The second dimension encoder three kinds of information:
            the sequence index of the corresponding feature matrix in `features`,
            the start frame index, and the number of frames for each segment.
        :return: an augmented tensor of shape ``(B, T, F)``.
        """
        assert len(features.shape) == 3, 'SpecAugment only supports batches of single-channel feature matrices.'
        features = features.clone()
        if supervision_segments is None:
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(features[sequence_idx])
        else:
            for sequence_idx, start_frame, num_frames in supervision_segments:
                end_frame = start_frame + num_frames
                features[sequence_idx, start_frame:end_frame] = self._forward_single(features[sequence_idx, start_frame:end_frame], warp=True, mask=False)
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(features[sequence_idx], warp=False, mask=True)
        return features

    def _forward_single(self, features: torch.Tensor, warp: bool=True, mask: bool=True) ->torch.Tensor:
        """
        Apply SpecAugment to a single feature matrix of shape (T, F).
        """
        if random.random() > self.p:
            return features
        if warp:
            if self.time_warp_factor is not None and self.time_warp_factor >= 1:
                features = time_warp(features, factor=self.time_warp_factor)
        if mask:
            mean = features.mean()
            features = mask_along_axis_optimized(features, mask_size=self.features_mask_size, mask_times=self.num_feature_masks, mask_value=mean, axis=2)
            max_tot_mask_frames = self.max_frames_mask_fraction * features.size(0)
            num_frame_masks = min(self.num_frame_masks, math.ceil(max_tot_mask_frames / self.frames_mask_size))
            max_mask_frames = min(self.frames_mask_size, max_tot_mask_frames // num_frame_masks)
            features = mask_along_axis_optimized(features, mask_size=max_mask_frames, mask_times=num_frame_masks, mask_value=mean, axis=1)
        return features

    def state_dict(self) ->Dict:
        return dict(time_warp_factor=self.time_warp_factor, num_feature_masks=self.num_feature_masks, features_mask_size=self.features_mask_size, num_frame_masks=self.num_frame_masks, frames_mask_size=self.frames_mask_size, max_frames_mask_fraction=self.max_frames_mask_fraction, p=self.p)

    def load_state_dict(self, state_dict: Dict):
        self.time_warp_factor = state_dict.get('time_warp_factor', self.time_warp_factor)
        self.num_feature_masks = state_dict.get('num_feature_masks', self.num_feature_masks)
        self.features_mask_size = state_dict.get('features_mask_size', self.features_mask_size)
        self.num_frame_masks = state_dict.get('num_frame_masks', self.num_frame_masks)
        self.frames_mask_size = state_dict.get('frames_mask_size', self.frames_mask_size)
        self.max_frames_mask_fraction = state_dict.get('max_frames_mask_fraction', self.max_frames_mask_fraction)
        self.p = state_dict.get('p', self.p)


def dereverb_wpe_torch(audio: torch.Tensor, n_fft: int=512, hop_length: int=128, taps: int=10, delay: int=3, iterations: int=3, statistics_mode: str='full') ->torch.Tensor:
    if not is_module_available('nara_wpe'):
        raise ImportError("Please install nara_wpe first using 'pip install git+https://github.com/fgnt/nara_wpe' (at the time of writing, only GitHub version has a PyTorch implementation).")
    assert audio.ndim == 2
    window = torch.blackman_window(n_fft)
    Y = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
    Y = Y.permute(1, 0, 2)
    Z = wpe_v6(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode=statistics_mode)
    z = torch.istft(Z.permute(1, 0, 2), n_fft=n_fft, hop_length=hop_length, window=window)
    return z


class DereverbWPE(torch.nn.Module):
    """
    Dereverberation with Weighted Prediction Error (WPE).
    The implementation and default values are borrowed from `nara_wpe` package:
    https://github.com/fgnt/nara_wpe

    The method and library are described in the following paper:
    https://groups.uni-paderborn.de/nt/pubs/2018/ITG_2018_Drude_Paper.pdf
    """

    def __init__(self, n_fft: int=512, hop_length: int=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, audio: torch.Tensor, *args, **kwargs) ->torch.Tensor:
        """
        Expects audio to be 2D or 3D tensor.
        2D means a batch of single-channel audio, shape (B, T).
        3D means a batch of multi-channel audio, shape (B, D, T).
        B => batch size; D => number of channels; T => number of audio samples.
        """
        if audio.ndim == 2:
            return torch.cat([dereverb_wpe_torch(a.unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop_length) for a in audio], dim=0)
        assert audio.ndim == 3
        return torch.stack([dereverb_wpe_torch(a, n_fft=self.n_fft, hop_length=self.hop_length) for a in audio], dim=0)


def _get_log_energy(x: torch.Tensor, energy_floor: float) ->torch.Tensor:
    """
    Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = (x.pow(2).sum(-1) + 1e-15).log()
    if energy_floor > 0.0:
        log_energy = torch.max(log_energy, torch.tensor(math.log(energy_floor), dtype=log_energy.dtype))
    return log_energy


def _get_strided_batch(waveform: torch.Tensor, window_length: int, window_shift: int, snip_edges: bool) ->torch.Tensor:
    """Given a waveform (2D tensor of size ``(batch_size, num_samples)``,
    it returns a 2D tensor ``(batch_size, num_frames, window_length)``
    representing how the window is shifted along the waveform. Each row is a frame.
    Args:
        waveform (torch.Tensor): Tensor of size ``(batch_size, num_samples)``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
    Returns:
        torch.Tensor: 3D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    num_samples = waveform.size(-1)
    if snip_edges:
        if num_samples < window_length:
            return torch.empty((0, 0, 0))
        else:
            num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        num_frames = (num_samples + window_shift // 2) // window_shift
        new_num_samples = (num_frames - 1) * window_shift + window_length
        npad = new_num_samples - num_samples
        npad_left = int((window_length - window_shift) // 2)
        npad_right = npad - npad_left
        pad_left = torch.flip(waveform[:, :npad_left], (1,))
        if npad_right >= 0:
            pad_right = torch.flip(waveform[:, -npad_right:], (1,))
        else:
            pad_right = torch.zeros(0, dtype=waveform.dtype)
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)
    strides = waveform.stride(0), window_shift * waveform.stride(1), waveform.stride(1)
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides)


def _get_strided_batch_streaming(waveform: torch.Tensor, window_shift: int, window_length: int, prev_remainder: Optional[torch.Tensor]=None, snip_edges: bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    A variant of _get_strided_batch that creates short frames of a batch of audio signals
    in a way suitable for streaming. It accepts a waveform, window size parameters, and
    an optional buffer of previously unused samples. It returns a pair of waveform windows tensor,
    and unused part of the waveform to be passed as ``prev_remainder`` in the next call to this
    function.

    Example usage::

        >>> # get the first buffer of audio and make frames
        >>> waveform = get_incoming_audio_from_mic()
        >>> frames, remainder = _get_strided_batch_streaming(
        ...     waveform,
        ...     window_shift=160,
        ...     window_length=200,
        ... )
        >>>
        >>> process(frames)  # do sth with the frames
        >>>
        >>> # get the next buffer and use previous remainder to make frames
        >>> waveform = get_incoming_audio_from_mic()
        >>> frames, remainder = _get_strided_batch_streaming(
        ...     waveform,
        ...     window_shift=160,
        ...     window_length=200,
        ...     prev_remainder=prev_remainder,
        ... )

    :param waveform: A waveform tensor of shape ``(batch_size, num_samples)``.
    :param window_shift: The shift between frames measured in the number of samples.
    :param window_length: The number of samples in each window (frame).
    :param prev_remainder: An optional waveform tensor of shape ``(batch_size, num_samples)``.
        Can be ``None`` which indicates the start of a recording.
    :param snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
        in the file, and the number of frames depends on the frame_length.  If False, the number of frames
        depends only on the frame_shift, and we reflect the data at the ends.
    :return: a pair of tensors with shapes ``(batch_size, num_frames, window_length)`` and
        ``(batch_size, remainder_len)``.
    """
    assert window_shift <= window_length
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    if prev_remainder is None:
        if not snip_edges:
            npad_left = int((window_length - window_shift) // 2)
            pad_left = torch.flip(waveform[:, :npad_left], (1,))
            waveform = torch.cat((pad_left, waveform), dim=1)
    else:
        assert prev_remainder.dim() == 2
        assert prev_remainder.size(0) == batch_size
        waveform = torch.cat((prev_remainder, waveform), dim=1)
    num_samples = waveform.size(-1)
    if snip_edges:
        if num_samples < window_length:
            return torch.empty((batch_size, 0, 0)), torch.empty(batch_size, 0)
        num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        window_remainder = window_length - window_shift
        num_frames = (num_samples - window_remainder) // window_shift
    remainder = waveform[:, num_frames * window_shift:]
    strides = waveform.stride(0), window_shift * waveform.stride(1), waveform.stride(1)
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides), remainder


BLACKMAN = 'blackman'


HAMMING = 'hamming'


HANNING = 'hanning'


POVEY = 'povey'


RECTANGULAR = 'rectangular'


def create_frame_window(window_size, window_type: str='povey', blackman_coeff=0.42):
    """Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return blackman_coeff - 0.5 * torch.cos(a * window_function) + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
    else:
        raise Exception(f'Invalid window type: {window_type}')


class Wav2Win(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and partition them into overlapping frames (of audio samples).
    Note: no feature extraction happens in here, the output is still a time-domain signal.

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Win()
        >>> t(x).shape
        torch.Size([1, 100, 400])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, window_length)``.
    When ``return_log_energy==True``, returns a tuple where the second element
    is a log-energy tensor of shape ``(batch_size, num_frames)``.
    """

    def __init__(self, sampling_rate: int=16000, frame_length: Seconds=0.025, frame_shift: Seconds=0.01, pad_length: Optional[int]=None, remove_dc_offset: bool=True, preemph_coeff: float=0.97, window_type: str='povey', dither: float=0.0, snip_edges: bool=False, energy_floor: float=EPSILON, raw_energy: bool=True, return_log_energy: bool=False) ->None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.remove_dc_offset = remove_dc_offset
        self.preemph_coeff = preemph_coeff
        self.window_type = window_type
        self.dither = dither
        self.snip_edges = snip_edges
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.return_log_energy = return_log_energy
        if snip_edges:
            warnings.warn('Setting snip_edges=True is generally incompatible with Lhotse -- you might experience mismatched duration/num_frames errors.')
        N = int(math.floor(frame_length * sampling_rate))
        self._length = N
        self._shift = int(math.floor(frame_shift * sampling_rate))
        self._window = nn.Parameter(create_frame_window(N, window_type=window_type), requires_grad=False)
        self.pad_length = N if pad_length is None else pad_length
        assert self.pad_length >= N, f'pad_length (or fft_length) = {pad_length} cannot be smaller than N = {N}'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '{}(sampling_rate={}, frame_length={}, frame_shift={}, pad_length={}, remove_dc_offset={}, preemph_coeff={}, window_type={} dither={}, snip_edges={}, energy_floor={}, raw_energy={}, return_log_energy={})'.format(self.__class__.__name__, self.sampling_rate, self.frame_length, self.frame_shift, self.pad_length, self.remove_dc_offset, self.preemph_coeff, self.window_type, self.dither, self.snip_edges, self.energy_floor, self.raw_energy, self.return_log_energy)
        return s

    def _forward_strided(self, x_strided: torch.Tensor) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.remove_dc_offset:
            mu = torch.mean(x_strided, dim=2, keepdim=True)
            x_strided = x_strided - mu
        log_energy: Optional[torch.Tensor] = None
        if self.return_log_energy and self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)
        if self.preemph_coeff != 0.0:
            x_offset = torch.nn.functional.pad(x_strided, (1, 0), mode='replicate')
            x_strided = x_strided - self.preemph_coeff * x_offset[:, :, :-1]
        x_strided = x_strided * self._window
        if self.pad_length != self._length:
            pad = self.pad_length - self._length
            x_strided = torch.nn.functional.pad(x_strided.unsqueeze(1), [0, pad], mode='constant', value=0.0).squeeze(1)
        if self.return_log_energy and not self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)
        return x_strided, log_energy

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n
        x_strided = _get_strided_batch(x, self._length, self._shift, self.snip_edges)
        return self._forward_strided(x_strided)

    @torch.jit.export
    def online_inference(self, x: torch.Tensor, context: Optional[torch.Tensor]=None) ->Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        The same as the ``forward()`` method, except it accepts an extra argument with the
        remainder waveform from the previous call of ``online_inference()``, and returns
        a tuple of ``((frames, log_energy), remainder)``.
        """
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n
        x_strided, remainder = _get_strided_batch_streaming(x, window_length=self._length, window_shift=self._shift, prev_remainder=context, snip_edges=self.snip_edges)
        x_strided, log_energy = self._forward_strided(x_strided)
        return (x_strided, log_energy), remainder


def next_power_of_2(x: int) ->int:
    """
    Returns the smallest power of 2 that is greater than x.

    Original source: TorchAudio (torchaudio/compliance/kaldi.py)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class Wav2FFT(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The output is a complex-valued tensor.

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2FFT()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``
    with dtype ``torch.complex64``.
    """

    def __init__(self, sampling_rate: int=16000, frame_length: Seconds=0.025, frame_shift: Seconds=0.01, round_to_power_of_two: bool=True, remove_dc_offset: bool=True, preemph_coeff: float=0.97, window_type: str='povey', dither: float=0.0, snip_edges: bool=False, energy_floor: float=EPSILON, raw_energy: bool=True, use_energy: bool=True) ->None:
        super().__init__()
        self.use_energy = use_energy
        N = int(math.floor(frame_length * sampling_rate))
        self.fft_length = next_power_of_2(N) if round_to_power_of_two else N
        self.wav2win = Wav2Win(sampling_rate, frame_length, frame_shift, pad_length=self.fft_length, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, return_log_energy=use_energy)

    @property
    def sampling_rate(self) ->int:
        return self.wav2win.sampling_rate

    @property
    def frame_length(self) ->Seconds:
        return self.wav2win.frame_length

    @property
    def frame_shift(self) ->Seconds:
        return self.wav2win.frame_shift

    @property
    def remove_dc_offset(self) ->bool:
        return self.wav2win.remove_dc_offset

    @property
    def preemph_coeff(self) ->float:
        return self.wav2win.preemph_coeff

    @property
    def window_type(self) ->str:
        return self.wav2win.window_type

    @property
    def dither(self) ->float:
        return self.wav2win.dither

    def _forward_strided(self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]) ->torch.Tensor:
        X = _rfft(x_strided)
        if self.use_energy and log_e is not None:
            X[:, :, 0] = log_e
        return X

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x_strided, log_e = self.wav2win(x)
        return self._forward_strided(x_strided=x_strided, log_e=log_e)

    @torch.jit.export
    def online_inference(self, x: torch.Tensor, context: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        (x_strided, log_e), remainder = self.wav2win.online_inference(x, context=context)
        return self._forward_strided(x_strided=x_strided, log_e=log_e), remainder


class Wav2Spec(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The STFT is transformed either to a magnitude spectrum (``use_fft_mag=True``)
    or a power spectrum (``use_fft_mag=False``).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Spec()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``.
    """

    def __init__(self, sampling_rate: int=16000, frame_length: Seconds=0.025, frame_shift: Seconds=0.01, round_to_power_of_two: bool=True, remove_dc_offset: bool=True, preemph_coeff: float=0.97, window_type: str='povey', dither: float=0.0, snip_edges: bool=False, energy_floor: float=EPSILON, raw_energy: bool=True, use_energy: bool=True, use_fft_mag: bool=False) ->None:
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def _forward_strided(self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]) ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        if self.use_energy and log_e is not None:
            pow_spec[:, :, 0] = log_e
        return pow_spec


class Wav2LogSpec(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The STFT is transformed either to a log-magnitude spectrum (``use_fft_mag=True``)
    or a log-power spectrum (``use_fft_mag=False``).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogSpec()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``.
    """

    def __init__(self, sampling_rate: int=16000, frame_length: Seconds=0.025, frame_shift: Seconds=0.01, round_to_power_of_two: bool=True, remove_dc_offset: bool=True, preemph_coeff: float=0.97, window_type: str='povey', dither: float=0.0, snip_edges: bool=False, energy_floor: float=EPSILON, raw_energy: bool=True, use_energy: bool=True, use_fft_mag: bool=False) ->None:
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def _forward_strided(self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]) ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = (pow_spec + 1e-15).log()
        if self.use_energy and log_e is not None:
            pow_spec[:, :, 0] = log_e
        return pow_spec


def lin2mel(x):
    return 1127.0 * np.log(1 + x / 700)


def create_mel_scale(num_filters: int, fft_length: int, sampling_rate: int, low_freq: float=0, high_freq: Optional[float]=None, norm_filters: bool=True) ->torch.Tensor:
    if high_freq is None or high_freq == 0:
        high_freq = sampling_rate / 2
    if high_freq < 0:
        high_freq = sampling_rate / 2 + high_freq
    mel_low_freq = lin2mel(low_freq)
    mel_high_freq = lin2mel(high_freq)
    melfc = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
    mels = lin2mel(np.linspace(0, sampling_rate, fft_length))
    B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=np.float32)
    for k in range(num_filters):
        left_mel = melfc[k]
        center_mel = melfc[k + 1]
        right_mel = melfc[k + 2]
        for j in range(int(fft_length / 2)):
            mel_j = mels[j]
            if left_mel < mel_j < right_mel:
                if mel_j <= center_mel:
                    B[j, k] = (mel_j - left_mel) / (center_mel - left_mel)
                else:
                    B[j, k] = (right_mel - mel_j) / (right_mel - center_mel)
    if norm_filters:
        B = B / np.sum(B, axis=0, keepdims=True)
    return torch.from_numpy(B)


class Wav2LogFilterBank(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their log-Mel filter bank energies (also known as "fbank").

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogFilterBank()
        >>> t(x).shape
        torch.Size([1, 100, 80])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_filters)``.
    """

    def __init__(self, sampling_rate: int=16000, frame_length: Seconds=0.025, frame_shift: Seconds=0.01, round_to_power_of_two: bool=True, remove_dc_offset: bool=True, preemph_coeff: float=0.97, window_type: str='povey', dither: float=0.0, snip_edges: bool=False, energy_floor: float=EPSILON, raw_energy: bool=True, use_energy: bool=False, use_fft_mag: bool=False, low_freq: float=20.0, high_freq: float=-400.0, num_filters: int=80, norm_filters: bool=False, torchaudio_compatible_mel_scale: bool=True):
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self._eps = nn.Parameter(torch.tensor(torch.finfo(torch.float).eps), requires_grad=False)
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram
        if torchaudio_compatible_mel_scale:
            from torchaudio.compliance.kaldi import get_mel_banks
            fb, _ = get_mel_banks(num_bins=num_filters, window_length_padded=self.fft_length, sample_freq=sampling_rate, low_freq=low_freq, high_freq=high_freq, vtln_warp_factor=1.0, vtln_low=100.0, vtln_high=-500.0)
            fb = torch.nn.functional.pad(fb, (0, 1), mode='constant', value=0).T
        else:
            fb = create_mel_scale(num_filters=num_filters, fft_length=self.fft_length, sampling_rate=sampling_rate, low_freq=low_freq, high_freq=high_freq, norm_filters=norm_filters)
        self._fb = nn.Parameter(fb, requires_grad=False)

    def _forward_strided(self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]) ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()
        if self.use_energy and log_e is not None:
            pow_spec = torch.cat((log_e.unsqueeze(-1), pow_spec), dim=-1)
        return pow_spec


class Wav2MFCC(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Mel-Frequency Cepstral Coefficients (MFCC).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2MFCC()
        >>> t(x).shape
        torch.Size([1, 100, 13])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_ceps)``.
    """

    def __init__(self, sampling_rate: int=16000, frame_length: Seconds=0.025, frame_shift: Seconds=0.01, round_to_power_of_two: bool=True, remove_dc_offset: bool=True, preemph_coeff: float=0.97, window_type: str='povey', dither: float=0.0, snip_edges: bool=False, energy_floor: float=EPSILON, raw_energy: bool=True, use_energy: bool=False, use_fft_mag: bool=False, low_freq: float=20.0, high_freq: float=-400.0, num_filters: int=23, norm_filters: bool=False, num_ceps: int=13, cepstral_lifter: int=22, torchaudio_compatible_mel_scale: bool=True):
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self.num_ceps = num_ceps
        self.cepstral_lifter = cepstral_lifter
        self._eps = nn.Parameter(torch.tensor(torch.finfo(torch.float).eps), requires_grad=False)
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram
        if torchaudio_compatible_mel_scale:
            from torchaudio.compliance.kaldi import get_mel_banks
            fb, _ = get_mel_banks(num_bins=num_filters, window_length_padded=self.fft_length, sample_freq=sampling_rate, low_freq=low_freq, high_freq=high_freq, vtln_warp_factor=1.0, vtln_low=100.0, vtln_high=-500.0)
            fb = torch.nn.functional.pad(fb, (0, 1), mode='constant', value=0).T
        else:
            fb = create_mel_scale(num_filters=num_filters, fft_length=self.fft_length, sampling_rate=sampling_rate, low_freq=low_freq, high_freq=high_freq, norm_filters=norm_filters)
        self._fb = nn.Parameter(fb, requires_grad=False)
        self._dct = nn.Parameter(self.make_dct_matrix(self.num_ceps, self.num_filters), requires_grad=False)
        self._lifter = nn.Parameter(self.make_lifter(self.num_ceps, self.cepstral_lifter), requires_grad=False)

    @staticmethod
    def make_lifter(N, Q):
        """Makes the liftering function

        Args:
          N: Number of cepstral coefficients.
          Q: Liftering parameter
        Returns:
          Liftering vector.
        """
        if Q == 0:
            return 1
        return 1 + 0.5 * Q * torch.sin(math.pi * torch.arange(N, dtype=torch.get_default_dtype()) / Q)

    @staticmethod
    def make_dct_matrix(num_ceps, num_filters):
        n = torch.arange(float(num_filters)).unsqueeze(1)
        k = torch.arange(float(num_ceps))
        dct = torch.cos(math.pi / float(num_filters) * (n + 0.5) * k)
        dct[:, 0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(num_filters))
        return dct

    def _forward_strided(self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]) ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()
        mfcc = torch.matmul(pow_spec, self._dct)
        if self.cepstral_lifter > 0:
            mfcc *= self._lifter
        if self.use_energy and log_e is not None:
            mfcc[:, 0] = log_e
        return mfcc


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (RandomizedSmoothing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lhotse_speech_lhotse(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

