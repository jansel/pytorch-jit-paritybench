import sys
_module = sys.modules[__name__]
del sys
style = _module
conf = _module
nerfstudio = _module
cameras = _module
camera_optimizers = _module
camera_paths = _module
camera_utils = _module
cameras = _module
lie_groups = _module
rays = _module
configs = _module
base_config = _module
config_utils = _module
experiment_config = _module
method_configs = _module
data = _module
datamanagers = _module
base_datamanager = _module
semantic_datamanager = _module
variable_res_datamanager = _module
dataparsers = _module
base_dataparser = _module
blender_dataparser = _module
dnerf_dataparser = _module
friends_dataparser = _module
instant_ngp_dataparser = _module
minimal_dataparser = _module
nerfstudio_dataparser = _module
nuscenes_dataparser = _module
phototourism_dataparser = _module
datasets = _module
base_dataset = _module
semantic_dataset = _module
pixel_samplers = _module
scene_box = _module
utils = _module
colmap_utils = _module
data_utils = _module
dataloaders = _module
nerfstudio_collate = _module
engine = _module
callbacks = _module
optimizers = _module
schedulers = _module
trainer = _module
exporter = _module
exporter_utils = _module
texture_utils = _module
tsdf_utils = _module
field_components = _module
activations = _module
base_field_component = _module
embedding = _module
encodings = _module
field_heads = _module
mlp = _module
spatial_distortions = _module
temporal_distortions = _module
fields = _module
base_field = _module
density_fields = _module
instant_ngp_field = _module
nerfacto_field = _module
nerfw_field = _module
semantic_nerf_field = _module
tensorf_field = _module
vanilla_nerf_field = _module
model_components = _module
losses = _module
ray_generators = _module
ray_samplers = _module
renderers = _module
scene_colliders = _module
models = _module
base_model = _module
instant_ngp = _module
mipnerf = _module
nerfacto = _module
semantic_nerfw = _module
tensorf = _module
vanilla_nerf = _module
pipelines = _module
base_pipeline = _module
dynamic_batch = _module
process_data = _module
hloc_utils = _module
insta360_utils = _module
metashape_utils = _module
polycam_utils = _module
process_data_utils = _module
record3d_utils = _module
colormaps = _module
colors = _module
comms = _module
decorators = _module
eval_utils = _module
install_checks = _module
io = _module
math = _module
misc = _module
plotly_utils = _module
poses = _module
printing = _module
profiler = _module
rich_utils = _module
scripts = _module
tensor_dataclass = _module
writer = _module
viewer = _module
run_deploy = _module
server = _module
path = _module
node = _module
state_node = _module
subprocess = _module
utils = _module
video_stream = _module
viewer_utils = _module
visualizer = _module
completions = _module
install = _module
process_nuscenes_masks = _module
docs = _module
add_nb_tags = _module
build_docs = _module
downloads = _module
download_data = _module
eval = _module
exporter = _module
github = _module
run_actions = _module
render = _module
texture = _module
train = _module
view_dataset = _module
test_cameras = _module
test_rays = _module
test_embedding = _module
test_encodings = _module
test_field_outputs = _module
test_fields = _module
test_mlp = _module
test_temporal_distortions = _module
test_ray_sampler = _module
test_renderers = _module
test_train = _module
test_poses = _module
test_tensor_dataclass = _module
test_visualization = _module

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


from typing import Type


from typing import Union


import torch


from torch import nn


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


import math


from typing import List


import numpy as np


from enum import Enum


from enum import auto


import torchvision


from torch.nn.functional import normalize


import random


from typing import Callable


from abc import abstractmethod


from torch.nn import Parameter


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


from copy import deepcopy


import numpy.typing as npt


from torch.utils.data.dataloader import DataLoader


import collections


import collections.abc


import re


import torch.utils.data


from torch._six import string_classes


from torch.cuda.amp.grad_scaler import GradScaler


from torch.nn.parameter import Parameter


from torch.optim import Optimizer


from torch.optim import lr_scheduler


import time


import torch.nn.functional as F


from torch.autograd import Function


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from typing import Set


from collections import defaultdict


import typing


from time import time


from typing import cast


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from matplotlib import cm


from math import floor


from math import log


from typing import NoReturn


from typing import TypeVar


import enum


from torch.utils.tensorboard import SummaryWriter


import warnings


import torch.multiprocessing as mp


from itertools import product


class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = '['
                for item in val:
                    flattened_val += str(item) + '\n'
                flattened_val = flattened_val.rstrip('\n')
                val = flattened_val + ']'
            lines += f'{key}: {str(val)}'.split('\n')
        return '\n    '.join(lines)


class IterableWrapper:
    """A helper that will allow an instance of a class to return multiple kinds of iterables bound
    to different functions of that class.

    To use this, take an instance of a class. From that class, pass in the <instance>.<new_iter_function>
    and <instance>.<new_next_function> to the IterableWrapper constructor. By passing in the instance's
    functions instead of just the class's functions, the self argument should automatically be accounted
    for.

    Args:
        new_iter: function that will be called instead as the __iter__() function
        new_next: function that will be called instead as the __next__() function
        length: length of the iterable. If -1, the iterable will be infinite.


    Attributes:
        new_iter: object's pointer to the function we are calling for __iter__()
        new_next: object's pointer to the function we are calling for __next__()
        length: length of the iterable. If -1, the iterable will be infinite.
        i: current index of the iterable.

    """
    i: int

    def __init__(self, new_iter: Callable, new_next: Callable, length: int=-1):
        self.new_iter = new_iter
        self.new_next = new_next
        self.length = length

    def __next__(self):
        if self.length != -1 and self.i >= self.length:
            raise StopIteration
        self.i += 1
        return self.new_next()

    def __iter__(self):
        self.new_iter()
        self.i = 0
        return self


class TrainingCallbackLocation(Enum):
    """Enum for specifying where the training callback should be run."""
    BEFORE_TRAIN_ITERATION = auto()
    AFTER_TRAIN_ITERATION = auto()


class TrainingCallback:
    """Callback class used during training.
    The function 'func' with 'args' and 'kwargs' will be called every 'update_every_num_iters' training iterations,
    including at iteration 0. The function is called after the training iteration.

    Args:
        where_to_run: List of locations for when to run callbak (before/after iteration)
        func: The function that will be called.
        update_every_num_iters: How often to call the function `func`.
        iters: Tuple of iteration steps to perform callback
        args: args for the function 'func'.
        kwargs: kwargs for the function 'func'.
    """

    def __init__(self, where_to_run: List[TrainingCallbackLocation], func: Callable, update_every_num_iters: Optional[int]=None, iters: Optional[Tuple[int, ...]]=None, args: Optional[List]=None, kwargs: Optional[Dict]=None):
        assert 'step' in signature(func).parameters.keys(), f"'step: int' must be an argument in the callback function 'func': {func.__name__}"
        self.where_to_run = where_to_run
        self.update_every_num_iters = update_every_num_iters
        self.iters = iters
        self.func = func
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def run_callback(self, step: int):
        """Callback to run after training step

        Args:
            step: current iteration step
        """
        if self.update_every_num_iters is not None:
            if step % self.update_every_num_iters == 0:
                self.func(*self.args, **self.kwargs, step=step)
        elif self.iters is not None:
            if step in self.iters:
                self.func(*self.args, **self.kwargs, step=step)

    def run_callback_at_location(self, step: int, location: TrainingCallbackLocation):
        """Runs the callback if it's supposed to be run at the given location.

        Args:
            step: current iteration step
            location: when to run callback (before/after iteration)
        """
        if location in self.where_to_run:
            self.run_callback(step=step)


UNICODE = str


class Path:
    """Path class

    Args:
        entries: component parts of the path
    """
    __slots__ = ['entries']

    def __init__(self, entries: Tuple=tuple()):
        self.entries = entries

    def append(self, other: str) ->'Path':
        """Methodthat appends a new component and returns new Path

        Args:
            other: _description_
        """
        new_path = self.entries
        for element in other.split('/'):
            if len(element) == 0:
                new_path = tuple()
            else:
                new_path = new_path + (element,)
        return Path(new_path)

    def lower(self):
        """Convert path object to serializable format"""
        return UNICODE('/' + '/'.join(self.entries))

    def __hash__(self):
        return hash(self.entries)

    def __eq__(self, other):
        return self.entries == other.entries


def get_dict_to_torch(stuff: Any, device: Union[torch.device, str]='cpu', exclude: Optional[List[str]]=None):
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff
    return stuff


class CameraType(Enum):
    """Supported camera types."""
    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()


TensorDataclassT = TypeVar('TensorDataclassT', bound='TensorDataclass')


class TensorDataclass:
    """@dataclass of tensors with the same size batch. Allows indexing and standard tensor ops.
    Fields that are not Tensors will not be batched unless they are also a TensorDataclass.
    Any fields that are dictionaries will have their Tensors or TensorDataclasses batched, and
    dictionaries will have their tensors or TensorDataclasses considered in the initial broadcast.
    Tensor fields must have at least 1 dimension, meaning that you must convert a field like torch.Tensor(1)
    to torch.Tensor([1])

    Example:

    .. code-block:: python

        @dataclass
        class TestTensorDataclass(TensorDataclass):
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor = None

        # Create a new tensor dataclass with batch size of [2,3,4]
        test = TestTensorDataclass(a=torch.ones((2, 3, 4, 2)), b=torch.ones((4, 3)))

        test.shape  # [2, 3, 4]
        test.a.shape  # [2, 3, 4, 2]
        test.b.shape  # [2, 3, 4, 3]

        test.reshape((6,4)).shape  # [6, 4]
        test.flatten().shape  # [24,]

        test[..., 0].shape  # [2, 3]
        test[:, 0, :].shape  # [2, 4]
    """
    _shape: tuple
    _field_custom_dimensions: Dict[str, int] = {}

    def __post_init__(self) ->None:
        """Finishes setting up the TensorDataclass

        This will 1) find the broadcasted shape and 2) broadcast all fields to this shape 3)
        set _shape to be the broadcasted shape.
        """
        if self._field_custom_dimensions is not None:
            for k, v in self._field_custom_dimensions.items():
                assert isinstance(v, int) and v > 1, f'Custom dimensions must be an integer greater than 1, since 1 is the default, received {k}: {v}'
        if not dataclasses.is_dataclass(self):
            raise TypeError('TensorDataclass must be a dataclass')
        batch_shapes = self._get_dict_batch_shapes({f.name: self.__getattribute__(f.name) for f in dataclasses.fields(self)})
        if len(batch_shapes) == 0:
            raise ValueError('TensorDataclass must have at least one tensor')
        batch_shape = torch.broadcast_shapes(*batch_shapes)
        broadcasted_fields = self._broadcast_dict_fields({f.name: self.__getattribute__(f.name) for f in dataclasses.fields(self)}, batch_shape)
        for f, v in broadcasted_fields.items():
            self.__setattr__(f, v)
        self.__setattr__('_shape', batch_shape)

    def _get_dict_batch_shapes(self, dict_: Dict) ->List:
        """Returns batch shapes of all tensors in a dictionary

        Args:
            dict_: The dictionary to get the batch shapes of.

        Returns:
            The batch shapes of all tensors in the dictionary.
        """
        batch_shapes = []
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                if isinstance(self._field_custom_dimensions, dict) and k in self._field_custom_dimensions:
                    batch_shapes.append(v.shape[:-self._field_custom_dimensions[k]])
                else:
                    batch_shapes.append(v.shape[:-1])
            elif isinstance(v, TensorDataclass):
                batch_shapes.append(v.shape)
            elif isinstance(v, Dict):
                batch_shapes.extend(self._get_dict_batch_shapes(v))
        return batch_shapes

    def _broadcast_dict_fields(self, dict_: Dict, batch_shape) ->Dict:
        """Broadcasts all tensors in a dictionary according to batch_shape

        Args:
            dict_: The dictionary to broadcast.

        Returns:
            The broadcasted dictionary.
        """
        new_dict = {}
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                if isinstance(self._field_custom_dimensions, dict) and k in self._field_custom_dimensions:
                    new_dict[k] = v.broadcast_to((*batch_shape, *v.shape[-self._field_custom_dimensions[k]:]))
                else:
                    new_dict[k] = v.broadcast_to((*batch_shape, v.shape[-1]))
            elif isinstance(v, TensorDataclass):
                new_dict[k] = v.broadcast_to(batch_shape)
            elif isinstance(v, Dict):
                new_dict[k] = self._broadcast_dict_fields(v, batch_shape)
        return new_dict

    def __getitem__(self: TensorDataclassT, indices) ->TensorDataclassT:
        if isinstance(indices, torch.Tensor):
            return self._apply_fn_to_fields(lambda x: x[indices])
        if isinstance(indices, (int, slice, type(Ellipsis))):
            indices = indices,
        assert isinstance(indices, tuple)
        tensor_fn = lambda x: x[indices + (slice(None),)]
        dataclass_fn = lambda x: x[indices]

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v[indices + (slice(None),) * custom_dims]
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn, custom_tensor_dims_fn=custom_tensor_dims_fn)

    def __setitem__(self, indices, value) ->NoReturn:
        raise RuntimeError('Index assignment is not supported for TensorDataclass')

    def __len__(self) ->int:
        if len(self._shape) == 0:
            raise TypeError('len() of a 0-d tensor')
        return self.shape[0]

    def __bool__(self) ->bool:
        if len(self) == 0:
            raise ValueError(f'The truth value of {self.__class__.__name__} when `len(x) == 0` is ambiguous. Use `len(x)` or `x is not None`.')
        return True

    @property
    def shape(self) ->Tuple[int, ...]:
        """Returns the batch shape of the tensor dataclass."""
        return self._shape

    @property
    def size(self) ->int:
        """Returns the number of elements in the tensor dataclass batch dimension."""
        if len(self._shape) == 0:
            return 1
        return int(np.prod(self._shape))

    @property
    def ndim(self) ->int:
        """Returns the number of dimensions of the tensor dataclass."""
        return len(self._shape)

    def reshape(self: TensorDataclassT, shape: Tuple[int, ...]) ->TensorDataclassT:
        """Returns a new TensorDataclass with the same data but with a new shape.

        This should deepcopy as well.

        Args:
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        if isinstance(shape, int):
            shape = shape,
        tensor_fn = lambda x: x.reshape((*shape, x.shape[-1]))
        dataclass_fn = lambda x: x.reshape(shape)

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v.reshape((*shape, *v.shape[-custom_dims:]))
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn, custom_tensor_dims_fn=custom_tensor_dims_fn)

    def flatten(self: TensorDataclassT) ->TensorDataclassT:
        """Returns a new TensorDataclass with flattened batch dimensions

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self.reshape((-1,))

    def broadcast_to(self: TensorDataclassT, shape: Union[torch.Size, Tuple[int, ...]]) ->TensorDataclassT:
        """Returns a new TensorDataclass broadcast to new shape.

        Changes to the original tensor dataclass should effect the returned tensor dataclass,
        meaning it is NOT a deepcopy, and they are still linked.

        Args:
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v.broadcast_to((*shape, *v.shape[-custom_dims:]))
        return self._apply_fn_to_fields(lambda x: x.broadcast_to((*shape, x.shape[-1])), custom_tensor_dims_fn=custom_tensor_dims_fn)

    def to(self: TensorDataclassT, device) ->TensorDataclassT:
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but on the specified device.
        """
        return self._apply_fn_to_fields(lambda x: x)

    def _apply_fn_to_fields(self: TensorDataclassT, fn: Callable, dataclass_fn: Optional[Callable]=None, custom_tensor_dims_fn: Optional[Callable]=None) ->TensorDataclassT:
        """Applies a function to all fields of the tensor dataclass.

        TODO: Someone needs to make a high level design choice for whether not not we want this
        to apply the function to any fields in arbitray superclasses. This is an edge case until we
        upgrade to python 3.10 and dataclasses can actually be subclassed with vanilla python and no
        janking, but if people try to jank some subclasses that are grandchildren of TensorDataclass
        (imagine if someone tries to subclass the RayBundle) this will matter even before upgrading
        to 3.10 . Currently we aren't going to be able to work properly for grandchildren, but you
        want to use self.__dict__ if you want to apply this to grandchildren instead of our dictionary
        from dataclasses.fields(self) as we do below and in other places.

        Args:
            fn: The function to apply to tensor fields.
            dataclass_fn: The function to apply to TensorDataclass fields.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        new_fields = self._apply_fn_to_dict({f.name: self.__getattribute__(f.name) for f in dataclasses.fields(self)}, fn, dataclass_fn, custom_tensor_dims_fn)
        return dataclasses.replace(self, **new_fields)

    def _apply_fn_to_dict(self, dict_: Dict, fn: Callable, dataclass_fn: Optional[Callable]=None, custom_tensor_dims_fn: Optional[Callable]=None) ->Dict:
        """A helper function for _apply_fn_to_fields, applying a function to all fields of dict_

        Args:
            dict_: The dictionary to apply the function to.
            fn: The function to apply to tensor fields.
            dataclass_fn: The function to apply to TensorDataclass fields.

        Returns:
            A new dictionary with the same data but with a new shape. Will deep copy"""
        field_names = dict_.keys()
        new_dict = {}
        for f in field_names:
            v = dict_[f]
            if v is not None:
                if isinstance(v, TensorDataclass) and dataclass_fn is not None:
                    new_dict[f] = dataclass_fn(v)
                elif isinstance(v, torch.Tensor) and isinstance(self._field_custom_dimensions, dict) and f in self._field_custom_dimensions and custom_tensor_dims_fn is not None:
                    new_dict[f] = custom_tensor_dims_fn(f, v)
                elif isinstance(v, (torch.Tensor, TensorDataclass)):
                    new_dict[f] = fn(v)
                elif isinstance(v, Dict):
                    new_dict[f] = self._apply_fn_to_dict(v, fn, dataclass_fn)
                else:
                    new_dict[f] = deepcopy(v)
        return new_dict


NERFSTUDIO_COLLATE_ERR_MSG_FORMAT = 'default_collate: batch must contain tensors, numpy arrays, numbers, dicts, lists or anything in {}; found {}'


np_str_obj_array_pattern = re.compile('[SaUO]')


def nerfstudio_collate(batch, extra_mappings: Union[Dict[type, Callable], None]=None):
    """
    This is the default pytorch collate function, but with support for nerfstudio types. All documentation
    below is copied straight over from pytorch's default_collate function, python version 3.8.13,
    pytorch version '1.12.1+cu113'. Custom nerfstudio types are accounted for at the end, and extra
    mappings can be passed in to handle custom types. These mappings are from types: callable (types
    being like int or float or the return value of type(3.), etc). The only code before we parse for custom types that
    was changed from default pytorch was the addition of the extra_mappings argument, a find and replace operation
    from default_collate to nerfstudio_collate, and the addition of the nerfstudio_collate_err_msg_format variable.


    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, nerfstudio_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # Example with a batch of `int`s:
        >>> nerfstudio_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> nerfstudio_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> nerfstudio_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> nerfstudio_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> nerfstudio_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> nerfstudio_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
    """
    if extra_mappings is None:
        extra_mappings = {}
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem.dtype))
            return nerfstudio_collate([torch.as_tensor(b) for b in batch], extra_mappings=extra_mappings)
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem})
        except TypeError:
            return {key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))
        if isinstance(elem, tuple):
            return [nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed]
        else:
            try:
                return elem_type([nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed])
            except TypeError:
                return [nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed]
    elif isinstance(elem, Cameras):
        assert all(isinstance(cam, Cameras) for cam in batch)
        assert all(cam.distortion_params is None for cam in batch) or all(cam.distortion_params is not None for cam in batch), 'All cameras must have distortion parameters or none of them should have distortion parameters.            Generalized batching will be supported in the future.'
        if elem.shape == ():
            op = torch.stack
        else:
            op = torch.cat
        return Cameras(op([cameras.camera_to_worlds for cameras in batch], dim=0), op([cameras.fx for cameras in batch], dim=0), op([cameras.fy for cameras in batch], dim=0), op([cameras.cx for cameras in batch], dim=0), op([cameras.cy for cameras in batch], dim=0), height=op([cameras.height for cameras in batch], dim=0), width=op([cameras.width for cameras in batch], dim=0), distortion_params=op([(cameras.distortion_params if cameras.distortion_params is not None else torch.zeros_like(cameras.distortion_params)) for cameras in batch], dim=0), camera_type=op([cameras.camera_type for cameras in batch], dim=0), times=torch.stack([(cameras.times if cameras.times is not None else -torch.ones_like(cameras.times)) for cameras in batch], dim=0))
    for type_key in extra_mappings:
        if isinstance(elem, type_key):
            return extra_mappings[type_key](batch)
    raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem_type))


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(self, dataset: Dataset, num_images_to_sample_from: int=-1, num_times_to_repeat_images: int=-1, device: Union[torch.device, str]='cpu', collate_fn=nerfstudio_collate, **kwargs):
        self.dataset = dataset
        super().__init__(dataset=dataset, **kwargs)
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = num_images_to_sample_from == -1 or num_images_to_sample_from >= len(self.dataset)
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_repeated = self.num_times_to_repeat_images
        self.first_time = True
        self.cached_collated_batch = None
        if self.cache_all_images:
            CONSOLE.print(f'Caching all {len(self.dataset)} images.')
            if len(self.dataset) > 500:
                CONSOLE.print('[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from.')
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(f'Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling.')
        else:
            CONSOLE.print(f'Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, resampling every {self.num_times_to_repeat_images} iters.')

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
        batch_list = []
        results = []
        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description='Loading data batch', transient=True):
                batch_list.append(res.result())
        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(collated_batch, device=self.device, exclude=['image'])
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images:
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


def to_immutable_dict(d: Dict[str, Any]):
    """Method to convert mutable dict to default factory dict

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda : dict(d))


BaseImage = collections.namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])


class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float=1.0) ->torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = int(width * scale_factor), int(height * scale_factor)
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    return mask_tensor


BLACK = torch.tensor([0.0, 0.0, 0.0])


BLUE = torch.tensor([0.0, 0.0, 1.0])


GREEN = torch.tensor([0.0, 1.0, 0.0])


RED = torch.tensor([1.0, 0.0, 0.0])


WHITE = torch.tensor([1.0, 1.0, 1.0])


COLORS_DICT = {'white': WHITE, 'black': BLACK, 'red': RED, 'green': GREEN, 'blue': BLUE}


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == '.json'
    with open(filename, encoding='UTF-8') as file:
        return json.load(file)


CAMERA_MODEL_TO_TYPE = {'SIMPLE_PINHOLE': CameraType.PERSPECTIVE, 'PINHOLE': CameraType.PERSPECTIVE, 'SIMPLE_RADIAL': CameraType.PERSPECTIVE, 'RADIAL': CameraType.PERSPECTIVE, 'OPENCV': CameraType.PERSPECTIVE, 'OPENCV_FISHEYE': CameraType.FISHEYE, 'EQUIRECTANGULAR': CameraType.EQUIRECTANGULAR}


MAX_AUTO_RESOLUTION = 1600


def qvec2rotmat(qvec):
    return np.array([[1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]], [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]], [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotation_translation_to_pose(r_vec, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""
    pose = np.eye(4)
    pose[:3, :3] = qvec2rotmat(r_vec)
    pose[:3, 3] = t_vec
    return pose


CameraModel = collections.namedtuple('CameraModel', ['model_id', 'model_name', 'num_params'])


CAMERA_MODELS = {CameraModel(model_id=0, model_name='SIMPLE_PINHOLE', num_params=3), CameraModel(model_id=1, model_name='PINHOLE', num_params=4), CameraModel(model_id=2, model_name='SIMPLE_RADIAL', num_params=4), CameraModel(model_id=3, model_name='RADIAL', num_params=5), CameraModel(model_id=4, model_name='OPENCV', num_params=8), CameraModel(model_id=5, model_name='OPENCV_FISHEYE', num_params=8), CameraModel(model_id=6, model_name='FULL_OPENCV', num_params=12), CameraModel(model_id=7, model_name='FOV', num_params=5), CameraModel(model_id=8, model_name='SIMPLE_RADIAL_FISHEYE', num_params=4), CameraModel(model_id=9, model_name='RADIAL_FISHEYE', num_params=5), CameraModel(model_id=10, model_name='THIN_PRISM_FISHEYE', num_params=12)}


CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


Camera = collections.namedtuple('Camera', ['id', 'model', 'width', 'height', 'params'])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character='<'):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, 'rb') as fid:
        num_cameras = read_next_bytes(fid, 8, 'Q')[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence='d' * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, 'Q')[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ''
            current_char = read_next_bytes(fid, 1, 'c')[0]
            while current_char != b'\x00':
                image_name += current_char.decode('utf-8')
                current_char = read_next_bytes(fid, 1, 'c')[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence='ddq' * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) ->torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


def print_tcnn_speed_warning(method_name: str):
    """Prints a warning about the speed of the TCNN."""
    CONSOLE.line()
    CONSOLE.print(f'[bold yellow]WARNING: Using a slow implementation of {method_name}. ')
    CONSOLE.print('[bold yellow]:person_running: :person_running: ' + 'Install tcnn for speedups :person_running: :person_running:')
    CONSOLE.print('[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch')
    CONSOLE.line()


class FieldHeadNames(Enum):
    """Possible field outputs"""
    RGB = 'rgb'
    SH = 'sh'
    DENSITY = 'density'
    NORMALS = 'normals'
    PRED_NORMALS = 'pred_normals'
    UNCERTAINTY = 'uncertainty'
    TRANSIENT_RGB = 'transient_rgb'
    TRANSIENT_DENSITY = 'transient_density'
    SEMANTICS = 'semantics'


class _TruncExp(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _TruncExp.apply


MSELoss = nn.MSELoss


def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    loss_intra = torch.sum(w ** 2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3
    return loss_inter + loss_intra


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)
    return sdist


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


EPS = 1e-07


def interlevel_loss(weights_list, ray_samples_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist
        wp = weights[..., 0]
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel

