import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
server = _module
tutorials = _module
advanced = _module
federated_sms_spam_prediction = _module
handcrafted_GRU = _module
preprocess = _module
monitor_network_traffic = _module
split_neural_network = _module
websockets_mnist = _module
run_websocket_client = _module
start_websocket_servers = _module
grid = _module
federated_learning = _module
mnist = _module
spam_prediction = _module
handcrafted_GRU = _module
websocket = _module
deploy_workers = _module
pen_testing = _module
steal_data_over_sockets = _module
run_websocket_server = _module
setup = _module
syft = _module
codes = _module
common = _module
util = _module
dependency_check = _module
exceptions = _module
execution = _module
action = _module
communication = _module
computation = _module
placeholder = _module
placeholder_id = _module
plan = _module
protocol = _module
role = _module
role_assignments = _module
state = _module
tracing = _module
translation = _module
abstract = _module
default = _module
torchscript = _module
type_wrapper = _module
federated = _module
fl_client = _module
fl_job = _module
floptimizer = _module
frameworks = _module
keras = _module
hook = _module
layers = _module
constructor = _module
model = _module
sequential = _module
tensorflow = _module
dp = _module
pate = _module
fl = _module
dataloader = _module
dataset = _module
utils = _module
functions = _module
he = _module
ciphertext = _module
context = _module
decryptor = _module
encryption_params = _module
encryptor = _module
evaluator = _module
integer_encoder = _module
key_generator = _module
modulus = _module
plaintext = _module
public_key = _module
secret_key = _module
base_converter = _module
global_variable = _module
numth = _module
operations = _module
rlwe = _module
rns_base = _module
rns_tool = _module
paillier = _module
hook = _module
hook_args = _module
linalg = _module
lr = _module
operations = _module
mpc = _module
beaver = _module
fss = _module
primitives = _module
securenn = _module
spdz = _module
nn = _module
conv = _module
functional = _module
pool = _module
rnn = _module
tensors = _module
decorators = _module
logging = _module
interpreters = _module
additive_shared = _module
autograd = _module
build_gradients = _module
gradients = _module
gradients_core = _module
native = _module
numpy = _module
paillier = _module
polynomial = _module
precision = _module
private = _module
torch_attributes = _module
generic = _module
hookable = _module
message_handler = _module
object = _module
sendable = _module
tensor = _module
attributes = _module
hook = _module
hook_args = _module
pointers = _module
string = _module
tensors = _module
overload = _module
remote = _module
types = _module
id_provider = _module
metrics = _module
object_storage = _module
callable_pointer = _module
multi_pointer = _module
object_pointer = _module
object_wrapper = _module
pointer_dataset = _module
pointer_plan = _module
pointer_tensor = _module
string_pointer = _module
abstract_grid = _module
authentication = _module
account = _module
credential = _module
gcloud = _module
test = _module
terraform_notebook = _module
terraform_script = _module
grid_client = _module
network = _module
nodes_manager = _module
peer_events = _module
private_grid = _module
public_grid = _module
webrtc_connection = _module
messaging = _module
message = _module
sandbox = _module
serde = _module
compression = _module
msgpack = _module
native_serde = _module
proto = _module
serde = _module
torch_serde = _module
protobuf = _module
native_serde = _module
serde = _module
torch_serde = _module
syft_serializable = _module
serde = _module
version = _module
workers = _module
base = _module
message_handler = _module
node_client = _module
tfe = _module
virtual = _module
websocket_client = _module
websocket_server = _module
test_util = _module
conftest = _module
efficiency = _module
assertions = _module
test_activations_time = _module
test_linalg_time = _module
test_communication = _module
test_package_wrapper = _module
test_placeholder = _module
test_plan = _module
test_protocol = _module
test_role = _module
test_role_assignments = _module
test_state = _module
test_translation = _module
test_callable_pointer = _module
test_dataset_pointer = _module
test_multi_pointer = _module
test_pointer_plan = _module
test_pointer_tensor = _module
test_autograd = _module
test_functions = _module
test_gc = _module
test_hookable = _module
test_id_provider = _module
test_logging = _module
test_object_storage = _module
test_private = _module
test_string = _module
test_sequential = _module
test_message = _module
test_notebooks = _module
test_msgpack_serde = _module
test_msgpack_serde_full = _module
test_protobuf_serde = _module
test_protobuf_serde_full = _module
serde_helpers = _module
test_dependency_check = _module
test_exceptions = _module
test_grid = _module
test_local_worker = _module
test_sandbox = _module
test_udacity = _module
differential_privacy = _module
test_pate = _module
test_dataloader = _module
test_dataset = _module
test_utils = _module
test_hook = _module
test_hook_args = _module
test_lr = _module
test_operations = _module
test_crypto_store = _module
test_fss = _module
test_multiparty_nn = _module
test_securenn = _module
test_functional = _module
test_nn = _module
test_additive_shared = _module
test_fv = _module
test_native = _module
test_numpy = _module
test_paillier = _module
test_parameter = _module
test_polynomial = _module
test_precision = _module
test_tensor = _module
test_federated_learning = _module
test_hook = _module
test_base = _module
test_virtual = _module
test_websocket_worker = _module
test_worker = _module

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


import torch as th


import numpy as np


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import torch.nn.functional as f


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import logging


from typing import Tuple


from typing import List


from typing import Union


import copy


import inspect


import warnings


from torch import jit


import math


from torch.utils.data import SequentialSampler


from torch.utils.data import RandomSampler


from torch.utils.data import BatchSampler


from torch._six import string_classes


from torch._six import int_classes


from torch._six import container_abcs


from torch.utils.data import Dataset


from typing import Dict


from typing import Any


from numpy.polynomial import polynomial as poly


from collections import defaultdict


from enum import Enum


import random


from torch.distributions import Normal


from math import gcd


from functools import wraps


from math import inf


from typing import Callable


from torch.nn import Module


from torch.nn import init


import re


from types import ModuleType


from abc import ABC


import functools


from typing import Set


from abc import abstractmethod


import types


from typing import TYPE_CHECKING


import time


from collections import OrderedDict


from typing import Collection


import numpy


import itertools


from itertools import starmap


from torch import Tensor


from functools import partial


from torch.nn import Parameter


from torch import optim


from types import MethodType


from time import time


class AbstractObject(ABC):
    """
    This is a generic object abstraction.
    """
    is_wrapper = False

    def __init__(self, id: int=None, owner: 'sy.workers.AbstractWorker'=None, tags: Set[str]=None, description: str=None, child=None):
        """Initializer for AbstractTensor

        Args:
            id: An optional string or integer id of the tensor
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
            child: an optional tensor to put in the .child attribute to build
                a chain of tensors
        """
        self.owner = owner or sy.local_worker
        self.id = id or sy.ID_PROVIDER.pop()
        self.tags = tags or set()
        self.description = description
        self.child = child

    def __str__(self) ->str:
        if hasattr(self, 'child'):
            return type(self).__name__ + '>' + self.child.__str__()
        else:
            return type(self).__name__

    def __repr__(self) ->str:
        if hasattr(self, 'child'):
            return type(self).__name__ + '>' + self.child.__repr__()
        else:
            return type(self).__name__

    def describe(self, description: str) ->'AbstractObject':
        self.description = description
        return self

    def tag(self, *tags: str) ->'AbstractObject':
        self.tags = self.tags or set()
        for tag in tags:
            self.tags.add(tag)
        self.owner.object_store.register_tags(self)
        return self

    def get_class_attributes(self):
        """
        Return all elements which defines an instance of a certain class.
        By default there is nothing so we return an empty dict, but for
        example for fixed precision tensor, the fractional precision is
        very important.
        """
        return {}

    @classmethod
    def on_function_call(cls, *args):
        """
        Override this to perform a specific action for each call of a torch
        function with arguments containing syft tensors of the class doing
        the overloading
        """
        pass

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Syft Tensor,
        Replace in the args_ all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a Syft Tensor on top of all tensors found in
        the response.

        Args:
            command: instruction of a function command: (command name,
            <no self>, arguments[, kwargs_])

        Returns:
            the response of the function command
        """
        cmd, _, args_, kwargs_ = command
        try:
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args_, **kwargs_)
        except AttributeError:
            pass
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args_, kwargs_)
        new_command = cmd, None, new_args, new_kwargs
        cls.on_function_call(new_command)
        response = new_type.handle_func_command(new_command)
        response = hook_args.hook_response(cmd, response, wrap_type=cls)
        return response

    @classmethod
    def rgetattr(cls, obj, attr, *args):
        """
        Get an attribute recursively.

        This is a core piece of functionality for the PySyft tensor chain.

        Args:
            obj: the object holding the attribute
            attr: nested attribute
            args: optional arguments to provide

        Returns:
            the attribute obj.attr

        Example:
            >>> rgetattr(obj, 'attr1.attr2.attr3')
            [Out] obj.attr1.attr2.attr3

        """

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [obj] + attr.split('.'))


class SyftSerializable:
    """
        Interface for the communication protocols in syft.

        syft-proto methods:
            1. bufferize
            2. unbufferize
            3. get_protobuf_schema

        msgpack methods:
            1. simplify
            2. detail

        Note: the interface can be inherited from parent class, but each class
        has to write it's own explicit methods, even if they are the ones from the parent class.
    """

    @staticmethod
    def simplify(worker, obj):
        """
            Serialization method for msgpack.

            Parameters:
                worker: the worker on which the serialization is being made.
                obj: the object to be serialized, an instantiated type of
                the class that implements SyftSerializable.

            Returns:
                Serialized object using msgpack.
        """
        raise NotImplementedError

    @staticmethod
    def detail(worker, obj):
        """
            Deserialization method for msgpack.

            Parameters:
                worker: the worker on which the serialization is being made.
                obj: the object to be deserialized, a serialized object of
                the class that implements SyftSerializable.

            Returns:
                Serialized object using msgpack.
        """
        raise NotImplementedError

    @staticmethod
    def bufferize(worker, obj):
        """
            Serialization method for protobuf.

            Parameters:
                worker: the worker on which the bufferize is being made.
                obj: the object to be bufferized using protobufers, an instantiated type
                of the class that implements SyftSerializable.

            Returns:
                Protobuf class for the current type.
        """
        raise NotImplementedError

    @staticmethod
    def get_msgpack_code():
        """
            Method that provides a code for msgpack if the type is not present in proto.json.

            The returned object should be similar to:
            {
                "code": int value,
                "forced_code": int value
            }

            Both keys are optional, the common and right way would be to add only the "code" key.

            Returns:
                dict: A dict with the "code" or "forced_code" keys.
        """
        raise NotImplementedError

    @staticmethod
    def unbufferize(worker, obj):
        """
            Deserialization method for protobuf.

            Parameters:
                worker: the worker on which the unbufferize is being made.
                obj: the object to be unbufferized using protobufers, an instantiated type
                of the class that implements SyftSerializable.

            Returns:
                Protobuf class for the current type.
        """
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema():
        """
            Returns the protobuf schema used for this type.

            Returns:
                Protobuf type.
        """
        raise NotImplementedError

    @staticmethod
    def get_original_class():
        """
            Returns the original type, only used in wrappers.

            Returns:
                Wrapped type.
        """
        return NotImplementedError


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def send(self, destination):
        return self.owner.send_obj(self, destination)


class AbstractTensor(AbstractSendable, SyftSerializable):

    def __init__(self, id: int=None, owner: 'sy.workers.AbstractWorker'=None, tags: List[str]=None, description: str=None, child=None):
        super(AbstractTensor, self).__init__(id, owner, tags, description, child)

    def wrap(self, register=True, type=None, **kwargs):
        """Wraps the class inside an empty object of class `type`.

        Because PyTorch/TF do not (yet) support functionality for creating
        arbitrary Tensor types (via subclassing torch.Tensor), in order for our
        new tensor types (such as PointerTensor) to be usable by the rest of
        PyTorch/TF (such as PyTorch's layers and loss functions), we need to
        wrap all of our new tensor types inside of a native PyTorch type.

        This function adds a .wrap() function to all of our tensor types (by
        adding it to AbstractTensor), such that (on any custom tensor
        my_tensor), my_tensor.wrap() will return a tensor that is compatible
        with the rest of the PyTorch/TensorFlow API.

        Returns:
            A wrapper tensor of class `type`, or whatever is specified as
            default by the current syft.framework.Tensor.
        """
        wrapper = sy.framework.hook.create_wrapper(type, **kwargs)
        wrapper.child = self
        wrapper.is_wrapper = True
        wrapper.child.parent = weakref.ref(wrapper)
        if self.id is None:
            self.id = sy.ID_PROVIDER.pop()
        if self.owner is not None and register:
            self.owner.register_obj(wrapper, obj_id=self.id)
        return wrapper

    def on(self, tensor: 'AbstractTensor', wrap: bool=True) ->'AbstractTensor':
        """
        Add a syft(log) tensor on top of the tensor.

        Args:
            tensor: the tensor to extend
            wrap: if true, add the syft tensor between the wrapper
            and the rest of the chain. If false, just add it at the top

        Returns:
            a syft/torch tensor
        """
        if not wrap:
            self.child = tensor
            return self
        else:
            if not hasattr(tensor, 'child'):
                tensor = tensor.wrap()
            self.id = tensor.id
            self.child = tensor.child
            tensor.child = self
            return tensor

    def copy(self):
        return self + 0

    def clone(self):
        """
        Clone should keep ids unchanged, contrary to copy
        """
        cloned_tensor = type(self)(**self.get_class_attributes())
        cloned_tensor.id = self.id
        cloned_tensor.owner = self.owner
        if hasattr(self, 'child') and self.child is not None:
            cloned_tensor.child = self.child.clone()
        return cloned_tensor

    def refresh(self):
        """
        Forward to Additive Shared Tensor the call to refresh shares
        """
        if hasattr(self, 'child'):
            self.child = self.child.refresh()
            return self
        else:
            raise AttributeError('Refresh should only be called on AdditiveSharedTensors')

    @property
    def shape(self):
        return self.child.shape

    def __len__(self) ->int:
        """Alias .shape[0] with len(), helpful for pointers"""
        try:
            if hasattr(self, 'child') and not isinstance(self.child, dict):
                return self.child.shape[0]
            else:
                return self.shape[0]
        except IndexError:
            return 0

    @property
    def grad(self):
        child_grad = self.child.grad
        if child_grad is None:
            return None
        else:
            return child_grad.wrap()

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        Syft tensor which has a child which is a pointer, an additive shared tensor,
        a multi-pointer, etc."""
        class_attributes = self.get_class_attributes()
        return type(self)(**class_attributes, owner=self.owner, tags=self.tags, description=self.description, id=self.id).on(self.child.get())

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""
        child_id = self.id
        tensor = self.get()
        tensor.id = child_id
        self.owner.register_obj(tensor)


class AbstractWorker(ABC, SyftSerializable):

    @abstractmethod
    def _send_msg(self, message: bin, location: 'AbstractWorker'):
        """Sends message from one worker to another.

        As AbstractWorker implies, you should never instantiate this class by
        itself. Instead, you should extend AbstractWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: A binary message to be sent from one worker
                to another.
            location: A AbstractWorker instance that lets you provide the
                destination to send the message.
        """
        pass

    @abstractmethod
    def _recv_msg(self, message: bin):
        """Receives the message.

        As AbstractWorker implies, you should never instantiate this class by
        itself. Instead, you should extend AbstractWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: The binary message being received.
        """
        pass


protocol_store = {}


def crypto_protocol(protocol_name):
    """
    Decorator to define a specific operation behaviour depending on the crypto
    protocol used

    Args:
        protocol_name: the name of the protocol. Currently supported:
            - snn: SecureNN
            - fss: Function Secret Sharing

    Example in a tensor file:
        ```
        @crypto_protocol("snn")
        def foo(...):
            # SNN specific code

        @crypto_protocol("fss")
        def foo(...):
            # FSS specific code
        ```

        See additive_sharing.py for more usage
    """

    def decorator(f):
        name = f.__qualname__
        protocol_store[name, protocol_name] = f

        def method(self, *args, **kwargs):
            f = protocol_store[name, self.protocol]
            return f(self, *args, **kwargs)
        return method
    return decorator


class memorize(dict):
    """
    This is a decorator to cache a function output when the function is
    deterministic and the input space is small. In such condition, the
    function will be called many times to perform the same computation
    so we want this computation to be cached.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result


no_wrap = {'no_wrap': True}


class Overloaded:

    def __init__(self):
        self.method = Overloaded.overload_method
        self.function = Overloaded.overload_function
        self.module = Overloaded.overload_module

    @staticmethod
    def overload_method(attr):
        """
        hook args and response for methods that hold the @overloaded.method decorator
        """

        def _hook_method_args(self, *args, **kwargs):
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(attr.__name__, self, args, kwargs)
            response = attr(self, new_self, *new_args, **new_kwargs)
            response = hook_args.hook_response(attr.__name__, response, wrap_type=type(self), wrap_args=self.get_class_attributes())
            return response
        return _hook_method_args

    @staticmethod
    def overload_function(attr):
        """
        hook args and response for functions that hold the @overloaded.function decorator
        """

        def _hook_function_args(*args, **kwargs):
            tensor = args[0] if not isinstance(args[0], (tuple, list)) else args[0][0]
            cls = type(tensor)
            new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(attr.__name__, args, kwargs)
            response = attr(*new_args, **new_kwargs)
            response = hook_args.hook_response(attr.__name__, response, wrap_type=cls, wrap_args=tensor.get_class_attributes())
            return response
        return _hook_function_args

    @staticmethod
    def overload_module(attr):
        module = Module()
        attr(module)
        return module


overloaded = Overloaded()


class Message(ABC, SyftSerializable):
    """All syft message types extend this class

    All messages in the pysyft protocol extend this class. This abstraction
    requires that every message has an integer type, which is important because
    this integer is what determines how the message is handled when a BaseWorker
    receives it.

    Additionally, this type supports a default simplifier and detailer, which are
    important parts of PySyft's serialization and deserialization functionality.
    You can read more abouty detailers and simplifiers in syft/serde/serde.py.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        """Return a human readable version of this message"""
        pass

    def __repr__(self):
        """Return a human readable version of this message"""
        return self.__str__()


class ForceObjectDeleteMessage(Message):
    """Garbage collect a remote object

    This is the dominant message for garbage collection of remote objects. When
    a pointer is deleted, this message is triggered by default to tell the object
    being pointed to to also delete itself.
    """

    def __init__(self, obj_id):
        """Initialize the message."""
        self.object_id = obj_id

    def __str__(self):
        """Return a human readable version of this message"""
        return f'({type(self).__name__} {self.object_id})'

    @staticmethod
    def simplify(worker: AbstractWorker, msg: 'ForceObjectDeleteMessage') ->tuple:
        """
        This function takes the attributes of a Message and saves them in a tuple.
        The detail() method runs the inverse of this method.
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            msg (Message): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(msg)
        """
        return sy.serde.msgpack.serde._simplify(worker, msg.object_id),

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) ->'ForceObjectDeleteMessage':
        """
        This function takes the simplified tuple version of this message and converts
        it into an ForceObjectDeleteMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            msg (ForceObjectDeleteMessage): a ForceObjectDeleteMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        return ForceObjectDeleteMessage(sy.serde.msgpack.serde._detail(worker, msg_tuple[0]))

    @staticmethod
    def bufferize(worker, msg):
        """
            This method serializes a ForceObjectDeleteMessage using ForceObjectDeleteMessagePB.

            Args:
                msg (ForceObjectDeleteMessage): input ForceObjectDeleteMessage to be serialized.

            Returns:
                proto_msg (ForceObjectDeleteMessagePB): serialized ForceObjectDeleteMessage.
        """
        proto_msg = ForceObjectDeleteMessagePB()
        sy.serde.protobuf.proto.set_protobuf_id(proto_msg.object_id, msg.object_id)
        return proto_msg

    @staticmethod
    def unbufferize(worker, proto_msg):
        """
            This method deserializes ForceObjectDeleteMessagePB into ForceObjectDeleteMessage.

            Args:
                proto_msg (ForceObjectDeleteMessagePB): input serialized ForceObjectDeleteMessagePB.

            Returns:
                ForceObjectDeleteMessage: deserialized ForceObjectDeleteMessagePB.
        """
        obj_id = sy.serde.protobuf.proto.get_protobuf_id(proto_msg.object_id)
        return ForceObjectDeleteMessage(obj_id=obj_id)

    @staticmethod
    def get_protobuf_schema():
        """
            Returns the protobuf schema used for ForceObjectDeleteMessage.

            Returns:
                Protobuf schema for ForceObjectDeleteMessage.
        """
        return ForceObjectDeleteMessagePB


class ObjectPointer(AbstractSendable, SyftSerializable):
    """A pointer to a remote object.

    An ObjectPointer forwards all API calls to the remote. ObjectPointer objects
    point to objects. They exist to mimic the entire
    API of an object, but instead of computing a function locally
    (such as addition, subtraction, etc.) they forward the computation to a
    remote machine as specified by self.location. Specifically, every
    ObjectPointer has a object located somewhere that it points to (they should
    never exist by themselves).
    The objects being pointed to can be on the same machine or (more commonly)
    on a different one. Note further that a ObjectPointer does not know the
    nature how it sends messages to the object it points to (whether over
    socket, http, or some other protocol) as that functionality is abstracted
    in the BaseWorker object in self.location.
    """

    def __init__(self, location: 'BaseWorker'=None, id_at_location: Union[str, int]=None, owner: 'BaseWorker'=None, id: Union[str, int]=None, garbage_collect_data: bool=True, point_to_attr: str=None, tags: List[str]=None, description: str=None):
        """Initializes a ObjectPointer.

        Args:
            location: An optional BaseWorker object which points to the worker
                on which this pointer's object can be found.
            id_at_location: An optional string or integer id of the object
                being pointed to.
            owner: An optional BaseWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the ObjectPointer.
            garbage_collect_data: If true (default), delete the remote object when the
                pointer is deleted.
            point_to_attr: string which can tell a pointer to not point directly to                an object, but to point to an attribute of that object such as .child or
                .grad. Note the string can be a chain (i.e., .child.child.child or
                .grad.child.child). Defaults to None, which means don't point to any attr,
                just point to then object corresponding to the id_at_location.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.location = location
        self.id_at_location = id_at_location
        self.garbage_collect_data = garbage_collect_data
        self.point_to_attr = point_to_attr

    @staticmethod
    def create_pointer(obj, location: 'AbstractWorker'=None, id_at_location: (str or int)=None, register: bool=False, owner: 'AbstractWorker'=None, ptr_id: (str or int)=None, garbage_collect_data=None) ->'ObjectPointer':
        """Creates a pointer to the "self" FrameworkTensor object.

        This method is called on a FrameworkTensor object, returning a pointer
        to that object. This method is the CORRECT way to create a pointer,
        and the parameters of this method give all possible attributes that
        a pointer can be created with.

        Args:
            location: The AbstractWorker object which points to the worker on which
                this pointer's object can be found. In nearly all cases, this
                is self.owner and so this attribute can usually be left blank.
                Very rarely you may know that you are about to move the Tensor
                to another worker so you can pre-initialize the location
                attribute of the pointer to some other worker, but this is a
                rare exception.
            id_at_location: A string or integer id of the tensor being pointed
                to. Similar to location, this parameter is almost always
                self.id and so you can leave this parameter to None. The only
                exception is if you happen to know that the ID is going to be
                something different than self.id, but again this is very rare
                and most of the time, setting this means that you are probably
                doing something you shouldn't.
            register: A boolean parameter (default False) that determines
                whether to register the new pointer that gets created. This is
                set to false by default because most of the time a pointer is
                initialized in this way so that it can be sent to someone else
                (i.e., "Oh you need to point to my tensor? let me create a
                pointer and send it to you" ). Thus, when a pointer gets
                created, we want to skip being registered on the local worker
                because the pointer is about to be sent elsewhere. However, if
                you are initializing a pointer you intend to keep, then it is
                probably a good idea to register it, especially if there is any
                chance that someone else will initialize a pointer to your
                pointer.
            owner: A AbstractWorker parameter to specify the worker on which the
                pointer is located. It is also where the pointer is registered
                if register is set to True.
            ptr_id: A string or integer parameter to specify the id of the pointer
                in case you wish to set it manually for any special reason.
                Otherwise, it will be set randomly.
            garbage_collect_data: If true (default), delete the remote tensor when the
                pointer is deleted.
            local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            preinitialize_grad: Initialize gradient for AutogradTensors to a tensor.

        Returns:
            A FrameworkTensor[ObjectPointer] pointer to self. Note that this
            object itself will likely be wrapped by a FrameworkTensor wrapper.
        """
        if owner is None:
            owner = obj.owner
        if location is None:
            location = obj.owner.id
        owner = obj.owner.get_worker(owner)
        location = obj.owner.get_worker(location)
        ptr = ObjectPointer(location=location, id_at_location=id_at_location, owner=owner, id=ptr_id, garbage_collect_data=True if garbage_collect_data is None else garbage_collect_data, tags=obj.tags, description=obj.description)
        return ptr

    def wrap(self, register=True, type=None, **kwargs):
        """Wraps the class inside framework tensor.

        Because PyTorch/TF do not (yet) support functionality for creating
        arbitrary Tensor types (via subclassing torch.Tensor), in order for our
        new tensor types (such as PointerTensor) to be usable by the rest of
        PyTorch/TF (such as PyTorch's layers and loss functions), we need to
        wrap all of our new tensor types inside of a native PyTorch type.

        This function adds a .wrap() function to all of our tensor types (by
        adding it to AbstractTensor), such that (on any custom tensor
        my_tensor), my_tensor.wrap() will return a tensor that is compatible
        with the rest of the PyTorch/TensorFlow API.

        Returns:
            A wrapper tensor of class `type`, or whatever is specified as
            default by the current syft.framework.Tensor.
        """
        wrapper = syft.framework.hook.create_wrapper(type, **kwargs)
        wrapper.child = self
        wrapper.is_wrapper = True
        wrapper.child.parent = weakref.ref(wrapper)
        if self.id is None:
            self.id = syft.ID_PROVIDER.pop()
        if self.owner is not None and register:
            self.owner.register_obj(wrapper, obj_id=self.id)
        return wrapper

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Pointer,
        Get the remote location to send the command, send it and get a
        pointer to the response, return.
        :param command: instruction of a function command: (command name,
        None, arguments[, kwargs_])
        :return: the response of the function command
        """
        pointer = cls.find_a_pointer(command)
        owner = pointer.owner
        location = pointer.location
        cmd, _, args_, kwargs_ = command
        response = owner.send_command(location, cmd_name=cmd, args_=args_, kwargs_=kwargs_)
        return response

    @classmethod
    def find_a_pointer(cls, command):
        """
        Find and return the first pointer in the args_ object, using a trick
        with the raising error RemoteObjectFoundError
        """
        try:
            cmd, _, args_, kwargs_ = command
            _ = hook_args.unwrap_args_from_function(cmd, args_, kwargs_)
        except exceptions.RemoteObjectFoundError as err:
            pointer = err.pointer
            return pointer

    def get(self, user=None, reason: str='', deregister_ptr: bool=True):
        """Requests the object being pointed to.

        The object to which the pointer points will be requested, serialized and returned.

        Note:
            This will typically mean that the remote object will be
            removed/destroyed.

        Args:
            user (obj, optional) : authenticate/allow user to perform get on remote private objects.
            reason (str, optional) : a description of why the data scientist wants to see it.
            deregister_ptr (bool, optional): this determines whether to
                deregister this pointer from the pointer's owner during this
                method. This defaults to True because the main reason people use
                this method is to move the tensor from the location to the
                local one, at which time the pointer has no use.

        Returns:
            An AbstractObject object which is the tensor (or chain) that this
            object used to point to on a location.

        TODO: add param get_copy which doesn't destroy remote if true.
        """
        if self.point_to_attr is not None:
            raise exceptions.CannotRequestObjectAttribute('You called .get() on a pointer to a tensor attribute. This is not yet supported. Call .clone().get() instead.')
        if self.location == self.owner:
            obj = self.owner.get_obj(self.id_at_location)
            if hasattr(obj, 'child'):
                obj = obj.child
        else:
            obj = self.owner.request_obj(self.id_at_location, self.location, user, reason)
        if deregister_ptr:
            self.owner.de_register_obj(self)
        if self.garbage_collect_data:
            self.garbage_collect_data = False
        return obj

    def __str__(self):
        """Returns a string version of this pointer.

        This is primarily for end users to quickly see things about the object.
        This tostring shouldn't be used for anything else though as it's likely
        to change. (aka, don't try to parse it to extract information. Read the
        attribute you need directly). Also, don't use this to-string as a
        serialized form of the pointer.
        """
        type_name = type(self).__name__
        out = f'[{type_name} | {str(self.owner.id)}:{self.id} -> {str(self.location.id)}:{self.id_at_location}]'
        if self.point_to_attr is not None:
            out += '::' + str(self.point_to_attr).replace('.', '::')
        big_str = False
        if self.tags is not None and len(self.tags):
            big_str = True
            out += '\n\tTags: '
            for tag in self.tags:
                out += str(tag) + ' '
        if big_str and hasattr(self, 'shape'):
            out += '\n\tShape: ' + str(self.shape)
        if self.description is not None:
            big_str = True
            out += '\n\tDescription: ' + str(self.description).split('\n')[0] + '...'
        return out

    def __repr__(self):
        """Returns the to-string method.

        When called using __repr__, most commonly seen when returned as cells
        in Jupyter notebooks.
        """
        return self.__str__()

    def __del__(self):
        """This method garbage collects the object this pointer is pointing to.
        By default, PySyft assumes that every object only has one pointer to it.
        Thus, if the pointer gets garbage collected, we want to automatically
        garbage collect the object being pointed to.
        """
        if hasattr(self, 'owner') and self.garbage_collect_data:
            if self.point_to_attr is None:
                self.owner.send_msg(ForceObjectDeleteMessage(self.id_at_location), self.location)

    def _create_attr_name_string(self, attr_name):
        if self.point_to_attr is not None:
            point_to_attr = f'{self.point_to_attr}.{attr_name}'
        else:
            point_to_attr = attr_name
        return point_to_attr

    def attr(self, attr_name):
        attr_ptr = syft.ObjectPointer(id=self.id, owner=self.owner, location=self.location, id_at_location=self.id_at_location, point_to_attr=self._create_attr_name_string(attr_name))
        self.__setattr__(attr_name, attr_ptr)
        return attr_ptr

    def setattr(self, name, value):
        self.owner.send_command(cmd_name='__setattr__', target=self, args_=(name, value), kwargs_={}, recipient=self.location)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: 'ObjectPointer') ->tuple:
        """
        This function takes the attributes of a ObjectPointer and saves them in a dictionary
        Args:
            ptr (ObjectPointer): a ObjectPointer
        Returns:
            tuple: a tuple holding the unique attributes of the pointer
        Examples:
            data = simplify(ptr)
        """
        return syft.serde.msgpack.serde._simplify(worker, ptr.id), syft.serde.msgpack.serde._simplify(worker, ptr.id_at_location), syft.serde.msgpack.serde._simplify(worker, ptr.location.id), syft.serde.msgpack.serde._simplify(worker, ptr.point_to_attr), ptr.garbage_collect_data

    @staticmethod
    def detail(worker: 'AbstractWorker', object_tuple: tuple) ->'ObjectPointer':
        """
        This function reconstructs an ObjectPointer given it's attributes in form of a dictionary.
        We use the spread operator to pass the dict data as arguments
        to the init method of ObjectPointer
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the ObjectPointer
        Returns:
            ObjectPointer: an ObjectPointer
        Examples:
            ptr = detail(data)
        """
        obj_id, id_at_location, worker_id, point_to_attr, garbage_collect_data = object_tuple
        obj_id = syft.serde.msgpack.serde._detail(worker, obj_id)
        id_at_location = syft.serde.msgpack.serde._detail(worker, id_at_location)
        worker_id = syft.serde.msgpack.serde._detail(worker, worker_id)
        point_to_attr = syft.serde.msgpack.serde._detail(worker, point_to_attr)
        if worker_id == worker.id:
            obj = worker.get_obj(id_at_location)
            if point_to_attr is not None and obj is not None:
                point_to_attrs = point_to_attr.split('.')
                for attr in point_to_attrs:
                    if len(attr) > 0:
                        obj = getattr(obj, attr)
                if obj is not None:
                    if not obj.is_wrapper and not isinstance(obj, FrameworkTensor):
                        obj = obj.wrap()
            return obj
        else:
            location = syft.hook.local_worker.get_worker(worker_id)
            ptr = ObjectPointer(location=location, id_at_location=id_at_location, owner=worker, id=obj_id, garbage_collect_data=garbage_collect_data)
            return ptr


class Action(ABC, SyftSerializable):
    """Describes the concrete steps workers can take with objects they own

    In Syft, an Action is when one worker wishes to tell another worker to do something with
    objects contained in the worker.object_store registry (or whatever the official object store is
    backed with in the case that it's been overridden). For example, telling a worker to take two
    tensors and add them together is an Action. Sending an object from one worker to another is
    also an Action."""

    def __init__(self, name: str, target, args_: tuple, kwargs_: dict, return_ids: tuple, return_value=False):
        """Initialize an action

        Args:
            name (String): The name of the method to be invoked (e.g. "__add__")
            target (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.),
                the id of action results are set by the client. This allows the client to be able to
                predict where the results will be ahead of time. Importantly, this allows the
                client to pre-initalize the pointers to the future data, regardless of whether
                the action has yet executed. It also reduces the size of the response from the
                action (which is very often empty).
            return_value (boolean): return the result or not. If true, the result is directly
                returned, if not, the command sender will create a pointer to the remote result
                using the return_ids and will need to do .get() later to get the result.

        """
        super().__init__()
        self.name = name
        self.target = target
        self.args = args_
        self.kwargs = kwargs_
        self.return_ids = return_ids
        self.return_value = return_value
        self._type_check('name', str)
        self._type_check('args', tuple)
        self._type_check('kwargs', dict)
        self._type_check('return_ids', tuple)

    def __eq__(self, other):
        return self.name == other.name and self.target == other.target and self.args == other.args and self.kwargs == other.kwargs and self.return_ids == other.return_ids

    def code(self, var_names=None) ->str:
        """Returns pseudo-code representation of computation action"""

        def stringify(obj):
            if isinstance(obj, PlaceholderId):
                id = obj.value
                if var_names is None:
                    ret = f'var_{id}'
                elif id in var_names:
                    ret = var_names[id]
                else:
                    idx = sum('var_' in k for k in var_names.values())
                    name = f'var_{idx}'
                    var_names[id] = name
                    ret = name
            elif isinstance(obj, PlaceHolder):
                ret = stringify(obj.id)
            elif isinstance(obj, (tuple, list)):
                ret = ', '.join(stringify(o) for o in obj)
            else:
                ret = str(obj)
            return ret
        out = ''
        if self.return_ids is not None:
            out += stringify(self.return_ids) + ' = '
        if self.target is not None:
            out += stringify(self.target) + '.'
        out += self.name + '('
        out += stringify(self.args)
        if self.kwargs:
            if len(self.args) > 0:
                out += ', '
            out += ', '.join(f'{k}={w}' for k, w in self.kwargs.items())
        out += ')'
        return out

    def __str__(self) ->str:
        """Returns string representation of this action"""
        return f'{type(self).__name__}[{self.code()}]'

    def _type_check(self, field_name, expected_type):
        actual_value = getattr(self, field_name)
        assert actual_value is None or isinstance(actual_value, expected_type), f'{field_name} must be {expected_type.__name__}, but was {type(actual_value).__name__}: {actual_value}.'

    @staticmethod
    @abstractmethod
    def simplify(worker: AbstractWorker, action: 'Action') ->tuple:
        """
        This function takes the attributes of a CommunicationAction and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (CommunicationAction): a CommunicationAction
        Returns:
            tuple: a tuple holding the unique attributes of the CommunicationAction
        Examples:
            data = simplify(worker, action)
        """
        return sy.serde.msgpack.serde._simplify(worker, action.name), sy.serde.msgpack.serde._simplify(worker, action.target), sy.serde.msgpack.serde._simplify(worker, action.args), sy.serde.msgpack.serde._simplify(worker, action.kwargs), sy.serde.msgpack.serde._simplify(worker, action.return_ids), sy.serde.msgpack.serde._simplify(worker, action.return_value)

    @staticmethod
    @abstractmethod
    def detail(worker: AbstractWorker, action_tuple: tuple) ->'Action':
        """
        This function takes the simplified tuple version of this message and converts
        it into a CommunicationAction. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            communication_tuple (Tuple): the raw information being detailed.
        Returns:
            communication (CommunicationAction): a CommunicationAction.
        Examples:
            communication = detail(sy.local_worker, communication_tuple)
        """
        name, target, args_, kwargs_, return_ids, return_value = action_tuple
        return sy.serde.msgpack.serde._detail(worker, name), sy.serde.msgpack.serde._detail(worker, target), sy.serde.msgpack.serde._detail(worker, args_), sy.serde.msgpack.serde._detail(worker, kwargs_), sy.serde.msgpack.serde._detail(worker, return_ids), sy.serde.msgpack.serde._detail(worker, return_value)

    @staticmethod
    @abstractmethod
    def bufferize(worker: AbstractWorker, action: 'Action', protobuf_action):
        """
        This function takes the attributes of a Action and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (Action): an Action
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_action.command = action.name
        protobuf_target = None
        if isinstance(action.target, sy.generic.pointers.pointer_tensor.PointerTensor):
            protobuf_target = protobuf_action.target_pointer
        elif isinstance(action.target, sy.execution.placeholder_id.PlaceholderId):
            protobuf_target = protobuf_action.target_placeholder_id
        elif isinstance(action.target, (int, str)):
            sy.serde.protobuf.proto.set_protobuf_id(protobuf_action.target_id, action.target)
        elif action.target is not None:
            protobuf_target = protobuf_action.target_tensor
        if protobuf_target is not None:
            protobuf_target.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, action.target))
        if action.args:
            protobuf_action.args.extend(sy.serde.protobuf.serde.bufferize_args(worker, action.args))
        if action.kwargs:
            for key, value in action.kwargs.items():
                protobuf_action.kwargs.get_or_create(key).CopyFrom(sy.serde.protobuf.serde.bufferize_arg(worker, value))
        if action.return_ids is not None:
            if not isinstance(action.return_ids, (list, tuple)):
                return_ids = action.return_ids,
            else:
                return_ids = action.return_ids
            for return_id in return_ids:
                if isinstance(return_id, PlaceholderId):
                    protobuf_action.return_placeholder_ids.append(sy.serde.protobuf.serde._bufferize(worker, return_id))
                else:
                    sy.serde.protobuf.proto.set_protobuf_id(protobuf_action.return_ids.add(), return_id)
        return protobuf_action

    @staticmethod
    @abstractmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj):
        """
        This function takes the Protobuf version of this message and converts
        it into an Action. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (ActionPB): the Protobuf message

        Returns:
            obj (tuple): a tuple of the args required to instantiate an Action object

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        command = protobuf_obj.command
        protobuf_target = protobuf_obj.WhichOneof('target')
        if protobuf_target:
            target = sy.serde.protobuf.serde._unbufferize(worker, getattr(protobuf_obj, protobuf_obj.WhichOneof('target')))
        else:
            target = None
        args_ = sy.serde.protobuf.serde.unbufferize_args(worker, protobuf_obj.args)
        kwargs_ = {}
        for key in protobuf_obj.kwargs:
            kwargs_[key] = sy.serde.protobuf.serde.unbufferize_arg(worker, protobuf_obj.kwargs[key])
        return_ids = tuple(sy.serde.protobuf.proto.get_protobuf_id(pb_id) for pb_id in protobuf_obj.return_ids)
        return_placeholder_ids = tuple(sy.serde.protobuf.serde._unbufferize(worker, placeholder) for placeholder in protobuf_obj.return_placeholder_ids)
        if return_placeholder_ids:
            action = command, target, args_, kwargs_, return_placeholder_ids
        else:
            action = command, target, args_, kwargs_, return_ids
        return action


COMMUNICATION_METHODS = ['get', 'mid_get', 'move', 'remote_get', 'remote_send', 'send', 'share', 'share_']


class TensorCommandMessage(Message):
    """All syft actions use this message type

    In Syft, an action is when one worker wishes to tell another worker to do something with
    objects contained in the worker.object_store registry (or whatever the official object store is
    backed with in the case that it's been overridden). Semantically, one could view all Messages
    as a kind of action, but when we say action this is what we mean. For example, telling a
    worker to take two tensors and add them together is an action. However, sending an object
    from one worker to another is not an action (and would instead use the ObjectMessage type)."""

    def __init__(self, action: Action):
        """Initialize an action message

        Args:
            message (Tuple): this is typically the args and kwargs of a method call on the client,
                but it can be any information necessary to execute the action properly.
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.),
                the id of action results are set by the client. This allows the client to be able
                to predict where the results will be ahead of time. Importantly, this allows the
                client to pre-initalize the pointers to the future data, regardless of whether
                the action has yet executed. It also reduces the size of the response from the
                action (which is very often empty).

        """
        self.action = action

    @property
    def name(self):
        return self.action.name

    @property
    def target(self):
        return self.action.target

    @property
    def args(self):
        return self.action.args

    @property
    def kwargs(self):
        return self.action.kwargs

    @property
    def return_ids(self):
        return self.action.return_ids

    @property
    def return_value(self):
        return self.action.return_value

    def __str__(self):
        """Return a human readable version of this message"""
        return f'({type(self).__name__} {self.action})'

    @staticmethod
    def computation(name, target, args_, kwargs_, return_ids, return_value=False):
        """ Helper function to build a TensorCommandMessage containing a ComputationAction
        directly from the action arguments.
        """
        action = ComputationAction(name, target, args_, kwargs_, return_ids, return_value)
        return TensorCommandMessage(action)

    @staticmethod
    def communication(name, target, args_, kwargs_, return_ids):
        """ Helper function to build a TensorCommandMessage containing a CommunicationAction
        directly from the action arguments.
        """
        action = CommunicationAction(name, target, args_, kwargs_, return_ids)
        return TensorCommandMessage(action)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: 'TensorCommandMessage') ->tuple:
        """
        This function takes the attributes of a TensorCommandMessage and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (TensorCommandMessage): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        return sy.serde.msgpack.serde._simplify(worker, ptr.action),

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) ->'TensorCommandMessage':
        """
        This function takes the simplified tuple version of this message and converts
        it into a TensorCommandMessage. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (TensorCommandMessage): an TensorCommandMessage.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        simplified_action = msg_tuple[0]
        detailed_action = sy.serde.msgpack.serde._detail(worker, simplified_action)
        return TensorCommandMessage(detailed_action)

    @staticmethod
    def bufferize(worker: AbstractWorker, action_message: 'TensorCommandMessage') ->'CommandMessagePB':
        """
        This function takes the attributes of a TensorCommandMessage and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action_message (TensorCommandMessage): an TensorCommandMessage
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_action_msg = CommandMessagePB()
        protobuf_action = sy.serde.protobuf.serde._bufferize(worker, action_message.action)
        if isinstance(action_message.action, ComputationAction):
            protobuf_action_msg.computation.CopyFrom(protobuf_action)
        elif isinstance(action_message.action, CommunicationAction):
            protobuf_action_msg.communication.CopyFrom(protobuf_action)
        return protobuf_action_msg

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj: 'CommandMessagePB') ->'TensorCommandMessage':
        """
        This function takes the Protobuf version of this message and converts
        it into an TensorCommandMessage. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (CommandMessagePB): the Protobuf message

        Returns:
            obj (TensorCommandMessage): an TensorCommandMessage

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        action = getattr(protobuf_obj, protobuf_obj.WhichOneof('action'))
        detailed_action = sy.serde.protobuf.serde._unbufferize(worker, action)
        return TensorCommandMessage(detailed_action)

    @staticmethod
    def get_protobuf_schema():
        """
            Returns the protobuf schema used for TensorCommandMessage.

            Returns:
                Protobuf schema for torch.Size.
        """
        return CommandMessagePB


class RNNCellBase(nn.Module):
    """
    Cell to be used as base for all RNN cells, including GRU and LSTM
    This class overrides the torch.nn.RNNCellBase
    Only Linear and Dropout layers are used to be able to use MPC
    """

    def __init__(self, input_size, hidden_size, bias, num_chunks, nonlinearity=None):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.nonlinearity = nonlinearity
        self.fc_xh = nn.Linear(input_size, self.num_chunks * hidden_size, bias=bias)
        self.fc_hh = nn.Linear(hidden_size, self.num_chunks * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """
        This method initializes or reset all the parameters of the cell.
        The paramaters are initiated following a uniform distribution.
        """
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            init.uniform_(w, -std, std)

    def init_hidden(self, input):
        """
        This method initializes a hidden state when no hidden state is provided
        in the forward method. It creates a hidden state with zero values.
        """
        h = torch.zeros(input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device)
        if input.has_child() and isinstance(input.child, PointerTensor):
            h = h.send(input.child.location)
        if input.has_child() and isinstance(input.child, precision.FixedPrecisionTensor):
            h = h.fix_precision()
            child = input.child
            if isinstance(child.child, AdditiveSharingTensor):
                crypto_provider = child.child.crypto_provider
                owners = child.child.locations
                h = h.share(*owners, crypto_provider=crypto_provider)
        return h


class GRUCell(RNNCellBase):
    """
    Python implementation of GRUCell for MPC
    This class overrides the torch.nn.GRUCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x)
        gate_x = self.fc_xh(x)
        gate_h = self.fc_hh(h)
        x_r, x_z, x_n = gate_x.chunk(self.num_chunks, 1)
        h_r, h_z, h_n = gate_h.chunk(self.num_chunks, 1)
        resetgate = torch.sigmoid(x_r + h_r)
        updategate = torch.sigmoid(x_z + h_z)
        newgate = torch.tanh(x_n + resetgate * h_n)
        h_ = newgate + updategate * (h - newgate)
        return h_


class LSTMCell(RNNCellBase):
    """
    Python implementation of LSTMCell for MPC
    This class overrides the torch.nn.LSTMCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def reset_parameters(self):
        super(LSTMCell, self).reset_parameters()
        incr_bias = 1.0 / self.hidden_size
        init.constant_(self.fc_xh.bias[self.hidden_size:2 * self.hidden_size], incr_bias)
        init.constant_(self.fc_hh.bias[self.hidden_size:2 * self.hidden_size], incr_bias)

    def forward(self, x, hc=None):
        if hc is None:
            hc = self.init_hidden(x), self.init_hidden(x)
        h, c = hc
        gate_x = self.fc_xh(x)
        gate_h = self.fc_hh(h)
        x_i, x_f, x_c, x_o = gate_x.chunk(self.num_chunks, 1)
        h_i, h_f, h_c, h_o = gate_h.chunk(self.num_chunks, 1)
        inputgate = torch.sigmoid(x_i + h_i)
        forgetgate = torch.sigmoid(x_f + h_f)
        cellgate = torch.tanh(x_c + h_c)
        outputgate = torch.sigmoid(x_o + h_o)
        c_ = torch.mul(forgetgate, c) + torch.mul(inputgate, cellgate)
        h_ = torch.mul(outputgate, torch.tanh(c_))
        return h_, c_


class RNNBase(nn.Module):
    """
    Module to be used as base for all RNN modules, including GRU and LSTM
    This class overrides the torch.nn.RNNBase
    Only Linear and Dropout layers are used to be able to use MPC
    """

    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, base_cell, nonlinearity=None):
        super(RNNBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.is_lstm = base_cell is LSTMCell
        self.nonlinearity = nonlinearity
        sizes = [input_size, *(hidden_size for _ in range(self.num_layers - 1))]
        self.rnn_forward = nn.ModuleList(base_cell(sz, hidden_size, bias, nonlinearity) for sz in sizes)
        if self.bidirectional:
            self.rnn_backward = nn.ModuleList(base_cell(sz, hidden_size, bias, nonlinearity) for sz in sizes)

    def forward(self, x, hc=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        if hc is None:
            hc = [self._init_hidden(x) for _ in range(2 if self.is_lstm else 1)]
        else:
            if not self.is_lstm:
                hc = [hc]
            if self.batch_first:
                hc = [item.transpose(0, 1) for item in hc]
        batch_size = x.shape[1]
        seq_len = x.shape[0]
        if self.bidirectional:
            hc = [item.contiguous().view(self.num_layers, 2, batch_size, self.hidden_size) for item in hc]
            hc_fwd = [item[:, (0), :, :] for item in hc]
            hc_back = [item[:, (1), :, :] for item in hc]
        else:
            hc_fwd = hc
        output = x.new(seq_len, batch_size, self.hidden_size).zero_()
        for t in range(seq_len):
            hc_fwd = self._apply_time_step(x, hc_fwd, t)
            output[(t), :, :] = hc_fwd[0][(-1), :, :]
        if self.bidirectional:
            output_back = x.new(seq_len, batch_size, self.hidden_size).zero_()
            for t in range(seq_len - 1, -1, -1):
                hc_back = self._apply_time_step(x, hc_back, t, reverse_direction=True)
                output_back[(t), :, :] = hc_back[0][(-1), :, :]
            output = torch.cat((output, output_back), dim=-1)
            hidden = [torch.cat((hid_item, back_item), dim=0) for hid_item, back_item in zip(hc_fwd, hc_back)]
        else:
            hidden = hc_fwd
        if self.batch_first:
            output = output.transpose(0, 1)
            hidden = [item.transpose(0, 1) for item in hidden]
        hidden = tuple(hidden) if self.is_lstm else hidden[0]
        return output, hidden

    def _init_hidden(self, input):
        """
        This method initializes a hidden state when no hidden state is provided
        in the forward method. It creates a hidden state with zero values for each
        layer of the network.
        """
        h = torch.zeros(self.num_layers * self.num_directions, input.shape[1], self.hidden_size, dtype=input.dtype, device=input.device)
        if input.has_child() and isinstance(input.child, PointerTensor):
            h = h.send(input.child.location)
        if input.has_child() and isinstance(input.child, precision.FixedPrecisionTensor):
            h = h.fix_precision()
            child = input.child
            if isinstance(child.child, AdditiveSharingTensor):
                crypto_provider = child.child.crypto_provider
                owners = child.child.locations
                h = h.share(*owners, crypto_provider=crypto_provider)
        return h

    def _apply_time_step(self, x, hc, t, reverse_direction=False):
        """
        Apply RNN layers at time t, given input and previous hidden states
        """
        rnn_layers = self.rnn_backward if reverse_direction else self.rnn_forward
        hc = torch.stack([*hc])
        hc_next = torch.zeros_like(hc)
        for layer in range(self.num_layers):
            inp = x[(t), :, :] if layer == 0 else hc_next[0][(layer - 1), :, :].clone()
            if self.is_lstm:
                hc_next[:, (layer), :, :] = torch.stack(rnn_layers[layer](inp, hc[:, (layer), :, :]))
            else:
                hc_next[0][(layer), :, :] = rnn_layers[layer](inp, hc[0][(layer), :, :])
        return hc_next


class GRU(RNNBase):
    """
    Python implementation of GRU for MPC
    This class overrides the torch.nn.GRU
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(GRU, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, GRUCell)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Overloads torch.nn.functional.conv2d to be able to use MPC on convolutional networks.
    The idea is to build new tensors from input and weight to compute a
    matrix multiplication equivalent to the convolution.
    Args:
        input: input image
        weight: convolution kernels
        bias: optional additive bias
        stride: stride of the convolution kernels
        padding:  implicit paddings on both sides of the input.
        dilation: spacing between kernel elements
        groups: split input into groups, in_channels should be divisible by the number of groups
    Returns:
        the result of the convolution (FixedPrecision Tensor)
    """
    assert len(input.shape) == 4
    assert len(weight.shape) == 4
    stride = torch.nn.modules.utils._pair(stride)
    padding = torch.nn.modules.utils._pair(padding)
    dilation = torch.nn.modules.utils._pair(dilation)
    batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
    nb_channels_out, nb_channels_kernel, nb_rows_kernel, nb_cols_kernel = weight.shape
    if bias is not None:
        assert len(bias) == nb_channels_out
    assert nb_channels_in == nb_channels_kernel * groups
    assert nb_channels_in % groups == 0
    assert nb_channels_out % groups == 0
    nb_rows_out = int((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0] + 1)
    nb_cols_out = int((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1] + 1)
    if padding != (0, 0):
        padding_mode = 'constant'
        input = torch.nn.functional.pad(input, (padding[1], padding[1], padding[0], padding[0]), padding_mode)
        nb_rows_in += 2 * padding[0]
        nb_cols_in += 2 * padding[1]
    pattern_ind = []
    for ch in range(nb_channels_in):
        for r in range(nb_rows_kernel):
            for c in range(nb_cols_kernel):
                pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
                pattern_ind.append(pixel + ch * nb_rows_in * nb_cols_in)
    im_flat = input.view(batch_size, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
            tmp = [(ind + offset) for ind in pattern_ind]
            im_reshaped.append(im_flat[:, (tmp)])
    im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)
    weight_reshaped = weight.view(nb_channels_out // groups, -1).t()
    if groups > 1:
        res = []
        chunks_im = torch.chunk(im_reshaped, groups, dim=2)
        chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
        for g in range(groups):
            tmp = chunks_im[g].matmul(chunks_weights[g])
            res.append(tmp)
        res = torch.cat(res, dim=2)
    else:
        res = im_reshaped.matmul(weight_reshaped)
    if bias is not None:
        if bias.is_wrapper and res.is_wrapper:
            res += bias
        elif bias.is_wrapper:
            res += bias.child
        else:
            res += bias
    res = res.permute(0, 2, 1).view(batch_size, nb_channels_out, nb_rows_out, nb_cols_out).contiguous()
    return res


class Conv2d(nn.Module):
    """
    This class tries to be an exact python port of the torch.nn.Conv2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.Conv2d and so we must have python ports available for all layer types
    which we seek to use.

    Note: This module is tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    This module has not yet been tested with GPUs but should work out of the box.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        """For information on the constructor arguments, please see PyTorch's
        documentation in torch.nn.Conv2d"""
        super().__init__()
        temp_init = th.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.weight = th.Tensor(temp_init.weight).fix_prec()
        if bias:
            self.bias = th.Tensor(temp_init.bias).fix_prec()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = temp_init.stride
        self.padding = temp_init.padding
        self.dilation = temp_init.dilation
        self.groups = groups
        self.padding_mode = padding_mode

    def forward(self, input):
        assert input.shape[1] == self.in_channels
        return conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AvgPool2d(Module):
    """
    This class is the beginning of an exact python port of the torch.nn.AvgPool2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.AvgPool2d and so we must have python ports available for all layer types
    which we seek to use.

    Note that this module has been tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    However, there is often some rounding error of unknown origin, usually less than
    1e-6 in magnitude.

    This module has not yet been tested with GPUs but should work out of the box.
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        """For information on the constructor arguments, please see PyTorch's
        documentation in torch.nn.AvgPool2d"""
        super().__init__()
        assert padding == 0
        assert ceil_mode is False
        assert count_include_pad is True
        assert divisor_override is None
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self._one_over_kernel_size = 1 / (self.kernel_size * self.kernel_size)

    def forward(self, data):
        batch_size, out_channels, rows, cols = data.shape
        kernel_results = []
        for i in range(0, rows - self.kernel_size + 1, self.stride):
            for j in range(0, cols - self.kernel_size + 1, self.stride):
                kernel_out = data[:, :, i:i + self.kernel_size, j:j + self.kernel_size].sum((2, 3)) * self._one_over_kernel_size
                kernel_results.append(kernel_out.unsqueeze(2))
        pred = th.cat(kernel_results, axis=2).view(batch_size, out_channels, int(rows / self.stride), int(cols / self.stride))
        return pred


class RNNCell(RNNCellBase):
    """
    Python implementation of RNNCell with tanh or relu non-linearity for MPC
    This class overrides the torch.nn.RNNCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = torch.relu
        else:
            raise ValueError(f'Unknown nonlinearity: {nonlinearity}')

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x)
        h_ = self.nonlinearity(self.fc_xh(x) + self.fc_hh(h))
        return h_


class RNN(RNNBase):
    """
    Python implementation of RNN for MPC
    This class overrides the torch.nn.RNN
    """

    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, RNNCell, nonlinearity)


class LSTM(RNNBase):
    """
    Python implementation of LSTM for MPC
    This class overrides the torch.nn.LSTM
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, LSTMCell)


class TopLevelTraceModel(torch.nn.Module):

    def __init__(self):
        super(TopLevelTraceModel, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(3, 1), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        x = x @ self.w1 + self.b1
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AvgPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_OpenMined_PySyft(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

