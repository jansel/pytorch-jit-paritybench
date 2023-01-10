import sys
_module = sys.modules[__name__]
del sys
su3 = _module
eval = _module
neurvps = _module
box = _module
config = _module
datasets = _module
models = _module
conic = _module
deformable = _module
hourglass_pose = _module
vanishing_net = _module
trainer = _module
utils = _module
train = _module

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


import random


import numpy as np


import torch


import matplotlib as mpl


import numpy.linalg as LA


import matplotlib.pyplot as plt


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


from torch import nn


from torch.nn.modules.utils import _pair


import warnings


from torch.autograd import Function


from torch.autograd.function import once_differentiable


import torch.nn as nn


import torch.nn.functional as F


import itertools


from collections import defaultdict


DCN = None


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, bias, stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = DCN.deform_conv_forward(input, weight, bias, offset, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1], ctx.group, ctx.deformable_groups, ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = DCN.deform_conv_backward(input, weight, bias, offset, grad_output, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1], ctx.group, ctx.deformable_groups, ctx.im2col_step)
        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None, None, None


def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, 'cpp')
    tar_dir = os.path.join(src_dir, 'build', ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob(f'{src_dir}/*.cu') + glob(f'{src_dir}/*.cpp')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from torch.utils.cpp_extension import load
        ext = load(name=ext_name, sources=srcs, extra_cflags=['-O3'], extra_cuda_cflags=[], build_directory=tar_dir)
    return ext


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=11, bias=True):
        global DCN
        DCN = load_cpp_ext('DCN')
        super(DeformConv, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            if self.use_bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.bias)

    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        return DeformConvFunction.apply(input.contiguous(), offset.contiguous(), self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups, self.im2col_step)


BOX_PARAMETERS = 'default_box', 'default_box_attr', 'conversion_box', 'frozen_box', 'camel_killer_box', 'box_it_up', 'box_safe_prefix', 'box_duplicates', 'ordered_box'


class BoxError(Exception):
    """Non standard dictionary exceptions"""


class BoxKeyError(BoxError, KeyError, AttributeError):
    """Key does not exist"""


def _from_json(json_string=None, filename=None, encoding='utf-8', errors='strict', multiline=False, **kwargs):
    if filename:
        with open(filename, 'r', encoding=encoding, errors=errors) as f:
            if multiline:
                data = [json.loads(line.strip(), **kwargs) for line in f if line.strip() and not line.strip().startswith('#')]
            else:
                data = json.load(f, **kwargs)
    elif json_string:
        data = json.loads(json_string, **kwargs)
    else:
        raise BoxError('from_json requires a string or filename')
    return data


def _from_yaml(yaml_string=None, filename=None, encoding='utf-8', errors='strict', **kwargs):
    if filename:
        with open(filename, 'r', encoding=encoding, errors=errors) as f:
            data = yaml.load(f, **kwargs)
    elif yaml_string:
        data = yaml.load(yaml_string, **kwargs)
    else:
        raise BoxError('from_yaml requires a string or filename')
    return data


def _to_json(obj, filename=None, encoding='utf-8', errors='strict', **json_kwargs):
    json_dump = json.dumps(obj, ensure_ascii=False, **json_kwargs)
    if filename:
        with open(filename, 'w', encoding=encoding, errors=errors) as f:
            f.write(json_dump if sys.version_info >= (3, 0) else json_dump.decode('utf-8'))
    else:
        return json_dump


def _to_yaml(obj, filename=None, default_flow_style=False, encoding='utf-8', errors='strict', **yaml_kwargs):
    if filename:
        with open(filename, 'w', encoding=encoding, errors=errors) as f:
            yaml.dump(obj, stream=f, default_flow_style=default_flow_style, **yaml_kwargs)
    else:
        return yaml.dump(obj, default_flow_style=default_flow_style, **yaml_kwargs)


yaml_support = True


_all_cap_re = re.compile('([a-z0-9])([A-Z])')


_first_cap_re = re.compile('(.)([A-Z][a-z]+)')


def _camel_killer(attr):
    """
    CamelKiller, qu'est-ce que c'est?

    Taken from http://stackoverflow.com/a/1176023/3244542
    """
    try:
        attr = str(attr)
    except UnicodeEncodeError:
        attr = attr.encode('utf-8', 'ignore')
    s1 = _first_cap_re.sub('\\1_\\2', attr)
    s2 = _all_cap_re.sub('\\1_\\2', s1)
    return re.sub('_+', '_', s2.casefold() if hasattr(s2, 'casefold') else s2.lower())


def _safe_key(key):
    try:
        return str(key)
    except UnicodeEncodeError:
        return key.encode('utf-8', 'ignore')


def _safe_attr(attr, camel_killer=False, replacement_char='x'):
    """Convert a key into something that is accessible as an attribute"""
    allowed = string.ascii_letters + string.digits + '_'
    attr = _safe_key(attr)
    if camel_killer:
        attr = _camel_killer(attr)
    attr = attr.replace(' ', '_')
    out = ''
    for character in attr:
        out += character if character in allowed else '_'
    out = out.strip('_')
    try:
        int(out[0])
    except (ValueError, IndexError):
        pass
    else:
        out = '{0}{1}'.format(replacement_char, out)
    if out in kwlist:
        out = '{0}{1}'.format(replacement_char, out)
    return re.sub('_+', '_', out)


def _conversion_checks(item, keys, box_config, check_only=False, pre_check=False):
    """
    Internal use for checking if a duplicate safe attribute already exists

    :param item: Item to see if a dup exists
    :param keys: Keys to check against
    :param box_config: Easier to pass in than ask for specfic items
    :param check_only: Don't bother doing the conversion work
    :param pre_check: Need to add the item to the list of keys to check
    :return: the original unmodified key, if exists and not check_only
    """
    if box_config['box_duplicates'] != 'ignore':
        if pre_check:
            keys = list(keys) + [item]
        key_list = [(k, _safe_attr(k, camel_killer=box_config['camel_killer_box'], replacement_char=box_config['box_safe_prefix'])) for k in keys]
        if len(key_list) > len(set(x[1] for x in key_list)):
            seen = set()
            dups = set()
            for x in key_list:
                if x[1] in seen:
                    dups.add('{0}({1})'.format(x[0], x[1]))
                seen.add(x[1])
            if box_config['box_duplicates'].startswith('warn'):
                warnings.warn('Duplicate conversion attributes exist: {0}'.format(dups))
            else:
                raise BoxError('Duplicate conversion attributes exist: {0}'.format(dups))
    if check_only:
        return
    for k in keys:
        if item == _safe_attr(k, camel_killer=box_config['camel_killer_box'], replacement_char=box_config['box_safe_prefix']):
            return k


def _get_box_config(cls, kwargs):
    return {'__converted': set(), '__box_heritage': kwargs.pop('__box_heritage', None), '__created': False, '__ordered_box_values': [], 'default_box': kwargs.pop('default_box', False), 'default_box_attr': kwargs.pop('default_box_attr', cls), 'conversion_box': kwargs.pop('conversion_box', True), 'box_safe_prefix': kwargs.pop('box_safe_prefix', 'x'), 'frozen_box': kwargs.pop('frozen_box', False), 'camel_killer_box': kwargs.pop('camel_killer_box', False), 'modify_tuples_box': kwargs.pop('modify_tuples_box', False), 'box_duplicates': kwargs.pop('box_duplicates', 'ignore'), 'ordered_box': kwargs.pop('ordered_box', False)}


def _recursive_tuples(iterable, box_class, recreate_tuples=False, **kwargs):
    out_list = []
    for i in iterable:
        if isinstance(i, dict):
            out_list.append(box_class(i, **kwargs))
        elif isinstance(i, list) or recreate_tuples and isinstance(i, tuple):
            out_list.append(_recursive_tuples(i, box_class, recreate_tuples, **kwargs))
        else:
            out_list.append(i)
    return tuple(out_list)


class Box(dict):
    """
    Improved dictionary access through dot notation with additional tools.

    :param default_box: Similar to defaultdict, return a default value
    :param default_box_attr: Specify the default replacement.
        WARNING: If this is not the default 'Box', it will not be recursive
    :param frozen_box: After creation, the box cannot be modified
    :param camel_killer_box: Convert CamelCase to snake_case
    :param conversion_box: Check for near matching keys as attributes
    :param modify_tuples_box: Recreate incoming tuples with dicts into Boxes
    :param box_it_up: Recursively create all Boxes from the start
    :param box_safe_prefix: Conversion box prefix for unsafe attributes
    :param box_duplicates: "ignore", "error" or "warn" when duplicates exists
        in a conversion_box
    :param ordered_box: Preserve the order of keys entered into the box
    """
    _protected_keys = dir({}) + ['to_dict', 'tree_view', 'to_json', 'to_yaml', 'from_yaml', 'from_json']

    def __new__(cls, *args, **kwargs):
        """
        Due to the way pickling works in python 3, we need to make sure
        the box config is created as early as possible.
        """
        obj = super(Box, cls).__new__(cls, *args, **kwargs)
        obj._box_config = _get_box_config(cls, kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        self._box_config = _get_box_config(self.__class__, kwargs)
        if self._box_config['ordered_box']:
            self._box_config['__ordered_box_values'] = []
        if not self._box_config['conversion_box'] and self._box_config['box_duplicates'] != 'ignore':
            raise BoxError('box_duplicates are only for conversion_boxes')
        if len(args) == 1:
            if isinstance(args[0], basestring):
                raise ValueError('Cannot extrapolate Box from string')
            if isinstance(args[0], Mapping):
                for k, v in args[0].items():
                    if v is args[0]:
                        v = self
                    self[k] = v
                    self.__add_ordered(k)
            elif isinstance(args[0], Iterable):
                for k, v in args[0]:
                    self[k] = v
                    self.__add_ordered(k)
            else:
                raise ValueError('First argument must be mapping or iterable')
        elif args:
            raise TypeError('Box expected at most 1 argument, got {0}'.format(len(args)))
        box_it = kwargs.pop('box_it_up', False)
        for k, v in kwargs.items():
            if args and isinstance(args[0], Mapping) and v is args[0]:
                v = self
            self[k] = v
            self.__add_ordered(k)
        if self._box_config['frozen_box'] or box_it or self._box_config['box_duplicates'] != 'ignore':
            self.box_it_up()
        self._box_config['__created'] = True

    def __add_ordered(self, key):
        if self._box_config['ordered_box'] and key not in self._box_config['__ordered_box_values']:
            self._box_config['__ordered_box_values'].append(key)

    def box_it_up(self):
        """
        Perform value lookup for all items in current dictionary,
        generating all sub Box objects, while also running `box_it_up` on
        any of those sub box objects.
        """
        for k in self:
            _conversion_checks(k, self.keys(), self._box_config, check_only=True)
            if self[k] is not self and hasattr(self[k], 'box_it_up'):
                self[k].box_it_up()

    def __hash__(self):
        if self._box_config['frozen_box']:
            hashing = 54321
            for item in self.items():
                hashing ^= hash(item)
            return hashing
        raise TypeError("unhashable type: 'Box'")

    def __dir__(self):
        allowed = string.ascii_letters + string.digits + '_'
        kill_camel = self._box_config['camel_killer_box']
        items = set(dir(dict) + ['to_dict', 'to_json', 'from_json', 'box_it_up'])
        for key in self.keys():
            key = _safe_key(key)
            if ' ' not in key and key[0] not in string.digits and key not in kwlist:
                for letter in key:
                    if letter not in allowed:
                        break
                else:
                    items.add(key)
        for key in self.keys():
            key = _safe_key(key)
            if key not in items:
                if self._box_config['conversion_box']:
                    key = _safe_attr(key, camel_killer=kill_camel, replacement_char=self._box_config['box_safe_prefix'])
                    if key:
                        items.add(key)
            if kill_camel:
                snake_key = _camel_killer(key)
                if snake_key:
                    items.remove(key)
                    items.add(snake_key)
        if yaml_support:
            items.add('to_yaml')
            items.add('from_yaml')
        return list(items)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            if isinstance(default, dict) and not isinstance(default, Box):
                return Box(default)
            if isinstance(default, list) and not isinstance(default, BoxList):
                return BoxList(default)
            return default

    def copy(self):
        return self.__class__(super(self.__class__, self).copy())

    def __copy__(self):
        return self.__class__(super(self.__class__, self).copy())

    def __deepcopy__(self, memodict=None):
        out = self.__class__()
        memodict = memodict or {}
        memodict[id(self)] = out
        for k, v in self.items():
            out[copy.deepcopy(k, memodict)] = copy.deepcopy(v, memodict)
        return out

    def __setstate__(self, state):
        self._box_config = state['_box_config']
        self.__dict__.update(state)

    def __getitem__(self, item, _ignore_default=False):
        try:
            value = super(Box, self).__getitem__(item)
        except KeyError as err:
            if item == '_box_config':
                raise BoxKeyError('_box_config should only exist as an attribute and is never defaulted')
            if self._box_config['default_box'] and not _ignore_default:
                return self.__get_default(item)
            raise BoxKeyError(str(err))
        else:
            return self.__convert_and_store(item, value)

    def keys(self):
        if self._box_config['ordered_box']:
            return self._box_config['__ordered_box_values']
        return super(Box, self).keys()

    def values(self):
        return [self[x] for x in self.keys()]

    def items(self):
        return [(x, self[x]) for x in self.keys()]

    def __get_default(self, item):
        default_value = self._box_config['default_box_attr']
        if default_value is self.__class__:
            return self.__class__(__box_heritage=(self, item), **self.__box_config())
        elif isinstance(default_value, Callable):
            return default_value()
        elif hasattr(default_value, 'copy'):
            return default_value.copy()
        return default_value

    def __box_config(self):
        out = {}
        for k, v in self._box_config.copy().items():
            if not k.startswith('__'):
                out[k] = v
        return out

    def __convert_and_store(self, item, value):
        if item in self._box_config['__converted']:
            return value
        if isinstance(value, dict) and not isinstance(value, Box):
            value = self.__class__(value, __box_heritage=(self, item), **self.__box_config())
            self[item] = value
        elif isinstance(value, list) and not isinstance(value, BoxList):
            if self._box_config['frozen_box']:
                value = _recursive_tuples(value, self.__class__, recreate_tuples=self._box_config['modify_tuples_box'], __box_heritage=(self, item), **self.__box_config())
            else:
                value = BoxList(value, __box_heritage=(self, item), box_class=self.__class__, **self.__box_config())
            self[item] = value
        elif self._box_config['modify_tuples_box'] and isinstance(value, tuple):
            value = _recursive_tuples(value, self.__class__, recreate_tuples=True, __box_heritage=(self, item), **self.__box_config())
            self[item] = value
        self._box_config['__converted'].add(item)
        return value

    def __create_lineage(self):
        if self._box_config['__box_heritage'] and self._box_config['__created']:
            past, item = self._box_config['__box_heritage']
            if not past[item]:
                past[item] = self
            self._box_config['__box_heritage'] = None

    def __getattr__(self, item):
        try:
            try:
                value = self.__getitem__(item, _ignore_default=True)
            except KeyError:
                value = object.__getattribute__(self, item)
        except AttributeError as err:
            if item == '__getstate__':
                raise AttributeError(item)
            if item == '_box_config':
                raise BoxError('_box_config key must exist')
            kill_camel = self._box_config['camel_killer_box']
            if self._box_config['conversion_box'] and item:
                k = _conversion_checks(item, self.keys(), self._box_config)
                if k:
                    return self.__getitem__(k)
            if kill_camel:
                for k in self.keys():
                    if item == _camel_killer(k):
                        return self.__getitem__(k)
            if self._box_config['default_box']:
                return self.__get_default(item)
            raise BoxKeyError(str(err))
        else:
            if item == '_box_config':
                return value
            return self.__convert_and_store(item, value)

    def __setitem__(self, key, value):
        if key != '_box_config' and self._box_config['__created'] and self._box_config['frozen_box']:
            raise BoxError('Box is frozen')
        if self._box_config['conversion_box']:
            _conversion_checks(key, self.keys(), self._box_config, check_only=True, pre_check=True)
        super(Box, self).__setitem__(key, value)
        self.__add_ordered(key)
        self.__create_lineage()

    def __setattr__(self, key, value):
        if key != '_box_config' and self._box_config['frozen_box'] and self._box_config['__created']:
            raise BoxError('Box is frozen')
        if key in self._protected_keys:
            raise AttributeError("Key name '{0}' is protected".format(key))
        if key == '_box_config':
            return object.__setattr__(self, key, value)
        try:
            object.__getattribute__(self, key)
        except (AttributeError, UnicodeEncodeError):
            if key not in self.keys() and (self._box_config['conversion_box'] or self._box_config['camel_killer_box']):
                if self._box_config['conversion_box']:
                    k = _conversion_checks(key, self.keys(), self._box_config)
                    self[key if not k else k] = value
                elif self._box_config['camel_killer_box']:
                    for each_key in self:
                        if key == _camel_killer(each_key):
                            self[each_key] = value
                            break
            else:
                self[key] = value
        else:
            object.__setattr__(self, key, value)
        self.__add_ordered(key)
        self.__create_lineage()

    def __delitem__(self, key):
        if self._box_config['frozen_box']:
            raise BoxError('Box is frozen')
        super(Box, self).__delitem__(key)
        if self._box_config['ordered_box'] and key in self._box_config['__ordered_box_values']:
            self._box_config['__ordered_box_values'].remove(key)

    def __delattr__(self, item):
        if self._box_config['frozen_box']:
            raise BoxError('Box is frozen')
        if item == '_box_config':
            raise BoxError('"_box_config" is protected')
        if item in self._protected_keys:
            raise AttributeError("Key name '{0}' is protected".format(item))
        try:
            object.__getattribute__(self, item)
        except AttributeError:
            del self[item]
        else:
            object.__delattr__(self, item)
        if self._box_config['ordered_box'] and item in self._box_config['__ordered_box_values']:
            self._box_config['__ordered_box_values'].remove(item)

    def pop(self, key, *args):
        if args:
            if len(args) != 1:
                raise BoxError('pop() takes only one optional argument "default"')
            try:
                item = self[key]
            except KeyError:
                return args[0]
            else:
                del self[key]
                return item
        try:
            item = self[key]
        except KeyError:
            raise BoxKeyError('{0}'.format(key))
        else:
            del self[key]
            return item

    def clear(self):
        self._box_config['__ordered_box_values'] = []
        super(Box, self).clear()

    def popitem(self):
        try:
            key = next(self.__iter__())
        except StopIteration:
            raise BoxKeyError('Empty box')
        return key, self.pop(key)

    def __repr__(self):
        return '<Box: {0}>'.format(str(self.to_dict()))

    def __str__(self):
        return str(self.to_dict())

    def __iter__(self):
        for key in self.keys():
            yield key

    def __reversed__(self):
        for key in reversed(list(self.keys())):
            yield key

    def to_dict(self):
        """
        Turn the Box and sub Boxes back into a native
        python dictionary.

        :return: python dictionary of this Box
        """
        out_dict = dict(self)
        for k, v in out_dict.items():
            if v is self:
                out_dict[k] = out_dict
            elif hasattr(v, 'to_dict'):
                out_dict[k] = v.to_dict()
            elif hasattr(v, 'to_list'):
                out_dict[k] = v.to_list()
        return out_dict

    def update(self, item=None, **kwargs):
        if not item:
            item = kwargs
        iter_over = item.items() if hasattr(item, 'items') else item
        for k, v in iter_over:
            if isinstance(v, dict):
                v = self.__class__(v)
                if k in self and isinstance(self[k], dict):
                    self[k].update(v)
                    continue
            if isinstance(v, list):
                v = BoxList(v)
            try:
                self.__setattr__(k, v)
            except (AttributeError, TypeError):
                self.__setitem__(k, v)

    def setdefault(self, item, default=None):
        if item in self:
            return self[item]
        if isinstance(default, dict):
            default = self.__class__(default)
        if isinstance(default, list):
            default = BoxList(default)
        self[item] = default
        return default

    def to_json(self, filename=None, encoding='utf-8', errors='strict', **json_kwargs):
        """
        Transform the Box object into a JSON string.

        :param filename: If provided will save to file
        :param encoding: File encoding
        :param errors: How to handle encoding errors
        :param json_kwargs: additional arguments to pass to json.dump(s)
        :return: string of JSON or return of `json.dump`
        """
        return _to_json(self.to_dict(), filename=filename, encoding=encoding, errors=errors, **json_kwargs)

    @classmethod
    def from_json(cls, json_string=None, filename=None, encoding='utf-8', errors='strict', **kwargs):
        """
        Transform a json object string into a Box object. If the incoming
        json is a list, you must use BoxList.from_json.

        :param json_string: string to pass to `json.loads`
        :param filename: filename to open and pass to `json.load`
        :param encoding: File encoding
        :param errors: How to handle encoding errors
        :param kwargs: parameters to pass to `Box()` or `json.loads`
        :return: Box object from json data
        """
        bx_args = {}
        for arg in kwargs.copy():
            if arg in BOX_PARAMETERS:
                bx_args[arg] = kwargs.pop(arg)
        data = _from_json(json_string, filename=filename, encoding=encoding, errors=errors, **kwargs)
        if not isinstance(data, dict):
            raise BoxError('json data not returned as a dictionary, but rather a {0}'.format(type(data).__name__))
        return cls(data, **bx_args)
    if yaml_support:

        def to_yaml(self, filename=None, default_flow_style=False, encoding='utf-8', errors='strict', **yaml_kwargs):
            """
            Transform the Box object into a YAML string.

            :param filename:  If provided will save to file
            :param default_flow_style: False will recursively dump dicts
            :param encoding: File encoding
            :param errors: How to handle encoding errors
            :param yaml_kwargs: additional arguments to pass to yaml.dump
            :return: string of YAML or return of `yaml.dump`
            """
            return _to_yaml(self.to_dict(), filename=filename, default_flow_style=default_flow_style, encoding=encoding, errors=errors, **yaml_kwargs)

        @classmethod
        def from_yaml(cls, yaml_string=None, filename=None, encoding='utf-8', errors='strict', loader=yaml.SafeLoader, **kwargs):
            """
            Transform a yaml object string into a Box object.

            :param yaml_string: string to pass to `yaml.load`
            :param filename: filename to open and pass to `yaml.load`
            :param encoding: File encoding
            :param errors: How to handle encoding errors
            :param loader: YAML Loader, defaults to SafeLoader
            :param kwargs: parameters to pass to `Box()` or `yaml.load`
            :return: Box object from yaml data
            """
            bx_args = {}
            for arg in kwargs.copy():
                if arg in BOX_PARAMETERS:
                    bx_args[arg] = kwargs.pop(arg)
            data = _from_yaml(yaml_string=yaml_string, filename=filename, encoding=encoding, errors=errors, Loader=loader, **kwargs)
            if not isinstance(data, dict):
                raise BoxError('yaml data not returned as a dictionarybut rather a {0}'.format(type(data).__name__))
            return cls(data, **bx_args)


M = Box()


class ConicConv(nn.Module):

    def __init__(self, c_in, c_out, kernel_size=3, bias=False):
        super().__init__()
        self.deform_conv = DeformConv(c_in, c_out, kernel_size=kernel_size, stride=1, padding=1, im2col_step=M.im2col_step, bias=bias)
        self.kernel_size = _pair(kernel_size)

    def forward(self, input, vpts):
        N, C, H, W = input.shape
        Kh, Kw = self.kernel_size
        with torch.no_grad():
            ys, xs = torch.meshgrid(torch.arange(0, H).float(), torch.arange(0, W).float())
            d = torch.cat([(vpts[:, 0, None, None] - ys)[..., None], (vpts[:, 1, None, None] - xs)[..., None]], dim=-1)
            d /= torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-05)
            n = torch.cat([-d[..., 1:2], d[..., 0:1]], dim=-1)
            offset = torch.zeros((N, H, W, Kh, Kw, 2))
            for i in range(Kh):
                for j in range(Kw):
                    offset[..., i, j, :] = d * (1 - i) + n * (1 - j)
                    offset[..., i, j, 0] += 1 - i
                    offset[..., i, j, 1] += 1 - j
            offset = offset.permute(0, 3, 4, 5, 1, 2).reshape((N, -1, H, W))
        return self.deform_conv(input, offset)


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, resample=None):
        super(Bottleneck2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.resample = resample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.resample is not None:
            residual = self.resample(x)
        out += residual
        return out


class Hourglass(nn.Module):

    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)
        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):

    def __init__(self, planes, block, head, depth, num_stacks, num_blocks):
        super(HourglassNet, self).__init__()
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, planes))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(planes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        resample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            resample = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride)
        layers = [block(self.inplanes, planes, stride, resample)]
        self.inplanes = planes * block.expansion
        for i in range(blocks - 1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1), nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out[::-1]


class ApolloniusNet(nn.Module):

    def __init__(self, output_stride, upsample_scale):
        super().__init__()
        self.fc0 = nn.Conv2d(64, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        if M.conic_6x:
            self.bn00 = nn.BatchNorm2d(32)
            self.conv00 = ConicConv(32, 32)
            self.bn0 = nn.BatchNorm2d(32)
            self.conv0 = ConicConv(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = ConicConv(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = ConicConv(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = ConicConv(128, 256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = ConicConv(256, 256)
        self.fc1 = nn.Linear(16384, M.fc_channel)
        self.fc2 = nn.Linear(M.fc_channel, M.fc_channel)
        self.fc3 = nn.Linear(M.fc_channel, len(M.multires))
        self.upsample_scale = upsample_scale
        self.stride = output_stride / upsample_scale

    def forward(self, input, vpts):
        if self.upsample_scale != 1:
            input = F.interpolate(input, scale_factor=self.upsample_scale)
        x = self.fc0(input)
        if M.conic_6x:
            x = self.bn00(x)
            x = self.relu(x)
            x = self.conv00(x, vpts / self.stride - 0.5)
            x = self.bn0(x)
            x = self.relu(x)
            x = self.conv0(x, vpts / self.stride - 0.5)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x, vpts / self.stride - 0.5)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x, vpts / self.stride / 2 - 0.5)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x, vpts / self.stride / 4 - 0.5)
        x = self.pool(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4(x, vpts / self.stride / 8 - 0.5)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


C = Box()


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


def sample_sphere(v, theta0, theta1):
    costheta = random.uniform(math.cos(theta1), math.cos(theta0))
    phi = random.random() * math.pi * 2
    v1 = orth(v)
    v2 = np.cross(v, v1)
    r = math.sqrt(1 - costheta ** 2)
    w = v * costheta + r * (v1 * math.cos(phi) + v2 * math.sin(phi))
    return w / LA.norm(w)


def to_label(w, vpts):
    degree = np.min(np.arccos(np.abs(vpts @ w).clip(max=1)))
    return [int(degree < res + 1e-06) for res in M.multires]


def to_pixel(w):
    x = w[0] / w[2] * C.io.focal_length * 256 + 256
    y = -w[1] / w[2] * C.io.focal_length * 256 + 256
    return y, x


class VanishingNet(nn.Module):

    def __init__(self, backbone, output_stride=4, upsample_scale=1):
        super().__init__()
        self.backbone = backbone
        self.anet = ApolloniusNet(output_stride, upsample_scale)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input_dict):
        x = self.backbone(input_dict['image'])[0]
        N, _, H, W = x.shape
        test = input_dict.get('test', False)
        if test:
            c = len(input_dict['vpts'])
        else:
            c = M.smp_rnd + C.io.num_vpts * len(M.multires) * (M.smp_pos + M.smp_neg)
        x = x[:, None].repeat(1, c, 1, 1, 1).reshape(N * c, _, H, W)
        if test:
            vpts = [to_pixel(v) for v in input_dict['vpts']]
            vpts = torch.tensor(vpts, device=x.device)
            return self.anet(x, vpts).sigmoid()
        vpts_gt = input_dict['vpts'].cpu().numpy()
        vpts, y = [], []
        for n in range(N):

            def add_sample(p):
                vpts.append(to_pixel(p))
                y.append(to_label(p, vpts_gt[n]))
            for vgt in vpts_gt[n]:
                for st, ed in zip([0] + M.multires[:-1], M.multires):
                    for _ in range(M.smp_pos):
                        add_sample(sample_sphere(vgt, st, ed))
                    for _ in range(M.smp_neg):
                        add_sample(sample_sphere(vgt, ed, ed * M.smp_multiplier))
            for _ in range(M.smp_rnd):
                add_sample(sample_sphere(np.array([0, 0, 1]), 0, math.pi / 2))
        y = torch.tensor(y, device=x.device, dtype=torch.float)
        vpts = torch.tensor(vpts, device=x.device)
        x = self.anet(x, vpts)
        L = self.loss(x, y)
        maskn = (y == 0).float()
        maskp = (y == 1).float()
        losses = {}
        for i in range(len(M.multires)):
            assert maskn[:, i].sum().item() != 0
            assert maskp[:, i].sum().item() != 0
            losses[f'lneg{i}'] = (L[:, i] * maskn[:, i]).sum() / maskn[:, i].sum()
            losses[f'lpos{i}'] = (L[:, i] * maskp[:, i]).sum() / maskp[:, i].sum()
        return {'losses': [losses], 'preds': {'vpts': vpts, 'scores': x.sigmoid(), 'ys': y}}

