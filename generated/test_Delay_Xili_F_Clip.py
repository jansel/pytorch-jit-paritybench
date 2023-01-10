import sys
_module = sys.modules[__name__]
del sys
FClip = _module
box = _module
config = _module
datasets = _module
line_parsing = _module
losses = _module
lr_schedulers = _module
metric = _module
models = _module
hourglass_line = _module
hourglass_pose = _module
pose_hrnet = _module
stage_1 = _module
nms = _module
postprocess = _module
trainer = _module
utils = _module
visualize = _module
crop = _module
input_parsing = _module
resolution = _module
wireframe = _module
wireframe_line = _module
york = _module
york_line = _module
demo = _module
test = _module
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


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


import torch.nn.functional as F


import math


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn as nn


import time


import logging


from collections import OrderedDict


from collections import defaultdict


import matplotlib as mpl


import matplotlib.pyplot as plt


import random


import warnings


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
        :return: Box object from json Good
        """
        bx_args = {}
        for arg in kwargs.copy():
            if arg in BOX_PARAMETERS:
                bx_args[arg] = kwargs.pop(arg)
        data = _from_json(json_string, filename=filename, encoding=encoding, errors=errors, **kwargs)
        if not isinstance(data, dict):
            raise BoxError('json Good not returned as a dictionary, but rather a {0}'.format(type(data).__name__))
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
            :return: Box object from yaml Good
            """
            bx_args = {}
            for arg in kwargs.copy():
                if arg in BOX_PARAMETERS:
                    bx_args[arg] = kwargs.pop(arg)
            data = _from_yaml(yaml_string=yaml_string, filename=filename, encoding=encoding, errors=errors, Loader=loader, **kwargs)
            if not isinstance(data, dict):
                raise BoxError('yaml Good not returned as a dictionarybut rather a {0}'.format(type(data).__name__))
            return cls(data, **bx_args)


M = Box()


class LineHead(nn.Module):

    def __init__(self, input_channels, m, output_channels):
        super(LineHead, self).__init__()
        ks = M.line_kernel
        self.branch1 = nn.Sequential(nn.Conv2d(input_channels, m, kernel_size=(1, ks), padding=(0, int(ks / 2))), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=(1, ks), padding=(0, int(ks / 2))))
        self.branch2 = nn.Sequential(nn.Conv2d(input_channels, m, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=1))
        self.branch3 = nn.Sequential(nn.Conv2d(input_channels, m, kernel_size=(ks, 1), padding=(int(ks / 2), 0)), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=(ks, 1), padding=(int(ks / 2), 0)))
        self.merge = nn.Conv2d(int(3 * output_channels), output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = torch.cat([x1, x2, x3], dim=1)
        return self.merge(x4)


class LCNNHead(nn.Module):

    def __init__(self, input_channels, num_class, head_size=[[2], [2], [1]]):
        super(LCNNHead, self).__init__()
        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(head_size, []):
            heads.append(nn.Sequential(nn.Conv2d(input_channels, m, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=1)))
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskHead(nn.Module):

    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()
        m = int(input_channels / 4)
        heads = []
        heads_size = sum(self._get_head_size(), [])
        heads_net = M.head_net
        for k, (output_channels, net) in enumerate(zip(heads_size, heads_net)):
            if net == 'raw':
                heads.append(nn.Sequential(nn.Conv2d(input_channels, m, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=1)))
                None
            elif net == 'raw_upsampler':
                heads.append(nn.Sequential(nn.UpsamplingBilinear2d(size=(M.resolution, M.resolution)), nn.Conv2d(input_channels, m, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=1)))
                None
            elif net == 'mask':
                heads.append(nn.Sequential(nn.Conv2d(input_channels, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, m, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=1)))
                None
            elif net == 'line':
                heads.append(LineHead(input_channels, m, output_channels))
                None
            else:
                raise NotImplementedError
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(self._get_head_size(), []))

    @staticmethod
    def _get_head_size():
        M_dic = M.to_dict()
        head_size = []
        for h in M_dic['head']['order']:
            head_size.append([M_dic['head'][h]['head_size']])
        return head_size

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class BottleneckLine(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckLine, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2, self.merge = self.build_line_layers(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def build_line_layers(self, planes):
        layer = []
        if 's' in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))
        if 'v' in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=(M.line_kernel, 1), padding=(int(M.line_kernel / 2), 0)))
        if 'h' in M.line.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=(1, M.line_kernel), padding=(0, int(M.line_kernel / 2))))
        assert len(layer) > 0
        if M.merge == 'cat':
            merge = nn.Conv2d(planes * len(layer), planes, kernel_size=1)
        elif M.merge == 'maxpool':
            ll = len(M.line.mode)
            merge = nn.MaxPool3d((ll, 1, 1), stride=(ll, 1, 1))
        else:
            raise ValueError()
        return nn.ModuleList(layer), merge

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        if M.merge == 'cat':
            tt = torch.cat([conv(out) for conv in self.conv2], dim=1)
        elif M.merge == 'maxpool':
            tt = torch.cat([torch.unsqueeze(conv(out), 2) for conv in self.conv2], dim=2)
        else:
            raise ValueError()
        out = self.merge(tt)
        out = torch.squeeze(out, 2)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck1D_v(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D_v, self).__init__()
        self.ks = M.line_kernel, 1
        self.padding = int(M.line_kernel / 2), 0
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=self.ks, stride=stride, padding=self.padding)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck1D_h(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D_h, self).__init__()
        self.ks = 1, M.line_kernel
        self.padding = 0, int(M.line_kernel / 2)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=self.ks, stride=stride, padding=self.padding)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
        if self.downsample is not None:
            residual = self.downsample(x)
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
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block, head, depth, num_stacks, num_blocks, num_classes):
        super(HourglassNet, self).__init__()
        self.inplanes = M.inplanes
        self.num_feats = self.inplanes * block.expansion
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
            score.append(head(ch, num_classes))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        extra_info = {'time_front': 0.0, 'time_stack0': 0.0, 'time_stack1': 0.0}
        t = time.time()
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        extra_info['time_front'] = time.time() - t
        for i in range(self.num_stacks):
            t = time.time()
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
            extra_info[f'time_stack{i}'] = time.time() - t
        return out[::-1], y, extra_info


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


logger = logging.getLogger(__name__)


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i]), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3), nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=False)
        self.final_layer = kwargs['head'](pre_stage_channels[0], kwargs['num_classes'])

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        extra_info = {'time_front': 0.0, 'time_stack0': 0.0, 'time_stack1': 0.0}
        t = time.time()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        extra_info['time_front'] = time.time() - t
        t = time.time()
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        extra_info['time_stack0'] = time.time() - t
        t = time.time()
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = self.final_layer(y_list[0])
        extra_info['time_stack1'] = time.time() - t
        return [x], y_list[0], extra_info


def non_maximum_suppression(a, delta=0.0, kernel=3):
    ap = F.max_pool2d(a, kernel, stride=1, padding=int((kernel - 1) / 2))
    mask = (a == ap).float().clamp(min=0.0)
    mask_n = (~mask.bool()).float() * delta
    return a * mask + a * mask_n


class PointParsing:

    @staticmethod
    def jheatmap_torch(jmap, joff, delta=0.8, K=1000, kernel=3, joff_type='raw', resolution=128):
        h, w = jmap.shape
        lcmap = non_maximum_suppression(jmap[None, ...], delta, kernel).reshape(-1)
        score, index = torch.topk(lcmap, k=int(K))
        if joff is not None:
            lcoff = joff.reshape(2, -1)
            if joff_type == 'raw':
                y = (index // w).float() + lcoff[0][index] + 0.5
                x = (index % w).float() + lcoff[1][index] + 0.5
            elif joff_type == 'gaussian':
                y = (index // w).float() + lcoff[0][index]
                x = (index % w).float() + lcoff[1][index]
            else:
                raise NotImplementedError
        else:
            y = (index // w).float()
            x = (index % w).float()
        yx = torch.cat([y[..., None], x[..., None]], dim=-1).clamp(0, resolution - 1e-06)
        return yx, score, index

    @staticmethod
    def jheatmap_numpy(jmap, joff, delta=0.8, K=1000, kernel=3, resolution=128):
        jmap = torch.from_numpy(jmap)
        if joff is not None:
            joff = torch.from_numpy(joff)
        xy, score, index = PointParsing.jheatmap_torch(jmap, joff, delta, K, kernel, resolution=resolution)
        v = torch.cat([xy, score[:, None]], 1)
        return v.numpy()


class OneStageLineParsing:

    @staticmethod
    def fclip_numpy(lcmap, lcoff, lleng, angle, delta=0.8, nlines=1000, ang_type='radian', kernel=3, resolution=128):
        lcmap = torch.from_numpy(lcmap)
        lcoff = torch.from_numpy(lcoff)
        lleng = torch.from_numpy(lleng)
        angle = torch.from_numpy(angle)
        lines, scores = OneStageLineParsing.fclip_torch(lcmap, lcoff, lleng, angle, delta, nlines, ang_type, kernel, resolution=resolution)
        return lines.numpy(), scores.numpy()

    @staticmethod
    def fclip_torch(lcmap, lcoff, lleng, angle, delta=0.8, nlines=1000, ang_type='radian', kernel=3, resolution=128):
        xy, score, index = PointParsing.jheatmap_torch(lcmap, lcoff, delta, nlines, kernel, resolution=resolution)
        lines = OneStageLineParsing.fclip_merge(xy, index, lleng, angle, ang_type, resolution=resolution)
        return lines, score

    @staticmethod
    def fclip_merge(xy, xy_idx, length_regress, angle_regress, ang_type='radian', resolution=128):
        """
        :param xy: (K, 2)
        :param xy_idx: (K,)
        :param length_regress: (H, W)
        :param angle_regress:  (H, W)
        :param ang_type
        :param resolution
        :return:
        """
        xy_idx = xy_idx.reshape(-1)
        lleng_regress = length_regress.reshape(-1)[xy_idx]
        angle_regress = angle_regress.reshape(-1)[xy_idx]
        lengths = lleng_regress * (resolution / 2)
        if ang_type == 'cosine':
            angles = angle_regress * 2 - 1
        elif ang_type == 'radian':
            angles = torch.cos(angle_regress * np.pi)
        else:
            raise NotImplementedError
        angles1 = -torch.sqrt(1 - angles ** 2)
        direction = torch.cat([angles1[:, None], angles[:, None]], 1)
        v1 = (xy + direction * lengths[:, None]).clamp(0, resolution)
        v2 = (xy - direction * lengths[:, None]).clamp(0, resolution)
        return torch.cat([v1[:, None], v2[:, None]], 1)


def ce_loss(logits, label, alpha=None):
    """
    merge 2 cls and multi-cls ce loss
    :param logits: [cls_num, bs, h, w]
    :param label: [cls_num, bs, h, w]
    :return:
    """
    cls_num, bs, h, w = logits.shape
    label = label.reshape(-1, bs, h, w)
    if cls_num == 2:
        label = torch.cat([1 - label, label], 0)
    nlogp = F.log_softmax(logits, 0)
    if cls_num == 2:
        loss = label * nlogp
        return -loss.sum(0).mean(2).mean(1)
    else:
        loss = label * nlogp
        w = label.sum(0).sum(2).sum(1)
        loss_ = -loss.sum(0).sum(2).sum(1)
        return loss_ / w


def focal_loss(logits, label, alpha):
    """
    TODO
    another version of focal loss from corner net
    only for two cls
    :param logits: [cls_num, bs, h, w]
    :param label: [bs, h, w]
    :return:
    """
    cls_num, bs, h, w = logits.shape
    assert cls_num == 2
    logp = F.log_softmax(logits, 0)
    p = F.softmax(logits, 0)
    loss = label * p[0] ** alpha * logp[1] + (1 - label) * p[1] ** alpha * logp[0]
    mask = label.sum(2).sum(1)
    return -loss.sum(2).sum(1) / mask


def l12loss(logits, target, mask=None, loss='L1'):
    if loss == 'L1':
        loss = torch.abs(logits - target)
    elif loss == 'L2':
        loss = (logits - target) ** 2
    else:
        raise ValueError('no such loss')
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean(2).mean(1)


def sigmoid_l1_loss(logits, target, scale=1.0, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) * scale + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean(2).mean(1)


def structure_nms_torch(lines, scores, threshold=2):
    diff = ((lines[:, None, :, None] - lines[:, None]) ** 2).sum(-1)
    diff0, diff1 = diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    diff = torch.cat([diff0[..., None], diff1[..., None]], -1)
    diff, _ = torch.min(diff, -1)
    idx = diff <= threshold
    ii = torch.eye(len(lines), device=lines.device).bool()
    idx[ii] = False
    i = 1
    k = lines.shape[0]
    hit = idx[0]
    while i < k - 2:
        if hit[i]:
            i += 1
            continue
        else:
            hit[i + 1:] = hit[i + 1:] | idx[i, i + 1:]
            i += 1
    drop = ~hit
    return lines[drop], scores[drop]


class FClip(nn.Module):

    def __init__(self, backbone):
        super(FClip, self).__init__()
        self.backbone = backbone
        self.M_dic = M.to_dict()
        self._get_head_size()

    def _get_head_size(self):
        head_size = []
        for h in self.M_dic['head']['order']:
            head_size.append([self.M_dic['head'][h]['head_size']])
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def lcmap_head(self, output, target):
        name = 'lcmap'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(self.M_dic['head'][name]['head_size'], batch, row, col)
        if self.M_dic['head'][name]['loss'] == 'Focal_loss':
            alpha = self.M_dic['head'][name]['focal_alpha']
            loss = focal_loss(pred, target, alpha)
        elif self.M_dic['head'][name]['loss'] == 'CE':
            loss = ce_loss(pred, target, None)
        else:
            raise NotImplementedError
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(1, 0, 2, 3).softmax(1)[:, 1], loss * weight

    def lcoff_head(self, output, target, mask):
        name = 'lcoff'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(self.M_dic['head'][name]['head_size'], batch, row, col)
        loss = sum(sigmoid_l1_loss(pred[j], target[j], offset=-0.5, mask=mask) for j in range(2))
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(1, 0, 2, 3).sigmoid() - 0.5, loss * weight

    def lleng_head(self, output, target, mask):
        name = 'lleng'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(batch, row, col)
        if self.M_dic['head'][name]['loss'] == 'sigmoid_L1':
            loss = sigmoid_l1_loss(pred, target, mask=mask)
            pred = pred.sigmoid()
        elif self.M_dic['head'][name]['loss'] == 'L1':
            loss = l12loss(pred, target, mask=mask)
            pred = pred.clamp(0.0, 1.0)
        else:
            raise NotImplementedError
        weight = self.M_dic['head'][name]['loss_weight']
        return pred, loss * weight

    def angle_head(self, output, target, mask):
        name = 'angle'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(batch, row, col)
        if self.M_dic['head'][name]['loss'] == 'sigmoid_L1':
            loss = sigmoid_l1_loss(pred, target, mask=mask)
            pred = pred.sigmoid()
        elif self.M_dic['head'][name]['loss'] == 'L1':
            loss = l12loss(pred, target, mask=mask)
            pred = pred.clamp(0.0, 1.0)
        else:
            raise NotImplementedError
        weight = self.M_dic['head'][name]['loss_weight']
        return pred, loss * weight

    def jmap_head(self, output, target, n_jtyp):
        name = 'jmap'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(n_jtyp, self.M_dic['head'][name]['head_size'], batch, row, col)
        if self.M_dic['head'][name]['loss'] == 'Focal_loss':
            alpha = self.M_dic['head'][name]['focal_alpha']
            loss = sum(focal_loss(pred[i], target[i], alpha) for i in range(n_jtyp))
        elif self.M_dic['head'][name]['loss'] == 'CE':
            loss = sum(ce_loss(pred[i], target[i], None) for i in range(n_jtyp))
        else:
            raise NotImplementedError
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1], loss * weight

    def joff_head(self, output, target, n_jtyp, mask):
        name = 'joff'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(n_jtyp, self.M_dic['head'][name]['head_size'], batch, row, col)
        loss = sum(sigmoid_l1_loss(pred[i, j], target[i, j], scale=1.0, offset=-0.5, mask=mask[i]) for i in range(n_jtyp) for j in range(2))
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(2, 0, 1, 3, 4).sigmoid() - 0.5, loss * weight

    def lmap_head(self, output, target):
        name = 'lmap'
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s:self.head_off[offidx]].reshape(batch, row, col)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean(2).mean(1)
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.sigmoid(), loss * weight

    def forward(self, input_dict, isTest=False):
        if isTest:
            return self.test_forward(input_dict)
        else:
            return self.trainval_forward(input_dict)

    def test_forward(self, input_dict):
        extra_info = {'time_front': 0.0, 'time_stack0': 0.0, 'time_stack1': 0.0, 'time_backbone': 0.0}
        extra_info['time_backbone'] = time.time()
        image = input_dict['image']
        outputs, feature, backbone_time = self.backbone(image)
        extra_info['time_front'] = backbone_time['time_front']
        extra_info['time_stack0'] = backbone_time['time_stack0']
        extra_info['time_stack1'] = backbone_time['time_stack1']
        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']
        output = outputs[0]
        heatmap = {}
        heatmap['lcmap'] = output[:, 0:self.head_off[0]].softmax(1)[:, 1]
        heatmap['lcoff'] = output[:, self.head_off[0]:self.head_off[1]].sigmoid() - 0.5
        heatmap['lleng'] = output[:, self.head_off[1]:self.head_off[2]].sigmoid()
        heatmap['angle'] = output[:, self.head_off[2]:self.head_off[3]].sigmoid()
        parsing = True
        if parsing:
            lines, scores = [], []
            for k in range(output.shape[0]):
                line, score = OneStageLineParsing.fclip_torch(lcmap=heatmap['lcmap'][k], lcoff=heatmap['lcoff'][k], lleng=heatmap['lleng'][k], angle=heatmap['angle'][k], delta=M.delta, resolution=M.resolution)
                if M.s_nms > 0:
                    line, score = structure_nms_torch(line, score, M.s_nms)
                lines.append(line[None])
                scores.append(score[None])
            heatmap['lines'] = torch.cat(lines)
            heatmap['score'] = torch.cat(scores)
        return {'heatmaps': heatmap, 'extra_info': extra_info}

    def trainval_forward(self, input_dict):
        image = input_dict['image']
        outputs, feature, backbone_time = self.backbone(image)
        result = {'feature': feature}
        batch, channel, row, col = outputs[0].shape
        T = input_dict['target'].copy()
        n_jtyp = 1
        T['lcoff'] = T['lcoff'].permute(1, 0, 2, 3)
        losses = []
        accuracy = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            L = OrderedDict()
            Acc = OrderedDict()
            heatmap = {}
            lcmap, L['lcmap'] = self.lcmap_head(output, T['lcmap'])
            lcoff, L['lcoff'] = self.lcoff_head(output, T['lcoff'], mask=T['lcmap'])
            heatmap['lcmap'] = lcmap
            heatmap['lcoff'] = lcoff
            lleng, L['lleng'] = self.lleng_head(output, T['lleng'], mask=T['lcmap'])
            angle, L['angle'] = self.angle_head(output, T['angle'], mask=T['lcmap'])
            heatmap['lleng'] = lleng
            heatmap['angle'] = angle
            losses.append(L)
            accuracy.append(Acc)
            if stack == 0 and input_dict['do_evaluation']:
                result['heatmaps'] = heatmap
        result['losses'] = losses
        result['accuracy'] = accuracy
        return result


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Delay_Xili_F_Clip(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

