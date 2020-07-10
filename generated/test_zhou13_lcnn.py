import sys
_module = sys.modules[__name__]
del sys
wireframe = _module
york = _module
demo = _module
lcnn = _module
box = _module
config = _module
datasets = _module
metric = _module
models = _module
hourglass_pose = _module
line_vectorizer = _module
multitask_learner = _module
postprocess = _module
trainer = _module
utils = _module
lsd = _module
post = _module
process = _module
train = _module

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


import random


import numpy as np


import torch


import math


import numpy.linalg as LA


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


import torch.nn as nn


import torch.nn.functional as F


import itertools


from collections import defaultdict


from collections import OrderedDict


import time


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
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
        return out[::-1], y


class Bottleneck1D(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()
        planes = outplanes // 2
        self.op = nn.Sequential(nn.BatchNorm1d(inplanes), nn.ReLU(inplace=True), nn.Conv1d(inplanes, planes, kernel_size=1), nn.BatchNorm1d(planes), nn.ReLU(inplace=True), nn.Conv1d(planes, planes, kernel_size=3, padding=1), nn.BatchNorm1d(planes), nn.ReLU(inplace=True), nn.Conv1d(planes, outplanes, kernel_size=1))

    def forward(self, x):
        return x + self.op(x)


FEATURE_DIM = 8


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


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


class LineVectorizer(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        lambda_ = torch.linspace(0, 1, M.n_pts0)[:, (None)]
        self.register_buffer('lambda_', lambda_)
        self.do_static_sampling = M.n_stc_posl + M.n_stc_negl > 0
        self.fc1 = nn.Conv2d(256, M.dim_loi, 1)
        scale_factor = M.n_pts0 // M.n_pts1
        if M.use_conv:
            self.pooling = nn.Sequential(nn.MaxPool1d(scale_factor, scale_factor), Bottleneck1D(M.dim_loi, M.dim_loi))
            self.fc2 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, 1))
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc), nn.ReLU(inplace=True), nn.Linear(M.dim_fc, M.dim_fc), nn.ReLU(inplace=True), nn.Linear(M.dim_fc, 1))
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input_dict):
        result = self.backbone(input_dict)
        h = result['preds']
        x = self.fc1(result['feature'])
        n_batch, n_channel, row, col = x.shape
        xs, ys, fs, ps, idx, jcs = [], [], [], [], [0], []
        for i, meta in enumerate(input_dict['meta']):
            p, label, feat, jc = self.sample_lines(meta, h['jmap'][i], h['joff'][i], input_dict['mode'])
            ys.append(label)
            if input_dict['mode'] == 'training' and self.do_static_sampling:
                p = torch.cat([p, meta['lpre']])
                feat = torch.cat([feat, meta['lpre_feat']])
                ys.append(meta['lpre_label'])
                del jc
            else:
                jcs.append(jc)
                ps.append(p)
            fs.append(feat)
            p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
            p = p.reshape(-1, 2)
            px, py = p[:, (0)].contiguous(), p[:, (1)].contiguous()
            px0 = px.floor().clamp(min=0, max=127)
            py0 = py.floor().clamp(min=0, max=127)
            px1 = (px0 + 1).clamp(min=0, max=127)
            py1 = (py0 + 1).clamp(min=0, max=127)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
            xp = (x[(i), :, (px0l), (py0l)] * (px1 - px) * (py1 - py) + x[(i), :, (px1l), (py0l)] * (px - px0) * (py1 - py) + x[(i), :, (px0l), (py1l)] * (px1 - px) * (py - py0) + x[(i), :, (px1l), (py1l)] * (px - px0) * (py - py0)).reshape(n_channel, -1, M.n_pts0).permute(1, 0, 2)
            xp = self.pooling(xp)
            xs.append(xp)
            idx.append(idx[-1] + xp.shape[0])
        x, y = torch.cat(xs), torch.cat(ys)
        f = torch.cat(fs)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)
        x = torch.cat([x, f], 1)
        x = self.fc2(x).flatten()
        if input_dict['mode'] != 'training':
            p = torch.cat(ps)
            s = torch.sigmoid(x)
            b = s > 0.5
            lines = []
            score = []
            for i in range(n_batch):
                p0 = p[idx[i]:idx[i + 1]]
                s0 = s[idx[i]:idx[i + 1]]
                mask = b[idx[i]:idx[i + 1]]
                p0 = p0[mask]
                s0 = s0[mask]
                if len(p0) == 0:
                    lines.append(torch.zeros([1, M.n_out_line, 2, 2], device=p.device))
                    score.append(torch.zeros([1, M.n_out_line], device=p.device))
                else:
                    arg = torch.argsort(s0, descending=True)
                    p0, s0 = p0[arg], s0[arg]
                    lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                    score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
                for j in range(len(jcs[i])):
                    if len(jcs[i][j]) == 0:
                        jcs[i][j] = torch.zeros([M.n_out_junc, 2], device=p.device)
                    jcs[i][j] = jcs[i][j][None, torch.arange(M.n_out_junc) % len(jcs[i][j])]
            result['preds']['lines'] = torch.cat(lines)
            result['preds']['score'] = torch.cat(score)
            result['preds']['juncs'] = torch.cat([jcs[i][0] for i in range(n_batch)])
            if len(jcs[i]) > 1:
                result['preds']['junts'] = torch.cat([jcs[i][1] for i in range(n_batch)])
        if input_dict['mode'] != 'testing':
            y = torch.cat(ys)
            loss = self.loss(x, y)
            lpos_mask, lneg_mask = y, 1 - y
            loss_lpos, loss_lneg = loss * lpos_mask, loss * lneg_mask

            def sum_batch(x):
                xs = [x[idx[i]:idx[i + 1]].sum()[None] for i in range(n_batch)]
                return torch.cat(xs)
            lpos = sum_batch(loss_lpos) / sum_batch(lpos_mask).clamp(min=1)
            lneg = sum_batch(loss_lneg) / sum_batch(lneg_mask).clamp(min=1)
            result['losses'][0]['lpos'] = lpos * M.loss_weight['lpos']
            result['losses'][0]['lneg'] = lneg * M.loss_weight['lneg']
        if input_dict['mode'] == 'training':
            del result['preds']
        return result

    def sample_lines(self, meta, jmap, joff, mode):
        with torch.no_grad():
            junc = meta['junc']
            jtyp = meta['jtyp']
            Lpos = meta['Lpos']
            Lneg = meta['Lneg']
            n_type = jmap.shape[0]
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            max_K = M.n_dyn_junc // n_type
            N = len(junc)
            if mode != 'training':
                K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)
            else:
                K = min(int(N * 2 + 2), max_K)
            if K < 2:
                K = 2
            device = jmap.device
            score, index = torch.topk(jmap, k=K)
            y = (index / 128).float() + torch.gather(joff[:, (0)], 1, index) + 0.5
            x = (index % 128).float() + torch.gather(joff[:, (1)], 1, index) + 0.5
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[(...), (None), :]
            del x, y, index
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)
            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()
            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]
            label = Lpos[up, vp]
            if mode == 'training':
                c = torch.zeros_like(label, dtype=torch.bool)
                cdx = label.nonzero().flatten()
                if len(cdx) > M.n_dyn_posl:
                    perm = torch.randperm(len(cdx), device=device)[:M.n_dyn_posl]
                    cdx = cdx[perm]
                c[cdx] = 1
                cdx = Lneg[up, vp].nonzero().flatten()
                if len(cdx) > M.n_dyn_negl:
                    perm = torch.randperm(len(cdx), device=device)[:M.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1
                cdx = torch.randint(len(c), (M.n_dyn_othr,), device=device)
                c[cdx] = 1
            else:
                c = (u < v).flatten()
            u, v, label = u[c], v[c], label[c]
            xy = xy.reshape(n_type * K, 2)
            xyu, xyv = xy[u], xy[v]
            u2v = xyu - xyv
            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-06)
            feat = torch.cat([xyu / 128 * M.use_cood, xyv / 128 * M.use_cood, u2v * M.use_slop, (u[:, (None)] > K).float(), (v[:, (None)] > K).float()], 1)
            line = torch.cat([xyu[:, (None)], xyv[:, (None)]], 1)
            xy = xy.reshape(n_type, K, 2)
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]
            return line, label.float(), feat, jcs


class MultitaskHead(nn.Module):

    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()
        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(nn.Sequential(nn.Conv2d(input_channels, m, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(m, output_channels, kernel_size=1)))
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean(2).mean(1)


class MultitaskLearner(nn.Module):

    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict['image']
        outputs, feature = self.backbone(image)
        result = {'feature': feature}
        batch, channel, row, col = outputs[0].shape
        T = input_dict['target'].copy()
        n_jtyp = T['jmap'].shape[1]
        for task in ['jmap']:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ['joff']:
            T[task] = T[task].permute(1, 2, 0, 3, 4)
        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            jmap = output[0:offset[0]].reshape(n_jtyp, 2, batch, row, col)
            lmap = output[offset[0]:offset[1]].squeeze(0)
            joff = output[offset[1]:offset[2]].reshape(n_jtyp, 2, batch, row, col)
            if stack == 0:
                result['preds'] = {'jmap': jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, (1)], 'lmap': lmap.sigmoid(), 'joff': joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5}
                if input_dict['mode'] == 'testing':
                    return result
            L = OrderedDict()
            L['jmap'] = sum(cross_entropy_loss(jmap[i], T['jmap'][i]) for i in range(n_jtyp))
            L['lmap'] = F.binary_cross_entropy_with_logits(lmap, T['lmap'], reduction='none').mean(2).mean(1)
            L['joff'] = sum(sigmoid_l1_loss(joff[i, j], T['joff'][i, j], -0.5, T['jmap'][i]) for i in range(n_jtyp) for j in range(2))
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result['losses'] = losses
        return result


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bottleneck1D,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_zhou13_lcnn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

