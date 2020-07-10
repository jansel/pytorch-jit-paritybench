import sys
_module = sys.modules[__name__]
del sys
audio = _module
wavallin = _module
evaluate = _module
hparams = _module
lrschedule = _module
mksubset = _module
preprocess = _module
preprocess_normalize = _module
setup = _module
synthesis = _module
test_audio = _module
test_misc = _module
test_mixture = _module
test_model = _module
tojson = _module
train = _module
wavenet_vocoder = _module
conv = _module
mixture = _module
modules = _module
tfcompat = _module
hparam = _module
upsample = _module
util = _module
version = _module
wavenet = _module

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


import torch


import numpy as np


from scipy.io import wavfile


from torch.utils import data as data_utils


from torch.nn import functional as F


from torch import nn


from functools import partial


import random


from torch import optim


import torch.backends.cudnn as cudnn


from torch.utils.data.sampler import Sampler


from warnings import warn


import math


from torch.distributions import Normal


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError('Should provide either lengths or mask')
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
        mask_ = mask.expand_as(target)
        losses = self.criterion(input, target)
        return (losses * mask_).sum() / mask_.sum()


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=256, log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.

    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3
    y_hat = y_hat.transpose(1, 2)
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)
    y = y.expand_as(means)
    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1.0 / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_y
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    inner_inner_cond = (cdf_delta > 1e-05).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = log_probs + F.log_softmax(logit_probs, -1)
    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def _cast_to_type_if_compatible(name, param_type, value):
    """Cast hparam to the provided type, if compatible.

  Args:
    name: Name of the hparam to be cast.
    param_type: The type of the hparam.
    value: The value to be cast, if compatible.

  Returns:
    The result of casting `value` to `param_type`.

  Raises:
    ValueError: If the type of `value` is not compatible with param_type.
      * If `param_type` is a string type, but `value` is not.
      * If `param_type` is a boolean, but `value` is not, or vice versa.
      * If `param_type` is an integer type, but `value` is not.
      * If `param_type` is a float type, but `value` is not a numeric type.
  """
    fail_msg = "Could not cast hparam '%s' of type '%s' from value %r" % (name, param_type, value)
    if issubclass(param_type, type(None)):
        return value
    if issubclass(param_type, (six.string_types, six.binary_type)) and not isinstance(value, (six.string_types, six.binary_type)):
        raise ValueError(fail_msg)
    if issubclass(param_type, bool) != isinstance(value, bool):
        raise ValueError(fail_msg)
    if issubclass(param_type, numbers.Integral) and not isinstance(value, numbers.Integral):
        raise ValueError(fail_msg)
    if issubclass(param_type, numbers.Number) and not isinstance(value, numbers.Number):
        raise ValueError(fail_msg)
    return param_type(value)


PARAM_RE = re.compile("""
  (?P<name>[a-zA-Z][\\w\\.]*)      # variable name: "var" or "x"
  (\\[\\s*(?P<index>\\d+)\\s*\\])?  # (optional) index: "1" or None
  \\s*=\\s*
  ((?P<val>[^,\\[]*)            # single value: "a" or None
   |
   \\[(?P<vals>[^\\]]*)\\])       # list of values: None or "1,2,3"
  ($|,\\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
    """Helper function for raising a value error for bad assignment."""
    raise ValueError("Could not parse hparam '%s' of type '%s' with value '%s' in %s" % (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
    """Helper function for raising a value error for reuse of name."""
    raise ValueError("Multiple assignments to variable '%s' in %s" % (name, values))


def _process_list_value(name, parse_fn, var_type, m_dict, values, results_dictionary):
    """Update results_dictionary from a list of values.

  Used to update results_dictionary to be returned by parse_values when
  encountering a clause with a list RHS (e.g.  "arr=[1,2,3]".)

  Mutates results_dictionary.

  Args:
    name: Name of variable in assignment ("arr").
    parse_fn: Function for parsing individual values.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.

  Raises:
    ValueError: If the name has an index or the values cannot be parsed.
  """
    if m_dict['index'] is not None:
        raise ValueError('Assignment of a list to a list index.')
    elements = filter(None, re.split('[ ,]', m_dict['vals']))
    if name in results_dictionary:
        raise _reuse_fail(name, values)
    try:
        results_dictionary[name] = [parse_fn(e) for e in elements]
    except ValueError:
        _parse_fail(name, var_type, m_dict['vals'], values)


def _process_scalar_value(name, parse_fn, var_type, m_dict, values, results_dictionary):
    """Update results_dictionary with a scalar value.

  Used to update the results_dictionary to be returned by parse_values when
  encountering a clause with a scalar RHS (e.g.  "s=5" or "arr[0]=5".)

  Mutates results_dictionary.

  Args:
    name: Name of variable in assignment ("s" or "arr").
    parse_fn: Function for parsing the actual value.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
      m_dict['index']: List index value (or None)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.

  Raises:
    ValueError: If the name has already been used.
  """
    try:
        parsed_value = parse_fn(m_dict['val'])
    except ValueError:
        _parse_fail(name, var_type, m_dict['val'], values)
    if not m_dict['index']:
        if name in results_dictionary:
            _reuse_fail(name, values)
        results_dictionary[name] = parsed_value
    else:
        if name in results_dictionary:
            if not isinstance(results_dictionary.get(name), dict):
                _reuse_fail(name, values)
        else:
            results_dictionary[name] = {}
        index = int(m_dict['index'])
        if index in results_dictionary[name]:
            _reuse_fail('{}[{}]'.format(name, index), values)
        results_dictionary[name][index] = parsed_value


def parse_values(values, type_map):
    """Parses hyperparameter values from a string into a python map.

  `values` is a string containing comma-separated `name=value` pairs.
  For each pair, the value of the hyperparameter named `name` is set to
  `value`.

  If a hyperparameter name appears multiple times in `values`, a ValueError
  is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').

  If a hyperparameter name in both an index assignment and scalar assignment,
  a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').

  The hyperparameter name may contain '.' symbols, which will result in an
  attribute name that is only accessible through the getattr and setattr
  functions.  (And must be first explicit added through add_hparam.)

  WARNING: Use of '.' in your variable names is allowed, but is not well
  supported and not recommended.

  The `value` in `name=value` must follows the syntax according to the
  type of the parameter:

  *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
     100, -12.
  *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
     -.54e89.
  *  Boolean: Either true or false.
  *  Scalar string: A non-empty sequence of characters, excluding comma,
     spaces, and square brackets.  E.g.: foo, bar_1.
  *  List: A comma separated list of scalar values of the parameter type
     enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].

  When index assignment is used, the corresponding type_map key should be the
  list name.  E.g. for "arr[1]=0" the type_map must have the key "arr" (not
  "arr[1]").

  Args:
    values: String.  Comma separated list of `name=value` pairs where
      'value' must follow the syntax described above.
    type_map: A dictionary mapping hyperparameter names to types.  Note every
      parameter name in values must be a key in type_map.  The values must
      conform to the types indicated, where a value V is said to conform to a
      type T if either V has type T, or V is a list of elements of type T.
      Hence, for a multidimensional parameter 'x' taking float values,
      'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.

  Returns:
    A python map mapping each name to either:
    * A scalar value.
    * A list of scalar values.
    * A dictionary mapping index numbers to scalar values.
    (e.g. "x=5,L=[1,2],arr[1]=3" results in {'x':5,'L':[1,2],'arr':{1:3}}")

  Raises:
    ValueError: If there is a problem with input.
    * If `values` cannot be parsed.
    * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
    * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
      'a[1]=1,a[1]=2', or 'a=1,a=[1]')
  """
    results_dictionary = {}
    pos = 0
    while pos < len(values):
        m = PARAM_RE.match(values, pos)
        if not m:
            raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
        pos = m.end()
        m_dict = m.groupdict()
        name = m_dict['name']
        if name not in type_map:
            raise ValueError('Unknown hyperparameter type for %s' % name)
        type_ = type_map[name]
        if type_ == bool:

            def parse_bool(value):
                if value in ['true', 'True']:
                    return True
                elif value in ['false', 'False']:
                    return False
                else:
                    try:
                        return bool(int(value))
                    except ValueError:
                        _parse_fail(name, type_, value, values)
            parse = parse_bool
        else:
            parse = type_
        if m_dict['val'] is not None:
            _process_scalar_value(name, parse, type_, m_dict, values, results_dictionary)
        elif m_dict['vals'] is not None:
            _process_list_value(name, parse, type_, m_dict, values, results_dictionary)
        else:
            _parse_fail(name, type_, '', values)
    return results_dictionary


class HParams(object):
    """Class to hold a set of hyperparameters as name-value pairs.

  A `HParams` object holds hyperparameters used to build and train a model,
  such as the number of hidden units in a neural net layer or the learning rate
  to use when training.

  You first create a `HParams` object by specifying the names and values of the
  hyperparameters.

  To make them easily accessible the parameter names are added as direct
  attributes of the class.  A typical usage is as follows:

  ```python
  # Create a HParams object specifying names and values of the model
  # hyperparameters:
  hparams = HParams(learning_rate=0.1, num_hidden_units=100)

  # The hyperparameter are available as attributes of the HParams object:
  hparams.learning_rate ==> 0.1
  hparams.num_hidden_units ==> 100
  ```

  Hyperparameters have type, which is inferred from the type of their value
  passed at construction type.   The currently supported types are: integer,
  float, boolean, string, and list of integer, float, boolean, or string.

  You can override hyperparameter values by calling the
  [`parse()`](#HParams.parse) method, passing a string of comma separated
  `name=value` pairs.  This is intended to make it possible to override
  any hyperparameter values from a single command-line flag to which
  the user passes 'hyper-param=value' pairs.  It avoids having to define
  one flag for each hyperparameter.

  The syntax expected for each value depends on the type of the parameter.
  See `parse()` for a description of the syntax.

  Example:

  ```python
  # Define a command line flag to pass name=value pairs.
  # For example using argparse:
  import argparse
  parser = argparse.ArgumentParser(description='Train my model.')
  parser.add_argument('--hparams', type=str,
                      help='Comma separated list of "name=value" pairs.')
  args = parser.parse_args()
  ...
  def my_program():
    # Create a HParams object specifying the names and values of the
    # model hyperparameters:
    hparams = tf.HParams(learning_rate=0.1, num_hidden_units=100,
                         activations=['relu', 'tanh'])

    # Override hyperparameters values by parsing the command line
    hparams.parse(args.hparams)

    # If the user passed `--hparams=learning_rate=0.3` on the command line
    # then 'hparams' has the following attributes:
    hparams.learning_rate ==> 0.3
    hparams.num_hidden_units ==> 100
    hparams.activations ==> ['relu', 'tanh']

    # If the hyperparameters are in json format use parse_json:
    hparams.parse_json('{"learning_rate": 0.3, "activations": "relu"}')
  ```
  """
    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self, hparam_def=None, model_structure=None, **kwargs):
        """Create an instance of `HParams` from keyword arguments.

    The keyword arguments specify name-values pairs for the hyperparameters.
    The parameter types are inferred from the type of the values passed.

    The parameter names are added as attributes of `HParams` object, so they
    can be accessed directly with the dot notation `hparams._name_`.

    Example:

    ```python
    # Define 3 hyperparameters: 'learning_rate' is a float parameter,
    # 'num_hidden_units' an integer parameter, and 'activation' a string
    # parameter.
    hparams = tf.HParams(
        learning_rate=0.1, num_hidden_units=100, activation='relu')

    hparams.activation ==> 'relu'
    ```

    Note that a few names are reserved and cannot be used as hyperparameter
    names.  If you use one of the reserved name the constructor raises a
    `ValueError`.

    Args:
      hparam_def: Serialized hyperparameters, encoded as a hparam_pb2.HParamDef
        protocol buffer. If provided, this object is initialized by
        deserializing hparam_def.  Otherwise **kwargs is used.
      model_structure: An instance of ModelStructure, defining the feature
        crosses to be used in the Trial.
      **kwargs: Key-value pairs where the key is the hyperparameter name and
        the value is the value for the parameter.

    Raises:
      ValueError: If both `hparam_def` and initialization values are provided,
        or if one of the arguments is invalid.

    """
        self._hparam_types = {}
        self._model_structure = model_structure
        if hparam_def:
            raise ValueError('hparam_def has been disabled in this version')
        else:
            for name, value in six.iteritems(kwargs):
                self.add_hparam(name, value)

    def add_hparam(self, name, value):
        """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
        if getattr(self, name, None) is not None:
            raise ValueError('Hyperparameter name is reserved: %s' % name)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError('Multi-valued hyperparameters cannot be empty: %s' % name)
            self._hparam_types[name] = type(value[0]), True
        else:
            self._hparam_types[name] = type(value), False
        setattr(self, name, value)

    def set_hparam(self, name, value):
        """Set the value of an existing hyperparameter.

    This function verifies that the type of the value matches the type of the
    existing hyperparameter.

    Args:
      name: Name of the hyperparameter.
      value: New value of the hyperparameter.

    Raises:
      ValueError: If there is a type mismatch.
    """
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError('Must not pass a list for single-valued parameter: %s' % name)
            setattr(self, name, [_cast_to_type_if_compatible(name, param_type, v) for v in value])
        else:
            if is_list:
                raise ValueError('Must pass a list for multi-valued parameter: %s.' % name)
            setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

    def del_hparam(self, name):
        """Removes the hyperparameter with key 'name'.

    Args:
      name: Name of the hyperparameter.
    """
        if hasattr(self, name):
            delattr(self, name)
            del self._hparam_types[name]

    def parse(self, values):
        """Override hyperparameter values, parsing new values from a string.

    See parse_values for more detail on the allowed format for values.

    Args:
      values: String.  Comma separated list of `name=value` pairs where
        'value' must follow the syntax described above.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values` cannot be parsed.
    """
        type_map = dict()
        for name, t in self._hparam_types.items():
            param_type, _ = t
            type_map[name] = param_type
        values_map = parse_values(values, type_map)
        return self.override_from_dict(values_map)

    def override_from_dict(self, values_dict):
        """Override hyperparameter values, parsing new values from a dictionary.

    Args:
      values_dict: Dictionary of name:value pairs.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values_dict` cannot be parsed.
    """
        for name, value in values_dict.items():
            self.set_hparam(name, value)
        return self

    def set_from_map(self, values_map):
        """DEPRECATED. Use override_from_dict."""
        return self.override_from_dict(values_dict=values_map)

    def set_model_structure(self, model_structure):
        self._model_structure = model_structure

    def get_model_structure(self):
        return self._model_structure

    def to_json(self, indent=None, separators=None, sort_keys=False):
        """Serializes the hyperparameters into JSON.

    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.

    Returns:
      A JSON string.
    """
        return json.dumps(self.values(), indent=indent, separators=separators, sort_keys=sort_keys)

    def parse_json(self, values_json):
        """Override hyperparameter values, parsing new values from a json object.

    Args:
      values_json: String containing a json object of name:value pairs.

    Returns:
      The `HParams` instance.

    Raises:
      ValueError: If `values_json` cannot be parsed.
    """
        values_map = json.loads(values_json)
        return self.override_from_dict(values_map)

    def values(self):
        """Return the hyperparameter values as a Python dictionary.

    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    """
        return {n: getattr(self, n) for n in self._hparam_types.keys()}

    def get(self, key, default=None):
        """Returns the value of `key` if it exists, else `default`."""
        if key in self._hparam_types:
            if default is not None:
                param_type, is_param_list = self._hparam_types[key]
                type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
                fail_msg = "Hparam '%s' of type '%s' is incompatible with default=%s" % (key, type_str, default)
                is_default_list = isinstance(default, list)
                if is_param_list != is_default_list:
                    raise ValueError(fail_msg)
                try:
                    if is_default_list:
                        for value in default:
                            _cast_to_type_if_compatible(key, param_type, value)
                    else:
                        _cast_to_type_if_compatible(key, param_type, default)
                except ValueError as e:
                    raise ValueError('%s. %s' % (fail_msg, e))
            return getattr(self, key)
        return default

    def __contains__(self, key):
        return key in self._hparam_types

    def __str__(self):
        return str(sorted(self.values().items()))

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self.__str__())

    @staticmethod
    def _get_kind_name(param_type, is_list):
        """Returns the field name given parameter type and is_list.

    Args:
      param_type: Data type of the hparam.
      is_list: Whether this is a list.

    Returns:
      A string representation of the field name.

    Raises:
      ValueError: If parameter type is not recognized.
    """
        if issubclass(param_type, bool):
            typename = 'bool'
        elif issubclass(param_type, six.integer_types):
            typename = 'int64'
        elif issubclass(param_type, (six.string_types, six.binary_type)):
            typename = 'bytes'
        elif issubclass(param_type, float):
            typename = 'float'
        else:
            raise ValueError('Unsupported parameter type: %s' % str(param_type))
        suffix = 'list' if is_list else 'value'
        return '_'.join([typename, suffix])


class DiscretizedMixturelogisticLoss(nn.Module):

    def __init__(self):
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError('Should provide either lengths or mask')
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
        mask_ = mask.expand_as(target)
        losses = discretized_mix_logistic_loss(input, target, num_classes=hparams.quantize_channels, log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return (losses * mask_).sum() / mask_.sum()


def mix_gaussian_loss(y_hat, y, log_scale_min=-7.0, reduce=True):
    """Mixture of continuous gaussian distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    C = y_hat.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y_hat.size(1) % 3 == 0
        nr_mix = y_hat.size(1) // 3
    y_hat = y_hat.transpose(1, 2)
    if C == 2:
        logit_probs = None
        means = y_hat[:, :, 0:1]
        log_scales = torch.clamp(y_hat[:, :, 1:2], min=log_scale_min)
    else:
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)
    y = y.expand_as(means)
    centered_y = y - means
    dist = Normal(loc=0.0, scale=torch.exp(log_scales))
    log_probs = dist.log_prob(centered_y)
    if nr_mix > 1:
        log_probs = log_probs + F.log_softmax(logit_probs, -1)
    if reduce:
        if nr_mix == 1:
            return -torch.sum(log_probs)
        else:
            return -torch.sum(log_sum_exp(log_probs))
    elif nr_mix == 1:
        return -log_probs
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


class MixtureGaussianLoss(nn.Module):

    def __init__(self):
        super(MixtureGaussianLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError('Should provide either lengths or mask')
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
        mask_ = mask.expand_as(target)
        losses = mix_gaussian_loss(input, target, log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return (losses * mask_).sum() / mask_.sum()


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def Conv1d1x1(in_channels, out_channels, bias=True):
    """1-by-1 convolution layer
    """
    return Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels=None, cin_channels=-1, gin_channels=-1, dropout=1 - 0.95, padding=None, dilation=1, causal=True, bias=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, *args, padding=padding, dilation=dilation, bias=bias, **kwargs)
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)
        else:
            self.conv1x1c = None
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=False)
        else:
            self.conv1x1g = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward

        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            x = x[:, :, :residual.size(-1)] if self.causal else x
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb
        x = torch.tanh(a) * torch.sigmoid(b)
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)
        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip, self.conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()


class Stretch2d(nn.Module):

    def __init__(self, x_scale, y_scale, mode='nearest'):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


def _get_activation(upsample_activation):
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear


class UpsampleNetwork(nn.Module):

    def __init__(self, upsample_scales, upsample_activation='none', upsample_activation_params={}, mode='nearest', freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = freq_axis_kernel_size, scale * 2 + 1
            padding = freq_axis_padding, scale
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            if upsample_activation != 'none':
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    def forward(self, c):
        """
        Args:
            c : B x C x T
        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        c = c.squeeze(1)
        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]
        return c


class ConvInUpsampleNetwork(nn.Module):

    def __init__(self, upsample_scales, upsample_activation='none', upsample_activation_params={}, mode='nearest', freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(ConvInUpsampleNetwork, self).__init__()
        ks = 2 * cin_pad + 1
        self.conv_in = nn.Conv1d(cin_channels, cin_channels, kernel_size=ks, bias=False)
        self.upsample = UpsampleNetwork(upsample_scales, upsample_activation, upsample_activation_params, mode, freq_axis_kernel_size, cin_pad=0, cin_channels=cin_channels)

    def forward(self, c):
        c_up = self.upsample(self.conv_in(c))
        return c_up


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Tensor): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Tensor: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


def receptive_field_size(total_layers, num_cycles, kernel_size, dilation=lambda x: 2 ** x):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0, clamp_log_scale=False):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = to_one_hot(argmax, nr_mix)
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    u = means.data.new(means.size()).uniform_(1e-05, 1.0 - 1e-05)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)
    return x


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample from (discretized) mixture of gaussian distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    C = y.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y.size(1) % 3 == 0
        nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]
    if nr_mix > 1:
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)
        temp = logit_probs.data - torch.log(-torch.log(temp))
        _, argmax = temp.max(dim=-1)
        one_hot = to_one_hot(argmax, nr_mix)
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    elif C == 2:
        means, log_scales = y[:, :, (0)], y[:, :, (1)]
    elif C == 3:
        means, log_scales = y[:, :, (1)], y[:, :, (2)]
    else:
        assert False, "shouldn't happen"
    scales = torch.exp(log_scales)
    dist = Normal(loc=means, scale=scales)
    x = dist.sample()
    x = torch.clamp(x, min=-1.0, max=1.0)
    return x


class WaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
    """

    def __init__(self, out_channels=256, layers=20, stacks=2, residual_channels=512, gate_channels=512, skip_out_channels=512, kernel_size=3, dropout=1 - 0.95, cin_channels=-1, gin_channels=-1, n_speakers=None, upsample_conditional_features=False, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}, scalar_input=False, use_speaker_embedding=False, output_distribution='Logistic', cin_pad=0):
        super(WaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.output_distribution = output_distribution
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)
        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(residual_channels, gate_channels, kernel_size=kernel_size, skip_out_channels=skip_out_channels, bias=True, dilation=dilation, dropout=dropout, cin_channels=cin_channels, gin_channels=gin_channels)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([nn.ReLU(inplace=True), Conv1d1x1(skip_out_channels, skip_out_channels), nn.ReLU(inplace=True), Conv1d1x1(skip_out_channels, out_channels)])
        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None
        if upsample_conditional_features:
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        else:
            self.upsample_net = None
        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        B, _, T = x.size()
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_bct = _expand_global_features(B, T, g, bct=True)
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        x = F.softmax(x, dim=1) if softmax else x
        return x

    def incremental_forward(self, initial_input=None, c=None, g=None, T=100, test_inputs=None, tqdm=lambda x: x, softmax=True, quantize=True, log_scale_min=-50.0):
        """Incremental forward step

        Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
        to (B x T x C) internally and fed to the network for each time step.
        Input of each time step will be of shape (B x 1 x C).

        Args:
            initial_input (Tensor): Initial decoder input, (B x C x 1)
            c (Tensor): Local conditioning features, shape (B x C' x T)
            g (Tensor): Global conditioning features, shape (B x C'' or B x C''x 1)
            T (int): Number of time steps to generate.
            test_inputs (Tensor): Teacher forcing inputs (for debugging)
            tqdm (lamda) : tqdm
            softmax (bool) : Whether applies softmax or not
            quantize (bool): Whether quantize softmax output before feeding the
              network output to input for the next time step. TODO: rename
            log_scale_min (float):  Log scale minimum value.

        Returns:
            Tensor: Generated one-hot encoded samples. B x C x T　
              or scaler vector B x 1 x T
        """
        self.clear_buffer()
        B = 1
        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.size(1) == 1:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            elif test_inputs.size(1) == self.out_channels:
                test_inputs = test_inputs.transpose(1, 2).contiguous()
            B = test_inputs.size(0)
            if T is None:
                T = test_inputs.size(1)
            else:
                T = max(T, test_inputs.size(1))
        T = int(T)
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_btc = _expand_global_features(B, T, g, bct=False)
        if c is not None:
            B = c.shape[0]
            if self.upsample_net is not None:
                c = self.upsample_net(c)
                assert c.size(-1) == T
            if c.size(-1) == T:
                c = c.transpose(1, 2).contiguous()
        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = torch.zeros(B, 1, 1)
            else:
                initial_input = torch.zeros(B, 1, self.out_channels)
                initial_input[:, :, (127)] = 1
            if next(self.parameters()).is_cuda:
                initial_input = initial_input
        elif initial_input.size(1) == self.out_channels:
            initial_input = initial_input.transpose(1, 2).contiguous()
        current_input = initial_input
        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, (t), :].unsqueeze(1)
            elif t > 0:
                current_input = outputs[-1]
            ct = None if c is None else c[:, (t), :].unsqueeze(1)
            gt = None if g is None else g_btc[:, (t), :].unsqueeze(1)
            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = 0
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt)
                skips += h
            skips *= math.sqrt(1.0 / len(self.conv_layers))
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)
            if self.scalar_input:
                if self.output_distribution == 'Logistic':
                    x = sample_from_discretized_mix_logistic(x.view(B, -1, 1), log_scale_min=log_scale_min)
                elif self.output_distribution == 'Normal':
                    x = sample_from_mix_gaussian(x.view(B, -1, 1), log_scale_min=log_scale_min)
                else:
                    assert False
            else:
                x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
                if quantize:
                    dist = torch.distributions.OneHotCategorical(x)
                    x = dist.sample()
            outputs += [x.data]
        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()
        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Stretch2d,
     lambda: ([], {'x_scale': 1.0, 'y_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_r9y9_wavenet_vocoder(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

