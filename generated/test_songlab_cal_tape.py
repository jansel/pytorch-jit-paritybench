import sys
_module = sys.modules[__name__]
del sys
adding_model = _module
adding_task = _module
fix_lmdb = _module
generate_plots = _module
lmdb_to_fasta = _module
tfrecord_to_json = _module
tfrecord_to_lmdb = _module
setup = _module
tape = _module
datasets = _module
errors = _module
main = _module
metrics = _module
models = _module
file_utils = _module
modeling_bert = _module
modeling_lstm = _module
modeling_onehot = _module
modeling_resnet = _module
modeling_trrosetta = _module
modeling_unirep = _module
modeling_utils = _module
optimization = _module
registry = _module
tokenizers = _module
training = _module
utils = _module
_sampler = _module
distributed_utils = _module
setup_utils = _module
visualization = _module
test_basic = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


from typing import Union


from typing import List


from typing import Tuple


from typing import Sequence


from typing import Dict


from typing import Any


from typing import Optional


from typing import Collection


from copy import copy


import logging


import random


import numpy as np


import torch.nn.functional as F


from torch.utils.data import Dataset


from scipy.spatial.distance import pdist


from scipy.spatial.distance import squareform


import typing


import warnings


import inspect


import math


from torch import nn


from torch.utils.checkpoint import checkpoint


from torch.nn.utils import weight_norm


import copy


from torch.nn.utils.weight_norm import weight_norm


import torch.optim as optim


from torch.utils.data import DataLoader


from abc import ABC


from abc import abstractmethod


class ProteinBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinBertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class ProteinBertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class ProteinBertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = ProteinBertSelfAttention(config)
        self.output = ProteinBertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) ->typing.Callable:
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f'Unrecognized activation fn: {name}')


class ProteinBertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ProteinBertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinBertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = ProteinBertAttention(config)
        self.intermediate = ProteinBertIntermediate(config)
        self.output = ProteinBertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class ProteinBertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([ProteinBertLayer(config) for _ in range(config.num_hidden_layers)])

    def run_function(self, start, chunk_size):

        def custom_forward(hidden_states, attention_mask):
            all_hidden_states = ()
            all_attentions = ()
            chunk_slice = slice(start, start + chunk_size)
            for layer in self.layer[chunk_slice]:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]
                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = hidden_states,
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs
        return custom_forward

    def forward(self, hidden_states, attention_mask, chunks=None):
        all_hidden_states = ()
        all_attentions = ()
        if chunks is not None:
            assert isinstance(chunks, int)
            chunk_size = (len(self.layer) + chunks - 1) // chunks
            for start in range(0, len(self.layer), chunk_size):
                outputs = checkpoint(self.run_function(start, chunk_size), hidden_states, attention_mask)
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + outputs[1]
                if self.output_attentions:
                    all_attentions = all_attentions + outputs[-1]
                hidden_states = outputs[0]
        else:
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer_module(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]
                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = hidden_states,
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs


class ProteinBertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinLSTMLayer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout: float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class ProteinLSTMPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scalar_reweighting = nn.Linear(2 * config.num_hidden_layers, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.scalar_reweighting(hidden_states).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


CONFIG_NAME = 'config.json'


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


logger = logging.getLogger(__name__)


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response['Error']['Code']) == 404:
                raise EnvironmentError('file {} not found'.format(url))
            else:
                raise
    return wrapper


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError('bad s3 path {}'.format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith('/'):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    return filename


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PROTEIN_MODELS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if url.startswith('s3://'):
        etag = s3_etag(url)
    else:
        try:
            response = requests.head(url, allow_redirects=True)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get('ETag')
        except EnvironmentError:
            etag = None
    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])
    if not os.path.exists(cache_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info('%s not found in cache, downloading to %s', url, temp_file.name)
            if url.startswith('s3://'):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            logger.info('copying %s to cache at %s', temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
            logger.info('creating metadata file for %s', cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(output_string, 'utf-8')
                meta_file.write(output_string)
            logger.info('removing temp file %s', temp_file.name)
    return cache_path


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PROTEIN_MODELS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    parsed = urlparse(url_or_filename)
    if parsed.scheme in ('http', 'https', 's3'):
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif parsed.scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.format(url_or_filename))


class ProteinConfig(object):
    """ Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods
        for loading/downloading/saving configurations.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names`
                (string) as keys and `url` (string) of associated pretrained model
                configurations as values.

        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune
                the model.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is
                a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns
                attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all
                hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map: typing.Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~ProteinConfig.from_pretrained`
            class method.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """ Instantiate a :class:`~ProteinConfig`
             (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to
                  load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~ProteinConfig.save_pretrained` method,
                  e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`,
                  e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict:
                key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will
                  be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration
                  attributes is controlled by the `return_unused_kwargs` keyword parameter.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)`
                  where `unused_kwargs` is a dictionary consisting of the key/value pairs
                  whose keys are not configuration attributes: ie the part of kwargs which
                  has not been used to update `config` and is otherwise ignored.

        Examples::

            # We can't instantiate directly the base class `ProteinConfig` so let's
              show the examples on a derived class: ProteinBertConfig
            # Download configuration from S3 and cache.
            config = ProteinBertConfig.from_pretrained('bert-base-uncased')
            # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = ProteinBertConfig.from_pretrained('./test/saved_model/')
            config = ProteinBertConfig.from_pretrained(
                './test/saved_model/my_configuration.json')
            config = ProteinBertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True,
                foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained model configuration file.".format(config_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_config_archive_map.keys()), config_file))
            return None
        if resolved_config_file == config_file:
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        config = cls.from_json_file(resolved_config_file)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info('Model config %s', config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class MaskedConv1d(nn.Conv1d):

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)


class ProteinResNetLayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ProteinResNetBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = MaskedConv1d(config.hidden_size, config.hidden_size, 3, padding=1, bias=False)
        self.bn1 = ProteinResNetLayerNorm(config)
        self.conv2 = MaskedConv1d(config.hidden_size, config.hidden_size, 3, padding=1, bias=False)
        self.bn2 = ProteinResNetLayerNorm(config)
        self.activation_fn = get_activation_fn(config.hidden_act)

    def forward(self, x, input_mask=None):
        identity = x
        out = self.conv1(x, input_mask)
        out = self.bn1(out)
        out = self.activation_fn(out)
        out = self.conv2(out, input_mask)
        out = self.bn2(out)
        out += identity
        out = self.activation_fn(out)
        return out


class ProteinResNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, embed_dim, padding_idx=0)
        inverse_frequency = 1 / 10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim)
        self.register_buffer('inverse_frequency', inverse_frequency)
        self.layer_norm = LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length - 1, -1, -1.0, dtype=words_embeddings.dtype, device=words_embeddings.device)
        sinusoidal_input = torch.ger(position_ids, self.inverse_frequency)
        position_embeddings = torch.cat([sinusoidal_input.sin(), sinusoidal_input.cos()], -1)
        position_embeddings = position_embeddings.unsqueeze(0)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinResNetPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention_weights = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None):
        attention_scores = self.attention_weights(hidden_states)
        if mask is not None:
            attention_scores += -10000.0 * (1 - mask)
        attention_weights = torch.softmax(attention_scores, -1)
        weighted_mean_embedding = torch.matmul(hidden_states.transpose(1, 2), attention_weights).squeeze(2)
        pooled_output = self.dense(weighted_mean_embedding)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([ProteinResNetBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, input_mask=None):
        all_hidden_states = ()
        for layer_module in self.layer:
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states, input_mask)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs


URL_PREFIX = 'https://s3.amazonaws.com/proteindata/pytorch-models/'


TRROSETTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {'xaa': URL_PREFIX + 'trRosetta-xaa-config.json', 'xab': URL_PREFIX + 'trRosetta-xab-config.json', 'xac': URL_PREFIX + 'trRosetta-xac-config.json', 'xad': URL_PREFIX + 'trRosetta-xad-config.json', 'xae': URL_PREFIX + 'trRosetta-xae-config.json'}


class TRRosettaConfig(ProteinConfig):
    pretrained_config_archive_map = TRROSETTA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, num_features: int=64, kernel_size: int=3, num_layers: int=61, dropout: float=0.15, msa_cutoff: float=0.8, penalty_coeff: float=4.5, initializer_range: float=0.02, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.msa_cutoff = msa_cutoff
        self.penalty_coeff = penalty_coeff
        self.initializer_range = initializer_range


class MSAFeatureExtractor(nn.Module):

    def __init__(self, config: TRRosettaConfig):
        super().__init__()
        self.msa_cutoff = config.msa_cutoff
        self.penalty_coeff = config.penalty_coeff

    def forward(self, msa1hot):
        initial_type = msa1hot.dtype
        msa1hot = msa1hot.float()
        seqlen = msa1hot.size(2)
        weights = self.reweight(msa1hot)
        features_1d = self.extract_features_1d(msa1hot, weights)
        features_2d = self.extract_features_2d(msa1hot, weights)
        left = features_1d.unsqueeze(2).repeat(1, 1, seqlen, 1)
        right = features_1d.unsqueeze(1).repeat(1, seqlen, 1, 1)
        features = torch.cat((left, right, features_2d), -1)
        features = features.type(initial_type)
        features = features.permute(0, 3, 1, 2)
        features = features.contiguous()
        return features

    def reweight(self, msa1hot, eps=1e-09):
        seqlen = msa1hot.size(2)
        id_min = seqlen * self.msa_cutoff
        id_mtx = torch.stack([torch.tensordot(el, el, [[1, 2], [1, 2]]) for el in msa1hot], 0)
        id_mask = id_mtx > id_min
        weights = 1.0 / (id_mask.type_as(msa1hot).sum(-1) + eps)
        return weights

    def extract_features_1d(self, msa1hot, weights):
        f1d_seq = msa1hot[:, (0), :, :20]
        batch_size = msa1hot.size(0)
        seqlen = msa1hot.size(2)
        beff = weights.sum()
        f_i = (weights[:, :, (None), (None)] * msa1hot).sum(1) / beff + 1e-09
        h_i = (-f_i * f_i.log()).sum(2, keepdims=True)
        f1d_pssm = torch.cat((f_i, h_i), dim=2)
        f1d = torch.cat((f1d_seq, f1d_pssm), dim=2)
        f1d = f1d.view(batch_size, seqlen, 42)
        return f1d

    def extract_features_2d(self, msa1hot, weights):
        batch_size = msa1hot.size(0)
        num_alignments = msa1hot.size(1)
        seqlen = msa1hot.size(2)
        num_symbols = 21
        if num_alignments == 1:
            f2d_dca = torch.zeros(batch_size, seqlen, seqlen, 442, dtype=torch.float, device=msa1hot.device)
            return f2d_dca
        x = msa1hot.view(batch_size, num_alignments, seqlen * num_symbols)
        num_points = weights.sum(1) - weights.mean(1).sqrt()
        mean = (x * weights.unsqueeze(2)).sum(1, keepdims=True) / num_points[:, (None), (None)]
        x = (x - mean) * weights[:, :, (None)].sqrt()
        cov = torch.matmul(x.transpose(-1, -2), x) / num_points[:, (None), (None)]
        reg = torch.eye(seqlen * num_symbols, device=weights.device, dtype=weights.dtype)[None]
        reg = reg * self.penalty_coeff / weights.sum(1, keepdims=True).sqrt().unsqueeze(2)
        cov_reg = cov + reg
        inv_cov = torch.stack([torch.inverse(cr) for cr in cov_reg.unbind(0)], 0)
        x1 = inv_cov.view(batch_size, seqlen, num_symbols, seqlen, num_symbols)
        x2 = x1.permute(0, 1, 3, 2, 4)
        features = x2.reshape(batch_size, seqlen, seqlen, num_symbols * num_symbols)
        x3 = (x1[:, :, :-1, :, :-1] ** 2).sum((2, 4)).sqrt() * (1 - torch.eye(seqlen, device=weights.device, dtype=weights.dtype)[None])
        apc = x3.sum(1, keepdims=True) * x3.sum(2, keepdims=True) / x3.sum((1, 2), keepdims=True)
        contacts = (x3 - apc) * (1 - torch.eye(seqlen, device=x3.device, dtype=x3.dtype).unsqueeze(0))
        f2d_dca = torch.cat([features, contacts[:, :, :, (None)]], axis=3)
        return f2d_dca

    @property
    def feature_size(self) ->int:
        return 526


class DilatedResidualBlock(nn.Module):

    def __init__(self, num_features: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = self._get_padding(kernel_size, dilation)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(num_features, affine=True, eps=1e-06)
        self.actv1 = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(num_features, affine=True, eps=1e-06)
        self.actv2 = nn.ELU(inplace=True)
        self.apply(self._init_weights)
        nn.init.constant_(self.norm2.weight, 0)

    def _get_padding(self, kernel_size: int, dilation: int) ->int:
        return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, features):
        shortcut = features
        features = self.conv1(features)
        features = self.norm1(features)
        features = self.actv1(features)
        features = self.dropout(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.actv2(features + shortcut)
        return features


class mLSTMCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        project_size = config.hidden_size * 4
        self.wmx = weight_norm(nn.Linear(config.input_size, config.hidden_size, bias=False))
        self.wmh = weight_norm(nn.Linear(config.hidden_size, config.hidden_size, bias=False))
        self.wx = weight_norm(nn.Linear(config.input_size, project_size, bias=False))
        self.wh = weight_norm(nn.Linear(config.hidden_size, project_size, bias=True))

    def forward(self, inputs, state):
        h_prev, c_prev = state
        m = self.wmx(inputs) * self.wmh(h_prev)
        z = self.wx(inputs) + self.wh(m)
        i, f, o, u = torch.chunk(z, 4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        c = f * c_prev + i * u
        h = o * torch.tanh(c)
        return h, c


class mLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlstm_cell = mLSTMCell(config)
        self.hidden_size = config.hidden_size

    def forward(self, inputs, state=None, mask=None):
        batch_size = inputs.size(0)
        seqlen = inputs.size(1)
        if mask is None:
            mask = torch.ones(batch_size, seqlen, 1, dtype=inputs.dtype, device=inputs.device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)
        if state is None:
            zeros = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            state = zeros, zeros
        steps = []
        for seq in range(seqlen):
            prev = state
            seq_input = inputs[:, (seq), :]
            hx, cx = self.mlstm_cell(seq_input, state)
            seqmask = mask[:, (seq)]
            hx = seqmask * hx + (1 - seqmask) * prev[0]
            cx = seqmask * cx + (1 - seqmask) * prev[1]
            state = hx, cx
            steps.append(hx)
        return torch.stack(steps, 1), (hx, cx)


WEIGHTS_NAME = 'pytorch_model.bin'


class ProteinModel(nn.Module):
    """ Base class for all models.

        :class:`~ProteinModel` takes care of storing the configuration of
        the models and handles methods for loading/downloading/saving models as well as a
        few methods commons to all models to (i) resize the input embeddings and (ii) prune
        heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~ProteinConfig`
              to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names`
              (string) as keys and `url` (string) of associated pretrained weights as values.

            - ``base_model_prefix``: a string indicating the attribute associated to the
              base model in derived classes of the same architecture adding modules on top
              of the base model.
    """
    config_class: typing.Type[ProteinConfig] = ProteinConfig
    pretrained_model_archive_map: typing.Dict[str, str] = {}
    base_model_prefix = ''

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, ProteinConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `ProteinConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if
            new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings
        self.init_weights(new_embeddings)
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if
            new_num_tokens != config.vocab_size. Take care of tying weights embeddings
            afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add
                newly initialized vectors at the end. Reducing the size will remove vectors
                from the end. If not provided or None: does nothing and just returns a
                pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        if hasattr(self, 'tie_weights'):
            self.tie_weights()
        return model_embeds

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        self.apply(self._init_weights)
        if getattr(self.config, 'pruned_heads', False):
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and
                    associated values being the list of heads to prune in said layer
                    (list of `int`).
        """
        base_model = getattr(self, self.base_model_prefix, self)
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~ProteinModel.from_pretrained`
            ` class method.
        """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.config.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()``
        (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that
        the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used
        by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache
                  or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using
                  :func:`~ProteinModel.save_pretrained`,
                  e.g.: ``./my_model_directory/``.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's
                ``__init__`` method

            config: (`optional`) instance of a class derived from
                :class:`~ProteinConfig`: Configuration for the model to
                use instead of an automatically loaded configuation. Configuration can be
                automatically loaded when:

                - the model is a model provided by the library (loaded with the
                  ``shortcut-name`` string of a pretrained model), or
                - the model was saved using
                  :func:`~ProteinModel.save_pretrained` and is reloaded
                  by suppling the save directory.
                - the model is loaded by suppling a local directory as
                  ``pretrained_model_name_or_path`` and a configuration JSON file named
                  `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state
                dictionary loaded from saved weights file. This option can be used if you
                want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~ProteinModel.save_pretrained` and
                :func:`~ProteinModel.from_pretrained` is not a
                simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys,
                unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and
                initiate the model. (e.g. ``output_attention=True``). Behave differently
                depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be
                  directly passed to the underlying model's ``__init__`` method (we assume
                  all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the
                  configuration class initialization function
                  (:func:`~ProteinConfig.from_pretrained`). Each key of
                  ``kwargs`` that corresponds to a configuration attribute will be used to
                  override said attribute with the supplied ``kwargs`` value. Remaining keys
                  that do not correspond to any configuration attribute will be passed to the
                  underlying model's ``__init__`` function.

        Examples::

            # Download model and configuration from S3 and cache.
            model = ProteinBertModel.from_pretrained('bert-base-uncased')
            # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = ProteinBertModel.from_pretrained('./test/saved_model/')
            # Update configuration during loading
            model = ProteinBertModel.from_pretrained('bert-base-uncased', output_attention=True)
            assert model.config.output_attention == True

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, *model_args, cache_dir=cache_dir, return_unused_kwargs=True, **kwargs)
        else:
            model_kwargs = kwargs
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained weights.".format(archive_file))
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(cls.pretrained_model_archive_map.keys()), archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info('loading weights file {}'.format(archive_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
        model = cls(config, *model_args, **model_kwargs)
        if state_dict is None:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        model_to_load = model
        if cls.base_model_prefix not in (None, ''):
            if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                model_to_load = getattr(model, cls.base_model_prefix)
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        model.eval()
        if output_loading_info:
            loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'error_msgs': error_msgs}
            return model, loading_info
        return model


class SimpleMLP(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float=0.0):
        super().__init__()
        self.main = nn.Sequential(weight_norm(nn.Linear(in_dim, hid_dim), dim=None), nn.ReLU(), nn.Dropout(dropout, inplace=True), weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


class SimpleConv(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float=0.0):
        super().__init__()
        self.main = nn.Sequential(nn.BatchNorm1d(in_dim), weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None), nn.ReLU(), nn.Dropout(dropout, inplace=True), weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


def accuracy(logits, labels, ignore_index: int=-100):
    with torch.no_grad():
        valid_mask = labels != ignore_index
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


class Accuracy(nn.Module):

    def __init__(self, ignore_index: int=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


class PredictionHeadTransform(nn.Module):

    def __init__(self, hidden_size: int, hidden_act: typing.Union[str, typing.Callable]='gelu', layer_norm_eps: float=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int, hidden_act: typing.Union[str, typing.Callable]='gelu', layer_norm_eps: float=1e-12, ignore_index: int=-100):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = hidden_states,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = masked_lm_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs


class ValuePredictionHead(nn.Module):

    def __init__(self, hidden_size: int, dropout: float=0.0):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size, 512, 1, dropout)

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = value_pred,
        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs
        return outputs


class SequenceClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 512, num_labels)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = logits,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            metrics = {'accuracy': accuracy(logits, targets)}
            loss_and_metrics = classification_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs


class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int, ignore_index: int=-100):
        super().__init__()
        self.classify = SimpleConv(hidden_size, 512, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = sequence_logits,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            classification_loss = loss_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))
            acc_fct = Accuracy(ignore_index=self._ignore_index)
            metrics = {'accuracy': acc_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))}
            loss_and_metrics = classification_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs


class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-100):
        super().__init__()
        self.predict = nn.Sequential(nn.Dropout(), nn.Linear(2 * hidden_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs, sequence_lengths, targets=None):
        prod = inputs[:, :, (None), :] * inputs[:, (None), :, :]
        diff = inputs[:, :, (None), :] - inputs[:, (None), :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()
        outputs = prediction,
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(prediction.view(-1, 2), targets.view(-1))
            metrics = {'precision_at_l5': self.compute_precision_at_l5(sequence_lengths, prediction, targets)}
            loss_and_metrics = contact_loss, metrics
            outputs = (loss_and_metrics,) + outputs
        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= (y_ind - x_ind >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, (1)]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Accuracy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (PairwiseContactPredictionHead,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ProteinBertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ProteinBertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ProteinBertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (ProteinLSTMLayer,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ProteinResNetPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SequenceClassificationHead,
     lambda: ([], {'hidden_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequenceToSequenceClassificationHead,
     lambda: ([], {'hidden_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SimpleConv,
     lambda: ([], {'in_dim': 4, 'hid_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SimpleMLP,
     lambda: ([], {'in_dim': 4, 'hid_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ValuePredictionHead,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (mLSTM,
     lambda: ([], {'config': _mock_config(hidden_size=4, input_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_songlab_cal_tape(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

