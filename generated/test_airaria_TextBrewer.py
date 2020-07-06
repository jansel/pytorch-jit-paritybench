import sys
_module = sys.modules[__name__]
del sys
conf = _module
cmrc2018_evaluate = _module
config = _module
matches = _module
modeling = _module
optimization = _module
processing = _module
pytorch_pretrained_bert = _module
convert_gpt2_checkpoint_to_pytorch = _module
convert_openai_checkpoint_to_pytorch = _module
convert_tf_checkpoint_to_pytorch = _module
convert_transfo_xl_checkpoint_to_pytorch = _module
file_utils = _module
modeling = _module
modeling_gpt2 = _module
modeling_openai = _module
modeling_transfo_xl = _module
modeling_transfo_xl_utilities = _module
my_modeling = _module
optimization = _module
optimization_openai = _module
tokenization = _module
tokenization_gpt2 = _module
tokenization_openai = _module
tokenization_transfo_xl = _module
train_eval = _module
utils = _module
run_ner = _module
utils_ner = _module
modeling = _module
optimization = _module
convert_gpt2_checkpoint_to_pytorch = _module
convert_openai_checkpoint_to_pytorch = _module
convert_tf_checkpoint_to_pytorch = _module
convert_transfo_xl_checkpoint_to_pytorch = _module
modeling = _module
modeling_gpt2 = _module
modeling_openai = _module
modeling_transfo_xl = _module
modeling_transfo_xl_utilities = _module
my_modeling = _module
optimization = _module
optimization_openai = _module
tokenization_transfo_xl = _module
utils = _module
utils_glue = _module
distill = _module
setup = _module
textbrewer = _module
compatibility = _module
configurations = _module
data_utils = _module
distillation = _module
distiller_basic = _module
distiller_general = _module
distiller_multitask = _module
distiller_multiteacher = _module
distiller_train = _module
distiller_utils = _module
distillers = _module
losses = _module
presets = _module
projections = _module
schedulers = _module
snippet = _module
utils = _module

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


import logging


import random


import numpy as np


import torch


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from functools import partial


from torch import nn


from torch.nn import CrossEntropyLoss


from torch.nn import BCEWithLogitsLoss


import torch.nn.functional as F


import math


from torch.optim import Optimizer


from torch.nn.utils import clip_grad_norm_


import re


import tensorflow as tf


import copy


import collections


import torch.nn as nn


from torch.nn.parameter import Parameter


from collections import defaultdict


from torch.optim.optimizer import required


import abc


from collections import Counter


from collections import OrderedDict


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DistributedSampler


from torch.utils.data import Dataset


from typing import Union


from typing import List


from typing import Optional


from typing import Dict


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
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


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_score = config.output_score
        self.output_sum = config.output_sum

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
        if self.output_score is True:
            attention_probs_sum = attention_scores
        else:
            attention_probs_sum = attention_probs
        if self.output_sum is True:
            attention_probs_sum = attention_probs_sum.sum(dim=1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs_sum


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, att_layer = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, att_layer


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, att_layer = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, att_layer


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, output_all_attention_layers=False):
        all_encoder_layers = [hidden_states]
        all_attention_layers = [None]
        for idx, layer_module in enumerate(self.layer):
            hidden_states, att_layer = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            if output_all_attention_layers:
                all_attention_layers.append(att_layer)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_attention_layers


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


BERT_CONFIG_NAME = 'bert_config.json'


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

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


CONFIG_NAME = 'config.json'


PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz', 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz', 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz', 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz'}


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


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


logger = logging.getLogger('Distillation')


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
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
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
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
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


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        None
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    None
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        None
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m', 'global_step'] for n in name):
            None
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    None
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        None
        pointer.data = torch.from_numpy(array)
    return model


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info('loading archive file {}'.format(archive_file))
        else:
            logger.info('loading archive file {} from cache at {}'.format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info('extracting archive file {} to temp dir {}'.format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
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
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, output_score=False, output_sum=False):
        super(BertModel, self).__init__(config)
        config.output_score = output_score
        config.output_sum = output_sum
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, output_all_attention_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, att_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, output_all_attention_layers=output_all_attention_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, att_layers


def initializer_builder(std):
    _std = std

    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_weights


class BertForQA(nn.Module):

    def __init__(self, config):
        super(BertForQA, self).__init__()
        self.bert = BertModel(config, output_score=True, output_sum=1)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.cls_outputs = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)
        self.num_heads = config.num_attention_heads

    def forward(self, input_ids, token_type_ids, attention_mask, doc_mask, start_positions=None, end_positions=None, start_logits_T=None, end_logits_T=None, attention_probs_sum_layer=None, attention_probs_sum_T=None):
        if attention_probs_sum_layer is not None:
            sequence_output, pooled_output, all_attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, output_attention_layer=[attention_probs_sum_layer])
            attention_probs_sum = all_attention_probs_sum[attention_probs_sum_layer]
        else:
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        output_for_cls = self.dropout(pooled_output)
        span_logits = self.qa_outputs(sequence_output)
        cls_logits = self.cls_outputs(output_for_cls).squeeze(-1)
        doc_mask[:, (0)] = 0
        span_logits = span_logits + (1.0 - doc_mask.unsqueeze(-1)) * -10000.0
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_logits_T is not None or start_positions is not None:
            total_loss = 0
            att_loss = None
            if start_logits_T is not None:
                temp = Conf.args.temperature
                temp_2 = temp * temp
                start_logits_T /= temp
                end_logits_T /= temp
                start_logits /= temp
                end_logits /= temp
                start_prob_T = F.softmax(start_logits_T, dim=-1)
                end_prob_T = F.softmax(end_logits_T, dim=-1)
                ce_loss = -(start_prob_T * F.log_softmax(start_logits, dim=-1) + end_prob_T * F.log_softmax(end_logits, dim=-1)).sum(dim=-1)
                ce_loss = ce_loss.mean()
                total_loss += ce_loss
                if attention_probs_sum_T is not None:
                    attention_probs_sum_T = attention_probs_sum_T / self.num_heads
                    attention_probs_sum = attention_probs_sum / self.num_heads
                    attention_probs_sum_T = F.softmax(attention_probs_sum_T, dim=-1)
                    att_loss = -((attention_probs_sum_T * F.log_softmax(attention_probs_sum, dim=-1)).sum(dim=-1) * attention_mask).sum() / attention_mask.sum() * Conf.args.att_loss_weight
                    total_loss += att_loss
            if start_positions is not None:
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                    end_positions = end_positions.squeeze(-1)
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                is_noans = (start_positions == 0).float()
                loss_fct_span = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
                start_loss = (loss_fct_span(start_logits, start_positions) * (1 - is_noans)).mean()
                end_loss = (loss_fct_span(end_logits, end_positions) * (1 - is_noans)).mean()
                total_loss += (start_loss + end_loss) / 2
            return total_loss, att_loss
        elif attention_probs_sum_layer is not None:
            return start_logits, end_logits, cls_logits, attention_probs_sum
        else:
            return start_logits, end_logits, cls_logits


class BertForQASimple(nn.Module):

    def __init__(self, config, args):
        super(BertForQASimple, self).__init__()
        self.output_encoded_layers = args.output_encoded_layers == 'true'
        self.output_attention_layers = args.output_attention_layers == 'true'
        self.bert = BertModel(config, output_score=args.output_att_score == 'true', output_sum=args.output_att_sum == 'true')
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.cls_outputs = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, token_type_ids, attention_mask, doc_mask, start_positions=None, end_positions=None):
        sequence_output, pooled_output, attention_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=self.output_encoded_layers, output_all_attention_layers=self.output_attention_layers)
        if self.output_encoded_layers is True:
            span_logits = self.qa_outputs(sequence_output[-1])
        else:
            span_logits = self.qa_outputs(sequence_output)
        doc_mask[:, (0)] = 0
        span_logits = span_logits + (1.0 - doc_mask.unsqueeze(-1)) * -10000.0
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None:
            total_loss = 0
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_noans = (start_positions == 0).float()
            loss_fct_span = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            start_loss = (loss_fct_span(start_logits, start_positions) * (1 - is_noans)).mean()
            end_loss = (loss_fct_span(end_logits, end_positions) * (1 - is_noans)).mean()
            total_loss += (start_loss + end_loss) / 2
            return start_logits, end_logits, sequence_output, attention_output, total_loss
        else:
            return start_logits, end_logits


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class Conv1D(nn.Module):

    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.weight = Parameter(w)
            self.bias = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1000000000.0 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


ACT_FNS = {'relu': nn.ReLU, 'swish': swish, 'gelu': gelu}


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2MultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(GPT2MultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        self.linear = nn.Linear(config.n_embd, 1)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        return multiple_choice_logits


class GPT2Config(object):
    """Configuration class to store the configuration of a `GPT2Model`.
    """

    def __init__(self, vocab_size_or_config_json_file=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, layer_norm_epsilon=1e-05, initializer_range=0.02):
        """Constructs GPT2Config.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_ctx = n_ctx
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPT2Config` from a Python dictionary of parameters."""
        config = GPT2Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPT2Config` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

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


PRETRAINED_CONFIG_ARCHIVE_MAP = {'transfo-xl-wt103': 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json'}


def load_tf_weights_in_gpt2(model, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        None
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    None
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        None
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    for name, array in zip(names, arrays):
        name = name[6:]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                l = re.split('(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'w' or l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'wpe' or l[0] == 'wte':
                pointer = getattr(pointer, l[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        None
        pointer.data = torch.from_numpy(array)
    return model


class GPT2PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `GPT2Config`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def set_tied(self):
        pass

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs):
        """
        Instantiate a GPT2PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `gpt2`
                - a path or url to a pretrained model archive containing:
                    . `gpt2_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a GPT2Model instance
                - a path or url to a pretrained model archive containing:
                    . `gpt2_config.json` a configuration file for the model
                    . a TensorFlow checkpoint with trained weights
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific GPT class
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find files {} and {} at this path or url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), pretrained_model_name_or_path, archive_file, config_file))
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info('loading weights file {}'.format(archive_file))
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        config = GPT2Config.from_json_file(resolved_config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            return load_tf_weights_in_gpt2(model, resolved_archive_file)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if key.endswith('.g'):
                new_key = key[:-2] + '.weight'
            elif key.endswith('.b'):
                new_key = key[:-2] + '.bias'
            elif key.endswith('.w'):
                new_key = key[:-2] + '.weight'
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
        start_model = model
        if hasattr(model, 'transformer') and all(not s.startswith('transformer.') for s in state_dict.keys()):
            start_model = model.transformer
        load(start_model, prefix='')
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        model.set_tied()
        return model


class GPT2Model(GPT2PreTrainedModel):
    """OpenAI GPT-2 model ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs a tuple consisting of:
        `hidden_states`: the encoded-hidden-states at the top of the model
            as a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
            (or more generally [d_1, ..., d_n, hidden_size] were d_1 ... d_n are the dimension of input_ids)
        `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
            torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2Model(config)
    hidden_states, presents = model(input_ids)
    ```
    """

    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` is not `None`:
            Outputs the language modeling loss.
        else a tuple:
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, config.vocab_size]
                (or more generally [d_1, ..., d_n, config.vocab_size] were d_1 ... d_n are the dimension of input_ids)
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2LMHeadModel(config)
    lm_logits, presents = model(input_ids)
    ```
    """

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        return lm_logits, presents


class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    """OpenAI GPT-2 model with a Language Modeling and a Multiple Choice head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
            indices selected in the range [0, config.vocab_size[
        `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
            which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with indices selected in [-1, 0, ..., config.vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., config.vocab_size]
        `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` and `multiple_choice_labels` are not `None`:
            Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
        else: a tuple with
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, config.vocab_size]
            `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
    mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2LMHeadModel(config)
    lm_logits, multiple_choice_logits, presents = model(input_ids, mc_token_ids)
    ```
    """

    def __init__(self, config):
        super(GPT2DoubleHeadsModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.multiple_choice_head = GPT2MultipleChoiceHead(config)
        self.apply(self.init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if losses:
            return losses
        return lm_logits, mc_logits, presents


class OpenAIGPTLMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(OpenAIGPTLMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class OpenAIGPTMultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(OpenAIGPTMultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout2d(config.resid_pdrop)
        self.linear = nn.Linear(config.n_embd, 1)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        return multiple_choice_logits


class OpenAIGPTConfig(object):
    """Configuration class to store the configuration of a `OpenAIGPTModel`.
    """

    def __init__(self, vocab_size_or_config_json_file=40478, n_special=0, n_positions=512, n_ctx=512, n_embd=768, n_layer=12, n_head=12, afn='gelu', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02):
        """Constructs OpenAIGPTConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `OpenAIGPTModel` or a configuration json file.
            n_special: The number of special tokens to learn during fine-tuning ('[SEP]', '[CLF]', ...)
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            afn: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_special = n_special
            self.n_ctx = n_ctx
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.afn = afn
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @property
    def total_tokens_embeddings(self):
        return self.vocab_size + self.n_special

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `OpenAIGPTConfig` from a Python dictionary of parameters."""
        config = OpenAIGPTConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `OpenAIGPTConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

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


def load_tf_weights_in_openai_gpt(model, openai_checkpoint_folder_path):
    """ Load tf pre-trained weights in a pytorch model (from NumPy arrays here)
    """
    import re
    import numpy as np
    None
    names = json.load(open(openai_checkpoint_folder_path + '/parameters_names.json', 'r', encoding='utf-8'))
    shapes = json.load(open(openai_checkpoint_folder_path + '/params_shapes.json', 'r', encoding='utf-8'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(openai_checkpoint_folder_path + '/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params = [arr.squeeze() for arr in init_params]
    try:
        assert model.tokens_embed.weight.shape == init_params[1].shape
        assert model.positions_embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += model.tokens_embed.weight.shape, init_params[1].shape
        e.args += model.positions_embed.weight.shape, init_params[0].shape
        raise
    model.tokens_embed.weight.data = torch.from_numpy(init_params[1])
    model.positions_embed.weight.data = torch.from_numpy(init_params[0])
    names.pop(0)
    init_params.pop(0)
    init_params.pop(0)
    for name, array in zip(names, init_params):
        name = name[6:]
        assert name[-2:] == ':0'
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                l = re.split('(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'w':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        None
        pointer.data = torch.from_numpy(array)
    return model


class OpenAIGPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(OpenAIGPTPreTrainedModel, self).__init__()
        if not isinstance(config, OpenAIGPTConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `OpenAIGPTConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_num_special_tokens(self, num_special_tokens):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_special_tokens=None, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs):
        """
        Instantiate a OpenAIGPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `openai-gpt`
                - a path or url to a pretrained model archive containing:
                    . `openai_gpt_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a OpenAIGPTModel instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . a series of NumPy files containing OpenAI TensorFlow trained weights
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find files {} and {} at this path or url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), pretrained_model_name_or_path, archive_file, config_file))
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info('loading weights file {}'.format(archive_file))
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        config = OpenAIGPTConfig.from_json_file(resolved_config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            return load_tf_weights_in_openai_gpt(model, resolved_archive_file)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if key.endswith('.g'):
                new_key = key[:-2] + '.weight'
            elif key.endswith('.b'):
                new_key = key[:-2] + '.bias'
            elif key.endswith('.w'):
                new_key = key[:-2] + '.weight'
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
        start_model = model
        if hasattr(model, 'transformer') and all(not s.startswith('transformer.') for s in state_dict.keys()):
            start_model = model.transformer
        load(start_model, prefix='')
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        model.set_num_special_tokens(num_special_tokens if num_special_tokens is not None else config.n_special)
        return model


class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model ("Improving Language Understanding by Generative Pre-Training").

    OpenAI GPT use a single embedding matrix to store the word and special embeddings.
    Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
    Special tokens need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    The embeddings are ordered as follow in the token embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1]                  ______________________

    where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
        total_tokens_embeddings = config.vocab_size + config.n_special
    You should use the associate indices to index the embeddings.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, total_tokens_embeddings[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.

    Outputs:
        `hidden_states`: the encoded-hidden-states at the top of the model
            as a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
            (or more generally [d_1, ..., d_n, hidden_size] were d_1 ... d_n are the dimension of input_ids)

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTModel(config)
    hidden_states = model(input_ids)
    ```
    """

    def __init__(self, config):
        super(OpenAIGPTModel, self).__init__(config)
        num_tokens = config.vocab_size + config.n_special
        self.tokens_embed = nn.Embedding(num_tokens, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        """ Update input embeddings with new embedding matrice if needed """
        if self.config.n_special == num_special_tokens:
            return
        self.config.n_special = num_special_tokens
        old_embed = self.tokens_embed
        self.tokens_embed = nn.Embedding(self.config.total_tokens_embeddings, self.config.n_embd)
        self.tokens_embed
        self.init_weights(self.tokens_embed)
        self.tokens_embed.weight.data[:self.config.vocab_size, :] = old_embed.weight.data[:self.config.vocab_size, :]

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))
        inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        for block in self.h:
            hidden_states = block(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)


class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model with a Language Modeling head ("Improving Language Understanding by Generative Pre-Training").

    OpenAI GPT use a single embedding matrix to store the word and special embeddings.
    Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
    Special tokens need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    The embeddings are ordered as follow in the token embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1]                  ______________________

    where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
        total_tokens_embeddings = config.vocab_size + config.n_special
    You should use the associate indices to index the embeddings.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, total_tokens_embeddings[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `lm_labels` is not `None`:
            Outputs the language modeling loss.
        else:
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, total_tokens_embeddings]
                (or more generally [d_1, ..., d_n, total_tokens_embeddings] were d_1 ... d_n are the dimension of input_ids)

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTLMHeadModel(config)
    lm_logits = model(input_ids)
    ```
    """

    def __init__(self, config):
        super(OpenAIGPTLMHeadModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = OpenAIGPTLMHead(self.transformer.tokens_embed.weight, config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        """ Update input and output embeddings with new embedding matrice
            Make sure we are sharing the embeddings
        """
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.tokens_embed.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None):
        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            shift_logits = lm_logits[(...), :-1, :].contiguous()
            shift_labels = lm_labels[(...), 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        return lm_logits


class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model with a Language Modeling and a Multiple Choice head ("Improving Language Understanding by Generative Pre-Training").

    OpenAI GPT use a single embedding matrix to store the word and special embeddings.
    Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
    Special tokens need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    The embeddings are ordered as follow in the token embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1]                  ______________________

    where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
        total_tokens_embeddings = config.vocab_size + config.n_special
    You should use the associate indices to index the embeddings.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
            indices selected in the range [0, total_tokens_embeddings[
        `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
            which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with indices selected in [-1, 0, ..., total_tokens_embeddings]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., total_tokens_embeddings]
        `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `lm_labels` and `multiple_choice_labels` are not `None`:
            Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
        else: a tuple with
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, total_tokens_embeddings]
            `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
    mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTLMHeadModel(config)
    lm_logits, multiple_choice_logits = model(input_ids, mc_token_ids)
    ```
    """

    def __init__(self, config):
        super(OpenAIGPTDoubleHeadsModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = OpenAIGPTLMHead(self.transformer.tokens_embed.weight, config)
        self.multiple_choice_head = OpenAIGPTMultipleChoiceHead(config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        """ Update input and output embeddings with new embedding matrice
            Make sure we are sharing the embeddings
        """
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.tokens_embed.weight)

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None):
        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[(...), :-1, :].contiguous()
            shift_labels = lm_labels[(...), 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if losses:
            return losses
        return lm_logits, mc_logits


class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, (None), :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, (None), :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False, r_r_bias=None, r_w_bias=None):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def forward(self, h, attn_mask=None, mems=None):
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        if self.pre_lnorm:
            c = self.layer_norm(c)
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[(None), :, :, (None)], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, (None)], -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)
        return output


class RelMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, r_r_bias=None, r_w_bias=None):
        super(RelMultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])
        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)
        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)
        x = x_padded.masked_select(mask[:, :, (None), (None)]).view(qlen, klen, x.size(2), x.size(3))
        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, (None), (None)]
        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        rw_head_q = w_head_q + self.r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(attn_mask[(None), :, :, (None)], -1e+30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, (None)], -1e+30).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]
        rw_head_q = w_head_q + r_w_bias[None]
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        D_ = r_bias[(None), :, (None)]
        BD = self._rel_shift(B_ + D_)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[(None), :, :, (None)], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, (None)], -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class TransfoXLConfig(object):
    """Configuration class to store the configuration of a `TransfoXLModel`.
    """

    def __init__(self, vocab_size_or_config_json_file=267735, cutoffs=[20000, 40000, 200000], d_model=1024, d_embed=1024, n_head=16, d_head=64, d_inner=4096, div_val=4, pre_lnorm=False, n_layer=18, tgt_len=128, ext_len=0, mem_len=1600, clamp_len=1000, same_length=True, proj_share_all_but_first=True, attn_type=0, sample_softmax=-1, adaptive=True, tie_weight=True, dropout=0.1, dropatt=0.0, untie_r=True, init='normal', init_range=0.01, proj_init_std=0.01, init_std=0.02):
        """Constructs TransfoXLConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TransfoXLModel` or a configuration json file.
            cutoffs: cutoffs for the adaptive softmax
            d_model: Dimensionality of the model's hidden states.
            d_embed: Dimensionality of the embeddings
            d_head: Dimensionality of the model's heads.
            div_val: divident value for adapative input and softmax
            pre_lnorm: apply LayerNorm to the input instead of the output
            d_inner: Inner dimension in FF
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            tgt_len: number of tokens to predict
            ext_len: length of the extended context
            mem_len: length of the retained previous heads
            same_length: use the same attn length for all tokens
            proj_share_all_but_first: True to share all but first projs, False not to share.
            attn_type: attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
            clamp_len: use the same pos embeddings after clamp_len
            sample_softmax: number of samples in sampled softmax
            adaptive: use adaptive softmax
            tie_weight: tie the word embedding and softmax weights
            dropout: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            dropatt: The dropout ratio for the attention probabilities.
            untie_r: untie relative position biases           
            embd_pdrop: The dropout ratio for the embeddings.
            init: parameter initializer to use
            init_range: parameters initialized by U(-init_range, init_range).
            proj_init_std: parameters initialized by N(0, init_std)
            init_std: parameters initialized by N(0, init_std)
        """
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.n_token = vocab_size_or_config_json_file
            self.cutoffs = []
            self.cutoffs.extend(cutoffs)
            self.tie_weight = tie_weight
            if proj_share_all_but_first:
                self.tie_projs = [False] + [True] * len(self.cutoffs)
            else:
                self.tie_projs = [False] + [False] * len(self.cutoffs)
            self.d_model = d_model
            self.d_embed = d_embed
            self.d_head = d_head
            self.d_inner = d_inner
            self.div_val = div_val
            self.pre_lnorm = pre_lnorm
            self.n_layer = n_layer
            self.n_head = n_head
            self.tgt_len = tgt_len
            self.ext_len = ext_len
            self.mem_len = mem_len
            self.same_length = same_length
            self.attn_type = attn_type
            self.clamp_len = clamp_len
            self.sample_softmax = sample_softmax
            self.adaptive = adaptive
            self.dropout = dropout
            self.dropatt = dropatt
            self.untie_r = untie_r
            self.init = init
            self.init_range = init_range
            self.proj_init_std = proj_init_std
            self.init_std = init_std
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransfoXLConfig` from a Python dictionary of parameters."""
        config = TransfoXLConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransfoXLConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

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


def build_tf_to_pytorch_map(model, config):
    """ A map of modules from TF to PyTorch.
        This time I use a map to keep the PyTorch model as identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}
    if hasattr(model, 'transformer'):
        tf_to_pt_map.update({'transformer/adaptive_softmax/cutoff_0/cluster_W': model.crit.cluster_weight, 'transformer/adaptive_softmax/cutoff_0/cluster_b': model.crit.cluster_bias})
        for i, (out_l, proj_l, tie_proj) in enumerate(zip(model.crit.out_layers, model.crit.out_projs, config.tie_projs)):
            layer_str = 'transformer/adaptive_softmax/cutoff_%d/' % i
            if config.tie_weight:
                tf_to_pt_map.update({(layer_str + 'b'): out_l.bias})
            else:
                raise NotImplementedError
                tf_to_pt_map.update({(layer_str + 'lookup_table'): out_l.weight, (layer_str + 'b'): out_l.bias})
            if not tie_proj:
                tf_to_pt_map.update({(layer_str + 'proj'): proj_l})
        model = model.transformer
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers, model.word_emb.emb_projs)):
        layer_str = 'transformer/adaptive_embed/cutoff_%d/' % i
        tf_to_pt_map.update({(layer_str + 'lookup_table'): embed_l.weight, (layer_str + 'proj_W'): proj_l})
    for i, b in enumerate(model.layers):
        layer_str = 'transformer/layer_%d/' % i
        tf_to_pt_map.update({(layer_str + 'rel_attn/LayerNorm/gamma'): b.dec_attn.layer_norm.weight, (layer_str + 'rel_attn/LayerNorm/beta'): b.dec_attn.layer_norm.bias, (layer_str + 'rel_attn/o/kernel'): b.dec_attn.o_net.weight, (layer_str + 'rel_attn/qkv/kernel'): b.dec_attn.qkv_net.weight, (layer_str + 'rel_attn/r/kernel'): b.dec_attn.r_net.weight, (layer_str + 'ff/LayerNorm/gamma'): b.pos_ff.layer_norm.weight, (layer_str + 'ff/LayerNorm/beta'): b.pos_ff.layer_norm.bias, (layer_str + 'ff/layer_1/kernel'): b.pos_ff.CoreNet[0].weight, (layer_str + 'ff/layer_1/bias'): b.pos_ff.CoreNet[0].bias, (layer_str + 'ff/layer_2/kernel'): b.pos_ff.CoreNet[3].weight, (layer_str + 'ff/layer_2/bias'): b.pos_ff.CoreNet[3].bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({'transformer/r_r_bias': r_r_list, 'transformer/r_w_bias': r_w_list})
    return tf_to_pt_map


def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        None
        raise
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        None
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array
    for name, pointer in tf_to_pt_map.items():
        assert name in tf_weights
        array = tf_weights[name]
        if 'kernel' in name or 'proj' in name:
            array = np.transpose(array)
        if ('r_r_bias' in name or 'r_w_bias' in name) and len(pointer) > 1:
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += p_i.shape, arr_i.shape
                    raise
                None
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += pointer.shape, array.shape
                raise
            None
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    None
    return model


class TransfoXLPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(TransfoXLPreTrainedModel, self).__init__()
        if not isinstance(config, TransfoXLConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `TransfoXLConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_weight(self, weight):
        if self.config.init == 'uniform':
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == 'normal':
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def init_weights(self, m):
        """ Initialize the weights.
        """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                self.init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                self.init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)

    def set_num_special_tokens(self, num_special_tokens):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs):
        """
        Instantiate a TransfoXLPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `transfo-xl`
                - a path or url to a pretrained model archive containing:
                    . `transfo_xl_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a TransfoXLModel instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find files {} and {} at this path or url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), pretrained_model_name_or_path, archive_file, config_file))
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info('loading weights file {}'.format(archive_file))
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        config = TransfoXLConfig.from_json_file(resolved_config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            return load_tf_weights_in_transfo_xl(model, config, pretrained_model_name_or_path)
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
        if not hasattr(model, 'transformer') and any(s.startswith('transformer.') for s in state_dict.keys()):
            start_prefix = 'transformer.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        return model


class TransfoXLModel(TransfoXLPreTrainedModel):
    """Transformer XL model ("Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context").

    Transformer XL use a relative positioning (with sinusiodal patterns) and adaptive softmax inputs which means that:
    - you don't need to specify positioning embeddings indices
    - the tokens in the vocabulary have to be sorted to decreasing frequency.

    Params:
        config: a TransfoXLConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the token indices selected in the range [0, self.config.n_token[
        `mems`: optional memomry of hidden states from previous forward passes
            as a list (num layers) of hidden states at the entry of each layer
            each hidden states has shape [self.config.mem_len, bsz, self.config.d_model]
            Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `target`
    Outputs:
        A tuple of (last_hidden_state, new_mems)
        `last_hidden_state`: the encoded-hidden-states at the top of the model
            as a torch.FloatTensor of size [batch_size, sequence_length, self.config.d_model]
        `new_mems`: list (num layers) of updated mem states at the entry of each layer
            each mem state is a torch.FloatTensor of size [self.config.mem_len, batch_size, self.config.d_model]
            Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `target`

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_ids_next = torch.LongTensor([[53, 21, 1], [64, 23, 100]])

    config = TransfoXLConfig()

    model = TransfoXLModel(config)
    last_hidden_state, new_mems = model(input_ids)

    # Another time on input_ids_next using the memory:
    last_hidden_state, new_mems = model(input_ids_next, new_mems)
    ```
    """

    def __init__(self, config):
        super(TransfoXLModel, self).__init__(config)
        self.n_token = config.n_token
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.word_emb = AdaptiveEmbedding(config.n_token, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val)
        self.drop = nn.Dropout(config.dropout)
        self.n_layer = config.n_layer
        self.tgt_len = config.tgt_len
        self.mem_len = config.mem_len
        self.ext_len = config.ext_len
        self.max_klen = config.tgt_len + config.ext_len + config.mem_len
        self.attn_type = config.attn_type
        if not config.untie_r:
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.layers = nn.ModuleList()
        if config.attn_type == 0:
            for i in range(config.n_layer):
                self.layers.append(RelPartialLearnableDecoderLayer(config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout, tgt_len=config.tgt_len, ext_len=config.ext_len, mem_len=config.mem_len, dropatt=config.dropatt, pre_lnorm=config.pre_lnorm, r_w_bias=None if config.untie_r else self.r_w_bias, r_r_bias=None if config.untie_r else self.r_r_bias))
        elif config.attn_type == 1:
            for i in range(config.n_layer):
                self.layers.append(RelLearnableDecoderLayer(config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout, tgt_len=config.tgt_len, ext_len=config.ext_len, mem_len=config.mem_len, dropatt=config.dropatt, pre_lnorm=config.pre_lnorm, r_w_bias=None if config.untie_r else self.r_w_bias, r_r_bias=None if config.untie_r else self.r_r_bias))
        elif config.attn_type in [2, 3]:
            for i in range(config.n_layer):
                self.layers.append(DecoderLayer(config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout, dropatt=config.dropatt, pre_lnorm=config.pre_lnorm, r_w_bias=None if config.untie_r else self.r_w_bias, r_r_bias=None if config.untie_r else self.r_r_bias))
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len
        if self.attn_type == 0:
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
        self.apply(self.init_weights)

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, data):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(self.mem_len, data.size(1), self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, (None)]
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, (None)]
        hids = []
        if self.attn_type == 0:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, dec_attn_mask=dec_attn_mask, mems=mems_i)
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i], r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        elif self.attn_type == 2:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb + pos_emb[-qlen:])
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)
            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        return core_out, new_mems

    def forward(self, input_ids, mems=None):
        """ Params:
                input_ids :: [bsz, len]
                mems :: optional mems from previous forwar passes (or init_mems)
                    list (num layers) of mem states at the entry of each layer
                        shape :: [self.config.mem_len, bsz, self.config.d_model]
                    Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `target`
            Returns:
                tuple (last_hidden, new_mems) where:
                    new_mems: list (num layers) of mem states at the entry of each layer
                        shape :: [self.config.mem_len, bsz, self.config.d_model]
                    last_hidden: output of the last layer:
                        shape :: [bsz, len, self.config.d_model]
        """
        input_ids = input_ids.transpose(0, 1).contiguous()
        if mems is None:
            mems = self.init_mems(input_ids)
        last_hidden, new_mems = self._forward(input_ids, mems=mems)
        last_hidden = last_hidden.transpose(0, 1).contiguous()
        return last_hidden, new_mems


class LogUniformSampler(object):

    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            self.log_q = (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()
        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """
        n_sample = self.n_sample
        n_tries = 2 * n_sample
        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples
            true_log_probs = self.log_q[labels]
            samp_log_probs = self.log_q[neg_samples]
            return true_log_probs, samp_log_probs, neg_samples


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))
                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, target=None, keep_order=False):
        """
            Params:
                hidden :: [len*bsz x d_proj]
                target :: [len*bsz]
            Return:
                if target is None:
                    out :: [len*bsz] Negative log likelihood
                else:
                    out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary
            We could replace this implementation by the native PyTorch one
            if their's had an option to set bias on all clusters in the native one.
            here: https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        """
        if target is not None:
            target = target.view(-1)
            if hidden.size(0) != target.size(0):
                raise RuntimeError('Input and target should have the same size in the batch dimension.')
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            if target is not None:
                output = -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            else:
                output = F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            if target is None:
                out = hidden.new_empty((head_logit.size(0), self.n_token))
            else:
                out = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                if target is not None:
                    mask_i = (target >= l_idx) & (target < r_idx)
                    indices_i = mask_i.nonzero().squeeze()
                    if indices_i.numel() == 0:
                        continue
                    target_i = target.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = hidden.index_select(0, indices_i)
                else:
                    hidden_i = hidden
                if i == 0:
                    if target is not None:
                        logprob_i = head_logprob_i.gather(1, target_i[:, (None)]).squeeze(1)
                    else:
                        out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    if target is not None:
                        logprob_i = head_logprob_i[:, (cluster_prob_idx)] + tail_logprob_i.gather(1, target_i[:, (None)]).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, (cluster_prob_idx), (None)] + tail_logprob_i
                        out[:, l_idx:r_idx] = logprob_i
                if target is not None:
                    if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                        out.index_copy_(0, indices_i, -logprob_i)
                    else:
                        out[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
                    offset += logprob_i.size(0)
        return out

    def log_prob(self, hidden):
        """ Computes log probabilities for all :math:`n\\_classes`
        From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py
        Args:
            hidden (Tensor): a minibatch of examples
        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= n\\_classes`, where :math:`n\\_classes` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.
        Shape:
            - Input: :math:`(N, in\\_features)`
            - Output: :math:`(N, n\\_classes)`
        """
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            return F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            out = hidden.new_empty((head_logit.size(0), self.n_token))
            head_logprob = F.log_softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
                if i == 0:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob[:, (-i)] + tail_logprob_i
                    out[:, (start_idx), (stop_idx)] = logprob_i
            return out


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)
    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]
    hit = (labels[:, :, (None)] == neg_samples).detach()
    true_logits = torch.einsum('ijk,ijk->ij', [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl', [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e+30)
    logits = torch.cat([true_logits[:, :, (None)], sample_logits], -1)
    return logits


class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    """Transformer XL model ("Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context").

    This model add an (adaptive) softmax head on top of the TransfoXLModel

    Transformer XL use a relative positioning (with sinusiodal patterns) and adaptive softmax inputs which means that:
    - you don't need to specify positioning embeddings indices
    - the tokens in the vocabulary have to be sorted to decreasing frequency.

    Call self.tie_weights() if you update/load the weights of the transformer to keep the weights tied.

    Params:
        config: a TransfoXLConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the token indices selected in the range [0, self.config.n_token[
        `target`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with the target token indices selected in the range [0, self.config.n_token[
        `mems`: an optional memory of hidden states from previous forward passes
            as a list (num layers) of hidden states at the entry of each layer
            each hidden states has shape [self.config.mem_len, bsz, self.config.d_model]
            Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `target`

    Outputs:
        A tuple of (last_hidden_state, new_mems)
        `softmax_output`: output of the (adaptive) softmax:
            if target is None:
                Negative log likelihood of shape [batch_size, sequence_length] 
            else:
                log probabilities of tokens, shape [batch_size, sequence_length, n_tokens]
        `new_mems`: list (num layers) of updated mem states at the entry of each layer
            each mem state is a torch.FloatTensor of size [self.config.mem_len, batch_size, self.config.d_model]
            Note that the first two dimensions are transposed in `mems` with regards to `input_ids` and `target`

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_ids_next = torch.LongTensor([[53, 21, 1], [64, 23, 100]])

    config = TransfoXLConfig()

    model = TransfoXLModel(config)
    last_hidden_state, new_mems = model(input_ids)

    # Another time on input_ids_next using the memory:
    last_hidden_state, new_mems = model(input_ids_next, mems=new_mems)
    ```
    """

    def __init__(self, config):
        super(TransfoXLLMHeadModel, self).__init__(config)
        self.transformer = TransfoXLModel(config)
        self.sample_softmax = config.sample_softmax
        if config.sample_softmax > 0:
            self.out_layer = nn.Linear(config.d_model, config.n_token)
            self.sampler = LogUniformSampler(config.n_token, config.sample_softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(config.n_token, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Run this to be sure output and input (adaptive) softmax weights are tied """
        if self.sample_softmax > 0:
            if self.config.tie_weight:
                self.out_layer.weight = self.transformer.word_emb.weight
        else:
            if self.config.tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.transformer.word_emb.emb_layers[i].weight
            if self.config.tie_projs:
                for i, tie_proj in enumerate(self.config.tie_projs):
                    if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[0]
                    elif tie_proj and self.config.div_val != 1:
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[i]

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.transformer.reset_length(tgt_len, ext_len, mem_len)

    def init_mems(self, data):
        return self.transformer.init_mems(data)

    def forward(self, input_ids, target=None, mems=None):
        """ Params:
                input_ids :: [bsz, len]
                target :: [bsz, len]
            Returns:
                tuple(softmax_output, new_mems) where:
                    new_mems: list (num layers) of hidden states at the entry of each layer
                        shape :: [mem_len, bsz, self.config.d_model] :: Warning: shapes are transposed here w. regards to input_ids
                    softmax_output: output of the (adaptive) softmax:
                        if target is None:
                            Negative log likelihood of shape :: [bsz, len] 
                        else:
                            log probabilities of tokens, shape :: [bsz, len, n_tokens]
        """
        bsz = input_ids.size(0)
        tgt_len = input_ids.size(1)
        last_hidden, new_mems = self.transformer(input_ids, mems)
        pred_hid = last_hidden[:, -tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.config.tie_weight
            logit = sample_logits(self.transformer.word_emb, self.out_layer.bias, target, pred_hid, self.sampler)
            softmax_output = -F.log_softmax(logit, -1)[:, :, (0)]
        else:
            softmax_output = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target)
            if target is None:
                softmax_output = softmax_output.view(bsz, tgt_len, -1)
            else:
                softmax_output = softmax_output.view(bsz, tgt_len)
        return softmax_output, new_mems


class ALBertEncoder(nn.Module):

    def __init__(self, config):
        super(ALBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.skip_connection = config.skip_connection
        self.num_repetitions = config.num_repetitions
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, output_attention_layer=None):
        all_encoder_layers = []
        all_attention_probs_sum = []
        for idx_outer, layer_module in enumerate(self.layer):
            x0 = hidden_states
            for idx_inner in range(self.num_repetitions):
                idx = idx_inner + idx_outer * self.num_repetitions
                if idx_inner == 0 or self.skip_connection is False:
                    hidden_states, attention_probs_sum = layer_module(hidden_states, attention_mask)
                else:
                    hidden_states, attention_probs_sum = layer_module(hidden_states + x0, attention_mask)
                if output_attention_layer is not None and idx in output_attention_layer:
                    all_attention_probs_sum.append(attention_probs_sum)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_attention_probs_sum


class ALBertModel(BertPreTrainedModel):

    def __init__(self, config, output_score=False, output_sum=1, skip_connection=False, num_repetitions=1):
        super(ALBertModel, self).__init__(config)
        config.output_score = output_score
        config.output_sum = output_sum
        config.skip_connection = skip_connection
        config.num_repetitions = num_repetitions
        self.embeddings = BertEmbeddings(config)
        self.encoder = ALBertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, output_attention_layer=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, attention_probs_sum = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, output_attention_layer=output_attention_layer)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, attention_probs_sum


class BertForGLUE(nn.Module):

    def __init__(self, config, num_labels):
        super(BertForGLUE, self).__init__()
        self.num_labels = num_labels
        output_sum = None if Conf.args.output_sum < 0 else Conf.args.output_sum
        self.bert = BertModel(config, Conf.args.output_score, output_sum)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, logits_T=None, attention_probs_sum_layer=None, attention_probs_sum_T=None, hidden_match_layer=None, hidden_match_T=None):
        if hidden_match_layer is not None:
            sequence_output, pooled_output, attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, output_attention_layer=attention_probs_sum_layer)
            hidden_states = [sequence_output[i] for i in hidden_match_layer]
            sequence_output = sequence_output[-1]
        else:
            sequence_output, pooled_output, attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, output_attention_layer=attention_probs_sum_layer)
            hidden_states = []
        output_for_cls = self.dropout(pooled_output)
        logits = self.classifier(output_for_cls)
        if logits_T is not None or labels is not None or attention_probs_sum_T is not None:
            total_loss = 0
            hidden_losses = None
            att_losses = None
            if logits_T is not None and self.num_labels != 1:
                temp = Conf.args.temperature
                logits_T /= temp
                logits /= temp
                prob_T = F.softmax(logits_T, dim=-1)
                ce_loss = -(prob_T * F.log_softmax(logits, dim=-1)).sum(dim=-1)
                ce_loss = ce_loss.mean()
                total_loss += ce_loss
            if attention_probs_sum_T:
                if Conf.args.mask_inter == 1:
                    mask = attention_mask
                    valid_count = torch.pow(mask.sum(dim=1), 2).sum()
                    att_losses = [((F.mse_loss(attention_probs_sum[i], attention_probs_sum_T[i], reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count) for i in range(len(attention_probs_sum_T))]
                else:
                    att_losses = [F.mse_loss(attention_probs_sum[i], attention_probs_sum_T[i]) for i in range(len(attention_probs_sum_T))]
                att_loss = sum(att_losses) * Conf.args.att_loss_weight
                total_loss += att_loss
            if hidden_match_T:
                if Conf.args.mask_inter == 1:
                    mask = attention_mask
                    valid_count = mask.sum() * hidden_states[0].size(-1)
                    hidden_losses = [((F.mse_loss(hidden_states[i], hidden_match_T[i], reduction='none') * mask.unsqueeze(-1)).sum() / valid_count) for i in range(len(hidden_match_layer))]
                else:
                    hidden_losses = [F.mse_loss(hidden_states[i], hidden_match_T[i]) for i in range(len(hidden_match_layer))]
                hidden_loss = sum(hidden_losses) * Conf.args.hidden_loss_weight
                total_loss += hidden_loss
            if labels is not None:
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.view(-1), labels.view(-1))
                else:
                    loss = F.cross_entropy(logits, labels)
                total_loss += loss
            return total_loss, att_losses, hidden_losses
        elif attention_probs_sum_layer is not None or hidden_match_layer is not None:
            return logits, attention_probs_sum, hidden_states
        else:
            return logits, None


class BertForGLUESimple(nn.Module):

    def __init__(self, config, num_labels, args):
        super(BertForGLUESimple, self).__init__()
        self.num_labels = num_labels
        self.output_encoded_layers = args.output_encoded_layers == 'true'
        self.output_attention_layers = args.output_attention_layers == 'true'
        self.bert = BertModel(config, output_score=args.output_att_score == 'true', output_sum=args.output_att_sum == 'true')
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        sequence_output, pooled_output, attention_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=self.output_encoded_layers, output_all_attention_layers=self.output_attention_layers)
        output_for_cls = self.dropout(pooled_output)
        logits = self.classifier(output_for_cls)
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits, labels)
            return logits, sequence_output, attention_output, loss
        else:
            return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4, attn_pdrop=0.5, resid_pdrop=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, output_score=4, output_sum=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_airaria_TextBrewer(_paritybench_base):
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

