import sys
_module = sys.modules[__name__]
del sys
doc_utils = _module
conf = _module
FastModel = _module
fastHan = _module
BertCharParser = _module
CharParser = _module
UserDict = _module
model = _module
bert = _module
bert_encoder_theseus = _module
finetune_dataloader = _module
model = _module
old_fastNLP_bert = _module
utils = _module
setup = _module
test = _module
core = _module
test_fastHan = _module

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


import re


import torch


import numpy as np


from torch import nn


from torch.nn import functional as F


import collections


import warnings


from itertools import chain


import copy


import math


import random


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


from torch.nn.parameter import Parameter


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    return word_embeddings


class BertCharParser(nn.Module):

    def __init__(self, app_index, vector_size, num_label, rnn_layers=3, arc_mlp_size=500, label_mlp_size=100, dropout=0.3, use_greedy_infer=False):
        super().__init__()
        self.parser = CharBiaffineParser(vector_size, num_label, rnn_layers, arc_mlp_size, label_mlp_size, dropout, use_greedy_infer)
        self.app_index = app_index

    def forward(self, feats, seq_lens, char_heads, char_labels):
        res_dict = self.parser(feats, seq_lens, gold_heads=char_heads)
        arc_pred = res_dict['arc_pred']
        label_pred = res_dict['label_pred']
        masks = res_dict['mask']
        loss = self.parser.loss(arc_pred, label_pred, char_heads, char_labels, masks)
        return {'loss': loss}

    def predict(self, feats, seq_lens):
        res = self.parser(feats, seq_lens, gold_heads=None)
        output = {}
        output['head_preds'] = res.pop('head_pred')
        size = res['label_pred'].size()
        res['label_pred'] = res['label_pred'].reshape(-1, size[-1])
        res['label_pred'][:, self.app_index] = -float('inf')
        res['label_pred'] = res['label_pred'].reshape(size)
        _, label_pred = res.pop('label_pred').max(2)
        output['label_preds'] = label_pred
        return output


class CharParser(nn.Module):

    def __init__(self, char_vocab_size, emb_dim, bigram_vocab_size, trigram_vocab_size, num_label, rnn_layers=3, rnn_hidden_size=400, arc_mlp_size=500, label_mlp_size=100, dropout=0.3, encoder='var-lstm', use_greedy_infer=False, app_index=0, pre_chars_embed=None, pre_bigrams_embed=None, pre_trigrams_embed=None):
        super().__init__()
        self.parser = CharBiaffineParser(char_vocab_size, emb_dim, bigram_vocab_size, trigram_vocab_size, num_label, rnn_layers, rnn_hidden_size, arc_mlp_size, label_mlp_size, dropout, encoder, use_greedy_infer, app_index, pre_chars_embed=pre_chars_embed, pre_bigrams_embed=pre_bigrams_embed, pre_trigrams_embed=pre_trigrams_embed)

    def forward(self, chars, bigrams, trigrams, seq_lens, char_heads, char_labels, pre_chars=None, pre_bigrams=None, pre_trigrams=None):
        res_dict = self.parser(chars, bigrams, trigrams, seq_lens, gold_heads=char_heads, pre_chars=pre_chars, pre_bigrams=pre_bigrams, pre_trigrams=pre_trigrams)
        arc_pred = res_dict['arc_pred']
        label_pred = res_dict['label_pred']
        masks = res_dict['mask']
        loss = self.parser.loss(arc_pred, label_pred, char_heads, char_labels, masks)
        return {'loss': loss}

    def predict(self, chars, bigrams, trigrams, seq_lens, pre_chars=None, pre_bigrams=None, pre_trigrams=None):
        res = self.parser(chars, bigrams, trigrams, seq_lens, gold_heads=None, pre_chars=pre_chars, pre_bigrams=pre_bigrams, pre_trigrams=pre_trigrams)
        output = {}
        output['head_preds'] = res.pop('head_pred')
        _, label_pred = res.pop('label_pred').max(2)
        output['label_preds'] = label_pred
        return output


BERT_KEY_RENAME_MAP_1 = {'gamma': 'weight', 'beta': 'bias', 'distilbert.embeddings': 'bert.embeddings', 'distilbert.transformer': 'bert.encoder'}


BERT_KEY_RENAME_MAP_2 = {'q_lin': 'self.query', 'k_lin': 'self.key', 'v_lin': 'self.value', 'out_lin': 'output.dense', 'sa_layer_norm': 'attention.output.LayerNorm', 'ffn.lin1': 'intermediate.dense', 'ffn.lin2': 'output.dense', 'output_layer_norm': 'output.LayerNorm'}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12):
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
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str):
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
            self.layer_norm_eps = layer_norm_eps
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


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        return context_layer


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
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
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, layers_cut=-1):
        all_encoder_layers = []
        layer_now = 0
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            layer_now += 1
            if layers_cut > 0 and layer_now >= layers_cut:
                break
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DistilBertEmbeddings(nn.Module):

    def __init__(self, config):
        super(DistilBertEmbeddings, self).__init__()

        def create_sinusoidal_embeddings(n_pos, dim, out):
            position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
            out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
            out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            out.detach_()
            out.requires_grad = False
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings, dim=config.hidden_size, out=self.position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.
        token_type_ids: no used.
        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def _get_bert_dir(model_dir_or_name: str='en-base-uncased'):
    if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
        model_url = _get_embedding_url('bert', model_dir_or_name.lower())
        None
        model_dir = cached_path(model_url, name='embedding')
    elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
        model_dir = os.path.abspath(os.path.expanduser(model_dir_or_name))
    else:
        logger.error(f'Cannot recognize BERT dir or name ``{model_dir_or_name}``.')
        raise ValueError(f'Cannot recognize BERT dir or name ``{model_dir_or_name}``.')
    return str(model_dir)


def _get_file_name_base_on_postfix(dir_path, postfix):
    """
    在dir_path中寻找后缀为postfix的文件.
    :param dir_path: str, 文件夹
    :param postfix: 形如".bin", ".json"等
    :return: str，文件的路径
    """
    files = list(filter(lambda filename: filename.endswith(postfix), os.listdir(os.path.join(dir_path))))
    if len(files) == 0:
        raise FileNotFoundError(f'There is no file endswith *{postfix} file in {dir_path}')
    elif len(files) > 1:
        raise FileExistsError(f'There are multiple *{postfix} files in {dir_path}')
    return os.path.join(dir_path, files[0])


class BertModel(nn.Module):
    """
    BERT(Bidirectional Embedding Representations from Transformers).

    用预训练权重矩阵来建立BERT模型::

        model = BertModel.from_pretrained(model_dir_or_name)

    用随机初始化权重矩阵来建立BERT模型::

        model = BertModel()

    :param int vocab_size: 词表大小，默认值为30522，为BERT English uncase版本的词表大小
    :param int hidden_size: 隐层大小，默认值为768，为BERT base的版本
    :param int num_hidden_layers: 隐藏层数，默认值为12，为BERT base的版本
    :param int num_attention_heads: 多头注意力头数，默认值为12，为BERT base的版本
    :param int intermediate_size: FFN隐藏层大小，默认值是3072，为BERT base的版本
    :param str hidden_act: FFN隐藏层激活函数，默认值为``gelu``
    :param float hidden_dropout_prob: FFN隐藏层dropout，默认值为0.1
    :param float attention_probs_dropout_prob: Attention层的dropout，默认值为0.1
    :param int max_position_embeddings: 最大的序列长度，默认值为512，
    :param int type_vocab_size: 最大segment数量，默认值为2
    :param int initializer_range: 初始化权重范围，默认值为0.02
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        super(BertModel, self).__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.model_type = 'bert'
        if hasattr(config, 'sinusoidal_pos_embds'):
            self.model_type = 'distilbert'
        elif 'model_type' in kwargs:
            self.model_type = kwargs['model_type'].lower()
        if self.model_type == 'distilbert':
            self.embeddings = DistilBertEmbeddings(config)
        else:
            self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if self.model_type != 'distilbert':
            self.pooler = BertPooler(config)
        else:
            logger.info('DistilBert has NOT pooler, will use hidden states of [CLS] token as pooled output.')
        self.apply(self.init_bert_weights)

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, layers_cut=-1):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, layers_cut=layers_cut)
        sequence_output = encoded_layers[-1]
        if self.model_type != 'distilbert':
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = sequence_output[:, 0]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        kwargs.pop('cache_dir', None)
        kwargs.pop('from_tf', None)
        pretrained_model_dir = _get_bert_dir(model_dir_or_name)
        config_file = _get_file_name_base_on_postfix(pretrained_model_dir, '.json')
        config = BertConfig.from_json_file(config_file)
        if state_dict is None:
            weights_path = _get_file_name_base_on_postfix(pretrained_model_dir, '.bin')
            state_dict = torch.load(weights_path, map_location='cpu')
        else:
            logger.error(f'Cannot load parameters through `state_dict` variable.')
            raise RuntimeError(f'Cannot load parameters through `state_dict` variable.')
        model_type = 'BERT'
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            for key_name in BERT_KEY_RENAME_MAP_1:
                if key_name in key:
                    new_key = key.replace(key_name, BERT_KEY_RENAME_MAP_1[key_name])
                    if 'distilbert' in key:
                        model_type = 'DistilBert'
                    break
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            for key_name in BERT_KEY_RENAME_MAP_2:
                if key_name in key:
                    new_key = key.replace(key_name, BERT_KEY_RENAME_MAP_2[key_name])
                    break
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        model = cls(config, *inputs, model_type=model_type, **kwargs)
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
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.warning('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.warning('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        logger.info(f'Load pre-trained {model_type} parameters from file {weights_path}.')
        return model


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <= 96 or cp >= 123 and cp <= 126:
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or cp >= 131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or cp >= 177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or cp >= 63744 and cp <= 64255 or cp >= 194560 and cp <= 195103:
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 65533 or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


VOCAB_NAME = 'vocab.txt'


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        if len(output_tokens) == 0:
            return [self.unk_token]
        return output_tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1000000000000.0)

    def _reinit_on_new_vocab(self, vocab):
        """
        在load bert之后，可能会对vocab进行重新排列。重新排列之后调用这个函数重新初始化与vocab相关的性质

        :param vocab:
        :return:
        """
        self.vocab = vocab
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning('Token indices sequence length is longer than the specified maximum  sequence length for this BERT model ({} > {}). Running this sequence through BERT will result in indexing errors'.format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        else:
            vocab_file = vocab_path
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning('Saving vocabulary to {}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!'.format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        """
        给定模型的名字或者路径，直接读取vocab.
        """
        model_dir = _get_bert_dir(model_dir_or_name)
        pretrained_model_name_or_path = _get_file_name_base_on_postfix(model_dir, '.txt')
        logger.info('loading vocabulary file {}'.format(pretrained_model_name_or_path))
        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_position_embeddings', int(1000000000000.0)), max_len)
        tokenizer = cls(pretrained_model_name_or_path, *inputs, **kwargs)
        return tokenizer


class _WordPieceBertModel(nn.Module):
    """
    这个模块用于直接计算word_piece的结果.

    """

    def __init__(self, model_dir_or_name: str, layers: str='-1', pooled_cls: bool=False):
        super().__init__()
        self.tokenzier = BertTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = BertModel.from_pretrained(model_dir_or_name)
        encoder_layer_number = len(self.encoder.encoder.layer)
        self.layers = list(map(int, layers.split(',')))
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f'The layer index:{layer} is out of scope for a bert model with {encoder_layer_number} layers.'
            else:
                assert layer < encoder_layer_number, f'The layer index:{layer} is out of scope for a bert model with {encoder_layer_number} layers.'
        self._cls_index = self.tokenzier.vocab['[CLS]']
        self._sep_index = self.tokenzier.vocab['[SEP]']
        self._wordpiece_unknown_index = self.tokenzier.vocab['[UNK]']
        self._wordpiece_pad_index = self.tokenzier.vocab['[PAD]']
        self.pooled_cls = pooled_cls

    def index_dataset(self, *datasets, field_name, add_cls_sep=True):
        """
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input。如果首尾不是
            [CLS]与[SEP]会在首尾额外加入[CLS]与[SEP], 且将word_pieces这一列的pad value设置为了bert的pad value。

        :param datasets: DataSet对象
        :param field_name: 基于哪一列index
        :return:
        """

        def convert_words_to_word_pieces(words):
            word_pieces = []
            for word in words:
                tokens = self.tokenzier.wordpiece_tokenizer.tokenize(word)
                word_piece_ids = self.tokenzier.convert_tokens_to_ids(tokens)
                word_pieces.extend(word_piece_ids)
            if add_cls_sep:
                if word_pieces[0] != self._cls_index:
                    word_pieces.insert(0, self._cls_index)
                if word_pieces[-1] != self._sep_index:
                    word_pieces.insert(-1, self._sep_index)
            return word_pieces
        for index, dataset in enumerate(datasets):
            try:
                dataset.apply_field(convert_words_to_word_pieces, field_name=field_name, new_field_name='word_pieces', is_input=True)
                dataset.set_pad_val('word_pieces', self._wordpiece_pad_index)
            except Exception as e:
                logger.error(f'Exception happens when processing the {index} dataset.')
                raise e

    def forward(self, word_pieces, token_type_ids=None):
        """

        :param word_pieces: torch.LongTensor, batch_size x max_len
        :param token_type_ids: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        batch_size, max_len = word_pieces.size()
        attn_masks = word_pieces.ne(self._wordpiece_pad_index)
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks, output_all_encoded_layers=True)
        outputs = bert_outputs[0].new_zeros((len(self.layers), batch_size, max_len, bert_outputs[0].size(-1)))
        for l_index, l in enumerate(self.layers):
            bert_output = bert_outputs[l]
            if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                bert_output[:, 0] = pooled_cls
            outputs[l_index] = bert_output
        return outputs


class BertWordPieceEncoder(nn.Module):
    """
    读取bert模型，读取之后调用index_dataset方法在dataset中生成word_pieces这一列。
    """

    def __init__(self, model_dir_or_name: str='en-base-uncased', layers: str='-1', pooled_cls: bool=False, word_dropout=0, dropout=0, requires_grad: bool=True):
        """
        
        :param str model_dir_or_name: 模型所在目录或者模型的名称。默认值为 ``en-base-uncased``
        :param str layers: 最终结果中的表示。以','隔开层数，可以以负数去索引倒数几层
        :param bool pooled_cls: 返回的句子开头的[CLS]是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取
            [CLS]做预测，一般该值为True。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool requires_grad: 是否需要gradient。
        """
        super().__init__()
        self.model = _WordPieceBertModel(model_dir_or_name=model_dir_or_name, layers=layers, pooled_cls=pooled_cls)
        self._sep_index = self.model._sep_index
        self._wordpiece_pad_index = self.model._wordpiece_pad_index
        self._wordpiece_unk_index = self.model._wordpiece_unknown_index
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad
        self.word_dropout = word_dropout
        self.dropout_layer = nn.Dropout(dropout)

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def embedding_dim(self):
        return self._embed_size

    @property
    def num_embedding(self):
        return self.model.encoder.config.vocab_size

    def index_datasets(self, *datasets, field_name, add_cls_sep=True):
        """
        使用bert的tokenizer新生成word_pieces列加入到datasets中，并将他们设置为input,且将word_pieces这一列的pad value设置为了
        bert的pad value。

        :param ~fastNLP.DataSet datasets: DataSet对象
        :param str field_name: 基于哪一列的内容生成word_pieces列。这一列中每个数据应该是List[str]的形式。
        :param bool add_cls_sep: 如果首尾不是[CLS]与[SEP]会在首尾额外加入[CLS]与[SEP]。
        :return:
        """
        self.model.index_dataset(*datasets, field_name=field_name, add_cls_sep=add_cls_sep)

    def forward(self, word_pieces, token_type_ids=None):
        """
        计算words的bert embedding表示。传入的words中应该自行包含[CLS]与[SEP]的tag。

        :param words: batch_size x max_len
        :param token_type_ids: batch_size x max_len, 用于区分前一句和后一句话. 如果不传入，则自动生成(大部分情况，都不需要输入),
            第一个[SEP]及之前为0, 第二个[SEP]及到第一个[SEP]之间为1; 第三个[SEP]及到第二个[SEP]之间为0，依次往后推。
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        with torch.no_grad():
            sep_mask = word_pieces.eq(self._sep_index)
            if token_type_ids is None:
                sep_mask_cumsum = sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():
                    token_type_ids = token_type_ids.eq(0).long()
        word_pieces = self.drop_word(word_pieces)
        outputs = self.model(word_pieces, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)
        return self.dropout_layer(outputs)

    def drop_word(self, words):
        """
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                if self._word_sep_index:
                    sep_mask = words.eq(self._wordpiece_unk_index)
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)
                pad_mask = words.ne(self._wordpiece_pad_index)
                mask = pad_mask.__and__(mask)
                words = words.masked_fill(mask, self._word_unk_index)
                if self._word_sep_index:
                    words.masked_fill_(sep_mask, self._wordpiece_unk_index)
        return words


class CharModel(nn.Module):

    def __init__(self, embed, label_vocab, pos_idx=31, Parsing_rnn_layers=3, Parsing_arc_mlp_size=500, Parsing_label_mlp_size=100, Parsing_use_greedy_infer=False, encoding_type='bmeso', embedding_dim=768, dropout=0.1, use_pos_embedding=True, use_average=True):
        super().__init__()
        self.embed = embed
        self.use_pos_embedding = use_pos_embedding
        self.use_average = use_average
        self.label_vocab = label_vocab
        self.pos_idx = pos_idx
        self.user_dict_weight = 0.05
        embedding_dim_1 = 512
        embedding_dim_2 = 256
        self.layers_map = {'CWS': '-1', 'POS': '-1', 'Parsing': '-1', 'NER': '-1'}
        self.ner_linear = nn.Linear(embedding_dim, len(label_vocab['NER']))
        trans = allowed_transitions(label_vocab['NER'], encoding_type='bmeso', include_start_end=True)
        self.ner_crf = ConditionalRandomField(len(label_vocab['NER']), include_start_end_trans=True, allowed_transitions=trans)
        self.biaffine_parser = BertCharParser(app_index=self.label_vocab['Parsing'].to_index('APP'), vector_size=768, num_label=len(label_vocab['Parsing']), rnn_layers=Parsing_rnn_layers, arc_mlp_size=Parsing_arc_mlp_size, label_mlp_size=Parsing_label_mlp_size, dropout=dropout, use_greedy_infer=Parsing_use_greedy_infer)
        if self.use_pos_embedding:
            self.pos_embedding = nn.Embedding(len(self.label_vocab['pos']), embedding_dim, padding_idx=0)
        self.loss = CrossEntropyLoss(padding_idx=0)
        self.cws_mlp = MLP([embedding_dim, embedding_dim_1, embedding_dim_2, len(label_vocab['CWS'])], 'relu', output_activation=None)
        trans = allowed_transitions(label_vocab['CWS'], include_start_end=True)
        self.cws_crf = ConditionalRandomField(len(label_vocab['CWS']), include_start_end_trans=True, allowed_transitions=trans)
        self.pos_mlp = MLP([embedding_dim, embedding_dim_1, embedding_dim_2, len(label_vocab['POS'])], 'relu', output_activation=None)
        trans = allowed_transitions(label_vocab['POS'], include_start_end=True)
        self.pos_crf = ConditionalRandomField(len(label_vocab['POS']), include_start_end_trans=True, allowed_transitions=trans)

    def _generate_embedding(self, feats, word_lens, seq_len, pos):
        device = feats.device
        new_feats = []
        batch_size = feats.size()[0]
        sentence_length = feats.size()[1]
        if self.use_average == False:
            for i in range(batch_size):
                new_feats.append(torch.index_select(feats[i], 0, word_lens[i]))
            new_feats = torch.stack(new_feats, 0)
        else:
            for i in range(batch_size):
                feats_for_one_sample = []
                for j in range(word_lens.size()[1]):
                    if word_lens[i][j] == 0 and j != 0:
                        feats_for_one_word = torch.zeros(feats.size()[-1])
                    else:
                        if j == word_lens.size()[1] - 1 or word_lens[i][j + 1] == 0:
                            index = range(word_lens[i][j], seq_len[i])
                        else:
                            index = range(word_lens[i][j], word_lens[i][j + 1])
                        index = torch.tensor(index)
                        index = index
                        feats_for_one_word = torch.index_select(feats[i], 0, index)
                        feats_for_one_word = torch.mean(feats_for_one_word, dim=0)
                        feats_for_one_word = feats_for_one_word
                    feats_for_one_sample.append(feats_for_one_word)
                feats_for_one_sample = torch.stack(feats_for_one_sample, dim=0)
                new_feats.append(feats_for_one_sample)
            new_feats = torch.stack(new_feats, 0)
        if self.use_pos_embedding:
            pos_feats = self.pos_embedding(pos)
            new_feats = new_feats + pos_feats
        return new_feats

    def _generate_from_pos(self, paths, seq_len):
        device = paths.device
        word_lens = []
        batch_size = paths.size()[0]
        new_seq_len = []
        batch_pos = []
        for i in range(batch_size):
            word_len = []
            pos = []
            for j in range(seq_len[i]):
                tag = paths[i][j]
                tag = self.label_vocab['POS'].to_word(int(tag))
                if tag.startswith('<'):
                    continue
                tag1, tag2 = tag.split('-')
                tag2 = self.label_vocab['pos'].to_index(tag2)
                if tag1 == 'S' or tag1 == 'B':
                    word_len.append(j)
                    pos.append(tag2)
            if len(pos) == 1:
                word_len.append(seq_len[i] - 1)
                pos.append(tag2)
            new_seq_len.append(len(pos))
            word_lens.append(word_len)
            batch_pos.append(pos)
        max_len = max(new_seq_len)
        for i in range(batch_size):
            word_lens[i] = word_lens[i] + [0] * (max_len - new_seq_len[i])
            batch_pos[i] = batch_pos[i] + [0] * (max_len - new_seq_len[i])
        word_lens = torch.tensor(word_lens, device=device)
        batch_pos = torch.tensor(batch_pos, device=device)
        new_seq_len = torch.tensor(new_seq_len, device=device)
        return word_lens, batch_pos, new_seq_len

    def _decode_parsing(self, dep_head, dep_label, seq_len, seq_len_for_wordlist, word_lens):
        device = dep_head.device
        heads = []
        labels = []
        batch_size = dep_head.size()[0]
        app_index = self.label_vocab['Parsing'].to_index('APP')
        max_len = seq_len.max()
        for i in range(batch_size):
            head = list(range(1, seq_len[i] + 1))
            label = [app_index] * int(seq_len[i])
            head[0] = 0
            for j in range(1, seq_len_for_wordlist[i]):
                if j + 1 == seq_len_for_wordlist[i]:
                    idx = seq_len[i] - 1
                else:
                    idx = word_lens[i][j + 1] - 1
                label[idx] = int(dep_label[i][j])
                root = dep_head[i][j]
                if root >= seq_len_for_wordlist[i] - 1:
                    head[idx] = int(seq_len[i] - 1)
                else:
                    try:
                        head[idx] = int(word_lens[i][root + 1] - 1)
                    except:
                        None
            head = head + [0] * int(max_len - seq_len[i])
            label = label + [0] * int(max_len - seq_len[i])
            heads.append(head)
            labels.append(label)
        heads = torch.tensor(heads, device=device)
        labels = torch.tensor(labels, device=device)
        return heads, labels

    def forward(self, chars, seq_len, task_class, target, seq_len_for_wordlist=None, dep_head=None, dep_label=None, pos=None, word_lens=None):
        task = task_class[0]
        self.current_task = task
        mask = chars.ne(0)
        layers = self.layers_map[task]
        feats = self.embed(chars, layers)
        if task == 'Parsing':
            parsing_feats = self._generate_embedding(feats, word_lens, seq_len, pos)
            loss_parsing = self.biaffine_parser(parsing_feats, seq_len_for_wordlist, dep_head, dep_label)
            return loss_parsing
        if task == 'NER':
            feats = F.relu(self.ner_linear(feats))
            logits = F.log_softmax(feats, dim=-1)
            loss = self.ner_crf(logits, target, mask)
            return {'loss': loss}
        if task == 'CWS':
            feats = self.cws_mlp(feats)
            logits = F.log_softmax(feats, dim=-1)
            loss = self.cws_crf(logits, target, mask)
            return {'loss': loss}
        if task == 'POS':
            feats = self.pos_mlp(feats)
            logits = F.log_softmax(feats, dim=-1)
            loss = self.pos_crf(logits, target, mask)
            return {'loss': loss}

    def __get_ud_diff(self, task, feats, tag_seqs):
        diff = torch.max(feats, dim=2)[0] - torch.mean(feats, dim=2)
        diff = diff.unsqueeze(dim=-1)
        diff = diff.expand(-1, -1, len(self.label_vocab[task]))
        diff = tag_seqs * diff * self.user_dict_weight
        return diff

    def predict(self, chars, seq_len, task_class, tag_seqs=None):
        task = task_class[0]
        mask = chars.ne(0)
        layers = self.layers_map[task]
        feats = self.embed(chars, layers)
        if task == 'Parsing':
            for sample in chars:
                sample[0] = self.pos_idx
            pos_feats = self.embed(chars, self.layers_map['POS'])
            pos_feats = self.pos_mlp(pos_feats)
            if tag_seqs is not None:
                diff = self.__get_ud_diff('POS', feats, tag_seqs)
                pos_feats = pos_feats + diff
            logits = F.log_softmax(pos_feats, dim=-1)
            paths, _ = self.pos_crf.viterbi_decode(logits, mask)
            word_lens, batch_pos, seq_len_for_wordlist = self._generate_from_pos(paths, seq_len)
            parsing_feats = self._generate_embedding(feats, word_lens, seq_len, batch_pos)
            answer = self.biaffine_parser.predict(parsing_feats, seq_len_for_wordlist)
            head_preds = answer['head_preds']
            label_preds = answer['label_preds']
            heads, labels = self._decode_parsing(head_preds, label_preds, seq_len, seq_len_for_wordlist, word_lens)
            return {'head_preds': heads, 'label_preds': labels, 'pred': paths}
        if task == 'CWS':
            feats = self.cws_mlp(feats)
            if tag_seqs is not None:
                diff = self.__get_ud_diff(task, feats, tag_seqs)
                feats = feats + diff
            logits = F.log_softmax(feats, dim=-1)
            paths, _ = self.cws_crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        if task == 'POS':
            feats = self.pos_mlp(feats)
            if tag_seqs is not None:
                diff = self.__get_ud_diff(task, feats, tag_seqs)
                feats = feats + diff
            logits = F.log_softmax(feats, dim=-1)
            paths, _ = self.pos_crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        if task == 'NER':
            feats = F.relu(self.ner_linear(feats))
            if tag_seqs is not None:
                diff = self.__get_ud_diff(task, feats, tag_seqs)
                feats = feats + diff
            logits = F.log_softmax(feats, dim=-1)
            paths, _ = self.ner_crf.viterbi_decode(logits, mask)
            return {'pred': paths}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertLayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_fastnlp_fastHan(_paritybench_base):
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

