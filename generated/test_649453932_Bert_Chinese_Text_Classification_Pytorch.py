import sys
_module = sys.modules[__name__]
del sys
ERNIE = _module
bert = _module
bert_CNN = _module
bert_DPCNN = _module
bert_RCNN = _module
bert_RNN = _module
pytorch_pretrained = _module
__main__ = _module
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
optimization = _module
optimization_openai = _module
tokenization = _module
tokenization_gpt2 = _module
tokenization_openai = _module
tokenization_transfo_xl = _module
run = _module
train_eval = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


import copy


import logging


import math


from torch import nn


from torch.nn import CrossEntropyLoss


import collections


from torch.nn.parameter import Parameter


from collections import defaultdict


import numpy as np


from torch.optim import Optimizer


from torch.optim.optimizer import required


from torch.nn.utils import clip_grad_norm_


import abc


import time


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask,
            output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k,
            config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.
            filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask,
            output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.
            convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.
            hidden_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1
            ), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask,
            output_all_encoded_layers=False)
        x = encoder_out.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px
        return x


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.
            num_layers, bidirectional=True, batch_first=True, dropout=
            config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size,
            config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask,
            output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.
            num_layers, bidirectional=True, batch_first=True, dropout=
            config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask,
            output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, (-1), :])
        return out


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.
            hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
            config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=
            input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


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
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
        torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
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
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(
            config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask,
        output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


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


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
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
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.
            size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
            bert_model_embedding_weights)

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
        self.predictions = BertLMPredictionHead(config,
            bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


BERT_CONFIG_NAME = 'bert_config.json'


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=
        3072, hidden_act='gelu', hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, max_position_embeddings=512,
        type_vocab_size=2, initializer_range=0.02):
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
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[
            0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
                ) as reader:
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
            raise ValueError(
                'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
                )

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


PRETRAINED_MODEL_ARCHIVE_MAP = {'transfo-xl-wt103':
    'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-pytorch_model.bin'
    }


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


class Conv1D(nn.Module):

    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).
            view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 10000.0 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
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

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[
                1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


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
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 
            -1, -1, hidden_states.size(-1))
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        return multiple_choice_logits


class GPT2Config(object):
    """Configuration class to store the configuration of a `GPT2Model`.
    """

    def __init__(self, vocab_size_or_config_json_file=50257, n_positions=
        1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12,
        layer_norm_epsilon=1e-05, initializer_range=0.02):
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
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[
            0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
                ) as reader:
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
            raise ValueError(
                'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
                )

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


PRETRAINED_CONFIG_ARCHIVE_MAP = {'transfo-xl-wt103':
    'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json'
    }


def load_tf_weights_in_gpt2(model, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            'Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    print('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print('Loading TF weight {} with shape {}'.format(name, shape))
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
        print('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


class GPT2PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `GPT2Config`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'
                .format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def set_tied(self):
        pass

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.
                initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None,
        cache_dir=None, from_tf=False, *inputs, **kwargs):
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
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[
                pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[
                pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path,
                WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path,
                CONFIG_NAME)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=
                cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir
                )
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find files {} and {} at this path or url."
                .format(pretrained_model_name_or_path, ', '.join(
                PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                pretrained_model_name_or_path, archive_file, config_file))
            return None
        if (resolved_archive_file == archive_file and resolved_config_file ==
            config_file):
            logger.info('loading weights file {}'.format(archive_file))
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(
                archive_file, resolved_archive_file))
            logger.info('loading configuration file {} from cache at {}'.
                format(config_file, resolved_config_file))
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
            local_metadata = {} if metadata is None else metadata.get(prefix
                [:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_model = model
        if hasattr(model, 'transformer') and all(not s.startswith(
            'transformer.') for s in state_dict.keys()):
            start_model = model.transformer
        load(start_model, prefix='')
        if len(missing_keys) > 0:
            logger.info(
                'Weights of {} not initialized from pretrained model: {}'.
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.
                format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'
                .format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        model.set_tied()
        return model


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
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).
            view(1, 1, n_ctx, n_ctx))
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
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 
            -1, -1, hidden_states.size(-1))
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)
            ).transpose(1, 2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        return multiple_choice_logits


class OpenAIGPTConfig(object):
    """Configuration class to store the configuration of a `OpenAIGPTModel`.
    """

    def __init__(self, vocab_size_or_config_json_file=40478, n_special=0,
        n_positions=512, n_ctx=512, n_embd=768, n_layer=12, n_head=12, afn=
        'gelu', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        layer_norm_epsilon=1e-05, initializer_range=0.02):
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
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[
            0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
                ) as reader:
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
            raise ValueError(
                'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
                )

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
    print('Loading weights...')
    names = json.load(open(openai_checkpoint_folder_path +
        '/parameters_names.json', 'r', encoding='utf-8'))
    shapes = json.load(open(openai_checkpoint_folder_path +
        '/params_shapes.json', 'r', encoding='utf-8'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(openai_checkpoint_folder_path + '/params_{}.npy'
        .format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params,
        shapes)]
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
        print('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


class OpenAIGPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(OpenAIGPTPreTrainedModel, self).__init__()
        if not isinstance(config, OpenAIGPTConfig):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `OpenAIGPTConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'
                .format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.
                initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_num_special_tokens(self, num_special_tokens):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,
        num_special_tokens=None, state_dict=None, cache_dir=None, from_tf=
        False, *inputs, **kwargs):
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
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[
                pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[
                pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path,
                WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path,
                CONFIG_NAME)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=
                cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir
                )
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find files {} and {} at this path or url."
                .format(pretrained_model_name_or_path, ', '.join(
                PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                pretrained_model_name_or_path, archive_file, config_file))
            return None
        if (resolved_archive_file == archive_file and resolved_config_file ==
            config_file):
            logger.info('loading weights file {}'.format(archive_file))
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(
                archive_file, resolved_archive_file))
            logger.info('loading configuration file {} from cache at {}'.
                format(config_file, resolved_config_file))
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
            local_metadata = {} if metadata is None else metadata.get(prefix
                [:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_model = model
        if hasattr(model, 'transformer') and all(not s.startswith(
            'transformer.') for s in state_dict.keys()):
            start_model = model.transformer
        load(start_model, prefix='')
        if len(missing_keys) > 0:
            logger.info(
                'Weights of {} not initialized from pretrained model: {}'.
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.
                format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'
                .format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        model.set_num_special_tokens(num_special_tokens if 
            num_special_tokens is not None else config.n_special)
        return model


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
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(
            inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model),
            nn.Dropout(dropout))
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

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
        pre_lnorm=False, r_r_bias=None, r_w_bias=None):
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
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
                )
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
                )
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
                attn_score.masked_fill_(attn_mask[(None), :, :, (None)], -
                    float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, (None)], -float(
                    'inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.
            size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)
        return output


class RelMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len
        =None, ext_len=None, mem_len=None, pre_lnorm=False, r_r_bias=None,
        r_w_bias=None):
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
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
                )
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head)
                )
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
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(
                3)), device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)
        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)
        x = x_padded.masked_select(mask[:, :, (None), (None)]).view(qlen,
            klen, x.size(2), x.size(3))
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
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, (None), (
                None)]
        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class DecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **
            kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=
            kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
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
                attn_score.masked_fill_(attn_mask[(None), :, :, (None)], -
                    float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, (None)], -float(
                    'inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.
            size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class RelLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
            dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=
            kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None,
        mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias, attn_mask=
            dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head,
            bias=False)

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
                attn_score = attn_score.float().masked_fill(attn_mask[(None
                    ), :, :, (None)], -1e+30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :,
                    :, (None)], -1e+30).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.
            size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=
            kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
        sample_softmax=False):
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
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=
                sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj,
                    d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj,
                    d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=
                param.dtype, device=param.device)
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

    def __init__(self, vocab_size_or_config_json_file=267735, cutoffs=[
        20000, 40000, 200000], d_model=1024, d_embed=1024, n_head=16,
        d_head=64, d_inner=4096, div_val=4, pre_lnorm=False, n_layer=18,
        tgt_len=128, ext_len=0, mem_len=1600, clamp_len=1000, same_length=
        True, proj_share_all_but_first=True, attn_type=0, sample_softmax=-1,
        adaptive=True, tie_weight=True, dropout=0.1, dropatt=0.0, untie_r=
        True, init='normal', init_range=0.01, proj_init_std=0.01, init_std=0.02
        ):
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
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[
            0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
                ) as reader:
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
            raise ValueError(
                'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
                )

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
        tf_to_pt_map.update({
            'transformer/adaptive_softmax/cutoff_0/cluster_W': model.crit.
            cluster_weight,
            'transformer/adaptive_softmax/cutoff_0/cluster_b': model.crit.
            cluster_bias})
        for i, (out_l, proj_l, tie_proj) in enumerate(zip(model.crit.
            out_layers, model.crit.out_projs, config.tie_projs)):
            layer_str = 'transformer/adaptive_softmax/cutoff_%d/' % i
            if config.tie_weight:
                tf_to_pt_map.update({(layer_str + 'b'): out_l.bias})
            else:
                raise NotImplementedError
                tf_to_pt_map.update({(layer_str + 'lookup_table'): out_l.
                    weight, (layer_str + 'b'): out_l.bias})
            if not tie_proj:
                tf_to_pt_map.update({(layer_str + 'proj'): proj_l})
        model = model.transformer
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers,
        model.word_emb.emb_projs)):
        layer_str = 'transformer/adaptive_embed/cutoff_%d/' % i
        tf_to_pt_map.update({(layer_str + 'lookup_table'): embed_l.weight,
            (layer_str + 'proj_W'): proj_l})
    for i, b in enumerate(model.layers):
        layer_str = 'transformer/layer_%d/' % i
        tf_to_pt_map.update({(layer_str + 'rel_attn/LayerNorm/gamma'): b.
            dec_attn.layer_norm.weight, (layer_str +
            'rel_attn/LayerNorm/beta'): b.dec_attn.layer_norm.bias, (
            layer_str + 'rel_attn/o/kernel'): b.dec_attn.o_net.weight, (
            layer_str + 'rel_attn/qkv/kernel'): b.dec_attn.qkv_net.weight,
            (layer_str + 'rel_attn/r/kernel'): b.dec_attn.r_net.weight, (
            layer_str + 'ff/LayerNorm/gamma'): b.pos_ff.layer_norm.weight,
            (layer_str + 'ff/LayerNorm/beta'): b.pos_ff.layer_norm.bias, (
            layer_str + 'ff/layer_1/kernel'): b.pos_ff.CoreNet[0].weight, (
            layer_str + 'ff/layer_1/bias'): b.pos_ff.CoreNet[0].bias, (
            layer_str + 'ff/layer_2/kernel'): b.pos_ff.CoreNet[3].weight, (
            layer_str + 'ff/layer_2/bias'): b.pos_ff.CoreNet[3].bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({'transformer/r_r_bias': r_r_list,
        'transformer/r_w_bias': r_w_list})
    return tf_to_pt_map


def load_tf_weights_in_transfo_xl(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            'Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        print('Loading TF weight {} with shape {}'.format(name, shape))
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
                print('Initialize PyTorch weight {} for layer {}'.format(
                    name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += pointer.shape, array.shape
                raise
            print('Initialize PyTorch weight {}'.format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    print('Weights not copied to PyTorch model: {}'.format(', '.join(
        tf_weights.keys())))
    return model


class TransfoXLPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(TransfoXLPreTrainedModel, self).__init__()
        if not isinstance(config, TransfoXLConfig):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `TransfoXLConfig`. To create a model from a pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'
                .format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_weight(self, weight):
        if self.config.init == 'uniform':
            nn.init.uniform_(weight, -self.config.init_range, self.config.
                init_range)
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
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.
                            proj_init_std)
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
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.
                            proj_init_std)
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
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None,
        cache_dir=None, from_tf=False, *inputs, **kwargs):
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
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[
                pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[
                pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path,
                WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path,
                CONFIG_NAME)
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=
                cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir
                )
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find files {} and {} at this path or url."
                .format(pretrained_model_name_or_path, ', '.join(
                PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                pretrained_model_name_or_path, archive_file, config_file))
            return None
        if (resolved_archive_file == archive_file and resolved_config_file ==
            config_file):
            logger.info('loading weights file {}'.format(archive_file))
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(
                archive_file, resolved_archive_file))
            logger.info('loading configuration file {} from cache at {}'.
                format(config_file, resolved_config_file))
        config = TransfoXLConfig.from_json_file(resolved_config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            return load_tf_weights_in_transfo_xl(model, config,
                pretrained_model_name_or_path)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix
                [:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'transformer') and any(s.startswith(
            'transformer.') for s in state_dict.keys()):
            start_prefix = 'transformer.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                'Weights of {} not initialized from pretrained model: {}'.
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.
                format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'
                .format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        return model


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
        keep_order=False):
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
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters,
                self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj,
                        d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj,
                    d_emb_i)))
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
                raise RuntimeError(
                    'Input and target should have the same size in the batch dimension.'
                    )
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                self.out_layers[0].bias, self.out_projs[0])
            if target is not None:
                output = -F.log_softmax(logit, dim=-1).gather(1, target.
                    unsqueeze(1)).squeeze(1)
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
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0
                        )
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0
                ], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias,
                head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            if target is None:
                out = hidden.new_empty((head_logit.size(0), self.n_token))
            else:
                out = torch.zeros_like(target, dtype=hidden.dtype, device=
                    hidden.device)
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
                        logprob_i = head_logprob_i.gather(1, target_i[:, (
                            None)]).squeeze(1)
                    else:
                        out[:, :self.cutoffs[0]] = head_logprob[:, :self.
                            cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i
                        ], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden_i, weight_i,
                        bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    if target is not None:
                        logprob_i = head_logprob_i[:, (cluster_prob_idx)
                            ] + tail_logprob_i.gather(1, target_i[:, (None)]
                            ).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, (cluster_prob_idx), (None)
                            ] + tail_logprob_i
                        out[:, l_idx:r_idx] = logprob_i
                if target is not None:
                    if hasattr(self, 'keep_order'
                        ) and self.keep_order or keep_order:
                        out.index_copy_(0, indices_i, -logprob_i)
                    else:
                        out[offset:offset + logprob_i.size(0)].copy_(-logprob_i
                            )
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
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                self.out_layers[0].bias, self.out_projs[0])
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
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0
                        )
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0
                ], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias,
                head_proj)
            out = hidden.new_empty((head_logit.size(0), self.n_token))
            head_logprob = F.log_softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
                if i == 0:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]
                        ]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i
                        ], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden, weight_i,
                        bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob[:, (-i)] + tail_logprob_i
                    out[:, (start_idx), (stop_idx)] = logprob_i
            return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_649453932_Bert_Chinese_Text_Classification_Pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Attention(*[], **{'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4, attn_pdrop=0.5, resid_pdrop=0.5)}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(BertIntermediate(*[], **{'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=ReLU())}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BertOnlyNSPHead(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BertPooler(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(BertSelfAttention(*[], **{'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

