import sys
_module = sys.modules[__name__]
del sys
inference = _module
run_albert = _module
run_basic_bert = _module
run_cnn_base = _module
run_colbert = _module
run_fast_text = _module
run_nezha = _module
run_poly_encoder = _module
run_re2 = _module
run_siamese_rnn = _module
run_simcse = _module
run_bm25 = _module
run_tfidf = _module
run_tfidf_sklearn = _module
setup = _module
sim = _module
base = _module
bm25 = _module
lsh = _module
pytorch = _module
common = _module
layers = _module
modeling_albert = _module
modeling_bert = _module
modeling_char_cnn = _module
modeling_colbert = _module
modeling_fasttext = _module
modeling_nezha = _module
modeling_poly_encoder = _module
modeling_re2 = _module
modeling_siamese_rnn = _module
modeling_text_cnn = _module
modeling_text_vdcnn = _module
pipeline = _module
sif_usif = _module
tensorflow = _module
optimizers = _module
tf_idf = _module
tools = _module
data_format = _module
process_cipher_text = _module
process_ngram = _module
process_oov_data = _module
process_plain_text = _module
settings = _module
similarity = _module
tokenizer = _module
word2vec = _module

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


import torch.nn as nn


from typing import NoReturn


from typing import Any


import torch.nn.functional as F


import random


import time


from collections import OrderedDict


from itertools import repeat


import copy


import string


import torch as torch


import math


class BertConfig(object):
    """BertModel的配置"""

    def __init__(self, vocab_size: int, hidden_size: int, num_attention_heads: int, num_hidden_layers: int, intermediate_size: int, hidden_act: Any, embedding_size: int=None, attention_head_size: int=None, attention_key_size: int=None, max_position_embeddings: int=None, max_position: int=None, layer_norm_eps: float=1e-07, type_vocab_size: int=None, hidden_dropout_prob: float=None, attention_probs_dropout_prob: float=None, shared_segment_embeddings: bool=False, hierarchical_position: Any=False, initializer_range: float=None, use_relative_position: bool=False, max_relative_positions: int=None, **kwargs):
        """构建BertConfig
        :param vocab_size: 词表大小
        :param hidden_size: 隐藏层大小
        :param num_attention_heads: encoder中的attention层的注意力头数量
        :param num_hidden_layers: encoder的层数
        :param intermediate_size: 前馈神经网络层维度
        :param hidden_act: encoder和pool中的非线性激活函数
        :param embedding_size: 词嵌入大小
        :param attention_head_size: Attention中V的head_size
        :param attention_key_size: Attention中Q,K的head_size
        :param max_position_embeddings: 最大编码位置
        :param max_position: 绝对位置编码最大位置数
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param type_vocab_size: segment_ids的词典大小
        :param hidden_dropout_prob: embedding、encoder和pool层中的全连接层dropout
        :param attention_probs_dropout_prob: attention的dropout
        :param shared_segment_embeddings: segment是否共享token embedding
        :param hierarchical_position: 是否层次分解位置编码
        :param initializer_range: truncated_normal_initializer初始化方法的stdev
        :param use_relative_position: 是否使用相对位置编码
        :param max_relative_positions: 相对位置编码最大位置数
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.max_position_embeddings = max_position_embeddings
        self.max_position = max_position or self.max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.shared_segment_embeddings = shared_segment_embeddings
        self.hierarchical_position = hierarchical_position
        self.initializer_range = initializer_range
        self.use_relative_position = use_relative_position
        self.max_relative_positions = max_relative_positions

    @classmethod
    def from_dict(cls, json_obj):
        """从字典对象中构建BertConfig
        :param json_obj: 字典对象
        :return: BertConfig
        """
        bert_config = BertConfig(**json_obj)
        for key, value in json_obj.items():
            if key == 'relative_attention':
                bert_config.use_relative_position = value
            else:
                bert_config.__dict__[key] = value
        return bert_config

    @classmethod
    def from_json_file(cls, json_file_path: str):
        """从json文件中构建BertConfig
        :param json_file_path: JSON文件路径
        :return: BertConfig
        """
        with open(json_file_path, 'r', encoding='utf-8') as reader:
            return cls.from_dict(json_obj=json.load(reader))

    def to_dict(self) ->dict:
        """将实例序列化为字典"""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self) ->str:
        """将实例序列化为json字符串"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def truncated_normal_(mean: float=0.0, stddev: float=0.02) ->Any:
    """截尾正态分布
    :param mean: 均值
    :param stddev: 标准差
    """

    def _truncated_norm(tensor: torch.Tensor):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.detach().copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.detach().mul_(stddev).add_(mean)
            return tensor
    return _truncated_norm


class BertEmbeddings(nn.Module):
    """Bert Embedding
    """

    def __init__(self, hidden_size: int, embedding_size: int, hidden_dropout_prob: float=None, shared_segment_embeddings: bool=False, type_vocab_size: int=None, layer_norm_eps: float=None, initializer: Any=truncated_normal_()):
        """NEZHA Embedding
        :param hidden_size: 编码维度
        :param embedding_size: 词嵌入大小
        :param hidden_dropout_prob: Dropout比例
        :param shared_segment_embeddings: 若True，则segment跟token共用embedding
        :param type_vocab_size: segment总数目
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param initializer: Embedding的初始化器
        """
        super(BertEmbeddings, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.shared_segment_embeddings = shared_segment_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer = initializer
        if self.type_vocab_size > 0 and not self.shared_segment_embeddings:
            self.segment_embeddings = nn.Embedding(num_embeddings=self.type_vocab_size, embedding_dim=self.embedding_size)
            self.initializer(self.segment_embeddings.weight)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)
        if self.embedding_size != self.hidden_size:
            self.outputs_dense = nn.Linear(in_features=self.embedding_size, out_features=self.hidden_size)

    def forward(self, input_ids, segment_ids, token_embeddings):
        outputs = token_embeddings(input_ids)
        if self.type_vocab_size > 0:
            if self.shared_segment_embeddings:
                segment_outputs = token_embeddings(segment_ids)
            else:
                segment_outputs = self.segment_embeddings(segment_ids)
            outputs = outputs + segment_outputs
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        if self.embedding_size != self.hidden_size:
            outputs = self.outputs_dense(outputs)
        return outputs


def scaled_dot_product_attention(query: Any, key: Any, value: Any, batch_size: int, num_heads: int, attention_head_size: int, dropout: float, mask: Any=None, pos_type: str=None, pos_ids: Any=None) ->tuple:
    """点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param batch_size: batch size
    :param num_heads: head num
    :param attention_head_size: 分头之后维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    :param pos_type: 指定位置编码种类，现支持经典的相对位置编码: "typical_relation"
    :param pos_ids: 位置编码
    """
    attention_scores = torch.matmul(input=query, other=key.permute(0, 1, 3, 2))
    if pos_type == 'typical_relation':
        attention_scores = attention_scores + torch.einsum('bhjd,kjd->bhjk', query, pos_ids)
    attention_scores = attention_scores / torch.sqrt(input=torch.tensor(data=attention_head_size))
    if mask is not None:
        attention_scores += mask * -1000000000.0
    attention_weights = torch.softmax(input=attention_scores, dim=-1)
    attention_weights = nn.Dropout(p=dropout)(attention_weights)
    context_layer = torch.matmul(input=attention_weights, other=value)
    if pos_type == 'typical_relation':
        context_layer = context_layer + torch.einsum('bhjk,jkd->bhjd', attention_weights, pos_ids)
    context_layer = context_layer.permute(0, 2, 1, 3)
    context_layer = torch.reshape(input=context_layer, shape=(batch_size, -1, attention_head_size * num_heads))
    return context_layer, attention_weights


class BertSelfAttention(nn.Module):
    """定义Self-Attention
    """

    def __init__(self, num_heads: int, head_size: int, batch_size: int, attention_dropout: float, use_bias: bool=True, key_size: int=None, hidden_size: int=None, initializer: Any=nn.init.xavier_normal_, pos_type: str=None):
        """
        :param num_heads: 注意力头数
        :param head_size: Attention中V的head_size
        :param batch_size: batch size
        :param attention_dropout: Attention矩阵的Dropout比例
        :param use_bias: 是否加上偏差项
        :param key_size: Attention中Q,K的head_size
        :param hidden_size: 编码维度
        :param initializer: 初始化器
        :param pos_type: 指定位置编码种类，现支持经典的相对位置编码: "typical_relation"
        """
        super(BertSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.batch_size = batch_size
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.key_size = key_size if key_size is not None else head_size
        self.hidden_size = hidden_size if hidden_size is not None else num_heads * head_size
        self.initializer = initializer
        self.pos_type = pos_type
        self.query_dense = nn.Linear(in_features=self.hidden_size, out_features=self.key_size * self.num_heads, bias=self.use_bias)
        self.initializer(self.query_dense.weight)
        self.key_dense = nn.Linear(in_features=self.hidden_size, out_features=self.key_size * self.num_heads, bias=self.use_bias)
        self.initializer(self.key_dense.weight)
        self.value_dense = nn.Linear(in_features=self.hidden_size, out_features=self.head_size * self.num_heads, bias=self.use_bias)
        self.initializer(self.value_dense.weight)
        self.output_dense = nn.Linear(in_features=self.head_size * self.num_heads, out_features=self.hidden_size, bias=self.use_bias)
        self.initializer(self.output_dense.weight)

    def transpose_for_scores(self, input_tensor: Any, head_size: int):
        """分拆最后一个维度到 (num_heads, depth)
        :param input_tensor: 输入
        :param head_size: 每个注意力头维数
        """
        input_tensor = torch.reshape(input=input_tensor, shape=(self.batch_size, -1, self.num_heads, head_size))
        return input_tensor.permute(0, 2, 1, 3)

    def forward(self, inputs):
        pos_ids = None
        if self.pos_type == 'typical_relation':
            query, key, value, pos_ids, mask = inputs
        else:
            query, key, value, mask = inputs
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.transpose_for_scores(input_tensor=query, head_size=self.key_size)
        key = self.transpose_for_scores(input_tensor=key, head_size=self.key_size)
        value = self.transpose_for_scores(input_tensor=value, head_size=self.head_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(query=query, key=key, value=value, batch_size=self.batch_size, num_heads=self.num_heads, attention_head_size=self.head_size, dropout=self.attention_dropout, mask=mask, pos_type=self.pos_type, pos_ids=pos_ids)
        attn_outputs = self.output_dense(scaled_attention)
        return attn_outputs, attention_weights


def linear_act(x):
    return x


def swish(x):
    return x * torch.sigmoid(x)


def get_activation(identifier: str):
    """获取激活函数
    """
    activations = {'gelu': F.gelu, 'glu': F.glu, 'relu': F.relu, 'elu': F.elu, 'hardtanh': F.hardtanh, 'relu6': F.relu6, 'selu': F.selu, 'leaky_relu': F.leaky_relu, 'sigmoid': F.sigmoid, 'gumbel_softmax': F.gumbel_softmax, 'tanh': torch.tanh, 'softmax': F.softmax, 'swish': swish, 'linear': linear_act}
    if identifier not in activations:
        raise ValueError(f'{identifier} not such activation')
    return activations[identifier]


class FeedForward(nn.Module):
    """FeedForward层
    """

    def __init__(self, in_features: int, mid_features: int, out_features: int, activation: Any='gelu', use_bias: bool=True, initializer: Any=truncated_normal_()):
        """
        https://arxiv.org/abs/2002.05202
        :param in_features: 输入维度
        :param mid_features: 中间层维度
        :param out_features: 输出维度
        :param use_bias: 是否使用偏差项
        :param activation: 激活函数，如果传入的是list，则将使用门控线性单元
        :param initializer: 初始化器
        """
        super(FeedForward, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.activation = [activation] if not isinstance(activation, list) else activation
        self.use_bias = use_bias
        self.initializer = initializer
        self.input_dense = nn.Linear(in_features=self.in_features, out_features=self.mid_features, bias=self.use_bias)
        self.initializer(self.input_dense.weight)
        for index in range(1, len(self.activation)):
            setattr(self, f'inner_dense_{index}', nn.Linear(in_features=self.in_features, out_features=self.mid_features, bias=self.use_bias))
            self.initializer(getattr(self, f'inner_dense_{index}').weight)
        self.output_dense = nn.Linear(in_features=self.mid_features, out_features=self.out_features, bias=self.use_bias)
        self.initializer(self.output_dense.weight)

    def forward(self, inputs):
        outputs = self.input_dense(inputs)
        outputs = get_activation(self.activation[0])(outputs)
        for index in range(1, len(self.activation)):
            inner_outputs = getattr(self, f'inner_dense_{index}')(inputs)
            inner_outputs = get_activation(self.activation[index])(inner_outputs)
            outputs = outputs * inner_outputs
        outputs = self.output_dense(outputs)
        return outputs


class RelativePositionEmbedding(nn.Module):
    """定义相对位置编码：https://arxiv.org/abs/1803.02155
    """

    def __init__(self, input_dim: int, output_dim: int, initializer: Any=truncated_normal_(), requires_grad: bool=False, device: Any=None, dtype: Any=None):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param initializer: 初始化器
        :param requires_grad: 是否可训练
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RelativePositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.weight = nn.Parameter(data=torch.empty((self.input_dim, self.output_dim), requires_grad=requires_grad, **factory_kwargs))
        self.initializer(self.weight)

    def forward(self, query, value):
        query_idx = torch.arange(0, query.shape[1])[:, None]
        value_idx = torch.arange(0, value.shape[1])[None, :]
        pos_ids = value_idx - query_idx
        max_position = (self.input_dim - 1) // 2
        pos_ids = torch.clamp(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return self.weight.index_select(dim=0, index=pos_ids)


def sinusoidal_init_(position: int, depth: int) ->NoReturn:
    """Sin-Cos位置向量初始化器
    :param position: 位置大小
    :param depth: 位置嵌入大小
    """

    def _sinusoidal_init(tensor: torch.Tensor):
        with torch.no_grad():
            pos = np.arange(position)[:, np.newaxis]
            index = np.arange(depth)[np.newaxis, :]
            angle_rates = 1 / np.power(10000, 2 * (index // 2) / np.float32(depth))
            angle_rads = pos * angle_rates
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            tensor.detach().copy_(torch.from_numpy(angle_rads))
            return tensor
    return _sinusoidal_init


class BertLayer(nn.Module):
    """Bert Block
    """

    def __init__(self, config: BertConfig, batch_size: int, initializer: Any=None):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param initializer: 初始化器
        """
        super(BertLayer, self).__init__()
        self.bert_config = config
        self.batch_size = batch_size
        self.initializer = initializer if initializer else truncated_normal_(stddev=config.initializer_range)
        self.key_size = self.bert_config.attention_key_size
        emb_input_dim = 2 * 64 + 1
        emb_output_dim = self.key_size if self.key_size is not None else self.bert_config.attention_head_size
        self.embeddings_initializer = sinusoidal_init_(position=emb_input_dim, depth=emb_output_dim)
        self.position_embeddings = RelativePositionEmbedding(input_dim=emb_input_dim, output_dim=emb_output_dim, initializer=self.embeddings_initializer, requires_grad=False)
        self.bert_self_attention = BertSelfAttention(num_heads=self.bert_config.num_attention_heads, head_size=self.bert_config.attention_head_size, batch_size=self.batch_size, attention_dropout=self.bert_config.attention_probs_dropout_prob, key_size=self.bert_config.attention_key_size, hidden_size=self.bert_config.hidden_size, initializer=self.initializer, pos_type='typical_relation')
        self.attn_dropout = nn.Dropout(p=self.bert_config.hidden_dropout_prob)
        self.attn_norm = nn.LayerNorm(normalized_shape=self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)
        self.feedforward = FeedForward(in_features=self.bert_config.hidden_size, mid_features=self.bert_config.intermediate_size, out_features=self.bert_config.hidden_size, activation=self.bert_config.hidden_act, initializer=self.initializer)
        self.feedforward_dropout = nn.Dropout(p=self.bert_config.hidden_dropout_prob)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)

    def forward(self, inputs, mask):
        pos_ids = self.position_embeddings(inputs, inputs)
        attn_outputs, attn_weights = self.bert_self_attention([inputs, inputs, inputs, pos_ids, mask])
        attn_outputs = self.attn_dropout(attn_outputs)
        attn_outputs = attn_outputs + inputs
        attn_outputs = self.attn_norm(attn_outputs)
        outputs = self.feedforward(attn_outputs)
        outputs = self.feedforward_dropout(outputs)
        outputs = outputs + attn_outputs
        outputs = self.feedforward_norm(outputs)
        return outputs


class BiasAdd(nn.Module):
    """偏置项
    """

    def __init__(self, shape: tuple, device: Any=None, dtype: Any=None):
        """
        :param shape:
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BiasAdd, self).__init__()
        self.weight = nn.Parameter(torch.empty(shape, **factory_kwargs))
        nn.init.zeros_(self.weight)

    def forward(self, inputs):
        return inputs + self.weight


class BertOutput(nn.Module):
    """Bert 规范化输出
    """

    def __init__(self, with_pool: Any=True, with_nsp: Any=False, with_mlm: Any=False, initializer: Any=truncated_normal_(), hidden_size: int=None, embedding_size: int=None, hidden_act: str=None, layer_norm_eps: float=None, mlm_decoder: Any=None, mlm_decoder_arg: dict=None, vocab_size: int=None):
        """
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, mlm_decoder
        :param initializer: 初始化器
        :param hidden_size: 隐藏层大小
        :param embedding_size: 词嵌入大小
        :param hidden_act: encoder和pool中的非线性激活函数
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param mlm_decoder: 用于给mlm做vocab分类的层，可训练，相当于无bias的dense
        :param
        :param vocab_size: mlm_decoder必传
        """
        super(BertOutput, self).__init__()
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = initializer
        self.hidden_size = hidden_size
        if self.with_pool:
            self.pool_activation = {'act': 'tanh', 'arg': {}} if with_pool is True else with_pool
            self.pooler_dense = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.initializer(self.pooler_dense.weight)
            if self.with_nsp:
                self.nsp_prob = nn.Linear(in_features=self.hidden_size, out_features=2)
                self.initializer(self.nsp_prob.weight)
        if self.with_mlm:
            self.mlm_activation = {'act': 'softmax', 'arg': {'dim': -1}} if with_mlm is True else with_mlm
            self.embedding_size = embedding_size
            self.hidden_act = hidden_act
            self.layer_norm_eps = layer_norm_eps
            self.mlm_decoder = mlm_decoder
            self.mlm_decoder_arg = {} if mlm_decoder_arg is None else mlm_decoder_arg
            self.mlm_dense = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size)
            self.initializer(self.mlm_dense.weight)
            self.mlm_norm = nn.LayerNorm(normalized_shape=self.embedding_size, eps=self.layer_norm_eps)
            self.mlm_bias = BiasAdd(shape=(vocab_size,))

    def forward(self, inputs):
        outputs = []
        if self.with_pool:
            sub_outputs = inputs[:, 0]
            sub_outputs = self.pooler_dense(sub_outputs)
            sub_outputs = get_activation(self.pool_activation['act'])(sub_outputs, **self.pool_activation['arg'])
            if self.with_nsp:
                sub_outputs = self.nsp_prob(sub_outputs)
                sub_outputs = get_activation('softmax')(sub_outputs, dim=-1)
            outputs.append(sub_outputs)
        if self.with_mlm:
            sub_outputs = self.mlm_dense(inputs)
            sub_outputs = get_activation(self.hidden_act)(sub_outputs)
            sub_outputs = self.mlm_norm(sub_outputs)(sub_outputs)
            sub_outputs = self.mlm_decoder(sub_outputs, **self.mlm_decoder_arg)
            sub_outputs = self.mlm_bias(sub_outputs)
            sub_outputs = get_activation(self.mlm_activation['act'])(sub_outputs, **self.mlm_activation['arg'])
            outputs.append(sub_outputs)
        if not outputs:
            return inputs
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


class Embedding(nn.Embedding):
    """扩展Embedding层
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int=None, max_norm: float=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, _weight: torch.Tensor=None, device=None, dtype=None) ->None:
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)

    def forward(self, inputs: torch.Tensor, mode: str='embedding') ->torch.Tensor:
        """新增mode参数，可以为embedding或dense。如果为embedding，
           则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        if mode == 'embedding':
            return F.embedding(inputs, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return torch.matmul(input=inputs, other=self.weight.permute(1, 0))


class ALBERT(nn.Module):
    """ALBERT Model
    """

    def __init__(self, config: BertConfig, batch_size: int, position_merge_mode: str='add', is_training: bool=True, add_pooling_layer: bool=True, with_pool: Any=False, with_nsp: Any=False, with_mlm: Any=False):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param position_merge_mode: 输入和position合并的方式
        :param is_training: train/eval
        :param add_pooling_layer: 处理输出，后面三个参数用于此
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, mlm_decoder
        """
        super(ALBERT, self).__init__()
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_prob_dropout_prob = 0.0
        self.batch_size = batch_size
        self.position_merge_mode = position_merge_mode
        self.is_training = is_training
        self.add_pooling_layer = add_pooling_layer
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = truncated_normal_(mean=0.0, stddev=self.config.initializer_range)
        self.token_embeddings = Embedding(num_embeddings=self.config.vocab_size, embedding_dim=self.config.embedding_size, padding_idx=0)
        self.initializer(self.token_embeddings.weight)
        self.bert_embeddings = BertEmbeddings(hidden_size=self.config.hidden_size, embedding_size=self.config.embedding_size, hidden_dropout_prob=self.config.hidden_dropout_prob, shared_segment_embeddings=self.config.shared_segment_embeddings, max_position=self.config.max_position, position_merge_mode=self.position_merge_mode, hierarchical_position=self.config.hierarchical_position, type_vocab_size=self.config.type_vocab_size, layer_norm_eps=self.config.layer_norm_eps, initializer=self.initializer)
        self.bert_layer = BertLayer(config=self.config, batch_size=self.batch_size, initializer=self.initializer)
        if self.add_pooling_layer:
            argument = {}
            if with_mlm:
                argument['embedding_size'] = self.config.embedding_size
                argument['hidden_act'] = self.config.hidden_act
                argument['layer_norm_eps'] = self.config.layer_norm_eps
                argument['mlm_decoder'] = nn.Linear(in_features=self.config.embedding_size, out_features=self.config.vocab_size)
                argument['vocab_size'] = self.config.vocab_size
            self.bert_output = BertOutput(with_pool, with_nsp, with_mlm, self.initializer, self.config.hidden_size, **argument)

    def forward(self, input_ids, token_type_ids):
        input_mask = torch.eq(input=input_ids, other=0).float()[:, None, None, :]
        outputs = self.bert_embeddings(input_ids, token_type_ids, self.token_embeddings)
        for index in range(self.config.num_hidden_layers):
            outputs = self.bert_layer(outputs, input_mask)
        if self.add_pooling_layer:
            outputs = self.bert_output(outputs)
        return outputs


class BertModel(nn.Module):
    """Bert Model
    """

    def __init__(self, config: BertConfig, batch_size: int, position_merge_mode: str='add', is_training: bool=True, add_pooling_layer: bool=True, with_pool: Any=False, with_nsp: Any=False, with_mlm: Any=False):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param position_merge_mode: 输入和position合并的方式
        :param is_training: train/eval
        :param add_pooling_layer: 处理输出，后面三个参数用于此
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, mlm_decoder
        """
        super(BertModel, self).__init__()
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_prob_dropout_prob = 0.0
        self.batch_size = batch_size
        self.position_merge_mode = position_merge_mode
        self.is_training = is_training
        self.add_pooling_layer = add_pooling_layer
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = truncated_normal_(mean=0.0, stddev=self.config.initializer_range)
        self.token_embeddings = Embedding(num_embeddings=self.config.vocab_size, embedding_dim=self.config.embedding_size, padding_idx=0)
        self.initializer(self.token_embeddings.weight)
        self.bert_embeddings = BertEmbeddings(hidden_size=self.config.hidden_size, embedding_size=self.config.embedding_size, hidden_dropout_prob=self.config.hidden_dropout_prob, shared_segment_embeddings=self.config.shared_segment_embeddings, max_position=self.config.max_position, position_merge_mode=self.position_merge_mode, hierarchical_position=self.config.hierarchical_position, type_vocab_size=self.config.type_vocab_size, layer_norm_eps=self.config.layer_norm_eps, initializer=self.initializer)
        for index in range(self.config.num_hidden_layers):
            setattr(self, f'bert_layer_{index}', BertLayer(config=self.config, batch_size=self.batch_size, initializer=self.initializer))
        if self.add_pooling_layer:
            argument = {}
            if with_mlm:
                argument['embedding_size'] = self.config.embedding_size
                argument['hidden_act'] = self.config.hidden_act
                argument['layer_norm_eps'] = self.config.layer_norm_eps
                argument['mlm_decoder'] = self.token_embeddings
                argument['mlm_decoder_arg'] = {'mode': 'dense'}
                argument['vocab_size'] = self.config.vocab_size
            self.bert_output = BertOutput(with_pool, with_nsp, with_mlm, self.initializer, self.config.hidden_size, **argument)

    def forward(self, input_ids, token_type_ids):
        input_mask = torch.eq(input=input_ids, other=0).float()[:, None, None, :]
        outputs = self.bert_embeddings(input_ids, token_type_ids, self.token_embeddings)
        for index in range(self.config.num_hidden_layers):
            outputs = getattr(self, f'bert_layer_{index}')(outputs, input_mask)
        if self.add_pooling_layer:
            outputs = self.bert_output(outputs)
        return outputs


class NEZHA(nn.Module):
    """NEZHA Model
    """

    def __init__(self, config: BertConfig, batch_size: int, position_merge_mode: str='add', is_training: bool=True, add_pooling_layer: bool=True, with_pool: Any=False, with_nsp: Any=False, with_mlm: Any=False):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param position_merge_mode: 输入和position合并的方式
        :param is_training: train/eval
        :param add_pooling_layer: 处理输出，后面三个参数用于此
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, mlm_decoder
        """
        super(NEZHA, self).__init__()
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_prob_dropout_prob = 0.0
        self.batch_size = batch_size
        self.position_merge_mode = position_merge_mode
        self.is_training = is_training
        self.add_pooling_layer = add_pooling_layer
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = truncated_normal_(mean=0.0, stddev=self.config.initializer_range)
        self.token_embeddings = Embedding(num_embeddings=self.config.vocab_size, embedding_dim=self.config.embedding_size, padding_idx=0)
        self.initializer(self.token_embeddings.weight)
        self.bert_embeddings = BertEmbeddings(hidden_size=self.config.hidden_size, embedding_size=self.config.embedding_size, hidden_dropout_prob=self.config.hidden_dropout_prob, shared_segment_embeddings=self.config.shared_segment_embeddings, type_vocab_size=self.config.type_vocab_size, layer_norm_eps=self.config.layer_norm_eps, initializer=self.initializer)
        for index in range(self.config.num_hidden_layers):
            setattr(self, f'bert_layer_{index}', BertLayer(config=self.config, batch_size=self.batch_size, initializer=self.initializer))
        if self.add_pooling_layer:
            argument = {}
            if with_mlm:
                argument['embedding_size'] = self.config.embedding_size
                argument['hidden_act'] = self.config.hidden_act
                argument['layer_norm_eps'] = self.config.layer_norm_eps
                argument['mlm_decoder'] = self.token_embeddings
                argument['mlm_decoder_arg'] = {'mode': 'dense'}
                argument['vocab_size'] = self.config.vocab_size
            self.bert_output = BertOutput(with_pool, with_nsp, with_mlm, self.initializer, self.config.hidden_size, **argument)

    def forward(self, input_ids, token_type_ids):
        input_mask = torch.eq(input=input_ids, other=0).float()[:, None, None, :]
        outputs = self.bert_embeddings(input_ids, token_type_ids, self.token_embeddings)
        for index in range(self.config.num_hidden_layers):
            outputs = getattr(self, f'bert_layer_{index}')(outputs, input_mask)
        if self.add_pooling_layer:
            outputs = self.bert_output(outputs)
        return outputs


class Model(nn.Module):
    """组合模型适配任务
    """

    def __init__(self, bert_config: BertConfig, batch_size: int, model_type: str, pooling: str):
        super(Model, self).__init__()
        self.first_block_layer_output = None
        self.last_block_layer_output = None
        self.pooling = pooling
        with_pool = 'linear' if pooling == 'pooler' else False
        if model_type == 'bert':
            self.model = BertModel(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
        elif model_type == 'albert':
            self.model = ALBERT(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
        elif model_type == 'nezha':
            self.model = NEZHA(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
        else:
            raise ValueError('model_type is None or not support')
        if model_type == 'albert':
            for name, module in self.model.named_modules():
                if name == 'bert_layer.feedforward_norm':
                    module.register_forward_hook(hook=self.first_block_hook)
                    module.register_forward_hook(hook=self.last_block_hook)
                    break
        else:
            for name, module in self.model.named_modules():
                if name == 'bert_layer_0.feedforward_norm':
                    module.register_forward_hook(hook=self.first_block_hook)
                elif name == f'bert_layer_{bert_config.num_hidden_layers - 1}.feedforward_norm':
                    module.register_forward_hook(hook=self.last_block_hook)

    def forward(self, input_ids, token_type_ids):
        outputs = self.model(input_ids, token_type_ids)
        if self.pooling == 'first-last-avg':
            first_block_outputs = self.first_block_layer_output.mean(dim=1)
            last_block_outputs = self.last_block_layer_output.mean(dim=1)
            outputs = (first_block_outputs + last_block_outputs) / 2.0
        elif self.pooling == 'last-avg':
            outputs = self.last_block_layer_output.mean(dim=1)
        elif self.pooling == 'cls':
            outputs = outputs[:, 0]
        elif self.pooling == 'pooler':
            pass
        else:
            raise ValueError('pooling is None or not support')
        return outputs

    def first_block_hook(self, module, fea_in, fea_out):
        self.first_block_layer_output = fea_out
        return None

    def last_block_hook(self, module, fea_in, fea_out):
        self.last_block_layer_output = fea_out
        return


class ContrastiveLoss(nn.Module):
    """ 对比损失函数"""

    def __init__(self) ->NoReturn:
        super(ContrastiveLoss, self).__init__()

    def forward(self, ew: Any, label: Any, m: float):
        """
        :param ew: Embedding向量之间的度量
        :param label: 样本句子的标签
        :param m: 负样本控制阈值
        :return:
        """
        l_1 = 0.25 * (1.0 - ew) * (1.0 - ew)
        l_0 = torch.where(ew < m * torch.ones_like(ew), torch.full_like(ew, 0), ew) * torch.where(ew < m * torch.ones_like(ew), torch.full_like(ew, 0), ew)
        loss = label * 1.0 * l_1 + (1 - label) * 1.0 * l_0
        return loss.sum()


class PositionEmbedding(nn.Module):
    """定义可训练的位置Embedding
    """

    def __init__(self, input_dim: int, output_dim: int, merge_mode: str='add', hierarchical: Any=None, custom_position_ids: bool=False, initializer: Any=truncated_normal_(), device: Any=None, dtype: Any=None):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param merge_mode: 输入和position合并的方式
        :param hierarchical: 是否层次分解位置编码
        :param custom_position_ids: 是否传入自定义位置编码id
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.custom_position_ids = custom_position_ids
        self.initializer = initializer
        self.weight = nn.Parameter(torch.empty((self.input_dim, self.output_dim), **factory_kwargs))
        self.initializer(self.weight)

    def forward(self, inputs):
        """如果传入自定义position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            position_ids = position_ids.int()
        else:
            batch_size, seq_len = inputs.size()[0], inputs.size()[1]
            position_ids = torch.arange(start=0, end=seq_len, step=1).unsqueeze(0)
        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.weight - alpha * self.weight[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = embeddings.index_select(dim=0, index=position_ids // self.input_dim)
            embeddings_y = embeddings.index_select(dim=0, index=position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        elif self.custom_position_ids:
            embeddings = self.weight.index_select(dim=0, index=position_ids)
        else:
            embeddings = self.weight[None, :seq_len]
        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = embeddings.repeat(batch_size, 1, 1)
            return torch.cat(tensors=(inputs, embeddings), dim=-1)


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__()
        self.p = p
        self.noise_shape = None

    def forward(self, inputs, noise_shape=None):
        """ noise_shape，tuple，应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1]
        self.noise_shape = noise_shape
        if not self.training or self.p == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.p == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.p).div_(1 - self.p)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


class Highway(nn.Module):
    """Highway
    """

    def __init__(self, feature_dim: int, transform_gate_bias: int=-2):
        """
        :param feature_dim: 输入特征大小
        :param transform_gate_bias: 常数初始化器scalar
        """
        super(Highway, self).__init__()
        self.transform_gate_bias = transform_gate_bias
        self.gate_transform = nn.Linear(in_features=feature_dim, out_features=feature_dim)
        nn.init.constant_(tensor=self.gate_transform.weight, val=self.transform_gate_bias)
        self.block_state = nn.Linear(in_features=feature_dim, out_features=feature_dim)
        nn.init.zeros_(tensor=self.block_state.weight)

    def forward(self, inputs):
        gate_transform = self.gate_transform(inputs)
        gate_transform = get_activation('sigmoid')(gate_transform)
        gate_cross = 1.0 - gate_transform
        block_state = self.block_state(inputs)
        block_state = get_activation('gelu')(block_state)
        highway = gate_transform * block_state + gate_cross * inputs
        return highway


class CharCNN(nn.Module):
    """Char CNN
    """

    def __init__(self, seq_len: int, embeddings_size: int, word_max_len: int, char_cnn_layers: list, highway_layers: int=2, num_rnn_layers: int=2, rnn_units: int=650, dropout: float=0.5, label_num: int=2, label_act: str='softmax'):
        """
        :param seq_len: 序列长度
        :param embeddings_size: 特征大小
        :param word_max_len: 单个token最大长度
        :param char_cnn_layers: 多层卷积列表，(filter_num, kernel_size_1)
                [[50, 1], [100, 2], [150, 3], [200, 4], [200, 5], [200, 6], [200, 7]]
        :param highway_layers: 使用highway层数
        :param num_rnn_layers: rnn层数
        :param rnn_units: rnn隐层大小
        :param dropout: 采样率
        :param label_num: 输出类别数
        :param label_act: 输出激活函数
        """
        super(CharCNN, self).__init__()
        self.seq_len = seq_len
        self.embeddings_size = embeddings_size
        self.word_max_len = word_max_len
        self.char_cnn_layers = char_cnn_layers
        self.highway_layers = highway_layers
        self.num_run_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.label_num = label_num
        self.label_act = label_act
        for index, char_cnn_size in enumerate(self.char_cnn_layers):
            setattr(self, f'conv_{index}', nn.Conv2d(in_channels=self.embeddings_size, out_channels=char_cnn_size[0], kernel_size=(1, char_cnn_size[1])))
            setattr(self, f'pool_{index}', nn.MaxPool2d(kernel_size=(1, self.word_max_len - char_cnn_size[1] + 1)))
        self.sum_filter_num = sum(np.array([ccl[0] for ccl in char_cnn_layers]))
        self.batch_norm = nn.BatchNorm1d(num_features=self.sum_filter_num)
        for highway_layer in range(highway_layers):
            setattr(self, f'highway_{highway_layer}', Highway(feature_dim=self.sum_filter_num))
        for index in range(num_rnn_layers):
            setattr(self, f'bi_lstm_{index}', nn.LSTM(input_size=self.sum_filter_num, hidden_size=rnn_units, bias=True, batch_first=True, bidirectional=True))
            setattr(self, f'lstm_dropout_{index}', nn.Dropout(p=self.dropout))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=self.seq_len * self.rnn_units * 2, out_features=self.label_num)

    def __int__(self, inputs):
        embeddings = inputs.unsqueeze(dim=-1)
        concat_embeddings = torch.concat(tensors=[embeddings for i in range(self.word_max_len)], dim=-1)
        embeddings_outputs = torch.permute(input=concat_embeddings, dims=[0, 2, 1, 3])
        conv_out = []
        for index, char_cnn_size in enumerate(self.char_cnn_layers):
            conv = getattr(self, f'conv_{index}')(embeddings_outputs)
            conv = get_activation('tanh')(conv)
            pooled = getattr(self, f'pool_{index}')(conv)
            pooled = torch.permute(input=pooled, dims=[0, 2, 3, 1])
            conv_out.append(pooled)
        outputs = torch.concat(tensors=conv_out, dim=-1)
        outputs = torch.reshape(input=outputs, shape=(outputs.shape[0], self.seq_len, outputs.shape[2] * self.sum_filter_num))
        outputs = self.batch_norm(outputs)
        for highway_layer in range(self.highway_layers):
            outputs = getattr(self, f'highway_{highway_layer}')(outputs)
        for index in range(self.num_rnn_layers):
            outputs = getattr(self, f'bi_lstm_{index}')(outputs)[0]
            outputs = getattr(self, f'lstm_dropout_{index}')(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        outputs = get_activation(self.label_act)(outputs)
        return outputs


def truncate_sequences(max_len: int, indices: Any, *sequences) ->list:
    """截断总长度至不超过max_len
    :param max_len: 最大长度
    :param indices: int/list
    :param sequences: 序列list
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > max_len:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


class BertTokenizerBase(object):
    """Bert分词器基类
    """

    def __init__(self, token_start: Any='[CLS]', token_end: Any='[SEP]', pre_tokenize: Any=None, token_translate: dict=None):
        """
        :param token_start: 起始token
        :param token_end: 结束token
        :param pre_tokenize: 外部传入的分词函数，用作对文本进行预分词。如果传入
                            pre_tokenize，则先执行pre_tokenize(text)，然后在它
                            的基础上执行原本的tokenize函数
        :param token_translate: 映射字典，主要用在tokenize之后，将某些特殊的token替换为对应的token
        """
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v: k for k, v in self._token_translate.items()}

    def tokenize(self, text: str, max_len: int=None) ->list:
        """分词
        :param text: 切词文本
        :param max_len: 填充长度
        """
        tokens = [(self._token_translate.get(token) or token) for token in self._tokenize(text)]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)
        if max_len is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(max_len, -index, tokens)
        return tokens

    def token_to_id(self, token: str) ->Any:
        """token转换为对应的id
        :param token: token
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens: list) ->list:
        """token序列转换为对应的id序列
        :param tokens: token list
        """
        return [self.token_to_id(token) for token in tokens]

    def encode(self, first_text: Any, second_text: Any=None, max_len: int=None, pattern: str='S*E*E', truncate_from: Any='post') ->tuple:
        """输出文本对应token id和segment id
        :param first_text: str/list
        :param second_text: str/list
        :param max_len: 最大长度
        :param pattern: pattern
        :param truncate_from: 填充位置，str/int
        """
        if isinstance(first_text, str):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text
        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, str):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text
        if max_len is not None:
            if truncate_from == 'post':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'pre':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                max_len += 1
            truncate_sequences(max_len, index, first_tokens, second_tokens)
        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)
        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        return first_token_ids, first_segment_ids

    def id_to_token(self, i: int) ->Any:
        """id转为对应个token
        :param i: id
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids: list) ->list:
        """id序列转为对应的token序列
        :param ids: id list
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids: list) ->Any:
        """转为可读文本
        :param ids: id list
        """
        raise NotImplementedError

    def _tokenize(self, text: str) ->list:
        """
        :param text: 切词文本
        """
        raise NotImplementedError


class BertTokenizer(BertTokenizerBase):
    """Bert原生分词器
    """

    def __init__(self, token_dict: Any, do_lower_case: bool=False, word_max_len: int=200, **kwargs):
        """
        :param token_dict: 映射字典或其文件路径
        :param do_lower_case: 小写化
        :param word_max_len: 最大长度
        """
        super(BertTokenizer, self).__init__(**kwargs)
        if isinstance(token_dict, str):
            token_dict = self.load_vocab(token_dict)
        self._do_lower_case = do_lower_case
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._vocab_size = len(token_dict)
        self._word_max_len = word_max_len
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = token_dict[getattr(self, f'_token_{token}')]
                setattr(self, f'_token_{token}_id', _token_id)
            except Exception as e:
                None

    def token_to_id(self, token: str) ->Any:
        """token转换为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i: int) ->Any:
        """id转换为对应的token
        """
        return self._token_dict_inv[i]

    def decode(self, ids: list, tokens: list=None) ->Any:
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self.is_special(token)]
        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self.is_cjk_character(token):
                text += token
            elif len(token) == 1 and self.is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self.is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token
        text = re.sub(' +', ' ', text)
        text = re.sub("' (re|m|s|t|ve|d|ll) ", "'\\1 ", text)
        punctuation = self.cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\\d\\.) (\\d)', '\\1\\2', text)
        return text.strip()

    def _tokenize(self, text: str, pre_tokenize: bool=True) ->list:
        """基本分词函数
        """
        if self._do_lower_case:
            text = BertTokenizer.lowercase_and_normalize(text)
        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens
        spaced = ''
        for ch in text:
            if self.is_punctuation(ch) or self.is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self.is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 65533 or self.is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self.word_piece_tokenize(word))
        return tokens

    def word_piece_tokenize(self, word: str):
        """word内分成subword
        """
        if len(word) > self._word_max_len:
            return [word]
        tokens, start, end = [], 0, 0
        while start < len(word):
            end = len(word)
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                end -= 1
            if start == end:
                return [word]
            else:
                tokens.append(sub)
                start = end
        return tokens

    @staticmethod
    def lowercase_and_normalize(text: str):
        """转小写，并进行简单的标准化
        """
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        return text

    @staticmethod
    def stem(token: str) ->str:
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def is_space(ch) ->bool:
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'

    @staticmethod
    def is_punctuation(ch) ->bool:
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'
        """
        code = ord(ch)
        return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126 or unicodedata.category(ch).startswith('P')

    @staticmethod
    def cjk_punctuation() ->str:
        return u'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'

    @staticmethod
    def is_cjk_character(ch) ->bool:
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 19968 <= code <= 40959 or 13312 <= code <= 19903 or 131072 <= code <= 173791 or 173824 <= code <= 177983 or 177984 <= code <= 178207 or 178208 <= code <= 183983 or 63744 <= code <= 64255 or 194560 <= code <= 195103

    @staticmethod
    def is_control(ch: Any) ->bool:
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def is_special(ch: Any) ->bool:
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and ch[0] == '[' and ch[-1] == ']'

    @staticmethod
    def is_redundant(token: str) ->bool:
        """判断该token是否冗余（默认情况下不可能分出来）
        """
        if len(token) > 1:
            for ch in BertTokenizer.stem(token):
                if BertTokenizer.is_cjk_character(ch) or BertTokenizer.is_punctuation(ch):
                    return True

    def rematch(self, text: str, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

    @staticmethod
    def load_vocab(dict_path: str, encoding: str='utf-8', simplified: bool=False, startswith: list=None) ->Any:
        """从bert的词典文件中读取词典
        :param dict_path: 字典文件路径
        :param encoding: 编码格式
        :param simplified: 是否过滤冗余部分token
        :param startswith: 附加在起始的list
        """
        token_dict = {}
        with open(dict_path, 'r', encoding=encoding) as reader:
            for line in reader:
                token = line.split()
                token = token[0] if token else line.strip()
                token_dict[token] = len(token_dict)
        if simplified:
            new_token_dict, keep_tokens = {}, []
            startswith = startswith or []
            for t in startswith:
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])
            for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
                if t not in new_token_dict and not BertTokenizer.is_redundant(t):
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])
            return new_token_dict, keep_tokens
        else:
            return token_dict

    @staticmethod
    def save_vocab(dict_path: str, token_dict: dict, encoding: str='utf-8') ->NoReturn:
        """将词典（比如精简过的）保存为文件
        """
        with open(dict_path, 'w+', encoding=encoding) as writer:
            for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
                writer.write(k + '\n')


class ColBERT(nn.Module):
    """ColBERT
    """

    def __init__(self, config: BertConfig, batch_size: int, bert_model_type: str='bert', feature_dim: int=128, mask_punctuation: bool=False, tokenizer: BertTokenizer=None, similarity_metric: str='cosine'):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param bert_model_type: bert模型
        :param feature_dim: feature size
        :param mask_punctuation: 是否mask标点符号
        :param tokenizer: 编码器，如果mask_punctuation为True，必传
        :param similarity_metric: 相似度计算方式
        """
        super(ColBERT, self).__init__()
        self.mask_punctuation = mask_punctuation
        self.similarity_metric = similarity_metric
        if bert_model_type == 'bert':
            self.bert_model = BertModel(config=config, batch_size=batch_size)
        elif bert_model_type == 'albert':
            self.bert_model = ALBERT(config=config, batch_size=batch_size)
        elif bert_model_type == 'nezha':
            self.bert_model = NEZHA(config=config, batch_size=batch_size)
        else:
            raise ValueError('`model_type` must in bert/albert/nezha')
        self.filter_dense = nn.Linear(in_features=config.hidden_size, out_features=feature_dim, bias=False)
        if mask_punctuation:
            skip_list = {w: (True) for symbol in string.punctuation for w in [symbol, tokenizer.encode(symbol)[0][1]]}

    def forward(self, query_input_ids, query_token_type_ids, doc_input_ids, doc_token_type_ids):
        query_embedding = self.bert_model(query_input_ids, query_token_type_ids)
        query_outputs = self.filter_dense(query_embedding)
        query_outputs = F.normalize(input=query_outputs, p=2, dim=2)
        doc_embedding = self.bert_model(doc_input_ids, doc_token_type_ids)
        doc_outputs = self.filter_dense(doc_embedding)
        if self.mask_punctuation:
            mask = [[(token not in self.skip_list and token != 0) for token in doc] for doc in doc_input_ids.tolist()]
            mask = torch.tensor(data=mask, dtype=torch.float32).unsqueeze(dim=-1)
            doc_outputs = doc_outputs * mask
        doc_outputs = F.normalize(input=doc_outputs, p=2, dim=2)
        if self.similarity_metric == 'cosine':
            outputs = (query_outputs @ doc_outputs.permute(0, 2, 1)).max(2).values.sum(1)
        elif self.similarity_metric == 'l2':
            outputs = (query_outputs.unsqueeze(2) - doc_outputs.unsqueeze(1)) ** 2
            outputs = (-1.0 * outputs.sum(-1)).max(-1).values.sum(-1)
        else:
            raise ValueError('`similarity_metric` must be cosine or l2')
        return outputs


class FastText(nn.Module):
    """ Fast Text Model
    """

    def __init__(self, embedding_size: int, seq_len: int, hidden_size: int, act: str='tanh', label_size: int=2, dropout: float=0.1):
        """
        :param embedding_size: 特征维度大小
        :param seq_len: 序列长度
        :param hidden_size: 中间隐层大小
        :param act: 激活函数
        :param label_size: 类别数
        """
        super(FastText, self).__init__()
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.act = act
        self.label_size = label_size
        self.dropout = dropout
        self.max_pooling = nn.MaxPool1d(kernel_size=self.embedding_size)
        self.avg_pooling = nn.AvgPool1d(kernel_size=self.embedding_size)
        self.mid_dense = nn.Linear(in_features=self.seq_len * 2, out_features=self.hidden_size)
        self.mid_dropout = nn.Dropout(p=self.dropout)
        self.output_dense = nn.Linear(in_features=self.hidden_size, out_features=self.label_size)

    def forward(self, inputs):
        inputs_m = self.max_pooling(inputs).squeeze()
        inputs_a = self.avg_pooling(inputs).squeeze()
        outputs = torch.concat([inputs_m, inputs_a], dim=-1)
        outputs = self.mid_dense(outputs)
        outputs = F.tanh(outputs)
        outputs = self.mid_dropout(outputs)
        outputs = self.output_dense(outputs)
        outputs = F.softmax(outputs, dim=-1)
        return outputs


def dot_product_attention(query: Any, key: Any, value: Any, depth: int, dropout: float, mask: Any=None) ->tuple:
    """通用点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param depth: 分头之后维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    """
    attention_scores = torch.matmul(input=query, other=key.permute(0, 2, 1))
    attention_scores = attention_scores / torch.sqrt(input=torch.tensor(data=depth))
    if mask is not None:
        attention_scores += mask * -1000000000.0
    attention_weights = torch.softmax(input=attention_scores, dim=-1)
    attention_weights = nn.Dropout(p=dropout)(attention_weights)
    context_layer = torch.matmul(input=attention_weights, other=value)
    return context_layer, attention_weights


class PolyEncoder(nn.Module):
    """ Poly Encoder
    """

    def __init__(self, config: BertConfig, batch_size: int, bert_model_type: str='bert', poly_type: str='learnt', candi_agg_type: str='cls', poly_m: int=16, has_labels: bool=True):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param bert_model_type: bert模型
        :param poly_type: m获取形式，learnt, first, last
        :param candi_agg_type: candidate表示类型，cls, avg
        :param poly_m: 控制阈值，m个global feature
        :param has_labels: 自监督/无监督
        """
        super(PolyEncoder, self).__init__()
        self.batch_size = batch_size
        self.embeddings_size = config.hidden_size
        self.poly_type = poly_type
        self.poly_m = poly_m
        self.candi_agg_type = candi_agg_type
        self.dropout_rate = config.hidden_dropout_prob
        self.has_labels = has_labels
        if poly_type == 'learnt':
            self.poly_embeddings = nn.Embedding(num_embeddings=poly_m + 1, embedding_dim=self.embeddings_size)
        if bert_model_type == 'bert':
            self.bert_model = BertModel(config=config, batch_size=batch_size)
        elif bert_model_type == 'albert':
            self.bert_model = ALBERT(config=config, batch_size=batch_size)
        elif bert_model_type == 'nezha':
            self.bert_model = NEZHA(config=config, batch_size=batch_size)
        else:
            raise ValueError('`model_type` must in bert/albert/nezha')
        if has_labels:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.dense = nn.Linear(in_features=self.embeddings_size, out_features=2)
            truncated_normal_(stddev=config.initializer_range)(self.dense.weight)

    def forward(self, context_input_ids, context_token_type_ids, candidate_input_ids, candidate_token_type_ids):
        context_embedding = self.bert_model(context_input_ids, context_token_type_ids)
        candidate_embedding = self.bert_model(candidate_input_ids, candidate_token_type_ids)
        if self.poly_type == 'learnt':
            context_poly_code_ids = torch.arange(start=1, end=self.poly_m + 1, step=1)
            context_poly_code_ids = torch.unsqueeze(input=context_poly_code_ids, dim=0)
            context_poly_code_ids = context_poly_code_ids.expand(self.batch_size, self.poly_m)
            context_poly_codes = self.poly_embeddings(context_poly_code_ids)
            context_vec, _ = dot_product_attention(query=context_poly_codes, key=context_embedding, value=context_embedding, depth=self.embeddings_size, dropout=self.dropout_rate)
        elif self.poly_type == 'first':
            context_vec = context_embedding[:, :self.poly_m]
        elif self.poly_type == 'last':
            context_vec = context_embedding[:, -self.poly_m:]
        else:
            raise ValueError('`poly_type` must in [learnt, first, last]')
        if self.candi_agg_type == 'cls':
            candidate_vec = candidate_embedding[:, 0]
        elif self.candi_agg_type == 'avg':
            candidate_vec = torch.sum(input=candidate_embedding, dim=1)
        else:
            raise ValueError('`candi_agg_type` must in [cls, avg]')
        final_vec, _ = dot_product_attention(query=candidate_vec, key=context_vec, value=context_vec, depth=self.embeddings_size, dropout=self.dropout_rate)
        outputs = torch.mean(input=final_vec * candidate_vec, dim=1)
        if self.has_labels:
            outputs = self.dropout(outputs)
            outputs = self.dense(outputs)
            outputs = get_activation('softmax')(input=outputs, dim=-1)
        return outputs


class Re2Encoder(nn.Module):
    """RE2 Encoder
    :param embedding_size: feature size
    :param filters_num: filter size
    :param enc_layers: encoder layer num
    :param kernel_size: 卷积核大小
    :param dropout: 采样率
    """

    def __init__(self, embedding_size: int, filters_num: int, enc_layers: int=2, kernel_size: Any=3, dropout: float=0.8):
        super(Re2Encoder, self).__init__()
        self.enc_layers = enc_layers
        for enc_index in range(enc_layers):
            if enc_index > 0:
                setattr(self, f'enc_dropout_{enc_index}', nn.Dropout(p=dropout))
            if enc_index == 0:
                setattr(self, f'enc_conv1d_{enc_index}', nn.Conv1d(in_channels=embedding_size, out_channels=filters_num, kernel_size=kernel_size, padding='same'))
            else:
                setattr(self, f'enc_conv1d_{enc_index}', nn.Conv1d(in_channels=filters_num, out_channels=filters_num, kernel_size=kernel_size, padding='same'))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, mask):
        outputs = inputs
        for enc_index in range(self.enc_layers):
            outputs = mask * outputs
            if enc_index > 0:
                outputs = getattr(self, f'enc_dropout_{enc_index}')(outputs)
            outputs = getattr(self, f'enc_conv1d_{enc_index}')(outputs.permute(0, 2, 1))
            outputs = outputs.permute(0, 2, 1)
            outputs = get_activation('relu')(outputs)
        outputs = self.dropout(outputs)
        return outputs


class Alignment(nn.Module):
    """对齐层"""

    def __init__(self, embedding_size: int, hidden_size: int, dropout: float, align_type: str='linear', device: Any=None, dtype: Any=None):
        """
        :param embedding_size: feature size
        :param hidden_size: hidden size
        :param dropout: 采样率
        :param align_type: 对齐方式，identity/linear
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Alignment, self).__init__()
        self.align_type = align_type
        if align_type == 'linear':
            self.linear_dropout1 = nn.Dropout(p=dropout)
            self.linear_dense1 = nn.Linear(in_features=embedding_size + hidden_size, out_features=hidden_size)
            self.linear_dropout2 = nn.Dropout(p=dropout)
            self.linear_dense2 = nn.Linear(in_features=embedding_size + hidden_size, out_features=hidden_size)
        self.temperature = nn.Parameter(nn.init.constant_(torch.empty((), **factory_kwargs), math.sqrt(1 / hidden_size)))

    def forward(self, a_inputs, a_mask, b_inputs, b_mask):
        if self.align_type == 'identity':
            attention_outputs = torch.matmul(input=a_inputs, other=b_inputs.permute(0, 2, 1)) * self.temperature
        elif self.align_type == 'linear':
            a_outputs = self.linear_dropout1(a_inputs)
            a_outputs = self.linear_dense1(a_outputs)
            a_outputs = get_activation('relu')(a_outputs)
            b_outputs = self.linear_dropout2(b_inputs)
            b_outputs = self.linear_dense2(b_outputs)
            b_outputs = get_activation('relu')(b_outputs)
            attention_outputs = torch.matmul(input=a_outputs, other=b_outputs.permute(0, 2, 1)) * self.temperature
        else:
            raise ValueError('`align_type` must be identity or linear')
        attention_mask = torch.matmul(input=a_mask, other=b_mask.permute(0, 2, 1))
        attention_outputs = attention_mask * attention_outputs + (1 - attention_mask) * -1000000000.0
        a_attention = nn.Softmax(dim=1)(attention_outputs)
        b_attention = nn.Softmax(dim=2)(attention_outputs)
        a_feature = torch.matmul(input=a_attention.permute(0, 2, 1), other=a_inputs)
        b_feature = torch.matmul(input=b_attention, other=b_inputs)
        return a_feature, b_feature


class Fusion(nn.Module):
    """Fusion Layer
    """

    def __init__(self, embedding_size: int, hidden_size: int, dropout: float, fusion_type: str='full'):
        """
        :param embedding_size: feature size
        :param hidden_size: feature size
        :param dropout: 采样率
        :param fusion_type: fusion type，simple/full
        """
        super(Fusion, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fusion_type = fusion_type
        if self.fusion_type == 'full':
            self.orig_dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
            self.sub_dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
            self.mul_dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
            self.dropout_layer = nn.Dropout(p=self.dropout)
            self.output_dense = nn.Linear(in_features=hidden_size * 3, out_features=hidden_size)
        elif self.fusion_type == 'simple':
            self.dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
        else:
            raise ValueError('`fusion_type` must be full or simple')

    def forward(self, inputs, align_inputs):
        if self.fusion_type == 'full':
            outputs = torch.concat(tensors=[get_activation('relu')(self.orig_dense(torch.concat(tensors=[inputs, align_inputs], dim=-1))), get_activation('relu')(self.sub_dense(torch.concat(tensors=[inputs, inputs - align_inputs], dim=-1))), get_activation('relu')(self.mul_dense(torch.concat(tensors=[inputs, inputs * align_inputs], dim=-1)))], dim=-1)
            outputs = self.dropout_layer(outputs)
            outputs = self.output_dense(outputs)
            outputs = get_activation('relu')(outputs)
        else:
            outputs = self.dense(torch.concat(tensors=[inputs, align_inputs], dim=-1))
            outputs = get_activation('relu')(outputs)
        return outputs


class Prediction(nn.Module):
    """Prediction Layer
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout: float, pred_type: str='full'):
        """
        :param num_classes: 类别数
        :param hidden_size: feature size
        :param dropout: 采样率
        :param pred_type: prediction type，simple/full/symmetric
        """
        super(Prediction, self).__init__()
        self.pred_type = pred_type
        self.dropout = dropout
        if self.pred_type == 'simple':
            in_features = hidden_size * 2
        elif self.pred_type == 'full' or self.pred_type == 'symmetric':
            in_features = hidden_size * 4
        else:
            raise ValueError('`pred_type` must be simple, full or symmetric')
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, a_feature, b_feature):
        if self.pred_type == 'simple':
            outputs = torch.concat(tensors=[a_feature, b_feature], dim=-1)
        elif self.pred_type == 'full':
            outputs = torch.concat(tensors=[a_feature, b_feature, a_feature * b_feature, a_feature - b_feature], dim=-1)
        elif self.pred_type == 'symmetric':
            outputs = torch.concat(tensors=[a_feature, b_feature, a_feature * b_feature, torch.abs(a_feature - b_feature)], dim=-1)
        else:
            raise ValueError('`pred_type` must be simple, full or symmetric')
        outputs = self.dropout1(outputs)
        outputs = self.dense1(outputs)
        outputs = get_activation('relu')(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.dense2(outputs)
        return outputs


class Re2Network(nn.Module):
    """Simple-Effective-Text-Matching
    """

    def __init__(self, vocab_size: int, embedding_size: int, block_layer_num: int=2, enc_layers: int=2, enc_kernel_size: Any=3, dropout: float=0.8, num_classes: int=2, hidden_size: int=None, connection_args: str='aug', align_type: str='linear', fusion_type: str='full', pred_type: str='full'):
        """
        :param vocab_size: 词表大小
        :param embedding_size: feature size
        :param block_layer_num: fusion block num
        :param enc_layers: encoder layer num
        :param enc_kernel_size: 卷积核大小
        :param dropout: 采样率
        :param num_classes: 类别数
        :param hidden_size: 隐藏层大小
        :param connection_args: 连接层模式，residual/aug
        :param align_type: 对齐方式，identity/linear
        :param fusion_type: fusion type，simple/full
        :param pred_type: prediction type，simple/full/symmetric
        """
        super(Re2Network, self).__init__()
        self.hidden_size = hidden_size
        if not hidden_size:
            self.hidden_size = embedding_size // 2
        self.block_layer_num = block_layer_num
        self.connection_args = connection_args
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.embeddings_dropout = nn.Dropout(p=dropout)
        self.connection = {'residual': self.residual, 'aug': self.augmented_residual}
        for index in range(block_layer_num):
            if index == 0:
                setattr(self, f're2_encoder_{index}', Re2Encoder(embedding_size=embedding_size, filters_num=hidden_size, enc_layers=enc_layers, kernel_size=enc_kernel_size, dropout=dropout))
                setattr(self, f'alignment_{index}', Alignment(embedding_size=embedding_size, hidden_size=hidden_size, dropout=dropout, align_type=align_type))
                setattr(self, f'fusion_{index}', Fusion(embedding_size=embedding_size, hidden_size=hidden_size, dropout=dropout, fusion_type=fusion_type))
            else:
                setattr(self, f're2_encoder_{index}', Re2Encoder(embedding_size=embedding_size + hidden_size, filters_num=hidden_size, enc_layers=enc_layers, kernel_size=enc_kernel_size, dropout=dropout))
                setattr(self, f'alignment_{index}', Alignment(embedding_size=embedding_size + hidden_size, hidden_size=hidden_size, dropout=dropout, align_type=align_type))
                setattr(self, f'fusion_{index}', Fusion(embedding_size=embedding_size + hidden_size, hidden_size=hidden_size, dropout=dropout, fusion_type=fusion_type))
        self.prediction = Prediction(num_classes=num_classes, hidden_size=hidden_size, dropout=dropout, pred_type=pred_type)

    def forward(self, text_a_input_ids, text_b_input_ids):
        text_a_mask = torch.eq(input=text_a_input_ids, other=0).float()[:, :, None]
        text_b_mask = torch.eq(input=text_b_input_ids, other=0).float()[:, :, None]
        a_embeddings = self.embeddings(text_a_input_ids)
        a_outputs = self.embeddings_dropout(a_embeddings)
        b_embeddings = self.embeddings(text_b_input_ids)
        b_outputs = self.embeddings_dropout(b_embeddings)
        a_residual, b_residual = a_outputs, b_outputs
        for index in range(self.block_layer_num):
            if index > 0:
                a_outputs = self.connection[self.connection_args](a_outputs, a_residual, index)
                b_outputs = self.connection[self.connection_args](b_outputs, b_residual, index)
                a_residual, b_residual = a_outputs, b_outputs
            a_encoder_outputs = getattr(self, f're2_encoder_{index}')(a_outputs, text_a_mask)
            b_encoder_outputs = getattr(self, f're2_encoder_{index}')(b_outputs, text_b_mask)
            a_outputs = torch.concat(tensors=[a_outputs, a_encoder_outputs], dim=-1)
            b_outputs = torch.concat(tensors=[b_outputs, b_encoder_outputs], dim=-1)
            a_align, b_align = getattr(self, f'alignment_{index}')(a_outputs, text_a_mask, b_outputs, text_b_mask)
            a_outputs = getattr(self, f'fusion_{index}')(a_outputs, a_align)
            b_outputs = getattr(self, f'fusion_{index}')(b_outputs, b_align)
        a_outputs = torch.sum(input=text_a_mask * a_outputs + (1.0 - text_a_mask) * -1000000000.0, dim=1)
        b_outputs = torch.sum(input=text_b_mask * b_outputs + (1.0 - text_b_mask) * -1000000000.0, dim=1)
        outputs = self.prediction(a_outputs, b_outputs)
        return outputs

    def residual(self, inputs: Any, res_inputs: Any, _) ->Any:
        """残差"""
        if inputs.shape[-1] != res_inputs.shape[-1]:
            inputs = nn.Linear(in_features=self.hidden_size, out_features=res_inputs.shape[-1])(inputs)
        return (inputs + res_inputs) * math.sqrt(0.5)

    def augmented_residual(self, inputs: Any, res_inputs: Any, index: int) ->Any:
        """增强残差"""
        outputs = inputs
        if index == 1:
            outputs = torch.concat(tensors=[res_inputs, inputs], dim=-1)
        elif index > 1:
            hidden_size = inputs.shape[-1]
            outputs = (res_inputs[:, :, -hidden_size:] + inputs) * math.sqrt(0.5)
            outputs = torch.concat(tensors=[res_inputs[:, :, :-hidden_size], outputs], dim=-1)
        return outputs


class SiameseRnnWithEmbedding(nn.Module):
    """ Siamese LSTM with Embedding """

    def __init__(self, emb_dim: int, vocab_size: int, units: int, dropout: float, num_layers: int, rnn: str, share: bool=True, if_bi: bool=True) ->NoReturn:
        """
        :param emb_dim: embedding dim
        :param vocab_size: 词表大小，例如为token最大整数index + 1.
        :param units: 输出空间的维度
        :param dropout: 采样率
        :param num_layers: RNN层数
        :param rnn: RNN的实现类型
        :param share: 是否共享权重
        :param if_bi: 是否双向
        :return:
        """
        super(SiameseRnnWithEmbedding, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.embedding2 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        if rnn not in ['lstm', 'gru']:
            raise ValueError('{} is unknown type'.format(rnn))
        if rnn == 'lstm':
            self.rnn1 = nn.LSTM(input_size=emb_dim, hidden_size=units, num_layers=num_layers, bidirectional=if_bi)
            self.rnn2 = nn.LSTM(input_size=emb_dim, hidden_size=units, num_layers=num_layers, bidirectional=if_bi)
        elif rnn == 'gru':
            self.rnn1 = nn.GRU(input_size=emb_dim, hidden_size=units, num_layers=num_layers, bidirectional=if_bi)
            self.rnn2 = nn.GRU(input_size=emb_dim, hidden_size=units, num_layers=num_layers, bidirectional=if_bi)
        self.if_bi = if_bi
        self.share = share
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs1: Any, inputs2: Any) ->tuple:
        """
        :param inputs1:
        :param inputs2:
        :return:
        """
        embedding1 = self.embedding1(inputs1.permute(1, 0))
        embedding2 = self.embedding2(inputs2.permute(1, 0))
        dropout1 = self.dropout(embedding1)
        dropout2 = self.dropout(embedding2)
        if self.share:
            outputs1 = self.rnn1(dropout1)
            outputs2 = self.rnn1(dropout2)
        else:
            outputs1 = self.rnn1(dropout1)
            outputs2 = self.rnn2(dropout2)
        if self.if_bi:
            state1 = torch.cat((outputs1[1][0][-2, :, :], outputs1[1][0][-1, :, :]), dim=-1)
            state2 = torch.cat((outputs2[1][0][-2, :, :], outputs2[1][0][-1, :, :]), dim=-1)
            return state1, state2
        return outputs1[1][0][-1:, :, :], outputs2[1][0][-1, :, :]


class SiameseBiRnnWithEmbedding(nn.Module):
    """ Learning Text Similarity with Siamese Recurrent Networks"""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, units: int, dropout: float, num_layers: int, rnn: str, cos_eps: float) ->NoReturn:
        """
        :param vocab_size: 词表大小，例如为token最大整数index + 1.
        :param emb_dim: embedding dim
        :param hidden_size: rnn隐藏层维度
        :param units: 全连接层输出维度
        :param dropout: 采样率
        :param num_layers: RNN层数
        :param rnn: RNN的实现类型
        :param cos_eps: 计算余弦相似度最小阈值
        :return:
        """
        super(SiameseBiRnnWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        if rnn not in ['lstm', 'gru']:
            raise ValueError('{} is unknown type'.format(rnn))
        if rnn == 'lstm':
            self.rnn_impl = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)
        elif rnn == 'gru':
            self.rnn_impl = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.dense = nn.Linear(in_features=units, out_features=units)
        self.dropout = nn.Dropout(dropout)
        self.eps = cos_eps

    def forward(self, inputs1: Any, inputs2: Any) ->torch.Tensor:
        embedding1 = self.embedding(inputs1.permute(1, 0))
        embedding2 = self.embedding(inputs2.permute(1, 0))
        rnn_outputs1 = self.rnn_impl(embedding1)
        rnn_outputs2 = self.rnn_impl(embedding2)
        avg1 = torch.mean(input=rnn_outputs1[0], dim=0)
        avg2 = torch.mean(input=rnn_outputs2[0], dim=0)
        output1 = self.dropout(torch.tanh(input=self.dense(avg1)))
        output2 = self.dropout(torch.tanh(input=self.dense(avg2)))
        output = torch.cosine_similarity(x1=output1, x2=output2, dim=-1, eps=self.eps)
        return output


class TextCNN(nn.Module):
    """Text CNN
    """

    def __init__(self, seq_len: int, embedding_size: int, units: int, filter_num: int, kernel_sizes: list, activations: list, padding: str='valid', dropout: float=0.1, act: str='tanh'):
        """
        :param seq_len: 序列长度
        :param embedding_size: 特征为大小
        :param units: 全连接层hidden size
        :param filter_num: 滤波器个数
        :param kernel_sizes: 卷积核大小，可以是多个
        :param activations: 激活函数列表，个数同kernel_sizes
        :param padding: 填充类型
        :param dropout: 采样率
        :param act: 全连接层激活函数
        """
        super(TextCNN, self).__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.units = units
        self.filter_num = filter_num
        self.kernel_sizes = kernel_sizes
        self.activations = activations
        self.padding = padding
        self.dropout = dropout
        self.act = act
        for index, kernel_size in enumerate(self.kernel_sizes):
            setattr(self, f'conv2d_{index}', nn.Conv2d(in_channels=1, out_channels=self.filter_num, kernel_size=(kernel_size, self.embedding_size), stride=(1, 1), padding=self.padding))
            setattr(self, f'pooled_{index}', nn.MaxPool2d(kernel_size=(seq_len - kernel_size + 1, 1), stride=(1, 1)))
        self.dropout = nn.Dropout(p=self.dropout)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=len(self.kernel_sizes) * self.filter_num, out_features=self.units)

    def forward(self, inputs):
        reshape_inputs = inputs.unsqueeze(dim=-1).permute(dims=(0, 3, 1, 2))
        conv_pools = []
        for index, activation in enumerate(self.activations):
            conv = getattr(self, f'conv2d_{index}')(reshape_inputs)
            conv_act = get_activation(activation)(conv)
            pooled = getattr(self, f'pooled_{index}')(conv_act)
            conv_pools.append(pooled)
        outputs = torch.concat(conv_pools, dim=-1)
        outputs = self.dropout(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        outputs = get_activation(self.act)(outputs)
        return outputs


class KMaxPooling(nn.Module):
    """动态K-max pooling
     k的选择为 k = max(k, s * (L-1) / L)
     其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
    """

    def __init__(self, top_k: int=8):
        super(KMaxPooling, self).__init__()
        self.top_k = top_k

    def forward(self, inputs):
        outputs = torch.topk(input=inputs, k=self.top_k, sorted=False).values
        return outputs


class ConvolutionalBlock(nn.Module):
    """Convolutional block
    """

    def __init__(self, in_channels: int, filter_num: int=256, kernel_size: int=3, strides: Any=1, padding: str='same', activation: Any='linear'):
        """
        :param in_channels:
        :param filter_num: 滤波器大小
        :param kernel_size: 卷积核大小
        :param strides: 移动步幅
        :param padding: 填充类型
        :param activation: 激活函数
        """
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=filter_num, kernel_size=kernel_size, padding=padding, stride=strides)
        self.batch_norm1 = nn.BatchNorm1d(num_features=filter_num)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=filter_num, out_channels=filter_num, kernel_size=kernel_size, stride=strides, padding=padding)
        self.batch_norm2 = nn.BatchNorm1d(num_features=filter_num)
        self.relu2 = nn.ReLU()
        self.activation = activation

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = get_activation(self.activation)(outputs)
        outputs = self.batch_norm1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = get_activation(self.activation)(outputs)
        outputs = self.batch_norm2(outputs)
        outputs = self.relu2(outputs)
        return outputs


class DownSampling(nn.Module):

    def __init__(self, pool_type: str='max', pool_size: Any=3, strides: Any=2, top_k: int=None, in_channels: int=None, padding: str='same'):
        """
        :param pool_type: "max", "k-max", "conv"
        :param pool_size: 池化窗口大小
        :param strides: 移动步幅
        :param top_k: top k, 如果是k-max，必传
        :param in_channels: 如果是conv，必传
        :param padding: 填充类型
        """
        super(DownSampling, self).__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=strides, padding=padding)
        elif pool_type == 'k-max':
            self.pool = KMaxPooling(top_k=top_k)
        elif pool_type == 'conv':
            self.pool = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=pool_size, stride=strides, padding=padding)
        else:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=strides, padding=padding)

    def forward(self, inputs):
        outputs = self.pool(inputs)
        return outputs


class ShortcutPool(nn.Module):
    """shortcut连接. 恒等映射, block+f(block)，加上 down sampling
    """

    def __init__(self, in_channels: int, filter_num: int=256, kernel_size: Any=1, strides: Any=2, padding: str='same', pool_type: str='max', shortcut: bool=True):
        """
        :param in_channels:
        :param filter_num: 滤波器大小
        :param kernel_size: 卷积核大小
        :param strides: 移动步幅
        :param padding: 填充类型
        :param pool_type: "max", "k-max", "conv"
        :param shortcut: 是否开启shortcut连接
        """
        super(ShortcutPool, self).__init__()
        self.shortcut = shortcut
        self.pool_type = pool_type
        if shortcut:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=filter_num, kernel_size=kernel_size, stride=strides, padding=padding)
            self.batch_norm = nn.BatchNorm1d(num_features=filter_num)
            self.down_sampling = DownSampling(self.pool_type)
        else:
            self.relu = nn.ReLU()
            self.down_sampling = DownSampling(self.pool_type)
        if pool_type is not None:
            self.conv1 = nn.Conv1d(in_channels=filter_num, out_channels=filter_num * 2, kernel_size=kernel_size, stride=strides, padding=padding)
            self.batch_norm1 = nn.BatchNorm1d(num_features=filter_num * 2)

    def forward(self, inputs, inputs_mid):
        if self.shortcut:
            conv = self.conv(inputs)
            conv = self.batch_norm(conv)
            outputs = self.down_sampling(inputs_mid)
            outputs = conv + outputs
        else:
            outputs = self.relu(inputs)
            outputs = self.down_sampling(outputs)
        if self.pool_type is not None:
            outputs = self.conv1(outputs)
            outputs = self.batch_norm1(outputs)
            return outputs


class TextVDCNN(nn.Module):
    """Text VDCNN
    """

    def __init__(self, embeddings_size: int, filters: list, dropout_spatial: float=0.2, dropout: float=0.32, activation_conv: Any='linear', pool_type: str='max', top_k: int=2, label_num: int=2, activate_classify: str='softmax'):
        """Text VDCNN
        :param embeddings_size: 特征大小
        :param filters: 滤波器配置, eg: [[64, 1], [128, 1], [256, 1], [512, 1]]
        :param dropout_spatial: 空间采样率
        :param dropout: 采样率
        :param activation_conv: 激活函数
        :param pool_type: "max", "k-max", "conv"
        :param top_k: max(k, s * (L-1) / L)
        :param label_num: 类别数
        :param activate_classify: 分类层激活函数
        """
        super(TextVDCNN, self).__init__()
        self.filters = filters
        self.activation_conv = activation_conv
        self.pool_type = pool_type
        self.activate_classify = activate_classify
        self.spatial_dropout = SpatialDropout(p=dropout_spatial)
        self.conv = nn.Conv1d(in_channels=embeddings_size, out_channels=filters[0][0], kernel_size=1, stride=1, padding='same')
        self.relu = nn.ReLU()
        for index, filters_block in enumerate(self.filters):
            for j in range(filters_block[1] - 1):
                setattr(self, f'{index}_{j}_conv_block', ConvolutionalBlock(embeddings_size, filters_block[0]))
            setattr(self, f'{index}_conv_block', ConvolutionalBlock(embeddings_size, filters_block[0]))
            setattr(self, f'{index}_shortcut_pool', ShortcutPool(embeddings_size, filters_block[0], strides=1, pool_type=self.pool_type))
        self.k_max_pool = KMaxPooling(top_k=top_k)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(in_features=embeddings_size, out_features=label_num)

    def forward(self, inputs):
        embeddings = self.spatial_dropout(inputs)
        conv = self.conv(embeddings)
        conv = get_activation(self.activation_conv)(conv)
        block = self.relu(conv)
        block = torch.permute(input=block, dims=[0, 2, 1])
        for index, filters_block in enumerate(self.filters):
            for j in range(filters_block[1] - 1):
                block_mid = getattr(self, f'{index}_{j}_conv_block')(block)
                block = block_mid + block
            block_mid = getattr(self, f'{index}_conv_block')(block)
            block = getattr(self, f'{index}_shortcut_pool')(block, block_mid)
        block = self.k_max_pool(block)
        block = torch.permute(input=block, dims=[0, 2, 1])
        block = self.flatten(block)
        block = self.dropout(block)
        outputs = self.dense(block)
        outputs = get_activation(self.activate_classify)(outputs)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiasAdd,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvolutionalBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (FastText,
     lambda: ([], {'embedding_size': 4, 'seq_len': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (FeedForward,
     lambda: ([], {'in_features': 4, 'mid_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Highway,
     lambda: ([], {'feature_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionEmbedding,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Prediction,
     lambda: ([], {'num_classes': 4, 'hidden_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Re2Encoder,
     lambda: ([], {'embedding_size': 4, 'filters_num': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (SpatialDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_DengBoCong_text_similarity(_paritybench_base):
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

