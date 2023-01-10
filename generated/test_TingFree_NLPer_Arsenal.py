import sys
_module = sys.modules[__name__]
del sys
codes = _module
explore = _module
text_classification = _module
text_generation = _module
nlper = _module
mini_pytorch_lightning = _module
model = _module
trainer = _module
models = _module
io = _module
text_clf = _module
text_gen = _module
modules = _module
decoders = _module
embeddings = _module
encoders = _module
metrics = _module
mlp = _module
modeling_outputs = _module
trainer = _module
utils = _module
corpus = _module
datasets = _module
fn = _module
format_convert = _module
tricks = _module
center_controller = _module
eight_bit = _module
specialModels = _module
fgm = _module
specialModels = _module
text_clf_handler = _module
unsup_simcse = _module
specialModels = _module

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


from torch import nn


import torch.nn.functional as F


from torch.optim import AdamW


from torch.utils.data import DataLoader


import torch


import numpy as np


import math


from typing import Dict


from typing import Callable


from typing import Optional


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import f1_score


from collections import OrderedDict


from typing import Union


from typing import List


import torch.nn as nn


import torch.nn.init as init


from torch.utils.data import Dataset


import warnings


import re


import random


import time


import matplotlib.pyplot as plt


from matplotlib.font_manager import FontProperties


def initial_parameter(net, initial_method=None):
    """source from fastnlp, A method used to initialize the weights of PyTorch models.
    https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/utils.py

    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)
                else:
                    init.normal_(w.data)
        elif m is not None and hasattr(m, 'weight') and hasattr(m.weight, 'requires_grad'):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)
                    else:
                        init.normal_(w.data)
    net.apply(weights_init)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron


    .. note::
        隐藏层的激活函数通过activation定义。一个str/function或者一个str/function的list可以被传入activation。
        如果只传入了一个str/function，那么所有隐藏层的激活函数都由这个str/function定义；
        如果传入了一个str/function的list，那么每一个隐藏层的激活函数由这个list中对应的元素定义，其中list的长度为隐藏层数。
        输出层的激活函数由output_activation定义，默认值为None，此时输出层没有激活函数。

    Examples::

        >>> net1 = MLP([5, 10, 5])
        >>> net2 = MLP([5, 10, 5], 'tanh')
        >>> net3 = MLP([5, 6, 7, 8, 5], 'tanh')
        >>> net4 = MLP([5, 6, 7, 8, 5], 'relu', output_activation='tanh')
        >>> net5 = MLP([5, 6, 7, 8, 5], ['tanh', 'relu', 'tanh'], 'tanh')
        >>> for net in [net1, net2, net3, net4, net5]:
        >>>     x = torch.randn(5, 5)
        >>>     y = net(x)
        >>>     print(x)
        >>>     print(y)
    """

    def __init__(self, size_layer, activation='relu', output_activation=None, initial_method=None, dropout=0.0):
        """

        :param List[int] size_layer: 一个int的列表，用来定义MLP的层数，列表中的数字为每一层是hidden数目。MLP的层数为 len(size_layer) - 1
        :param Union[str,func,List[str]] activation: 一个字符串或者函数的列表，用来定义每一个隐层的激活函数，字符串包括relu，tanh和
            sigmoid，默认值为relu
        :param Union[str,func] output_activation:  字符串或者函数，用来定义输出层的激活函数，默认值为None，表示输出层没有激活函数
        :param str initial_method: 参数初始化方式
        :param float dropout: dropout概率，默认值为0
        """
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        self.output_activation = output_activation
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i - 1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i - 1], size_layer[i]))
        self.dropout = nn.Dropout(p=dropout)
        actives = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        if not isinstance(activation, list):
            activation = [activation] * (len(size_layer) - 2)
        elif len(activation) == len(size_layer) - 2:
            pass
        else:
            raise ValueError(f'the length of activation function list except {len(size_layer) - 2} but got {len(activation)}!')
        self.hidden_active = []
        for func in activation:
            if callable(func):
                self.hidden_active.append(func)
            elif func.lower() in actives:
                self.hidden_active.append(actives[func])
            else:
                raise ValueError('should set activation correctly: {}'.format(activation))
        if self.output_activation is not None:
            if callable(self.output_activation):
                pass
            elif self.output_activation.lower() in actives:
                self.output_activation = actives[self.output_activation]
            else:
                raise ValueError('should set activation correctly: {}'.format(activation))
        initial_parameter(self, initial_method)

    def forward(self, x):
        """
        :param torch.Tensor x: the input of MLP
        :return: torch.Tensor : the output of MLP
        """
        for layer, func in zip(self.hiddens, self.hidden_active):
            x = self.dropout(func(layer(x)))
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)
        return x


class ModelOutput(OrderedDict):
    """修改自transformers.file_utils.ModelOutput，允许下标、索引、属性访问，根据创建时的顺序决定

    >>> t = ModelOutput(lr=1e-5)
    >>> t.update(n_epochs=2)
    >>> t
    >>> ModelOutput([('lr', 1e-05), ('n_epochs', 2)])
    >>> t[0] == t['lr'] == t.lr
    >>> True
    >>> t.lr = 5e-5  # equals t[0]=5e-5 or t['lr']=5e-5
    >>> t.batch_size = 8  # equal t['batch_size']=8
    >>> del t.lr  # equals "del t[0]" or "del t['lr']"
    """

    def __getitem__(self, k):
        """允许索引访问，同时也允许下标"""
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        """设置属性时，会覆盖同名item"""
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        """设置item的同时，也设置属性"""
        if isinstance(key, int):
            key = list(self.keys())[key]
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        """同时删除item和属性，只有通过该模型注册的才能被删除"""
        super().__delattr__(item)
        if item in self.keys():
            self.__delitem__(item)

    def __delitem__(self, key):
        """同时删除item和属性，只有通过该模型注册的才能被删除"""
        if isinstance(key, int):
            key = list(self.keys())[key]
        super().__delitem__(key)
        if key in self.__dict__:
            self.__delattr__(key)

    def pop(self, key):
        result = self[key]
        del self[key]
        return result

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class TextCLFOutput(ModelOutput):
    """
    封装TextCLF模型的输出

    Args:
        logits: Tensor, [batch_size, num_labels], 模型最后的输出结果，用于计算损失，非概率值
        seqEmb: Tensor, [batch_size, hidden_size], 最终用于分类的句子级表示
    """
    logits: torch.Tensor = None
    seqEmb: torch.Tensor = None


class BertCLF(nn.Module):

    def __init__(self, args):
        super(BertCLF, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.clf = MLP([self.bert.config.hidden_size, args.num_class], 'tanh', dropout=args.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.clf(outputs[1])
        return TextCLFOutput(logits=logits, seqEmb=outputs[1])


class TextGenOutput(ModelOutput):
    """
    封装TextGen模型的输出

    Args:
        pred: Tensor, [batch_size, seq_len, vocab_size], 用于计算loss

    """


class DecoderOutput(ModelOutput):
    """
    封装Decoder的输出

    Args:
        last_hidden_state: Tensor, [batch_size, seq_len, hidden_size], 解码器的预测结果
    """
    last_hidden_state: torch.Tensor = None


class TransformerDecoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, tgt, memory, memory_padding_mask=None, tgt_padding_mask=None):
        """

        :param tgt: tensor, [batch_size, tgt_len, d_model]
        :param memory: tensor, [batch_size, src_len, d_model]
        :param memory_padding_mask: tensor, [batch_size, src_len], nn.transformerDecoderLayer中的memory_key_padding_mask
        :param tgt_padding_mask: tensor/None, [batch_size, tgt_len], nn.transformerDecoderLayer中的tgt_key_padding_mask
        :return: DecoderOutput(last_hidden_state), last_hidden_state: [batch_size, tgt_len, d_model]
        """
        tgt_len = tgt.size(1)
        outputs = self.decoder(tgt.transpose(0, 1), memory.transpose(0, 1), tgt_mask=self.generate_square_subsequent_mask(tgt_len), memory_key_padding_mask=memory_padding_mask, tgt_key_padding_mask=tgt_padding_mask).transpose(0, 1)
        return DecoderOutput(last_hidden_state=outputs)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Roberta2Transformer(nn.Module):

    def __init__(self, args):
        super(Roberta2Transformer, self).__init__()
        roberta_config = RobertaConfig.from_pretrained(args.pretrained_model)
        self.encoder = RobertaModel.from_pretrained(args.pretrained_model)
        self.decoder = TransformerDecoder(d_model=roberta_config.hidden_size)
        self.generator = nn.Linear(roberta_config.hidden_size, roberta_config.vocab_size)

    def forward(self, encoded_src, encoded_tgt=None, encoder_hidden_states=False, decoder_hidden_states=True):
        """
        :param encoded_src: {'input_ids':[batch_size, src_len], 'token_type_ids':[batch_size, src_len],'attention_mask':[batch_size, src_len]}
        :param encoded_tgt: 和encoded_src类似
        :param encoder_hidden_states: 是否返回encoder的输出信息
        :param decoder_hidden_states: 是否返回decoder的输出信息
        :return: TextGenOutput
        """
        if not encoded_tgt:
            encoded_tgt = encoded_src
        src_input_ids, src_token_type_ids, src_attention_mask = encoded_src['input_ids'], encoded_src['token_type_ids'], encoded_src['attention_mask']
        tgt_input_ids, tgt_token_type_ids, tgt_attention_mask = encoded_tgt['input_ids'], encoded_tgt['token_type_ids'], encoded_tgt['attention_mask']
        embed_tgt = self.encoder.embeddings(input_ids=tgt_input_ids, token_type_ids=tgt_token_type_ids)
        encode_outputs = self.encoder(src_input_ids, src_attention_mask, src_token_type_ids)
        memory = encode_outputs.last_hidden_state
        decode_output = self.decoder(embed_tgt, memory, tgt_attention_mask == 0)
        final = decode_output.last_hidden_state
        output = TextGenOutput(pred=self.generator(final))
        if encoder_hidden_states:
            output.update(**{('encoder_' + k): v for k, v in encode_outputs.items()})
        if decoder_hidden_states:
            del decode_output.last_hidden_state
            output.update(**{('decoder_' + k): v for k, v in decode_output.items()})
        return output


class Dict2Obj:
    """
    将嵌套字典转换成对象，将关键字访问替换成属性访问

    >>> t = Dict2Obj()
    >>> t.x1 = 3e-5
    >>> t.x2.x21 = [8]
    >>> t.x2.x22 = 16
    >>> t.update({
    >>>     'x3': 0.1,
    >>>     'x2': {'x22': 32, 'x23': 64},
    >>>     'x4': {'x41':'yyy'}
    >>> })
    >>> t.toDict()  # {'x1': 3e-05, 'x2': {'x21': [8], 'x22': 32, 'x23': 64},
    >>>             #  'x3': 0.1, 'x4': {'x41': 'yyy'}}
    >>> print(t)  # str of t.toDict()
    """

    def __init__(self, init_dict=None):
        if init_dict:
            for key, value in init_dict.items():
                if self._is_valid(key):
                    if type(value) is dict:
                        self.__setattr__(key, Dict2Obj(value))
                    else:
                        self.__setattr__(key, value)

    def __getattr__(self, key):
        """访问一个不存在的属性时，调用该函数"""
        if self._is_valid(key):
            self.__setattr__(key, Dict2Obj({}))
            return self.__getattribute__(key)

    def __repr__(self):
        return str(self.toDict())

    def update(self, aux_dict):
        for key, value in aux_dict.items():
            if self._is_valid(key):
                if type(value) is dict:
                    if hasattr(self, key):
                        self.__getattribute__(key).update(value)
                    else:
                        self.__getattr__(key).update(value)
                else:
                    self.__setattr__(key, value)

    def _is_valid(self, key):
        if type(key) is str and re.match('[a-zA-Z_][0-9a-zA-Z_]*', key):
            return True
        raise ValueError(f'{key} is not a valid variable, please check manually')

    def toDict(self):
        target = {}
        for key, value in self.__dict__.items():
            if type(value) is not Dict2Obj:
                target[key] = value
            else:
                target[key] = value.toDict()
        return target


def type_transfor(data, target_type, in_type=None):
    """transform data into specific type(list, numpy and tensor)

    :param data: list, np.ndarray, torch.tensor
    :param target_type: list, np.ndarray, torch.tensor
    :param in_type: data[0] type
    :return: new data with target type
    """
    if in_type and isinstance(data, target_type) and isinstance(data[0], in_type):
        return data
    if not in_type and isinstance(data, target_type):
        return data
    elif isinstance(data, list):
        if target_type is np.ndarray:
            return np.array(data)
        else:
            return torch.tensor(data, dtype=target_type)
    elif isinstance(data, np.ndarray):
        if target_type is list:
            return data.tolist()
        else:
            return torch.from_numpy(data)
    elif target_type is list:
        return data.tolist()
    else:
        return data.numpy()


class MetricBase:

    def __init__(self, name: str, fun: Optional[Callable], data_type, in_type=None, **kwargs):
        """
        封装基础的评估函数

        :param name: 自定义评估函数名
        :param fun: 函数
        :param data_type: 函数接受的数据类型
        :param in_type: data[0]的数据类型
        :param kwargs: 函数的其它参数，非golds和preds
        """
        self.name = name
        self.metric = fun
        self.data_type = data_type
        self.in_type = in_type
        self.kwargs = kwargs

    def score(self, golds, preds):
        golds = type_transfor(golds, self.data_type, self.in_type)
        preds = type_transfor(preds, self.data_type, self.in_type)
        return self.metric(golds, preds, **self.kwargs)

    def score_end(self, scores):
        return scores


class Metrics:

    def __init__(self, metrics: Dict[str, MetricBase], target_metric: str):
        """
        指定模型在训练与测试过程中用来评估性能的指标

        :param metrics: 通过字典的形式添加评估指标，例如{'F1': f1_metric}, f1_metric:MetricBase
        :param target_metric: 用于早停以及保存最佳模型，必须为metrics中的一个，例如'F1'
        """
        self._metrics = metrics
        self.target = target_metric
        if self.target and self.target not in metrics.keys():
            raise ValueError('target_metric must be one of metrics or None')
        self.metric_values = {}

    def print_values(self):
        if not self.metric_values:
            None
        else:
            for name, value in self.metric_values.items():
                None

    def scores(self, golds, preds) ->Dict[str, float]:
        for metric_name in self._metrics.keys():
            scores = self._metrics[metric_name].score_end(self._metrics[metric_name].score(golds, preds))
            self.metric_values[metric_name] = scores
        return self.metric_values

    def target_score(self, golds, preds):
        self.metric_values[self.target] = self._metrics[self.target].score(golds, preds)
        return self.metric_values[self.target]

    def return_target_score(self):
        if self.target in self.metric_values.keys():
            if self.target == 'rouge-1' or self.target == 'rouge-2':
                return self.metric_values[self.target]['r']
            if self.target == 'rouge-l':
                return self.metric_values[self.target]['f']
            return self.metric_values[self.target]


class StandardModel(torch.nn.Module):

    def __init__(self, configs, metrics: Metrics, **kwargs):
        super(StandardModel, self).__init__()
        self.configs = configs
        self.aux_configs = Dict2Obj()
        self.metrics = metrics
        self.auto_optimization = True

    def __call__(self):
        self.forward()

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        """对训练的封装，处理一个batch，要求返回loss，其余自定义

        # :param batch: 一个batch_size大小的数据
        :param batch_idx: 该批次数据在整个数据中的顺序
        :returns: LightningOutput(loss)
        """
        raise NotImplementedError()

    def training_step_end(self, batch_outputs):
        """对training_step返回值的后处理，默认不处理，在自动计算梯度之后调用，详见mpl.Trainer

        :param batch_outputs: 列表，training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    def training_epoch_end(self, outputs):
        """对整个train epoch的输出数据进行处理

        :param outputs: 列表，每一个元素都是training_step_end的返回值
        :return: mpl.Trainer不处理返回值
        """
        pass

    def validation_step(self, batch, batch_idx):
        """对验证的封装，处理一个batch， 要求返回loss, preds, golds, 其余可以自定义，
        前三位是必须的（如果没有重写validation_epoch_end的话），预测值和真实值用于计算指标，
        要求符合相应metric的输入

        :param batch: 一个batch_size大小的数据
        :param batch_idx: 一个batch_size大小的数据
        :returns: LightningOutput(loss, preds, golds)
        """
        raise NotImplementedError()

    def validation_step_end(self, batch_outputs):
        """对validation_step返回值的后处理，默认不处理

        :param batch_outputs: 列表，training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    def validation_epoch_end(self, outputs) ->float:
        """对整个eval epoch的输出数据进行处理

        :param outputs: 元组，每一个元素都是validation_step_end的返回值
        :return: 目标指标值，用于early stop以及保存最佳模型，由metric.target而定
        """
        preds, golds = [], []
        for batch_outputs in outputs:
            preds += batch_outputs.preds
            golds += batch_outputs.golds
        self.metrics.scores(golds, preds)
        self.metrics.print_values()
        return self.metrics.return_target_score()

    def test_step(self, batch, batch_idx):
        """对预测的封装，处理一个batch，推荐返回预测值，但并不强制约束

        :param batch: 一个batch_size大小的数据
        :param batch_idx: 一个batch_size大小的数据
        :return: Any
        """
        raise NotImplementedError()

    def test_step_end(self, batch_outputs):
        """对test_step返回值的后处理，默认不处理

        :param batch_outputs: 列表，training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    def test_epoch_end(self, outputs):
        """对整个test的输出数据进行处理

        :param outputs: 列表，每一个元素都是training_step_end的返回值
        :return: mpl.Trainer不处理返回值
        """
        return None

    def configure_optimizers(self):
        """配置optimizer和lr scheduler，在加载数据之后调用

        :returns: optimizer(required), lr_scheduler(optional)
        """
        raise NotImplementedError()

    def prepare_data(self):
        """ 加载数据，预处理

        :return: None
        """
        return None

    def train_dataloader(self):
        """返回训练集数据迭代器

        :return: torch.utils.data.DataLoader
        """
        return None

    def val_dataloader(self):
        """返回开发集数据迭代器

        :return: torch.utils.data.DataLoader
        """
        return None

    def test_dataloader(self):
        """返回测试集数据迭代器

        :return: torch.utils.data.DataLoader
        """
        return None


class DataCollatorWithPadding:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        with_tgt = len(batch[0]) == 2
        src = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        tgt = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        for example in batch:
            ex_src = example['encoded_src']
            for k, v in ex_src.items():
                src[k].append(v)
            if with_tgt:
                ex_tgt = example['encoded_tgt']
                for k, v in ex_tgt.items():
                    tgt[k].append(v)
        batch_src = self.tokenizer.pad(src, padding=True, return_tensors='pt')
        if with_tgt:
            batch_tgt = self.tokenizer.pad(tgt, padding=True, return_tensors='pt')
            return {'encoded_src': batch_src.data, 'encoded_tgt': batch_tgt.data}
        else:
            return {'encoded_src': batch_src.data}


class DatasetCLF(Dataset):

    def __init__(self, data, tokenizer, max_len=512, load_label=True, **kwargs):
        """封装文本分类数据集，现在仅支持二分类和多分类，暂不支持多标签分类

        :param data: 标准数据格式为List[List[str, int]]，例如[[’今天天气很好‘, 1], ['我心情不好', 0]]
        :param tokenizer: transformers.xxxTokenizer
        :param max_len: 分词后的文本长度上限，default=512
        :param load_label: 是否加载标签，default=True
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.load_label = load_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][0]
        encode_inputs = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors='pt')
        example = {}
        for key, value in encode_inputs.items():
            example[key] = value[0]
        if self.load_label:
            label = self.data[index][1]
            example['labels'] = torch.tensor(label, dtype=torch.long)
            return example
        return example


class LightningOutput(ModelOutput):
    """
    封装mpl.StandardModel中training_step、validation_step的输出

    Args:
        loss: 可求导的损失
        preds: list, [batch_size, xxx], 模型的输出值，用于和golds计算指标，具体形式根据不同的任务决定
        golds: list, [batch_size, xxx], 数据的真实标签，用于和preds计算指标
    """
    loss: torch.Tensor = None
    preds: list = []
    golds: list = []
    target_score: float = 0.0


class Reader:

    def check(self, filepath, delete=False):
        """检查文件是否存在，若存在且delete=True，则删除该文件

        :param filepath: 文件路径
        :param delete: 是否删除该文件
        :return:
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'不存在{filepath}, 请仔细检查该文件路径')
        elif delete:
            os.remove(filepath)

    def read_txt(self, filepath) ->List[str]:
        """按行读取文件，去除样本两侧空格和换行符

        :param filepath: 文件路径
        :return: 列表
        """
        self.check(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f]
        None
        return data

    def get_loader(self, filepath, batch_size=1):
        """按行读取文件，每次读取batch_size行，返回一个迭代器，自动去除末尾换行符

        :param filepath: 文件路径
        :param batch_size: 每次读取的样本数
        :return: 生成器
        """
        self.check(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                examples = [f.readline() for _ in range(batch_size)]
                examples = [ex.strip() for ex in examples if ex]
                if len(examples) != batch_size:
                    if examples:
                        yield examples
                    break
                if batch_size == 1:
                    yield examples[0]
                else:
                    yield examples

    def read_json(self, filepath):
        """读取一个json文件

        :param filepath: 文件路径
        :return: 类型和json文件内容相关
        """
        self.check(filepath)
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        None
        return data

    def read_jsonl(self, filepath) ->list:
        """每一行是一个json字符串，按行读取，返回列表

        :param filepath: 文件路径
        :return: 列表
        """
        self.check(filepath)
        data = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        None
        return data

    def read_table(self, filepath, f_type=None, **kwargs):
        """读取表格，支持csv/tsv/xls/xlsx，如果要返回xls/xlsx中的所有页面，需要指定sheet_name=None

        :param filepath: 文件路径
        :param f_type: ['csv', 'tsv', 'xls', 'xlsx']中的一个，如果不指定，则自动识别文件名后缀
        :return: 表格
        """
        self.check(filepath)
        f_type = os.path.splitext(filepath)[-1].replace('.', '') if not f_type else f_type
        assert f_type in ['csv', 'tsv', 'xls', 'xlsx']
        if f_type == 'csv':
            csv_kwargs = select_kwargs(kwargs, pd.read_csv)
            data = pd.read_csv(filepath, sep=',', **csv_kwargs)
        elif f_type == 'tsv':
            tsv_kwargs = select_kwargs(kwargs, pd.read_csv)
            data = pd.read_csv(filepath, sep='\t', **tsv_kwargs)
        else:
            excel_kwargs = select_kwargs(kwargs, pd.read_excel)
            data = pd.read_excel(filepath, **excel_kwargs, engine='openpyxl')
        None
        return data

    def read_yaml(self, filepath) ->dict:
        """读取yaml

        :param filepath: 文件路径
        :return: 字典
        """
        self.check(filepath)
        with open(filepath, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def load_nlp_data(self, file, task_name='text_clf'):
        """读取不同任务的标准数据

        :param file: data file
        :param task_name: one of ['text_clf']
        :return: target data
        """
        if task_name == 'text_clf':
            raw_data = self.read_txt(file)
            target_data = []
            for raw_instance in raw_data:
                split_instance = raw_instance.split('\t')
                if len(split_instance) == 2:
                    split_instance[1] = int(split_instance[1])
                target_data.append(split_instance)
        else:
            raise ValueError(f'load {task_name} failed, we only support load text_clf data now')
        return target_data


class PosEmbedding(nn.Module):
    """绝对位置编码
    """

    def __init__(self, emb_size: int, max_len: int=512):
        super(PosEmbedding).__init__()
        assert emb_size % 2 == 0, 'emb_size must be even'
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, seq_len: int):
        """

        :param seq_len: 每个batch中的序列长度
        :return: [1, seq_len, emb_size]
        """
        emb = self.pe[:seq_len]
        return emb.unsqueeze(0)


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, emb_size=300, dropout=0.1, max_len=512):
        super(TransformerEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.tok_embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = PosEmbedding(emb_size, max_len)

    def forward(self, x):
        """

        :param x: Tensor, [batch, seq_len]
        :return: [batch, seq_len, emb_size]
        """
        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(x.size(1))
        final_emb = tok_emb + pos_emb
        return self.dropout(final_emb)


class EncoderOutput(ModelOutput):
    """
    封装Encoder的输出

    Args:
        seqEmb: Tensor, [batch_size, seq_len, hidden_size]
    """
    seqEmb: torch.Tensor = None


class TransformerEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, src, src_mask=None):
        """

        :param src: tensor, [batch_size, src_len, emb_size]
        :param src_mask: tensor/None, [batch_size, src_len];
        :return: EncoderOutput(seqEmb), seqEmb: [batch_size, src_len, d_model]
        """
        outputs = self.encoder(src, src_key_padding_mask=src_mask)
        return EncoderOutput(seqEmb=outputs)


def get_simcse_loss(once_emb, twice_emb, t=0.05):
    """用于无监督SimCSE训练的loss

    :param once_emb: [batch_size, emb_dim], 第一次dropout后的句子编码
    :param twice_emb: [batch_size, emb_dim], 第二次dropout后的句子编码
    :param t: 温度系数
    """
    batch_size = once_emb.size(0)
    y_true = torch.cat([torch.arange(1, batch_size * 2, step=2, dtype=torch.long).unsqueeze(1), torch.arange(0, batch_size * 2, step=2, dtype=torch.long).unsqueeze(1)], dim=1).reshape([batch_size * 2])
    batch_emb = torch.cat([once_emb, twice_emb], dim=1).reshape(batch_size * 2, -1)
    norm_emb = F.normalize(batch_emb, dim=1, p=2)
    sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
    sim_score = sim_score - torch.eye(batch_size * 2, device=once_emb.device) * 1000000000000.0
    sim_score = sim_score / t
    loss = F.cross_entropy(sim_score, y_true)
    return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'size_layer': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StandardModel,
     lambda: ([], {'configs': _mock_config(), 'metrics': 4}),
     lambda: ([], {}),
     True),
]

class Test_TingFree_NLPer_Arsenal(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

