import sys
_module = sys.modules[__name__]
del sys
bert_seq2seq = _module
bart_chinese = _module
basic_bert = _module
bert_cls_classifier = _module
bert_cls_multi_classifier = _module
bert_cls_multi_seq2seq = _module
bert_relation_extraction = _module
bert_seq_labeling = _module
bert_seq_labeling_crf = _module
config = _module
dataset = _module
extend_model_method = _module
gpt2_generate_model = _module
helper = _module
model = _module
bart_model = _module
bert_model = _module
crf = _module
gpt2_model = _module
nezha_model = _module
roberta_model = _module
t5_model = _module
data = _module
collate = _module
iterator = _module
sampler = _module
tokenizer = _module
vocab = _module
transformers = _module
attention_utils = _module
bert = _module
modeling = _module
generation_utils = _module
gpt = _module
model_utils = _module
nezha = _module
optimization = _module
roberta = _module
tokenizer_utils = _module
utils = _module
batch_sampler = _module
downloader = _module
env = _module
log = _module
profiler = _module
tools = _module
seq2seq_model = _module
simbert_model = _module
t5_ch = _module
utils = _module
bart_auto_title_train = _module
bert_couplet_paddle_train = _module
mBart_translation_en_ro = _module
roberta_autotitle_paddle_train = _module
roberta_math_paddle_train = _module
gpt2_ancient_translation_train = _module
gpt2_english_story_train = _module
gpt2_explain_dream_train = _module
gpt2_generate_article = _module
nezha_auto_title_train = _module
nezha_couplets_train = _module
nezha_relation_extract_train = _module
relationship_classify_train = _module
roberta_THUCNews_auto_title = _module
roberta_auto_title_train = _module
roberta_coarsness_NER_CRF_train = _module
roberta_coarsness_NER_train = _module
roberta_couplets_train = _module
roberta_fine_grained_NER_CRF_train = _module
roberta_large_auto_article_gen = _module
roberta_large_auto_title_train = _module
roberta_math_ques_train = _module
roberta_medical_ner_train = _module
roberta_news_classification_train = _module
roberta_participle_CRF_train = _module
roberta_poem_train = _module
roberta_relation_extract_train = _module
roberta_semantic_matching_train = _module
simbert_train = _module
t5_ancient_translation_train = _module
t5_auto_title_train = _module
setup = _module
auto_title_test = _module
bert_english_autotitle_test = _module
english_t5_test = _module
get_bert_embedding = _module
gpt_ancient_translation_test = _module
gpt_english_story_test = _module
gpt_explain_dream_test = _module
gpt_test_english = _module
nezha_auto_title_test = _module
nezha_relation_extract_test = _module
poem_test = _module
relation_extract_test = _module
semantic_matching_test = _module
t5_chinese_autotitle_test = _module
t5_chinese_test = _module
bert_couplet_test_paddle = _module
roberta_autotitle_test_paddle = _module
roberta_math_test_paddle = _module

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


import torch


import torch.nn.functional as F


import torch.nn as nn


import random


import time


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from typing import List


import math


import warnings


from typing import Optional


from typing import Tuple


import torch.utils.checkpoint


from torch import nn


from torch.nn import CrossEntropyLoss


import logging


from torch._C import device


from torch.nn import MSELoss


import re


from typing import Any


from typing import Callable


from typing import Dict


from typing import Set


from typing import Union


import copy


import pandas as pd


from torch.utils import data


def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


def _is_punctuation(ch):
    """标点符号类字符判断（全/半角均在此内）
    """
    code = ord(ch)
    return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126 or unicodedata.category(ch).startswith('P')


def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    Args:
        text (str|bytes): Text to be converted to unicode.
    Returns:
        str: converted text.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError('Unsupported string type: %s' % type(text))


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a peice of text.
    Args:
        text (str): Text to be tokened.
    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing.
            Defaults to `True`.

    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer."""
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text using basic tokenizer.

        Args:
            text (str): A piece of text.

        Returns: 
            list(str): A list of tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import BasicTokenizer
                basictokenizer = BasicTokenizer()
                tokens = basictokenizer.tokenize('He was a puppeteer')
                '''
                ['he', 'was', 'a', 'puppeteer']
                '''

        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text.
        """
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """
        Splits punctuation on a piece of text.
        """
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
        """
        Adds whitespace around any CJK character.
        """
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
        """
        Checks whether CP is the codepoint of a CJK character.
        """
        if cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or cp >= 131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or cp >= 177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or cp >= 63744 and cp <= 64255 or cp >= 194560 and cp <= 195103:
            return True
        return False

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
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


class Tokenizer(BasicTokenizer):

    def __init__(self, token_dict):
        """初始化
        """
        super(Tokenizer, self).__init__()
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        for token in ['pad', 'cls', 'sep', 'unk', 'mask']:
            try:
                _token_id = token_dict[getattr(self, '_token_' + str(token))]
                setattr(self, '_token_' + str(token) + '_id', _token_id)
                self.token_start_id = self._token_cls_id
                self.token_end_id = self._token_sep_id
            except Exception as e:
                pass
        self._vocab_size = len(token_dict)

    def token_to_id(self, token):
        """token转换为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        return self._token_dict_inv[i]

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = ''.join([c for c in ch if not (ord(c) == 0 or ord(c) == 65533 or self._is_control(c))])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))
        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]
        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token
        text = re.sub(' +', ' ', text)
        text = re.sub("' (re|m|s|t|ve|d|ll) ", "'\\1 ", text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\\d\\.) (\\d)', '\\1\\2', text)
        return text.strip()

    def _tokenize(self, text):
        """基本分词函数
        """
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 65533 or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))
        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self._token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        """
        code = ord(ch)
        return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126 or unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 19968 <= code <= 40959 or 13312 <= code <= 19903 or 131072 <= code <= 173791 or 173824 <= code <= 177983 or 177984 <= code <= 178207 or 178208 <= code <= 183983 or 63744 <= code <= 64255 or 194560 <= code <= 195103

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and ch[0] == '[' and ch[-1] == ']'


def get_model(model_name, word2ix):
    if model_name == 'roberta':
        config = BertConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)
    elif model_name == 'bert':
        config = BertConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)
    elif model_name == 'nezha':
        config = BertConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)
    elif model_name == 'roberta-large':
        config = RobertaLargeConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)
    else:
        raise Exception('model_name_err')
    return config, bert, layer_norm_cond, CLS


class BasicBert(nn.Module):

    def __init__(self, word2ix, model_name='roberta', tokenizer=None):
        super().__init__()
        self.config = ''
        self.word2ix = word2ix
        if tokenizer is None:
            self.tokenizer = Tokenizer(word2ix)
        else:
            self.tokenizer = tokenizer
        self.model_name = model_name
        self.config, self.bert, self.layer_norm_cond, self.cls = get_model(model_name, word2ix)
        self.device = torch.device('cpu')

    def load_pretrain_params(self, pretrain_model_path, keep_tokens=None, strict=False):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {k: v for k, v in checkpoint.items()}
        if keep_tokens is not None:
            embedding_weight_name = 'bert.embeddings.word_embeddings.weight'
            cls_pre_weight = 'cls.predictions.decoder.weight'
            cls_pre_bias = 'cls.predictions.bias'
            checkpoint[embedding_weight_name] = checkpoint[embedding_weight_name][keep_tokens]
            checkpoint[cls_pre_weight] = checkpoint[cls_pre_weight][keep_tokens]
            checkpoint[cls_pre_bias] = checkpoint[cls_pre_bias][keep_tokens]
        self.load_state_dict(checkpoint, strict=strict)
        torch.cuda.empty_cache()
        None

    def load_all_params(self, model_path, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        None

    def forward(self, input_text):
        input_ids, _ = self.tokenizer.encode(input_text, max_length=512)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).view(1, -1)
        enc_layers, _ = self.bert(input_ids, position_ids=None, token_type_ids=None, output_all_encoded_layers=True)
        squence_out = enc_layers[-1]
        tokens_hidden_state, _ = self.cls(squence_out)
        return tokens_hidden_state

    def set_device(self, device):
        self.device = torch.device(device)
        self

    def save_all_params(self, save_path):
        if self.model_name == 'nezha':
            checkpoints = {k: v for k, v in self.state_dict().items() if 'relative' not in k}
            torch.save(checkpoints, save_path)
            return
        torch.save(self.state_dict(), save_path)


class BasicGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {('model.' + k): v for k, v in checkpoint.items()}
        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        None

    def load_all_params(self, model_path, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        None

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


class BasicT5(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {('model.' + k): v for k, v in checkpoint.items()}
        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        None

    def load_all_params(self, model_path, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        None

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


class BasicBart(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {('model.' + k): v for k, v in checkpoint.items()}
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        None

    def load_all_params(self, model_path, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        None

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


class BertClsClassifier(BasicBert):
    """
    """

    def __init__(self, word2ix, target_size, model_name='roberta'):
        super(BertClsClassifier, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)

    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.target_size)
        labels = labels.view(-1)
        loss = nn.CrossEntropyLoss(reduction='mean')
        return loss(predictions, labels)

    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        text = text
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        all_layers, pooled_out = self.bert(text, output_all_encoded_layers=True)
        if use_layer_num != -1:
            pooled_out = all_layers[use_layer_num][:, 0]
        predictions = self.final_dense(pooled_out)
        if labels is not None:
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions


class BertClsMultiClassifier(BasicBert):
    """
    """

    def __init__(self, word2ix, target_size, model_name='roberta'):
        super(BertClsMultiClassifier, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)

    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        batch_size = predictions.shape[0]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        return loss(predictions, labels).sum() / batch_size

    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        text = text
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        all_layers, pooled_out = self.bert(text, output_all_encoded_layers=True)
        if use_layer_num != -1:
            pooled_out = all_layers[use_layer_num][:, 0]
        predictions = self.final_dense(pooled_out)
        if labels is not None:
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions


class ClsMultiSeq2SeqModel(BasicBert):
    """
    """

    def __init__(self, word2idx, target, model_name='roberta'):
        super(ClsMultiSeq2SeqModel, self).__init__(word2ix=word2idx, model_name=model_name)
        self.target = target
        self.final_dense = nn.Linear(self.config.hidden_size, len(self.target))

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, len(self.target))
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        input_tensor = input_tensor
        token_type_id = token_type_id
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril()
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, output_all_encoded_layers=True)
        squence_out = enc_layers[-1]
        tokens_hidden_state, _ = self.cls(squence_out)
        predictions = self.final_dense(tokens_hidden_state)
        if labels is not None:
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=40, beam_size=1, is_poem=False, max_length=256):
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        try:
            token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        except:
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=input_max_length, truncation=True)
            token_ids = tokenizer_out['input_ids']
            token_type_ids = tokenizer_out['token_type_ids']
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
        out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device='cpu'):
        """
        beam-search操作
        """
        sep_id = word2ix['[SEP]']
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score = output_scores.view(-1, 1) + logit_score
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = hype_pos // scores.shape[-1]
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                end_counts = (output_ids == sep_id).sum(1)
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    return output_ids[best_one][:-1]
                else:
                    flag = end_counts < 1
                    if not flag.all():
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        end_counts = end_counts[flag]
                        beam_size = flag.sum()
            return output_ids[output_scores.argmax()]


class BertRelationExtrac(BasicBert):
    """
    """

    def __init__(self, word2ix, predicate_num, model_name='roberta'):
        super(BertRelationExtrac, self).__init__(word2ix=word2ix, model_name=model_name)
        self.predicate_num = predicate_num
        self.subject_pred_norm = nn.LayerNorm(self.config.hidden_size)
        self.subject_pred = nn.Linear(self.config.hidden_size, 2)
        self.activation = nn.Sigmoid()
        self.object_pred = nn.Linear(self.config.hidden_size, 2 * self.predicate_num)

    def binary_crossentropy(self, labels, pred):
        labels = labels.float()
        loss = -labels * torch.log(pred) - (1.0 - labels) * torch.log(1.0 - pred)
        return loss

    def compute_total_loss(self, subject_pred, object_pred, subject_labels, object_labels):
        """
        计算loss
        """
        subject_loss = self.binary_crossentropy(subject_labels, subject_pred)
        subject_loss = torch.mean(subject_loss, dim=2)
        subject_loss = (subject_loss * self.target_mask).sum() / self.target_mask.sum()
        object_loss = self.binary_crossentropy(object_labels, object_pred)
        object_loss = torch.mean(object_loss, dim=3).sum(dim=2)
        object_loss = (object_loss * self.target_mask).sum() / self.target_mask.sum()
        return subject_loss + object_loss

    def extrac_subject(self, output, subject_ids):
        batch_size = output.shape[0]
        hidden_size = output.shape[-1]
        start_end = torch.gather(output, index=subject_ids.unsqueeze(-1).expand((batch_size, 2, hidden_size)), dim=1)
        subject = torch.cat((start_end[:, 0], start_end[:, 1]), dim=-1)
        return subject

    def forward(self, text, subject_ids, position_enc=None, subject_labels=None, object_labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            raise Exception('目前 use_layer_num 只支持-1')
        text = text
        subject_ids = subject_ids
        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)
        subject_pred_out = self.subject_pred(self.subject_pred_norm(tokens_hidden_state))
        subject_pred_act = self.activation(subject_pred_out)
        subject_pred_act = subject_pred_act ** 2
        subject_vec = self.extrac_subject(tokens_hidden_state, subject_ids)
        object_layer_norm = self.layer_norm_cond([tokens_hidden_state, subject_vec])
        object_pred_out = self.object_pred(object_layer_norm)
        object_pred_act = self.activation(object_pred_out)
        object_pred_act = object_pred_act ** 4
        batch_size, seq_len, target_size = object_pred_act.shape
        object_pred_act = object_pred_act.reshape((batch_size, seq_len, int(target_size / 2), 2))
        predictions = object_pred_act
        if subject_labels is not None and object_labels is not None:
            subject_labels = subject_labels
            object_labels = object_labels
            loss = self.compute_total_loss(subject_pred_act, object_pred_act, subject_labels, object_labels)
            return predictions, loss
        else:
            return predictions

    def predict_subject(self, text, use_layer_num=-1):
        if use_layer_num != -1:
            raise Exception('use_layer_num目前只支持-1')
        text = text
        self.target_mask = (text > 0).float()
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)
        subject_pred_out = self.subject_pred(self.subject_pred_norm(tokens_hidden_state))
        subject_pred_act = self.activation(subject_pred_out)
        subject_pred_act = subject_pred_act ** 2
        return subject_pred_act

    def predict_object_predicate(self, text, subject_ids, use_layer_num=-1):
        if use_layer_num != -1:
            raise Exception('use_layer_num目前只支持-1')
        text = text
        subject_ids = subject_ids
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)
        subject_vec = self.extrac_subject(tokens_hidden_state, subject_ids)
        object_layer_norm = self.layer_norm_cond([tokens_hidden_state, subject_vec])
        object_pred_out = self.object_pred(object_layer_norm)
        object_pred_act = self.activation(object_pred_out)
        object_pred_act = object_pred_act ** 4
        batch_size, seq_len, target_size = object_pred_act.shape
        object_pred_act = object_pred_act.view((batch_size, seq_len, int(target_size / 2), 2))
        predictions = object_pred_act
        return predictions


class BertSeqLabeling(BasicBert):
    """
    """

    def __init__(self, word2ix, target_size, model_name='roberta'):
        super(BertSeqLabeling, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)

    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.target_size)
        labels = labels.view(-1)
        self.target_mask = self.target_mask.view(-1)
        loss = nn.CrossEntropyLoss(reduction='none')
        return (loss(predictions, labels) * self.target_mask).sum() / self.target_mask.sum()

    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                raise Exception('层数选择错误，因为bert base模型共8层，所以参数只只允许0 - 7， 默认为-1，取最后一层')
        self.target_mask = (text > 0).float()
        text = text
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)
        predictions = self.final_dense(tokens_hidden_state)
        if labels is not None:
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions


class CRFLayer(nn.Module):
    """
    """

    def __init__(self, output_dim):
        super(CRFLayer, self).__init__()
        self.output_dim = output_dim
        self.trans = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.trans.data.uniform_(-0.1, 0.1)

    def compute_loss(self, y_pred, y_true, mask):
        """
        计算CRF损失
        """
        y_pred = y_pred * mask
        y_true = y_true * mask
        target_score = self.target_score(y_pred, y_true)
        log_norm = self.log_norm_step(y_pred, mask)
        log_norm = self.logsumexp(log_norm, dim=1)
        return log_norm - target_score

    def forward(self, y_pred, y_true, mask):
        """
        y_true: [[1, 2, 3], [2, 3, 0] ]
        mask: [[1, 1, 1], [1, 1, 0]]
        """
        if y_pred.shape[0] != mask.shape[0] or y_pred.shape[1] != mask.shape[1]:
            raise Exception('mask shape is not match to y_pred shape')
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        mask = mask.float()
        y_true = y_true.reshape(y_pred.shape[:-1])
        y_true = y_true.long()
        y_true_onehot = F.one_hot(y_true, self.output_dim)
        y_true_onehot = y_true_onehot.float()
        return self.compute_loss(y_pred, y_true_onehot, mask)

    def target_score(self, y_pred, y_true):
        """
        计算状态标签得分 + 转移标签得分
        y_true: (batch, seq_len, out_dim)
        y_pred: (batch, seq_len, out_dim)
        """
        point_score = torch.einsum('bni,bni->b', y_pred, y_true)
        trans_score = torch.einsum('bni,ij,bnj->b', y_true[:, :-1], self.trans, y_true[:, 1:])
        return point_score + trans_score

    def log_norm_step(self, y_pred, mask):
        """
        计算归一化因子Z(X)
        """
        state = y_pred[:, 0]
        y_pred = y_pred[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()
        batch, seq_len, out_dim = y_pred.shape
        for t in range(seq_len):
            cur_mask = mask[:, t]
            state = torch.unsqueeze(state, 2)
            g = torch.unsqueeze(self.trans, 0)
            outputs = self.logsumexp(state + g, dim=1)
            outputs = outputs + y_pred[:, t]
            outputs = cur_mask * outputs + (1 - cur_mask) * state.squeeze(-1)
            state = outputs
        return outputs

    def logsumexp(self, x, dim=None, keepdim=False):
        """
        避免溢出
        """
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        out = xm + torch.log(torch.sum(torch.exp(x - xm), dim=dim, keepdim=True))
        return out if keepdim else out.squeeze(dim)


class BertSeqLabelingCRF(BasicBert):
    """
    """

    def __init__(self, word2ix, target_size, model_name='roberta'):
        super(BertSeqLabelingCRF, self).__init__(word2ix=word2ix, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)
        self.crf_layer = CRFLayer(self.target_size)

    def compute_loss(self, predictions, labels):
        """
        计算loss
        """
        loss = self.crf_layer(predictions, labels, self.target_mask)
        return loss.mean()

    def forward(self, text, position_enc=None, labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            raise Exception('use_layer_num目前只支持-1')
        self.target_mask = (text > 0).float()
        text = text
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        enc_layers, _ = self.bert(text, output_all_encoded_layers=True)
        squence_out = enc_layers[use_layer_num]
        tokens_hidden_state, _ = self.cls(squence_out)
        predictions = self.final_dense(tokens_hidden_state)
        if labels is not None:
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions


class GPT2Config:

    def __init__(self, vocab_size=21128, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, gradient_checkpointing=False, use_cache=True):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cache = use_cache
        self.add_cross_attention = False
        self.use_return_dict = False
        self.output_attentions = False
        self.output_hidden_states = False

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer('masked_bias', torch.tensor(-10000.0))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / float(v.size(-1)) ** 0.5
        nd, ns = w.size(-2), w.size(-1)
        if not self.is_cross_attention:
            mask = self.bias[:, :, ns - nd:ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias)
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = torch.matmul(w, v),
        if output_attentions:
            outputs += w,
        return outputs

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

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        if encoder_hidden_states is not None:
            assert hasattr(self, 'q_attn'), 'If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`.'
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = key.transpose(-2, -1), value
        else:
            present = None
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return (a, present) + attn_outputs[1:]


gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def linear_act(x):
    return x


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


ACT2FN = {'relu': F.relu, 'gelu': gelu, 'tanh': torch.tanh, 'gelu_new': gelu_new, 'gelu_fast': gelu_fast, 'mish': mish, 'linear': linear_act, 'sigmoid': torch.sigmoid}


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        attn_outputs = self.attn(self.ln_1(hidden_states), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + hidden_states
        if encoder_hidden_states is not None:
            assert hasattr(self, 'crossattention'), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'
            cross_attn_outputs = self.crossattention(self.ln_cross_attn(hidden_states), attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
            attn_output = cross_attn_outputs[0]
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.config = config
        self.device_map = None

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        if attention_mask is not None:
            assert batch_size > 0, 'batch_size has to be defined and > 0'
            attention_mask = (1.0 - attention_mask) * -10000.0
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = encoder_batch_size, encoder_sequence_length
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=None, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states


class GPT2LMHeadModel(nn.Module):
    _keys_to_ignore_on_load_missing = ['h\\.\\d+\\.attn\\.masked_bias', 'lm_head\\.weight']

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.model_parallel = False
        self.device_map = None

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get('token_type_ids', None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {'input_ids': input_ids, 'past_key_values': past, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, lm_logits


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) ->torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.')


class ListProcessor(LogitsProcessor):

    def __init__(self, list_processor: List[LogitsProcessor]) ->None:
        super().__init__()
        self.list_processor = list_processor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) ->torch.FloatTensor:
        for processor in self.list_processor:
            scores = processor(input_ids, scores)
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(f'`penalty` has to be a strictly positive float, but is {penalty}')
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) ->torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores


class TemperatureLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).
    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not temperature > 0:
            raise ValueError(f'`temperature` has to be a strictly positive float, but is {temperature}')
        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) ->torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class TopKLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f'`top_k` has to be a strictly positive integer, but is {top_k}')
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) ->torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.
    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to top_p or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f'`top_p` has to be a float > 0 and < 1, but is {top_p}')
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) ->torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class GPT2(BasicGPT):

    def __init__(self, word2ix, tokenizer=None):
        super().__init__()
        self.word2ix = word2ix
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(word2ix)
        self.config = GPT2Config(len(word2ix))
        self.model = GPT2LMHeadModel(self.config)

    def sample_generate(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=1.0, add_eos=False, repetition_penalty=1.0, temperature=1.0):
        lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty), TemperatureLogitsProcessor(temperature=temperature), TopKLogitsProcessor(top_k=top_k), TopPLogitsProcessor(top_p=top_p)]
        self.list_processor = ListProcessor(lp)
        token_ids, _ = self.tokenizer.encode(text, max_length=input_max_length)
        if not add_eos:
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)[:-1].view(1, -1)
        else:
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []
        sep_id = self.word2ix['[SEP]']
        with torch.no_grad():
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score[:, self.word2ix['[UNK]']] = -float('Inf')
                filtered_logits = self.list_processor(token_ids, logit_score)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long()), dim=1)
        return self.tokenizer.decode(np.array(output_ids))

    def sample_generate_once(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, sep='。'):
        token_ids, _ = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)[1:-1].view(1, -1)
        output_ids = []
        sep_id = self.word2ix[sep]
        with torch.no_grad():
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.word2ix['[UNK]']] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
        return self.tokenizer.decode(np.array(output_ids))

    def sample_generate_english(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):
        token_ids = self.tokenizer.encode(text, max_length=input_max_length, truncation=True)
        if add_eos:
            token_ids = token_ids + [self.word2ix['<EOS>']]
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []
        sep_id = self.word2ix['<EOS>']
        with torch.no_grad():
            for step in range(out_max_length):
                _, scores = self.model(token_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.word2ix['unk']] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
        return self.tokenizer.decode(output_ids)

    def _make_causal_mask(self, input_ids_shape: torch.Size):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), 0.0)
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1.0)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    def forward(self, x, labels=None):
        if labels is not None:
            labels = labels
        x = x
        attention_mask = self._make_causal_mask(x.shape)
        pad_mask = (labels != -100).float()
        attention_mask = attention_mask * pad_mask.unsqueeze(1).unsqueeze(1)
        loss, lm_logit = self.model(x, labels=labels, attention_mask=attention_mask)
        return loss, lm_logit


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, '`padding_idx` should not be None, but of type int'
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).'
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}'
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len), f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}'
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}'
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}'
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BartConfig:
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BartModel`. It is used to
    instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BART `facebook/bart-large
    <https://huggingface.co/facebook/bart-large>`__ architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BartModel` or
            :class:`~transformers.TFBartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``), only
            :obj:`True` for `bart-large-cnn`.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels: (:obj:`int`, `optional`, defaults to 3):
            The number of labels to use in :class:`~transformers.BartForSequenceClassification`.
    Example::
        >>> from transformers import BartModel, BartConfig
        >>> # Initializing a BART facebook/bart-large style configuration
        >>> configuration = BartConfig()
        >>> # Initializing a model from the facebook/bart-large style configuration
        >>> model = BartModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'bart'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=21128, max_position_embeddings=512, encoder_layers=6, encoder_ffn_dim=3072, encoder_attention_heads=12, decoder_layers=6, decoder_ffn_dim=3072, decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0, activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, force_bos_token_to_be_generated=False, use_cache=True, num_labels=3, pad_token_id=0, bos_token_id=101, eos_token_id=102, is_encoder_decoder=True, decoder_start_token_id=102):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding
        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated
        self.output_attentions = False

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


class BartEncoderLayer(nn.Module):

    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer_head_mask: torch.Tensor, output_attentions: bool=False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


class BartDecoderLayer(nn.Module):

    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, encoder_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=encoder_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights, cross_attn_weights
        if use_cache:
            outputs += present_key_value,
        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int]=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is None:
            attention_mask = (input_ids > 0).float()
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        else:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        return hidden_states


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int=0):
    """
    可以用于cross attention return (tgt_len, tgt_len + past_key_values_len) , row is output, column is input.
    生成一个下三角的mask，lm model 用。
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float('-inf'))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model, self.padding_idx)
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, encoder_head_mask=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        device = input_ids.device
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length)
            combined_attention_mask = combined_attention_mask
        if attention_mask is not None and combined_attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        positions = self.embed_positions(input_shape, past_key_values_length)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += hidden_states,
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(hidden_states, attention_mask=combined_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, encoder_layer_head_mask=encoder_head_mask[idx] if encoder_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[3 if output_attentions else 1],
            if output_attentions:
                all_self_attns += layer_outputs[1],
                all_cross_attentions += layer_outputs[2],
        if output_hidden_states:
            all_hidden_states += hidden_states,
        next_cache = next_decoder_cache if use_cache else None
        return hidden_states


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class BartModel(nn.Module):

    def __init__(self, config: BartConfig):
        super().__init__()
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.config = config
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = False
        output_hidden_states = False
        use_cache = False
        return_dict = False
        if attention_mask is None:
            attention_mask = (input_ids > 0).float()
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if decoder_attention_mask is None:
            decoder_attention_mask = (decoder_input_ids > 0).float()
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, encoder_head_mask=head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return decoder_outputs, encoder_outputs


class BartForConditionalGeneration(nn.Module):

    def __init__(self, config: BartConfig):
        super().__init__()
        self.model = BartModel(config)
        self.config = config
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = True
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        decoder_out, encoder_out = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_out)
        target_mask = (decoder_input_ids > 0).float().view(-1)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1)) * target_mask / target_mask.sum()
        output = lm_logits,
        return (masked_lm_loss,) + output if masked_lm_loss is not None else output


class BertLayerNorm(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第3部分"""

    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.conditional = conditional
        if conditional == True:
            self.weight_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.weight_dense.weight.data.uniform_(0, 0)
            self.bias_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.bias_dense.weight.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional == False:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        else:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            weight = self.weight + self.weight_dense(cond)
            bias = self.bias + self.bias_dense(cond)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.variance_epsilon)
            return weight * x + bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertConfig(object):

    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12):
        self.vocab_size = vocab_size
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


class BertSelfAttention(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
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

    def forward(self, hidden_states, attention_mask, output_attentions=False):
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
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
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
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        self_outputs, attention_metrix = self.self(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output, attention_metrix


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
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
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        attention_output, attention_matrix = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_matrix


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, output_attentions=False):
        all_encoder_layers = []
        all_attention_matrices = []
        for i, layer_module in enumerate(self.layer):
            layer_output, attention_matrix = layer_module(hidden_states, attention_mask, output_attentions=output_attentions)
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrix)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrix)
        return all_encoder_layers, all_attention_matrices


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Predictions(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform(x)
        return x, self.decoder(x)


class CLS(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = Predictions(config)

    def forward(self, x):
        return self.predictions(x)


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class RobertaLargeConfig(object):

    def __init__(self, vocab_size, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12):
        self.vocab_size = vocab_size
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


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig) and not isinstance(config, RobertaLargeConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, output_all_encoded_layers=True, output_attentions=False):
        extended_attention_mask = (input_ids > 0).float()
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        if attention_mask is not None:
            extended_attention_mask = attention_mask * extended_attention_mask
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_layers, all_attention_matrices = self.encoder(embedding_output, attention_mask=extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, output_attentions=output_attentions)
        sequence_output = encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if output_attentions:
            return all_attention_matrices
        if not output_all_encoded_layers:
            encoder_layers = encoder_layers[-1]
        return encoder_layers, pooled_output


class NeZhaEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.use_relative_position = config.use_relative_position
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RelativePositionsEncoding(nn.Module):

    def __init__(self, length, depth, max_relative_position=127):
        super(RelativePositionsEncoding, self).__init__()
        vocab_size = max_relative_position * 2 + 1
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        embeddings_table = torch.zeros(vocab_size, depth)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
        embeddings_table[:, 0::2] = torch.sin(position * div_term)
        embeddings_table[:, 1::2] = torch.cos(position * div_term)
        embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)
        flat_relative_positions_matrix = final_mat.view(-1)
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix, num_classes=vocab_size).float()
        positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
        my_shape = list(final_mat.size())
        my_shape.append(depth)
        positions_encoding = positions_encoding.view(my_shape)
        self.register_buffer('positions_encoding', positions_encoding)

    def forward(self, length):
        return self.positions_encoding[:length, :length, :]


class NeZhaSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.relative_positions_encoding = RelativePositionsEncoding(length=config.max_position_embeddings, depth=self.attention_head_size, max_relative_position=config.max_relative_position)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        batch_size, num_attention_heads, from_seq_length, to_seq_length = attention_scores.size()
        relations_keys = self.relative_positions_encoding(to_seq_length)
        query_layer_t = query_layer.permute(2, 0, 1, 3)
        query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, self.attention_head_size)
        key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
        key_position_scores_r = key_position_scores.view(from_seq_length, batch_size, num_attention_heads, from_seq_length)
        key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        relations_values = self.relative_positions_encoding(to_seq_length)
        attention_probs_t = attention_probs.permute(2, 0, 1, 3)
        attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, to_seq_length)
        value_position_scores = torch.matmul(attentions_probs_r, relations_values)
        value_position_scores_r = value_position_scores.view(from_seq_length, batch_size, num_attention_heads, self.attention_head_size)
        value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        context_layer = context_layer + value_position_scores_r_t
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = context_layer,
        return outputs


class NeZhaAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = NeZhaSelfAttention(config)
        self.pruned_heads = set()
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class NeZhaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = NeZhaAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class NeZhaEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([NeZhaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_all_encoded_layers=True):
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layer):
            if output_all_encoded_layers:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
        outputs = all_hidden_states + (hidden_states,)
        return outputs


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states
        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN['gelu_new']

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == 'relu':
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == 'gated-gelu':
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(f'{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`')
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Config:
    model_type = 't5'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50000, d_model=768, d_kv=64, d_ff=2048, num_layers=12, num_decoder_layers=12, num_heads=12, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='gated-gelu', is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1, is_decoder=False):
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class T5Attention(nn.Module):

    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets)
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length
        if past_key_value is not None:
            assert len(past_key_value) == 2, 'past_key_value should have 2 past states: keys and values. Got {} past states'.format(len(past_key_value))
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """  projection """
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """  reshape """
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """ projects hidden states correctly to key/query states """
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states))
            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    hidden_states = past_key_value
            return hidden_states
        query_states = shape(self.q(hidden_states))
        key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros((1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype)
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]
            if mask is not None:
                position_bias = position_bias + mask
        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        present_key_value_state = (key_states, value_states) if self.is_decoder and use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, encoder_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, return_dict=True):
        if past_key_value is not None:
            assert self.is_decoder, 'Only decoder can use `past_key_values`'
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            error_message = 'There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states'.format(expected_num_past_key_values, '2 (past / key) for cross attention' if expected_num_past_key_values == 4 else '', len(past_key_value))
            assert len(past_key_value) == expected_num_past_key_values, error_message
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, layer_head_mask=encoder_layer_head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = cross_attention_outputs[0]
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states)
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs


class T5Stack(nn.Module):

    def __init__(self, config, embed_tokens=None):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList([T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat([torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype), causal_mask], axis=-1)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError('Wrong shape for input_ids (shape {}) or attention_mask (shape {})'.format(input_shape, attention_mask.shape))
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.
        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1000000000.0
        return encoder_extended_attention_mask

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, encoder_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        input_shape = input_ids.size()
        if inputs_embeds is None:
            assert self.embed_tokens is not None, 'You have to initialize the model with valid token embeddings'
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            assert self.is_decoder, ':obj:`use_cache` can only be set to `True` if {} is used as a decoder'.format(self)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        present_key_value_states = () if use_cache else None
        all_hidden_states = None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.is_decoder else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = None
            encoder_layer_head_mask = None
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if attention_mask is not None:
                    attention_mask = attention_mask
                if position_bias is not None:
                    position_bias = position_bias
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask
                if encoder_layer_head_mask is not None:
                    encoder_layer_head_mask = encoder_layer_head_mask
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, layer_head_mask=None, encoder_layer_head_mask=None, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and 'cuda:' + str(k) != self.last_device:
                        hidden_states = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states,


class T5ForConditionalGeneration(nn.Module):
    _keys_to_ignore_on_load_missing = ['encoder\\.embed_tokens\\.weight', 'decoder\\.embed_tokens\\.weight', 'lm_head\\.weight']
    _keys_to_ignore_on_load_unexpected = ['decoder\\.block\\.0\\.layer\\.1\\.EncDecAttention\\.relative_attention_bias\\.weight']

    def __init__(self, config):
        super().__init__()
        self.model_dim = config.d_model
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        if past_key_values is not None:
            assert labels is None, 'Decoder should not use cached key value states when training.'
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, encoder_head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss, lm_logits
        return lm_logits,

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {'decoder_input_ids': input_ids, 'past_key_values': past, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)


class T5PegasusTokenizer(Tokenizer):

    def __init__(self, token_dict, pre_tokenizer=lambda x: jieba.cut(x, HMM=False)):
        super().__init__(token_dict)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self._token_dict:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class T5SmallConfig:
    model_type = 't5'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50000, d_model=512, d_kv=64, d_ff=1024, num_layers=8, num_decoder_layers=8, num_heads=6, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='gated-gelu', is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1, is_decoder=False):
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class T5Model(BasicT5):

    def __init__(self, word2idx, size='base'):
        super().__init__()
        if size == 'base':
            config = T5Config()
        elif size == 'small':
            config = T5SmallConfig()
        else:
            raise Exception('not support this model type')
        self.model = T5ForConditionalGeneration(config)
        self.word2idx = word2idx
        self.tokenizer = T5PegasusTokenizer(self.word2idx)
        self.bos_id = self.word2idx['[CLS]']
        self.eos_id = self.word2idx['[SEP]']
        self.unk_id = self.word2idx['[UNK]']

    def forward(self, input_ids, decoder_input_ids, labels=None):
        input_ids = input_ids
        decoder_input_ids = decoder_input_ids
        if labels is not None:
            labels = labels
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)

    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=1.0, add_eos=True):
        token_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(token_out) == 2:
            token_ids = token_out[0]
        else:
            token_ids = token_out
        if not add_eos:
            token_ids = token_ids[:-1]
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []
        input_decoder_ids = torch.tensor(self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.model(input_ids=token_ids, decoder_input_ids=input_decoder_ids)[0]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.unk_id] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                filterd_logits_prob = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(filterd_logits_prob, num_samples=1)
                if self.eos_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)
        return self.tokenizer.decode(output_ids)


yayun_list = ['东同铜桐筒童僮瞳中衷忠虫终戎崇嵩弓躬宫融雄熊穹穷冯风枫丰充隆空公功工攻蒙笼聋珑洪红鸿虹丛翁聪通蓬烘潼胧砻峒螽梦讧冻忡酆恫总侗窿懵庞种盅芎倥艨绒葱匆骢', '冬农宗钟龙舂松冲容蓉庸封胸雍浓重从逢缝踪茸峰锋烽蛩慵恭供淙侬松凶墉镛佣溶邛共憧喁邕壅纵龚枞脓淞匈汹禺蚣榕彤', '江扛窗邦缸降双庞逄腔撞幢桩淙豇', '支枝移为垂吹陂碑奇宜仪皮儿离施知驰池规危夷师姿迟眉悲之芝时诗棋旗辞词期祠基疑姬丝司葵医帷思滋持随痴维卮麋螭麾墀弥慈遗肌脂雌披嬉尸狸炊篱兹差疲茨卑亏蕤陲骑曦歧岐谁斯私窥熙欺疵赀笞羁彝颐资糜饥衰锥姨楣夔涯伊蓍追', '缁箕椎罴篪萎匙脾坻嶷治骊尸綦怡尼漪累牺饴而鸱推縻璃祁绥逵羲羸肢骐訾狮奇嗤咨堕其睢漓蠡噫馗辎胝鳍蛇陴淇淄丽筛厮氏痍貔比僖贻祺嘻鹂瓷琦嵋怩熹孜台蚩罹魑丕琪耆衰惟剂提禧居栀戏畸椅磁痿离佳虽仔寅委崎隋逶倭黎犁郦', '微薇晖徽挥韦围帏违霏菲妃绯飞非扉肥腓威畿机几讥矶稀希衣依沂巍归诽痱欷葳颀圻', '鱼渔初书舒居裾车渠余予誉舆胥狙锄疏蔬梳虚嘘徐猪闾庐驴诸除储如墟与畲疽苴于茹蛆且沮祛蜍榈淤好雎纾躇趄滁屠据匹咀衙涂虑', '虞愚娱隅刍无芜巫于盂衢儒濡襦须株诛蛛殊瑜榆谀愉腴区驱躯朱珠趋扶符凫雏敷夫肤纡输枢厨俱驹模谟蒲胡湖瑚乎壶狐弧孤辜姑觚菰徒途涂荼图屠奴呼吾七虞梧吴租卢鲈苏酥乌枯都铺禺诬竽吁瞿劬需俞逾觎揄萸臾渝岖镂娄夫孚桴俘迂姝拘摹糊鸪沽呱蛄驽逋舻垆徂孥泸栌嚅蚨诹扶母毋芙喁颅轳句邾洙麸机膜瓠恶芋呕驺喻枸侏龉葫懦帑拊', '齐蛴脐黎犁梨黧妻萋凄堤低氐诋题提荑缔折篦鸡稽兮奚嵇蹊倪霓西栖犀嘶撕梯鼙批挤迷泥溪圭闺睽奎携畦骊鹂儿', '佳街鞋牌柴钗差涯阶偕谐骸排乖怀淮豺侪埋霾斋娲蜗娃哇皆喈揩蛙楷槐俳', '灰恢魁隈回徊枚梅媒煤瑰雷催摧堆陪杯醅嵬推开哀埃台苔该才材财裁来莱栽哉灾猜胎孩虺崔裴培坏垓陔徕皑傀崃诙煨桅唉颏能茴酶偎隗咳', '真因茵辛新薪晨辰臣人仁神亲申伸绅身宾滨邻鳞麟珍尘陈春津秦频苹颦银垠筠巾民珉缗贫淳醇纯唇伦纶轮沦匀旬巡驯钧均臻榛姻寅彬鹑皴遵循振甄岷谆椿询恂峋莘堙屯呻粼磷辚濒闽豳逡填狺泯洵溱夤荀竣娠纫鄞抡畛嶙斌氤', '文闻纹云氛分纷芬焚坟群裙君军勤斤筋勋薰曛熏荤耘芸汾氲员欣芹殷昕贲郧雯蕲', '元原源园猿辕坦烦繁蕃樊翻萱喧冤言轩藩魂浑温孙门尊存蹲敦墩暾屯豚村盆奔论坤昏婚阍痕根恩吞沅媛援爰幡番反埙鸳宛掀昆琨鲲扪荪髡跟垠抡蕴犍袁怨蜿溷昆炖饨臀喷纯', '寒韩翰丹殚单安难餐滩坛檀弹残干肝竿乾阑栏澜兰看刊丸桓纨端湍酸团抟攒官观冠鸾銮栾峦欢宽盘蟠漫汗郸叹摊奸剜棺钻瘢谩瞒潘胖弁拦完莞獾拌掸萑倌繁曼馒鳗谰洹滦', '删潸关弯湾还环鹌鬟寰班斑颁般蛮颜菅攀顽山鳏艰闲娴悭孱潺殷扳讪患', '先前千阡笺天坚肩贤弦烟燕莲怜田填钿年颠巅牵妍研眠渊涓蠲编玄县泉迁仙鲜钱煎然延筵禅蝉缠连联涟篇偏便全宣镌穿川缘鸢铅捐旋娟船涎鞭专圆员乾虔愆骞权拳椽传焉跹溅舷咽零骈阗鹃翩扁平沿诠痊悛荃遄卷挛戋佃滇婵颛犍搴嫣癣澶单竣鄢扇键蜷棉', '萧箫挑貂刁凋雕迢条跳苕调枭浇聊辽寥撩僚寮尧幺宵消霄绡销超朝潮嚣樵谯骄娇焦蕉椒饶烧遥姚摇谣瑶韶昭招飚标杓镳瓢苗描猫要腰邀乔桥侨妖夭漂飘翘祧佻徼侥哨娆陶橇劭潇骁獠料硝灶鹞钊蛲峤轿荞嘹逍燎憔剽', '肴巢交郊茅嘲钞包胶爻苞梢蛟庖匏坳敲胞抛鲛崤铙炮哮捎茭淆泡跑咬啁教咆鞘剿刨佼抓姣唠', '豪毫操髦刀萄猱桃糟漕旄袍挠蒿涛皋号陶翱敖遭篙羔高嘈搔毛艘滔骚韬缫膏牢醪逃槽劳洮叨绸饕骜熬臊涝淘尻挑嚣捞嗥薅咎谣', '歌多罗河戈阿和波科柯陀娥蛾鹅萝荷过磨螺禾哥娑驼佗沱峨那苛诃珂轲莎蓑梭婆摩魔讹坡颇俄哦呵皤么涡窝茄迦伽磋跎番蹉搓驮献蝌箩锅倭罗嵯锣', '麻花霞家茶华沙车牙蛇瓜斜邪芽嘉瑕纱鸦遮叉葩奢楂琶衙赊涯夸巴加耶嗟遐笳差蟆蛙虾拿葭茄挝呀枷哑娲爬杷蜗爷芭鲨珈骅娃哇洼畲丫夸裟瘕些桠杈痂哆爹椰咤笆桦划迦揶吾佘', '阳杨扬香乡光昌堂章张王房芳长塘妆常凉霜藏场央泱鸯秧嫱床方浆觞梁娘庄黄仓皇装殇襄骧相湘箱缃创忘芒望尝偿樯枪坊囊郎唐狂强肠康冈苍匡荒遑行妨棠翔良航倡伥羌庆姜僵缰疆粮穰将墙桑刚祥详洋徉佯粱量羊伤汤鲂樟彰漳璋猖商防', '筐煌隍凰蝗惶璜廊浪裆沧纲亢吭潢钢丧盲簧忙茫傍汪臧琅当庠裳昂障糖疡锵杭邙赃滂禳攘瓤抢螳踉眶炀阊彭蒋亡殃蔷镶孀搪彷胱磅膀螃八庚更羹盲横觥彭棚亨英瑛烹平评京惊荆明盟鸣荣莹兵卿生甥笙牲檠擎鲸迎行衡耕萌氓宏闳茎莺樱泓橙筝争清情晴精睛菁旌晶盈瀛嬴营婴缨贞成盛城诚呈程声征正轻名令并倾萦琼赓撑瞠枪伧峥猩珩蘅铿嵘丁嘤鹦铮\ue3dc砰绷轰訇瞪侦顷榜抨趟坪请', '青经泾形刑邢型陉亭庭廷霆蜓停丁宁钉仃馨星腥醒惺娉灵棂龄铃苓伶零玲翎瓴囹聆听厅汀冥溟螟铭瓶屏萍荧萤荥扃町瞑暝', '蒸承丞惩陵凌绫冰膺鹰应蝇绳渑乘升胜兴缯凭仍兢矜征凝称登灯僧增曾憎层能棱朋鹏弘肱腾滕藤恒冯瞢扔誊', '尤邮优忧流留榴骝刘由油游猷悠攸牛修羞秋周州洲舟酬仇柔俦畴筹稠邱抽湫遒收鸠不愁休囚求裘球浮谋牟眸矛侯猴喉讴沤鸥瓯楼娄陬偷头投钩沟幽彪疣绸浏瘤犹啾酋售蹂揉搜叟邹貅泅球逑俅蜉桴罘欧搂抠髅蝼兜句妯惆呕缪繇偻篓馗区', '侵寻浔林霖临针箴斟沈深淫心琴禽擒钦衾吟今襟金音阴岑簪琳琛椹谌忱壬任黔歆禁喑森参淋郴妊湛', '覃潭谭参骖南男谙庵含涵函岚蚕探贪耽龛堪戡谈甘三酣篮柑惭蓝郯婪庵颔褴澹', '盐檐廉帘嫌严占髯谦奁纤签瞻蟾炎添兼缣尖潜阎镰粘淹箝甜恬拈暹詹渐歼黔沾苫占崦阉砭', '咸缄谗衔岩帆衫杉监凡馋芟喃嵌掺搀严']


class Seq2SeqModel(BasicBert):
    """
    """

    def __init__(self, word2ix, model_name='roberta', tokenizer=None):
        super(Seq2SeqModel, self).__init__(word2ix=word2ix, model_name=model_name, tokenizer=tokenizer)
        self.hidden_dim = self.config.hidden_size
        self.vocab_size = len(word2ix)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        input_tensor = input_tensor
        token_type_id = token_type_id
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril()
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, output_all_encoded_layers=True)
        squence_out = enc_layers[-1]
        tokens_hidden_state, predictions = self.cls(squence_out)
        if labels is not None:
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=40, beam_size=1, is_poem=False, max_length=256):
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        try:
            token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        except:
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=input_max_length, truncation=True)
            token_ids = tokenizer_out['input_ids']
            token_type_ids = tokenizer_out['token_type_ids']
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
        if is_poem:
            out_puts_ids = self.beam_search_poem(text, token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        else:
            out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def sample_generate(self, text, out_max_length=40, top_k=30, top_p=0.0, max_length=256, repetition_penalty=1.0, temperature=1.0):
        input_max_length = max_length - out_max_length
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty), TemperatureLogitsProcessor(temperature=temperature), TopKLogitsProcessor(top_k=top_k), TopPLogitsProcessor(top_p=top_p)]
        list_processor = ListProcessor(lp)
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long).view(1, -1)
        device = self.device
        output_ids = []
        sep_id = self.word2ix['[SEP]']
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.forward(token_ids, token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score[:, self.word2ix['[UNK]']] = -float('Inf')
                filtered_logits = list_processor(token_ids, logit_score)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long()), dim=1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
        return self.tokenizer.decode(np.array(output_ids))

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device='cpu'):
        """
        beam-search操作
        """
        sep_id = word2ix['[SEP]']
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score = output_scores.view(-1, 1) + logit_score
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = hype_pos // scores.shape[-1]
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                end_counts = (output_ids == sep_id).sum(1)
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    return output_ids[best_one][:-1]
                else:
                    flag = end_counts < 1
                    if not flag.all():
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        end_counts = end_counts[flag]
                        beam_size = flag.sum()
            return output_ids[output_scores.argmax()]

    def beam_search_poem(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device='cpu'):
        """
        beam-search操作
        """
        yayun_pos = []
        title = text.split('##')[0]
        if '五言律诗' in text:
            yayun_pos = [10, 22, 34, 46]
        elif '五言绝句' in text:
            yayun_pos = [10, 22]
        elif '七言律诗' in text:
            yayun_pos = [14, 30, 46, 62]
        elif '七言绝句' in text:
            yayun_pos = [14, 30]
        sep_id = word2ix['[SEP]']
        douhao_id = word2ix['，']
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix['。']
        repeat_word = [[] for i in range(beam_size)]
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = -1 * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                for i, char in enumerate(last_chars):
                    for word in repeat_word[i]:
                        logit_score[i, word] -= 5
                    for word in title:
                        ix = word2ix.get(word, -1)
                        if ix != -1:
                            logit_score[i, ix] += 2
                if step in yayun_pos:
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    logit_score[i, ix] += 10
                logit_score = output_scores.view(-1, 1) + logit_score
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = hype_pos // scores.shape[-1]
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    if each_out in repeat_word[index]:
                        pass
                    else:
                        repeat_word[index].append(each_out)
                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break
                output_scores = hype_score
                last_chars = indice2
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                end_counts = (output_ids == sep_id).sum(1)
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    return output_ids[best_one][:-1]
                else:
                    flag = end_counts < 1
                    if not flag.all():
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        end_counts = end_counts[flag]
                        beam_size = flag.sum()
                        flag = flag.long()
                        new_repeat_word = []
                        for index, i in enumerate(flag):
                            if i.item() == 1:
                                new_repeat_word.append(repeat_word[index])
                        repeat_word = new_repeat_word
            return output_ids[output_scores.argmax()]

    def beam_search_poem_v2(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device='cpu'):
        """
        beam-search操作
        """
        yayun_pos = []
        if '五言律诗' in text:
            yayun_pos = [10, 22, 34, 46]
        elif '五言绝句' in text:
            yayun_pos = [10, 22]
        elif '七言律诗' in text:
            yayun_pos = [14, 30, 46, 62]
        elif '七言绝句' in text:
            yayun_pos = [14, 30]
        sep_id = word2ix['[SEP]']
        douhao_id = word2ix['，']
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix['。']
        repeat_word = []
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = -1 * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                for i, char in enumerate(last_chars):
                    logit_score[i, char] -= 2
                    for word in repeat_word:
                        logit_score[i, word] -= 1
                if step in yayun_pos:
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    logit_score[i, ix] += 3
                logit_score = output_scores.view(-1, 1) + logit_score
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = hype_pos // scores.shape[-1]
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)
                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    if each_out in repeat_word:
                        pass
                    else:
                        repeat_word.append(each_out)
                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break
                output_scores = hype_score
                last_chars = indice2
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                end_counts = (output_ids == sep_id).sum(1)
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    return output_ids[best_one]
                else:
                    flag = end_counts < 1
                    if not flag.all():
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        end_counts = end_counts[flag]
                        beam_size = flag.sum()
                        flag = flag.long()
            return output_ids[output_scores.argmax()]


class SimBertModel(BasicBert):
    """
    """

    def __init__(self, word2ix, model_name='roberta', tokenizer=None):
        super(SimBertModel, self).__init__(word2ix=word2ix, model_name=model_name, tokenizer=tokenizer)
        self.word2ix = word2ix
        self.hidden_dim = self.config.hidden_size
        self.vocab_size = len(word2ix)

    def compute_loss(self, cls_token_state, predictions, labels, target_mask):
        loss1 = self.compute_loss_of_seq2seq(predictions, labels, target_mask)
        loss2 = self.compute_loss_of_similarity(cls_token_state)
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, predictions, labels, target_mask):
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()

    def compute_loss_of_similarity(self, y_pred):
        y_true = self.get_labels_of_similarity(y_pred)
        y_true = y_true
        norm_a = torch.nn.functional.normalize(y_pred, dim=-1, p=2)
        similarities = norm_a.matmul(norm_a.t())
        similarities = similarities - torch.eye(y_pred.shape[0]) * 1000000000000.0
        similarities = similarities * 20
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(similarities, y_true)
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = torch.arange(0, y_pred.shape[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = (idxs_1 == idxs_2).float().argmax(dim=-1).long()
        return labels

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None):
        input_tensor = input_tensor
        token_type_id = token_type_id
        if position_enc is not None:
            position_enc = position_enc
        if labels is not None:
            labels = labels
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril()
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, output_all_encoded_layers=True)
        squence_out = enc_layers[-1]
        sequence_hidden, predictions = self.cls(squence_out)
        if labels is not None:
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(sequence_hidden[0], predictions, labels, target_mask)
            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=40, beam_size=1, max_length=256):
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        try:
            token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        except:
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=input_max_length, truncation=True)
            token_ids = tokenizer_out['input_ids']
            token_type_ids = tokenizer_out['token_type_ids']
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
        out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=self.device)
        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def random_sample(self, inputs, n, topk=None, topp=None, states=None, temperature=1, min_ends=1):
        """随机采样n个结果
        说明：非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回：n个解码序列组成的list。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(inputs, output_ids, states, temperature, 'probas')
            probas /= probas.sum(axis=1, keepdims=True)
            if step == 0:
                probas = np.repeat(probas, n, axis=0)
                inputs = [np.repeat(i, n, axis=0) for i in inputs]
                output_ids = np.repeat(output_ids, n, axis=0)
            if topk is not None:
                k_indices = probas.argpartition(-topk, axis=1)[:, -topk:]
                probas = np.take_along_axis(probas, k_indices, axis=1)
                probas /= probas.sum(axis=1, keepdims=True)
            if topp is not None:
                p_indices = probas.argsort(axis=1)[:, ::-1]
                probas = np.take_along_axis(probas, p_indices, axis=1)
                cumsum_probas = np.cumsum(probas, axis=1)
                flag = np.roll(cumsum_probas >= topp, 1, axis=1)
                flag[:, 0] = False
                probas[flag] = 0
                probas /= probas.sum(axis=1, keepdims=True)
            sample_func = lambda p: np.random.choice(len(p), p=p)
            sample_ids = np.apply_along_axis(sample_func, 1, probas)
            sample_ids = sample_ids.reshape((-1, 1))
            if topp is not None:
                sample_ids = np.take_along_axis(p_indices, sample_ids, axis=1)
            if topk is not None:
                sample_ids = np.take_along_axis(k_indices, sample_ids, axis=1)
            output_ids = np.concatenate([output_ids, sample_ids], 1)
            is_end = output_ids[:, -1] == self.end_id
            end_counts = (output_ids == self.end_id).sum(1)
            if output_ids.shape[1] >= self.minlen:
                flag = is_end & (end_counts >= min_ends)
                if flag.any():
                    for ids in output_ids[flag]:
                        results.append(ids)
                    flag = flag == False
                    inputs = [i[flag] for i in inputs]
                    output_ids = output_ids[flag]
                    end_counts = end_counts[flag]
                    if len(output_ids) == 0:
                        break
        for ids in output_ids:
            results.append(ids)
        return results

    def sample_generate(self, text, out_max_length=40, top_k=30, top_p=0.0, max_length=256, repetition_penalty=1.0, temperature=1.0, sample_num=1):
        input_max_length = max_length - out_max_length
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        result_list = []
        lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty), TemperatureLogitsProcessor(temperature=temperature), TopKLogitsProcessor(top_k=top_k), TopPLogitsProcessor(top_p=top_p)]
        list_processor = ListProcessor(lp)
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long).view(1, -1)
        device = self.device
        output_ids = []
        sep_id = self.word2ix['[SEP]']
        with torch.no_grad():
            for step in range(out_max_length):
                if step == 0:
                    token_ids = token_ids.repeat((sample_num, 1))
                    token_type_ids = token_type_ids.repeat((sample_num, 1))
                scores = self.forward(token_ids, token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score[:, self.word2ix['[UNK]']] = -float('Inf')
                filtered_logits = list_processor(token_ids, logit_score)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if step == 0:
                    output_ids = next_token.view((sample_num, 1))
                else:
                    output_ids = torch.cat([output_ids, next_token.view((sample_num, 1))], dim=1)
                token_ids = torch.cat([token_ids, next_token.view((sample_num, 1)).long()], dim=1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((sample_num, 1), device=device, dtype=torch.long)], dim=1)
                is_end = output_ids[:, -1] == sep_id
                if is_end.any():
                    for ids in output_ids[is_end]:
                        sample_num -= 1
                        result_list.append(self.tokenizer.decode(ids.cpu().numpy()[:-1]))
                    is_end = is_end == False
                    token_ids = token_ids[is_end]
                    output_ids = output_ids[is_end]
                    if len(output_ids) == 0:
                        break
                    token_type_ids = token_type_ids[is_end]
        return result_list

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device='cpu'):
        """
        beam-search操作
        """
        sep_id = word2ix['[SEP]']
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logit_score = output_scores.view(-1, 1) + logit_score
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = hype_pos // scores.shape[-1]
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                end_counts = (output_ids == sep_id).sum(1)
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    return output_ids[best_one][:-1]
                else:
                    flag = end_counts < 1
                    if not flag.all():
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        end_counts = end_counts[flag]
                        beam_size = flag.sum()
            return output_ids[output_scores.argmax()]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4, attn_pdrop=0.5, resid_pdrop=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BartAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BartClassificationHead,
     lambda: ([], {'input_dim': 4, 'inner_dim': 4, 'num_classes': 4, 'pooler_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BartLearnedPositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([[4, 4, 4, 4]], {}),
     False),
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertLayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
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
     False),
    (CRFLayer,
     lambda: ([], {'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 1])], {}),
     False),
    (Conv1D,
     lambda: ([], {'nf': 4, 'nx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NeZhaAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, max_position_embeddings=4, max_relative_position=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (NeZhaSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, max_position_embeddings=4, max_relative_position=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RelativePositionsEncoding,
     lambda: ([], {'length': 4, 'depth': 1}),
     lambda: ([0], {}),
     True),
    (T5Attention,
     lambda: ([], {'config': _mock_config(is_decoder=4, relative_attention_num_buckets=4, d_model=4, d_kv=4, num_heads=4, dropout_rate=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (T5DenseGatedGeluDense,
     lambda: ([], {'config': _mock_config(d_model=4, d_ff=4, dropout_rate=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (T5DenseReluDense,
     lambda: ([], {'config': _mock_config(d_model=4, d_ff=4, dropout_rate=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (T5LayerCrossAttention,
     lambda: ([], {'config': _mock_config(is_decoder=4, relative_attention_num_buckets=4, d_model=4, d_kv=4, num_heads=4, dropout_rate=0.5, layer_norm_epsilon=1)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (T5LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (T5LayerSelfAttention,
     lambda: ([], {'config': _mock_config(is_decoder=4, relative_attention_num_buckets=4, d_model=4, d_kv=4, num_heads=4, dropout_rate=0.5, layer_norm_epsilon=1)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_920232796_bert_seq2seq(_paritybench_base):
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

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

