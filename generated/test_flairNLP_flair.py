import sys
_module = sys.modules[__name__]
del sys
flair = _module
data = _module
data_fetcher = _module
datasets = _module
base = _module
document_classification = _module
sequence_labeling = _module
text_image = _module
text_text = _module
treebanks = _module
embeddings = _module
base = _module
document = _module
image = _module
legacy = _module
token = _module
file_utils = _module
hyperparameter = _module
param_selection = _module
parameter = _module
inference_utils = _module
models = _module
language_model = _module
sequence_tagger_model = _module
similarity_learning_model = _module
text_classification_model = _module
text_regression_model = _module
nn = _module
optim = _module
samplers = _module
trainers = _module
language_model_trainer = _module
trainer = _module
training_utils = _module
visual = _module
activations = _module
manifold = _module
ner_html = _module
training_curves = _module
predict = _module
setup = _module
conftest = _module
test_data = _module
test_datasets = _module
test_embeddings = _module
test_hyperparameter = _module
test_language_model = _module
test_sequence_tagger = _module
test_text_classifier = _module
test_text_regressor = _module
test_utils = _module
test_visual = _module
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


import torch


import logging.config


from abc import abstractmethod


from typing import List


from typing import Dict


from typing import Union


from typing import Callable


import re


import logging


from collections import Counter


from collections import defaultdict


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data.dataset import Subset


import torch.utils.data.dataloader


import numpy as np


from torch.nn import ParameterList


from torch.nn import Parameter


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn.functional as F


from torch.nn import Sequential


from torch.nn import Linear


from torch.nn import Conv2d


from torch.nn import ReLU


from torch.nn import MaxPool2d


from torch.nn import Dropout2d


from torch.nn import AdaptiveAvgPool2d


from torch.nn import AdaptiveMaxPool2d


from torch.nn import TransformerEncoderLayer


from torch.nn import TransformerEncoder


from typing import Tuple


from functools import lru_cache


import torch.nn as nn


import math


from torch.optim import Optimizer


from typing import Optional


import torch.nn


from torch.nn.parameter import Parameter


from torch.utils.data import DataLoader


from torch import nn


import itertools


import warnings


from torch.optim.optimizer import required


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data.sampler import Sampler


import random


import time


from torch import cuda


from torch.optim.sgd import SGD


import copy


import inspect


from enum import Enum


from math import inf


from functools import reduce


from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_absolute_error


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from torch.optim import SGD


from torch.optim.adam import Adam


class Label:
    """
    This class represents a label. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, value: str, score: float=1.0):
        self.value = value
        self.score = score
        super().__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != '':
            raise ValueError('Incorrect label value provided. Label value needs to be set.')
        else:
            self._value = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if 0.0 <= score <= 1.0:
            self._score = score
        else:
            self._score = 1.0

    def to_dict(self):
        return {'value': self.value, 'confidence': self.score}

    def __str__(self):
        return f'{self._value} ({round(self._score, 4)})'

    def __repr__(self):
        return f'{self._value} ({round(self._score, 4)})'


class DataPoint:
    """
    This is the parent class of all data points in Flair (including Token, Sentence, Image, etc.). Each DataPoint
    must be embeddable (hence the abstract property embedding() and methods to() and clear_embeddings()). Also,
    each DataPoint may have Labels in several layers of annotation (hence the functions add_label(), get_labels()
    and the property 'label')
    """

    def __init__(self):
        self.annotation_layers = {}

    @property
    @abstractmethod
    def embedding(self):
        pass

    @abstractmethod
    def to(self, device: str, pin_memory: bool=False):
        pass

    @abstractmethod
    def clear_embeddings(self, embedding_names: List[str]=None):
        pass

    def add_label(self, label_type: str, value: str, score: float=1.0):
        if label_type not in self.annotation_layers:
            self.annotation_layers[label_type] = [Label(value, score)]
        else:
            self.annotation_layers[label_type].append(Label(value, score))
        return self

    def set_label(self, label_type: str, value: str, score: float=1.0):
        self.annotation_layers[label_type] = [Label(value, score)]
        return self

    def get_labels(self, label_type: str=None):
        if label_type is None:
            return self.labels
        return self.annotation_layers[label_type] if label_type in self.annotation_layers else []

    @property
    def labels(self) ->List[Label]:
        all_labels = []
        for key in self.annotation_layers.keys():
            all_labels.extend(self.annotation_layers[key])
        return all_labels


class Image(DataPoint):

    def __init__(self, data=None, imageURL=None):
        super().__init__()
        self.data = data
        self._embeddings: Dict = {}
        self.imageURL = imageURL

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self):
        image_repr = self.data.size() if self.data else ''
        image_url = self.imageURL if self.imageURL else ''
        return f'Image: {image_repr} {image_url}'

    def get_embedding(self) ->torch.tensor:
        embeddings = [self._embeddings[embed] for embed in sorted(self._embeddings.keys())]
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.tensor([], device=flair.device)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.pin_memory()
                else:
                    self._embeddings[name] = vector

    def clear_embeddings(self, embedding_names: List[str]=None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]


class Token(DataPoint):
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(self, text: str, idx: int=None, head_id: int=None, whitespace_after: bool=True, start_position: int=None):
        super().__init__()
        self.text: str = text
        self.idx: int = idx
        self.head_id: int = head_id
        self.whitespace_after: bool = whitespace_after
        self.start_pos = start_position
        self.end_pos = start_position + len(text) if start_position is not None else None
        self.sentence: Sentence = None
        self._embeddings: Dict = {}
        self.tags_proba_dist: Dict[str, List[Label]] = {}

    def add_tag_label(self, tag_type: str, tag: Label):
        self.set_label(tag_type, tag.value, tag.score)

    def add_tags_proba_dist(self, tag_type: str, tags: List[Label]):
        self.tags_proba_dist[tag_type] = tags

    def add_tag(self, tag_type: str, tag_value: str, confidence=1.0):
        self.set_label(tag_type, tag_value, confidence)

    def get_tag(self, label_type):
        if len(self.get_labels(label_type)) == 0:
            return Label('')
        return self.get_labels(label_type)[0]

    def get_tags_proba_dist(self, tag_type: str) ->List[Label]:
        if tag_type in self.tags_proba_dist:
            return self.tags_proba_dist[tag_type]
        return []

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.pin_memory()
                else:
                    self._embeddings[name] = vector

    def clear_embeddings(self, embedding_names: List[str]=None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

    def get_each_embedding(self) ->torch.tensor:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embed = self._embeddings[embed]
            if flair.embedding_storage_mode == 'cpu' and embed.device != flair.device:
                embed = embed
            embeddings.append(embed)
        return embeddings

    def get_embedding(self) ->torch.tensor:
        embeddings = self.get_each_embedding()
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.tensor([], device=flair.device)

    @property
    def start_position(self) ->int:
        return self.start_pos

    @property
    def end_position(self) ->int:
        return self.end_pos

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self) ->str:
        return 'Token: {} {}'.format(self.idx, self.text) if self.idx is not None else 'Token: {}'.format(self.text)

    def __repr__(self) ->str:
        return 'Token: {} {}'.format(self.idx, self.text) if self.idx is not None else 'Token: {}'.format(self.text)


class Span(DataPoint):
    """
    This class represents one textual span consisting of Tokens.
    """

    def __init__(self, tokens: List[Token]):
        super().__init__()
        self.tokens = tokens
        self.start_pos = None
        self.end_pos = None
        if tokens:
            self.start_pos = tokens[0].start_position
            self.end_pos = tokens[len(tokens) - 1].end_position

    @property
    def text(self) ->str:
        return ' '.join([t.text for t in self.tokens])

    def to_original_text(self) ->str:
        pos = self.tokens[0].start_pos
        if pos is None:
            return ' '.join([t.text for t in self.tokens])
        str = ''
        for t in self.tokens:
            while t.start_pos != pos:
                str += ' '
                pos += 1
            str += t.text
            pos += len(t.text)
        return str

    def to_dict(self):
        return {'text': self.to_original_text(), 'start_pos': self.start_pos, 'end_pos': self.end_pos, 'labels': self.labels}

    def __str__(self) ->str:
        ids = ','.join([str(t.idx) for t in self.tokens])
        label_string = ' '.join([str(label) for label in self.labels])
        labels = f'   [− Labels: {label_string}]' if self.labels is not None else ''
        return 'Span [{}]: "{}"{}'.format(ids, self.text, labels)

    def __repr__(self) ->str:
        ids = ','.join([str(t.idx) for t in self.tokens])
        return '<{}-span ({}): "{}">'.format(self.tag, ids, self.text) if self.tag is not None else '<span ({}): "{}">'.format(ids, self.text)

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def score(self):
        return self.labels[0].score


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag.value == 'O':
            continue
        split = tag.value.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1].value == 'O':
            tags[i].value = 'B' + tag.value[1:]
        elif tags[i - 1].value[1:] == tag.value[1:]:
            continue
        else:
            tags[i].value = 'B' + tag.value[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.value == 'O':
            new_tags.append(tag.value)
        elif tag.value.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i + 1].value.split('-')[0] == 'I':
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace('B-', 'S-'))
        elif tag.value.split('-')[0] == 'I':
            if i + 1 < len(tags) and tags[i + 1].value.split('-')[0] == 'I':
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


log = logging.getLogger('flair')


def segtok_tokenizer(text: str) ->List[Token]:
    """
    Tokenizer using segtok, a third party library dedicated to rules-based Indo-European languages.
    https://github.com/fnl/segtok
    """
    tokens: List[Token] = []
    words: List[str] = []
    sentences = split_single(text)
    for sentence in sentences:
        contractions = split_contractions(word_tokenizer(sentence))
        words.extend(contractions)
    index = text.index
    current_offset = 0
    previous_word_offset = -1
    previous_token = None
    for word in words:
        try:
            word_offset = index(word, current_offset)
            start_position = word_offset
        except:
            word_offset = previous_word_offset + 1
            start_position = current_offset + 1 if current_offset > 0 else current_offset
        if word:
            token = Token(text=word, start_position=start_position, whitespace_after=True)
            tokens.append(token)
        if previous_token is not None and word_offset - 1 == previous_word_offset:
            previous_token.whitespace_after = False
        current_offset = word_offset + len(word)
        previous_word_offset = current_offset - 1
        previous_token = token
    return tokens


def space_tokenizer(text: str) ->List[Token]:
    """
    Tokenizer based on space character only.
    """
    tokens: List[Token] = []
    word = ''
    index = -1
    for index, char in enumerate(text):
        if char == ' ':
            if len(word) > 0:
                start_position = index - len(word)
                tokens.append(Token(text=word, start_position=start_position, whitespace_after=True))
            word = ''
        else:
            word += char
    index += 1
    if len(word) > 0:
        start_position = index - len(word)
        tokens.append(Token(text=word, start_position=start_position, whitespace_after=False))
    return tokens


class Sentence(DataPoint):
    """
       A Sentence is a list of Tokens and is used to represent a sentence or text fragment.
    """

    def __init__(self, text: str=None, use_tokenizer: Union[bool, Callable[[str], List[Token]]]=False, language_code: str=None):
        """
        Class to hold all meta related to a text (tokens, predictions, language code, ...)
        :param text: original string
        :param use_tokenizer: a custom tokenizer (default is space based tokenizer,
        more advanced options are segtok_tokenizer to use segtok or build_spacy_tokenizer to use Spacy library
        if available). Check the code of space_tokenizer to implement your own (if you need it).
        If instead of providing a function, this parameter is just set to True, segtok will be used.
        :param labels:
        :param language_code:
        """
        super().__init__()
        self.tokens: List[Token] = []
        self._embeddings: Dict = {}
        self.language_code: str = language_code
        tokenizer = use_tokenizer
        if type(use_tokenizer) == bool:
            tokenizer = segtok_tokenizer if use_tokenizer else space_tokenizer
        if text is not None:
            text = self._restore_windows_1252_characters(text)
            [self.add_token(token) for token in tokenizer(text)]
        if text == '':
            log.warning('ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?')
        self.tokenized = None

    def get_token(self, token_id: int) ->Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Union[Token, str]):
        if type(token) is str:
            token = Token(token)
        token.text = token.text.replace('\u200c', '')
        token.text = token.text.replace('\u200b', '')
        token.text = token.text.replace('️', '')
        token.text = token.text.replace('\ufeff', '')
        if token.text.strip() == '':
            return
        self.tokens.append(token)
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def get_label_names(self):
        label_names = []
        for label in self.labels:
            label_names.append(label.value)
        return label_names

    def get_spans(self, label_type: str, min_score=-1) ->List[Span]:
        spans: List[Span] = []
        current_span = []
        tags = defaultdict(lambda : 0.0)
        previous_tag_value: str = 'O'
        for token in self:
            tag: Label = token.get_tag(label_type)
            tag_value = tag.value
            if tag_value == '' or tag_value == 'O':
                tag_value = 'O-'
            if tag_value[0:2] not in ['B-', 'I-', 'O-', 'E-', 'S-']:
                tag_value = 'S-' + tag_value
            in_span = False
            if tag_value[0:2] not in ['O-']:
                in_span = True
            starts_new_span = False
            if tag_value[0:2] in ['B-', 'S-']:
                starts_new_span = True
            if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
                starts_new_span = True
            if (starts_new_span or not in_span) and len(current_span) > 0:
                scores = [t.get_labels(label_type)[0].score for t in current_span]
                span_score = sum(scores) / len(scores)
                if span_score > min_score:
                    span = Span(current_span)
                    span.add_label(label_type=label_type, value=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0], score=span_score)
                    spans.append(span)
                current_span = []
                tags = defaultdict(lambda : 0.0)
            if in_span:
                current_span.append(token)
                weight = 1.1 if starts_new_span else 1.0
                tags[tag_value[2:]] += weight
            previous_tag_value = tag_value
        if len(current_span) > 0:
            scores = [t.get_labels(label_type)[0].score for t in current_span]
            span_score = sum(scores) / len(scores)
            if span_score > min_score:
                span = Span(current_span)
                span.add_label(label_type=label_type, value=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0], score=span_score)
                spans.append(span)
        return spans

    @property
    def embedding(self):
        return self.get_embedding()

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector
        self._embeddings[name] = vector

    def get_embedding(self) ->torch.tensor:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embedding = self._embeddings[embed]
            embeddings.append(embedding)
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.Tensor()

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.pin_memory()
                else:
                    self._embeddings[name] = vector
        for token in self:
            token

    def clear_embeddings(self, embedding_names: List[str]=None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]
        for token in self:
            token.clear_embeddings(embedding_names)

    def to_tagged_string(self, main_tag=None) ->str:
        list = []
        for token in self.tokens:
            list.append(token.text)
            tags: List[str] = []
            for label_type in token.annotation_layers.keys():
                if main_tag is not None and main_tag != label_type:
                    continue
                if token.get_labels(label_type)[0].value == 'O':
                    continue
                tags.append(token.get_labels(label_type)[0].value)
            all_tags = '<' + '/'.join(tags) + '>'
            if all_tags != '<>':
                list.append(all_tags)
        return ' '.join(list)

    def to_tokenized_string(self) ->str:
        if self.tokenized is None:
            self.tokenized = ' '.join([t.text for t in self.tokens])
        return self.tokenized

    def to_plain_string(self):
        plain = ''
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after:
                plain += ' '
        return plain.rstrip()

    def convert_tag_scheme(self, tag_type: str='ner', target_scheme: str='iob'):
        tags: List[Label] = []
        for token in self.tokens:
            tags.append(token.get_tag(tag_type))
        if target_scheme == 'iob':
            iob2(tags)
        if target_scheme == 'iobes':
            iob2(tags)
            tags = iob_iobes(tags)
        for index, tag in enumerate(tags):
            self.tokens[index].set_label(tag_type, tag)

    def infer_space_after(self):
        """
        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP
        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.
        :return:
        """
        last_token = None
        quote_count: int = 0
        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False
            if last_token is not None:
                if token.text in ['.', ':', ',', ';', ')', "n't", '!', '?']:
                    last_token.whitespace_after = False
                if token.text.startswith("'"):
                    last_token.whitespace_after = False
            if token.text in ['(']:
                token.whitespace_after = False
            last_token = token
        return self

    def to_original_text(self) ->str:
        if len(self.tokens) > 0 and self.tokens[0].start_pos is None:
            return ' '.join([t.text for t in self.tokens])
        str = ''
        pos = 0
        for t in self.tokens:
            while t.start_pos > pos:
                str += ' '
                pos += 1
            str += t.text
            pos += len(t.text)
        return str

    def to_dict(self, tag_type: str=None):
        labels = []
        entities = []
        if tag_type:
            entities = [span.to_dict() for span in self.get_spans(tag_type)]
        if self.labels:
            labels = [l.to_dict() for l in self.labels]
        return {'text': self.to_original_text(), 'labels': labels, 'entities': entities}

    def __getitem__(self, idx: int) ->Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()
        sentence_labels = f'  − Sentence-Labels: {self.annotation_layers}' if self.annotation_layers != {} else ''
        token_labels = f'  − Token-Labels: "{tagged_string}"' if tokenized_string != tagged_string else ''
        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_label(tag_type, token.get_tag(tag_type).value, token.get_tag(tag_type).score)
            s.add_token(nt)
        return s

    def __str__(self) ->str:
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()
        sentence_labels = f'  − Sentence-Labels: {self.annotation_layers}' if self.annotation_layers != {} else ''
        token_labels = f'  − Token-Labels: "{tagged_string}"' if tokenized_string != tagged_string else ''
        return f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'

    def __len__(self) ->int:
        return len(self.tokens)

    def get_language_code(self) ->str:
        if self.language_code is None:
            try:
                self.language_code = langdetect.detect(self.to_plain_string())
            except:
                self.language_code = 'en'
        return self.language_code

    @staticmethod
    def _restore_windows_1252_characters(text: str) ->str:

        def to_windows_1252(match):
            try:
                return bytes([ord(match.group(0))]).decode('windows-1252')
            except UnicodeDecodeError:
                return ''
        return re.sub('[\\u0080-\\u0099]', to_windows_1252, text)


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    def __init__(self):
        """Set some attributes that would otherwise result in errors. Overwrite these in your embedding class."""
        if not hasattr(self, 'name'):
            self.name: str = 'unnamed_embedding'
        if not hasattr(self, 'static_embeddings'):
            self.static_embeddings = False
        super().__init__()

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) ->str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) ->List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""
        if type(sentences) is Sentence or type(sentences) is Image:
            sentences = [sentences]
        everything_embedded: bool = True
        if self.embedding_type == 'word-level':
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False
        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)
        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors.
    This method was proposed by Liu et al. (2019) in the paper:
    "Linguistic Knowledge and Transferability of Contextual Representations" (https://arxiv.org/abs/1903.08855)

    The implementation is copied and slightly modified from the allennlp repository and is licensed under Apache 2.0.
    It can be found under:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py.
    """

    def __init__(self, mixture_size: int, trainable: bool=False) ->None:
        """
        Inits scalar mix implementation.
        ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
        :param mixture_size: size of mixtures (usually the number of layers)
        """
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        initial_scalar_parameters = [0.0] * mixture_size
        self.scalar_parameters = ParameterList([Parameter(torch.tensor([initial_scalar_parameters[i]], dtype=torch.float, device=flair.device), requires_grad=trainable) for i in range(mixture_size)])
        self.gamma = Parameter(torch.tensor([1.0], dtype=torch.float, device=flair.device), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) ->torch.Tensor:
        """
        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        :param tensors: list of input tensors
        :return: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error('{} tensors were passed, but the module was initialized to mix {} tensors.'.format(len(tensors), self.mixture_size))
        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class DocumentEmbeddings(Embeddings):
    """Abstract base class for all document-level embeddings. Ever new type of document embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) ->str:
        return 'sentence-level'


class TransformerDocumentEmbeddings(DocumentEmbeddings):

    def __init__(self, model: str='bert-base-uncased', fine_tune: bool=True, batch_size: int=1, layers: str='-1', use_scalar_mix: bool=False):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model, config=config)
        self.name = 'transformer-document-' + str(model)
        self.model.eval()
        self.model
        if layers == 'all':
            hidden_states = self.model(torch.tensor([1], device=flair.device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(',')]
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size
        self.initial_cls_token: bool = False
        if isinstance(self.tokenizer, BertTokenizer) or isinstance(self.tokenizer, AlbertTokenizer):
            self.initial_cls_token = True

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        """Add embeddings to all words in a list of sentences."""
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]
        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)
        return sentences

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        """Extract sentence embedding from CLS token or similar and add to Sentence object."""
        gradient_context = torch.enable_grad() if self.fine_tune and self.training else torch.no_grad()
        with gradient_context:
            subtokenized_sentences = []
            for sentence in sentences:
                subtokenized_sentence = self.tokenizer.encode(sentence.to_tokenized_string(), add_special_tokens=True, max_length=512)
                subtokenized_sentences.append(torch.tensor(subtokenized_sentence, dtype=torch.long, device=flair.device))
            longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))
            input_ids = torch.zeros([len(sentences), longest_sequence_in_batch], dtype=torch.long, device=flair.device)
            mask = torch.zeros([len(sentences), longest_sequence_in_batch], dtype=torch.long, device=flair.device)
            for s_id, sentence in enumerate(subtokenized_sentences):
                sequence_length = len(sentence)
                input_ids[s_id][:sequence_length] = sentence
                mask[s_id][:sequence_length] = torch.ones(sequence_length)
            hidden_states = self.model(input_ids, attention_mask=mask)[-1] if len(sentences) > 1 else self.model(input_ids)[-1]
            for sentence_idx, (sentence, subtokens) in enumerate(zip(sentences, subtokenized_sentences)):
                index_of_CLS_token = 0 if self.initial_cls_token else len(subtokens) - 1
                cls_embeddings_all_layers: List[torch.FloatTensor] = [hidden_states[layer][sentence_idx][index_of_CLS_token] for layer in self.layer_indexes]
                if self.use_scalar_mix:
                    sm = ScalarMix(mixture_size=len(cls_embeddings_all_layers))
                    sm_embeddings = sm(cls_embeddings_all_layers)
                    cls_embeddings_all_layers = [sm_embeddings]
                sentence.set_embedding(self.name, torch.cat(cls_embeddings_all_layers))

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        return len(self.layer_indexes) * self.model.config.hidden_size if not self.use_scalar_mix else self.model.config.hidden_size

    def __setstate__(self, d):
        self.__dict__ = d
        model_name = self.name.split('transformer-document-')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) ->str:
        return 'word-level'


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()
        self.embeddings = embeddings
        for i, embedding in enumerate(embeddings):
            embedding.name = f'{str(i)}-{embedding.name}'
            self.add_module(f'list_embedding_{str(i)}', embedding)
        self.name: str = 'Stack'
        self.static_embeddings: bool = True
        self.__embedding_type: str = embeddings[0].embedding_type
        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool=True):
        if type(sentences) is Sentence:
            sentences = [sentences]
        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) ->str:
        return self.__embedding_type

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)
        return sentences

    def __str__(self):
        return f"StackedEmbeddings [{','.join([str(e) for e in self.embeddings])}]"


class DocumentPoolEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings: List[TokenEmbeddings], fine_tune_mode='linear', pooling: str='mean'):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param pooling: a string which can any value from ['mean', 'max', 'min']
        """
        super().__init__()
        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length
        self.fine_tune_mode = fine_tune_mode
        if self.fine_tune_mode in ['nonlinear', 'linear']:
            self.embedding_flex = torch.nn.Linear(self.embedding_length, self.embedding_length, bias=False)
            self.embedding_flex.weight.data.copy_(torch.eye(self.embedding_length))
        if self.fine_tune_mode in ['nonlinear']:
            self.embedding_flex_nonlinear = torch.nn.ReLU(self.embedding_length)
            self.embedding_flex_nonlinear_map = torch.nn.Linear(self.embedding_length, self.embedding_length)
        self.__embedding_length: int = self.embeddings.embedding_length
        self
        self.pooling = pooling
        if self.pooling == 'mean':
            self.pool_op = torch.mean
        elif pooling == 'max':
            self.pool_op = torch.max
        elif pooling == 'min':
            self.pool_op = torch.min
        else:
            raise ValueError(f'Pooling operation for {self.mode!r} is not defined')
        self.name: str = f'document_{self.pooling}'

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""
        if isinstance(sentences, Sentence):
            sentences = [sentences]
        self.embeddings.embed(sentences)
        for sentence in sentences:
            word_embeddings = []
            for token in sentence.tokens:
                word_embeddings.append(token.get_embedding().unsqueeze(0))
            word_embeddings = torch.cat(word_embeddings, dim=0)
            if self.fine_tune_mode in ['nonlinear', 'linear']:
                word_embeddings = self.embedding_flex(word_embeddings)
            if self.fine_tune_mode in ['nonlinear']:
                word_embeddings = self.embedding_flex_nonlinear(word_embeddings)
                word_embeddings = self.embedding_flex_nonlinear_map(word_embeddings)
            if self.pooling == 'mean':
                pooled_embedding = self.pool_op(word_embeddings, 0)
            else:
                pooled_embedding, _ = self.pool_op(word_embeddings, 0)
            sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass

    def extra_repr(self):
        return f'fine_tune_mode={self.fine_tune_mode}, pooling={self.pooling}'


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.dropout_rate, inplace_str)


class DocumentRNNEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings: List[TokenEmbeddings], hidden_size=128, rnn_layers=1, reproject_words: bool=True, reproject_words_dimension: int=None, bidirectional: bool=False, dropout: float=0.5, word_dropout: float=0.0, locked_dropout: float=0.0, rnn_type='GRU', fine_tune: bool=True):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()
        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.rnn_type = rnn_type
        self.reproject_words = reproject_words
        self.bidirectional = bidirectional
        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length
        self.static_embeddings = False if fine_tune else True
        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4
        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension
        self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings, self.embeddings_dimension)
        if rnn_type == 'LSTM':
            self.rnn = torch.nn.LSTM(self.embeddings_dimension, hidden_size, num_layers=rnn_layers, bidirectional=self.bidirectional, batch_first=True)
        else:
            self.rnn = torch.nn.GRU(self.embeddings_dimension, hidden_size, num_layers=rnn_layers, bidirectional=self.bidirectional, batch_first=True)
        self.name = 'document_' + self.rnn._get_name()
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None
        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)
        self
        self.eval()

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""
        if not hasattr(self, 'locked_dropout'):
            self.locked_dropout = None
        if not hasattr(self, 'word_dropout'):
            self.word_dropout = None
        if type(sentences) is Sentence:
            sentences = [sentences]
        self.rnn.zero_grad()
        self.embeddings.embed(sentences)
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(self.embeddings.embedding_length * longest_token_sequence_in_batch, dtype=torch.float, device=flair.device)
        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[:self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)
        sentence_tensor = torch.cat(all_embs).view([len(sentences), longest_token_sequence_in_batch, self.embeddings.embedding_length])
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        packed = pack_padded_sequence(sentence_tensor, lengths, enforce_sorted=False, batch_first=True)
        rnn_out, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(rnn_out, batch_first=True)
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[sentence_no, length - 1]
            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[sentence_no, 0]
                embedding = torch.cat([first_rep, last_rep], 0)
            if self.static_embeddings:
                embedding = embedding.detach()
            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):
        major, minor, build, *_ = (int(info) for info in torch.__version__.replace('+', '.').split('.') if info.isdigit())
        if major >= 1 and minor >= 4:
            for child_module in self.children():
                if isinstance(child_module, torch.nn.RNNBase):
                    _flat_weights_names = []
                    num_direction = None
                    if child_module.__dict__['bidirectional']:
                        num_direction = 2
                    else:
                        num_direction = 1
                    for layer in range(child_module.__dict__['num_layers']):
                        for direction in range(num_direction):
                            suffix = '_reverse' if direction == 1 else ''
                            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                            if child_module.__dict__['bias']:
                                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                            param_names = [x.format(layer, suffix) for x in param_names]
                            _flat_weights_names.extend(param_names)
                    setattr(child_module, '_flat_weights_names', _flat_weights_names)
                child_module._apply(fn)
        else:
            super()._apply(fn)


class Tqdm:
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) ->None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) ->None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {'mininterval': Tqdm.default_mininterval, **kwargs}
        return _tqdm(*args, **new_kwargs)


logger = logging.getLogger('flair')


def replace_with_language_code(string: str):
    string = string.replace('arabic-', 'ar-')
    string = string.replace('basque-', 'eu-')
    string = string.replace('bulgarian-', 'bg-')
    string = string.replace('croatian-', 'hr-')
    string = string.replace('czech-', 'cs-')
    string = string.replace('danish-', 'da-')
    string = string.replace('dutch-', 'nl-')
    string = string.replace('farsi-', 'fa-')
    string = string.replace('persian-', 'fa-')
    string = string.replace('finnish-', 'fi-')
    string = string.replace('french-', 'fr-')
    string = string.replace('german-', 'de-')
    string = string.replace('hebrew-', 'he-')
    string = string.replace('hindi-', 'hi-')
    string = string.replace('indonesian-', 'id-')
    string = string.replace('italian-', 'it-')
    string = string.replace('japanese-', 'ja-')
    string = string.replace('norwegian-', 'no')
    string = string.replace('polish-', 'pl-')
    string = string.replace('portuguese-', 'pt-')
    string = string.replace('slovenian-', 'sl-')
    string = string.replace('spanish-', 'es-')
    string = string.replace('swedish-', 'sv-')
    return string


class FlairEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(self, model, fine_tune: bool=False, chars_per_chunk: int=512, with_whitespace: bool=True, tokenized_lm: bool=True):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward',
                etc (see https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)
                depending on which character language model is desired.
        :param fine_tune: if set to True, the gradient will propagate into the language model. This dramatically slows
                down training and often leads to overfitting, so use with caution.
        :param chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster
                but requires more memory. Lower means slower but less memory.
        :param with_whitespace: If True, use hidden state after whitespace after word. If False, use hidden
                 state at last character of word.
        :param tokenized_lm: Whether this lm is tokenized. Default is True, but for LMs trained over unprocessed text
                False might be better.
        """
        super().__init__()
        cache_dir = Path('embeddings')
        aws_path: str = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources'
        hu_path: str = 'https://flair.informatik.hu-berlin.de/resources'
        clef_hipe_path: str = 'https://files.ifi.uzh.ch/cl/siclemat/impresso/clef-hipe-2020/flair'
        self.PRETRAINED_MODEL_ARCHIVE_MAP = {'multi-forward': f'{aws_path}/embeddings-v0.4.3/lm-jw300-forward-v0.1.pt', 'multi-backward': f'{aws_path}/embeddings-v0.4.3/lm-jw300-backward-v0.1.pt', 'multi-v0-forward': f'{aws_path}/embeddings-v0.4/lm-multi-forward-v0.1.pt', 'multi-v0-backward': f'{aws_path}/embeddings-v0.4/lm-multi-backward-v0.1.pt', 'multi-v0-forward-fast': f'{aws_path}/embeddings-v0.4/lm-multi-forward-fast-v0.1.pt', 'multi-v0-backward-fast': f'{aws_path}/embeddings-v0.4/lm-multi-backward-fast-v0.1.pt', 'en-forward': f'{aws_path}/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt', 'en-backward': f'{aws_path}/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt', 'en-forward-fast': f'{aws_path}/embeddings/lm-news-english-forward-1024-v0.2rc.pt', 'en-backward-fast': f'{aws_path}/embeddings/lm-news-english-backward-1024-v0.2rc.pt', 'news-forward': f'{aws_path}/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt', 'news-backward': f'{aws_path}/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt', 'news-forward-fast': f'{aws_path}/embeddings/lm-news-english-forward-1024-v0.2rc.pt', 'news-backward-fast': f'{aws_path}/embeddings/lm-news-english-backward-1024-v0.2rc.pt', 'mix-forward': f'{aws_path}/embeddings/lm-mix-english-forward-v0.2rc.pt', 'mix-backward': f'{aws_path}/embeddings/lm-mix-english-backward-v0.2rc.pt', 'ar-forward': f'{aws_path}/embeddings-stefan-it/lm-ar-opus-large-forward-v0.1.pt', 'ar-backward': f'{aws_path}/embeddings-stefan-it/lm-ar-opus-large-backward-v0.1.pt', 'bg-forward-fast': f'{aws_path}/embeddings-v0.3/lm-bg-small-forward-v0.1.pt', 'bg-backward-fast': f'{aws_path}/embeddings-v0.3/lm-bg-small-backward-v0.1.pt', 'bg-forward': f'{aws_path}/embeddings-stefan-it/lm-bg-opus-large-forward-v0.1.pt', 'bg-backward': f'{aws_path}/embeddings-stefan-it/lm-bg-opus-large-backward-v0.1.pt', 'cs-forward': f'{aws_path}/embeddings-stefan-it/lm-cs-opus-large-forward-v0.1.pt', 'cs-backward': f'{aws_path}/embeddings-stefan-it/lm-cs-opus-large-backward-v0.1.pt', 'cs-v0-forward': f'{aws_path}/embeddings-v0.4/lm-cs-large-forward-v0.1.pt', 'cs-v0-backward': f'{aws_path}/embeddings-v0.4/lm-cs-large-backward-v0.1.pt', 'da-forward': f'{aws_path}/embeddings-stefan-it/lm-da-opus-large-forward-v0.1.pt', 'da-backward': f'{aws_path}/embeddings-stefan-it/lm-da-opus-large-backward-v0.1.pt', 'de-forward': f'{aws_path}/embeddings/lm-mix-german-forward-v0.2rc.pt', 'de-backward': f'{aws_path}/embeddings/lm-mix-german-backward-v0.2rc.pt', 'de-historic-ha-forward': f'{aws_path}/embeddings-stefan-it/lm-historic-hamburger-anzeiger-forward-v0.1.pt', 'de-historic-ha-backward': f'{aws_path}/embeddings-stefan-it/lm-historic-hamburger-anzeiger-backward-v0.1.pt', 'de-historic-wz-forward': f'{aws_path}/embeddings-stefan-it/lm-historic-wiener-zeitung-forward-v0.1.pt', 'de-historic-wz-backward': f'{aws_path}/embeddings-stefan-it/lm-historic-wiener-zeitung-backward-v0.1.pt', 'de-historic-rw-forward': f'{hu_path}/embeddings/redewiedergabe_lm_forward.pt', 'de-historic-rw-backward': f'{hu_path}/embeddings/redewiedergabe_lm_backward.pt', 'es-forward': f'{aws_path}/embeddings-v0.4/language_model_es_forward_long/lm-es-forward.pt', 'es-backward': f'{aws_path}/embeddings-v0.4/language_model_es_backward_long/lm-es-backward.pt', 'es-forward-fast': f'{aws_path}/embeddings-v0.4/language_model_es_forward/lm-es-forward-fast.pt', 'es-backward-fast': f'{aws_path}/embeddings-v0.4/language_model_es_backward/lm-es-backward-fast.pt', 'eu-forward': f'{aws_path}/embeddings-stefan-it/lm-eu-opus-large-forward-v0.2.pt', 'eu-backward': f'{aws_path}/embeddings-stefan-it/lm-eu-opus-large-backward-v0.2.pt', 'eu-v1-forward': f'{aws_path}/embeddings-stefan-it/lm-eu-opus-large-forward-v0.1.pt', 'eu-v1-backward': f'{aws_path}/embeddings-stefan-it/lm-eu-opus-large-backward-v0.1.pt', 'eu-v0-forward': f'{aws_path}/embeddings-v0.4/lm-eu-large-forward-v0.1.pt', 'eu-v0-backward': f'{aws_path}/embeddings-v0.4/lm-eu-large-backward-v0.1.pt', 'fa-forward': f'{aws_path}/embeddings-stefan-it/lm-fa-opus-large-forward-v0.1.pt', 'fa-backward': f'{aws_path}/embeddings-stefan-it/lm-fa-opus-large-backward-v0.1.pt', 'fi-forward': f'{aws_path}/embeddings-stefan-it/lm-fi-opus-large-forward-v0.1.pt', 'fi-backward': f'{aws_path}/embeddings-stefan-it/lm-fi-opus-large-backward-v0.1.pt', 'fr-forward': f'{aws_path}/embeddings/lm-fr-charlm-forward.pt', 'fr-backward': f'{aws_path}/embeddings/lm-fr-charlm-backward.pt', 'he-forward': f'{aws_path}/embeddings-stefan-it/lm-he-opus-large-forward-v0.1.pt', 'he-backward': f'{aws_path}/embeddings-stefan-it/lm-he-opus-large-backward-v0.1.pt', 'hi-forward': f'{aws_path}/embeddings-stefan-it/lm-hi-opus-large-forward-v0.1.pt', 'hi-backward': f'{aws_path}/embeddings-stefan-it/lm-hi-opus-large-backward-v0.1.pt', 'hr-forward': f'{aws_path}/embeddings-stefan-it/lm-hr-opus-large-forward-v0.1.pt', 'hr-backward': f'{aws_path}/embeddings-stefan-it/lm-hr-opus-large-backward-v0.1.pt', 'id-forward': f'{aws_path}/embeddings-stefan-it/lm-id-opus-large-forward-v0.1.pt', 'id-backward': f'{aws_path}/embeddings-stefan-it/lm-id-opus-large-backward-v0.1.pt', 'it-forward': f'{aws_path}/embeddings-stefan-it/lm-it-opus-large-forward-v0.1.pt', 'it-backward': f'{aws_path}/embeddings-stefan-it/lm-it-opus-large-backward-v0.1.pt', 'ja-forward': f'{aws_path}/embeddings-v0.4.1/lm__char-forward__ja-wikipedia-3GB/japanese-forward.pt', 'ja-backward': f'{aws_path}/embeddings-v0.4.1/lm__char-backward__ja-wikipedia-3GB/japanese-backward.pt', 'ml-forward': f'https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-forward.pt', 'ml-backward': f'https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-backward.pt', 'nl-forward': f'{aws_path}/embeddings-stefan-it/lm-nl-opus-large-forward-v0.1.pt', 'nl-backward': f'{aws_path}/embeddings-stefan-it/lm-nl-opus-large-backward-v0.1.pt', 'nl-v0-forward': f'{aws_path}/embeddings-v0.4/lm-nl-large-forward-v0.1.pt', 'nl-v0-backward': f'{aws_path}/embeddings-v0.4/lm-nl-large-backward-v0.1.pt', 'no-forward': f'{aws_path}/embeddings-stefan-it/lm-no-opus-large-forward-v0.1.pt', 'no-backward': f'{aws_path}/embeddings-stefan-it/lm-no-opus-large-backward-v0.1.pt', 'pl-forward': f'{aws_path}/embeddings/lm-polish-forward-v0.2.pt', 'pl-backward': f'{aws_path}/embeddings/lm-polish-backward-v0.2.pt', 'pl-opus-forward': f'{aws_path}/embeddings-stefan-it/lm-pl-opus-large-forward-v0.1.pt', 'pl-opus-backward': f'{aws_path}/embeddings-stefan-it/lm-pl-opus-large-backward-v0.1.pt', 'pt-forward': f'{aws_path}/embeddings-v0.4/lm-pt-forward.pt', 'pt-backward': f'{aws_path}/embeddings-v0.4/lm-pt-backward.pt', 'pubmed-forward': f'{aws_path}/embeddings-v0.4.1/pubmed-2015-fw-lm.pt', 'pubmed-backward': f'{aws_path}/embeddings-v0.4.1/pubmed-2015-bw-lm.pt', 'sl-forward': f'{aws_path}/embeddings-stefan-it/lm-sl-opus-large-forward-v0.1.pt', 'sl-backward': f'{aws_path}/embeddings-stefan-it/lm-sl-opus-large-backward-v0.1.pt', 'sl-v0-forward': f'{aws_path}/embeddings-v0.3/lm-sl-large-forward-v0.1.pt', 'sl-v0-backward': f'{aws_path}/embeddings-v0.3/lm-sl-large-backward-v0.1.pt', 'sv-forward': f'{aws_path}/embeddings-stefan-it/lm-sv-opus-large-forward-v0.1.pt', 'sv-backward': f'{aws_path}/embeddings-stefan-it/lm-sv-opus-large-backward-v0.1.pt', 'sv-v0-forward': f'{aws_path}/embeddings-v0.4/lm-sv-large-forward-v0.1.pt', 'sv-v0-backward': f'{aws_path}/embeddings-v0.4/lm-sv-large-backward-v0.1.pt', 'ta-forward': f'{aws_path}/embeddings-stefan-it/lm-ta-opus-large-forward-v0.1.pt', 'ta-backward': f'{aws_path}/embeddings-stefan-it/lm-ta-opus-large-backward-v0.1.pt', 'de-impresso-hipe-v1-forward': f'{clef_hipe_path}/de-hipe-flair-v1-forward/best-lm.pt', 'de-impresso-hipe-v1-backward': f'{clef_hipe_path}/de-hipe-flair-v1-backward/best-lm.pt', 'en-impresso-hipe-v1-forward': f'{clef_hipe_path}/en-flair-v1-forward/best-lm.pt', 'en-impresso-hipe-v1-backward': f'{clef_hipe_path}/en-flair-v1-backward/best-lm.pt', 'fr-impresso-hipe-v1-forward': f'{clef_hipe_path}/fr-hipe-flair-v1-forward/best-lm.pt', 'fr-impresso-hipe-v1-backward': f'{clef_hipe_path}/fr-hipe-flair-v1-backward/best-lm.pt'}
        if type(model) == str:
            if model.lower() in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[model.lower()]
                if 'impresso-hipe' in model.lower():
                    cache_dir = cache_dir / model.lower()
                model = cached_path(base_path, cache_dir=cache_dir)
            elif replace_with_language_code(model) in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[replace_with_language_code(model)]
                model = cached_path(base_path, cache_dir=cache_dir)
            elif not Path(model).exists():
                raise ValueError(f'The given model "{model}" is not available or is not a valid path.')
        if type(model) == LanguageModel:
            self.lm: LanguageModel = model
            self.name = f'Task-LSTM-{self.lm.hidden_size}-{self.lm.nlayers}-{self.lm.is_forward_lm}'
        else:
            self.lm: LanguageModel = LanguageModel.load_language_model(model)
            self.name = str(model)
        self.fine_tune = fine_tune
        self.static_embeddings = not fine_tune
        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.with_whitespace: bool = with_whitespace
        self.tokenized_lm: bool = tokenized_lm
        self.chars_per_chunk: int = chars_per_chunk
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(embedded_dummy[0].get_token(1).get_embedding())
        self.eval()

    def train(self, mode=True):
        if 'fine_tune' not in self.__dict__:
            self.fine_tune = False
        if 'chars_per_chunk' not in self.__dict__:
            self.chars_per_chunk = 512
        if not self.fine_tune:
            pass
        else:
            super(FlairEmbeddings, self).train(mode)

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        if 'with_whitespace' not in self.__dict__:
            self.with_whitespace = True
        if 'tokenized_lm' not in self.__dict__:
            self.tokenized_lm = True
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()
        with gradient_context:
            text_sentences = [sentence.to_tokenized_string() for sentence in sentences] if self.tokenized_lm else [sentence.to_plain_string() for sentence in sentences]
            start_marker = self.lm.document_delimiter if 'document_delimiter' in self.lm.__dict__ else '\n'
            end_marker = ' '
            all_hidden_states_in_lm = self.lm.get_representation(text_sentences, start_marker, end_marker, self.chars_per_chunk)
            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string() if self.tokenized_lm else sentence.to_plain_string()
                offset_forward: int = len(start_marker)
                offset_backward: int = len(sentence_text) + len(start_marker)
                for token in sentence.tokens:
                    offset_forward += len(token.text)
                    if self.is_forward_lm:
                        offset_with_whitespace = offset_forward
                        offset_without_whitespace = offset_forward - 1
                    else:
                        offset_with_whitespace = offset_backward
                        offset_without_whitespace = offset_backward - 1
                    if self.with_whitespace:
                        embedding = all_hidden_states_in_lm[(offset_with_whitespace), (i), :]
                    else:
                        embedding = all_hidden_states_in_lm[(offset_without_whitespace), (i), :]
                    if self.tokenized_lm or token.whitespace_after:
                        offset_forward += 1
                        offset_backward -= 1
                    offset_backward -= len(token.text)
                    if flair.embedding_storage_mode == 'gpu':
                        embedding = embedding.clone()
                    token.set_embedding(self.name, embedding)
            del all_hidden_states_in_lm
        return sentences

    def __str__(self):
        return self.name


class DocumentLMEmbeddings(DocumentEmbeddings):

    def __init__(self, flair_embeddings: List[FlairEmbeddings]):
        super().__init__()
        self.embeddings = flair_embeddings
        self.name = 'document_lm'
        for i, embedding in enumerate(flair_embeddings):
            self.add_module('lm_embedding_{}'.format(i), embedding)
            if not embedding.static_embeddings:
                self.static_embeddings = False
        self._embedding_length: int = sum(embedding.embedding_length for embedding in flair_embeddings)

    @property
    def embedding_length(self) ->int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        if type(sentences) is Sentence:
            sentences = [sentences]
        for embedding in self.embeddings:
            embedding.embed(sentences)
            for sentence in sentences:
                sentence: Sentence = sentence
                if embedding.is_forward_lm:
                    sentence.set_embedding(embedding.name, sentence[len(sentence) - 1]._embeddings[embedding.name])
                else:
                    sentence.set_embedding(embedding.name, sentence[0]._embeddings[embedding.name])
        return sentences


class ImageEmbeddings(Embeddings):

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) ->str:
        return 'image-level'


class IdentityImageEmbeddings(ImageEmbeddings):

    def __init__(self, transforms):
        self.PIL = pythonimagelib
        self.name = 'Identity'
        self.transforms = transforms
        self.__embedding_length = None
        self.static_embeddings = True
        super().__init__()

    def _add_embeddings_internal(self, images: List[Image]) ->List[Image]:
        for image in images:
            image_data = self.PIL.Image.open(image.imageURL)
            image_data.load()
            image.set_embedding(self.name, self.transforms(image_data))

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class PrecomputedImageEmbeddings(ImageEmbeddings):

    def __init__(self, url2tensor_dict, name):
        self.url2tensor_dict = url2tensor_dict
        self.name = name
        self.__embedding_length = len(list(self.url2tensor_dict.values())[0])
        self.static_embeddings = True
        super().__init__()

    def _add_embeddings_internal(self, images: List[Image]) ->List[Image]:
        for image in images:
            if image.imageURL in self.url2tensor_dict:
                image.set_embedding(self.name, self.url2tensor_dict[image.imageURL])
            else:
                image.set_embedding(self.name, torch.zeros(self.__embedding_length, device=flair.device))

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class NetworkImageEmbeddings(ImageEmbeddings):

    def __init__(self, name, pretrained=True, transforms=None):
        super().__init__()
        try:
            import torchvision as torchvision
        except ModuleNotFoundError:
            log.warning('-' * 100)
            log.warning('ATTENTION! The library "torchvision" is not installed!')
            log.warning('To use convnets pretraned on ImageNet, please first install with "pip install torchvision"')
            log.warning('-' * 100)
            pass
        model_info = {'resnet50': (torchvision.models.resnet50, lambda x: list(x)[:-1], 2048), 'mobilenet_v2': (torchvision.models.mobilenet_v2, lambda x: list(x)[:-1] + [torch.nn.AdaptiveAvgPool2d((1, 1))], 1280)}
        transforms = [] if transforms is None else transforms
        transforms += [torchvision.transforms.ToTensor()]
        if pretrained:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            transforms += [torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
        self.transforms = torchvision.transforms.Compose(transforms)
        if name in model_info:
            model_constructor = model_info[name][0]
            model_features = model_info[name][1]
            embedding_length = model_info[name][2]
            net = model_constructor(pretrained=pretrained)
            modules = model_features(net.children())
            self.features = torch.nn.Sequential(*modules)
            self.__embedding_length = embedding_length
            self.name = name
        else:
            raise Exception(f'Image embeddings {name} not available.')

    def _add_embeddings_internal(self, images: List[Image]) ->List[Image]:
        image_tensor = torch.stack([self.transforms(image.data) for image in images])
        image_embeddings = self.features(image_tensor)
        image_embeddings = image_embeddings.view(image_embeddings.shape[:2]) if image_embeddings.dim() == 4 else image_embeddings
        if image_embeddings.dim() != 2:
            raise Exception(f'Unknown embedding shape of length {image_embeddings.dim()}')
        for image_id, image in enumerate(images):
            image.set_embedding(self.name, image_embeddings[image_id])

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class ConvTransformNetworkImageEmbeddings(ImageEmbeddings):

    def __init__(self, feats_in, convnet_parms, posnet_parms, transformer_parms):
        super(ConvTransformNetworkImageEmbeddings, self).__init__()
        adaptive_pool_func_map = {'max': AdaptiveMaxPool2d, 'avg': AdaptiveAvgPool2d}
        convnet_arch = [] if convnet_parms['dropout'][0] <= 0 else [Dropout2d(convnet_parms['dropout'][0])]
        convnet_arch.extend([Conv2d(in_channels=feats_in, out_channels=convnet_parms['n_feats_out'][0], kernel_size=convnet_parms['kernel_sizes'][0], padding=convnet_parms['kernel_sizes'][0][0] // 2, stride=convnet_parms['strides'][0], groups=convnet_parms['groups'][0]), ReLU()])
        if '0' in convnet_parms['pool_layers_map']:
            convnet_arch.append(MaxPool2d(kernel_size=convnet_parms['pool_layers_map']['0']))
        for layer_id, (kernel_size, n_in, n_out, groups, stride, dropout) in enumerate(zip(convnet_parms['kernel_sizes'][1:], convnet_parms['n_feats_out'][:-1], convnet_parms['n_feats_out'][1:], convnet_parms['groups'][1:], convnet_parms['strides'][1:], convnet_parms['dropout'][1:])):
            if dropout > 0:
                convnet_arch.append(Dropout2d(dropout))
            convnet_arch.append(Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size, padding=kernel_size[0] // 2, stride=stride, groups=groups))
            convnet_arch.append(ReLU())
            if str(layer_id + 1) in convnet_parms['pool_layers_map']:
                convnet_arch.append(MaxPool2d(kernel_size=convnet_parms['pool_layers_map'][str(layer_id + 1)]))
        convnet_arch.append(adaptive_pool_func_map[convnet_parms['adaptive_pool_func']](output_size=convnet_parms['output_size']))
        self.conv_features = Sequential(*convnet_arch)
        conv_feat_dim = convnet_parms['n_feats_out'][-1]
        if posnet_parms is not None and transformer_parms is not None:
            self.use_transformer = True
            if posnet_parms['nonlinear']:
                posnet_arch = [Linear(2, posnet_parms['n_hidden']), ReLU(), Linear(posnet_parms['n_hidden'], conv_feat_dim)]
            else:
                posnet_arch = [Linear(2, conv_feat_dim)]
            self.position_features = Sequential(*posnet_arch)
            transformer_layer = TransformerEncoderLayer(d_model=conv_feat_dim, **transformer_parms['transformer_encoder_parms'])
            self.transformer = TransformerEncoder(transformer_layer, num_layers=transformer_parms['n_blocks'])
            self.cls_token = Parameter(torch.ones(conv_feat_dim, 1) / conv_feat_dim)
            self._feat_dim = conv_feat_dim
        else:
            self.use_transformer = False
            self._feat_dim = convnet_parms['output_size'][0] * convnet_parms['output_size'][1] * conv_feat_dim

    def forward(self, x):
        x = self.conv_features(x)
        b, d, h, w = x.shape
        if self.use_transformer:
            y = torch.stack([torch.cat([torch.arange(h).unsqueeze(1)] * w, dim=1), torch.cat([torch.arange(w).unsqueeze(0)] * h, dim=0)])
            y = y.view([2, h * w]).transpose(1, 0)
            y = y.type(torch.float32)
            y = self.position_features(y).transpose(1, 0).view([d, h, w])
            y = y.unsqueeze(dim=0)
            x = x + y
            x = x.view([b, d, h * w])
            x = F.layer_norm(x.permute([0, 2, 1]), (d,)).permute([0, 2, 1])
            x = torch.cat([x, torch.stack([self.cls_token] * b)], dim=2)
            x = x.view([b * d, h * w + 1]).transpose(1, 0).view([h * w + 1, b, d])
            x = self.transformer(x)
            x = x[(-1), :, :]
        else:
            x = x.view([-1, self._feat_dim])
            x = F.layer_norm(x, (self._feat_dim,))
        return x

    def _add_embeddings_internal(self, images: List[Image]) ->List[Image]:
        image_tensor = torch.stack([image.data for image in images])
        image_embeddings = self.forward(image_tensor)
        for image_id, image in enumerate(images):
            image.set_embedding(self.name, image_embeddings[image_id])

    @property
    def embedding_length(self):
        return self._feat_dim

    def __str__(self):
        return self.name


class WordEmbeddings(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str=None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        self.embeddings = embeddings
        old_base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/'
        base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/'
        embeddings_path_v4 = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/'
        embeddings_path_v4_1 = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/'
        cache_dir = Path('embeddings')
        if embeddings.lower() == 'glove' or embeddings.lower() == 'en-glove':
            cached_path(f'{old_base_path}glove.gensim.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{old_base_path}glove.gensim', cache_dir=cache_dir)
        elif embeddings.lower() == 'turian' or embeddings.lower() == 'en-turian':
            cached_path(f'{embeddings_path_v4_1}turian.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4_1}turian', cache_dir=cache_dir)
        elif embeddings.lower() == 'extvec' or embeddings.lower() == 'en-extvec':
            cached_path(f'{old_base_path}extvec.gensim.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{old_base_path}extvec.gensim', cache_dir=cache_dir)
        elif embeddings.lower() == 'crawl' or embeddings.lower() == 'en-crawl':
            cached_path(f'{base_path}en-fasttext-crawl-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{base_path}en-fasttext-crawl-300d-1M', cache_dir=cache_dir)
        elif embeddings.lower() == 'news' or embeddings.lower() == 'en-news' or embeddings.lower() == 'en':
            cached_path(f'{base_path}en-fasttext-news-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{base_path}en-fasttext-news-300d-1M', cache_dir=cache_dir)
        elif embeddings.lower() == 'twitter' or embeddings.lower() == 'en-twitter':
            cached_path(f'{old_base_path}twitter.gensim.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{old_base_path}twitter.gensim', cache_dir=cache_dir)
        elif len(embeddings.lower()) == 2:
            cached_path(f'{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M', cache_dir=cache_dir)
        elif len(embeddings.lower()) == 7 and embeddings.endswith('-wiki'):
            cached_path(f'{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M', cache_dir=cache_dir)
        elif len(embeddings.lower()) == 8 and embeddings.endswith('-crawl'):
            cached_path(f'{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy', cache_dir=cache_dir)
            embeddings = cached_path(f'{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M', cache_dir=cache_dir)
        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        self.name: str = str(embeddings)
        self.static_embeddings = True
        if str(embeddings).endswith('.bin'):
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(str(embeddings), binary=True)
        else:
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(str(embeddings))
        self.field = field
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    @lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) ->torch.Tensor:
        if word in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word]
        elif word.lower() in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word.lower()]
        elif re.sub('\\d', '#', word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[re.sub('\\d', '#', word.lower())]
        elif re.sub('\\d', '0', word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[re.sub('\\d', '0', word.lower())]
        else:
            word_embedding = np.zeros(self.embedding_length, dtype='float')
        word_embedding = torch.tensor(word_embedding.tolist(), device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                word_embedding = self.get_cached_vec(word=word)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        if 'embeddings' not in self.__dict__:
            self.embeddings = self.name
        return f"'{self.embeddings}'"


class Dictionary:
    """
    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.
    """

    def __init__(self, add_unk=True):
        self.item2idx: Dict[str, int] = {}
        self.idx2item: List[str] = []
        self.multi_label: bool = False
        if add_unk:
            self.add_item('<unk>')

    def add_item(self, item: str) ->int:
        """
        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.
        :param item: a string for which to assign an id.
        :return: ID of string
        """
        item = item.encode('utf-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) ->int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item = item.encode('utf-8')
        if item in self.item2idx.keys():
            return self.item2idx[item]
        else:
            return 0

    def get_idx_for_items(self, items: List[str]) ->List[int]:
        """
        returns the IDs for each item of the list of string, otherwise 0 if not found
        :param items: List of string for which IDs are requested
        :return: List of ID of strings
        """
        if not hasattr(self, 'item2idx_not_encoded'):
            d = dict([(key.decode('UTF-8'), value) for key, value in self.item2idx.items()])
            self.item2idx_not_encoded = defaultdict(int, d)
        if not items:
            return []
        results = itemgetter(*items)(self.item2idx_not_encoded)
        if isinstance(results, int):
            return [results]
        return list(results)

    def get_items(self) ->List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))
        return items

    def __len__(self) ->int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode('UTF-8')

    def save(self, savefile):
        with open(savefile, 'wb') as f:
            mappings = {'idx2item': self.idx2item, 'item2idx': self.item2idx}
            pickle.dump(mappings, f)

    @classmethod
    def load_from_file(cls, filename: str):
        dictionary: Dictionary = Dictionary()
        with open(filename, 'rb') as f:
            mappings = pickle.load(f, encoding='latin1')
            idx2item = mappings['idx2item']
            item2idx = mappings['item2idx']
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name: str):
        if name == 'chars' or name == 'common-chars':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/common_characters'
            char_dict = cached_path(base_path, cache_dir='datasets')
            return Dictionary.load_from_file(char_dict)
        if name == 'chars-large' or name == 'common-chars-large':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/common_characters_large'
            char_dict = cached_path(base_path, cache_dir='datasets')
            return Dictionary.load_from_file(char_dict)
        if name == 'chars-xl' or name == 'common-chars-xl':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/common_characters_xl'
            char_dict = cached_path(base_path, cache_dir='datasets')
            return Dictionary.load_from_file(char_dict)
        return Dictionary.load_from_file(name)

    def __str__(self):
        tags = ', '.join(self.get_item_for_index(i) for i in range(min(len(self), 30)))
        return f'Dictionary with {len(self)} tags: {tags}'


class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(self, path_to_char_dict: str=None, char_embedding_dim: int=25, hidden_size_char: int=25):
        """Uses the default character dictionary if none provided."""
        super().__init__()
        self.name = 'Char'
        self.static_embeddings = False
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load('common-chars')
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(path_to_char_dict)
        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(len(self.char_dictionary.item2idx), self.char_embedding_dim)
        self.char_rnn = torch.nn.LSTM(self.char_embedding_dim, self.hidden_size_char, num_layers=1, bidirectional=True)
        self.__embedding_length = self.hidden_size_char * 2
        self

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        for sentence in sentences:
            tokens_char_indices = []
            for token in sentence.tokens:
                char_indices = [self.char_dictionary.get_idx_for_item(char) for char in token.text]
                tokens_char_indices.append(char_indices)
            tokens_sorted_by_length = sorted(tokens_char_indices, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(tokens_char_indices):
                for j, cj in enumerate(tokens_sorted_by_length):
                    if ci == cj:
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros((len(tokens_sorted_by_length), longest_token_in_sentence), dtype=torch.long, device=flair.device)
            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[(i), :chars2_length[i]] = torch.tensor(c, dtype=torch.long, device=flair.device)
            chars = tokens_mask
            character_embeddings = self.char_embedding(chars).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(character_embeddings, chars2_length)
            lstm_out, self.hidden = self.char_rnn(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros((outputs.size(0), outputs.size(2)), dtype=torch.float, device=flair.device)
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]
            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


class PooledFlairEmbeddings(TokenEmbeddings):

    def __init__(self, contextual_embeddings: Union[str, FlairEmbeddings], pooling: str='min', only_capitalized: bool=False, **kwargs):
        super().__init__()
        if type(contextual_embeddings) is str:
            self.context_embeddings: FlairEmbeddings = FlairEmbeddings(contextual_embeddings, **kwargs)
        else:
            self.context_embeddings: FlairEmbeddings = contextual_embeddings
        self.embedding_length = self.context_embeddings.embedding_length * 2
        self.name = self.context_embeddings.name + '-context'
        self.word_embeddings = {}
        self.word_count = {}
        self.only_capitalized = only_capitalized
        self.static_embeddings = False
        self.pooling = pooling
        if pooling == 'mean':
            self.aggregate_op = torch.add
        elif pooling == 'fade':
            self.aggregate_op = torch.add
        elif pooling == 'max':
            self.aggregate_op = torch.max
        elif pooling == 'min':
            self.aggregate_op = torch.min

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            None
            self.word_embeddings = {}
            self.word_count = {}

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        self.context_embeddings.embed(sentences)
        for sentence in sentences:
            for token in sentence.tokens:
                local_embedding = token._embeddings[self.context_embeddings.name].cpu()
                if token.text:
                    if token.text[0].isupper() or not self.only_capitalized:
                        if token.text not in self.word_embeddings:
                            self.word_embeddings[token.text] = local_embedding
                            self.word_count[token.text] = 1
                        else:
                            aggregated_embedding = self.aggregate_op(self.word_embeddings[token.text], local_embedding)
                            if self.pooling == 'fade':
                                aggregated_embedding /= 2
                            self.word_embeddings[token.text] = aggregated_embedding
                            self.word_count[token.text] += 1
        for sentence in sentences:
            for token in sentence.tokens:
                if token.text in self.word_embeddings:
                    base = self.word_embeddings[token.text] / self.word_count[token.text] if self.pooling == 'mean' else self.word_embeddings[token.text]
                else:
                    base = token._embeddings[self.context_embeddings.name]
                token.set_embedding(self.name, base)
        return sentences

    def embedding_length(self) ->int:
        return self.embedding_length

    def __setstate__(self, d):
        self.__dict__ = d
        if flair.device != 'cpu':
            for key in self.word_embeddings:
                self.word_embeddings[key] = self.word_embeddings[key].cpu()


class TransformerWordEmbeddings(TokenEmbeddings):

    def __init__(self, model: str='bert-base-uncased', layers: str='-1,-2,-3,-4', pooling_operation: str='first', batch_size: int=1, use_scalar_mix: bool=False, fine_tune: bool=False):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either take the first
        subtoken ('first'), the last subtoken ('last'), both first and last ('first_last') or a mean over all ('mean')
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model, config=config)
        self.name = 'transformer-word-' + str(model)
        self.model.eval()
        self.model
        if layers == 'all':
            hidden_states = self.model(torch.tensor([1], device=flair.device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size
        self.special_tokens = []
        if self.tokenizer._bos_token:
            self.special_tokens.append(self.tokenizer.bos_token)
        if self.tokenizer._cls_token:
            self.special_tokens.append(self.tokenizer.cls_token)
        self.begin_offset = 1
        if type(self.tokenizer) == XLNetTokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == T5Tokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == GPT2Tokenizer:
            self.begin_offset = 0

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        """Add embeddings to all words in a list of sentences."""
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]
        for batch in sentence_batches:
            self._add_embeddings_to_sentences(batch)
        return sentences

    @staticmethod
    def _remove_special_markup(text: str):
        text = re.sub('^Ġ', '', text)
        text = re.sub('^##', '', text)
        text = re.sub('^▁', '', text)
        text = re.sub('</w>$', '', text)
        return text

    def _get_processed_token_text(self, token: Token) ->str:
        pieces = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(token.text, add_special_tokens=False))
        token_text = ''
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        """Match subtokenization to Flair tokenization and extract embeddings from transformers for each token."""
        subtokenized_sentences = []
        subtokenized_sentences_token_lengths = []
        for sentence in sentences:
            tokenized_string = sentence.to_tokenized_string()
            ids = self.tokenizer.encode(tokenized_string, add_special_tokens=False)
            subtokenized_sentence = self.tokenizer.build_inputs_with_special_tokens(ids)
            subtokenized_sentences.append(torch.tensor(subtokenized_sentence, dtype=torch.long))
            subtokens = self.tokenizer.convert_ids_to_tokens(subtokenized_sentence)
            word_iterator = iter(sentence)
            token = next(word_iterator)
            token_text = self._get_processed_token_text(token)
            token_subtoken_lengths = []
            reconstructed_token = ''
            subtoken_count = 0
            for subtoken_id, subtoken in enumerate(subtokens):
                subtoken_count += 1
                subtoken = self._remove_special_markup(subtoken)
                reconstructed_token = reconstructed_token + subtoken
                if reconstructed_token in self.special_tokens and subtoken_id == 0:
                    reconstructed_token = ''
                    subtoken_count = 0
                if reconstructed_token.lower() == token_text:
                    token_subtoken_lengths.append(subtoken_count)
                    reconstructed_token = ''
                    subtoken_count = 0
                    if len(token_subtoken_lengths) < len(sentence):
                        token = next(word_iterator)
                        token_text = self._get_processed_token_text(token)
                    else:
                        break
            if token != sentence[-1]:
                log.error(f"Tokenization MISMATCH in sentence '{sentence.to_tokenized_string()}'")
                log.error(f"Last matched: '{token}'")
                log.error(f"Last sentence: '{sentence[-1]}'")
                log.error(f"subtokenized: '{subtokens}'")
            subtokenized_sentences_token_lengths.append(token_subtoken_lengths)
        longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))
        input_ids = torch.zeros([len(sentences), longest_sequence_in_batch], dtype=torch.long, device=flair.device)
        mask = torch.zeros([len(sentences), longest_sequence_in_batch], dtype=torch.long, device=flair.device)
        for s_id, sentence in enumerate(subtokenized_sentences):
            sequence_length = len(sentence)
            input_ids[s_id][:sequence_length] = sentence
            mask[s_id][:sequence_length] = torch.ones(sequence_length)
        hidden_states = self.model(input_ids, attention_mask=mask)[-1]
        gradient_context = torch.enable_grad() if self.fine_tune and self.training else torch.no_grad()
        with gradient_context:
            for sentence_idx, (sentence, subtoken_lengths) in enumerate(zip(sentences, subtokenized_sentences_token_lengths)):
                subword_start_idx = self.begin_offset
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):
                    subword_end_idx = subword_start_idx + number_of_subtokens
                    subtoken_embeddings: List[torch.FloatTensor] = []
                    for layer in self.layer_indexes:
                        current_embeddings = hidden_states[layer][sentence_idx][subword_start_idx:subword_end_idx]
                        if self.pooling_operation == 'first':
                            final_embedding: torch.FloatTensor = current_embeddings[0]
                        if self.pooling_operation == 'last':
                            final_embedding: torch.FloatTensor = current_embeddings[-1]
                        if self.pooling_operation == 'first_last':
                            final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])
                        if self.pooling_operation == 'mean':
                            all_embeddings: List[torch.FloatTensor] = [embedding.unsqueeze(0) for embedding in current_embeddings]
                            final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)
                        subtoken_embeddings.append(final_embedding)
                    if self.use_scalar_mix:
                        sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                        subtoken_embeddings = [sm_embeddings]
                    token.set_embedding(self.name, torch.cat(subtoken_embeddings))
                    subword_start_idx += number_of_subtokens

    def train(self, mode=True):
        if not self.fine_tune:
            pass
        else:
            super().train(mode)

    @property
    @abstractmethod
    def embedding_length(self) ->int:
        """Returns the length of the embedding vector."""
        if not self.use_scalar_mix:
            length = len(self.layer_indexes) * self.model.config.hidden_size
        else:
            length = self.model.config.hidden_size
        if self.pooling_operation == 'first_last':
            length *= 2
        return length

    def __setstate__(self, d):
        self.__dict__ = d
        model_name = self.name.split('transformer-word-')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


class FastTextEmbeddings(TokenEmbeddings):
    """FastText Embeddings with oov functionality"""

    def __init__(self, embeddings: str, use_local: bool=True, field: str=None):
        """
        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache
        if use_local is False.

        :param embeddings: path to your embeddings '.bin' file
        :param use_local: set this to False if you are using embeddings from a remote source
        """
        cache_dir = Path('embeddings')
        if use_local:
            if not Path(embeddings).exists():
                raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            embeddings = cached_path(f'{embeddings}', cache_dir=cache_dir)
        self.embeddings = embeddings
        self.name: str = str(embeddings)
        self.static_embeddings = True
        self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(str(embeddings))
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        self.field = field
        super().__init__()

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    @lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) ->torch.Tensor:
        try:
            word_embedding = self.precomputed_word_embeddings[word]
        except:
            word_embedding = np.zeros(self.embedding_length, dtype='float')
        word_embedding = torch.tensor(word_embedding.tolist(), device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                word_embedding = self.get_cached_vec(word)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"


class FlairDataset(Dataset):

    @abstractmethod
    def is_in_memory(self) ->bool:
        pass


class Corpus:

    def __init__(self, train: FlairDataset, dev: FlairDataset=None, test: FlairDataset=None, name: str='corpus'):
        self.name: str = name
        if test is None:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]
        if dev is None:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]
        self._train: FlairDataset = train
        self._test: FlairDataset = test
        self._dev: FlairDataset = dev

    @property
    def train(self) ->FlairDataset:
        return self._train

    @property
    def dev(self) ->FlairDataset:
        return self._dev

    @property
    def test(self) ->FlairDataset:
        return self._test

    def downsample(self, percentage: float=0.1, downsample_train=True, downsample_dev=True, downsample_test=True):
        if downsample_train:
            self._train = self._downsample_to_proportion(self.train, percentage)
        if downsample_dev:
            self._dev = self._downsample_to_proportion(self.dev, percentage)
        if downsample_test:
            self._test = self._downsample_to_proportion(self.test, percentage)
        return self

    def filter_empty_sentences(self):
        log.info('Filtering empty sentences')
        self._train = Corpus._filter_empty_sentences(self._train)
        self._test = Corpus._filter_empty_sentences(self._test)
        self._dev = Corpus._filter_empty_sentences(self._dev)
        log.info(self)

    @staticmethod
    def _filter_empty_sentences(dataset) ->Dataset:
        empty_sentence_indices = []
        non_empty_sentence_indices = []
        index = 0
        for batch in DataLoader(dataset):
            for sentence in batch:
                if len(sentence) == 0:
                    empty_sentence_indices.append(index)
                else:
                    non_empty_sentence_indices.append(index)
                index += 1
        subset = Subset(dataset, non_empty_sentence_indices)
        return subset

    def make_vocab_dictionary(self, max_tokens=-1, min_freq=1) ->Dictionary:
        """
        Creates a dictionary of all tokens contained in the corpus.
        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
        If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered
        to be added to the dictionary.
        :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)
        :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)
        :return: dictionary of tokens
        """
        tokens = self._get_most_common_tokens(max_tokens, min_freq)
        vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)
        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq) ->List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()
        tokens = []
        for token, freq in tokens_and_frequencies:
            if min_freq != -1 and freq < min_freq or max_tokens != -1 and len(tokens) == max_tokens:
                break
            tokens.append(token)
        return tokens

    def _get_all_tokens(self) ->List[str]:
        tokens = list(map(lambda s: s.tokens, self.train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map(lambda t: t.text, tokens))

    @staticmethod
    def _downsample_to_proportion(dataset: Dataset, proportion: float):
        sampled_size: int = round(len(dataset) * proportion)
        splits = random_split(dataset, [len(dataset) - sampled_size, sampled_size])
        return splits[1]

    def obtain_statistics(self, label_type: str=None, pretty_print: bool=True) ->dict:
        """
        Print statistics about the class distribution (only labels of sentences are taken into account) and sentence
        sizes.
        """
        json_string = {'TRAIN': self._obtain_statistics_for(self.train, 'TRAIN', label_type), 'TEST': self._obtain_statistics_for(self.test, 'TEST', label_type), 'DEV': self._obtain_statistics_for(self.dev, 'DEV', label_type)}
        if pretty_print:
            json_string = json.dumps(json_string, indent=4)
        return json_string

    @staticmethod
    def _obtain_statistics_for(sentences, name, tag_type) ->dict:
        if len(sentences) == 0:
            return {}
        classes_to_count = Corpus._count_sentence_labels(sentences)
        tags_to_count = Corpus._count_token_labels(sentences, tag_type)
        tokens_per_sentence = Corpus._get_tokens_per_sentence(sentences)
        label_size_dict = {}
        for l, c in classes_to_count.items():
            label_size_dict[l] = c
        tag_size_dict = {}
        for l, c in tags_to_count.items():
            tag_size_dict[l] = c
        return {'dataset': name, 'total_number_of_documents': len(sentences), 'number_of_documents_per_class': label_size_dict, 'number_of_tokens_per_tag': tag_size_dict, 'number_of_tokens': {'total': sum(tokens_per_sentence), 'min': min(tokens_per_sentence), 'max': max(tokens_per_sentence), 'avg': sum(tokens_per_sentence) / len(sentences)}}

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map(lambda x: len(x.tokens), sentences))

    @staticmethod
    def _count_sentence_labels(sentences):
        label_count = defaultdict(lambda : 0)
        for sent in sentences:
            for label in sent.labels:
                label_count[label.value] += 1
        return label_count

    @staticmethod
    def _count_token_labels(sentences, label_type):
        label_count = defaultdict(lambda : 0)
        for sent in sentences:
            for token in sent.tokens:
                if label_type in token.annotation_layers.keys():
                    label = token.get_tag(label_type)
                    label_count[label.value] += 1
        return label_count

    def __str__(self) ->str:
        return 'Corpus: %d train + %d dev + %d test sentences' % (len(self.train), len(self.dev), len(self.test))

    def make_label_dictionary(self, label_type: str=None) ->Dictionary:
        """
        Creates a dictionary of all labels assigned to the sentences in the corpus.
        :return: dictionary of labels
        """
        label_dictionary: Dictionary = Dictionary(add_unk=False)
        label_dictionary.multi_label = False
        data = ConcatDataset([self.train, self.test])
        loader = DataLoader(data, batch_size=1)
        log.info('Computing label dictionary. Progress:')
        for batch in Tqdm.tqdm(iter(loader)):
            for sentence in batch:
                labels = sentence.get_labels(label_type) if label_type is not None else sentence.labels
                for label in labels:
                    label_dictionary.add_item(label.value)
                if isinstance(sentence, Sentence):
                    for token in sentence.tokens:
                        for label in token.get_labels(label_type):
                            label_dictionary.add_item(label.value)
                if not label_dictionary.multi_label:
                    if len(labels) > 1:
                        label_dictionary.multi_label = True
        log.info(label_dictionary.idx2item)
        return label_dictionary

    def get_label_distribution(self):
        class_to_count = defaultdict(lambda : 0)
        for sent in self.train:
            for label in sent.labels:
                class_to_count[label.value] += 1
        return class_to_count

    def get_all_sentences(self) ->Dataset:
        return ConcatDataset([self.train, self.dev, self.test])

    def make_tag_dictionary(self, tag_type: str) ->Dictionary:
        tag_dictionary: Dictionary = Dictionary()
        tag_dictionary.add_item('O')
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                tag_dictionary.add_item(token.get_tag(tag_type).value)
        tag_dictionary.add_item('<START>')
        tag_dictionary.add_item('<STOP>')
        return tag_dictionary


class OneHotEmbeddings(TokenEmbeddings):
    """One-hot encoded embeddings. """

    def __init__(self, corpus: Corpus, field: str='text', embedding_length: int=300, min_freq: int=3):
        """
        Initializes one-hot encoded word embeddings and a trainable embedding layer
        :param corpus: you need to pass a Corpus in order to construct the vocabulary
        :param field: by default, the 'text' of tokens is embedded, but you can also embed tags such as 'pos'
        :param embedding_length: dimensionality of the trainable embedding layer
        :param min_freq: minimum frequency of a word to become part of the vocabulary
        """
        super().__init__()
        self.name = 'one-hot'
        self.static_embeddings = False
        self.min_freq = min_freq
        self.field = field
        tokens = list(map(lambda s: s.tokens, corpus.train))
        tokens = [token for sublist in tokens for token in sublist]
        if field == 'text':
            most_common = Counter(list(map(lambda t: t.text, tokens))).most_common()
        else:
            most_common = Counter(list(map(lambda t: t.get_tag(field).value, tokens))).most_common()
        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)
        self.vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            self.vocab_dictionary.add_item(token)
        self.__embedding_length = embedding_length
        None
        None
        self.embedding_layer = torch.nn.Embedding(len(self.vocab_dictionary), self.__embedding_length)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)
        self

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        one_hot_sentences = []
        for i, sentence in enumerate(sentences):
            if self.field == 'text':
                context_idxs = [self.vocab_dictionary.get_idx_for_item(t.text) for t in sentence.tokens]
            else:
                context_idxs = [self.vocab_dictionary.get_idx_for_item(t.get_tag(self.field).value) for t in sentence.tokens]
            one_hot_sentences.extend(context_idxs)
        one_hot_sentences = torch.tensor(one_hot_sentences, dtype=torch.long)
        embedded = self.embedding_layer.forward(one_hot_sentences)
        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return 'min_freq={}'.format(self.min_freq)


class HashEmbeddings(TokenEmbeddings):
    """Standard embeddings with Hashing Trick."""

    def __init__(self, num_embeddings: int=1000, embedding_length: int=300, hash_method='md5'):
        super().__init__()
        self.name = 'hash'
        self.static_embeddings = False
        self.__num_embeddings = num_embeddings
        self.__embedding_length = embedding_length
        self.__hash_method = hash_method
        self.embedding_layer = torch.nn.Embedding(self.__num_embeddings, self.__embedding_length)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)
        self

    @property
    def num_embeddings(self) ->int:
        return self.__num_embeddings

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:

        def get_idx_for_item(text):
            hash_function = hashlib.new(self.__hash_method)
            hash_function.update(bytes(str(text), 'utf-8'))
            return int(hash_function.hexdigest(), 16) % self.__num_embeddings
        hash_sentences = []
        for i, sentence in enumerate(sentences):
            context_idxs = [get_idx_for_item(t.text) for t in sentence.tokens]
            hash_sentences.extend(context_idxs)
        hash_sentences = torch.tensor(hash_sentences, dtype=torch.long)
        embedded = self.embedding_layer.forward(hash_sentences)
        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1
        return sentences

    def __str__(self):
        return self.name


class MuseCrosslingualEmbeddings(TokenEmbeddings):

    def __init__(self):
        self.name: str = f'muse-crosslingual'
        self.static_embeddings = True
        self.__embedding_length: int = 300
        self.language_embeddings = {}
        super().__init__()

    @lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, language_code: str, word: str) ->torch.Tensor:
        current_embedding_model = self.language_embeddings[language_code]
        if word in current_embedding_model:
            word_embedding = current_embedding_model[word]
        elif word.lower() in current_embedding_model:
            word_embedding = current_embedding_model[word.lower()]
        elif re.sub('\\d', '#', word.lower()) in current_embedding_model:
            word_embedding = current_embedding_model[re.sub('\\d', '#', word.lower())]
        elif re.sub('\\d', '0', word.lower()) in current_embedding_model:
            word_embedding = current_embedding_model[re.sub('\\d', '0', word.lower())]
        else:
            word_embedding = np.zeros(self.embedding_length, dtype='float')
        word_embedding = torch.tensor(word_embedding, device=flair.device, dtype=torch.float)
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[Sentence]:
        for i, sentence in enumerate(sentences):
            language_code = sentence.get_language_code()
            supported = ['en', 'de', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'et', 'fi', 'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl', 'pt', 'ro', 'ru', 'sk']
            if language_code not in supported:
                language_code = 'en'
            if language_code not in self.language_embeddings:
                log.info(f"Loading up MUSE embeddings for '{language_code}'!")
                webpath = 'https://alan-nlp.s3.eu-central-1.amazonaws.com/resources/embeddings-muse'
                cache_dir = Path('embeddings') / 'MUSE'
                cached_path(f'{webpath}/muse.{language_code}.vec.gensim.vectors.npy', cache_dir=cache_dir)
                embeddings_file = cached_path(f'{webpath}/muse.{language_code}.vec.gensim', cache_dir=cache_dir)
                self.language_embeddings[language_code] = gensim.models.KeyedVectors.load(str(embeddings_file))
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                word_embedding = self.get_cached_vec(language_code=language_code, word=word)
                token.set_embedding(self.name, word_embedding)
        return sentences

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def __str__(self):
        return self.name


def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str]=None) ->str:
    if path_inside_archive:
        return '({})#{}'.format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


class NILCEmbeddings(WordEmbeddings):

    def __init__(self, embeddings: str, model: str='skip', size: int=100):
        """
        Initializes portuguese classic word embeddings trained by NILC Lab (http://www.nilc.icmc.usp.br/embeddings).
        Constructor downloads required files if not there.
        :param embeddings: one of: 'fasttext', 'glove', 'wang2vec' or 'word2vec'
        :param model: one of: 'skip' or 'cbow'. This is not applicable to glove.
        :param size: one of: 50, 100, 300, 600 or 1000.
        """
        base_path = 'http://143.107.183.175:22980/download.php?file=embeddings/'
        cache_dir = Path('embeddings') / embeddings.lower()
        if embeddings.lower() == 'glove':
            cached_path(f'{base_path}{embeddings}/{embeddings}_s{size}.zip', cache_dir=cache_dir)
            embeddings = cached_path(f'{base_path}{embeddings}/{embeddings}_s{size}.zip', cache_dir=cache_dir)
        elif embeddings.lower() in ['fasttext', 'wang2vec', 'word2vec']:
            cached_path(f'{base_path}{embeddings}/{model}_s{size}.zip', cache_dir=cache_dir)
            embeddings = cached_path(f'{base_path}{embeddings}/{model}_s{size}.zip', cache_dir=cache_dir)
        elif not Path(embeddings).exists():
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        self.name: str = str(embeddings)
        self.static_embeddings = True
        log.info('Reading embeddings from %s' % embeddings)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(open_inside_zip(str(embeddings), cache_dir=cache_dir))
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super(TokenEmbeddings, self).__init__()

    @property
    def embedding_length(self) ->int:
        return self.__embedding_length

    def __str__(self):
        return self.name


class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass


class PairwiseBCELoss(SimilarityLoss):
    """
    Binary cross entropy between pair similarities and pair labels.
    """

    def __init__(self, balanced=False):
        super(PairwiseBCELoss, self).__init__()
        self.balanced = balanced

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets) - targets
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.balanced:
            weight_matrix = n * (targets / 2.0 + neg_targets / (2.0 * (n - 1)))
            bce_loss *= weight_matrix
        loss = bce_loss.mean()
        return loss


class RankingLoss(SimilarityLoss):
    """
    Triplet ranking loss between pair similarities and pair labels.
    """

    def __init__(self, margin=0.1, direction_weights=[0.5, 0.5]):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.direction_weights = direction_weights

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets) - targets
        ranking_loss_matrix_01 = neg_targets * F.relu(self.margin + inputs - torch.diag(inputs).view(n, 1))
        ranking_loss_matrix_10 = neg_targets * F.relu(self.margin + inputs - torch.diag(inputs).view(1, n))
        neg_targets_01_sum = torch.sum(neg_targets, dim=1)
        neg_targets_10_sum = torch.sum(neg_targets, dim=0)
        loss = self.direction_weights[0] * torch.mean(torch.sum(ranking_loss_matrix_01 / neg_targets_01_sum, dim=1)) + self.direction_weights[1] * torch.mean(torch.sum(ranking_loss_matrix_10 / neg_targets_10_sum, dim=0))
        return loss


class Result(object):

    def __init__(self, main_score: float, log_header: str, log_line: str, detailed_results: str):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LockedDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PairwiseBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RankingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (SimilarityLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (WordDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_flairNLP_flair(_paritybench_base):
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

