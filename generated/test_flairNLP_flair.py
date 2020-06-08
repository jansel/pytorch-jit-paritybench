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


from abc import abstractmethod


from typing import Union


from typing import List


from torch.nn import ParameterList


from torch.nn import Parameter


import torch


import logging


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


from typing import Dict


from collections import Counter


from functools import lru_cache


import re


import numpy as np


import torch.nn as nn


import math


from torch.optim import Optimizer


from typing import Optional


from typing import Callable


import torch.nn


from torch.nn.parameter import Parameter


from torch.utils.data import DataLoader


from torch import nn


import itertools


import warnings


import random


from torch import cuda


from torch.utils.data import Dataset


from torch.optim.sgd import SGD


import copy


import inspect


from torch.utils.data.dataset import ConcatDataset


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
            raise ValueError(
                'Incorrect label value provided. Label value needs to be set.')
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
        return self.annotation_layers[label_type
            ] if label_type in self.annotation_layers else []

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
        embeddings = [self._embeddings[embed] for embed in sorted(self.
            _embeddings.keys())]
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.tensor([], device=flair.device)

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.
            keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True)

    def clear_embeddings(self, embedding_names: List[str]=None):
        if embedding_names is None:
            self._embeddings: Dict = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]


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


class Token(DataPoint):
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(self, text: str, idx: int=None, head_id: int=None,
        whitespace_after: bool=True, start_position: int=None):
        super().__init__()
        self.text: str = text
        self.idx: int = idx
        self.head_id: int = head_id
        self.whitespace_after: bool = whitespace_after
        self.start_pos = start_position
        self.end_pos = start_position + len(text
            ) if start_position is not None else None
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
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.
            keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device: str, pin_memory: bool=False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True)

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
            embed = self._embeddings[embed].to(flair.device)
            if (flair.embedding_storage_mode == 'cpu' and embed.device !=
                flair.device):
                embed = embed.to(flair.device)
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
        return 'Token: {} {}'.format(self.idx, self.text
            ) if self.idx is not None else 'Token: {}'.format(self.text)

    def __repr__(self) ->str:
        return 'Token: {} {}'.format(self.idx, self.text
            ) if self.idx is not None else 'Token: {}'.format(self.text)


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
                tokens.append(Token(text=word, start_position=
                    start_position, whitespace_after=True))
            word = ''
        else:
            word += char
    index += 1
    if len(word) > 0:
        start_position = index - len(word)
        tokens.append(Token(text=word, start_position=start_position,
            whitespace_after=False))
    return tokens


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
        return {'text': self.to_original_text(), 'start_pos': self.
            start_pos, 'end_pos': self.end_pos, 'labels': self.labels}

    def __str__(self) ->str:
        ids = ','.join([str(t.idx) for t in self.tokens])
        label_string = ' '.join([str(label) for label in self.labels])
        labels = (f'   [− Labels: {label_string}]' if self.labels is not
            None else '')
        return 'Span [{}]: "{}"{}'.format(ids, self.text, labels)

    def __repr__(self) ->str:
        ids = ','.join([str(t.idx) for t in self.tokens])
        return '<{}-span ({}): "{}">'.format(self.tag, ids, self.text
            ) if self.tag is not None else '<span ({}): "{}">'.format(ids,
            self.text)

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def score(self):
        return self.labels[0].score


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
            start_position = (current_offset + 1 if current_offset > 0 else
                current_offset)
        if word:
            token = Token(text=word, start_position=start_position,
                whitespace_after=True)
            tokens.append(token)
        if (previous_token is not None and word_offset - 1 ==
            previous_word_offset):
            previous_token.whitespace_after = False
        current_offset = word_offset + len(word)
        previous_word_offset = current_offset - 1
        previous_token = token
    return tokens


log = logging.getLogger('flair')


class Sentence(DataPoint):
    """
       A Sentence is a list of Tokens and is used to represent a sentence or text fragment.
    """

    def __init__(self, text: str=None, use_tokenizer: Union[bool, Callable[
        [str], List[Token]]]=False, language_code: str=None):
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
            log.warning(
                'ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?'
                )
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
            if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:
                ] != tag_value[2:] and in_span:
                starts_new_span = True
            if (starts_new_span or not in_span) and len(current_span) > 0:
                scores = [t.get_labels(label_type)[0].score for t in
                    current_span]
                span_score = sum(scores) / len(scores)
                if span_score > min_score:
                    span = Span(current_span)
                    span.add_label(label_type=label_type, value=sorted(tags
                        .items(), key=lambda k_v: k_v[1], reverse=True)[0][
                        0], score=span_score)
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
                span.add_label(label_type=label_type, value=sorted(tags.
                    items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                    score=span_score)
                spans.append(span)
        return spans

    @property
    def embedding(self):
        return self.get_embedding()

    def set_embedding(self, name: str, vector: torch.tensor):
        device = flair.device
        if flair.embedding_storage_mode == 'cpu' and len(self._embeddings.
            keys()) > 0:
            device = next(iter(self._embeddings.values())).device
        if device != vector.device:
            vector = vector.to(device)
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
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking
                        =True)
        for token in self:
            token.to(device, pin_memory)

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

    def convert_tag_scheme(self, tag_type: str='ner', target_scheme: str='iob'
        ):
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
        return {'text': self.to_original_text(), 'labels': labels,
            'entities': entities}

    def __getitem__(self, idx: int) ->Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()
        sentence_labels = (f'  − Sentence-Labels: {self.annotation_layers}' if
            self.annotation_layers != {} else '')
        token_labels = (f'  − Token-Labels: "{tagged_string}"' if 
            tokenized_string != tagged_string else '')
        return (
            f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'
            )

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_label(tag_type, token.get_tag(tag_type).value, token
                    .get_tag(tag_type).score)
            s.add_token(nt)
        return s

    def __str__(self) ->str:
        tagged_string = self.to_tagged_string()
        tokenized_string = self.to_tokenized_string()
        sentence_labels = (f'  − Sentence-Labels: {self.annotation_layers}' if
            self.annotation_layers != {} else '')
        token_labels = (f'  − Token-Labels: "{tagged_string}"' if 
            tokenized_string != tagged_string else '')
        return (
            f'Sentence: "{tokenized_string}"   [− Tokens: {len(self)}{token_labels}{sentence_labels}]'
            )

    def __len__(self) ->int:
        return len(self.tokens)

    def get_language_code(self) ->str:
        if self.language_code is None:
            import langdetect
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

    def embed(self, sentences: Union[Sentence, List[Sentence]]) ->List[Sentence
        ]:
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
    def _add_embeddings_internal(self, sentences: List[Sentence]) ->List[
        Sentence]:
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
        self.scalar_parameters = ParameterList([Parameter(torch.tensor([
            initial_scalar_parameters[i]], dtype=torch.float, device=flair.
            device), requires_grad=trainable) for i in range(mixture_size)])
        self.gamma = Parameter(torch.tensor([1.0], dtype=torch.float,
            device=flair.device), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) ->torch.Tensor:
        """
        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        :param tensors: list of input tensors
        :return: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error(
                '{} tensors were passed, but the module was initialized to mix {} tensors.'
                .format(len(tensors), self.mixture_size))
        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for
            parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass


class Result(object):

    def __init__(self, main_score: float, log_header: str, log_line: str,
        detailed_results: str):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


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
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.
                dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.
                dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.
            dropout_rate)
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
        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.
            dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.dropout_rate, inplace_str)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_flairNLP_flair(_paritybench_base):
    pass

    def test_000(self):
        self._check(SimilarityLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(LockedDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(WordDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
