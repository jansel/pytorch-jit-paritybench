import sys
_module = sys.modules[__name__]
del sys
test_krl = _module
test_ner = _module
test_re = _module
test_srl = _module
lightkg = _module
base = _module
config = _module
model = _module
module = _module
tool = _module
common = _module
entity = _module
relation = _module
ede = _module
srl = _module
model = _module
module = _module
tool = _module
utils = _module
convert = _module
ere = _module
re = _module
model = _module
module = _module
tool = _module
dataset = _module
preprocess = _module
erl = _module
ner = _module
model = _module
module = _module
tool = _module
kr = _module
krl = _module
models = _module
transE = _module
model = _module
module = _module
tool = _module
get_neg_batch = _module
score_func = _module
ksq = _module
learning = _module
log = _module
setup = _module

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


import torch


import torch.nn as nn


from torchtext.vocab import Vectors


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torchtext.data import Dataset


from torchtext.data import Field


from torchtext.data import BucketIterator


from torchtext.data import ReversibleField


from torchtext.datasets import SequenceTaggingDataset


from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score


from sklearn.metrics import recall_score


from sklearn.metrics import precision_score


import torch.nn.functional as F


import re


from torchtext.data import Iterator


from torchtext.data import TabularDataset


import random


LEVEL_COLOR = {'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}


class ColoredFormatter(logging.Formatter):
    COLOR_MAP = {'black': '30', 'red': '31', 'green': '32', 'yellow': '33', 'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37', 'bg_black': '40', 'bg_red': '41', 'bg_green': '42', 'bg_yellow': '43', 'bg_blue': '44', 'bg_magenta': '45', 'bg_cyan': '46', 'bg_white': '47', 'light_black': '1;30', 'light_red': '1;31', 'light_green': '1;32', 'light_yellow': '1;33', 'light_blue': '1;34', 'light_magenta': '1;35', 'light_cyan': '1;36', 'light_white': '1;37', 'light_bg_black': '100', 'light_bg_red': '101', 'light_bg_green': '102', 'light_bg_yellow': '103', 'light_bg_blue': '104', 'light_bg_magenta': '105', 'light_bg_cyan': '106', 'light_bg_white': '107'}

    def __init__(self, fmt, datefmt):
        super(ColoredFormatter, self).__init__(fmt, datefmt)

    def parse_color(self, level_name):
        color_name = LEVEL_COLOR.get(level_name, '')
        if not color_name:
            return ''
        color_value = []
        color_name = color_name.split(',')
        for _cn in color_name:
            color_code = self.COLOR_MAP.get(_cn, '')
            if color_code:
                color_value.append(color_code)
        return '\x1b[' + ';'.join(color_value) + 'm'

    def format(self, record):
        record.log_color = self.parse_color(record.levelname)
        message = super(ColoredFormatter, self).format(record) + '\x1b[0m'
        return message


FILE_DATE_FMT = '%Y-%m-%d %H:%M:%S'


FILE_LOG_FMT = '[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'


STDOUT_DATE_FMT = '%Y-%m-%d %H:%M:%S'


STDOUT_LOG_FMT = '%(log_color)s[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'


def _get_logger(log_to_file=True, log_filename='default.log', log_level='DEBUG'):
    _logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(ColoredFormatter(fmt=STDOUT_LOG_FMT, datefmt=STDOUT_DATE_FMT))
    _logger.addHandler(stdout_handler)
    if log_to_file:
        _tmp_path = os.path.dirname(os.path.abspath(__file__))
        _tmp_path = os.path.join(_tmp_path, '../logs/{}'.format(log_filename))
        file_handler = logging.handlers.TimedRotatingFileHandler(_tmp_path, when='midnight', backupCount=30)
        file_formatter = logging.Formatter(fmt=FILE_LOG_FMT, datefmt=FILE_DATE_FMT)
        file_handler.setFormatter(file_formatter)
        _logger.addHandler(file_handler)
    _logger.setLevel(log_level)
    return _logger


logger = _get_logger(log_to_file=False)


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_path

    def load(self, path=None):
        path = path if path else self.save_path
        map_location = None if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, 'model.pkl')
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loadding model from {}'.format(model_path))

    def save(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, 'model.pkl')
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLstmCrf(BaseModel):

    def __init__(self, args):
        super(BiLstmCrf, self).__init__(args)
        self.args = args
        self.hidden_dim = 300
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static)
        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.dropout)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num)
        self.crflayer = CRF(self.tag_num)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def loss(self, x, sent_lengths, y):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        return self.crflayer(emissions, y, mask=mask)

    def forward(self, x, sent_lengths):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, sent_lengths):
        x = self.embedding(sentence.to(DEVICE))
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size)
        y = self.hidden2label(lstm_out)
        return y


DEFAULT_CONFIG = {'save_path': './saves'}


class BaseConfig(object):

    def __init__(self):
        pass

    @staticmethod
    def load(path=DEFAULT_CONFIG['save_path']):
        config = None
        config_path = os.path.join(path, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logger.info('loadding config from {}'.format(config_path))
        return config

    def save(self, path=None):
        if not hasattr(self, 'save_path'):
            raise AttributeError('config object must init save_path attr in init method!')
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, 'config.pkl')
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved config to {}'.format(config_path))


class Config(BaseConfig):

    def __init__(self, entity_vocab, rel_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.entity_vocab = entity_vocab
        self.rel_vocab = rel_vocab
        self.entity_num = len(self.entity_vocab)
        self.rel_num = len(self.rel_vocab)
        for name, value in kwargs.items():
            setattr(self, name, value)


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    assert len(words) == len(tags)
    events = {}

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('-')[0] == 'O' or tags[i + 1] == 'rel':
            events[temp_type] = ''.join(words[begin:i + 1])
    for i, tag in enumerate(tags):
        if tag == 'rel':
            events['rel'] = words[i]
        elif tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            begin = i
            temp_type = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[0] == 'I':
            check_if_closing_range()
    return events


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'rel':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iobes_ranges(words, tags):
    new_tags = iobes_iob(tags)
    return iob_ranges(words, new_tags)


def light_tokenize(text):
    return [text]


ENTITY = Field(tokenize=light_tokenize, batch_first=True)


RELATION = Field(tokenize=light_tokenize, batch_first=True)


Fields = [('head', ENTITY), ('rel', RELATION), ('tail', ENTITY)]


POS = Field(sequential=True, tokenize=light_tokenize)


TAG = ReversibleField(sequential=True, tokenize=light_tokenize, is_target=True, unk_token=None)


TEXT = Field(sequential=True, tokenize=light_tokenize, include_lengths=True)


class SRL(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None
        self._pos_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = srl_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = srl_tool.get_dataset(dev_path)
            word_vocab, pos_vocab, tag_vocab = srl_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, pos_vocab, tag_vocab = srl_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._pos_vocab = pos_vocab
        self._tag_vocab = tag_vocab
        train_iter = srl_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, pos_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = -bilstmcrf.loss(item_text_sentences, item_text_lengths, item.pos, item.rel, item.tag) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        bilstmcrf.save()

    def predict(self, word_list, pos_list, rel_list):
        self._model.eval()
        assert len(word_list) == len(pos_list) == len(rel_list)
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in word_list]).view(-1, 1)
        len_text = torch.tensor([len(vec_text)])
        vec_pos = torch.tensor([self._pos_vocab.stoi[x] for x in pos_list]).view(-1, 1)
        vec_rel = torch.tensor([int(x) for x in rel_list]).view(-1, 1)
        vec_predict = self._model(vec_text, vec_pos, vec_rel, len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        return iobes_ranges([x for x in word_list], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
        self._pos_vocab = config.pos_vocab

    def test(self, test_path):
        test_dataset = srl_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        for dev_item in tqdm(dev_dataset):
            item_score = srl_tool.get_score(self._model, dev_item.text, dev_item.tag, dev_item.pos, dev_item.rel, self._word_vocab, self._tag_vocab, self._pos_vocab)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)


class TextCNN(BaseModel):

    def __init__(self, args):
        super(TextCNN, self).__init__(args)
        self.class_num = args.class_num
        self.chanel_num = 1
        self.filter_num = args.filter_num
        self.filter_sizes = args.filter_sizes
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path))
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.embedding_dimension).from_pretrained(args.vectors)
            self.chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv2d(self.chanel_num, self.filter_num, (size, self.embedding_dimension)) for size in self.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(self.filter_sizes) * self.filter_num, self.class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack((self.embedding(x), self.embedding2(x)), dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class LSTMClassifier(BaseModel):

    def __init__(self, args):
        super(LSTMClassifier, self).__init__(args)
        self.hidden_dim = 300
        self.class_num = args.class_num
        self.batch_size = args.batch_size
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.embedding_dimension).from_pretrained(args.vectors)
        else:
            self.embedding2 = None
        self.lstm = nn.LSTM(self.embedding_dimension, self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, self.class_num)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out
        final = lstm_out[-1]
        y = self.hidden2label(final)
        return y


LABEL = Field(sequential=False, unk_token=None)


def handle_line(entity1, entity2, sentence, begin_e1_token='<e1>', end_e1_token='</e1>', begin_e2_token='<e2>', end_e2_token='</e2>'):
    assert entity1 in sentence
    assert entity2 in sentence
    sentence = sentence.replace(entity1, begin_e1_token + entity1 + end_e1_token)
    sentence = sentence.replace(entity2, begin_e2_token + entity2 + end_e2_token)
    sentence = ' '.join(jieba.cut(sentence))
    sentence = sentence.replace('< e1 >', begin_e1_token)
    sentence = sentence.replace('< / e1 >', end_e1_token)
    sentence = sentence.replace('< e2 >', begin_e2_token)
    sentence = sentence.replace('< / e2 >', end_e2_token)
    return sentence.split()


class REDataset(Dataset):
    """Defines a Dataset of relation extraction format.
    eg:
    钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
    元武	元华	unknown	于师傅在一次京剧表演中，选了元龙（洪金宝）、元楼（元奎）、元彪、成龙、元华、元武、元泰7人担任七小福的主角。
    """

    def __init__(self, path, fields, encoding='utf-8', **kwargs):
        examples = []
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                chunks = line.split()
                entity_1, entity_2, relation, sentence = tuple(chunks)
                sentence_list = handle_line(entity_1, entity_2, sentence)
                examples.append(Example.fromlist((sentence_list, relation), fields))
        super(REDataset, self).__init__(examples, fields, **kwargs)


class NER(Module):
    """
    """

    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None

    def train(self, train_path, save_path=DEFAULT_CONFIG['save_path'], dev_path=None, vectors_path=None, **kwargs):
        train_dataset = ner_tool.get_dataset(train_path)
        if dev_path:
            dev_dataset = ner_tool.get_dataset(dev_path)
            word_vocab, tag_vocab = ner_tool.get_vocab(train_dataset, dev_dataset)
        else:
            word_vocab, tag_vocab = ner_tool.get_vocab(train_dataset)
        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        train_iter = ner_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, save_path=save_path, vector_path=vectors_path, **kwargs)
        bilstmcrf = BiLstmCrf(config)
        self._model = bilstmcrf
        optim = torch.optim.Adam(bilstmcrf.parameters(), lr=config.lr)
        for epoch in range(config.epoch):
            bilstmcrf.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                bilstmcrf.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = -bilstmcrf.loss(item_text_sentences, item_text_lengths, item.tag) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
            if dev_path:
                dev_score = self._validate(dev_dataset)
                logger.info('dev score:{}'.format(dev_score))
            adjust_learning_rate(optim, config.lr / (1 + (epoch + 1) * config.lr_decay))
        config.save()
        bilstmcrf.save()

    def predict(self, text):
        self._model.eval()
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text])
        len_text = torch.tensor([len(vec_text)])
        vec_predict = self._model(vec_text.view(-1, 1), len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        return iob_ranges([x for x in text], tag_predict)

    def load(self, save_path=DEFAULT_CONFIG['save_path']):
        config = Config.load(save_path)
        bilstmcrf = BiLstmCrf(config)
        bilstmcrf.load()
        self._model = bilstmcrf
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab

    def test(self, test_path):
        test_dataset = ner_tool.get_dataset(test_path)
        test_score = self._validate(test_dataset)
        logger.info('test score:{}'.format(test_score))

    def _validate(self, dev_dataset):
        self._model.eval()
        dev_score_list = []
        for dev_item in tqdm(dev_dataset):
            item_score = ner_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab, self._tag_vocab)
            dev_score_list.append(item_score)
        return sum(dev_score_list) / len(dev_score_list)


p1 = torch.nn.PairwiseDistance(p=1)


def l1_score(head, rel, tail):
    return p1(tail - head, rel)


p2 = torch.nn.PairwiseDistance(p=2)


def l2_score(head, rel, tail):
    return p2(tail - head, rel)


class TransE(BaseModel):

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.entity_num = args.entity_num
        self.rel_num = args.rel_num
        self.embedding_dimension = args.embedding_dim
        self.entity_embedding = nn.Embedding(self.entity_num, self.embedding_dimension)
        self.rel_embedding = nn.Embedding(self.rel_num, self.embedding_dimension)
        if args.score_func == 'l1':
            self.score_func = l1_score
        else:
            self.score_func = l2_score

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embedding.weight)
        nn.init.xavier_normal_(self.rel_embedding.weight)

    def forward(self, head, rel, tail):
        vec_head = self.entity_embedding(head).view(-1, self.embedding_dimension)
        vec_rel = self.rel_embedding(rel).view(-1, self.embedding_dimension)
        vec_tail = self.entity_embedding(tail).view(-1, self.embedding_dimension)
        vec_head = F.normalize(vec_head)
        vec_rel = F.normalize(vec_rel)
        vec_tail = F.normalize(vec_tail)
        return self.score_func(vec_head, vec_rel, vec_tail)


MODELS = {'TransE': TransE}


def get_neg_batch(head, tail, entity_num):
    neg_head = head.clone()
    neg_tail = tail.clone()
    if random.random() > 0.5:
        offset_tensor = torch.randint_like(neg_head, entity_num)
        neg_head = (neg_head + offset_tensor) % entity_num
    else:
        offset_tensor = torch.randint_like(neg_tail, entity_num)
        neg_tail = (neg_tail + offset_tensor) % entity_num
    return neg_head, neg_tail


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (TransE,
     lambda: ([], {'args': _mock_config(save_path=4, entity_num=4, rel_num=4, embedding_dim=4, score_func=4)}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     True),
]

class Test_smilelight_lightKG(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

