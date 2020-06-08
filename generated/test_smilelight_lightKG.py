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
utils = _module
convert = _module
ere = _module
re = _module
model = _module
module = _module
dataset = _module
preprocess = _module
erl = _module
ner = _module
model = _module
kr = _module
krl = _module
models = _module
transE = _module
model = _module
module = _module
get_neg_batch = _module
score_func = _module
ksq = _module
learning = _module
log = _module
setup = _module

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


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn.functional as F


FILE_LOG_FMT = (
    '[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
    )


STDOUT_LOG_FMT = (
    '%(log_color)s[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
    )


LEVEL_COLOR = {'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
    'ERROR': 'red', 'CRITICAL': 'red,bg_white'}


class ColoredFormatter(logging.Formatter):
    COLOR_MAP = {'black': '30', 'red': '31', 'green': '32', 'yellow': '33',
        'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37',
        'bg_black': '40', 'bg_red': '41', 'bg_green': '42', 'bg_yellow':
        '43', 'bg_blue': '44', 'bg_magenta': '45', 'bg_cyan': '46',
        'bg_white': '47', 'light_black': '1;30', 'light_red': '1;31',
        'light_green': '1;32', 'light_yellow': '1;33', 'light_blue': '1;34',
        'light_magenta': '1;35', 'light_cyan': '1;36', 'light_white':
        '1;37', 'light_bg_black': '100', 'light_bg_red': '101',
        'light_bg_green': '102', 'light_bg_yellow': '103', 'light_bg_blue':
        '104', 'light_bg_magenta': '105', 'light_bg_cyan': '106',
        'light_bg_white': '107'}

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


STDOUT_DATE_FMT = '%Y-%m-%d %H:%M:%S'


FILE_DATE_FMT = '%Y-%m-%d %H:%M:%S'


def _get_logger(log_to_file=True, log_filename='default.log', log_level='DEBUG'
    ):
    _logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(ColoredFormatter(fmt=STDOUT_LOG_FMT,
        datefmt=STDOUT_DATE_FMT))
    _logger.addHandler(stdout_handler)
    if log_to_file:
        _tmp_path = os.path.dirname(os.path.abspath(__file__))
        _tmp_path = os.path.join(_tmp_path, '../logs/{}'.format(log_filename))
        file_handler = logging.handlers.TimedRotatingFileHandler(_tmp_path,
            when='midnight', backupCount=30)
        file_formatter = logging.Formatter(fmt=FILE_LOG_FMT, datefmt=
            FILE_DATE_FMT)
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


def handle_line(entity1, entity2, sentence, begin_e1_token='<e1>',
    end_e1_token='</e1>', begin_e2_token='<e2>', end_e2_token='</e2>'):
    assert entity1 in sentence
    assert entity2 in sentence
    sentence = sentence.replace(entity1, begin_e1_token + entity1 +
        end_e1_token)
    sentence = sentence.replace(entity2, begin_e2_token + entity2 +
        end_e2_token)
    sentence = ' '.join(jieba.cut(sentence))
    sentence = sentence.replace('< e1 >', begin_e1_token)
    sentence = sentence.replace('< / e1 >', end_e1_token)
    sentence = sentence.replace('< e2 >', begin_e2_token)
    sentence = sentence.replace('< / e2 >', end_e2_token)
    return sentence.split()


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
            raise AttributeError(
                'config object must init save_path attr in init method!')
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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextCNN(BaseModel):

    def __init__(self, args):
        super(TextCNN, self).__init__(args)
        self.class_num = args.class_num
        self.chanel_num = 1
        self.filter_num = args.filter_num
        self.filter_sizes = args.filter_sizes
        self.vocabulary_size = args.vocabulary_size
        self.embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(self.vocabulary_size, self.
            embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}'.format(args.vector_path)
                )
            vectors = Vectors(args.vector_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze
                =not args.non_static).to(DEVICE)
        if args.multichannel:
            self.embedding2 = nn.Embedding(self.vocabulary_size, self.
                embedding_dimension).from_pretrained(args.vectors).to(DEVICE)
            self.chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv2d(self.chanel_num, self.
            filter_num, (size, self.embedding_dimension)) for size in self.
            filter_sizes]).to(DEVICE)
        self.dropout = nn.Dropout(args.dropout).to(DEVICE)
        self.fc = nn.Linear(len(self.filter_sizes) * self.filter_num, self.
            class_num).to(DEVICE)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack((self.embedding(x), self.embedding2(x)), dim=1).to(
                DEVICE)
        else:
            x = self.embedding(x).to(DEVICE)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def light_tokenize(sequence: str):
    return [sequence]


p2 = torch.nn.PairwiseDistance(p=2)


def l2_score(head, rel, tail):
    return p2(tail - head, rel)


p1 = torch.nn.PairwiseDistance(p=1)


def l1_score(head, rel, tail):
    return p1(tail - head, rel)


class TransE(BaseModel):

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.entity_num = args.entity_num
        self.rel_num = args.rel_num
        self.embedding_dimension = args.embedding_dim
        self.entity_embedding = nn.Embedding(self.entity_num, self.
            embedding_dimension).to(DEVICE)
        self.rel_embedding = nn.Embedding(self.rel_num, self.
            embedding_dimension).to(DEVICE)
        if args.score_func == 'l1':
            self.score_func = l1_score
        else:
            self.score_func = l2_score

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embedding.weight)
        nn.init.xavier_normal_(self.rel_embedding.weight)

    def forward(self, head, rel, tail):
        vec_head = self.entity_embedding(head).view(-1, self.
            embedding_dimension)
        vec_rel = self.rel_embedding(rel).view(-1, self.embedding_dimension)
        vec_tail = self.entity_embedding(tail).view(-1, self.
            embedding_dimension)
        vec_head = F.normalize(vec_head)
        vec_rel = F.normalize(vec_rel)
        vec_tail = F.normalize(vec_tail)
        return self.score_func(vec_head, vec_rel, vec_tail)


MODELS = {'TransE': TransE}


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_smilelight_lightKG(_paritybench_base):
    pass
