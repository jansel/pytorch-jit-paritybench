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


FILE_DATE_FMT = '%Y-%m-%d %H:%M:%S'


STDOUT_LOG_FMT = (
    '%(log_color)s[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
    )


STDOUT_DATE_FMT = '%Y-%m-%d %H:%M:%S'


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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def light_tokenize(sequence: str):
    return [sequence]


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_smilelight_lightKG(_paritybench_base):
    pass
