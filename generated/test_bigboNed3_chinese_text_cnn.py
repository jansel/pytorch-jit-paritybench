import sys
_module = sys.modules[__name__]
del sys
dataset = _module
main = _module
model = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,
                freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension
                ).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv2d(chanel_num, filter_num, (size,
            embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_bigboNed3_chinese_text_cnn(_paritybench_base):
    pass
