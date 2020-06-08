import sys
_module = sys.modules[__name__]
del sys
__main__ = _module
extractor = _module
base_extractor = _module
sphinx_stt_extractor = _module
keyword_data_generator = _module
url_fetcher = _module
utils = _module
color_print = _module
file_utils = _module
wordset = _module
youtube_processor = _module
youtube_searcher = _module
measure_power = _module
power_consumption_benchmark = _module
wattsup_server = _module
server = _module
service = _module
client = _module
manage_audio = _module
model = _module
record = _module
freeze = _module
input_data = _module
models = _module
train = _module
speech_demo = _module
speech_demo_tk = _module
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


import uuid


import numpy as np


from enum import Enum


import math


import random


import re


from torch.autograd import Variable


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data as data


from collections import ChainMap


import copy


class SerializableModule(nn.Module):

    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda
            storage, loc: storage))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_castorini_honk(_paritybench_base):
    pass
