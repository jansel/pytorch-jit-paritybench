import sys
_module = sys.modules[__name__]
del sys
aeslc = _module
ai2_arc = _module
art = _module
billsum = _module
blimp = _module
blog_authorship_corpus = _module
boolq = _module
break_data = _module
cfq = _module
civil_comments = _module
cmrc2018 = _module
cnn_dailymail = _module
coarse_discourse = _module
com_qa = _module
commonsense_qa = _module
coqa = _module
cornell_movie_dialog = _module
cos_e = _module
cosmos_qa = _module
crime_and_punish = _module
csv = _module
definite_pronoun_resolution = _module
discofuse = _module
drop = _module
empathetic_dialogues = _module
eraser_multi_rc = _module
esnli = _module
event2Mind = _module
flores = _module
fquad = _module
gap = _module
germeval_14 = _module
gigaword = _module
glue = _module
hansards = _module
hellaswag = _module
imdb = _module
jeopardy = _module
json = _module
kor_nli = _module
lc_quad = _module
librispeech_lm = _module
lm1b = _module
math_dataset = _module
math_qa = _module
mlqa = _module
movie_rationales = _module
multi_news = _module
multi_nli = _module
multi_nli_mismatch = _module
newsroom = _module
openbookqa = _module
opinosis = _module
para_crawl = _module
qa4mre = _module
qangaroo = _module
qasc = _module
quarel = _module
quartz = _module
quoref = _module
race = _module
reclor = _module
reddit = _module
reddit_tifu = _module
scan = _module
scicite = _module
scientific_papers = _module
scifact = _module
sciq = _module
scitail = _module
sentiment140 = _module
snli = _module
social_i_qa = _module
squad = _module
squad_it = _module
squad_v1_pt = _module
squad_v2 = _module
super_glue = _module
ted_hrlr = _module
ted_multi = _module
tiny_shakespeare = _module
trivia_qa = _module
tydiqa = _module
ubuntu_dialogs_corpus = _module
wiki40b = _module
wiki_qa = _module
wiki_split = _module
wikihow = _module
wikipedia = _module
wikitext = _module
winogrande = _module
wiqa = _module
wmt14 = _module
wmt_utils = _module
wmt15 = _module
wmt16 = _module
wmt17 = _module
wmt18 = _module
wmt19 = _module
wmt_t2t = _module
x_stance = _module
xcopa = _module
xnli = _module
xquad = _module
xsum = _module
xtreme = _module
yelp_polarity = _module
bertscore = _module
bleu = _module
coval = _module
gleu = _module
rouge = _module
sacrebleu = _module
seqeval = _module
evaluate = _module
setup = _module
nlp = _module
arrow_dataset = _module
arrow_reader = _module
arrow_writer = _module
builder = _module
commands = _module
convert = _module
download = _module
dummy_data = _module
env = _module
test = _module
user = _module
datasets = _module
features = _module
hf_api = _module
info = _module
inspect = _module
load = _module
metric = _module
metrics = _module
naming = _module
splits = _module
utils = _module
download_manager = _module
file_utils = _module
info_utils = _module
mock_download_manager = _module
py_utils = _module
tqdm_utils = _module
version = _module
tests = _module
test_arrow_reader = _module
test_dataset_common = _module
test_py_utils = _module

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


import itertools


import logging


from collections.abc import Mapping


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import numpy as np


from functools import partial

