import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
tests = _module
base = _module
test_base = _module
test_index = _module
bi_encoders = _module
qa = _module
tfhub = _module
test_lareqa_qa = _module
test_use_qa = _module
text_image = _module
test_clip = _module
conftest = _module
encoders = _module
audio = _module
pytorch = _module
test_fairseq = _module
test_speech_embedding = _module
test_trill = _module
test_vggish = _module
test_yamnet = _module
test_vi_audio2vec = _module
code = _module
test_code2vec = _module
face = _module
test_face2vec = _module
image = _module
fastai = _module
test_resnet = _module
test_bit = _module
test_inception = _module
test_inception_resnet = _module
test_mobilenet = _module
vectorai = _module
test_vi_image2vec = _module
text = _module
sentence_transformers = _module
test_sentence_transformers = _module
tf_transformers = _module
test_tf_transformers = _module
test_albert = _module
test_bert = _module
test_elmo = _module
test_labse = _module
test_use = _module
test_use_multi_transformer = _module
test_use_transformer = _module
torch_transformers = _module
test_legal_bert = _module
test_torch_longformers = _module
test_torch_transformers = _module
test_vi_text2vec = _module
test_autoencoder = _module
test_encode_chunk_documents = _module
test_encode_document = _module
test_import_utils = _module
test_model_to_dict = _module
test_utils = _module
download_badges = _module
upload_cards = _module
vectorhub = _module
auto_encoder = _module
distilroberta_qa = _module
lareqa_qa = _module
use_multi_qa = _module
use_qa = _module
dpr = _module
clip = _module
doc_utils = _module
wav2vec = _module
speech_embedding = _module
trill = _module
trill_distilled = _module
vggish = _module
yamnet = _module
vi_encoder = _module
transformers = _module
codebert = _module
tf = _module
face2vec = _module
color = _module
base = _module
resnet = _module
bit = _module
bit_medium = _module
inception_resnet = _module
inceptionv1 = _module
inceptionv2 = _module
inceptionv3 = _module
mobilenet = _module
mobilenetv2 = _module
resnetv2 = _module
sentence_auto_transformers = _module
tf_auto_transformers = _module
albert = _module
bert = _module
elmo = _module
experts_bert = _module
labse = _module
use = _module
use_lite = _module
use_multi = _module
use_multi_transformer = _module
use_transformer = _module
legal_bert = _module
torch_auto_transformers = _module
torch_longformers = _module
video = _module
sampler = _module
errors = _module
import_utils = _module
indexer = _module
models_dict = _module
options = _module
utils = _module

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


import warnings


from collections import defaultdict


from typing import List


from abc import abstractproperty


from typing import Union

