import sys
_module = sys.modules[__name__]
del sys
conf = _module
gnes = _module
base = _module
cli = _module
api = _module
parser = _module
client = _module
http = _module
stream = _module
component = _module
composer = _module
flask = _module
encoder = _module
audio = _module
mfcc = _module
vggish = _module
vggish_cores = _module
vggish_params = _module
vggish_postprocess = _module
vggish_slim = _module
image = _module
cvae = _module
cvae_cores = _module
model = _module
inception = _module
inception_cores = _module
inception_utils = _module
inception_v4 = _module
onnx = _module
torchvision = _module
numeric = _module
hash = _module
pca = _module
pooling = _module
pq = _module
quantizer = _module
standarder = _module
tf_pq = _module
vlad = _module
text = _module
bert = _module
char = _module
flair = _module
transformer = _module
w2v = _module
video = _module
incep_mixture = _module
mixture_core = _module
yt8m_feature_extractor = _module
yt8m_feature_extractor_cores = _module
inception_v3 = _module
yt8m_model = _module
flow = _module
helper = _module
indexer = _module
chunk = _module
annoy = _module
bindexer = _module
faiss = _module
hbindexer = _module
numpy = _module
doc = _module
dict = _module
filesys = _module
leveldb = _module
rocksdb = _module
preprocessor = _module
audio_vanilla = _module
vggish_example = _module
vggish_example_helper = _module
mel_features = _module
resize = _module
segmentation = _module
sliding_window = _module
io_utils = _module
ffmpeg = _module
gif = _module
webp = _module
split = _module
frame_select = _module
shot_detector = _module
video_decoder = _module
video_encoder = _module
proto = _module
gnes_pb2 = _module
gnes_pb2_grpc = _module
router = _module
map = _module
reduce = _module
score_fn = _module
normalize = _module
service = _module
frontend = _module
grpc = _module
uuid = _module
setup = _module
tests = _module
dummy2 = _module
dummy3 = _module
dummy_contrib = _module
fake_faiss = _module
fake_faiss2 = _module
dummy_pb2 = _module
dummy_pb2_grpc = _module
test_annoyindexer = _module
test_audio_preprocessor = _module
test_batching = _module
test_bindexer = _module
test_client_cli = _module
test_compose = _module
test_contrib_module = _module
test_dict_indexer = _module
test_dummy_transformer = _module
test_dump_loads = _module
test_encoder = _module
test_encoder_service = _module
test_euclidean_indexer = _module
test_ffmpeg_tools = _module
test_flair_encoder = _module
test_frame_selector = _module
test_gif = _module
test_gnes_flow = _module
test_grpc_service = _module
test_hash_encoder = _module
test_hash_indexer = _module
test_healthcheck = _module
test_image_encoder = _module
test_image_preprocessor = _module
test_indexer_service = _module
test_joint_indexer = _module
test_leveldbindexer = _module
test_leveldbindexerasync = _module
test_load_dump_pipeline = _module
test_mfcc_encoder = _module
test_mh_indexer = _module
test_onnx_image_encoder = _module
test_parser = _module
test_partition = _module
test_pca_encoder = _module
test_pipeline_train = _module
test_pipeline_train_ext = _module
test_pipelinepreprocess = _module
test_pooling_encoder = _module
test_preprocessor = _module
test_pretrain_encoder = _module
test_progressbar = _module
test_proto = _module
test_pytorch_transformers_encoder = _module
test_quantizer_encoder = _module
test_raw_bytes_send = _module
test_router = _module
test_score_fn = _module
test_service_mgr = _module
test_simple_indexer = _module
test_stream_grpc = _module
test_uuid = _module
test_vggish = _module
test_vggish_example = _module
test_video_decode_preprocessor = _module
test_video_encoder_preprocessor = _module
test_video_preprocessor = _module
test_video_shotdetect_preprocessor = _module
test_vlad = _module
test_w2v_encoder = _module
test_yaml = _module
test_yt8m_encoder = _module
test_yt8m_feature_extractor = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from typing import List


from typing import Callable


import numpy as np


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gnes_ai_gnes(_paritybench_base):
    pass
