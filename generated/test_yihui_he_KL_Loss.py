import sys
_module = sys.modules[__name__]
del sys
detectron = _module
core = _module
config = _module
rpn_generator = _module
test = _module
test_engine = _module
test_retinanet = _module
datasets = _module
cityscapes_json_dataset_evaluator = _module
coco_to_cityscapes_id = _module
dataset_catalog = _module
dummy_datasets = _module
json_dataset = _module
json_dataset_evaluator = _module
roidb = _module
task_evaluation = _module
voc_dataset_evaluator = _module
voc_eval = _module
FPN = _module
ResNet = _module
VGG16 = _module
VGG_CNN_M_1024 = _module
modeling = _module
detector = _module
fast_rcnn_heads = _module
generate_anchors = _module
gradient_clipping = _module
keypoint_rcnn_heads = _module
mask_rcnn_heads = _module
model_builder = _module
name_compat = _module
optimizer = _module
retinanet_heads = _module
rfcn_heads = _module
rpn_heads = _module
ops = _module
collect_and_distribute_fpn_rpn_proposals = _module
generate_proposal_labels = _module
generate_proposals = _module
roi_data = _module
data_utils = _module
fast_rcnn = _module
keypoint_rcnn = _module
loader = _module
mask_rcnn = _module
minibatch = _module
retinanet = _module
rpn = _module
data_loader_benchmark = _module
test_batch_permutation_op = _module
test_bbox_transform = _module
test_cfg = _module
test_loader = _module
test_restore_checkpoint = _module
test_smooth_l1_loss_op = _module
test_spatial_narrow_as_op = _module
test_zero_even_op = _module
utils = _module
blob = _module
boxes = _module
c2 = _module
collections = _module
colormap = _module
coordinator = _module
env = _module
gmm = _module
image = _module
io = _module
keypoints = _module
logging = _module
lr_policy = _module
model_convert_utils = _module
net = _module
py_cpu_nms = _module
segms = _module
subprocess = _module
timer = _module
train = _module
training_stats = _module
vis = _module
setup = _module
convert_cityscapes_to_coco = _module
convert_coco_model_to_cityscapes = _module
convert_pkl_to_pb = _module
convert_selective_search = _module
generate_testdev_from_test = _module
infer = _module
infer_simple = _module
pickle_caffe_blobs = _module
reval = _module
test_net = _module
train_net = _module
visualize_results = _module

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

