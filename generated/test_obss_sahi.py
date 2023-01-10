import sys
_module = sys.modules[__name__]
del sys
sahi = _module
annotation = _module
auto_model = _module
cli = _module
models = _module
base = _module
detectron2 = _module
huggingface = _module
mmdet = _module
torchvision = _module
yolov5 = _module
postprocess = _module
combine = _module
legacy = _module
utils = _module
predict = _module
prediction = _module
scripts = _module
coco2fiftyone = _module
coco2yolov5 = _module
coco_error_analysis = _module
coco_evaluation = _module
predict_fiftyone = _module
slice_coco = _module
slicing = _module
coco = _module
compatibility = _module
cv = _module
fiftyone = _module
file = _module
import_utils = _module
shapely = _module
versions = _module
run_code_style = _module
setup = _module
tests = _module
cascade_mask_rcnn_r50_fpn = _module
cascade_mask_rcnn_r50_fpn_1x_coco = _module
cascade_mask_rcnn_r50_fpn_1x_coco_v280 = _module
cascade_mask_rcnn_r50_fpn_v280 = _module
coco_instance = _module
default_runtime = _module
schedule_1x = _module
coco_detection = _module
retinanet_r50_fpn = _module
retinanet_r50_fpn_1x_coco = _module
retinanet_r50_fpn_1x_coco_v280 = _module
retinanet_r50_fpn_v280 = _module
yolox_tiny_8x8_300e_coco = _module
test_annotation = _module
test_autoslice = _module
test_cocoutils = _module
test_detectron2 = _module
test_fileutils = _module
test_highlevelapi = _module
test_huggingfacemodel = _module
test_mmdetectionmodel = _module
test_predict = _module
test_shapelyutils = _module
test_slicing = _module
test_torchvision = _module
test_yolov5model = _module

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


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import logging


import torch


from collections.abc import Sequence


import time


from typing import Sequence

