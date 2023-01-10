import sys
_module = sys.modules[__name__]
del sys
audioonly_64x1x1 = _module
slowonly_u54_kinetics = _module
timesformer_divST_16x12x1_kinetics = _module
audioonly_r101_64x1x1_200e_audio_feature = _module
tsn_r18_64x1x1_100e_kinetics200_audio_feature = _module
tsn_r50_64x1x1_100e_kinetics400_audio = _module
i3d_r50_video_32x2x1_256e_kinetics400_rgb = _module
slowonly_r50_8x8x1_256e_omnisource_rgb = _module
slowfast_r50_video_4x16x1_256e_kinetics400_rgb = _module
slowonly_nl_embedded_gaussian_r50_8x8x1_150e = _module
timesformer_divST_16x12x1_15e_kinetics400_rgb = _module
__int__ = _module
audio_filter = _module
class_distribution_clips = _module
class_distribution_time = _module
evaluate_acc_per_cls = _module
pose_feasibility = _module
print_layers = _module
augment_dataset = _module
build_file_list = _module
generate_dataset = _module
generate_dataset_pose = _module
pose_extraction = _module
demo_audio = _module
demo_skeleton = _module
long_video_demo_clips = _module
multimodial_demo = _module
visualize_heatmap_volume = _module
late_fusion = _module
misc = _module
record_experiment = _module
schedule_stuff = _module
top_tags = _module
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


import pandas as pd


import torch


import numpy as np


import time


from itertools import repeat

