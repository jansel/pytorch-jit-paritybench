import sys
_module = sys.modules[__name__]
del sys
download_pdc_data = _module
dense_correspondence = _module
correspondence_tools = _module
correspondence_augmentation = _module
correspondence_finder = _module
correspondence_plotter = _module
dataset = _module
dense_correspondence_dataset_masked = _module
scene_structure = _module
spartan_dataset_masked = _module
evaluation = _module
plotting = _module
utils = _module
loss_functions = _module
loss_composer = _module
network = _module
dense_correspondence_network = _module
test = _module
numpy_correspondence_finder = _module
training = _module
training = _module
training_script = _module
docker_build = _module
docker_run = _module
modules = _module
dense_correspondence_manipulation = _module
change_detection = _module
depthscanner = _module
mesh_processing = _module
tsdf_converter = _module
fusion = _module
fusion_reconstruction = _module
scripts = _module
batch_run_change_detection_pipeline = _module
compute_descriptor_images = _module
convertPlyToVtp = _module
convert_data_to_new_format = _module
convert_ply_to_vtp = _module
director_dev_app = _module
mesh_descriptor_color_app = _module
mesh_processing_app = _module
render_depth_images = _module
run_change_detection = _module
run_change_detection_pipeline = _module
tsdf_to_mesh = _module
simple_pixel_correspondence_labeler = _module
annotate_correspondences = _module
visualize_saved_correspondences = _module
constants = _module
director_utils = _module
image_utils = _module
segmentation = _module
transformations = _module
visualization = _module
live_heatmap_visualization = _module
start_notebook = _module

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


import numpy as np


import warnings


import logging


import torch


import torch.nn as nn


import copy


from torch.autograd import Variable


import torch.optim as optim


class ImageType:
    RGB = 0
    DEPTH = 1
    MASK = 2


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_RobotLocomotion_pytorch_dense_correspondence(_paritybench_base):
    pass
