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


import time


import copy


from torch.autograd import Variable


import torch.optim as optim


def getDictFromYamlFilename(filename):
    """
    Read data from a YAML files
    """
    return yaml.load(open(filename), Loader=CLoader)


class CameraIntrinsics(object):
    """
    Useful class for wrapping camera intrinsics and loading them from a
    camera_info.yaml file
    """

    def __init__(self, cx, cy, fx, fy, width, height):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height
        self.K = self.get_camera_matrix()

    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 
            0, 1]])

    @staticmethod
    def from_yaml_file(filename):
        config = getDictFromYamlFilename(filename)
        fx = config['camera_matrix']['data'][0]
        cx = config['camera_matrix']['data'][2]
        fy = config['camera_matrix']['data'][4]
        cy = config['camera_matrix']['data'][5]
        width = config['image_width']
        height = config['image_height']
        return CameraIntrinsics(cx, cy, fx, fy, width, height)


class ImageType:
    RGB = 0
    DEPTH = 1
    MASK = 2


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_RobotLocomotion_pytorch_dense_correspondence(_paritybench_base):
    pass
