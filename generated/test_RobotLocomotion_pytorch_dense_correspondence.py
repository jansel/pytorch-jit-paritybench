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
labelfusion_masked = _module
scene_structure = _module
spartan_dataset_masked = _module
evaluation = _module
evaluation = _module
plotting = _module
utils = _module
loss_functions = _module
loss_composer = _module
pixelwise_contrastive_loss = _module
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
utils = _module
visualization = _module
live_heatmap_visualization = _module
start_notebook = _module

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


import numpy as np


import random


import torch


import numpy as numpy


from numpy.linalg import inv


import warnings


from torchvision import transforms


import torch.utils.data as data


import copy


import math


import logging


import matplotlib.pyplot as plt


import pandas as pd


import scipy.stats as ss


import itertools


from torch.autograd import Variable


import torch.nn as nn


import time


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
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

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


class DenseCorrespondenceDataset(data.Dataset):

    def __init__(self, debug=False):
        self.debug = debug
        self.mode = 'train'
        self.both_to_tensor = ComposeJoint([[transforms.ToTensor(), transforms.ToTensor()]])

    def __len__(self):
        return self.num_images_total

    def __getitem__(self, index):
        """
        The method through which the dataset is accessed for training.

        The index param is not currently used, and instead each dataset[i] is the result of
        a random sampling over:
        - random scene
        - random rgbd frame from that scene
        - random rgbd frame (different enough pose) from that scene
        - various randomization in the match generation and non-match generation procedure

        returns a large amount of variables, separated by commas.

        0th return arg: the type of data sampled (this can be used as a flag for different loss functions)
        0th rtype: string

        1st, 2nd return args: image_a_rgb, image_b_rgb
        1st, 2nd rtype: 3-dimensional torch.FloatTensor of shape (image_height, image_width, 3)

        3rd, 4th return args: matches_a, matches_b
        3rd, 4th rtype: 1-dimensional torch.LongTensor of shape (num_matches)

        5th, 6th return args: non_matches_a, non_matches_b
        5th, 6th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        Return values 3,4,5,6 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        """
        metadata = dict()
        scene_name = self.get_random_scene_name()
        metadata['scene_name'] = scene_name
        image_a_idx = self.get_random_image_index(scene_name)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_idx)
        metadata['image_a_idx'] = image_a_idx
        image_b_idx = self.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)
        metadata['image_b_idx'] = image_b_idx
        if image_b_idx is None:
            logging.info('no frame with sufficiently different pose found, returning')
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_idx)
        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose, image_b_depth_numpy, image_b_pose, num_attempts=self.num_matching_attempts, img_a_mask=np.asarray(image_a_mask))
        if uv_a is None:
            logging.info('no matches found, returning')
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        if self.debug:
            num_matches_to_plot = 10
            indexes_to_keep = (torch.rand(num_matches_to_plot) * len(uv_a[0])).floor().type(torch.LongTensor)
            uv_a = torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep)
            uv_b = torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep)
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)
        if not self.debug:
            [image_a_rgb], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb], uv_a)
            [image_b_rgb, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_mask], uv_b)
        else:
            [image_a_rgb, image_a_depth], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_depth], uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_depth, image_b_mask], uv_b)
            image_a_depth_numpy = np.asarray(image_a_depth)
            image_b_depth_numpy = np.asarray(image_b_depth)
        if index % 2:
            metadata['non_match_type'] = 'masked'
            logging.debug('masking non-matches')
            image_b_mask = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)
        else:
            metadata['non_match_type'] = 'non_masked'
            logging.debug('not masking non-matches')
            image_b_mask = None
        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[1]
        uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b, image_b_shape, num_non_matches_per_match=self.num_non_matches_per_match, img_b_mask=image_b_mask)
        if self.debug:
            uv_a_long = torch.t(uv_a[0].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1, 1), torch.t(uv_a[1].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1, 1)
            uv_b_non_matches_long = uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1)
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth_numpy, image_b_rgb, image_b_depth_numpy, uv_a, uv_b, show=False)
                correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth_numpy, image_b_rgb, image_b_depth_numpy, uv_a_long, uv_b_non_matches_long, use_previous_plot=(fig, axes), circ_color='r')
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)
        uv_a_long = torch.t(uv_a[0].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1, 1), torch.t(uv_a[1].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1, 1)
        uv_b_non_matches_long = uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1)
        matches_a = uv_a[1].long() * image_width + uv_a[0].long()
        matches_b = uv_b[1].long() * image_width + uv_b[0].long()
        non_matches_a = uv_a_long[1].long() * image_width + uv_a_long[0].long()
        non_matches_a = non_matches_a.squeeze(1)
        non_matches_b = uv_b_non_matches_long[1].long() * image_width + uv_b_non_matches_long[0].long()
        non_matches_b = non_matches_b.squeeze(1)
        return 'matches', image_a_rgb, image_b_rgb, matches_a, matches_b, non_matches_a, non_matches_b, metadata

    def return_empty_data(self, image_a_rgb, image_b_rgb, metadata=None):
        if metadata is None:
            metadata = dict()
        empty = DenseCorrespondenceDataset.empty_tensor()
        return -1, image_a_rgb, image_b_rgb, empty, empty, empty, empty, empty, empty, empty, empty, metadata

    @staticmethod
    def empty_tensor():
        """
        Makes a placeholder tensor
        :return:
        :rtype:
        """
        return torch.LongTensor([-1])

    @staticmethod
    def is_empty(tensor):
        """
        Tells if the tensor is the same as that created by empty_tensor()
        """
        return len(tensor) == 1 and tensor[0] == -1

    def get_rgbd_mask_pose(self, scene_name, img_idx):
        """
        Returns rgb image, depth image, mask and pose.
        :param scene_name:
        :type scene_name: str
        :param img_idx:
        :type img_idx: int
        :return: rgb, depth, mask, pose
        :rtype: PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, a 4x4 numpy array
        """
        rgb_file = self.get_image_filename(scene_name, img_idx, ImageType.RGB)
        rgb = self.get_rgb_image(rgb_file)
        depth_file = self.get_image_filename(scene_name, img_idx, ImageType.DEPTH)
        depth = self.get_depth_image(depth_file)
        mask_file = self.get_image_filename(scene_name, img_idx, ImageType.MASK)
        mask = self.get_mask_image(mask_file)
        pose = self.get_pose_from_scene_name_and_idx(scene_name, img_idx)
        return rgb, depth, mask, pose

    def get_random_rgbd_mask_pose(self):
        """
        Simple wrapper method for `get_rgbd_mask_pose`.
        Returns rgb image, depth image, mask and pose.
        :return: rgb, depth, mask, pose
        :rtype: PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, a 4x4 numpy array
        """
        scene_name = self.get_random_scene_name()
        img_idx = self.get_random_image_index(scene_name)
        return self.get_rgbd_mask_pose(scene_name, img_idx)

    def get_img_idx_with_different_pose(self, scene_name, pose_a, threshold=0.2, angle_threshold=20, num_attempts=10):
        """
        Try to get an image with a different pose to the one passed in. If one can't be found
        then return None
        :param scene_name:
        :type scene_name:
        :param pose_a:
        :type pose_a:
        :param threshold:
        :type threshold:
        :param num_attempts:
        :type num_attempts:
        :return: an index with a different-enough pose
        :rtype: int or None
        """
        counter = 0
        while counter < num_attempts:
            img_idx = self.get_random_image_index(scene_name)
            pose = self.get_pose_from_scene_name_and_idx(scene_name, img_idx)
            diff = utils.compute_distance_between_poses(pose_a, pose)
            angle_diff = utils.compute_angle_between_poses(pose_a, pose)
            if diff > threshold or angle_diff > angle_threshold:
                return img_idx
            counter += 1
        return None

    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """
        raise NotImplementedError('subclass must implement this method')

    @staticmethod
    def load_rgb_image(rgb_filename):
        """
        Returns PIL.Image.Image
        :param rgb_filename:
        :type rgb_filename:
        :return:
        :rtype: PIL.Image.Image
        """
        return Image.open(rgb_filename).convert('RGB')

    @staticmethod
    def load_mask_image(mask_filename):
        """
        Loads the mask image, returns a PIL.Image.Image
        :param mask_filename:
        :type mask_filename:
        :return:
        :rtype: PIL.Image.Image
        """
        return Image.open(mask_filename)

    def get_rgb_image(self, rgb_filename):
        """
        :param depth_filename: string of full path to depth image
        :return: PIL.Image.Image, in particular an 'RGB' PIL image
        """
        return Image.open(rgb_filename).convert('RGB')

    def get_rgb_image_from_scene_name_and_idx(self, scene_name, img_idx):
        """
        Returns an rgb image given a scene_name and image index
        :param scene_name:
        :param img_idx: str or int
        :return: PIL.Image.Image
        """
        img_filename = self.get_image_filename(scene_name, img_idx, ImageType.RGB)
        return self.get_rgb_image(img_filename)

    def get_depth_image(self, depth_filename):
        """
        :param depth_filename: string of full path to depth image
        :return: PIL.Image.Image
        """
        return Image.open(depth_filename)

    def get_depth_image_from_scene_name_and_idx(self, scene_name, img_idx):
        """
        Returns a depth image given a scene_name and image index
        :param scene_name:
        :param img_idx: str or int
        :return: PIL.Image.Image
        """
        img_filename = self.get_image_filename(scene_name, img_idx, ImageType.DEPTH)
        return self.get_depth_image(img_filename)

    def get_mask_image(self, mask_filename):
        """
        :param mask_filename: string of full path to mask image
        :return: PIL.Image.Image
        """
        return Image.open(mask_filename)

    def get_mask_image_from_scene_name_and_idx(self, scene_name, img_idx):
        """
        Returns a depth image given a scene_name and image index
        :param scene_name:
        :param img_idx: str or int
        :return: PIL.Image.Image
        """
        img_filename = self.get_image_filename(scene_name, img_idx, ImageType.MASK)
        return self.get_mask_image(img_filename)

    def get_image_filename(self, scene_name, img_index, image_type):
        raise NotImplementedError('Implement in superclass')

    def load_all_pose_data(self):
        """
        Efficiently pre-loads all pose data for the scenes. This is because when used as
        part of torch DataLoader in threaded way it behaves strangely
        :return:
        :rtype:
        """
        raise NotImplementedError('subclass must implement this method')

    def get_pose_from_scene_name_and_idx(self, scene_name, idx):
        """

        :param scene_name: str
        :param img_idx: int
        :return: 4 x 4 numpy array
        """
        raise NotImplementedError('subclass must implement this method')

    def quaternion_matrix(self, quaternion):
        _EPS = np.finfo(float).eps * 4.0
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0], [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0], [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0], [0.0, 0.0, 0.0, 1.0]])

    def elasticfusion_pose_to_homogeneous_transform(self, lf_pose):
        homogeneous_transform = self.quaternion_matrix([lf_pose[6], lf_pose[3], lf_pose[4], lf_pose[5]])
        homogeneous_transform[0, 3] = lf_pose[0]
        homogeneous_transform[1, 3] = lf_pose[1]
        homogeneous_transform[2, 3] = lf_pose[2]
        return homogeneous_transform

    def get_pose_list(self, scene_directory, pose_list_filename):
        posegraph_filename = os.path.join(scene_directory, pose_list_filename)
        with open(posegraph_filename) as f:
            content = f.readlines()
        pose_list = [x.strip().split() for x in content]
        return pose_list

    def get_full_path_for_scene(self, scene_name):
        raise NotImplementedError('subclass must implement this method')

    def get_random_scene_name(self):
        """
        Returns a random scene_name
        The result will depend on whether we are in test or train mode
        :return:
        :rtype:
        """
        return random.choice(self.scenes)

    def get_random_image_index(self, scene_name):
        """
        Returns a random image index from a given scene
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        raise NotImplementedError('subclass must implement this method')

    def get_random_scene_directory(self):
        scene_name = self.get_random_scene_name()
        scene_directory = self.get_full_path_for_scene(scene_name)
        return scene_directory

    def scene_generator(self, mode=None):
        """
        Returns an generator that traverses all the scenes
        :return:
        :rtype:
        """
        NotImplementedError('subclass must implement this method')

    def init_length(self):
        """
        Computes the total number of images and scenes in this dataset.
        Sets the result to the class variables self.num_images_total and self._num_scenes
        :return:
        :rtype:
        """
        self.num_images_total = 0
        self._num_scenes = 0
        for scene_name in self.scene_generator():
            scene_directory = self.get_full_path_for_scene(scene_name)
            rgb_images_regex = os.path.join(scene_directory, 'images/*_rgb.png')
            all_rgb_images_in_scene = glob.glob(rgb_images_regex)
            num_images_this_scene = len(all_rgb_images_in_scene)
            self.num_images_total += num_images_this_scene
            self._num_scenes += 1

    def load_from_config_yaml(self, key):
        this_file_path = os.path.dirname(__file__)
        yaml_path = os.path.join(this_file_path, 'config.yaml')
        with open(yaml_path, 'r') as stream:
            try:
                config_dict = yaml.load(stream)
            except yaml.YAMLError as exc:
                None
        username = getpass.getuser()
        relative_path = config_dict[username][key]
        full_path = os.path.join(os.environ['HOME'], relative_path)
        return full_path

    def use_all_available_scenes(self):
        self.scenes = [os.path.basename(x) for x in glob.glob(self.logs_root_path + '*')]

    def set_train_test_split_from_yaml(self, yaml_config_file_full_path):
        """
        Sets self.train and self.test attributes from config file
        :param yaml_config_file_full_path:
        :return:
        """
        if isinstance(yaml_config_file_full_path, str):
            with open(yaml_config_file_full_path, 'r') as stream:
                try:
                    config_dict = yaml.load(stream)
                except yaml.YAMLError as exc:
                    None
        else:
            config_dict = yaml_config_file_full_path
        self.train = config_dict['train']
        self.test = config_dict['test']
        self.set_train_mode()

    def set_parameters_from_training_config(self, training_config):
        """
        Some parameters that are really associated only with training, for example
        those associated with random sampling during the training process,
        should be passed in from a training.yaml config file.

        :param training_config: a dict() holding params
        """
        if self.mode == 'train' and training_config['training']['domain_randomize']:
            logging.info('enabling domain randomization')
            self.enable_domain_randomization()
        else:
            self.disable_domain_randomization()
        self.num_matching_attempts = int(training_config['training']['num_matching_attempts'])
        self.sample_matches_only_off_mask = training_config['training']['sample_matches_only_off_mask']
        self.num_non_matches_per_match = training_config['training']['num_non_matches_per_match']
        self.num_masked_non_matches_per_match = int(training_config['training']['fraction_masked_non_matches'] * self.num_non_matches_per_match)
        self.num_background_non_matches_per_match = int(training_config['training']['fraction_background_non_matches'] * self.num_non_matches_per_match)
        self.cross_scene_num_samples = training_config['training']['cross_scene_num_samples']
        self._use_image_b_mask_inv = training_config['training']['use_image_b_mask_inv']
        self._data_load_types = []
        self._data_load_type_probabilities = []
        p = training_config['training']['data_type_probabilities']['SINGLE_OBJECT_WITHIN_SCENE']
        if p > 0:
            None
            self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE)
            self._data_load_type_probabilities.append(p)
        p = training_config['training']['data_type_probabilities']['SINGLE_OBJECT_ACROSS_SCENE']
        if p > 0:
            None
            self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE)
            self._data_load_type_probabilities.append(p)
        p = training_config['training']['data_type_probabilities']['DIFFERENT_OBJECT']
        if p > 0:
            None
            self._data_load_types.append(SpartanDatasetDataType.DIFFERENT_OBJECT)
            self._data_load_type_probabilities.append(p)
        p = training_config['training']['data_type_probabilities']['MULTI_OBJECT']
        if p > 0:
            None
            self._data_load_types.append(SpartanDatasetDataType.MULTI_OBJECT)
            self._data_load_type_probabilities.append(p)
        p = training_config['training']['data_type_probabilities']['SYNTHETIC_MULTI_OBJECT']
        if p > 0:
            None
            self._data_load_types.append(SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT)
            self._data_load_type_probabilities.append(p)
        self._data_load_type_probabilities = np.array(self._data_load_type_probabilities)
        self._data_load_type_probabilities /= np.sum(self._data_load_type_probabilities)

    def set_train_mode(self):
        self.mode = 'train'

    def set_test_mode(self):
        self.mode = 'test'

    def enable_domain_randomization(self):
        """
        Turns on background domain randomization
        :return:
        :rtype:
        """
        self._domain_randomize = True

    def disable_domain_randomization(self):
        """
        Turns off background domain randomization
        :return:
        :rtype:
        """
        self._domain_randomize = False

    def compute_image_mean_and_std_dev(self, num_image_samples=10):
        """
        Computes the image_mean and std_dev using the specified number of samples.
        Returns two torch.FloatTensor objects, each of size [3]
        :param num_image_samples:
        :type num_image_samples:
        :return:
        :rtype:
        """

        def get_random_image():
            scene_name = self.get_random_scene_name()
            img_idx = self.get_random_image_index(scene_name)
            img_filename = self.get_image_filename(scene_name, img_idx, ImageType.RGB)
            img = self.get_rgb_image(img_filename)
            return img

        def get_image_mean(img_tensor):
            """

            :param img_tensor: torch.FloatTensor with shape [3, 480, 640]
            :type img_tensor:
            :return: torch.FloatTensor with shape [3]
            :rtype:
            """
            img_mean = torch.mean(img_tensor, 1)
            img_mean = torch.mean(img_mean, 1)
            return img_mean

        def get_image_std_dev(img_tensor):
            shape = img_tensor.shape
            img_height = shape[1]
            img_width = shape[2]
            v = img_tensor.view(-1, img_height * img_width)
            std_dev = torch.std(v, 1)
            return std_dev
        to_tensor = transforms.ToTensor()
        img_mean_sum = None
        img_mean_sum = torch.zeros([3])
        for i in range(0, num_image_samples):
            img = get_random_image()
            img_tensor = to_tensor(img)
            single_img_mean = get_image_mean(img_tensor)
            if img_mean_sum is None:
                img_mean_sum = torch.zeros_like(single_img_mean)
            img_mean_sum = img_mean_sum + single_img_mean
        std_dev_sum = None
        for i in range(0, num_image_samples):
            img = get_random_image()
            img_tensor = to_tensor(img)
            single_std_dev = get_image_std_dev(img_tensor)
            if std_dev_sum is None:
                std_dev_sum = torch.zeros_like(single_std_dev)
            std_dev_sum += single_std_dev
        img_mean = 1.0 / num_image_samples * img_mean_sum
        std_dev = 1.0 / num_image_samples * std_dev_sum
        return img_mean, std_dev

    @property
    def test_scene_directories(self):
        """
        Get the list of testing scene directories
        :return: list of strings
        """
        return self.test

    @property
    def train_scene_directories(self):
        """
        Get the list of training scene directories
        :return: list of strings
        """
        return self.train
    """
    Debug
    """

    def debug_show_data(self, image_a_rgb, image_a_depth, image_a_pose, image_b_rgb, image_b_depth, image_b_pose):
        plt.imshow(image_a_rgb)
        plt.show()
        plt.imshow(image_a_depth)
        plt.show()
        None
        plt.imshow(image_b_rgb)
        plt.show()
        plt.imshow(image_b_depth)
        plt.show()
        None


class SceneStructure(object):

    def __init__(self, processed_folder_dir):
        self._processed_folder_dir = processed_folder_dir

    @property
    def fusion_reconstruction_file(self):
        """
        The filepath for the fusion reconstruction
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'fusion_mesh.ply')

    @property
    def foreground_fusion_reconstruction_file(self):
        """
        The filepath for the fusion reconstruction corresponding only to the
        foreground. Note, this may not exist if you haven't done some processing
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'fusion_mesh_foreground.ply')

    @property
    def camera_info_file(self):
        """
        Full filepath for yaml file containing camera intrinsics parameters
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'images', 'camera_info.yaml')

    @property
    def camera_pose_file(self):
        """
        Full filepath for yaml file containing the camera poses
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'images', 'pose_data.yaml')

    @property
    def rendered_images_dir(self):
        return os.path.join(self._processed_folder_dir, 'rendered_images')

    @property
    def images_dir(self):
        return os.path.join(self._processed_folder_dir, 'images')

    @property
    def metadata_file(self):
        return os.path.join(self.images_dir, 'metadata.yaml')

    def mesh_descriptors_dir(self, network_name):
        """
        Directory where we store descriptors corresponding to a particular network
        :param network_name:
        :type network_name:
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'mesh_descriptors', network_name)

    def mesh_cells_image_filename(self, img_idx):
        """
        Returns the full filename for the cell labels image
        :param img_idx:
        :type img_idx:
        :return:
        :rtype:
        """
        filename = utils.getPaddedString(img_idx) + '_mesh_cells.png'
        return os.path.join(self.rendered_images_dir, filename)

    def mesh_descriptors_filename(self, network_name, img_idx):
        """
        Returns the full filename for the .npz file that contains two arrays

        .npz reference https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savez.html#numpy.savez

        D = descriptor dimension

        - cell_ids: np.array of size N, dtype=np.int64
        - cell_descriptors: np.array with np.shape = [N,D dtype = np.float64
        -
        :param img_idx:
        :type img_idx:
        :return:
        :rtype:
        """
        filename = utils.getPaddedString(img_idx) + '_mesh_descriptors.npz'
        return os.path.join(self.mesh_descriptors_dir(network_name), filename)

    def mesh_descriptor_statistics_filename(self, network_name):
        """
        Filename containing mesh descriptor statistics

        N = number of cells for which we have descriptor information

        - cell_valid: np.array of size N, dtype=np.int64. Value is the 
        index of that cell in the ply file description
        - cell_descriptor_mean: np.array with np.shape = [N,D] dtype = np.float64
        - cell_location: Location of the cell in object frame np.array with
                        np.shape = [N,3], dtype=np.float64

        :param: network_name
        :return: filename
        :rtype: str
        """
        return os.path.join(self.mesh_descriptors_dir(network_name), 'mesh_descriptor_stats.npz')

    @staticmethod
    def descriptor_image_filename(img_idx):
        filename = utils.getPaddedString(img_idx) + '_descriptor_image.npy'
        return filename


class SpartanDatasetDataType:
    SINGLE_OBJECT_WITHIN_SCENE = 0
    SINGLE_OBJECT_ACROSS_SCENE = 1
    DIFFERENT_OBJECT = 2
    MULTI_OBJECT = 3
    SYNTHETIC_MULTI_OBJECT = 4


class SpartanDataset(DenseCorrespondenceDataset):
    PADDED_STRING_WIDTH = 6

    def __init__(self, debug=False, mode='train', config=None, config_expanded=None, verbose=False):
        """
        :param config: This is for creating a dataset from a composite dataset config file.
            This is of the form:

                logs_root_path: logs_proto # path relative to utils.get_data_dir()

                single_object_scenes_config_files:
                - caterpillar_17_scenes.yaml
                - baymax.yaml

                multi_object_scenes_config_files:
                - multi_object.yaml

        :type config: dict()

        :param config_expanded: When a config is read, it is parsed into an expanded form
            which is actually used as self._config.  See the function _setup_scene_data()
            for how this is done.  We want to save this expanded config to disk as it contains
            all config information.  If loading a previously-used dataset configuration, we want
            to pass in the config_expanded.
        :type config_expanded: dict()
        """
        DenseCorrespondenceDataset.__init__(self, debug=debug)
        if self.debug:
            self._domain_randomize = False
            self.num_masked_non_matches_per_match = 5
            self.num_background_non_matches_per_match = 5
            self.cross_scene_num_samples = 1000
            self._use_image_b_mask_inv = True
            self.num_matching_attempts = 10000
            self.sample_matches_only_off_mask = True
        self._verbose = verbose
        if config is not None:
            self._setup_scene_data(config)
        elif config_expanded is not None:
            self._parse_expanded_config(config_expanded)
        else:
            raise ValueError('You need to give me either a config or config_expanded')
        self._pose_data = dict()
        self._initialize_rgb_image_to_tensor()
        if mode == 'test':
            self.set_test_mode()
        elif mode == 'train':
            self.set_train_mode()
        else:
            raise ValueError('mode should be one of [test, train]')
        self.init_length()
        None
        None
        None
        None

    def __getitem__(self, index):
        """
        This overloads __getitem__ and is what is actually returned
        using a torch dataloader.

        This small function randomly chooses one of our different
        img pair types, then returns that type of data.
        """
        data_load_type = self._get_data_load_type()
        if data_load_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
            if self._verbose:
                None
            return self.get_single_object_within_scene_data()
        if data_load_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE:
            if self._verbose:
                None
            return self.get_single_object_across_scene_data()
        if data_load_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
            if self._verbose:
                None
            return self.get_different_object_data()
        if data_load_type == SpartanDatasetDataType.MULTI_OBJECT:
            if self._verbose:
                None
            return self.get_multi_object_within_scene_data()
        if data_load_type == SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT:
            if self._verbose:
                None
            return self.get_synthetic_multi_object_within_scene_data()

    def _setup_scene_data(self, config):
        """
        Initializes the data for all the different types of scenes

        Creates two class attributes

        self._single_object_scene_dict

        Each entry of self._single_object_scene_dict is a dict with keys {"test", "train"}. The
        values are lists of scenes

        self._single_object_scene_dict has (key, value) = (object_id, scene config for that object)

        self._multi_object_scene_dict has (key, value) = ("train"/"test", list of scenes)

        Note that the scenes have absolute paths here
        """
        self.logs_root_path = utils.convert_data_relative_path_to_absolute_path(config['logs_root_path'], assert_path_exists=True)
        self._single_object_scene_dict = dict()
        prefix = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset')
        for config_file in config['single_object_scenes_config_files']:
            config_file = os.path.join(prefix, 'single_object', config_file)
            single_object_scene_config = utils.getDictFromYamlFilename(config_file)
            object_id = single_object_scene_config['object_id']
            if object_id not in self._single_object_scene_dict:
                self._single_object_scene_dict[object_id] = single_object_scene_config
            else:
                existing_config = self._single_object_scene_dict[object_id]
                merged_config = SpartanDataset.merge_single_object_configs([existing_config, single_object_scene_config])
                self._single_object_scene_dict[object_id] = merged_config
        self._multi_object_scene_dict = {'train': [], 'test': [], 'evaluation_labeled_data_path': []}
        for config_file in config['multi_object_scenes_config_files']:
            config_file = os.path.join(prefix, 'multi_object', config_file)
            multi_object_scene_config = utils.getDictFromYamlFilename(config_file)
            for key, val in self._multi_object_scene_dict.items():
                for item in multi_object_scene_config[key]:
                    val.append(item)
        self._config = dict()
        self._config['logs_root_path'] = config['logs_root_path']
        self._config['single_object'] = self._single_object_scene_dict
        self._config['multi_object'] = self._multi_object_scene_dict
        self._setup_data_load_types()

    def _parse_expanded_config(self, config_expanded):
        """
        If we have previously saved to disk a dict with:
        "single_object", and
        "multi_object" keys,
        then we want to recreate a config from these.
        """
        self._config = config_expanded
        self._single_object_scene_dict = self._config['single_object']
        self._multi_object_scene_dict = self._config['multi_object']
        self.logs_root_path = utils.convert_data_relative_path_to_absolute_path(self._config['logs_root_path'], assert_path_exists=True)

    def _setup_data_load_types(self):
        self._data_load_types = []
        self._data_load_type_probabilities = []
        if self.debug:
            self._data_load_types.append(SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT)
            self._data_load_type_probabilities.append(1)

    def _get_data_load_type(self):
        """
        Gets a random data load type from the allowable types
        :return: SpartanDatasetDataType
        :rtype:
        """
        return np.random.choice(self._data_load_types, 1, p=self._data_load_type_probabilities)[0]

    def scene_generator(self, mode=None):
        """
        Returns a generator that traverses all the scenes
        :return:
        :rtype:
        """
        if mode is None:
            mode = self.mode
        for object_id, single_object_scene_dict in self._single_object_scene_dict.items():
            for scene_name in single_object_scene_dict[mode]:
                yield scene_name
        for scene_name in self._multi_object_scene_dict[mode]:
            yield scene_name

    def get_scene_list(self, mode=None):
        """
        Returns a list of all scenes in this dataset
        :return:
        :rtype:
        """
        scene_generator = self.scene_generator(mode=mode)
        scene_list = []
        for scene_name in scene_generator:
            scene_list.append(scene_name)
        return scene_list

    def get_list_of_objects(self):
        """
        Returns a list of object ids
        :return: list of object_ids
        :rtype:
        """
        return list(self._single_object_scene_dict.keys())

    def get_scene_list_for_object(self, object_id, mode=None):
        """
        Returns list of scenes for a given object. Return test/train
        scenes depending on the mode
        :param object_id:
        :type object_id: string
        :param mode: either "test" or "train"
        :type mode:
        :return:
        :rtype:
        """
        if mode is None:
            mode = self.mode
        return copy.copy(self._single_object_scene_dict[object_id][mode])

    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])

    def get_full_path_for_scene(self, scene_name):
        """
        Returns the full path to the processed logs folder
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        return os.path.join(self.logs_root_path, scene_name, 'processed')

    def load_all_pose_data(self):
        """
        Efficiently pre-loads all pose data for the scenes. This is because when used as
        part of torch DataLoader in threaded way it behaves strangely
        :return:
        :rtype:
        """
        for scene_name in self.scene_generator():
            self.get_pose_data(scene_name)

    def get_pose_data(self, scene_name):
        """
        Checks if have not already loaded the pose_data.yaml for this scene,
        if haven't then loads it. Then returns the dict of the pose_data.yaml.
        :type scene_name: str
        :return: a dict() of the pose_data.yaml for the scene.
        :rtype: dict()
        """
        if scene_name not in self._pose_data:
            logging.info('Loading pose data for scene %s' % scene_name)
            pose_data_filename = os.path.join(self.get_full_path_for_scene(scene_name), 'images', 'pose_data.yaml')
            self._pose_data[scene_name] = utils.getDictFromYamlFilename(pose_data_filename)
        return self._pose_data[scene_name]

    def get_pose_from_scene_name_and_idx(self, scene_name, idx):
        """
        :param scene_name: str
        :param img_idx: int
        :return: 4 x 4 numpy array
        """
        idx = int(idx)
        scene_pose_data = self.get_pose_data(scene_name)
        pose_data = scene_pose_data[idx]['camera_to_world']
        return utils.homogenous_transform_from_dict(pose_data)

    def get_image_filename(self, scene_name, img_index, image_type):
        """
        Get the image filename for that scene and image index
        :param scene_name: str
        :param img_index: str or int
        :param image_type: ImageType
        :return:
        """
        scene_directory = self.get_full_path_for_scene(scene_name)
        if image_type == ImageType.RGB:
            images_dir = os.path.join(scene_directory, 'images')
            file_extension = '_rgb.png'
        elif image_type == ImageType.DEPTH:
            images_dir = os.path.join(scene_directory, 'rendered_images')
            file_extension = '_depth.png'
        elif image_type == ImageType.MASK:
            images_dir = os.path.join(scene_directory, 'image_masks')
            file_extension = '_mask.png'
        else:
            raise ValueError('unsupported image type')
        if isinstance(img_index, int):
            img_index = utils.getPaddedString(img_index, width=SpartanDataset.PADDED_STRING_WIDTH)
        scene_directory = self.get_full_path_for_scene(scene_name)
        if not os.path.isdir(scene_directory):
            raise ValueError("scene_name = %s doesn't exist" % scene_name)
        return os.path.join(images_dir, img_index + file_extension)

    def get_camera_intrinsics(self, scene_name=None):
        """
        Returns the camera matrix for that scene
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        if scene_name is None:
            scene_directory = self.get_random_scene_directory()
        else:
            scene_directory = os.path.join(self.logs_root_path, scene_name)
        camera_info_file = os.path.join(scene_directory, 'processed', 'images', 'camera_info.yaml')
        return CameraIntrinsics.from_yaml_file(camera_info_file)

    def get_random_image_index(self, scene_name):
        """
        Returns a random image index from a given scene
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        pose_data = self.get_pose_data(scene_name)
        image_idxs = list(pose_data.keys())
        random.choice(image_idxs)
        random_idx = random.choice(image_idxs)
        return random_idx

    def get_random_object_id(self):
        """
        Returns a random object_id
        :return:
        :rtype:
        """
        object_id_list = list(self._single_object_scene_dict.keys())
        return random.choice(object_id_list)

    def get_random_object_id_and_int(self):
        """
        Returns a random object_id (a string) and its "int" (i.e. numerical unique id)
        :return:
        :rtype:
        """
        object_id_list = list(self._single_object_scene_dict.keys())
        random_object_id = random.choice(object_id_list)
        object_id_int = sorted(self._single_object_scene_dict.keys()).index(random_object_id)
        return random_object_id, object_id_int

    def get_random_single_object_scene_name(self, object_id):
        """
        Returns a random scene name for that object
        :param object_id: str
        :type object_id:
        :return: str
        :rtype:
        """
        scene_list = self._single_object_scene_dict[object_id][self.mode]
        return random.choice(scene_list)

    def get_different_scene_for_object(self, object_id, scene_name):
        """
        Return a different scene name
        :param object_id:
        :type object_id:
        :return:
        :rtype:
        """
        scene_list = self._single_object_scene_dict[object_id][self.mode]
        if len(scene_list) == 1:
            raise ValueError("There is only one scene of this object, can't sample a different one")
        idx_array = np.arange(0, len(scene_list))
        rand_idxs = np.random.choice(idx_array, 2, replace=False)
        for idx in rand_idxs:
            scene_name_b = scene_list[idx]
            if scene_name != scene_name_b:
                return scene_name_b
        raise ValueError('It (should) be impossible to get here!!!!')

    def get_two_different_object_ids(self):
        """
        Returns two different random object ids
        :return: two object ids
        :rtype: two strings separated by commas
        """
        object_id_list = list(self._single_object_scene_dict.keys())
        if len(object_id_list) == 1:
            raise ValueError("There is only one object, can't sample a different one")
        idx_array = np.arange(0, len(object_id_list))
        rand_idxs = np.random.choice(idx_array, 2, replace=False)
        object_1_id = object_id_list[rand_idxs[0]]
        object_2_id = object_id_list[rand_idxs[1]]
        assert object_1_id != object_2_id
        return object_1_id, object_2_id

    def get_random_multi_object_scene_name(self):
        """
        Returns a random multi object scene name
        :return:
        :rtype:
        """
        return random.choice(self._multi_object_scene_dict[self.mode])

    def get_number_of_unique_single_objects(self):
        """
        Returns the number of unique objects in this dataset with single object scenes
        :return:
        :rtype:
        """
        return len(list(self._single_object_scene_dict.keys()))

    def has_multi_object_scenes(self):
        """
        Returns true if there are multi-object scenes in this datase
        :return:
        :rtype:
        """
        return len(self._multi_object_scene_dict['train']) > 0

    def get_random_scene_name(self):
        """
        Gets a random scene name across both single and multi object
        """
        types = []
        if self.has_multi_object_scenes():
            for _ in range(len(self._multi_object_scene_dict[self.mode])):
                types.append('multi')
        if self.get_number_of_unique_single_objects() > 0:
            for _ in range(self.get_number_of_unique_single_objects()):
                types.append('single')
        if len(types) == 0:
            raise ValueError("I don't think you have any scenes?")
        scene_type = random.choice(types)
        if scene_type == 'multi':
            return self.get_random_multi_object_scene_name()
        if scene_type == 'single':
            object_id = self.get_random_object_id()
            return self.get_random_single_object_scene_name(object_id)

    def get_single_object_within_scene_data(self):
        """
        Simple wrapper around get_within_scene_data(), for the single object case
        """
        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError('There are no single object scenes in this dataset')
        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)
        metadata = dict()
        metadata['object_id'] = object_id
        metadata['object_id_int'] = sorted(self._single_object_scene_dict.keys()).index(object_id)
        metadata['scene_name'] = scene_name
        metadata['type'] = SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE
        return self.get_within_scene_data(scene_name, metadata)

    def get_multi_object_within_scene_data(self):
        """
        Simple wrapper around get_within_scene_data(), for the multi object case
        """
        if not self.has_multi_object_scenes():
            raise ValueError('There are no multi object scenes in this dataset')
        scene_name = self.get_random_multi_object_scene_name()
        metadata = dict()
        metadata['scene_name'] = scene_name
        metadata['type'] = SpartanDatasetDataType.MULTI_OBJECT
        return self.get_within_scene_data(scene_name, metadata)

    def get_within_scene_data(self, scene_name, metadata, for_synthetic_multi_object=False):
        """
        The method through which the dataset is accessed for training.

        Each call is is the result of
        a random sampling over:
        - random scene
        - random rgbd frame from that scene
        - random rgbd frame (different enough pose) from that scene
        - various randomization in the match generation and non-match generation procedure

        returns a large amount of variables, separated by commas.

        0th return arg: the type of data sampled (this can be used as a flag for different loss functions)
        0th rtype: string

        1st, 2nd return args: image_a_rgb, image_b_rgb
        1st, 2nd rtype: 3-dimensional torch.FloatTensor of shape (image_height, image_width, 3)

        3rd, 4th return args: matches_a, matches_b
        3rd, 4th rtype: 1-dimensional torch.LongTensor of shape (num_matches)

        5th, 6th return args: masked_non_matches_a, masked_non_matches_b
        5th, 6th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        9th, 10th return args: blind_non_matches_a, blind_non_matches_b
        9th, 10th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        11th return arg: metadata useful for plotting, and-or other flags for loss functions
        11th rtype: dict

        Return values 3,4,5,6,7,8,9,10 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        If no datapoints were found for some type of match or non-match then we return
        our "special" empty tensor. Note that due to the way the pytorch data loader
        functions you cannot return an empty tensor like torch.FloatTensor([]). So we
        return SpartanDataset.empty_tensor()

        """
        SD = SpartanDataset
        image_a_idx = self.get_random_image_index(scene_name)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_idx)
        metadata['image_a_idx'] = image_a_idx
        image_b_idx = self.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)
        metadata['image_b_idx'] = image_b_idx
        if image_b_idx is None:
            logging.info('no frame with sufficiently different pose found, returning')
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_idx)
        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)
        if self.sample_matches_only_off_mask:
            correspondence_mask = np.asarray(image_a_mask)
        else:
            correspondence_mask = None
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose, image_b_depth_numpy, image_b_pose, img_a_mask=correspondence_mask, num_attempts=self.num_matching_attempts)
        if for_synthetic_multi_object:
            return image_a_rgb, image_b_rgb, image_a_depth, image_b_depth, image_a_mask, image_b_mask, uv_a, uv_b
        if uv_a is None:
            logging.info('no matches found, returning')
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)
        if not self.debug:
            [image_a_rgb, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_mask], uv_a)
            [image_b_rgb, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_mask], uv_b)
        else:
            [image_a_rgb, image_a_depth, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_depth, image_a_mask], uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_depth, image_b_mask], uv_b)
        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)
        image_b_mask_torch = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)
        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]
        uv_b_masked_non_matches = correspondence_finder.create_non_correspondences(uv_b, image_b_shape, num_non_matches_per_match=self.num_masked_non_matches_per_match, img_b_mask=image_b_mask_torch)
        if self._use_image_b_mask_inv:
            image_b_mask_inv = 1 - image_b_mask_torch
        else:
            image_b_mask_inv = None
        uv_b_background_non_matches = correspondence_finder.create_non_correspondences(uv_b, image_b_shape, num_non_matches_per_match=self.num_background_non_matches_per_match, img_b_mask=image_b_mask_inv)
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)
        matches_a = SD.flatten_uv_tensor(uv_a, image_width)
        matches_b = SD.flatten_uv_tensor(uv_b, image_width)
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(uv_a, uv_b_masked_non_matches, self.num_masked_non_matches_per_match)
        masked_non_matches_a = SD.flatten_uv_tensor(uv_a_masked_long, image_width).squeeze(1)
        masked_non_matches_b = SD.flatten_uv_tensor(uv_b_masked_non_matches_long, image_width).squeeze(1)
        uv_a_background_long, uv_b_background_non_matches_long = self.create_non_matches(uv_a, uv_b_background_non_matches, self.num_background_non_matches_per_match)
        background_non_matches_a = SD.flatten_uv_tensor(uv_a_background_long, image_width).squeeze(1)
        background_non_matches_b = SD.flatten_uv_tensor(uv_b_background_non_matches_long, image_width).squeeze(1)
        matches_a_mask = SD.mask_image_from_uv_flat_tensor(matches_a, image_width, image_height)
        image_a_mask_torch = torch.from_numpy(np.asarray(image_a_mask)).long()
        mask_a_flat = image_a_mask_torch.view(-1, 1).squeeze(1)
        blind_non_matches_a = (mask_a_flat - matches_a_mask).nonzero()
        no_blind_matches_found = False
        if len(blind_non_matches_a) == 0:
            no_blind_matches_found = True
        else:
            blind_non_matches_a = blind_non_matches_a.squeeze(1)
            num_blind_samples = blind_non_matches_a.size()[0]
            if num_blind_samples > 0:
                blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(image_b_mask_torch, num_blind_samples)
                if blind_uv_b[0] is None:
                    no_blind_matches_found = True
                elif len(blind_uv_b[0]) == 0:
                    no_blind_matches_found = True
                else:
                    blind_non_matches_b = utils.uv_to_flattened_pixel_locations(blind_uv_b, image_width)
                    if len(blind_non_matches_b) == 0:
                        no_blind_matches_found = True
            else:
                no_blind_matches_found = True
        if no_blind_matches_found:
            blind_non_matches_a = blind_non_matches_b = SD.empty_tensor()
        if self.debug:
            num_matches_to_plot = 10
            plot_uv_a, plot_uv_b = SD.subsample_tuple_pair(uv_a, uv_b, num_samples=num_matches_to_plot)
            plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long = SD.subsample_tuple_pair(uv_a_masked_long, uv_b_masked_non_matches_long, num_samples=num_matches_to_plot * 3)
            plot_uv_a_background_long, plot_uv_b_background_non_matches_long = SD.subsample_tuple_pair(uv_a_background_long, uv_b_background_non_matches_long, num_samples=num_matches_to_plot * 3)
            blind_uv_a = utils.flattened_pixel_locations_to_u_v(blind_non_matches_a, image_width)
            plot_blind_uv_a, plot_blind_uv_b = SD.subsample_tuple_pair(blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot * 10)
        if self.debug:
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL, image_b_depth_numpy, plot_uv_a, plot_uv_b, show=False)
                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL, image_b_depth_numpy, plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long, use_previous_plot=(fig, axes), circ_color='r')
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL, image_b_depth_numpy, plot_uv_a, plot_uv_b, show=False)
                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL, image_b_depth_numpy, plot_uv_a_background_long, plot_uv_b_background_non_matches_long, use_previous_plot=(fig, axes), circ_color='b')
                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL, image_b_depth_numpy, plot_blind_uv_a, plot_blind_uv_b, circ_color='k', show=True)
                import matplotlib.pyplot as plt
                plt.imshow(np.asarray(image_a_mask))
                plt.title('Mask of img a object pixels')
                plt.show()
                plt.imshow(np.asarray(image_a_mask) - 1)
                plt.title('Mask of img a background')
                plt.show()
                temp = matches_a_mask.view(image_height, -1)
                plt.imshow(temp)
                plt.title('Mask of img a object pixels for which there was a match')
                plt.show()
                temp2 = (mask_a_flat - matches_a_mask).view(image_height, -1)
                plt.imshow(temp2)
                plt.title('Mask of img a object pixels for which there was NO match')
                plt.show()
        return metadata['type'], image_a_rgb, image_b_rgb, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, metadata

    def create_non_matches(self, uv_a, uv_b_non_matches, multiplier):
        """
        Simple wrapper for repeated code
        :param uv_a:
        :type uv_a:
        :param uv_b_non_matches:
        :type uv_b_non_matches:
        :param multiplier:
        :type multiplier:
        :return:
        :rtype:
        """
        uv_a_long = torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1), torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1)
        uv_b_non_matches_long = uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1)
        return uv_a_long, uv_b_non_matches_long

    def get_single_object_across_scene_data(self):
        """
        Simple wrapper for get_across_scene_data(), for the single object case
        """
        metadata = dict()
        object_id = self.get_random_object_id()
        scene_name_a = self.get_random_single_object_scene_name(object_id)
        scene_name_b = self.get_different_scene_for_object(object_id, scene_name_a)
        metadata['object_id'] = object_id
        metadata['scene_name_a'] = scene_name_a
        metadata['scene_name_b'] = scene_name_b
        metadata['type'] = SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE
        return self.get_across_scene_data(scene_name_a, scene_name_b, metadata)

    def get_different_object_data(self):
        """
        Simple wrapper for get_across_scene_data(), for the different object case
        """
        metadata = dict()
        object_id_a, object_id_b = self.get_two_different_object_ids()
        scene_name_a = self.get_random_single_object_scene_name(object_id_a)
        scene_name_b = self.get_random_single_object_scene_name(object_id_b)
        metadata['object_id_a'] = object_id_a
        metadata['scene_name_a'] = scene_name_a
        metadata['object_id_b'] = object_id_b
        metadata['scene_name_b'] = scene_name_b
        metadata['type'] = SpartanDatasetDataType.DIFFERENT_OBJECT
        return self.get_across_scene_data(scene_name_a, scene_name_b, metadata)

    def get_synthetic_multi_object_within_scene_data(self):
        """
        Synthetic case
        """
        object_id_a, object_id_b = self.get_two_different_object_ids()
        scene_name_a = self.get_random_single_object_scene_name(object_id_a)
        scene_name_b = self.get_random_single_object_scene_name(object_id_b)
        metadata = dict()
        metadata['object_id_a'] = object_id_a
        metadata['scene_name_a'] = scene_name_a
        metadata['object_id_b'] = object_id_b
        metadata['scene_name_b'] = scene_name_b
        metadata['type'] = SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT
        image_a1_rgb, image_a2_rgb, image_a1_depth, image_a2_depth, image_a1_mask, image_a2_mask, uv_a1, uv_a2 = self.get_within_scene_data(scene_name_a, metadata, for_synthetic_multi_object=True)
        if uv_a1 is None:
            logging.info('no matches found, returning')
            image_a1_rgb_tensor = self.rgb_image_to_tensor(image_a1_rgb)
            return self.return_empty_data(image_a1_rgb_tensor, image_a1_rgb_tensor)
        image_b1_rgb, image_b2_rgb, image_b1_depth, image_b2_depth, image_b1_mask, image_b2_mask, uv_b1, uv_b2 = self.get_within_scene_data(scene_name_b, metadata, for_synthetic_multi_object=True)
        if uv_b1 is None:
            logging.info('no matches found, returning')
            image_b1_rgb_tensor = self.rgb_image_to_tensor(image_b1_rgb)
            return self.return_empty_data(image_b1_rgb_tensor, image_b1_rgb_tensor)
        uv_a1 = uv_a1[0].long(), uv_a1[1].long()
        uv_a2 = uv_a2[0].long(), uv_a2[1].long()
        uv_b1 = uv_b1[0].long(), uv_b1[1].long()
        uv_b2 = uv_b2[0].long(), uv_b2[1].long()
        matches_pair_a = uv_a1, uv_a2
        matches_pair_b = uv_b1, uv_b2
        merged_rgb_1, merged_mask_1, uv_a1, uv_a2, uv_b1, uv_b2 = correspondence_augmentation.merge_images_with_occlusions(image_a1_rgb, image_b1_rgb, image_a1_mask, image_b1_mask, matches_pair_a, matches_pair_b)
        if uv_a1 is None or uv_a2 is None or uv_b1 is None or uv_b2 is None:
            logging.info('something got fully occluded, returning')
            image_b1_rgb_tensor = self.rgb_image_to_tensor(image_b1_rgb)
            return self.return_empty_data(image_b1_rgb_tensor, image_b1_rgb_tensor)
        matches_pair_a = uv_a2, uv_a1
        matches_pair_b = uv_b2, uv_b1
        merged_rgb_2, merged_mask_2, uv_a2, uv_a1, uv_b2, uv_b1 = correspondence_augmentation.merge_images_with_occlusions(image_a2_rgb, image_b2_rgb, image_a2_mask, image_b2_mask, matches_pair_a, matches_pair_b)
        if uv_a1 is None or uv_a2 is None or uv_b1 is None or uv_b2 is None:
            logging.info('something got fully occluded, returning')
            image_b1_rgb_tensor = self.rgb_image_to_tensor(image_b1_rgb)
            return self.return_empty_data(image_b1_rgb_tensor, image_b1_rgb_tensor)
        matches_1 = correspondence_augmentation.merge_matches(uv_a1, uv_b1)
        matches_2 = correspondence_augmentation.merge_matches(uv_a2, uv_b2)
        matches_2 = matches_2[0].float(), matches_2[1].float()
        merged_mask_2_torch = torch.from_numpy(merged_mask_2).type(torch.FloatTensor)
        image_b_shape = merged_mask_2_torch.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]
        matches_2_masked_non_matches = correspondence_finder.create_non_correspondences(matches_2, image_b_shape, num_non_matches_per_match=self.num_masked_non_matches_per_match, img_b_mask=merged_mask_2_torch)
        if self._use_image_b_mask_inv:
            merged_mask_2_torch_inv = 1 - merged_mask_2_torch
        else:
            merged_mask_2_torch_inv = None
        matches_2_background_non_matches = correspondence_finder.create_non_correspondences(matches_2, image_b_shape, num_non_matches_per_match=self.num_background_non_matches_per_match, img_b_mask=merged_mask_2_torch_inv)
        SD = SpartanDataset
        merged_rgb_1_PIL = merged_rgb_1
        merged_rgb_2_PIL = merged_rgb_2
        merged_rgb_1 = self.rgb_image_to_tensor(merged_rgb_1)
        merged_rgb_2 = self.rgb_image_to_tensor(merged_rgb_2)
        matches_a = SD.flatten_uv_tensor(matches_1, image_width)
        matches_b = SD.flatten_uv_tensor(matches_2, image_width)
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(matches_1, matches_2_masked_non_matches, self.num_masked_non_matches_per_match)
        masked_non_matches_a = SD.flatten_uv_tensor(uv_a_masked_long, image_width).squeeze(1)
        masked_non_matches_b = SD.flatten_uv_tensor(uv_b_masked_non_matches_long, image_width).squeeze(1)
        uv_a_background_long, uv_b_background_non_matches_long = self.create_non_matches(matches_1, matches_2_background_non_matches, self.num_background_non_matches_per_match)
        background_non_matches_a = SD.flatten_uv_tensor(uv_a_background_long, image_width).squeeze(1)
        background_non_matches_b = SD.flatten_uv_tensor(uv_b_background_non_matches_long, image_width).squeeze(1)
        if self.debug:
            num_matches_to_plot = 10
            None
            plot_uv_a1, plot_uv_a2 = SpartanDataset.subsample_tuple_pair(uv_a1, uv_a2, num_samples=num_matches_to_plot)
            plot_uv_b1, plot_uv_b2 = SpartanDataset.subsample_tuple_pair(uv_b1, uv_b2, num_samples=num_matches_to_plot)
            None
            plot_uv_1, plot_uv_2 = SpartanDataset.subsample_tuple_pair(matches_1, matches_2, num_samples=num_matches_to_plot)
            plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long = SpartanDataset.subsample_tuple_pair(uv_a_masked_long, uv_b_masked_non_matches_long, num_samples=num_matches_to_plot)
            plot_uv_a_background_long, plot_uv_b_background_non_matches_long = SpartanDataset.subsample_tuple_pair(uv_a_background_long, uv_b_background_non_matches_long, num_samples=num_matches_to_plot)
            fig, axes = correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth), merged_rgb_2_PIL, np.asarray(image_b2_depth), plot_uv_1, plot_uv_2, circ_color='g', show=False)
            correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth), merged_rgb_2_PIL, np.asarray(image_b2_depth), plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long, use_previous_plot=(fig, axes), circ_color='r', show=True)
            fig, axes = correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth), merged_rgb_2_PIL, np.asarray(image_b2_depth), plot_uv_1, plot_uv_2, circ_color='g', show=False)
            correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth), merged_rgb_2_PIL, np.asarray(image_b2_depth), plot_uv_a_background_long, plot_uv_b_background_non_matches_long, use_previous_plot=(fig, axes), circ_color='b')
        return metadata['type'], merged_rgb_1, merged_rgb_2, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, SD.empty_tensor(), SD.empty_tensor(), metadata

    def get_across_scene_data(self, scene_name_a, scene_name_b, metadata):
        """
        Essentially just returns a bunch of samples off the masks from scene_name_a, and scene_name_b.

        Since this data is across scene, we can't generate matches.

        Return args are for returning directly from __getitem__

        See get_within_scene_data() for documentation of return args.

        :param scene_name_a, scene_name_b: Names of scenes from which to each randomly sample an image
        :type scene_name_a, scene_name_b: strings
        :param metadata: a dict() holding metadata of the image pair, both for logging and for different downstream loss functions
        :type metadata: dict()
        """
        SD = SpartanDataset
        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError('There are no single object scenes in this dataset')
        image_a_idx = self.get_random_image_index(scene_name_a)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name_a, image_a_idx)
        metadata['image_a_idx'] = image_a_idx
        image_b_idx = self.get_random_image_index(scene_name_b)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name_b, image_b_idx)
        metadata['image_b_idx'] = image_b_idx
        num_samples = self.cross_scene_num_samples
        blind_uv_a = correspondence_finder.random_sample_from_masked_image_torch(np.asarray(image_a_mask), num_samples)
        blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(np.asarray(image_b_mask), num_samples)
        if blind_uv_a[0] is None or blind_uv_b[0] is None:
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)
        if not self.debug:
            [image_a_rgb, image_a_mask], blind_uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_mask], blind_uv_a)
            [image_b_rgb, image_b_mask], blind_uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_mask], blind_uv_b)
        else:
            [image_a_rgb, image_a_depth, image_a_mask], blind_uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_depth, image_a_mask], blind_uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], blind_uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_depth, image_b_mask], blind_uv_b)
        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)
        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]
        blind_uv_a_flat = SD.flatten_uv_tensor(blind_uv_a, image_width)
        blind_uv_b_flat = SD.flatten_uv_tensor(blind_uv_b, image_width)
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)
        empty_tensor = SD.empty_tensor()
        if self.debug and (blind_uv_a[0] is not None and blind_uv_b[0] is not None):
            num_matches_to_plot = 10
            plot_blind_uv_a, plot_blind_uv_b = SD.subsample_tuple_pair(blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot * 10)
            correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL, image_b_depth_numpy, plot_blind_uv_a, plot_blind_uv_b, circ_color='k', show=True)
        return metadata['type'], image_a_rgb, image_b_rgb, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, blind_uv_a_flat, blind_uv_b_flat, metadata

    def get_image_mean(self):
        """
        Returns dataset image_mean
        :return: list
        :rtype:
        """
        return constants.DEFAULT_IMAGE_MEAN

    def get_image_std_dev(self):
        """
        Returns dataset image std_dev
        :return: list
        :rtype:
        """
        return constants.DEFAULT_IMAGE_STD_DEV

    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """
        return self._rgb_image_to_tensor(img)

    def get_first_image_index(self, scene_name):
        """
        Gets the image index for the "first" image in that scene.
        Correctly handles the case where we did a close-up data collection
        :param scene_name:
        :type scene_name: string
        :return: index of first image in scene
        :rtype: int
        """
        full_path_for_scene = self.get_full_path_for_scene(scene_name)
        ss = SceneStructure(full_path_for_scene)
        metadata_file = ss.metadata_file
        first_image_index = None
        if os.path.exists(metadata_file):
            metadata = utils.getDictFromYamlFilename(metadata_file)
            if len(metadata['close_up_image_indices']) > 0:
                first_image_index = min(metadata['close_up_image_indices'])
            else:
                first_image_index = min(metadata['normal_image_indices'])
        else:
            pose_data = self.get_pose_data(scene_name)
            first_image_index = min(pose_data.keys())
        return first_image_index

    @property
    def config(self):
        return self._config

    @staticmethod
    def merge_single_object_configs(config_list):
        """
        Given a list of single object configs, merge them. This basically concatenates
        all the fields ('train', 'test', 'logs_root_path')

        Asserts that 'object_id' is the same for all of the configs
        Asserts that `logs_root_path` is the same for all the configs
        :param config_list:
        :type config_list:
        :return: single object config
        :rtype: dict
        """
        config = config_list[0]
        logs_root_path = config['logs_root_path']
        object_id = config['object_id']
        train_scenes = []
        test_scenes = []
        evaluation_labeled_data_path = []
        for config in config_list:
            assert config['object_id'] == object_id
            assert config['logs_root_path'] == logs_root_path
            train_scenes += config['train']
            test_scenes += config['test']
            evaluation_labeled_data_path += config['evaluation_labeled_data_path']
        merged_config = dict()
        merged_config['logs_root_path'] = logs_root_path
        merged_config['object_id'] = object_id
        merged_config['train'] = train_scenes
        merged_config['test'] = test_scenes
        merged_config['evaluation_labeled_data_path'] = evaluation_labeled_data_path
        return merged_config

    @staticmethod
    def flatten_uv_tensor(uv_tensor, image_width):
        """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
        """
        return uv_tensor[1].long() * image_width + uv_tensor[0].long()

    @staticmethod
    def mask_image_from_uv_flat_tensor(uv_flat_tensor, image_width, image_height):
        """
        Returns a torch.LongTensor with shape [image_width*image_height]. It has a 1 exactly
        at the indices specified by uv_flat_tensor
        :param uv_flat_tensor:
        :type uv_flat_tensor:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :return:
        :rtype:
        """
        image_flat = torch.zeros(image_width * image_height).long()
        image_flat[uv_flat_tensor] = 1
        return image_flat

    @staticmethod
    def subsample_tuple(uv, num_samples):
        """
        Subsamples a tuple of (torch.Tensor, torch.Tensor)
        """
        indexes_to_keep = (torch.rand(num_samples) * len(uv[0])).floor().type(torch.LongTensor)
        return torch.index_select(uv[0], 0, indexes_to_keep), torch.index_select(uv[1], 0, indexes_to_keep)

    @staticmethod
    def subsample_tuple_pair(uv_a, uv_b, num_samples):
        """
        Subsamples a pair of tuples, i.e. (torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)
        """
        assert len(uv_a[0]) == len(uv_b[0])
        indexes_to_keep = (torch.rand(num_samples) * len(uv_a[0])).floor().type(torch.LongTensor)
        uv_a_downsampled = torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep)
        uv_b_downsampled = torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep)
        return uv_a_downsampled, uv_b_downsampled

    @staticmethod
    def make_default_10_scenes_drill():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset', '10_drill_scenes.yaml')
        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode='train', config=config)
        return dataset

    @staticmethod
    def make_default_caterpillar():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset', 'composite', 'caterpillar_only.yaml')
        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode='train', config=config)
        return dataset


class DenseCorrespondenceNetwork(nn.Module):
    IMAGE_TO_TENSOR = valid_transform = transforms.Compose([transforms.ToTensor()])

    def __init__(self, fcn, descriptor_dimension, image_width=640, image_height=480, normalize=False):
        """

        :param fcn:
        :type fcn:
        :param descriptor_dimension:
        :type descriptor_dimension:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :param normalize: If True normalizes the feature vectors to lie on unit ball
        :type normalize:
        """
        super(DenseCorrespondenceNetwork, self).__init__()
        self._fcn = fcn
        self._descriptor_dimension = descriptor_dimension
        self._image_width = image_width
        self._image_height = image_height
        self._image_mean = np.zeros(3)
        self._image_std_dev = np.ones(3)
        self.config = dict()
        self._descriptor_image_stats = None
        self._normalize = normalize
        self._constructed_from_model_folder = False

    @property
    def fcn(self):
        return self._fcn

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def descriptor_dimension(self):
        return self._descriptor_dimension

    @property
    def image_shape(self):
        return [self._image_height, self._image_width]

    @property
    def image_mean(self):
        return self._image_mean

    @image_mean.setter
    def image_mean(self, value):
        """
        Sets the image mean used in normalizing the images before
        being passed through the network
        :param value: list of floats
        :type value:
        :return:
        :rtype:
        """
        self._image_mean = value
        self.config['image_mean'] = value
        self._update_normalize_tensor_transform()

    @property
    def image_std_dev(self):
        return self._image_std_dev

    @image_std_dev.setter
    def image_std_dev(self, value):
        """
        Sets the image std dev used in normalizing the images before
        being passed through the network
        :param value: list of floats
        :type value:
        :return:
        :rtype:
        """
        self._image_std_dev = value
        self.config['image_std_dev'] = value
        self._update_normalize_tensor_transform()

    @property
    def image_to_tensor(self):
        return self._image_to_tensor

    @image_to_tensor.setter
    def image_to_tensor(self, value):
        self._image_to_tensor = value

    @property
    def normalize_tensor_transform(self):
        return self._normalize_tensor_transform

    @property
    def path_to_network_params_folder(self):
        if not 'path_to_network_params_folder' in self.config:
            raise ValueError("DenseCorrespondenceNetwork: Config doesn't have a `path_to_network_params_folder`entry")
        return self.config['path_to_network_params_folder']

    @property
    def descriptor_image_stats(self):
        """
        Returns the descriptor normalization parameters, if possible.
        If they have not yet been loaded then it loads them
        :return:
        :rtype:
        """
        if self._descriptor_image_stats is None:
            path_to_params = utils.convert_to_absolute_path(self.path_to_network_params_folder)
            descriptor_stats_file = os.path.join(path_to_params, 'descriptor_statistics.yaml')
            self._descriptor_image_stats = utils.getDictFromYamlFilename(descriptor_stats_file)
        return self._descriptor_image_stats

    @property
    def constructed_from_model_folder(self):
        """
        Returns True if this model was constructed from
        :return:
        :rtype:
        """
        return self._constructed_from_model_folder

    @constructed_from_model_folder.setter
    def constructed_from_model_folder(self, value):
        self._constructed_from_model_folder = value

    @property
    def unique_identifier(self):
        """
        Return the unique identifier for this network, if it has one.
        If no identifier.yaml found (or we don't even have a model params folder)
        then return None
        :return:
        :rtype:
        """
        try:
            path_to_network_params_folder = self.path_to_network_params_folder
        except ValueError:
            return None
        identifier_file = os.path.join(path_to_network_params_folder, 'identifier.yaml')
        if not os.path.exists(identifier_file):
            return None
        if not self.constructed_from_model_folder:
            return None
        d = utils.getDictFromYamlFilename(identifier_file)
        unique_identifier = d['id'] + '+' + self.config['model_param_filename_tail']
        return unique_identifier

    def _update_normalize_tensor_transform(self):
        """
        Updates the image to tensor transform using the current image mean and
        std dev
        :return: None
        :rtype:
        """
        self._normalize_tensor_transform = transforms.Normalize(self.image_mean, self.image_std_dev)

    def forward_on_img(self, img, cuda=True):
        """
        Runs the network forward on an image
        :param img: img is an image as a numpy array in opencv format [0,255]
        :return:
        """
        img_tensor = DenseCorrespondenceNetwork.IMAGE_TO_TENSOR(img)
        if cuda:
            img_tensor
        return self.forward(img_tensor)

    def forward_on_img_tensor(self, img):
        """
        Deprecated, use `forward` instead
        Runs the network forward on an img_tensor
        :param img: (C x H X W) in range [0.0, 1.0]
        :return:
        """
        warnings.warn('use forward method instead', DeprecationWarning)
        img = img.unsqueeze(0)
        img = torch.tensor(img, device=torch.device('cuda'))
        res = self.fcn(img)
        res = res.squeeze(0)
        res = res.permute(1, 2, 0)
        res = res.data.cpu().numpy().squeeze()
        return res

    def forward(self, img_tensor):
        """
        Simple forward pass on the network.

        Does NOT normalize the image

        D = descriptor dimension
        N = batch size

        :param img_tensor: input tensor img.shape = [N, D, H , W] where
                    N is the batch size
        :type img_tensor: torch.Variable or torch.Tensor
        :return: torch.Variable with shape [N, D, H, W],
        :rtype:
        """
        res = self.fcn(img_tensor)
        if self._normalize:
            norm = torch.norm(res, 2, 1)
            res = res / norm
        return res

    def forward_single_image_tensor(self, img_tensor):
        """
        Simple forward pass on the network.

        Assumes the image has already been normalized (i.e. subtract mean, divide by std dev)

        Color channel should be RGB

        :param img_tensor: torch.FloatTensor with shape [3,H,W]
        :type img_tensor:
        :return: torch.FloatTensor with shape  [H, W, D]
        :rtype:
        """
        assert len(img_tensor.shape) == 3
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = torch.tensor(img_tensor, device=torch.device('cuda'))
        res = self.forward(img_tensor)
        res = res.squeeze(0)
        res = res.permute(1, 2, 0)
        return res

    def process_network_output(self, image_pred, N):
        """
        Processes the network output into a new shape

        :param image_pred: output of the network img.shape = [N,descriptor_dim, H , W]
        :type image_pred: torch.Tensor
        :param N: batch size
        :type N: int
        :return: same as input, new shape is [N, W*H, descriptor_dim]
        :rtype:
        """
        W = self._image_width
        H = self._image_height
        image_pred = image_pred.view(N, self.descriptor_dimension, W * H)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred

    def clip_pixel_to_image_size_and_round(self, uv):
        """
        Clips pixel to image coordinates and converts to int
        :param uv:
        :type uv:
        :return:
        :rtype:
        """
        u = min(int(round(uv[0])), self._image_width - 1)
        v = min(int(round(uv[1])), self._image_height - 1)
        return [u, v]

    def load_training_dataset(self):
        """
        Loads the dataset that this was trained on
        :return: a dataset object, loaded with the config as set in the dataset.yaml
        :rtype: SpartanDataset
        """
        network_params_folder = self.path_to_network_params_folder
        network_params_folder = utils.convert_to_absolute_path(network_params_folder)
        dataset_config_file = os.path.join(network_params_folder, 'dataset.yaml')
        config = utils.getDictFromYamlFilename(dataset_config_file)
        return SpartanDataset(config_expanded=config)

    @staticmethod
    def get_unet(config):
        """
        Returns a Unet nn.module that satisifies the fcn properties stated in get_fcn() docstring
        """
        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        sys.path.append(os.path.join(dc_source_dir, 'external/unet-pytorch'))
        model = UNet(num_classes=config['descriptor_dimension'])
        return model

    @staticmethod
    def get_fcn(config):
        """
        Returns a pytorch nn.module that satisfies these properties:

        1. autodiffs
        2. has forward() overloaded
        3. can accept a ~Nx3xHxW (should double check)
        4. outputs    a ~NxDxHxW (should double check)

        :param config: Dict with dcn configuration parameters

        """
        if config['backbone']['model_class'] == 'Resnet':
            resnet_model = config['backbone']['resnet_name']
            fcn = getattr(resnet_dilated, resnet_model)(num_classes=config['descriptor_dimension'])
        elif config['backbone']['model_class'] == 'Unet':
            fcn = DenseCorrespondenceNetwork.get_unet(config)
        else:
            raise ValueError("Can't build backbone network.  I don't know this backbone model class!")
        return fcn

    @staticmethod
    def from_config(config, load_stored_params=True, model_param_file=None):
        """
        Load a network from a configuration


        :param config: Dict specifying details of the network architecture

        :param load_stored_params: whether or not to load stored params, if so there should be
            a "path_to_network" entry in the config
        :type load_stored_params: bool

        e.g.
            path_to_network: /home/manuelli/code/dense_correspondence/recipes/trained_models/10_drill_long_3d
            parameter_file: dense_resnet_34_8s_03505.pth
            descriptor_dimensionality: 3
            image_width: 640
            image_height: 480

        :return: DenseCorrespondenceNetwork
        :rtype:
        """
        if 'backbone' not in config:
            config['backbone'] = dict()
            config['backbone']['model_class'] = 'Resnet'
            config['backbone']['resnet_name'] = 'Resnet34_8s'
        fcn = DenseCorrespondenceNetwork.get_fcn(config)
        if 'normalize' in config:
            normalize = config['normalize']
        else:
            normalize = False
        dcn = DenseCorrespondenceNetwork(fcn, config['descriptor_dimension'], image_width=config['image_width'], image_height=config['image_height'], normalize=normalize)
        if load_stored_params:
            assert model_param_file is not None
            config['model_param_file'] = model_param_file
            try:
                dcn.load_state_dict(torch.load(model_param_file))
            except:
                logging.info('loading params with the new style failed, falling back to dcn.fcn.load_state_dict')
                dcn.fcn.load_state_dict(torch.load(model_param_file))
        dcn
        dcn.train()
        dcn.config = config
        return dcn

    @staticmethod
    def from_model_folder(model_folder, load_stored_params=True, model_param_file=None, iteration=None):
        """
        Loads a DenseCorrespondenceNetwork from a model folder
        :param model_folder: the path to the folder where the model is stored. This direction contains
        files like

            - 003500.pth
            - training.yaml

        :type model_folder:
        :return: a DenseCorrespondenceNetwork objecc t
        :rtype:
        """
        from_model_folder = False
        model_folder = utils.convert_to_absolute_path(model_folder)
        if model_param_file is None:
            model_param_file, _, _ = utils.get_model_param_file_from_directory(model_folder, iteration=iteration)
            from_model_folder = True
        model_param_file = utils.convert_to_absolute_path(model_param_file)
        training_config_filename = os.path.join(model_folder, 'training.yaml')
        training_config = utils.getDictFromYamlFilename(training_config_filename)
        config = training_config['dense_correspondence_network']
        config['path_to_network_params_folder'] = model_folder
        config['model_param_filename_tail'] = os.path.split(model_param_file)[1]
        dcn = DenseCorrespondenceNetwork.from_config(config, load_stored_params=load_stored_params, model_param_file=model_param_file)
        dcn.constructed_from_model_folder = from_model_folder
        dcn.model_folder = model_folder
        return dcn

    @staticmethod
    def find_best_match(pixel_a, res_a, res_b, debug=False):
        """
        Compute the correspondences between the pixel_a location in image_a
        and image_b

        :param pixel_a: vector of (u,v) pixel coordinates
        :param res_a: array of dense descriptors res_a.shape = [H,W,D]
        :param res_b: array of dense descriptors
        :param pixel_b: Ground truth . . .
        :return: (best_match_uv, best_match_diff, norm_diffs)
        best_match_idx is again in (u,v) = (right, down) coordinates

        """
        descriptor_at_pixel = res_a[pixel_a[1], pixel_a[0]]
        height, width, _ = res_a.shape
        if debug:
            None
            None
            None
        norm_diffs = np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))
        best_match_flattened_idx = np.argmin(norm_diffs)
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_xy]
        best_match_uv = best_match_xy[1], best_match_xy[0]
        return best_match_uv, best_match_diff, norm_diffs

    @staticmethod
    def find_best_match_for_descriptor(descriptor, res):
        """
        Compute the correspondences between the given descriptor and the descriptor image
        res
        :param descriptor:
        :type descriptor:
        :param res: array of dense descriptors res = [H,W,D]
        :type res: numpy array with shape [H,W,D]
        :return: (best_match_uv, best_match_diff, norm_diffs)
        best_match_idx is again in (u,v) = (right, down) coordinates
        :rtype:
        """
        height, width, _ = res.shape
        norm_diffs = np.sqrt(np.sum(np.square(res - descriptor), axis=2))
        best_match_flattened_idx = np.argmin(norm_diffs)
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_xy]
        best_match_uv = best_match_xy[1], best_match_xy[0]
        return best_match_uv, best_match_diff, norm_diffs

    def evaluate_descriptor_at_keypoints(self, res, keypoint_list):
        """

        :param res: result of evaluating the network
        :type res: torch.FloatTensor [D,W,H]
        :param img:
        :type img: img_tensor
        :param kp: list of cv2.KeyPoint
        :type kp:
        :return: numpy.ndarray (N,D) N = num keypoints, D = descriptor dimension
        This is the same format as sift.compute from OpenCV
        :rtype:
        """
        raise NotImplementedError('This function is currently broken')
        N = len(keypoint_list)
        D = self.descriptor_dimension
        des = np.zeros([N, D])
        for idx, kp in enumerate(keypoint_list):
            uv = self.clip_pixel_to_image_size_and_round([kp.pt[0], kp.pt[1]])
            des[idx, :] = res[uv[1], uv[0], :]
        des = np.array(des, dtype=np.float32)
        return des


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseCorrespondenceNetwork,
     lambda: ([], {'fcn': _mock_layer(), 'descriptor_dimension': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_RobotLocomotion_pytorch_dense_correspondence(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

