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
visualization = _module
live_heatmap_visualization = _module
start_notebook = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
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


from torchvision import transforms


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
            des[(idx), :] = res[(uv[1]), (uv[0]), :]
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

