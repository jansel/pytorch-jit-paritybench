import sys
_module = sys.modules[__name__]
del sys
images = _module
mano = _module
joints_info = _module
lbs = _module
model = _module
utils = _module
setup = _module

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


import torch


import torch.nn.functional as F


import numpy as np


from collections import namedtuple


import torch.nn as nn


def points2sphere(points, radius=0.001, vc=[0.0, 0.0, 1.0], count=[5, 5]):
    points = points.reshape(-1, 3)
    n_points = points.shape[0]
    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count=count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)
        spheres.append(sphs)
    spheres = Mesh.concatenate_meshes(spheres)
    return spheres


ModelOutput = namedtuple('ModelOutput', ['vertices', 'joints', 'full_pose', 'betas', 'transl', 'global_orient', 'hand_pose'])


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


TIP_IDS = {'mano': {'thumb': 744, 'index': 320, 'middle': 443, 'ring': 554, 'pinky': 671}}


colors = {'pink': [1.0, 0.75, 0.8], 'skin': [0.96, 0.75, 0.69], 'purple': [0.63, 0.13, 0.94], 'red': [1.0, 0.0, 0.0], 'green': [0.0, 1.0, 0.0], 'yellow': [1.0, 1.0, 0], 'brown': [1.0, 0.25, 0.25], 'blue': [0.0, 0.0, 1.0], 'white': [1.0, 1.0, 1.0], 'orange': [1.0, 0.65, 0.0], 'grey': [0.75, 0.75, 0.75], 'black': [0.0, 0.0, 0.0]}


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype=torch.float32):
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-08, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def blend_shapes(betas, shape_disps):
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor, vertices):
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot=True, dtype=torch.float32):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todense())
    elif 'chumpy' in str(type(array)):
        array = np.array(array)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array.astype(dtype)


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array


class MANO(nn.Module):
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS
    NUM_BETAS = 10

    def __init__(self, model_path, is_rhand=True, data_struct=None, create_betas=True, betas=None, create_global_orient=True, global_orient=None, create_transl=True, transl=None, create_hand_pose=True, hand_pose=None, use_pca=True, num_pca_comps=6, flat_hand_mean=False, batch_size=1, joint_mapper=None, v_template=None, dtype=torch.float32, vertex_ids=None, use_compressed=True, ext='pkl', **kwargs):
        """ MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the
                hand. (default = True)
            hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        """
        self.num_pca_comps = num_pca_comps
        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format('RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if 'RIGHT' in os.path.basename(model_path) else False
            assert osp.exists(mano_path), 'Path {} does not exist!'.format(mano_path)
            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)
        self.tip_ids = TIP_IDS['mano']
        super(MANO, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.joint_mapper = joint_mapper
        self.faces = data_struct.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.NUM_BETAS], dtype=dtype)
            elif 'torch.Tensor' in str(type(betas)):
                default_betas = betas.clone().detach()
            else:
                default_betas = torch.tensor(betas, dtype=dtype)
            self.register_parameter('betas', nn.Parameter(default_betas, requires_grad=True))
        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3], dtype=dtype)
            elif torch.is_tensor(global_orient):
                default_global_orient = global_orient.clone().detach()
            else:
                default_global_orient = torch.tensor(global_orient, dtype=dtype)
            global_orient = nn.Parameter(default_global_orient, requires_grad=True)
            self.register_parameter('global_orient', global_orient)
        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3], dtype=dtype, requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter('transl', nn.Parameter(default_transl, requires_grad=True))
        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        self.register_buffer('v_template', to_tensor(v_template, dtype=dtype))
        shapedirs = data_struct.shapedirs
        self.register_buffer('shapedirs', to_tensor(to_np(shapedirs), dtype=dtype))
        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)
        num_pose_basis = data_struct.posedirs.shape[-1]
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=dtype))
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(data_struct.weights), dtype=dtype))
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean
        hand_components = data_struct.hands_components[:num_pca_comps]
        self.np_hand_components = hand_components
        if self.use_pca:
            self.register_buffer('hand_components', torch.tensor(hand_components, dtype=dtype))
        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean
        self.register_buffer('hand_mean', to_tensor(hand_mean, dtype=self.dtype))
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)
            hand_pose_param = nn.Parameter(default_hand_pose, requires_grad=True)
            self.register_parameter('hand_pose', hand_pose_param)
        pose_mean = self.create_mean_pose(data_struct, flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = pose_mean.clone()
        self.register_buffer('pose_mean', pose_mean_tensor)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)
        return pose_mean

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        msg = 'Number of betas: {}'.format(self.NUM_BETAS)
        if self.use_pca:
            msg += '\nNumber of PCA components: {}'.format(self.num_pca_comps)
        msg += '\nFlat hand mean: {}'.format(self.flat_hand_mean)
        return msg

    def add_joints(self, vertices, joints, joint_ids=None):
        dev = vertices.device
        if joint_ids is None:
            joint_ids = to_tensor(list(self.tip_ids.values()), dtype=torch.long)
        extra_joints = torch.index_select(vertices, 1, joint_ids)
        joints = torch.cat([joints, extra_joints], dim=1)
        return joints

    def forward(self, betas=None, global_orient=None, hand_pose=None, transl=None, return_verts=True, return_tips=False, return_full_pose=False, pose2rot=True, **kwargs):
        """
        """
        global_orient = global_orient if global_orient is not None else self.global_orient
        betas = betas if betas is not None else self.betas
        hand_pose = hand_pose if hand_pose is not None else self.hand_pose
        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl
        if self.use_pca:
            hand_pose = torch.einsum('bi,ij->bj', [hand_pose, self.hand_components])
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean
        if return_verts:
            vertices, joints = lbs(betas, full_pose, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype)
            if return_tips:
                joints = self.add_joints(vertices, joints)
            if self.joint_mapper is not None:
                joints = self.joint_mapper(joints)
            if apply_trans:
                joints = joints + transl.unsqueeze(dim=1)
                vertices = vertices + transl.unsqueeze(dim=1)
        output = ModelOutput(vertices=vertices if return_verts else None, joints=joints if return_verts else None, betas=betas, global_orient=global_orient, hand_pose=hand_pose, full_pose=full_pose if return_full_pose else None)
        return output

    def hand_meshes(self, output, vc=colors['skin']):
        vertices = to_np(output.vertices)
        if vertices.ndim < 3:
            vertices = vertices.reshape(-1, 778, 3)
        meshes = []
        for v in vertices:
            hand_mesh = Mesh(vertices=v, faces=self.faces, vc=vc)
            meshes.append(hand_mesh)
        return meshes

    def joint_meshes(self, output, radius=0.002, vc=colors['green']):
        joints = to_np(output.joints)
        if joints.ndim < 3:
            joints = joints.reshape(1, -1, 3)
        meshes = []
        for j in joints:
            joint_mesh = Mesh(vertices=j, radius=radius, vc=vc)
            meshes.append(joint_mesh)
        return meshes

