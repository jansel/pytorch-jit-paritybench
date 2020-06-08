import sys
_module = sys.modules[__name__]
del sys
Logger = _module
config = _module
MovingObj3D = _module
SimpleLoader = _module
TUM_RGBD = _module
dataloader = _module
evaluate = _module
LeastSquareTracking = _module
models = _module
algorithms = _module
criterions = _module
geometry = _module
submodules = _module
run_example = _module
timers = _module
train = _module
train_utils = _module

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


import torch


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as func


from torch import sin


from torch import cos


from torch import atan2


from torch import acos


from torch.nn import init


from torch.utils.data import DataLoader


def color_normalize(color):
    rgb_mean = torch.Tensor([0.4914, 0.4822, 0.4465]).type_as(color)
    rgb_std = torch.Tensor([0.2023, 0.1994, 0.201]).type_as(color)
    return (color - rgb_mean.view(1, 3, 1, 1)) / rgb_std.view(1, 3, 1, 1)


class LeastSquareTracking(nn.Module):
    NONE = -1
    RGB = 0
    CONV_RGBD = 1
    CONV_RGBD2 = 2

    def __init__(self, encoder_name, max_iter_per_pyr, mEst_type,
        solver_type, tr_samples=10, no_weight_sharing=False, timers=None):
        """
        :param the backbone network used for regression.
        :param the maximum number of iterations at each pyramid levels
        :param the type of weighting functions.
        :param the type of solver. 
        :param number of samples in trust-region solver
        :param True if we do not want to share weight at different pyramid levels
        :param (optional) time to benchmark time consumed at each step
        """
        super(LeastSquareTracking, self).__init__()
        self.construct_image_pyramids = ImagePyramids([0, 1, 2, 3], pool='avg')
        self.construct_depth_pyramids = ImagePyramids([0, 1, 2, 3], pool='max')
        self.timers = timers
        """ =============================================================== """
        """             Initialize the Deep Feature Extractor               """
        """ =============================================================== """
        if encoder_name == 'RGB':
            None
            self.encoder = None
            self.encoder_type = self.RGB
            context_dim = 1
        elif encoder_name == 'ConvRGBD':
            None
            context_dim = 4
            self.encoder = FeaturePyramid(D=context_dim)
            self.encoder_type = self.CONV_RGBD
        elif encoder_name == 'ConvRGBD2':
            None
            context_dim = 8
            self.encoder = FeaturePyramid(D=context_dim)
            self.encoder_type = self.CONV_RGBD2
        else:
            raise NotImplementedError()
        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """
        if no_weight_sharing:
            self.mEst_func0 = DeepRobustEstimator(mEst_type)
            self.mEst_func1 = DeepRobustEstimator(mEst_type)
            self.mEst_func2 = DeepRobustEstimator(mEst_type)
            self.mEst_func3 = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func0, self.mEst_func1, self.mEst_func2,
                self.mEst_func3]
        else:
            self.mEst_func = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
                self.mEst_func]
        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """
        if no_weight_sharing:
            self.solver_func0 = DirectSolverNet(solver_type, samples=tr_samples
                )
            self.solver_func1 = DirectSolverNet(solver_type, samples=tr_samples
                )
            self.solver_func2 = DirectSolverNet(solver_type, samples=tr_samples
                )
            self.solver_func3 = DirectSolverNet(solver_type, samples=tr_samples
                )
            solver_funcs = [self.solver_func0, self.solver_func1, self.
                solver_func2, self.solver_func3]
        else:
            self.solver_func = DirectSolverNet(solver_type, samples=tr_samples)
            solver_funcs = [self.solver_func, self.solver_func, self.
                solver_func, self.solver_func]
        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """
        self.tr_update0 = TrustRegion(max_iter_per_pyr, mEst_func=
            mEst_funcs[0], solver_func=solver_funcs[0], timers=timers)
        self.tr_update1 = TrustRegion(max_iter_per_pyr, mEst_func=
            mEst_funcs[1], solver_func=solver_funcs[1], timers=timers)
        self.tr_update2 = TrustRegion(max_iter_per_pyr, mEst_func=
            mEst_funcs[2], solver_func=solver_funcs[2], timers=timers)
        self.tr_update3 = TrustRegion(max_iter_per_pyr, mEst_func=
            mEst_funcs[3], solver_func=solver_funcs[3], timers=timers)

    def forward(self, img0, img1, depth0, depth1, K, init_only=False):
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        """
        if self.timers:
            self.timers.tic('extract features')
        invD0 = torch.clamp(1.0 / depth0, 0, 10)
        invD1 = torch.clamp(1.0 / depth1, 0, 10)
        invD0[invD0 == invD0.min()] = 0
        invD1[invD1 == invD1.min()] = 0
        invD0[invD0 == invD0.max()] = 0
        invD1[invD1 == invD1.max()] = 0
        I0 = color_normalize(img0)
        I1 = color_normalize(img1)
        x0 = self.__encode_features(I0, invD0, I1, invD1)
        x1 = self.__encode_features(I1, invD1, I0, invD0)
        d0 = self.construct_depth_pyramids(invD0)
        d1 = self.construct_depth_pyramids(invD1)
        if self.timers:
            self.timers.toc('extract features')
        poses_to_train = [[], []]
        B = invD0.shape[0]
        R0 = torch.eye(3, dtype=torch.float).expand(B, 3, 3).type_as(I0)
        t0 = torch.zeros(B, 3, 1, dtype=torch.float).type_as(I0)
        poseI = [R0, t0]
        prior_W = torch.ones(d0[3].shape).type_as(d0[3])
        if self.timers:
            self.timers.tic('trust-region update')
        K3 = K >> 3
        output3 = self.tr_update3(poseI, x0[3], x1[3], d0[3], d1[3], K3,
            prior_W)
        pose3, mEst_W3 = output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])
        K2 = K >> 2
        output2 = self.tr_update2(pose3, x0[2], x1[2], d0[2], d1[2], K2,
            mEst_W3)
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])
        K1 = K >> 1
        output1 = self.tr_update1(pose2, x0[1], x1[1], d0[1], d1[1], K1,
            mEst_W2)
        pose1, mEst_W1 = output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])
        output0 = self.tr_update0(pose1, x0[0], x1[0], d0[0], d1[0], K, mEst_W1
            )
        pose0 = output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])
        if self.timers:
            self.timers.toc('trust-region update')
        if self.training:
            pyr_R = torch.stack(tuple(poses_to_train[0]), dim=1)
            pyr_t = torch.stack(tuple(poses_to_train[1]), dim=1)
            return pyr_R, pyr_t
        else:
            return pose0

    def __encode_features(self, img0, invD0, img1, invD1):
        """ get the encoded features
        """
        if self.encoder_type == self.RGB:
            I = self.__color3to1(img0)
            x = self.construct_image_pyramids(I)
        elif self.encoder_type == self.CONV_RGBD:
            m = torch.cat((img0, invD0), dim=1)
            x = self.encoder.forward(m)
        elif self.encoder_type in [self.CONV_RGBD2]:
            m = torch.cat((img0, invD0, img1, invD1), dim=1)
            x = self.encoder.forward(m)
        else:
            raise NotImplementedError()
        x = [self.__Nto1(a) for a in x]
        return x

    def __Nto1(self, x):
        """ Take the average of multi-dimension feature into one dimensional,
            which boostrap the optimization speed
        """
        C = x.shape[1]
        return x.sum(dim=1, keepdim=True) / C

    def __color3to1(self, img):
        """ Return a gray-scale image
        """
        B, _, H, W = img.shape
        return (img[:, (0)] * 0.299 + img[:, (1)] * 0.587 + img[:, (2)] * 0.114
            ).view(B, 1, H, W)


def compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p):
    """ chained gradient of image w.r.t. the pose
    :param the Jacobian of the feature map in x direction
    :param the Jacobian of the feature map in y direction
    :param the Jacobian of the x map to manifold p
    :param the Jacobian of the y map to manifold p
    ------------
    :return the image jacobian in x, y, direction, Bx2x6 each
    """
    B, C, H, W = Jf_x.shape
    Jf_p = Jf_x.view(B, C, -1, 1) * Jx_p.view(B, 1, -1, 6) + Jf_y.view(B, C,
        -1, 1) * Jy_p.view(B, 1, -1, 6)
    return Jf_p.view(B, -1, 6)


def compute_warped_residual(pose, invD0, invD1, x0, x1, px, py, K, obj_mask
    =None):
    """ Compute the residual error of warped target image w.r.t. the reference feature map.
    refer to equation (12) in the paper

    :param the forward warping pose from the reference camera to the target frame.
        Note that warping from the target frame to the reference frame is the inverse of this operation.
    :param the reference inverse depth
    :param the target inverse depth
    :param the reference feature image
    :param the target feature image
    :param the pixel x map
    :param the pixel y map
    :param the intrinsic calibration
    -----------
    :return the residual (of reference image), and occlusion information
    """
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(px,
        py, invD0, pose, K)
    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
    occ = geometry.check_occ(inv_z_warped, invD1, u_warped, v_warped)
    residuals = x1_1to0 - x0
    B, C, H, W = x0.shape
    if obj_mask is not None:
        occ = occ & (obj_mask.view(B, 1, H, W) < 1)
    residuals[occ.expand(B, C, H, W)] = 0.001
    return residuals, occ


def feature_gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image 
    -----------
    :return the gradient of the image in x, y direction
    """
    B, C, H, W = img.shape
    wx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1,
        3, 3).type_as(img)
    wy = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1,
        3, 3).type_as(img)
    img_reshaped = img.view(-1, 1, H, W)
    img_pad = func.pad(img_reshaped, (1, 1, 1, 1), mode='replicate')
    img_dx = func.conv2d(img_pad, wx, stride=1, padding=0)
    img_dy = func.conv2d(img_pad, wy, stride=1, padding=0)
    if normalize_gradient:
        mag = torch.sqrt(img_dx ** 2 + img_dy ** 2 + 1e-08)
        img_dx = img_dx / mag
        img_dy = img_dy / mag
    return img_dx.view(B, C, H, W), img_dy.view(B, C, H, W)


def compute_jacobian_warping(p_invdepth, K, px, py):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    B, C, H, W = p_invdepth.size()
    assert C == 1
    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invd = p_invdepth.view(B, -1, 1)
    xy = x * y
    O = torch.zeros((B, H * W, 1)).type_as(p_invdepth)
    dx_dp = torch.cat((-xy, 1 + x ** 2, -y, invd, O, -invd * x), dim=2)
    dy_dp = torch.cat((-1 - y ** 2, xy, x, O, invd, -invd * y), dim=2)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)
    return dx_dp * fx.view(B, 1, 1), dy_dp * fy.view(B, 1, 1)


class TrustRegionBase(nn.Module):
    """ 
    This is the the base function of the trust-region based inverse compositional algorithm. 
    """

    def __init__(self, max_iter=3, mEst_func=None, solver_func=None, timers
        =None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network 
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionBase, self).__init__()
        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers

    def forward(self, pose, x0, x1, invD0, invD1, K, wPrior=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)
        if self.timers:
            self.timers.tic('pre-compute Jacobians')
        J_F_p = self.precompute_Jacobian(invD0, x0, px, py, K)
        if self.timers:
            self.timers.toc('pre-compute Jacobians')
        if self.timers:
            self.timers.tic('compute warping residuals')
        residuals, occ = compute_warped_residual(pose, invD0, invD1, x0, x1,
            px, py, K)
        if self.timers:
            self.timers.toc('compute warping residuals')
        if self.timers:
            self.timers.tic('robust estimator')
        weights = self.mEstimator(residuals, x0, x1, wPrior)
        wJ = weights.view(B, -1, 1) * J_F_p
        if self.timers:
            self.timers.toc('robust estimator')
        if self.timers:
            self.timers.tic('pre-compute JtWJ')
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2), wJ)
        if self.timers:
            self.timers.toc('pre-compute JtWJ')
        for idx in range(self.max_iterations):
            if self.timers:
                self.timers.tic('solve x=A^{-1}b')
            pose = self.directSolver(JtWJ, torch.transpose(J_F_p, 1, 2),
                weights, residuals, pose, invD0, invD1, x0, x1, K)
            if self.timers:
                self.timers.toc('solve x=A^{-1}b')
            if self.timers:
                self.timers.tic('compute warping residuals')
            residuals, occ = compute_warped_residual(pose, invD0, invD1, x0,
                x1, px, py, K)
            if self.timers:
                self.timers.toc('compute warping residuals')
        return pose, weights

    def precompute_Jacobian(self, invD, x, px, py, K):
        """ Pre-compute the image Jacobian on the reference frame
        refer to equation (13) in the paper
        
        :param invD, template depth
        :param x, template feature
        :param px, normalized image coordinate in cols (x)
        :param py, normalized image coordinate in rows (y)
        :param K, the intrinsic parameters, [fx, fy, cx, cy]

        ------------
        :return precomputed image Jacobian on template
        """
        Jf_x, Jf_y = feature_gradient(x)
        Jx_p, Jy_p = compute_jacobian_warping(invD, K, px, py)
        J_F_p = compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p)
        return J_F_p


class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """

    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1 << i, 1 << i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1 << i, 1 << i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        x_out = [f(x) for f in self.multiscales]
        return x_out


def initialize_weights(modules, method='xavier'):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                m.bias.data.zero_()
            if method == 'xavier':
                init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                m.bias.data.zero_()
            if method == 'xavier':
                init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                init.kaiming_uniform_(m.weight)


class FeaturePyramid(nn.Module):
    """ 
    The proposed feature-encoder (A).
    It also supports to extract features using one-view only.
    """

    def __init__(self, D):
        super(FeaturePyramid, self).__init__()
        self.net0 = nn.Sequential(conv(True, D, 16, 3), conv(True, 16, 32, 
            3, dilation=2), conv(True, 32, 32, 3, dilation=2))
        self.net1 = nn.Sequential(conv(True, 32, 32, 3), conv(True, 32, 64,
            3, dilation=2), conv(True, 64, 64, 3, dilation=2))
        self.net2 = nn.Sequential(conv(True, 64, 64, 3), conv(True, 64, 96,
            3, dilation=2), conv(True, 96, 96, 3, dilation=2))
        self.net3 = nn.Sequential(conv(True, 96, 96, 3), conv(True, 96, 128,
            3, dilation=2), conv(True, 128, 128, 3, dilation=2))
        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x0 = self.net0(x)
        x0s = self.downsample(x0)
        x1 = self.net1(x0s)
        x1s = self.downsample(x1)
        x2 = self.net2(x1s)
        x2s = self.downsample(x2)
        x3 = self.net3(x2s)
        return x0, x1, x2, x3


class DeepRobustEstimator(nn.Module):
    """ The M-estimator 

    When use estimator_type = 'MultiScale2w', it is the proposed convolutional M-estimator
    """

    def __init__(self, estimator_type):
        super(DeepRobustEstimator, self).__init__()
        if estimator_type == 'MultiScale2w':
            self.D = 4
        elif estimator_type == 'None':
            self.mEst_func = self.__constant_weight
            self.D = -1
        else:
            raise NotImplementedError()
        if self.D > 0:
            self.net = nn.Sequential(conv(True, self.D, 16, 3, dilation=1),
                conv(True, 16, 32, 3, dilation=2), conv(True, 32, 64, 3,
                dilation=4), conv(True, 64, 1, 3, dilation=1), nn.Sigmoid())
            initialize_weights(self.net)
        else:
            self.net = None

    def forward(self, residual, x0, x1, ws=None):
        """
        :param residual, the residual map
        :param x0, the feature map of the template
        :param x1, the feature map of the image
        :param ws, the initial weighted residual
        """
        if self.D == 1:
            context = residual.abs()
            w = self.net(context)
        elif self.D == 4:
            B, C, H, W = residual.shape
            wl = func.interpolate(ws, (H, W), mode='bilinear',
                align_corners=True)
            context = torch.cat((residual.abs(), x0, x1, wl), dim=1)
            w = self.net(context)
        elif self.D < 0:
            w = self.mEst_func(residual)
        return w

    def __weight_Huber(self, x, alpha=0.02):
        """ weight function of Huber loss:
        refer to P. 24 w(x) at
        https://members.loria.fr/moberger/Enseignement/Master2/Documents/ZhangIVC-97-01.pdf

        Note this current implementation is not differentiable.
        """
        abs_x = torch.abs(x)
        linear_mask = abs_x > alpha
        w = torch.ones(x.shape).type_as(x)
        if linear_mask.sum().item() > 0:
            w[linear_mask] = alpha / abs_x[linear_mask]
        return w

    def __constant_weight(self, x):
        """ mimic the standard least-square when weighting function is constant
        """
        return torch.ones(x.shape).type_as(x)


def fcLayer(in_planes, out_planes, bias=True):
    return nn.Sequential(nn.Linear(in_planes, out_planes, bias), nn.ReLU(
        inplace=True))


def deep_damping_regressor(D):
    """ Output a damping vector at each dimension
    """
    net = nn.Sequential(fcLayer(in_planes=D, out_planes=128, bias=True),
        fcLayer(in_planes=128, out_planes=256, bias=True), fcLayer(
        in_planes=256, out_planes=6, bias=True))
    return net


def invH(H):
    """ Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    if H.is_cuda:
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH


def inverse_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update 
    in the inverse compositional form
    refer to equation (10) in the paper 

    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """
    inv_H = invH(H)
    xi = torch.bmm(inv_H, Rhs)
    d_R = geometry.batch_twist2Mat(-xi[:, :3].view(-1, 3))
    d_t = -torch.bmm(d_R, xi[:, 3:])
    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t)
    return pose


class DirectSolverNet(nn.Module):
    SOLVER_NO_DAMPING = 0
    SOLVER_RESIDUAL_VOLUME = 1

    def __init__(self, solver_type, samples=10):
        super(DirectSolverNet, self).__init__()
        if solver_type == 'Direct-Nodamping':
            self.net = None
            self.type = self.SOLVER_NO_DAMPING
        elif solver_type == 'Direct-ResVol':
            self.samples = samples
            self.net = deep_damping_regressor(D=6 * 6 + 6 * samples)
            self.type = self.SOLVER_RESIDUAL_VOLUME
            initialize_weights(self.net)
        else:
            raise NotImplementedError()

    def forward(self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K):
        """
        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param weights, the weight matrix
        :param R, the residual
        :param pose0, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param x0, the template feature map
        :param x1, the image feature map
        :param K, the intrinsic parameters

        -----------
        :return updated pose
        """
        B = JtJ.shape[0]
        wR = (weights * R).view(B, -1, 1)
        JtR = torch.bmm(Jt, wR)
        if self.type == self.SOLVER_NO_DAMPING:
            diag_mask = torch.eye(6).view(1, 6, 6).type_as(JtJ)
            diagJtJ = diag_mask * JtJ
            traceJtJ = torch.sum(diagJtJ, (2, 1))
            epsilon = (traceJtJ * 1e-06).view(B, 1, 1) * diag_mask
            Hessian = JtJ + epsilon
            pose_update = inverse_update_pose(Hessian, JtR, pose0)
        elif self.type == self.SOLVER_RESIDUAL_VOLUME:
            Hessian = self.__regularize_residual_volume(JtJ, Jt, JtR,
                weights, pose0, invD0, invD1, x0, x1, K, sample_range=self.
                samples)
            pose_update = inverse_update_pose(Hessian, JtR, pose0)
        else:
            raise NotImplementedError()
        return pose_update

    def __regularize_residual_volume(self, JtJ, Jt, JtR, weights, pose,
        invD0, invD1, x0, x1, K, sample_range):
        """ regularize the approximate with residual volume

        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param JtR, the Right-hand size residual
        :param weights, the weight matrix
        :param pose, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param K, the intrinsic parameters
        :param x0, the template feature map
        :param x1, the image feature map
        :param sample_range, the numerb of samples

        ---------------
        :return the damped Hessian matrix
        """
        JtR_volumes = []
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)
        diag_mask = torch.eye(6).view(1, 6, 6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2, 1))
        epsilon = (traceJtJ * 1e-06).view(B, 1, 1) * diag_mask
        n = sample_range
        lambdas = torch.logspace(-5, 5, n).type_as(JtJ)
        for s in range(n):
            D = lambdas[s] * diagJtJ + epsilon
            Hessian = JtJ + D
            pose_s = inverse_update_pose(Hessian, JtR, pose)
            res_s, _ = compute_warped_residual(pose_s, invD0, invD1, x0, x1,
                px, py, K)
            JtR_s = torch.bmm(Jt, (weights * res_s).view(B, -1, 1))
            JtR_volumes.append(JtR_s)
        JtR_flat = torch.cat(tuple(JtR_volumes), dim=2).view(B, -1)
        JtJ_flat = JtJ.view(B, -1)
        damp_est = self.net(torch.cat((JtR_flat, JtJ_flat), dim=1))
        R = diag_mask * damp_est.view(B, 6, 1) + epsilon
        return JtJ + R


class ListModule(nn.Module):
    """ The implementation of a list of modules from
    https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lvzhaoyang_DeeperInverseCompositionalAlgorithm(_paritybench_base):
    pass
