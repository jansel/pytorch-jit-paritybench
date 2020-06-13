import sys
_module = sys.modules[__name__]
del sys
ltr = _module
actors = _module
base_actor = _module
bbreg = _module
tracking = _module
admin = _module
environment = _module
loading = _module
model_constructor = _module
multigpu = _module
settings = _module
stats = _module
tensorboard = _module
data = _module
bounding_box_utils = _module
image_loader = _module
loader = _module
processing = _module
processing_utils = _module
sampler = _module
transforms = _module
dataset = _module
base_image_dataset = _module
base_video_dataset = _module
coco = _module
coco_seq = _module
davis = _module
ecssd = _module
got10k = _module
hku_is = _module
imagenetvid = _module
lasot = _module
lvis = _module
msra10k = _module
sbd = _module
synthetic_video = _module
synthetic_video_blend = _module
tracking_net = _module
vos_base = _module
youtubevos = _module
models = _module
backbone = _module
base = _module
resnet = _module
resnet18_vggm = _module
atom = _module
atom_iou_net = _module
layers = _module
activation = _module
blocks = _module
distance = _module
filter = _module
normalization = _module
transform = _module
loss = _module
kl_regression = _module
target_classification = _module
meta = _module
steepestdescent = _module
target_classifier = _module
features = _module
initializer = _module
linear_filter = _module
optimizer = _module
residual_modules = _module
dimpnet = _module
run_training = _module
train_settings = _module
atom = _module
atom_gmm_sampl = _module
atom_paper = _module
atom_prob_ml = _module
dimp = _module
dimp18 = _module
dimp50 = _module
prdimp18 = _module
prdimp50 = _module
super_dimp = _module
trainers = _module
base_trainer = _module
ltr_trainer = _module
vot = _module
pytracking = _module
analysis = _module
evaluate_vos = _module
extract_results = _module
playback_results = _module
plot_results = _module
vos_utils = _module
evaluation = _module
datasets = _module
got10kdataset = _module
lasotdataset = _module
mobifacedataset = _module
multi_object_wrapper = _module
nfsdataset = _module
otbdataset = _module
running = _module
tpldataset = _module
tracker = _module
trackingnetdataset = _module
uavdataset = _module
vot2020 = _module
votdataset = _module
experiments = _module
myexperiments = _module
augmentation = _module
color = _module
deep = _module
extractor = _module
featurebase = _module
net_wrappers = _module
preprocessing = _module
util = _module
libs = _module
complex = _module
dcf = _module
fourier = _module
operation = _module
optimization = _module
tensordict = _module
tensorlist = _module
parameter = _module
default = _module
default_vot = _module
multiscale_no_iounet = _module
dimp18_vot = _module
dimp50_vot = _module
dimp50_vot19 = _module
eco = _module
run_experiment = _module
run_tracker = _module
run_video = _module
run_vot = _module
run_webcam = _module
atom = _module
optim = _module
basetracker = _module
dimp = _module
eco = _module
optim = _module
util_scripts = _module
download_results = _module
pack_got10k_results = _module
pack_trackingnet_results = _module
utils = _module
convert_vot_anno_to_rect = _module
load_text = _module
params = _module
plotting = _module
visdom = _module

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


import torch.nn as nn


import torch


import math


import random


import torch.nn.functional as F


import numpy as np


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


from torch import nn


from torch.nn import functional as F


import torch.optim as optim


import torch.nn


class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)


class Backbone(nn.Module):
    """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """

    def __init__(self, frozen_layers=()):
        super().__init__()
        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError(
                    'Unknown option for frozen layers: "{}". Should be "all", "none" or list of layer names.'
                    .format(frozen_layers))
        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False

    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True

    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower(
            ) == 'all':
            self.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self, layer).eval()

    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower(
            ) == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self, layer).parameters():
                    p.requires_grad_(False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SpatialCrossMapLRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1,
        ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1,
                padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class ATOMnet(nn.Module):
    """ ATOM network module"""

    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer,
        extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ATOMnet, self).__init__()
        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer
        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1,
            *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *
            test_imgs.shape[-3:]))
        train_feat_iou = [feat for feat in train_feat.values()]
        test_feat_iou = [feat for feat in test_feat.values()]
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
            train_bb.reshape(num_train_images, num_sequences, 4),
            test_proposals.reshape(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1
    ):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
        kernel_size, stride=stride, padding=padding, dilation=dilation,
        bias=True), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class AtomIoUNet(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=(128, 256), pred_input_dim=(256, 256),
        pred_inter_dim=(256, 256)):
        super().__init__()
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)
        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)
        self.prroi_pool3r = PrRoIPool2D(3, 3, 1 / 8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1 / 8)
        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)
        self.prroi_pool4r = PrRoIPool2D(1, 1, 1 / 16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)
        self.fc34_3r = conv(256 + 256, pred_input_dim[0], kernel_size=1,
            stride=1, padding=0)
        self.fc34_4r = conv(256 + 256, pred_input_dim[1], kernel_size=1,
            stride=1, padding=0)
        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)
        self.iou_predictor = nn.Linear(pred_inter_dim[0] + pred_inter_dim[1
            ], 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d
                ) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""
        assert bb1.dim() == 3
        assert proposals2.dim() == 4
        num_images = proposals2.shape[0]
        num_sequences = proposals2.shape[1]
        feat1 = [(f[0, ...] if f.dim() == 5 else f.reshape(-1,
            num_sequences, *f.shape[-3:])[0, ...]) for f in feat1]
        bb1 = bb1[0, ...]
        modulation = self.get_modulation(feat1, bb1)
        iou_feat = self.get_iou_feat(feat2)
        modulation = [f.reshape(1, num_sequences, -1).repeat(num_images, 1,
            1).reshape(num_sequences * num_images, -1) for f in modulation]
        proposals2 = proposals2.reshape(num_sequences * num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.reshape(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""
        fc34_3_r, fc34_4_r = modulation
        c3_t, c4_t = feat
        batch_size = c3_t.size()[0]
        c3_t_att = c3_t * fc34_3_r.reshape(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.reshape(batch_size, -1, 1, 1)
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(
            -1, 1).to(c3_t.device)
        num_proposals_per_batch = proposals.shape[1]
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0
            :2] + proposals[:, :, 2:4]), dim=2)
        roi2 = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1,
            num_proposals_per_batch, -1), proposals_xyxy), dim=2)
        roi2 = roi2.reshape(-1, 5).to(proposals_xyxy.device)
        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)
        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)
        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)
        iou_pred = self.iou_predictor(fc34_rt_cat).reshape(batch_size,
            num_proposals_per_batch)
        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (batch, 4)."""
        feat3_r, feat4_r = feat
        c3_r = self.conv3_1r(feat3_r)
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(
            -1, 1).to(bb.device)
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)
        roi3r = self.prroi_pool3r(c3_r, roi1)
        c4_r = self.conv4_1r(feat4_r)
        roi4r = self.prroi_pool4r(c4_r, roi1)
        fc3_r = self.fc3_1r(roi3r)
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)
        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)
        return fc34_3_r, fc34_4_r

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [(f.reshape(-1, *f.shape[-3:]) if f.dim() == 5 else f) for
            f in feat2]
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))
        return c3_t, c4_t


class MLU(nn.Module):
    """MLU activation
    """

    def __init__(self, min_val, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.inplace = inplace

    def forward(self, input):
        return F.elu(F.leaky_relu(input, 1 / self.min_val, inplace=self.
            inplace), self.min_val, inplace=self.inplace)


class LeakyReluPar(nn.Module):
    """LeakyRelu parametric activation
    """

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * torch.abs(x) + (1.0 + a) / 2.0 * x


class LeakyReluParDeriv(nn.Module):
    """Derivative of the LeakyRelu parametric activation, wrt x.
    """

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * torch.sign(x.detach()) + (1.0 + a) / 2.0


class BentIdentPar(nn.Module):
    """BentIdent parametric activation
    """

    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * (torch.sqrt(x * x + 4.0 * self.b * self.b) -
            2.0 * self.b) + (1.0 + a) / 2.0 * x


class BentIdentParDeriv(nn.Module):
    """BentIdent parametric activation deriv
    """

    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * (x / torch.sqrt(x * x + 4.0 * self.b * self.b)
            ) + (1.0 + a) / 2.0


class LinearBlock(nn.Module):

    def __init__(self, in_planes, out_planes, input_sz, bias=True,
        batch_norm=True, relu=True):
        super().__init__()
        self.linear = nn.Linear(in_planes * input_sz * input_sz, out_planes,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)


class DistanceMap(nn.Module):
    """Generate a distance map from a origin center location.
    args:
        num_bins:  Number of bins in the map.
        bin_displacement:  Displacement of the bins.
    """

    def __init__(self, num_bins, bin_displacement=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.bin_displacement = bin_displacement

    def forward(self, center, output_sz):
        """Create the distance map.
        args:
            center: Torch tensor with (y,x) center position. Dims (batch, 2)
            output_sz: Size of output distance map. 2-dimensional tuple."""
        center = center.view(-1, 2)
        bin_centers = torch.arange(self.num_bins, dtype=torch.float32,
            device=center.device).view(1, -1, 1, 1)
        k0 = torch.arange(output_sz[0], dtype=torch.float32, device=center.
            device).view(1, 1, -1, 1)
        k1 = torch.arange(output_sz[1], dtype=torch.float32, device=center.
            device).view(1, 1, 1, -1)
        d0 = k0 - center[:, (0)].view(-1, 1, 1, 1)
        d1 = k1 - center[:, (1)].view(-1, 1, 1, 1)
        dist = torch.sqrt(d0 * d0 + d1 * d1)
        bin_diff = dist / self.bin_displacement - bin_centers
        bin_val = torch.cat((F.relu(1.0 - torch.abs(bin_diff[:, :-1, :, :]),
            inplace=True), (1.0 + bin_diff[:, -1:, :, :]).clamp(0, 1)), dim=1)
        return bin_val


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """

    def __init__(self, size_average=True, eps=1e-05, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * (input.shape[1] * input.shape[2] *
                input.shape[3] / (torch.sum((input * input).view(input.
                shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())
        else:
            return input * (self.scale / (torch.sum((input * input).view(
                input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)
                .sqrt())


def interpolate(x, sz):
    """Interpolate 4D tensor x to size sz."""
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    return F.interpolate(x, sz, mode='bilinear', align_corners=False
        ) if x.shape[-2:] != sz else x


class InterpCat(nn.Module):
    """Interpolate and concatenate features of different resolutions."""

    def forward(self, input):
        if isinstance(input, (dict, OrderedDict)):
            input = list(input.values())
        output_shape = None
        for x in input:
            if output_shape is None or output_shape[0] > x.shape[-2]:
                output_shape = x.shape[-2:]
        return torch.cat([interpolate(x, output_shape) for x in input], dim=-3)


class KLRegression(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density, mc_dim=-1):
        """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""
        exp_val = scores - torch.log(sample_density + self.eps)
        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[
            mc_dim]) - torch.mean(scores * (gt_density / (sample_density +
            self.eps)), dim=mc_dim)
        return L.mean()


class MLRegression(nn.Module):
    """Maximum likelihood loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: probability density of the sample distribution
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""
        assert mc_dim == 1
        assert (sample_density[:, (0), (...)] == -1).all()
        exp_val = scores[:, 1:, (...)] - torch.log(sample_density[:, 1:, (
            ...)] + self.eps)
        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[
            mc_dim] - 1) - scores[:, (0), (...)]
        loss = L.mean()
        return loss


class KLRegressionGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""
        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)
        L = torch.logsumexp(scores, dim=grid_dim) + math.log(grid_scale
            ) - score_corr
        return L.mean()


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = 1.0 - negative_mask
        prediction = negative_mask * F.relu(prediction
            ) + positive_mask * prediction
        loss = self.error_metric(prediction, positive_mask * label)
        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.
                device))
        return loss


class TensorList(list):
    """Container mainly used for lists of torch tensors. Extends lists with pytorch functionality."""

    def __init__(self, list_of_tensors=None):
        if list_of_tensors is None:
            list_of_tensors = list()
        super(TensorList, self).__init__(list_of_tensors)

    def __deepcopy__(self, memodict={}):
        return TensorList(copy.deepcopy(list(self), memodict))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in
                item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 + e2) for e1, e2 in zip(self, other)])
        return TensorList([(e + other) for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 + e1) for e1, e2 in zip(self, other)])
        return TensorList([(other + e) for e in self])

    def __iadd__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 - e2) for e1, e2 in zip(self, other)])
        return TensorList([(e - other) for e in self])

    def __rsub__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 - e1) for e1, e2 in zip(self, other)])
        return TensorList([(other - e) for e in self])

    def __isub__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 * e2) for e1, e2 in zip(self, other)])
        return TensorList([(e * other) for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 * e1) for e1, e2 in zip(self, other)])
        return TensorList([(other * e) for e in self])

    def __imul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 / e2) for e1, e2 in zip(self, other)])
        return TensorList([(e / other) for e in self])

    def __rtruediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 / e1) for e1, e2 in zip(self, other)])
        return TensorList([(other / e) for e in self])

    def __itruediv__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 @ e2) for e1, e2 in zip(self, other)])
        return TensorList([(e @ other) for e in self])

    def __rmatmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 @ e1) for e1, e2 in zip(self, other)])
        return TensorList([(other @ e) for e in self])

    def __imatmul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] @= e2
        else:
            for i in range(len(self)):
                self[i] @= other
        return self

    def __mod__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 % e2) for e1, e2 in zip(self, other)])
        return TensorList([(e % other) for e in self])

    def __rmod__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 % e1) for e1, e2 in zip(self, other)])
        return TensorList([(other % e) for e in self])

    def __pos__(self):
        return TensorList([(+e) for e in self])

    def __neg__(self):
        return TensorList([(-e) for e in self])

    def __le__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 <= e2) for e1, e2 in zip(self, other)])
        return TensorList([(e <= other) for e in self])

    def __ge__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 >= e2) for e1, e2 in zip(self, other)])
        return TensorList([(e >= other) for e in self])

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorList) for t in self):
            return self
        new_list = TensorList()
        for t in self:
            if isinstance(t, TensorList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def list(self):
        return list(self)

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError("'TensorList' object has not attribute '{}'"
                .format(name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self]
                )
        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))


class GNSteepestDescent(nn.Module):
    """General module for steepest descent based meta learning."""

    def __init__(self, residual_module, num_iter=1, compute_losses=False,
        detach_length=float('Inf'), parameter_batch_dim=0,
        residual_batch_dim=0, steplength_reg=0.0, filter_dilation_factors=None
        ):
        super().__init__()
        self.residual_module = residual_module
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.steplength_reg = steplength_reg
        self._parameter_batch_dim = parameter_batch_dim
        self._residual_batch_dim = residual_batch_dim
        self.filter_dilation_factors = filter_dilation_factors

    def _sqr_norm(self, x: TensorList, batch_dim=0):
        sum_keep_batch_dim = lambda e: e.sum(dim=[d for d in range(e.dim()) if
            d != batch_dim])
        return sum((x * x).apply(sum_keep_batch_dim))

    def _compute_loss(self, res):
        return sum((res * res).sum()) / sum(res.numel())

    def forward(self, meta_parameter: TensorList, num_iter=None, *args, **
        kwargs):
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        num_iter = self.num_iter if num_iter is None else num_iter
        meta_parameter_iterates = [meta_parameter]
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()
            meta_parameter.requires_grad_(True)
            r = self.residual_module(meta_parameter,
                filter_dilation_factors=self.filter_dilation_factors, **kwargs)
            if self.compute_losses:
                losses.append(self._compute_loss(r))
            u = r.clone()
            g = TensorList(torch.autograd.grad(r, meta_parameter, u,
                create_graph=True))
            h = TensorList(torch.autograd.grad(g, u, g, create_graph=True))
            ip_gg = self._sqr_norm(g, batch_dim=self._parameter_batch_dim)
            ip_hh = self._sqr_norm(h, batch_dim=self._residual_batch_dim)
            alpha = ip_gg / (ip_hh + self.steplength_reg * ip_gg).clamp(1e-08)
            step = g.apply(lambda e: alpha.reshape([(-1 if d == self.
                _parameter_batch_dim else 1) for d in range(e.dim())]) * e)
            meta_parameter = meta_parameter - step
            meta_parameter_iterates.append(meta_parameter)
        if self.compute_losses:
            losses.append(self._compute_loss(self.residual_module(
                meta_parameter, filter_dilation_factors=self.
                filter_dilation_factors, **kwargs)))
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()
        return meta_parameter, meta_parameter_iterates, losses


class FilterPool(nn.Module):
    """Pool the target region in a feature map.
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(filter_size, filter_size, 1 /
            feature_stride)
        self.pool_square = pool_square

    def forward(self, feat, bb):
        """Pool the regions in bb.
        args:
            feat:  Input feature maps. Dims (num_samples, feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (num_samples, 4).
        returns:
            pooled_feat:  Pooled features. Dims (num_samples, feat_dim, wH, wW)."""
        bb = bb.reshape(-1, 4)
        num_images_total = bb.shape[0]
        batch_index = torch.arange(num_images_total, dtype=torch.float32
            ).reshape(-1, 1).to(bb.device)
        pool_bb = bb.clone()
        if self.pool_square:
            bb_sz = pool_bb[:, 2:4].prod(dim=1, keepdim=True).sqrt()
            pool_bb[:, :2] += pool_bb[:, 2:] / 2 - bb_sz / 2
            pool_bb[:, 2:] = bb_sz
        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        roi1 = torch.cat((batch_index, pool_bb), dim=1)
        return self.prroi_pool(feat, roi1)


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
    dilation=1, bias=True, batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'
    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FilterInitializer(nn.Module):
    """Initializes a target classification filter by applying a number of conv layers before and after pooling the target region.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end.
        num_filter_pre_convs:  Conv layers before pooling.
        num_filter_post_convs:  Conv layers after pooling."""

    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16,
        pool_square=False, filter_norm=True, num_filter_pre_convs=1,
        num_filter_post_convs=0):
        super().__init__()
        self.filter_pool = FilterPool(filter_size=filter_size,
            feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm
        pre_conv_layers = []
        for i in range(num_filter_pre_convs):
            pre_conv_layers.append(conv_block(feature_dim, feature_dim,
                kernel_size=3, padding=1))
        self.filter_pre_layers = nn.Sequential(*pre_conv_layers
            ) if pre_conv_layers else None
        post_conv_layers = []
        for i in range(num_filter_post_convs):
            post_conv_layers.append(conv_block(feature_dim, feature_dim,
                kernel_size=1, padding=0))
        post_conv_layers.append(nn.Conv2d(feature_dim, feature_dim,
            kernel_size=1, padding=0))
        self.filter_post_layers = nn.Sequential(*post_conv_layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_images = bb.shape[0] if bb.dim() == 3 else 1
        if self.filter_pre_layers is not None:
            feat = self.filter_pre_layers(feat.reshape(-1, feat.shape[-3],
                feat.shape[-2], feat.shape[-1]))
        feat_post = self.filter_pool(feat, bb)
        weights = self.filter_post_layers(feat_post)
        if num_images > 1:
            weights = torch.mean(weights.reshape(num_images, -1, weights.
                shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)
        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] *
                weights.shape[3])
        return weights


class FilterInitializerLinear(nn.Module):
    """Initializes a target classification filter by applying a linear conv layer and then pooling the target region.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end.
        conv_ksz:  Kernel size of the conv layer before pooling."""

    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16,
        pool_square=False, filter_norm=True, conv_ksz=3, init_weights='default'
        ):
        super().__init__()
        self.filter_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=
            conv_ksz, padding=conv_ksz // 2)
        self.filter_pool = FilterPool(filter_size=filter_size,
            feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_weights == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif init_weights == 'zero':
                    m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_images = feat.shape[0]
        feat = self.filter_conv(feat.reshape(-1, feat.shape[-3], feat.shape
            [-2], feat.shape[-1]))
        weights = self.filter_pool(feat, bb)
        if num_images > 1:
            weights = torch.mean(weights.reshape(num_images, -1, weights.
                shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)
        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] *
                weights.shape[3])
        return weights


class FilterInitializerZero(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, filter_size=1, feature_dim=256):
        super().__init__()
        self.filter_size = feature_dim, filter_size, filter_size

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        return feat.new_zeros(num_sequences, self.filter_size[0], self.
            filter_size[1], self.filter_size[2])


class FilterInitializerSiamese(nn.Module):
    """Initializes a target classification filter by only pooling the target region (similar to Siamese trackers).
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False,
        filter_norm=True):
        super().__init__()
        self.filter_pool = FilterPool(filter_size=filter_size,
            feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_images = feat.shape[0]
        feat = feat.reshape(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1])
        weights = self.filter_pool(feat, bb)
        if num_images > 1:
            weights = torch.mean(weights.reshape(num_images, -1, weights.
                shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)
        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] *
                weights.shape[3])
        return weights


class LinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=
        None, feature_extractor=None):
        super().__init__()
        self.filter_size = filter_size
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""
        assert train_bb.dim() == 3
        num_sequences = train_bb.shape[1]
        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])
        train_feat = self.extract_classification_feat(train_feat, num_sequences
            )
        test_feat = self.extract_classification_feat(test_feat, num_sequences)
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb,
            *args, **kwargs)
        test_scores = [self.classify(f, test_feat) for f in filter_iter]
        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""
        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        weights = self.filter_initializer(feat, bb)
        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights,
                *args, feat=feat, bb=bb, **kwargs)
        else:
            weights_iter = [weights]
            losses = None
        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]
        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:]
                )
        train_feat = self.extract_classification_feat(backbone_feat,
            num_sequences)
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:]
                )
        else:
            num_sequences = None
        test_feat = self.extract_classification_feat(backbone_feat,
            num_sequences)
        scores = filter_layer.apply_filter(test_feat, filter_weights)
        return scores


class DiMPSteepestDescentGN(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0,
        init_filter_reg=0.01, init_gauss_sigma=1.0, num_dist_bins=5,
        bin_displacement=1.0, mask_init_factor=4.0, score_act='relu',
        act_param=None, min_filter_reg=0.001, mask_act='sigmoid',
        detach_length=float('Inf'), alpha_eps=0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) *
            torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(1, -1,
            1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1 / 2 * (d / init_gauss_sigma) ** 2)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=
            1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0
            ].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1,
            kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(
                act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None,
        compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = weights.shape[-2], weights.shape[-1]
        output_sz = feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1
            ] + (weights.shape[-1] + 1) % 2
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.
            min_filter_reg ** 2)
        dmap_offset = torch.Tensor(filter_sz).to(bb.device) % 2 / 2.0
        center = ((bb[(...), :2] + bb[(...), 2:] / 2) / self.feat_stride
            ).reshape(-1, 2).flip((1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)
        label_map = self.label_map_predictor(dist_map).reshape(num_images,
            num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images,
            num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images,
                num_sequences, 1, 1) * spatial_weight
        backprop_through_learning = self.detach_length > 0
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if (not backprop_through_learning or i > 0 and i % self.
                detach_length == 0):
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)
            if compute_losses:
                losses.append(((residuals ** 2).sum() + reg_weight * (
                    weights ** 2).sum()) / num_sequences)
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat,
                residuals_mapped, filter_sz, training=self.training
                ) + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images,
                num_sequences, -1).sum(dim=(0, 2)) + (reg_weight + self.
                alpha_eps) * alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1
                ) * weights_grad
            weight_iterates.append(weights)
        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = self.score_activation(scores, target_mask)
            losses.append((((sample_weight * (scores - label_map)) ** 2).
                sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
        return weights, weight_iterates, losses


class DiMPL2SteepestDescentGN(nn.Module):
    """A simpler optimizer module that uses L2 loss.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        gauss_sigma:  The standard deviation of the label function.
        hinge_threshold:  Threshold for the hinge-based loss (see DiMP paper).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0,
        gauss_sigma=1.0, hinge_threshold=-999, init_filter_reg=0.01,
        min_filter_reg=0.001, detach_length=float('Inf'), alpha_eps=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) *
            torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.hinge_threshold = hinge_threshold
        self.gauss_sigma = gauss_sigma
        self.alpha_eps = alpha_eps

    def get_label(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, 
            -1, 1).to(center.device)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 
            1, -1).to(center.device)
        g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k0 - center[:,
            :, (0)].reshape(*center.shape[:2], 1, 1)) ** 2)
        g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k1 - center[:,
            :, (1)].reshape(*center.shape[:2], 1, 1)) ** 2)
        gauss = g0 * g1
        return gauss

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None,
        compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = weights.shape[-2], weights.shape[-1]
        output_sz = feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1
            ] + (weights.shape[-1] + 1) % 2
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.
            min_filter_reg ** 2)
        dmap_offset = torch.Tensor(filter_sz).to(bb.device) % 2 / 2.0
        center = ((bb[(...), :2] + bb[(...), 2:] / 2) / self.feat_stride).flip(
            (-1,)) - dmap_offset
        label_map = self.get_label(center, output_sz)
        target_mask = (label_map > self.hinge_threshold).float()
        label_map *= target_mask
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images,
                num_sequences, 1, 1)
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = target_mask * scores + (1.0 - target_mask) * F.relu(
                scores)
            score_mask = target_mask + (1.0 - target_mask) * (scores.detach
                () > 0).float()
            residuals = sample_weight * (scores_act - label_map)
            if compute_losses:
                losses.append(((residuals ** 2).sum() + reg_weight * (
                    weights ** 2).sum()) / num_sequences)
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat,
                residuals_mapped, filter_sz, training=self.training
                ) + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images,
                num_sequences, -1).sum(dim=(0, 2)) + (reg_weight + self.
                alpha_eps) * alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1
                ) * weights_grad
            weight_iterates.append(weights)
        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = target_mask * scores + (1.0 - target_mask) * F.relu(scores
                )
            losses.append((((sample_weight * (scores - label_map)) ** 2).
                sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
        return weights, weight_iterates, losses


class PrDiMPSteepestDescentNewton(nn.Module):
    """Optimizer module for PrDiMP.
    It unrolls the steepest descent with Newton iterations to optimize the target filter. See the PrDiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        gauss_sigma:  The standard deviation to use for the label density function.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
        init_uni_weight:  Weight of uniform label distribution.
        normalize_label:  Wheter to normalize the label distribution.
        label_shrink:  How much to shrink to label distribution.
        softmax_reg:  Regularization in the denominator of the SoftMax.
        label_threshold:  Threshold probabilities smaller than this.
    """

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0,
        init_filter_reg=0.01, gauss_sigma=1.0, min_filter_reg=0.001,
        detach_length=float('Inf'), alpha_eps=0.0, init_uni_weight=None,
        normalize_label=False, label_shrink=0, softmax_reg=None,
        label_threshold=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) *
            torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        self.uni_weight = 0 if init_uni_weight is None else init_uni_weight
        self.normalize_label = normalize_label
        self.label_shrink = label_shrink
        self.softmax_reg = softmax_reg
        self.label_threshold = label_threshold

    def get_label_density(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, 
            -1, 1).to(center.device)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 
            1, -1).to(center.device)
        dist0 = (k0 - center[:, :, (0)].reshape(*center.shape[:2], 1, 1)) ** 2
        dist1 = (k1 - center[:, :, (1)].reshape(*center.shape[:2], 1, 1)) ** 2
        if self.gauss_sigma == 0:
            dist0_view = dist0.reshape(-1, dist0.shape[-2])
            dist1_view = dist1.reshape(-1, dist1.shape[-1])
            one_hot0 = torch.zeros_like(dist0_view)
            one_hot1 = torch.zeros_like(dist1_view)
            one_hot0[torch.arange(one_hot0.shape[0]), dist0_view.argmin(dim=-1)
                ] = 1.0
            one_hot1[torch.arange(one_hot1.shape[0]), dist1_view.argmin(dim=-1)
                ] = 1.0
            gauss = one_hot0.reshape(dist0.shape) * one_hot1.reshape(dist1.
                shape)
        else:
            g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist0)
            g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist1)
            gauss = g0 / (2 * math.pi * self.gauss_sigma ** 2) * g1
        gauss = gauss * (gauss > self.label_threshold).float()
        if self.normalize_label:
            gauss /= gauss.sum(dim=(-2, -1), keepdim=True) + 1e-08
        label_dens = (1.0 - self.label_shrink) * ((1.0 - self.uni_weight) *
            gauss + self.uni_weight / (output_sz[0] * output_sz[1]))
        return label_dens

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None,
        compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = weights.shape[-2], weights.shape[-1]
        output_sz = feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1
            ] + (weights.shape[-1] + 1) % 2
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.
            min_filter_reg ** 2)
        offset = torch.Tensor(filter_sz).to(bb.device) % 2 / 2.0
        center = ((bb[(...), :2] + bb[(...), 2:] / 2) / self.feat_stride).flip(
            (-1,)) - offset
        label_density = self.get_label_density(center, output_sz)
        if sample_weight is None:
            sample_weight = torch.Tensor([1.0 / num_images]).to(feat.device)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.reshape(num_images, num_sequences,
                1, 1)
        exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)

        def _compute_loss(scores, weights):
            return torch.sum(sample_weight.reshape(sample_weight.shape[0], 
                -1) * (torch.log(scores.exp().sum(dim=(-2, -1)) + exp_reg) -
                (label_density * scores).sum(dim=(-2, -1)))
                ) / num_sequences + reg_weight * (weights ** 2).sum(
                ) / num_sequences
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_softmax = activation.softmax_reg(scores.reshape(
                num_images, num_sequences, -1), dim=2, reg=self.softmax_reg
                ).reshape(scores.shape)
            res = sample_weight * (scores_softmax - label_density)
            if compute_losses:
                losses.append(_compute_loss(scores, weights))
            weights_grad = filter_layer.apply_feat_transpose(feat, res,
                filter_sz, training=self.training) + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            sm_scores_grad = scores_softmax * scores_grad
            hes_scores_grad = sm_scores_grad - scores_softmax * torch.sum(
                sm_scores_grad, dim=(-2, -1), keepdim=True)
            grad_hes_grad = (scores_grad * hes_scores_grad).reshape(num_images,
                num_sequences, -1).sum(dim=2).clamp(min=0)
            grad_hes_grad = (sample_weight.reshape(sample_weight.shape[0], 
                -1) * grad_hes_grad).sum(dim=0)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = (grad_hes_grad + (reg_weight + self.alpha_eps) *
                alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1
                ) * weights_grad
            weight_iterates.append(weights)
        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            losses.append(_compute_loss(scores, weights))
        return weights, weight_iterates, losses


class LinearFilterLearnGen(nn.Module):

    def __init__(self, feat_stride=16, init_filter_reg=0.01,
        init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
        mask_init_factor=4.0, score_act='bentpar', act_param=None, mask_act
        ='sigmoid'):
        super().__init__()
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.feat_stride = feat_stride
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(1, -1,
            1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1 / 2 * (d / init_gauss_sigma) ** 2)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=
            1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0
            ].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1,
            kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
        else:
            raise ValueError('Unknown activation')

    def forward(self, meta_parameter: TensorList, feat, bb, sample_weight=
        None, is_distractor=None):
        filter = meta_parameter[0]
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = filter.shape[-2], filter.shape[-1]
        scores = filter_layer.apply_filter(feat, filter)
        center = ((bb[(...), :2] + bb[(...), 2:] / 2) / self.feat_stride
            ).reshape(-1, 2).flip((1,))
        if is_distractor is not None:
            center[(is_distractor.reshape(-1)), :] = 99999
        dist_map = self.distance_map(center, scores.shape[-2:])
        label_map = self.label_map_predictor(dist_map).reshape(num_images,
            num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images,
            num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(-1, 1, 1, 1
                ) * spatial_weight
        scores_act = self.score_activation(scores, target_mask)
        data_residual = sample_weight * (scores_act - label_map)
        reg_residual = self.filter_reg * filter.reshape(1, num_sequences, -1)
        return TensorList([data_residual, reg_residual])


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor,
        classification_layer, bb_regressor_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(
            classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer +
            self.bb_regressor_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *
        args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""
        assert train_imgs.dim() == 5 and test_imgs.dim(
            ) == 5, 'Expect 5 dimensional inputs'
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1,
            *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *
            test_imgs.shape[-3:]))
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)
        target_scores = self.classifier(train_feat_clf, test_feat_clf,
            train_bb, *args, **kwargs)
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
            train_bb, test_proposals)
        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.
            classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.
            get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.
            classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_visionml_pytracking(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BentIdentPar(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BentIdentParDeriv(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FilterInitializerZero(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(InstanceL2Norm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(InterpCat(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(KLRegression(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(KLRegressionGrid(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(LBHinge(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(LeakyReluPar(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(LeakyReluParDeriv(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(LinearBlock(*[], **{'in_planes': 4, 'out_planes': 4, 'input_sz': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(MLU(*[], **{'min_val': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(SpatialCrossMapLRN(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

