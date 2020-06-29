import sys
_module = sys.modules[__name__]
del sys
locator = _module
__main__ = _module
argparser = _module
bmm = _module
data = _module
data_plant_stuff = _module
find_lr = _module
get_image_size = _module
locate = _module
logger = _module
losses = _module
make_metric_plots = _module
metrics = _module
metrics_from_results = _module
models = _module
unet_model = _module
unet_parts = _module
utils = _module
paint = _module
train = _module
generate_csv = _module
parseResults = _module
spacing_stats_to_csv = _module
setup = _module

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


import math


import torch


import numpy as np


from torch.nn import functional as F


import time


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import warnings


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances


class AveragedHausdorffLoss(nn.Module):

    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()
        assert set1.size()[1] == set2.size()[1
            ], 'The points in both sets must have the same number of dimensions, got %s and %s.' % (
            set2.size()[1], set2.size()[1])
        d2_matrix = cdist(set1, set2)
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])
        res = term_1 + term_2
        return res


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, "nn criterions don't compute the gradient w.r.t. targets - please mark these variables as volatile or not requiring gradients"


def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res = torch.mean((tensor + 1e-06) ** p, dim, keepdim=keepdim) ** (1.0 / p)
    return res


class WeightedHausdorffDistance(nn.Module):

    def __init__(self, resized_height, resized_width, p=-9, return_2_terms=
        False, device=torch.device('cpu')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(nn.Module, self).__init__()
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height, resized_width],
            dtype=torch.get_default_dtype(), device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(
            resized_height), np.arange(resized_width)]))
        self.all_img_locations = self.all_img_locations
        self.return_2_terms = return_2_terms
        self.p = p

    def forward(self, prob_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :param orig_widths: List of the original widths for each image
                            in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """
        _assert_no_grad(gt)
        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width
            ), 'You must configure the WeightedHausdorffDistance with the height and width of the probability map that you are using, got a probability map of size %s' % str(
            prob_map.size())
        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)
        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            prob_map_b = prob_map[(b), :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[(b), :]
            norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0], dtype=torch.
                    get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist], dtype=torch.
                    get_default_dtype()))
                continue
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1
                ) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = cdist(normalized_x, normalized_y)
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
            term_1 = 1 / (n_est_pts + 1e-06) * torch.sum(p * torch.min(
                d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated
                ) * self.max_dist + p_replicated * d_matrix
            minn = generaliz_mean(weighted_d_matrix, p=self.p, dim=0,
                keepdim=False)
            term_2 = torch.mean(minn)
            terms_1.append(term_1)
            terms_2.append(term_2)
        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)
        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()
        return res


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, height, width, known_n_points
        =None, ultrasmall=False, device=torch.device('cuda')):
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 3 for RGB)
        :param n_classes: Number of output classes
        :param height: Height of the input images
        :param known_n_points: If you know the number of points,
                               (e.g, one pupil), then set it.
                               Otherwise it will be estimated by a lateral NN.
                               If provided, no lateral network will be build
                               and the resulting UNet will be a FCN.
        :param ultrasmall: If True, the 5 central layers are removed,
                           resulting in a much smaller UNet.
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super(UNet, self).__init__()
        self.ultrasmall = ultrasmall
        self.device = device
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'
                .format(height, width))
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        if self.ultrasmall:
            self.down3 = down(256, 512, normaliz=False)
            self.up1 = up(768, 128)
            self.up2 = up(256, 64)
            self.up3 = up(128, 64, activ=False)
        else:
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)
            self.down5 = down(512, 512)
            self.down6 = down(512, 512)
            self.down7 = down(512, 512)
            self.down8 = down(512, 512, normaliz=False)
            self.up1 = up(1024, 512)
            self.up2 = up(1024, 512)
            self.up3 = up(1024, 512)
            self.up4 = up(1024, 512)
            self.up5 = up(1024, 256)
            self.up6 = up(512, 128)
            self.up7 = up(256, 64)
            self.up8 = up(128, 64, activ=False)
        self.outc = outconv(64, n_classes)
        self.out_nonlin = nn.Sigmoid()
        self.known_n_points = known_n_points
        if known_n_points is None:
            steps = 3 if self.ultrasmall else 8
            height_mid_features = height // 2 ** steps
            width_mid_features = width // 2 ** steps
            self.branch_1 = nn.Sequential(nn.Linear(height_mid_features *
                width_mid_features * 512, 64), nn.ReLU(inplace=True), nn.
                Dropout(p=0.5))
            self.branch_2 = nn.Sequential(nn.Linear(height * width, 64), nn
                .ReLU(inplace=True), nn.Dropout(p=0.5))
            self.regressor = nn.Sequential(nn.Linear(64 + 64, 1), nn.ReLU())
        self.lin = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.ultrasmall:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        else:
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x8 = self.down7(x7)
            x9 = self.down8(x8)
            x = self.up1(x9, x8)
            x = self.up2(x, x7)
            x = self.up3(x, x6)
            x = self.up4(x, x5)
            x = self.up5(x, x4)
            x = self.up6(x, x3)
            x = self.up7(x, x2)
            x = self.up8(x, x1)
        x = self.outc(x)
        x = self.out_nonlin(x)
        x = x.squeeze(1)
        if self.known_n_points is None:
            middle_layer = x4 if self.ultrasmall else x9
            middle_layer_flat = middle_layer.view(batch_size, -1)
            x_flat = x.view(batch_size, -1)
            lateral_flat = self.branch_1(middle_layer_flat)
            x_flat = self.branch_2(x_flat)
            regression_features = torch.cat((x_flat, lateral_flat), dim=1)
            regression = self.regressor(regression_features)
            return x, regression
        else:
            n_pts = torch.tensor([self.known_n_points] * batch_size, dtype=
                torch.get_default_dtype())
            n_pts = n_pts
            return x, n_pts


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch, normaliz=True, activ=True):
        super(double_conv, self).__init__()
        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]
        ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch, normaliz=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch,
            out_ch, normaliz=normaliz))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, normaliz=True, activ=True):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.conv = double_conv(in_ch, out_ch, normaliz=normaliz, activ=activ)

    def forward(self, x1, x2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2)), diffY // 2,
            int(math.ceil(diffY / 2))))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_javiribera_locating_objects_without_bboxes(_paritybench_base):
    pass
    def test_000(self):
        self._check(double_conv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(down(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(inconv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(outconv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(up(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

