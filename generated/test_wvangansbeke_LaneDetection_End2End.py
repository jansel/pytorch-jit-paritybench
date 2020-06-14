import sys
_module = sys.modules[__name__]
del sys
Load_Data_new = _module
Loss_crit = _module
ERFNet = _module
LSQ_layer = _module
Least_squares_net = _module
Networks = _module
gels = _module
utils = _module
eval_lane = _module
main = _module
test = _module
Loss_crit = _module
ERFNet = _module
LSQ_layer = _module
utils = _module
main = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import torch.nn.init as init


import torch.optim


from math import ceil


from torch.nn import init


from torch.optim import lr_scheduler


from scipy.optimize import fsolve


class CrossEntropyLoss2d(nn.Module):
    """
    Standard 2d cross entropy loss on all pixels of image
    My implemetation (but since Pytorch 0.2.0 libs have their
    owm optimized implementation, consider using theirs)
    """

    def __init__(self, weight=None, size_average=True, seg=False, nclasses=2):
        if seg:
            weights = torch.Tensor([1] + [weight] * nclasses)
            weights = weights
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weights, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets[:, (0),
            :, :])


class Area_Loss(nn.Module):
    """
    Compute area between curves by integrating (x1 - x2)^2 over y
    *Area:
        *order 0: int((c1 - c2)**2)dy
        *order 1: int((b1*y - b2*y + c1 - c2)**2)dy
        *order 2: int((a1*y**2 - a2*y**2 + b1*y - b2*y + c1 - c2)**2)dy

    *A weight function W can be added:
        Weighted area: int(W(y)*diff**2)dy
        with W(y):
            *1
            *(1-y)
            *(1-y**0.5)
    """

    def __init__(self, order, weight_funct):
        super(Area_Loss, self).__init__()
        self.order = order
        self.weight_funct = weight_funct

    def forward(self, params, gt_params, compute=True):
        diff = params.squeeze(-1) - gt_params
        a = diff[:, (0)]
        b = diff[:, (1)]
        t = 0.7
        if self.order == 2:
            c = diff[:, (2)]
            if self.weight_funct == 'none':
                loss_fit = a ** 2 * t ** 5 / 5 + 2 * a * b * t ** 4 / 4 + (
                    b ** 2 + c * 2 * a
                    ) * t ** 3 / 3 + 2 * b * c * t ** 2 / 2 + c ** 2 * t
            elif self.weight_funct == 'linear':
                loss_fit = c ** 2 * t - t ** 5 * (2 * a * b / 5 - a ** 2 / 5
                    ) + t ** 2 * (b * c - c ** 2 / 2
                    ) - a ** 2 * t ** 6 / 6 - t ** 4 * (b ** 2 / 4 - a * b /
                    2 + a * c / 2) + t ** 3 * (b ** 2 / 3 - 2 * c * b / 3 +
                    2 * a * c / 3)
            elif self.weight_funct == 'quadratic':
                loss_fit = (t ** 3 * (1 / 3 * b ** 2 + 2 / 3 * a * c) - t **
                    (7 / 2) * (2 / 7 * b ** 2 + 4 / 7 * a * c) + c ** 2 * t +
                    0.2 * a ** 2 * t ** 5 - 2 / 11 * a ** 2 * t ** (11 / 2) -
                    2 / 3 * c ** 2 * t ** (3 / 2) + 0.5 * a * b * t ** 4 - 
                    4 / 9 * a * b * t ** (9 / 2) + b * c * t ** 2 - 0.8 * b *
                    c * t ** (5 / 2))
            else:
                return NotImplementedError(
                    'The requested weight function is                         not implemented, only order 1 or order 2 possible'
                    )
        elif self.order == 1:
            loss_fit = b ** 2 * t + a * b * t ** 2 + a ** 2 * t ** 3 / 3
        else:
            return NotImplementedError(
                'The requested order is not implemented, only none, linear or quadratic possible'
                )
        mask = torch.prod(gt_params != 0, 1).byte()
        loss_fit = torch.masked_select(loss_fit, mask)
        loss_fit = loss_fit.mean(0) if loss_fit.size()[0] != 0 else 0
        return loss_fit


class MSE_Loss(nn.Module):
    """
    Compute mean square error loss on curve parameters
    in ortho or normal view
    """

    def __init__(self, options):
        super(MSE_Loss, self).__init__()
        self.loss_crit = nn.MSELoss()
        if not options.no_cuda:
            self.loss_crit = self.loss_crit

    def forward(self, params, gt_params, compute=True):
        loss = self.loss_crit(params.squeeze(-1), gt_params)
        return loss


def get_homography(resize=256, no_mapping=False):
    if no_mapping:
        M = np.identity(3)
        M_inv = np.identity(3)
    else:
        y_start = 0.2 * resize
        y_stop = resize - 1
        src = np.float32([[0.45 * (2 * resize), y_start], [0.55 * (2 *
            resize), y_start], [0.02 * (2 * resize), y_stop], [0.97 * (2 *
            resize), y_stop]])
        dst = np.float32([[0.45 * (2 * resize), y_start], [0.55 * (2 *
            resize), y_start], [0.45 * (2 * resize), y_stop], [0.55 * (2 *
            resize), y_stop]])
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


class backprojection_loss(nn.Module):
    """
    Compute mean square error loss on points in normal view
    instead of parameters in ortho view
    """

    def __init__(self, options):
        super(backprojection_loss, self).__init__()
        M, M_inv = get_homography(options.resize, options.no_mapping)
        self.M, self.M_inv = torch.from_numpy(M), torch.from_numpy(M_inv)
        start = 160
        delta = 10
        num_heights = (720 - start) // delta
        self.y_d = (torch.arange(start, 720, delta) - 80).double() / 2.5
        self.ones = torch.ones(num_heights).double()
        self.y_prime = (self.M[(1), 1:2] * self.y_d + self.M[(1), 2:]) / (
            self.M[(2), 1:2] * self.y_d + self.M[(2), 2:])
        self.y_eval = 255 - self.y_prime
        if options.order == 0:
            self.Y = self.tensor_ones
        elif options.order == 1:
            self.Y = torch.stack((self.y_eval, self.ones), 1)
        elif options.order == 2:
            self.Y = torch.stack((self.y_eval ** 2, self.y_eval, self.ones), 1)
        elif options.order == 3:
            self.Y = torch.stack((self.y_eval ** 3, self.y_eval ** 2, self.
                y_eval, self.ones), 1)
        else:
            raise NotImplementedError(
                'Requested order {} for polynomial fit is not implemented'.
                format(options.order))
        self.Y = self.Y.unsqueeze(0).repeat(options.batch_size, 1, 1)
        self.ones = torch.ones(options.batch_size, num_heights, 1).double()
        self.y_prime = self.y_prime.unsqueeze(0).repeat(options.batch_size, 1
            ).unsqueeze(2)
        self.M_inv = self.M_inv.unsqueeze(0).repeat(options.batch_size, 1, 1)
        if not options.no_cuda:
            self.M = self.M
            self.M_inv = self.M_inv
            self.y_prime = self.y_prime
            self.Y = self.Y
            self.ones = self.ones

    def forward(self, params, x_gt, valid_samples):
        bs = params.size(0)
        x_prime = torch.bmm(self.Y[:bs], params)
        coordinates = torch.stack((x_prime, self.y_prime[:bs], self.ones[:
            bs]), 2).squeeze(3).permute((0, 2, 1))
        trans = torch.bmm(self.M_inv[:bs], coordinates)
        x_cal = trans[:, (0), :] / trans[:, (2), :]
        x_err = (x_gt - x_cal) * valid_samples
        loss = torch.sum(x_err ** 2) / valid_samples.sum()
        if valid_samples.sum() == 0:
            loss = 0
        return loss, x_cal * valid_samples


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2,
            padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding
            =0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2,
            padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):

    def __init__(self, num_classes, pretrain, do_segmentation=False):
        super().__init__()
        self.pretrain = pretrain
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2,
            padding=0, output_padding=0, bias=True)
        if pretrain:
            self.output_conv2 = nn.ConvTranspose2d(16, num_classes + 1, 2,
                stride=2, padding=0, output_padding=0, bias=True)
        self.do_segmentation = do_segmentation
        if do_segmentation:
            self.layers1 = nn.ModuleList()
            self.layers1.append(UpsamplerBlock(128, 64))
            self.layers1.append(non_bottleneck_1d(64, 0, 1))
            self.layers1.append(non_bottleneck_1d(64, 0, 1))
            self.layers1.append(UpsamplerBlock(64, 16))
            self.layers1.append(non_bottleneck_1d(16, 0, 1))
            self.layers1.append(non_bottleneck_1d(16, 0, 1))
            output_conv3 = nn.ConvTranspose2d(16, num_classes + 1, 2,
                stride=2, padding=0, output_padding=0, bias=True)
            self.layers1.append(output_conv3)

    def forward(self, input, flag):
        output = input
        output_seg = input
        for layer in self.layers:
            output = layer(output)
        if self.pretrain:
            if flag:
                output = self.output_conv(output)
            else:
                output = self.output_conv2(output)
        else:
            output = self.output_conv(output)
        if self.do_segmentation:
            for layer1 in self.layers1:
                output_seg = layer1(output_seg)
        return output, output_seg


class Net(nn.Module):

    def __init__(self, layers=18, in_channels=1, out_channels=1, pretrained
        =False, pool=False):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels, pretrained)

    def forward(self, input, flag, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            encoder_output = self.encoder(input)
            decoder_output, output_seg = self.decoder.forward(encoder_output,
                flag)
            return encoder_output, decoder_output, output_seg


class GELS(Function):

    @staticmethod
    def forward(ctx, A, b):
        u = torch.cholesky(torch.matmul(A.transpose(-1, -2), A), upper=True)
        ret = torch.cholesky_solve(torch.matmul(A.transpose(-1, -2), b), u,
            upper=True)
        ctx.save_for_backward(u, ret, A, b)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        chol, x, a, b = ctx.saved_tensors
        z = torch.cholesky_solve(grad_output, chol, upper=True)
        xzt = torch.matmul(x, z.transpose(-1, -2))
        zx_sym = xzt + xzt.transpose(-1, -2)
        grad_A = -torch.matmul(a, zx_sym) + torch.matmul(b, z.transpose(-1, -2)
            )
        grad_b = torch.matmul(a, z)
        return grad_A, grad_b


class Weighted_least_squares(nn.Module):

    def __init__(self, size, nclasses, order, no_cuda, reg_ls=0,
        use_cholesky=False):
        super().__init__()
        N, C, self.H, W = size
        self.nclasses = nclasses
        self.tensor_ones = torch.ones(N, self.H * W, 1).float()
        self.order = order
        self.reg_ls = reg_ls * torch.eye(order + 1)
        self.use_cholesky = use_cholesky
        if not no_cuda:
            self.reg_ls = self.reg_ls
            self.tensor_ones = self.tensor_ones

    def forward(self, W, grid):
        beta2, beta3 = None, None
        W = W.view(-1, self.nclasses, grid.size(1))
        bs = W.size(0)
        grid = grid[:bs]
        tensor_ones = self.tensor_ones[:bs]
        x_map = grid[:, :, (0)].unsqueeze(2)
        y_map = (255 - grid[:, :, (1)]).unsqueeze(2)
        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((y_map, tensor_ones), 2)
        elif self.order == 2:
            Y = torch.cat((y_map ** 2, y_map, tensor_ones), 2)
        elif self.order == 3:
            Y = torch.cat((y_map ** 3, y_map ** 2, y_map, tensor_ones), 2)
        else:
            raise NotImplementedError(
                'Requested order {} for polynomial fit is not implemented'.
                format(self.order))
        W0 = W[:, (0), :].unsqueeze(2)
        Y0 = torch.mul(W0, Y)
        if not self.use_cholesky:
            Z = torch.bmm(Y0.transpose(1, 2), Y0) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            beta0 = GELS.apply(Y0, torch.mul(W0, x_map))
        W1 = W[:, (1), :].unsqueeze(2)
        Y1 = torch.mul(W1, Y)
        if not self.use_cholesky:
            Z = torch.bmm(Y1.transpose(1, 2), Y1) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y1.transpose(1, 2), torch.mul(W1, x_map))
            beta1 = torch.bmm(Z_inv, X)
        else:
            beta1 = GELS.apply(Y1, torch.mul(W1, x_map))
        if self.nclasses > 3:
            W2 = W[:, (2), :].unsqueeze(2)
            Y2 = torch.mul(W2, Y)
            if not self.use_cholesky:
                Z = torch.bmm(Y2.transpose(1, 2), Y2) + self.reg_ls
                Z_inv = torch.inverse(Z)
                X = torch.bmm(Y2.transpose(1, 2), torch.mul(W2, x_map))
                beta2 = torch.bmm(Z_inv, X)
            else:
                beta2 = GELS.apply(Y2, torch.mul(W2, x_map))
            beta2 = beta2.double()
            W3 = W[:, (3), :].unsqueeze(2)
            Y3 = torch.mul(W3, Y)
            if not self.use_cholesky:
                Z = torch.bmm(Y3.transpose(1, 2), Y3) + self.reg_ls
                Z_inv = torch.inverse(Z)
                X = torch.bmm(Y3.transpose(1, 2), torch.mul(W3, x_map))
                beta3 = torch.bmm(Z_inv, X)
            else:
                beta3 = GELS.apply(Y3, torch.mul(W3, x_map))
            beta3 = beta3.double()
        return beta0.double(), beta1.double(), beta2, beta3


class Classification(nn.Module):

    def __init__(self, class_type, size, channels_in, resize):
        super().__init__()
        self.class_type = class_type
        filter_size = 1
        pad = (filter_size - 1) // 2
        self.conv1 = nn.Conv2d(channels_in, 128, filter_size, stride=1,
            padding=pad, bias=True)
        self.conv1_bn = nn.BatchNorm2d(128)
        filter_size = 3
        pad = (filter_size - 1) // 2
        self.conv2 = nn.Conv2d(128, 128, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        rows, cols = size
        self.avgpool = nn.AvgPool2d((1, cols))
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        if class_type == 'line':
            self.fully_connected1 = nn.Linear(64 * rows * cols // 4, 128)
            self.fully_connected_line1 = nn.Linear(128, 4)
        else:
            self.fully_connected_horizon = nn.Linear(64 * rows, resize)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        if self.class_type == 'line':
            x = self.maxpool(x)
        else:
            x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        batch_size = x.size(0)
        if self.class_type == 'line':
            x = F.relu(self.fully_connected1(x))
            x = self.fully_connected_line1(x)
        else:
            x = self.fully_connected_horizon(x)
        return x


def ProjectiveGridGenerator(size, theta, no_cuda):
    N, C, H, W = size
    linear_points_W = torch.linspace(0, W - 1, W)
    linear_points_H = torch.linspace(0, H - 1, H)
    base_grid = theta.new(N, H, W, 3)
    base_grid[:, :, :, (0)] = torch.ger(torch.ones(H), linear_points_W
        ).expand_as(base_grid[:, :, :, (0)])
    base_grid[:, :, :, (1)] = torch.ger(linear_points_H, torch.ones(W)
        ).expand_as(base_grid[:, :, :, (1)])
    base_grid[:, :, :, (2)] = 1
    grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
    grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
    if not no_cuda:
        grid = grid.cuda()
    return grid


def return_tensor(x):
    return x


def square_tensor(x):
    return x ** 2


def activation_layer(activation='square', no_cuda=False):
    place_cuda = True
    if activation == 'sigmoid':
        layer = nn.Sigmoid()
    elif activation == 'relu':
        layer = nn.ReLU()
    elif activation == 'softplus':
        layer = nn.Softplus()
    elif activation == 'square':
        layer = square_tensor
        place_cuda = False
    elif activation == 'abs':
        layer = torch.abs
        place_cuda = False
    elif activation == 'none':
        layer = return_tensor
    else:
        raise NotImplementedError('Activation type: {} is not implemented'.
            format(activation))
    if not no_cuda and place_cuda:
        layer = layer.cuda()
    return layer


class Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.nclasses = args.nclasses
        resize = args.resize
        size = torch.Size([args.batch_size, args.nclasses, args.resize, 2 *
            args.resize])
        M, _ = get_homography(args.resize, args.no_mapping)
        M = torch.from_numpy(M).unsqueeze_(0).expand([args.batch_size, 3, 3]
            ).float()
        out_channels = args.nclasses + int(not args.end_to_end)
        self.net = Networks.define_model(mod=args.mod, layers=args.layers,
            in_channels=args.channels_in, out_channels=out_channels,
            pretrained=args.pretrained, pool=args.pool)
        self.activation = activation_layer(args.activation_layer, args.no_cuda)
        self.grid = ProjectiveGridGenerator(size, M, args.no_cuda)
        self.ls_layer = Weighted_least_squares(size, args.nclasses, args.
            order, args.no_cuda, args.reg_ls, args.use_cholesky)
        zero_rows = ceil(args.resize * args.mask_percentage)
        self.idx_row = torch.linspace(0, zero_rows - 1, zero_rows).long()
        n_row = 13
        self.idx_col1 = Variable(torch.linspace(1, n_row, n_row + 1).long())
        self.idx_col2 = Variable(torch.linspace(0, n_row, n_row + 1).long()
            ) + 2 * resize - (n_row + 1)
        idx_mask = (np.arange(resize)[:, (None)] < np.arange(2 * resize) -
            (resize + 10)) * 1
        idx_mask = np.flip(idx_mask, 1).copy() + idx_mask
        self.idx_mask = Variable(torch.from_numpy(idx_mask)).type(torch.
            ByteTensor).expand(args.batch_size, args.nclasses, resize, 2 *
            resize)
        self.end_to_end = args.end_to_end
        self.pretrained = args.pretrained
        self.classification_branch = args.clas
        if self.classification_branch:
            size_enc = 32, 64
            chan = 128
            self.line_classification = Classification('line', size=size_enc,
                channels_in=chan, resize=resize)
            self.horizon_estimation = Classification('horizon', size=
                size_enc, channels_in=chan, resize=resize)
        if not args.no_cuda:
            self.idx_row = self.idx_row
            self.idx_col1 = self.idx_col1
            self.idx_col2 = self.idx_col2
            self.idx_mask = self.idx_mask
            if self.classification_branch:
                self.line_classification = self.line_classification
                self.horizon_estimation = self.horizon_estimation

    def forward(self, input, gt_line, end_to_end, early_return=False, gt=None):
        line, horizon = None, None
        shared_encoder, output, output_seg = self.net(input, end_to_end *
            self.pretrained)
        if early_return:
            return output
        if not end_to_end:
            activated = output.detach()
            _, activated = torch.max(activated, 1)
            activated = activated.float()
            if self.nclasses < 3:
                left = activated * (activated == 1).float()
                right = activated * (activated == 2).float()
                activated = torch.stack((left, right), 1)
            else:
                left1 = activated * (activated == 1).float()
                right1 = activated * (activated == 2).float()
                left2 = activated * (activated == 3).float()
                right2 = activated * (activated == 4).float()
                activated = torch.stack((left1, right1, left2, right2), 1)
        else:
            activated = self.activation(output)
            if self.classification_branch:
                line = self.line_classification(shared_encoder)
                horizon = self.horizon_estimation(shared_encoder)
        masked = activated.index_fill(2, self.idx_row, 0)
        if gt_line.sum() != 0 and end_to_end == False:
            gt_mask = gt_line[:, :, (None), (None)].byte().expand_as(masked)
            masked[gt_mask] = masked[0, 0].unsqueeze(0).repeat(gt_line.sum(
                ).item(), 1, 1).view(-1)
        beta0, beta1, beta2, beta3 = self.ls_layer(masked, self.grid)
        return (beta0, beta1, beta2, beta3, masked, output, line, horizon,
            output_seg)


class resnet_block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, encode=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
            padding=dilation, dilation=dilation, bias=False)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride
            =1, padding=0, bias=False)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x = x + shortcut
        return x


class simple_net(nn.Module):

    def __init__(self, nclasses):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.resnet_block1 = resnet_block(32, 32, 1)
        self.pool = nn.MaxPool2d(2, (2, 2))
        self.resnet_block2 = resnet_block(32, 64, 1)
        self.resnet_block3 = resnet_block(64, 128, 2)
        self.resnet_block4 = resnet_block(128, 256, 4)
        self.resnet_block5 = resnet_block(256, 128, 2)
        self.resnet_block6 = resnet_block(128, 64, 1)
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.resnet_block7 = resnet_block(64, 32, 1)
        self.conv_out = nn.Conv2d(32, nclasses, 1, stride=1, padding=0,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        output_encoder = x
        x = self.resnet_block5(x)
        x = self.resnet_block6(x)
        x = self.resnet_block7(x)
        x = self.conv_out(x)
        return x, output_encoder


class Classification_old(nn.Module):

    def __init__(self, class_type, size, resize=320):
        super().__init__()
        self.class_type = class_type
        self.conv1 = nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=True)
        self.conv5_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)
        rows, cols = size
        self.fully_connected1 = nn.Linear(32 * rows * cols // 4, 1024)
        if class_type == 'line':
            self.fully_connected2 = nn.Linear(1024, 128)
            self.fully_connected_line1 = nn.Linear(128, 3)
            self.fully_connected_line2 = nn.Linear(128, 3)
            self.fully_connected_line3 = nn.Linear(128, 3)
            self.fully_connected_line4 = nn.Linear(128, 3)
        else:
            self.fully_connected_horizon = nn.Linear(1024, resize)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.pool1(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fully_connected1(x))
        batch_size = x.size(0)
        if self.class_type == 'line':
            x = F.relu(self.fully_connected2(x))
            x1 = self.fully_connected_line1(x).view(batch_size, 3, 1, 1)
            x2 = self.fully_connected_line2(x).view(batch_size, 3, 1, 1)
            x3 = self.fully_connected_line3(x).view(batch_size, 3, 1, 1)
            x4 = self.fully_connected_line4(x).view(batch_size, 3, 1, 1)
            x = torch.cat((x1, x2, x3, x4), 2)
        else:
            x = self.fully_connected_horizon(x)
        return x


class Classification(nn.Module):

    def __init__(self, class_type, size, channels_in, resize):
        super().__init__()
        self.class_type = class_type
        filter_size = 1
        pad = (filter_size - 1) // 2
        self.conv1 = nn.Conv2d(channels_in, 128, filter_size, stride=1,
            padding=pad, bias=True)
        self.conv1_bn = nn.BatchNorm2d(128)
        filter_size = 3
        pad = (filter_size - 1) // 2
        self.conv2 = nn.Conv2d(128, 128, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        rows, cols = size
        self.avgpool = nn.AvgPool2d((1, cols))
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        if class_type == 'line':
            self.fully_connected1 = nn.Linear(64 * rows * cols // 4, 128)
            self.fully_connected_line1 = nn.Linear(128, 3)
            self.fully_connected_line2 = nn.Linear(128, 3)
            self.fully_connected_line3 = nn.Linear(128, 3)
            self.fully_connected_line4 = nn.Linear(128, 3)
        else:
            self.fully_connected_horizon = nn.Linear(64 * rows, resize)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        if self.class_type == 'line':
            x = self.maxpool(x)
        else:
            x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        batch_size = x.size(0)
        if self.class_type == 'line':
            x = F.relu(self.fully_connected1(x))
            x1 = self.fully_connected_line1(x).view(batch_size, 3, 1, 1)
            x2 = self.fully_connected_line2(x).view(batch_size, 3, 1, 1)
            x3 = self.fully_connected_line3(x).view(batch_size, 3, 1, 1)
            x4 = self.fully_connected_line4(x).view(batch_size, 3, 1, 1)
            x = torch.cat((x1, x2, x3, x4), 2)
        else:
            x = self.fully_connected_horizon(x)
        return x


class DLT(nn.Module):

    def __init__(self, batch_size, cuda, size, channels_in):
        super().__init__()
        self.activation = nn.Tanh()
        self.spatial_trans = Spatial_transformer_net(size, channels_in)
        self.xs1, self.xs2, self.xs3, self.xs4 = 0.1, 0.9, 0.45, 0.55
        self.ys1, self.ys2 = 1, 0.3
        self.xd1, self.xd2, self.xd3, self.xd4 = 0.45, 0.55, 0.45, 0.55
        self.yd1, self.yd2 = 1, 0.3
        A = torch.FloatTensor([[0, 0, 0, -self.ys1, -1, self.ys1 * self.yd1
            ], [self.xs1, self.ys1, 1, 0, 0, 0], [self.xs2, self.ys1, 1, 0,
            0, 0], [0, 0, 0, -self.ys2, -1, 0], [self.xs3, self.ys2, 1, 0, 
            0, 0], [self.xs4, self.ys2, 1, 0, 0, 0]])
        B = torch.FloatTensor([[-self.yd1], [0], [0], [0], [0], [0]])
        self.A = A.expand(batch_size, 6, 6)
        self.B = B.expand(batch_size, 6, 1)
        self.zeros = Variable(torch.zeros(batch_size, 1, 1))
        self.ones = Variable(torch.ones(batch_size, 1, 1))
        if cuda:
            self.activation = self.activation
            self.A = self.A
            self.B = self.B
            self.zeros = self.zeros
            self.ones = self.ones
            self.spatial_trans = self.spatial_trans

    def forward(self, output_encoder):
        x = self.spatial_trans(output_encoder)
        x = self.activation(x) / 16
        A = Variable(self.A.clone())
        B = Variable(self.B.clone())
        A[:, (1), (5)] = -self.ys1 * (self.xd1 + x[:, (0)])
        A[:, (2), (5)] = -self.ys1 * (self.xd2 + x[:, (1)])
        A[:, (3), (5)] = self.ys2 * (self.yd2 + x[:, (2)])
        A[:, (4), (5)] = -self.ys2 * (self.xd3 + x[:, (0)])
        A[:, (5), (5)] = -self.ys2 * (self.xd4 + x[:, (1)])
        B[:, (1), (0)] = self.xd1 + x[:, (0)]
        B[:, (2), (0)] = self.xd2 + x[:, (1)]
        B[:, (3), (0)] = -(self.yd2 + x[:, (2)])
        B[:, (4), (0)] = self.xd3 + x[:, (0)]
        B[:, (5), (0)] = self.xd4 + x[:, (1)]
        A_prime = torch.bmm(A.transpose(1, 2), A)
        B_prime = torch.bmm(A.transpose(1, 2), B)
        h = torch.stack([torch.gesv(b, a)[0] for b, a in zip(torch.unbind(
            B_prime), torch.unbind(A_prime))])
        h = torch.cat((h[:, 0:3, :], self.zeros, h[:, 3:5, :], self.zeros,
            h[:, 5:6, :], self.ones), 1)
        h = h.view(-1, 3, 3)
        return h, x


class Spatial_transformer_net(nn.Module):

    def __init__(self, size, channels_in):
        super().__init__()
        filter_size = 1
        pad = (filter_size - 1) // 2
        self.conv1 = nn.Conv2d(channels_in, 128, filter_size, stride=1,
            padding=pad, bias=True)
        self.conv1_bn = nn.BatchNorm2d(128)
        filter_size = 3
        pad = (filter_size - 1) // 2
        self.conv2 = nn.Conv2d(128, 128, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        rows, cols = size
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        self.conv5 = nn.Conv2d(64, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.fully_connected1 = nn.Linear(64 * rows * cols // 4, 128)
        self.fully_connected2 = nn.Linear(128, 3)
        self.fully_connected2.weight.data.fill_(0)
        self.fully_connected2.bias.data = torch.FloatTensor([0, 0, 0])

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.maxpool(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fully_connected1(x))
        x = self.fully_connected2(x)
        return x


class Proxy_branch_segmentation(nn.Module):

    def __init__(self, channels_in, resize, nclasses):
        super().__init__()
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.dec4 = _DecoderBlock(channels_in, 256, 256, pad=padding)
        self.dec3 = _DecoderBlock(256, 128, 128, pad=padding)
        self.dec2 = _DecoderBlock(128, 64, 64, pad=padding)
        self.final = nn.Conv2d(64, nclasses + 1, kernel_size=1)
        self.resize = resize

    def forward(self, x):
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.final(x)
        x = F.upsample(x, (self.resize, 2 * self.resize), mode='bilinear')
        return x


class ProjectiveGridGenerator(nn.Module):

    def __init__(self, size, theta, cuda):
        super().__init__()
        self.N, self.C, self.H, self.W = size
        linear_points_W = torch.linspace(0, 1, self.W)
        linear_points_H = torch.linspace(0, 1, self.H)
        self.base_grid = theta.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, (0)] = torch.ger(torch.ones(self.H),
            linear_points_W).expand_as(self.base_grid[:, :, :, (0)])
        self.base_grid[:, :, :, (1)] = torch.ger(linear_points_H, torch.
            ones(self.W)).expand_as(self.base_grid[:, :, :, (1)])
        self.base_grid[:, :, :, (2)] = 1
        self.base_grid = Variable(self.base_grid)
        if cuda:
            self.base_grid = self.base_grid

    def forward(self, theta, no_ortho_view):
        if no_ortho_view:
            return self.base_grid.view(self.N, -1, 3)
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3),
            theta.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        return grid


class Weighted_least_squares(nn.Module):

    def __init__(self, size, nclasses, order, cuda, reg_ls=0, use_cholesky=
        False, sample_factor=5):
        super().__init__()
        self.sample_factor = sample_factor
        N, C, self.H, W = size
        self.nclasses = nclasses
        self.tensor_ones = Variable(torch.ones(N, self.H * W, 1))
        self.order = order
        self.reg_ls = Variable(reg_ls * torch.eye(order + 1))
        self.use_cholesky = use_cholesky
        if cuda:
            self.reg_ls = self.reg_ls
            self.tensor_ones = self.tensor_ones

    def forward(self, W, grid, sample_grid):
        beta1, beta2, beta3 = None, None, None
        if sample_grid:
            W = W[:, :, ::self.sample_factor, ::self.sample_factor].contiguous(
                )
        W = W.view(-1, self.nclasses, grid.size(1))
        W0 = W[:, (0), :].unsqueeze(2)
        x_map = grid[:, :, (0)].unsqueeze(2)
        y_map = (1 - grid[:, :, (1)]).unsqueeze(2)
        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((y_map, self.tensor_ones), 2)
        elif self.order == 2:
            Y = torch.cat((y_map ** 2, y_map, self.tensor_ones), 2)
        else:
            raise NotImplementedError(
                'Requested order for polynomial fit is not implemented')
        Y = Y[0:W.size(0), :, :]
        x_map = x_map[0:W.size(0), :, :]
        Y0 = torch.mul(W0, Y)
        Z = torch.bmm(Y0.transpose(1, 2), Y0) + self.reg_ls
        if not self.use_cholesky:
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            beta0 = []
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            for image, rhs in zip(torch.unbind(Z), torch.unbind(X)):
                R = torch.potrf(image)
                opl = torch.trtrs(rhs, R.transpose(0, 1))
                beta0.append(torch.trtrs(opl[0], R, upper=False)[0])
            beta0 = torch.cat(beta0, 1).transpose(0, 1).unsqueeze(2)
        if self.nclasses > 1:
            W1 = W[:, (1), :].unsqueeze(2)
            Y1 = torch.mul(W1, Y)
            Z = torch.bmm(Y1.transpose(1, 2), Y1) + self.reg_ls
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y1.transpose(1, 2), torch.mul(W1, x_map))
            beta1 = torch.bmm(Z_inv, X)
        if self.nclasses > 2:
            W2 = W[:, (2), :].unsqueeze(2)
            Y2 = torch.mul(W2, Y)
            Z = torch.bmm(Y2.transpose(1, 2), Y2) + self.reg_ls
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y2.transpose(1, 2), torch.mul(W2, x_map))
            beta2 = torch.bmm(Z_inv, X)
            W3 = W[:, (3), :].unsqueeze(2)
            Y3 = torch.mul(W3, Y)
            Z = torch.bmm(Y3.transpose(1, 2), Y3) + self.reg_ls
            Z_inv = [torch.inverse(matrix) for matrix in torch.unbind(Z)]
            Z_inv = torch.stack(Z_inv)
            X = torch.bmm(Y3.transpose(1, 2), torch.mul(W3, x_map))
            beta3 = torch.bmm(Z_inv, X)
        return beta0, beta1, beta2, beta3


def Init_Projective_transform(nclasses, batch_size, resize):
    size = torch.Size([batch_size, nclasses, resize, 2 * resize])
    y_start = 0.3
    y_stop = 1
    xd1, xd2, xd3, xd4 = 0.45, 0.55, 0.45, 0.55
    src = np.float32([[0.45, y_start], [0.55, y_start], [0.1, y_stop], [0.9,
        y_stop]])
    dst = np.float32([[xd3, y_start], [xd4, y_start], [xd1, y_stop], [xd2,
        y_stop]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    M = torch.from_numpy(M).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch
        .FloatTensor)
    M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([batch_size, 3, 3]
        ).type(torch.FloatTensor)
    return size, M, M_inv


def define_network(options, norm='batch'):
    print('defining network')
    if options.model_seg == 'resnet':
        net = simple_net(options.nclasses)
    elif options.model_seg == 'resnet18':
        net = ResNet18(options.nclasses)
    elif options.model_seg == 'unet':
        net = UNet(options.nclasses, options.activation_net, options.pad)
    elif options.model_seg == 'enet':
        net = ENet(options.nclasses, options.norm, options.no_dropout)
    elif options.model_seg == 'segnet':
        net = SegNet(options.nclasses)
    else:
        raise NotImplementedError('The requested {} is not yet implemented'
            .format(options.model_seg))
    if options.cuda:
        net = net.cuda()
    print(net)
    return net


class Net(nn.Module):

    def __init__(self, options):
        super().__init__()
        resize = options.resize
        size = torch.Size([options.batch_size, options.nclasses, options.
            resize, 2 * options.resize])
        size, M, M_inv = Init_Projective_transform(options.nclasses,
            options.batch_size, options.resize, options.grid_sample,
            options.sample_factor)
        self.M = Variable(M)
        self.net = define_network(options)
        self.activation = activation_layer(options.activation_layer,
            options.cuda)
        self.project_layer = ProjectiveGridGenerator(size, M, options.cuda)
        self.ls_layer = Weighted_least_squares(size, options.nclasses,
            options.order, options.cuda, options.reg_ls, options.use_cholesky)
        if options.classification_branch:
            factor = options.resize / 256
            if options.model_seg == 'unet':
                chan = 512
                size_enc = 32, 64
                if options.pad == 1:
                    size_enc = tuple(int(i * factor) for i in size_enc)
                elif options.pad == 0:
                    size_enc = (16, 48) if factor == 1 else (24, 64)
                else:
                    raise NotImplementedError
            elif options.model_seg == 'segnet':
                chan = 512
                size_enc = 8, 16
                size_enc = tuple(int(i * factor) for i in size_enc)
            elif options.model_seg == 'enet':
                chan = 128
                size_enc = 32, 64
                size_enc = tuple(int(i * factor) for i in size_enc)
            elif options.model_seg == 'resnet':
                chan = 256
                num_downsample = 0
                size_enc = tuple(int(i * factor) for i in size_enc)
                size_enc = (options.resize // 2 ** num_downsample, 2 *
                    options.resize // 2 ** num_downsample)
            elif options.model_seg == 'resnet18':
                chan = 512
                size_enc = 8, 16
                size_enc = tuple(int(i * factor) for i in size_enc)
            else:
                raise NotImplementedError
            self.line_classification = Classification('line', size=size_enc,
                channels_in=chan, resize=resize)
            self.horizon_estimation = Classification('horizon', size=
                size_enc, channels_in=chan, resize=resize)
        self.classification_branch = options.classification_branch
        self.DLT = DLT(options.batch_size, options.cuda, size=size_enc,
            channels_in=chan)
        self.proxy = Proxy_branch_segmentation(chan, resize, options.nclasses)
        self.DLT_on = options.DLT_on
        self.proxy_branch = options.proxy_branch
        zero_rows = ceil(options.resize * options.mask_percentage)
        self.idx_row = Variable(torch.linspace(0, zero_rows - 1, zero_rows)
            .long())
        n_row = 13
        self.idx_col1 = Variable(torch.linspace(0, n_row, n_row + 1).long())
        self.idx_col2 = Variable(torch.linspace(0, n_row, n_row + 1).long()
            ) + 2 * resize - (n_row + 1)
        idx_mask = (np.arange(resize)[:, (None)] < np.arange(2 * resize) -
            (resize + 10)) * 1
        idx_mask = np.flip(idx_mask, 1).copy() + idx_mask
        self.idx_mask = Variable(torch.from_numpy(idx_mask)).type(torch.
            ByteTensor).expand(options.batch_size, options.nclasses, resize,
            2 * resize)
        if options.cuda:
            self.M = self.M
            self.idx_row = self.idx_row
            self.idx_col1 = self.idx_col1
            self.idx_col2 = self.idx_col2
            self.idx_mask = self.idx_mask
            if options.classification_branch:
                self.line_classification = self.line_classification
                self.horizon_estimation = self.horizon_estimation
                self.DLT = self.DLT
            if options.proxy_branch:
                self.proxy = self.proxy

    def forward(self, input, no_ortho_view, sample_grid):
        output_seg, line, horizon, x = None, None, None, None
        output, output_encoder = self.net(input)
        if self.classification_branch:
            line = self.line_classification(output_encoder)
            horizon = self.horizon_estimation(output_encoder)
        if self.proxy_branch:
            output_seg = self.proxy(output_encoder)
        output = self.activation(output)
        if not no_ortho_view:
            output = output.index_fill(2, self.idx_row, 0)
        if self.DLT_on:
            M, x = self.DLT(output_encoder)
            grid = self.project_layer(M, no_ortho_view)
        else:
            M = self.M
            grid = self.project_layer(self.M, no_ortho_view)
        beta0, beta1, beta2, beta3 = self.ls_layer(output, grid, sample_grid)
        return (beta0, beta1, beta2, beta3, output ** 2, output_seg, line,
            horizon, M, x)


class CrossEntropyLoss2d(nn.Module):
    """
    Standard 2d cross entropy loss on all pixels of image
    My implemetation (but since Pytorch 0.2.0 libs have their
    owm optimized implementation, consider using theirs)
    """

    def __init__(self, weight=None, size_average=True, seg=False):
        if seg:
            weights = torch.Tensor([1] + [weight] * 2)
            weights = weights
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weights, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets[:, (0),
            :, :])


class Area_Loss(nn.Module):
    """
    Compute area between curves by integrating (x1 - x2)^2 over y
    *Area:
        *order 0: int((c1 - c2)**2)dy
        *order 1: int((b1*y - b2*y + c1 - c2)**2)dy
        *order 2: int((a1*y**2 - a2*y**2 + b1*y - b2*y + c1 - c2)**2)dy

    *A weight function W can be added:
        Weighted area: int(W(y)*diff**2)dy
        with W(y):
            *1
            *(1-y)
            *(1-y**0.5)
    """

    def __init__(self, order, weight_funct):
        super(Area_Loss, self).__init__()
        self.order = order
        self.weight_funct = weight_funct

    def forward(self, params, gt_params, compute=True):
        diff = params.squeeze(-1) - gt_params
        a = diff[:, (0)]
        b = diff[:, (1)]
        t = 0.7
        if self.order == 2:
            c = diff[:, (2)]
            if self.weight_funct == 'none':
                loss_fit = a ** 2 * t ** 5 / 5 + 2 * a * b * t ** 4 / 4 + (
                    b ** 2 + c * 2 * a
                    ) * t ** 3 / 3 + 2 * b * c * t ** 2 / 2 + c ** 2 * t
            elif self.weight_funct == 'linear':
                loss_fit = c ** 2 * t - t ** 5 * (2 * a * b / 5 - a ** 2 / 5
                    ) + t ** 2 * (b * c - c ** 2 / 2
                    ) - a ** 2 * t ** 6 / 6 - t ** 4 * (b ** 2 / 4 - a * b /
                    2 + a * c / 2) + t ** 3 * (b ** 2 / 3 - 2 * c * b / 3 +
                    2 * a * c / 3)
            elif self.weight_funct == 'quadratic':
                loss_fit = (t ** 3 * (1 / 3 * b ** 2 + 2 / 3 * a * c) - t **
                    (7 / 2) * (2 / 7 * b ** 2 + 4 / 7 * a * c) + c ** 2 * t +
                    0.2 * a ** 2 * t ** 5 - 2 / 11 * a ** 2 * t ** (11 / 2) -
                    2 / 3 * c ** 2 * t ** (3 / 2) + 0.5 * a * b * t ** 4 - 
                    4 / 9 * a * b * t ** (9 / 2) + b * c * t ** 2 - 0.8 * b *
                    c * t ** (5 / 2))
            else:
                return NotImplementedError(
                    'The requested weight function is                         not implemented, only order 1 or order 2 possible'
                    )
        elif self.order == 1:
            loss_fit = b ** 2 * t + a * b * t ** 2 + a ** 2 * t ** 3 / 3
        else:
            return NotImplementedError(
                'The requested order is not implemented, only none, linear or quadratic possible'
                )
        mask = torch.prod(gt_params != 0, 1).byte()
        loss_fit = torch.masked_select(loss_fit, mask)
        loss_fit = loss_fit.mean(0) if loss_fit.size()[0] != 0 else 0
        return loss_fit


class MSE_Loss(nn.Module):
    """
    Compute mean square error loss on curve parameters
    in ortho or normal view
    """

    def __init__(self, options):
        super(MSE_Loss, self).__init__()
        self.loss_crit = nn.MSELoss()
        if not options.no_cuda:
            self.loss_crit = self.loss_crit

    def forward(self, params, gt_params, compute=True):
        loss = self.loss_crit(params.squeeze(-1), gt_params)
        return loss


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2,
            padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding
            =0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2,
            padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):

    def __init__(self, num_classes, pretrain):
        super().__init__()
        self.pretrain = pretrain
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2,
            padding=0, output_padding=0, bias=True)
        if pretrain:
            self.output_conv2 = nn.ConvTranspose2d(16, num_classes + 1, 2,
                stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input, flag):
        output = input
        for layer in self.layers:
            output = layer(output)
        if self.pretrain:
            if flag:
                output = self.output_conv(output)
            else:
                output = self.output_conv2(output)
        else:
            output = self.output_conv(output)
        return output


class Net(nn.Module):

    def __init__(self, layers=18, in_channels=1, out_channels=1, pretrained
        =False, pool=False):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels, pretrained)

    def forward(self, input, flag, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            encoder_output = self.encoder(input)
            decoder_output = self.decoder.forward(encoder_output, flag)
            return encoder_output, decoder_output


class ProjectiveGridGenerator(nn.Module):

    def __init__(self, size, theta, no_cuda):
        super().__init__()
        self.N, self.C, self.H, self.W = size
        linear_points_W = torch.linspace(0, 1 - 1 / self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1 / self.H, self.H)
        self.base_grid = theta.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, (0)] = torch.ger(torch.ones(self.H),
            linear_points_W).expand_as(self.base_grid[:, :, :, (0)])
        self.base_grid[:, :, :, (1)] = torch.ger(linear_points_H, torch.
            ones(self.W)).expand_as(self.base_grid[:, :, :, (1)])
        self.base_grid[:, :, :, (2)] = 1
        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid

    def forward(self, theta):
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3),
            theta.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        return grid


class Weighted_least_squares(nn.Module):

    def __init__(self, size, nclasses, order, no_cuda, reg_ls=0,
        use_cholesky=False):
        super().__init__()
        N, C, self.H, W = size
        self.nclasses = nclasses
        self.tensor_ones = Variable(torch.ones(N, self.H * W, 1))
        self.order = order
        self.reg_ls = Variable(reg_ls * torch.eye(order + 1))
        self.use_cholesky = use_cholesky
        if not no_cuda:
            self.reg_ls = self.reg_ls
            self.tensor_ones = self.tensor_ones

    def forward(self, W, grid):
        beta2, beta3 = None, None
        W = W.view(-1, self.nclasses, grid.size(1))
        W0 = W[:, (0), :].unsqueeze(2)
        x_map = grid[:, :, (0)].unsqueeze(2)
        y_map = (1 - grid[:, :, (1)]).unsqueeze(2)
        if self.order == 0:
            Y = self.tensor_ones
        elif self.order == 1:
            Y = torch.cat((y_map, self.tensor_ones), 2)
        elif self.order == 2:
            Y = torch.cat((y_map ** 2, y_map, self.tensor_ones), 2)
        else:
            raise NotImplementedError(
                'Requested order {} for polynomial fit is not implemented'.
                format(self.order))
        Y0 = torch.mul(W0, Y)
        Z = torch.bmm(Y0.transpose(1, 2), Y0) + self.reg_ls
        if not self.use_cholesky:
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            beta0 = torch.bmm(Z_inv, X)
        else:
            beta0 = []
            X = torch.bmm(Y0.transpose(1, 2), torch.mul(W0, x_map))
            for image, rhs in zip(torch.unbind(Z), torch.unbind(X)):
                R = torch.potrf(image)
                opl = torch.trtrs(rhs, R.transpose(0, 1))
                beta0.append(torch.trtrs(opl[0], R, upper=False)[0])
            beta0 = torch.cat(beta0, 1).transpose(0, 1).unsqueeze(2)
        W1 = W[:, (1), :].unsqueeze(2)
        Y1 = torch.mul(W1, Y)
        Z = torch.bmm(Y1.transpose(1, 2), Y1) + self.reg_ls
        Z_inv = torch.inverse(Z)
        X = torch.bmm(Y1.transpose(1, 2), torch.mul(W1, x_map))
        beta1 = torch.bmm(Z_inv, X)
        if self.nclasses > 3:
            W2 = W[:, (2), :].unsqueeze(2)
            Y2 = torch.mul(W2, Y)
            Z = torch.bmm(Y2.transpose(1, 2), Y2) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y2.transpose(1, 2), torch.mul(W2, x_map))
            beta2 = torch.bmm(Z_inv, X)
            W3 = W[:, (3), :].unsqueeze(2)
            Y3 = torch.mul(W3, Y)
            Z = torch.bmm(Y3.transpose(1, 2), Y3) + self.reg_ls
            Z_inv = torch.inverse(Z)
            X = torch.bmm(Y3.transpose(1, 2), torch.mul(W3, x_map))
            beta3 = torch.bmm(Z_inv, X)
        return beta0, beta1, beta2, beta3


class Classification(nn.Module):

    def __init__(self, class_type, size, channels_in, resize):
        super().__init__()
        self.class_type = class_type
        filter_size = 1
        pad = (filter_size - 1) // 2
        self.conv1 = nn.Conv2d(channels_in, 128, filter_size, stride=1,
            padding=pad, bias=True)
        self.conv1_bn = nn.BatchNorm2d(128)
        filter_size = 3
        pad = (filter_size - 1) // 2
        self.conv2 = nn.Conv2d(128, 128, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, filter_size, stride=1, padding=pad,
            bias=True)
        self.conv4_bn = nn.BatchNorm2d(64)
        rows, cols = size
        self.avgpool = nn.AvgPool2d((1, cols))
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        if class_type == 'line':
            self.fully_connected1 = nn.Linear(64 * rows * cols // 4, 128)
            self.fully_connected_line1 = nn.Linear(128, 3)
            self.fully_connected_line2 = nn.Linear(128, 3)
            self.fully_connected_line3 = nn.Linear(128, 3)
            self.fully_connected_line4 = nn.Linear(128, 3)
        else:
            self.fully_connected_horizon = nn.Linear(64 * rows, resize)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        if self.class_type == 'line':
            x = self.maxpool(x)
        else:
            x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        batch_size = x.size(0)
        if self.class_type == 'line':
            x = F.relu(self.fully_connected1(x))
            x1 = self.fully_connected_line1(x).view(batch_size, 3, 1, 1)
            x2 = self.fully_connected_line2(x).view(batch_size, 3, 1, 1)
            x3 = self.fully_connected_line3(x).view(batch_size, 3, 1, 1)
            x4 = self.fully_connected_line4(x).view(batch_size, 3, 1, 1)
            x = torch.cat((x1, x2, x3, x4), 2).squeeze(3)
        else:
            x = self.fully_connected_horizon(x)
        return x


class Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        resize = args.resize
        size = torch.Size([args.batch_size, args.nclasses, args.resize, 2 *
            args.resize])
        size, M, M_inv = Init_Projective_transform(args.nclasses, args.
            batch_size, args.resize)
        self.M = M
        out_channels = args.nclasses + int(not args.end_to_end)
        self.net = Networks.define_model(mod=args.mod, layers=args.layers,
            in_channels=args.channels_in, out_channels=out_channels,
            pretrained=args.pretrained, pool=args.pool)
        self.activation = activation_layer(args.activation_layer, args.no_cuda)
        self.project_layer = ProjectiveGridGenerator(size, M, args.no_cuda)
        self.ls_layer = Weighted_least_squares(size, args.nclasses, args.
            order, args.no_cuda, args.reg_ls, args.use_cholesky)
        zero_rows = ceil(args.resize * args.mask_percentage)
        self.idx_row = Variable(torch.linspace(0, zero_rows - 1, zero_rows)
            .long())
        n_row = 13
        self.idx_col1 = Variable(torch.linspace(1, n_row, n_row + 1).long())
        self.idx_col2 = Variable(torch.linspace(0, n_row, n_row + 1).long()
            ) + 2 * resize - (n_row + 1)
        idx_mask = (np.arange(resize)[:, (None)] < np.arange(2 * resize) -
            (resize + 10)) * 1
        idx_mask = np.flip(idx_mask, 1).copy() + idx_mask
        self.idx_mask = Variable(torch.from_numpy(idx_mask)).type(torch.
            ByteTensor).expand(args.batch_size, args.nclasses, resize, 2 *
            resize)
        self.end_to_end = args.end_to_end
        self.pretrained = args.pretrained
        self.classification_branch = args.clas
        if self.classification_branch:
            size_enc = 32, 64
            chan = 128
            self.line_classification = Classification('line', size=size_enc,
                channels_in=chan, resize=resize)
            self.horizon_estimation = Classification('horizon', size=
                size_enc, channels_in=chan, resize=resize)
        if not args.no_cuda:
            self.M = self.M
            self.idx_row = self.idx_row
            self.idx_col1 = self.idx_col1
            self.idx_col2 = self.idx_col2
            self.idx_mask = self.idx_mask
            if self.classification_branch:
                self.line_classification = self.line_classification
                self.horizon_estimation = self.horizon_estimation

    def forward(self, input, end_to_end):
        line, horizon = None, None
        shared_encoder, output = self.net(input, end_to_end * self.pretrained)
        if not end_to_end:
            activated = output.detach()
            _, activated = torch.max(activated, 1)
            activated = activated.float()
            left = activated * (activated == 1).float()
            right = activated * (activated == 2).float()
            activated = torch.stack((left, right), 1)
        else:
            activated = self.activation(output)
            if self.classification_branch:
                line = self.line_classification(shared_encoder)
                horizon = self.horizon_estimation(shared_encoder)
        masked = activated.index_fill(2, self.idx_row, 0)
        grid = self.project_layer(self.M)
        beta0, beta1, beta2, beta3 = self.ls_layer(masked, grid)
        return (beta0, beta1, beta2, beta3, masked, self.M, output, line,
            horizon)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_wvangansbeke_LaneDetection_End2End(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Area_Loss(*[], **{'order': 4, 'weight_funct': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Classification(*[], **{'class_type': 4, 'size': [4, 4], 'channels_in': 4, 'resize': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Decoder(*[], **{'num_classes': 4, 'pretrain': False}), [torch.rand([4, 128, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Encoder(*[], **{'in_channels': 4, 'num_classes': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MSE_Loss(*[], **{'options': _mock_config(no_cuda=4)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Spatial_transformer_net(*[], **{'size': [4, 4], 'channels_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(UpsamplerBlock(*[], **{'ninput': 4, 'noutput': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(resnet_block(*[], **{'in_channels': 4, 'out_channels': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(simple_net(*[], **{'nclasses': 4}), [torch.rand([4, 3, 64, 64])], {})

