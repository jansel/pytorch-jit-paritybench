import sys
_module = sys.modules[__name__]
del sys
train = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.backends import cudnn


import math


class LinearFAFunction(torch.autograd.Function):
    """Autograd function for linear feedback alignment module.
    """

    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None
        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_weight_fa, grad_bias


_global_config['cuda'] = 4


class LinearFA(nn.Module):
    """Linear feedback alignment module.

    Args:
        input_features (int): Number of input features to linear layer.
        output_features (int): Number of output features from linear layer.
        bias (bool): True if to use trainable bias.
    """

    def __init__(self, input_features, output_features, bias=True):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features,
            input_features))
        self.weight_fa = nn.Parameter(torch.Tensor(output_features,
            input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if args.cuda:
            self.weight.data = self.weight.data
            self.weight_fa.data = self.weight_fa.data
            if bias:
                self.bias.data = self.bias.data

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_fa.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa,
            self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            input_features) + ', out_features=' + str(self.output_features
            ) + ', bias=' + str(self.bias is not None) + ')'


_global_config['no_similarity_std'] = 4


def similarity_matrix(x):
    """ Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). """
    if x.dim() == 4:
        if not args.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-08 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


_global_config['nonlin'] = 4


_global_config['backprop'] = 4


_global_config['momentum'] = 4


_global_config['no_print_stats'] = 4


_global_config['target_proj_size'] = 4


class LocalLossBlockLinear(nn.Module):
    """A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
       
    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    """

    def __init__(self, num_in, num_out, num_classes, first_layer=False,
        dropout=None, batchnorm=None):
        super(LocalLossBlockLinear, self).__init__()
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        self.batchnorm = (not args.no_batch_norm if batchnorm is None else
            batchnorm)
        self.encoder = nn.Linear(num_in, num_out, bias=True)
        if not args.backprop and args.loss_unsup == 'recon':
            self.decoder_x = nn.Linear(num_out, num_in, bias=True)
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup ==
            'predsim'):
            if args.bio:
                self.decoder_y = LinearFA(num_out, args.target_proj_size)
            else:
                self.decoder_y = nn.Linear(num_out, num_classes)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size,
                bias=False)
        if not args.backprop and not args.bio and (args.loss_unsup == 'sim' or
            args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        if args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0,
                weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0,
                weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad'
                )
        self.clear_stats()

    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = (
                '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'
                .format(self.encoder, self.loss_sim / self.examples, self.
                loss_pred / self.examples, 100.0 * float(self.examples -
                self.correct) / self.examples, self.examples))
            return stats
        else:
            return ''

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot):
        h = self.encoder(x)
        if self.batchnorm:
            h = self.bn(h)
        h = self.nonlin(h)
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)
        if (self.training or not args.no_print_stats) and not args.backprop:
            if (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.
                loss_sup == 'predsim'):
                if args.bio:
                    h_loss = h
                else:
                    h_loss = self.linear_loss(h)
                Rh = similarity_matrix(h_loss)
            if args.loss_unsup == 'sim':
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif args.loss_unsup == 'recon' and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            elif args.cuda:
                loss_unsup = torch.cuda.FloatTensor([0])
            else:
                loss_unsup = torch.FloatTensor([0])
            if args.loss_sup == 'sim':
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif args.loss_sup == 'pred':
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    float_type = (torch.cuda.FloatTensor if args.cuda else
                        torch.FloatTensor)
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type
                        ).detach()
                    loss_sup = F.binary_cross_entropy_with_logits(y_hat_local,
                        y_onehot_pred)
                else:
                    loss_sup = F.cross_entropy(y_hat_local, y.detach())
                if not args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif args.loss_sup == 'predsim':
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type = (torch.cuda.FloatTensor if args.cuda else
                        torch.FloatTensor)
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type
                        ).detach()
                    loss_pred = (1 - args.beta
                        ) * F.binary_cross_entropy_with_logits(y_hat_local,
                        y_onehot_pred)
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1 - args.beta) * F.cross_entropy(y_hat_local,
                        y.detach())
                loss_sim = args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            loss = args.alpha * loss_unsup + (1 - args.alpha) * loss_sup
            if self.training:
                loss.backward(retain_graph=args.no_detach)
            if self.training and not args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()
            loss = loss.item()
        else:
            loss = 0.0
        return h_return, loss


_global_config['dim_in_decoder'] = 4


class LocalLossBlockConv(nn.Module):
    """
    A block containing nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d
    The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        ch_in (int): Number of input features maps.
        ch_out (int): Number of output features maps.
        kernel_size (int): Kernel size in Conv2d.
        stride (int): Stride in Conv2d.
        padding (int): Padding in Conv2d.
        num_classes (int): Number of classes (used in local prediction loss).
        dim_out (int): Feature map height/width for input (and output).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        bias (bool): True if to use trainable bias.
        pre_act (bool): True if to apply layer order nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d -> nn.Conv2d (used for PreActResNet).
        post_act (bool): True if to apply layer order nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d.
    """

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding,
        num_classes, dim_out, first_layer=False, dropout=None, bias=None,
        pre_act=False, post_act=True):
        super(LocalLossBlockConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        self.bias = True if bias is None else bias
        self.pre_act = pre_act
        self.post_act = post_act
        self.encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride,
            padding=padding, bias=self.bias)
        if not args.backprop and args.loss_unsup == 'recon':
            self.decoder_x = nn.ConvTranspose2d(ch_out, ch_in, kernel_size,
                stride=stride, padding=padding)
        if args.bio or not args.backprop and (args.loss_sup == 'pred' or 
            args.loss_sup == 'predsim'):
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = dim_out, dim_out
            dim_in_decoder = ch_out * dim_out_h * dim_out_w
            while dim_in_decoder > args.dim_in_decoder and ks_h < dim_out:
                ks_h *= 2
                dim_out_h = math.ceil(dim_out / ks_h)
                dim_in_decoder = ch_out * dim_out_h * dim_out_w
                if dim_in_decoder > args.dim_in_decoder:
                    ks_w *= 2
                    dim_out_w = math.ceil(dim_out / ks_w)
                    dim_in_decoder = ch_out * dim_out_h * dim_out_w
            if ks_h > 1 or ks_w > 1:
                pad_h = ks_h * (dim_out_h - dim_out // ks_h) // 2
                pad_w = ks_w * (dim_out_w - dim_out // ks_w) // 2
                self.avg_pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h,
                    pad_w))
            else:
                self.avg_pool = None
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup ==
            'predsim'):
            if args.bio:
                self.decoder_y = LinearFA(dim_in_decoder, args.target_proj_size
                    )
            else:
                self.decoder_y = nn.Linear(dim_in_decoder, num_classes)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size,
                bias=False)
        if not args.backprop and (args.loss_unsup == 'sim' or args.loss_sup ==
            'sim' or args.loss_sup == 'predsim'):
            self.conv_loss = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding
                =1, bias=False)
        if not args.no_batch_norm:
            if pre_act:
                self.bn_pre = torch.nn.BatchNorm2d(ch_in)
            if not (pre_act and args.backprop):
                self.bn = torch.nn.BatchNorm2d(ch_out)
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)
        if args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0,
                weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0,
                weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad'
                )
        self.clear_stats()

    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = (
                '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'
                .format(self.encoder, self.loss_sim / self.examples, self.
                loss_pred / self.examples, 100.0 * float(self.examples -
                self.correct) / self.examples, self.examples))
            return stats
        else:
            return ''

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot, x_shortcut=None):
        if self.pre_act:
            if not args.no_batch_norm:
                x = self.bn_pre(x)
            x = self.nonlin(x)
            if self.dropout_p > 0:
                x = self.dropout(x)
        h = self.encoder(x)
        if self.post_act and not args.no_batch_norm:
            h = self.bn(h)
        if x_shortcut is not None:
            h = h + x_shortcut
        if self.post_act:
            h = self.nonlin(h)
        h_return = h
        if self.post_act and self.dropout_p > 0:
            h_return = self.dropout(h_return)
        if (not args.no_print_stats or self.training) and not args.backprop:
            if not self.post_act:
                if not args.no_batch_norm:
                    h = self.bn(h)
                h = self.nonlin(h)
            if (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.
                loss_sup == 'predsim'):
                if args.bio:
                    h_loss = h
                    if self.avg_pool is not None:
                        h_loss = self.avg_pool(h_loss)
                else:
                    h_loss = self.conv_loss(h)
                Rh = similarity_matrix(h_loss)
            if args.loss_unsup == 'sim':
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif args.loss_unsup == 'recon' and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            elif args.cuda:
                loss_unsup = torch.cuda.FloatTensor([0])
            else:
                loss_unsup = torch.FloatTensor([0])
            if args.loss_sup == 'sim':
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif args.loss_sup == 'pred':
                if self.avg_pool is not None:
                    h = self.avg_pool(h)
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    float_type = (torch.cuda.FloatTensor if args.cuda else
                        torch.FloatTensor)
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type
                        ).detach()
                    loss_sup = F.binary_cross_entropy_with_logits(y_hat_local,
                        y_onehot_pred)
                else:
                    loss_sup = F.cross_entropy(y_hat_local, y.detach())
                if not args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif args.loss_sup == 'predsim':
                if self.avg_pool is not None:
                    h = self.avg_pool(h)
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type = (torch.cuda.FloatTensor if args.cuda else
                        torch.FloatTensor)
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type
                        ).detach()
                    loss_pred = (1 - args.beta
                        ) * F.binary_cross_entropy_with_logits(y_hat_local,
                        y_onehot_pred)
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1 - args.beta) * F.cross_entropy(y_hat_local,
                        y.detach())
                loss_sim = args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            loss = args.alpha * loss_unsup + (1 - args.alpha) * loss_sup
            if self.training:
                loss.backward(retain_graph=args.no_detach)
            if self.training and not args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()
            loss = loss.item()
        else:
            loss = 0.0
        return h_return, loss


_global_config['pre_act'] = 4


_global_config['no_detach'] = 4


_global_config['optim'] = 4


_global_config['weight_decay'] = 4


class BasicBlock(nn.Module):
    """ Used in ResNet() """
    expansion = 1

    def __init__(self, in_planes, planes, stride, num_classes, input_dim):
        super(BasicBlock, self).__init__()
        self.input_dim = input_dim
        self.stride = stride
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, stride, 1,
            num_classes, input_dim, bias=False, pre_act=args.pre_act,
            post_act=not args.pre_act)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, 1, 1,
            num_classes, input_dim, bias=False, pre_act=args.pre_act,
            post_act=not args.pre_act)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False, groups=1), nn.BatchNorm2d(self.expansion * planes))
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0,
                    weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=
                    0, weight_decay=args.weight_decay, amsgrad=args.optim ==
                    'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        if len(self.shortcut) > 0:
            self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        if len(self.shortcut) > 0:
            self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out, loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        out, loss = self.conv2(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return out, y, y_onehot, loss_total


class Bottleneck(nn.Module):
    """ Used in ResNet() """
    expansion = 4

    def __init__(self, in_planes, planes, stride, num_classes, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = LocalLossBlockConv(in_planes, planes, 1, 1, 0,
            num_classes, input_dim, bias=False)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1,
            num_classes, input_dim // stride, bias=False)
        self.conv3 = LocalLossBlockConv(planes, self.expansion * planes, 1,
            1, 0, num_classes, input_dim // stride, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0,
                    weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=
                    0, weight_decay=args.weight_decay, amsgrad=args.optim ==
                    'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        self.conv3.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        self.conv3.optim_zero_grad()
        if len(self.shortcut) > 0:
            self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        self.conv3.optim_step()
        if len(self.shortcut) > 0:
            self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out, loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        out, loss = self.conv2(out, y, y_onehot)
        loss_total += loss
        out, loss = self.conv3(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return out, y, y_onehot, loss_total


class ResNet(nn.Module):
    """
    Residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    """

    def __init__(self, block, num_blocks, num_classes, input_ch,
        feature_multiplyer, input_dim):
        super(ResNet, self).__init__()
        self.in_planes = int(feature_multiplyer * 64)
        self.conv1 = LocalLossBlockConv(input_ch, int(feature_multiplyer * 
            64), 3, 1, 1, num_classes, input_dim, bias=False, post_act=not
            args.pre_act)
        self.layer1 = self._make_layer(block, int(feature_multiplyer * 64),
            num_blocks[0], 1, num_classes, input_dim)
        self.layer2 = self._make_layer(block, int(feature_multiplyer * 128),
            num_blocks[1], 2, num_classes, input_dim)
        self.layer3 = self._make_layer(block, int(feature_multiplyer * 256),
            num_blocks[2], 2, num_classes, input_dim // 2)
        self.layer4 = self._make_layer(block, int(feature_multiplyer * 512),
            num_blocks[3], 2, num_classes, input_dim // 4)
        self.linear = nn.Linear(int(feature_multiplyer * 512 * block.
            expansion), num_classes)
        if not args.backprop:
            self.linear.weight.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, num_classes,
        input_dim):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        stride_cum = 1
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_classes,
                input_dim // stride_cum))
            stride_cum *= stride
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def parameters(self):
        if not args.backprop:
            return self.linear.parameters()
        else:
            return super(ResNet, self).parameters()

    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)
        for layer in self.layer4:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()
        for layer in self.layer4:
            layer.optim_zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()
        for layer in self.layer4:
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x, loss = self.conv1(x, y, y_onehot)
        x, _, _, loss = self.layer1((x, y, y_onehot, loss))
        x, _, _, loss = self.layer2((x, y, y_onehot, loss))
        x, _, _, loss = self.layer3((x, y, y_onehot, loss))
        x, _, _, loss = self.layer4((x, y, y_onehot, loss))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, loss


class wide_basic(nn.Module):
    """ Used in WideResNet() """

    def __init__(self, in_planes, planes, dropout_rate, stride, num_classes,
        input_dim, adapted):
        super(wide_basic, self).__init__()
        self.adapted = adapted
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, 1, 1,
            num_classes, input_dim * stride, dropout=None if self.adapted else
            0, bias=True, pre_act=True, post_act=False)
        if not self.adapted:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1,
            num_classes, input_dim, dropout=None if self.adapted else 0,
            bias=True, pre_act=True, post_act=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True))
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0,
                    weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=
                    0, weight_decay=args.weight_decay, amsgrad=args.optim ==
                    'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        if len(self.shortcut) > 0:
            self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        if len(self.shortcut) > 0:
            self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out, loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        if not self.adapted:
            out = self.dropout(out)
        out, loss = self.conv2(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return out, y, y_onehot, loss_total


class Wide_ResNet(nn.Module):
    """
    Wide residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    """

    def __init__(self, depth, widen_factor, dropout_rate, num_classes,
        input_ch, input_dim, adapted=False):
        super(Wide_ResNet, self).__init__()
        self.adapted = adapted
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        None
        if self.adapted:
            nStages = [16 * k, 16 * k, 32 * k, 64 * k]
        else:
            nStages = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = nStages[0]
        self.conv1 = LocalLossBlockConv(input_ch, nStages[0], 3, 1, 1,
            num_classes, 32, dropout=0, bias=True, post_act=False)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n,
            dropout_rate, 1, num_classes, input_dim, adapted)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n,
            dropout_rate, 2, num_classes, input_dim, adapted)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n,
            dropout_rate, 2, num_classes, input_dim // 2, adapted)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3] * (16 if self.adapted else 1),
            num_classes)
        if not args.backprop:
            self.linear.weight.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride,
        num_classes, input_dim, adapted):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        stride_cum = 1
        for stride in strides:
            stride_cum *= stride
            layers.append(block(self.in_planes, planes, dropout_rate,
                stride, num_classes, input_dim // stride_cum, adapted))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def parameters(self):
        if not args.backprop:
            return self.linear.parameters()
        else:
            return super(Wide_ResNet, self).parameters()

    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x, loss = self.conv1(x, y, y_onehot)
        x, _, _, loss = self.layer1((x, y, y_onehot, loss))
        x, _, _, loss = self.layer2((x, y, y_onehot, loss))
        x, _, _, loss = self.layer3((x, y, y_onehot, loss))
        x = F.relu(self.bn1(x))
        if self.adapted:
            x = F.max_pool2d(x, 2)
        else:
            x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, loss


class Net(nn.Module):
    """
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    """

    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes
        ):
        super(Net, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList([LocalLossBlockLinear(input_dim *
            input_dim * input_ch, num_hidden, num_classes, first_layer=True)])
        self.layers.extend([LocalLossBlockLinear(int(num_hidden // 
            reduce_factor ** (i - 1)), int(num_hidden // reduce_factor ** i
            ), num_classes) for i in range(1, num_layers)])
        self.layer_out = nn.Linear(int(num_hidden // reduce_factor ** (
            num_layers - 1)), num_classes)
        if not args.backprop:
            self.layer_out.weight.data.zero_()

    def parameters(self):
        if not args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)
        return x, total_loss


_global_config['num_hidden'] = 4


_global_config['num_layers'] = 1


class VGGn(nn.Module):
    """
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        vgg_name (str): The name of the network.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    """

    def __init__(self, vgg_name, input_dim, input_ch, num_classes, feat_mult=1
        ):
        super(VGGn, self).__init__()
        self.cfg = cfg[vgg_name]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes
        self.features, output_dim = self._make_layers(self.cfg, input_ch,
            input_dim, feat_mult)
        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        if args.num_layers > 0:
            self.classifier = Net(args.num_layers, args.num_hidden,
                output_dim, int(output_ch * feat_mult), num_classes)
        else:
            self.classifier = nn.Linear(output_dim * output_dim * int(
                output_ch * feat_mult), num_classes)

    def parameters(self):
        if not args.backprop:
            return self.classifier.parameters()
        else:
            return super(VGGn, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].set_learning_rate(lr)
        if args.num_layers > 0:
            self.classifier.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_zero_grad()
        if args.num_layers > 0:
            self.classifier.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_step()
        if args.num_layers > 0:
            self.classifier.optim_step()

    def forward(self, x, y, y_onehot):
        loss_total = 0
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                x, loss = self.features[i](x, y, y_onehot)
                loss_total += loss
            else:
                x = self.features[i](x)
        if args.num_layers > 0:
            x, loss = self.classifier(x, y, y_onehot)
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        loss_total += loss
        return x, loss_total

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale_cum *= 2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *= 2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                scale_cum *= 4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scale_cum *= 2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *= 2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                scale_cum *= 4
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=
                        7, stride=2, padding=3, num_classes=num_classes,
                        dim_out=input_dim // scale_cum, first_layer=
                        first_layer)]
                else:
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=
                        3, stride=1, padding=1, num_classes=num_classes,
                        dim_out=input_dim // scale_cum, first_layer=
                        first_layer)]
                input_ch = x
                first_layer = False
        return nn.Sequential(*layers), input_dim // scale_cum


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_anokland_local_loss(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LinearFA(*[], **{'input_features': 4, 'output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

