import sys
_module = sys.modules[__name__]
del sys
dataset = _module
cars196 = _module
cub200 = _module
stanford = _module
metric = _module
batchsampler = _module
loss = _module
pairsampler = _module
utils = _module
model = _module
backbone = _module
inception = _module
google = _module
v1bn = _module
resnet = _module
embedding = _module
run = _module
run_distill = _module
run_distill_fitnet = _module

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


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min
        =eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class _Triplet(nn.Module):

    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True,
        size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=p == 2)
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)
        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]
        loss = F.triplet_margin_loss(anchor_embed, positive_embed,
            negative_embed, margin=self.margin, p=self.p, reduction='none')
        if not self.reduce:
            return loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)
        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]
        pos_loss = F.pairwise_distance(anchor_embed, positive_embed, p=2).pow(2
            )
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed,
            negative_embed, p=2)).clamp(min=0).pow(2)
        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):

    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(
            self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(
            self.beta)
        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:
            self.permute_len + 1]
        ordered_student = torch.gather(score_student, 1, permute_idx)
        log_prob = (ordered_student - torch.stack([torch.logsumexp(
            ordered_student[:, i:], dim=1) for i in range(permute_idx.size(
            1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()
        return loss


class FitNet(nn.Module):

    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)
        return (self.transform(student) - teacher).pow(2).mean()


class AttentionTransfer(nn.Module):

    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(
            0), -1))
        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.
                size(0), -1))
        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):

    def forward(self, student, teacher):
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class _Sampler(nn.Module):

    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        raise NotImplementedError


class GoogleNet(nn.Sequential):
    output_size = 1024
    input_side = 227
    rescale = 255.0
    rgb_mean = [122.7717, 115.9465, 102.9801]
    rgb_std = [1, 1, 1]
    url = (
        'https://github.com/vadimkantorov/metriclearningbench/releases/download/data/googlenet.h5'
        )
    md5hash = 'c7d7856bd1ab5cb02618b3f7f564e3c6'
    model_filename = 'googlenet.h5'

    def __init__(self, pretrained=True, root='data'):
        super(GoogleNet, self).__init__(OrderedDict([('conv1', nn.
            Sequential(OrderedDict([('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2,
            2), (3, 3))), ('relu1', nn.ReLU(True)), ('pool1', nn.MaxPool2d(
            (3, 3), (2, 2), ceil_mode=True)), ('lrn1', nn.CrossMapLRN2d(5, 
            0.0001, 0.75, 1))]))), ('conv2', nn.Sequential(OrderedDict([(
            '3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))), (
            'relu1', nn.ReLU(True)), ('3x3', nn.Conv2d(64, 192, (3, 3), (1,
            1), (1, 1))), ('relu2', nn.ReLU(True)), ('lrn2', nn.
            CrossMapLRN2d(5, 0.0001, 0.75, 1)), ('pool2', nn.MaxPool2d((3, 
            3), (2, 2), ceil_mode=True))]))), ('inception_3a',
            InceptionModule(192, 64, 96, 128, 16, 32, 32)), ('inception_3b',
            InceptionModule(256, 128, 128, 192, 32, 96, 64)), ('pool3', nn.
            MaxPool2d((3, 3), (2, 2), ceil_mode=True)), ('inception_4a',
            InceptionModule(480, 192, 96, 208, 16, 48, 64)), (
            'inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)
            ), ('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64,
            64)), ('inception_4d', InceptionModule(512, 112, 144, 288, 32, 
            64, 64)), ('inception_4e', InceptionModule(528, 256, 160, 320, 
            32, 128, 128)), ('pool4', nn.MaxPool2d((3, 3), (2, 2),
            ceil_mode=True)), ('inception_5a', InceptionModule(832, 256, 
            160, 320, 32, 128, 128)), ('inception_5b', InceptionModule(832,
            384, 192, 384, 48, 128, 128)), ('pool5', nn.AvgPool2d((7, 7), (
            1, 1), ceil_mode=True))]))
        if pretrained:
            self.load(root)

    def load(self, root):
        download_url(self.url, root, self.model_filename, self.md5hash)
        h5_file = h5py.File(os.path.join(root, self.model_filename), 'r')
        group_key = list(h5_file.keys())[0]
        self.load_state_dict({k: torch.from_numpy(v[group_key].value) for k,
            v in h5_file[group_key].items()})


class InceptionModule(nn.Module):

    def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce,
        outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5, outplane_pool_proj
        ):
        super(InceptionModule, self).__init__()
        a = nn.Sequential(OrderedDict([('1x1', nn.Conv2d(inplane,
            outplane_a1x1, (1, 1), (1, 1), (0, 0))), ('1x1_relu', nn.ReLU(
            True))]))
        b = nn.Sequential(OrderedDict([('3x3_reduce', nn.Conv2d(inplane,
            outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0))), ('3x3_relu1',
            nn.ReLU(True)), ('3x3', nn.Conv2d(outplane_b3x3_reduce,
            outplane_b3x3, (3, 3), (1, 1), (1, 1))), ('3x3_relu2', nn.ReLU(
            True))]))
        c = nn.Sequential(OrderedDict([('5x5_reduce', nn.Conv2d(inplane,
            outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0))), ('5x5_relu1',
            nn.ReLU(True)), ('5x5', nn.Conv2d(outplane_c5x5_reduce,
            outplane_c5x5, (5, 5), (1, 1), (2, 2))), ('5x5_relu2', nn.ReLU(
            True))]))
        d = nn.Sequential(OrderedDict([('pool_pool', nn.MaxPool2d((3, 3), (
            1, 1), (1, 1))), ('pool_proj', nn.Conv2d(inplane,
            outplane_pool_proj, (1, 1), (1, 1), (0, 0))), ('pool_relu', nn.
            ReLU(True))]))
        for container in [a, b, c, d]:
            for name, module in container.named_children():
                self.add_module(name, module)
        self.branches = [a, b, c, d]

    def forward(self, input):
        return torch.cat([branch(input) for branch in self.branches], 1)


pretrained_settings = {'bninception': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth',
    'input_space': 'BGR', 'input_size': [3, 224, 224], 'input_range': [0, 
    255], 'mean': [104, 117, 128], 'std': [1, 1, 1], 'num_classes': 1000}}}


def bninception(num_classes=1000, pretrained='imagenet'):
    """BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper.
    """
    model = BNInception(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['bninception'][pretrained]
        assert num_classes == settings['num_classes'
            ], 'num_classes should be {}, but is {}'.format(settings[
            'num_classes'], num_classes)
        weight = model_zoo.load_url(settings['url'])
        weight = {k: (v.squeeze(0) if v.size(0) == 1 else v) for k, v in
            weight.items()}
        model.load_state_dict(weight)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


class InceptionV1BN(nn.Module):
    output_size = 1024

    def __init__(self, pretrained=True):
        super(InceptionV1BN, self).__init__()
        self.model = bninception(num_classes=1000, pretrained='imagenet' if
            pretrained else None)

    def forward(self, input):
        x = self.model.features(input)
        x = self.model.global_pool(x)
        return x.view(x.size(0), -1)


class BNInception(nn.Module):

    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2,
            2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9,
            affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1,
            1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 
            1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9,
            affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1,
            1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2),
            dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3,
            3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2),
            dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1,
            1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1,
            1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=
            True, count_include_pad=True)
        self.last_linear = nn.Linear(1024, num_classes)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out
            )
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(
            conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out
            )
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(
            inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(
            pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(
            inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(
            inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(
            inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out
            )
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(
            inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = (self.
            inception_3a_double_3x3_reduce(pool2_3x3_s2_out))
        inception_3a_double_3x3_reduce_bn_out = (self.
            inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out))
        inception_3a_relu_double_3x3_reduce_out = (self.
            inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out))
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(
            inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = (self.
            inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out))
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(
            inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(
            inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = (self.
            inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out))
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(
            inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(
            inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(
            inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out,
            inception_3a_3x3_bn_out, inception_3a_double_3x3_2_bn_out,
            inception_3a_pool_proj_bn_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out
            )
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(
            inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(
            inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(
            inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(
            inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(
            inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out
            )
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(
            inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = (self.
            inception_3b_double_3x3_reduce(inception_3a_output_out))
        inception_3b_double_3x3_reduce_bn_out = (self.
            inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out))
        inception_3b_relu_double_3x3_reduce_out = (self.
            inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out))
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(
            inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = (self.
            inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out))
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(
            inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(
            inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = (self.
            inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out))
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(
            inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(
            inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(
            inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out,
            inception_3b_3x3_bn_out, inception_3b_double_3x3_2_bn_out,
            inception_3b_pool_proj_bn_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(
            inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(
            inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(
            inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(
            inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out
            )
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(
            inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = (self.
            inception_3c_double_3x3_reduce(inception_3b_output_out))
        inception_3c_double_3x3_reduce_bn_out = (self.
            inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out))
        inception_3c_relu_double_3x3_reduce_out = (self.
            inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out))
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = (self.
            inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out))
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(
            inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(
            inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = (self.
            inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out))
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_3x3_bn_out,
            inception_3c_double_3x3_2_bn_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out
            )
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(
            inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(
            inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(
            inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(
            inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(
            inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out
            )
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(
            inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = (self.
            inception_4a_double_3x3_reduce(inception_3c_output_out))
        inception_4a_double_3x3_reduce_bn_out = (self.
            inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out))
        inception_4a_relu_double_3x3_reduce_out = (self.
            inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out))
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(
            inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(
            inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = (self.
            inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out))
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(
            inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(
            inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = (self.
            inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out))
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(
            inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(
            inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(
            inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out,
            inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out,
            inception_4a_pool_proj_bn_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out
            )
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(
            inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(
            inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(
            inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(
            inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(
            inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out
            )
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(
            inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = (self.
            inception_4b_double_3x3_reduce(inception_4a_output_out))
        inception_4b_double_3x3_reduce_bn_out = (self.
            inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out))
        inception_4b_relu_double_3x3_reduce_out = (self.
            inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out))
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(
            inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(
            inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = (self.
            inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out))
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(
            inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(
            inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = (self.
            inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out))
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(
            inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(
            inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(
            inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out,
            inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out,
            inception_4b_pool_proj_bn_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out
            )
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(
            inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(
            inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(
            inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(
            inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(
            inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out
            )
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(
            inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = (self.
            inception_4c_double_3x3_reduce(inception_4b_output_out))
        inception_4c_double_3x3_reduce_bn_out = (self.
            inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out))
        inception_4c_relu_double_3x3_reduce_out = (self.
            inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out))
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(
            inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(
            inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = (self.
            inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out))
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(
            inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(
            inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = (self.
            inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out))
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(
            inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(
            inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(
            inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out,
            inception_4c_3x3_bn_out, inception_4c_double_3x3_2_bn_out,
            inception_4c_pool_proj_bn_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out
            )
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(
            inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(
            inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(
            inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(
            inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(
            inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out
            )
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(
            inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = (self.
            inception_4d_double_3x3_reduce(inception_4c_output_out))
        inception_4d_double_3x3_reduce_bn_out = (self.
            inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out))
        inception_4d_relu_double_3x3_reduce_out = (self.
            inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out))
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(
            inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(
            inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = (self.
            inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out))
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(
            inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(
            inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = (self.
            inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out))
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(
            inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(
            inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(
            inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out,
            inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out,
            inception_4d_pool_proj_bn_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(
            inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(
            inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(
            inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(
            inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out
            )
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(
            inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = (self.
            inception_4e_double_3x3_reduce(inception_4d_output_out))
        inception_4e_double_3x3_reduce_bn_out = (self.
            inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out))
        inception_4e_relu_double_3x3_reduce_out = (self.
            inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out))
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(
            inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(
            inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = (self.
            inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out))
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(
            inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(
            inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = (self.
            inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out))
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_3x3_bn_out,
            inception_4e_double_3x3_2_bn_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out
            )
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(
            inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(
            inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(
            inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(
            inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(
            inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out
            )
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(
            inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = (self.
            inception_5a_double_3x3_reduce(inception_4e_output_out))
        inception_5a_double_3x3_reduce_bn_out = (self.
            inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out))
        inception_5a_relu_double_3x3_reduce_out = (self.
            inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out))
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(
            inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(
            inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = (self.
            inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out))
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(
            inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(
            inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = (self.
            inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out))
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(
            inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(
            inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(
            inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out,
            inception_5a_3x3_bn_out, inception_5a_double_3x3_2_bn_out,
            inception_5a_pool_proj_bn_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out
            )
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(
            inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(
            inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(
            inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(
            inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(
            inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out
            )
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(
            inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = (self.
            inception_5b_double_3x3_reduce(inception_5a_output_out))
        inception_5b_double_3x3_reduce_bn_out = (self.
            inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out))
        inception_5b_relu_double_3x3_reduce_out = (self.
            inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out))
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(
            inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(
            inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = (self.
            inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out))
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(
            inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(
            inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = (self.
            inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out))
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(
            inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(
            inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(
            inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out,
            inception_5b_3x3_bn_out, inception_5b_double_3x3_2_bn_out,
            inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def logits(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1',
            'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        if get_ha:
            return b1, b2, b3, b4, pool
        return pool


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1',
            'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        if get_ha:
            return b1, b2, b3, b4, pool
        return pool


class LinearEmbedding(nn.Module):

    def __init__(self, base, output_size=512, embedding_size=128, normalize
        =True):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, x, get_ha=False):
        if get_ha:
            b1, b2, b3, b4, pool = self.base(x, True)
        else:
            pool = self.base(x)
        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        if get_ha:
            return b1, b2, b3, b4, pool, embedding
        return embedding


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lenscloth_RKD(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(HardDarkRank(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_001(self):
        self._check(FitNet(*[], **{'in_feature': 4, 'out_feature': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(AttentionTransfer(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(RKdAngle(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(RkdDistance(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(InceptionModule(*[], **{'inplane': 4, 'outplane_a1x1': 4, 'outplane_b3x3_reduce': 4, 'outplane_b3x3': 4, 'outplane_c5x5_reduce': 4, 'outplane_c5x5': 4, 'outplane_pool_proj': 4}), [torch.rand([4, 4, 4, 4])], {})

