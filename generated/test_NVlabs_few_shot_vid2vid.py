import sys
_module = sys.modules[__name__]
del sys
data = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
fewshot_face_dataset = _module
fewshot_pose_dataset = _module
fewshot_street_dataset = _module
image_folder = _module
keypoint2img = _module
lmdb_dataset = _module
download_youTube_playlist = _module
preprocess = _module
check_valid = _module
get_poses = _module
track = _module
util = _module
models = _module
base_model = _module
face_refiner = _module
flownet = _module
input_process = _module
loss_collector = _module
models = _module
networks = _module
architecture = _module
base_network = _module
discriminator = _module
flownet2_pytorch = _module
convert = _module
datasets = _module
losses = _module
main = _module
models = _module
FlowNetC = _module
FlowNetFusion = _module
FlowNetS = _module
FlowNetSD = _module
channelnorm_package = _module
channelnorm = _module
setup = _module
correlation_package = _module
correlation = _module
setup = _module
resample2d_package = _module
resample2d = _module
setup = _module
submodules = _module
utils = _module
flow_utils = _module
frame_utils = _module
param_utils = _module
tools = _module
generator = _module
loss = _module
normalization = _module
sync_batchnorm = _module
batchnorm = _module
comm = _module
replicate = _module
unittest = _module
vgg = _module
trainer = _module
vid2vid_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
download_datasets = _module
download_flownet2 = _module
download_gdrive = _module
test = _module
train = _module
distributed = _module
html = _module
image_pool = _module
util = _module
visualizer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.utils.data as data


import torchvision.transforms as transforms


import torch.utils.data


import torch.distributed as dist


import copy


import torch.nn.functional as F


import torch.nn as nn


import functools


import torch.nn.utils.spectral_norm as sn


from torch.nn import init


import math


from torch.utils.data import DataLoader


from torch.autograd import Variable


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import time


from inspect import isclass


import inspect


import re


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import torchvision


from collections import OrderedDict


def get_rank():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def is_master():
    """check if current process is the master"""
    return get_rank() == 0


class Visualizer:

    def __init__(self, opt):
        self.tf_log = opt.tf_log
        self.use_visdom = opt.use_visdom
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)
        if self.use_visdom:
            self.vis = visdom.Visdom()
            self.visdom_id = opt.visdom_id
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            None
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            if hasattr(opt, 'model_idx') and opt.model_idx != -1:
                self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_%03d.txt' % opt.model_idx)
            else:
                self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_visdom_results(self, visuals, epoch, step):
        ncols = self.ncols
        if ncols > 0:
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = """<style>
                    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)
            title = self.name
            label_html = ''
            label_html_row = ''
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                label_html_row += '<td>%s</td>' % label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % ncols != 0:
                images.append(white_image)
                label_html_row += '<td></td>'
                idx += 1
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row
            self.vis.images(images, nrow=ncols, win=self.visdom_id + 1, padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=self.visdom_id + 2, opts=dict(title=title + ' labels'))

    def display_current_results(self, visuals, epoch, step):
        if self.use_visdom:
            self.display_visdom_results(visuals, epoch, step)
        if self.tf_log:
            img_summaries = []
            for label, image_numpy in visuals.items():
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                scipy.misc.toimage(image_numpy).save(s, format='jpeg')
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)
        if self.use_html:
            for label, image_numpy in visuals.items():
                if image_numpy is None:
                    continue
                ext = 'png' if 'label' in label else 'jpg'
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%03d_iter%07d_%s_%d.%s' % (epoch, step, label, i, ext))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%03d_iter%07d_%s.%s' % (epoch, step, label, ext))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    util.save_image(image_numpy, img_path)
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=300)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visuals.items():
                    if image_numpy is None:
                        continue
                    ext = 'png' if 'label' in label else 'jpg'
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            if n == epoch:
                                img_path = 'epoch%03d_iter%07d_%s_%d.%s' % (n, step, label, i, ext)
                            else:
                                img_paths = sorted(glob.glob(os.path.join(self.img_dir, 'epoch%03d_iter*_%s_%d.%s' % (n, label, i, ext))))
                                img_path = os.path.basename(img_paths[-1]) if len(img_paths) else 'img.jpg'
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        if n == epoch:
                            img_path = 'epoch%03d_iter%07d_%s.%s' % (n, step, label, ext)
                        else:
                            img_paths = sorted(glob.glob(os.path.join(self.img_dir, 'epoch%03d_iter*_%s.%s' % (n, label, ext))))
                            img_path = os.path.basename(img_paths[-1]) if len(img_paths) else 'img.jpg'
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 6:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)
        None
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        webpage.add_header(name)
        ims = []
        txts = []
        links = []
        for label, image_numpy in visuals.items():
            if image_numpy is None:
                continue
            ext = 'png' if 'label' in label else 'jpg'
            image_name = os.path.join(label, '%s.%s' % (name, ext))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    @staticmethod
    def vis_print(opt, message):
        None
        if is_master() and opt.isTrain and not opt.debug:
            log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(log_name, 'a') as log_file:
                log_file.write('%s\n' % message)


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.old_lr = opt.lr
        self.pose = 'pose' in opt.dataset_mode
        self.face = 'face' in opt.dataset_mode
        self.street = 'street' in opt.dataset_mode
        self.warp_ref = opt.warp_ref
        self.has_fg = self.pose
        self.add_face_D = opt.add_face_D
        self.concat_ref_for_D = (opt.isTrain or opt.finetune) and opt.netD_subarch == 'n_layers'
        self.concat_fg_mask_for_D = self.has_fg

    def forward(self):
        pass

    def get_optimizer(self, params, for_discriminator=False):
        opt = self.opt
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, 0.999
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, opt.beta2
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        lr = D_lr if for_discriminator else G_lr
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            Visualizer.vis_print(self.opt, '%s not exists yet!' % save_path)
        else:
            try:
                loaded_weights = torch.load(save_path)
                network.load_state_dict(loaded_weights)
                Visualizer.vis_print(self.opt, 'network loaded from %s' % save_path)
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    Visualizer.vis_print(self.opt, 'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    Visualizer.vis_print(self.opt, 'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add('.'.join(k.split('.')[:2]))
                            if 'flow_network_temp' in k:
                                network.flow_temp_is_initalized = False
                    Visualizer.vis_print(self.opt, sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if self.isTrain and tensors[0].get_device() == 0:
            if remove_size == 0:
                return tensors
            if isinstance(tensors, list):
                return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]
            if tensors is None:
                return None
            if isinstance(tensors, torch.Tensor):
                tensors = tensors[remove_size:]
        return tensors

    def concat(self, tensors, dim=0):
        if tensors[0] is not None and tensors[1] is not None:
            if isinstance(tensors[0], list):
                tensors_cat = []
                for i in range(len(tensors[0])):
                    tensors_cat.append(self.concat([tensors[0][i], tensors[1][i]], dim=dim))
                return tensors_cat
            return torch.cat([tensors[0], tensors[1].unsqueeze(1)], dim=dim)
        elif tensors[1] is not None:
            if isinstance(tensors[1], list):
                return [(t.unsqueeze(1) if t is not None else None) for t in tensors[1]]
            return tensors[1].unsqueeze(1)
        return tensors[0]

    def reshape(self, tensors, for_temporal=False):
        if isinstance(tensors, list):
            return [self.reshape(tensor, for_temporal) for tensor in tensors]
        if tensors is None or type(tensors) != torch.Tensor or len(tensors.size()) <= 4:
            return tensors
        bs, t, ch, h, w = tensors.size()
        if not for_temporal:
            tensors = tensors.contiguous().view(-1, ch, h, w)
        elif not self.opt.isTrain:
            tensors = tensors.contiguous().view(bs, -1, h, w)
        else:
            nD = self.tD
            if t > nD:
                if t % nD == 0:
                    tensors = tensors.contiguous().view(-1, ch * nD, h, w)
                else:
                    n = t // nD
                    tensors = tensors[:, -n * nD:].contiguous().view(-1, ch * nD, h, w)
            else:
                tensors = tensors.contiguous().view(bs, ch * t, h, w)
        return tensors

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = [[tensor[:tensor.size(0) // 2] for tensor in p] for p in pred]
            real = [[tensor[tensor.size(0) // 2:] for tensor in p] for p in pred]
            return fake, real
        else:
            return torch.chunk(pred, 2, dim=0)

    def get_train_params(self, netG, train_names):
        train_list = set()
        params = []
        params_dict = netG.state_dict()
        for key, value in params_dict.items():
            do_train = False
            for model_name in train_names:
                if model_name in key:
                    do_train = True
            if do_train:
                module = netG
                key_list = key.split('.')
                for k in key_list:
                    module = getattr(module, k)
                params += [module]
                train_list.add('.'.join(key_list[:1]))
        Visualizer.vis_print(self.opt, ('training layers: ', train_list))
        return params, train_list

    def define_networks(self, start_epoch):
        opt = self.opt
        input_nc = opt.label_nc if opt.label_nc != 0 and not self.pose else opt.input_nc
        netG_input_nc = input_nc
        opt.for_face = False
        self.netG = networks.define_G(opt)
        if self.refine_face:
            opt_face = copy.deepcopy(opt)
            opt_face.n_downsample_G -= 1
            if opt_face.n_adaptive_layers > 0:
                opt_face.n_adaptive_layers -= 1
            opt_face.input_nc = opt.output_nc
            opt_face.fineSize = self.faceRefiner.face_size
            opt_face.aspect_ratio = 1
            opt_face.for_face = True
            self.netGf = networks.define_G(opt_face)
        if self.isTrain or opt.finetune:
            netD_input_nc = input_nc + opt.output_nc + (1 if self.concat_fg_mask_for_D else 0)
            if self.concat_ref_for_D:
                netD_input_nc *= 2
            self.netD = networks.define_D(opt, netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm_D, opt.netD_subarch, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            if self.add_face_D:
                self.netDf = networks.define_D(opt, opt.output_nc * 2, opt.ndf, opt.n_layers_D, opt.norm_D, 'n_layers', 1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            else:
                self.netDf = None
        self.temporal = False
        self.netDT = None
        Visualizer.vis_print(self.opt, '---------- Networks initialized -------------')
        if self.isTrain:
            params = list(self.netG.parameters())
            if self.refine_face:
                params += list(self.netGf.parameters())
            self.optimizer_G = self.get_optimizer(params, for_discriminator=False)
            params = list(self.netD.parameters())
            if self.add_face_D:
                params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)
        Visualizer.vis_print(self.opt, '---------- Optimizers initialized -------------')
        if (not opt.isTrain or start_epoch > opt.niter_single) and opt.n_frames_G > 1:
            self.init_temporal_model()

    def save_networks(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        if self.refine_face:
            self.save_network(self.netGf, 'Gf', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.temporal:
            self.save_network(self.netDT, 'DT', which_epoch, self.gpu_ids)
        if self.add_face_D:
            self.save_network(self.netDf, 'Df', which_epoch, self.gpu_ids)

    def load_networks(self):
        opt = self.opt
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain or opt.continue_train else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.temporal and opt.warp_ref and not self.netG.flow_temp_is_initalized:
                self.netG.load_pretrained_net(self.netG.flow_network_ref, self.netG.flow_network_temp)
            if self.refine_face:
                self.load_network(self.netGf, 'Gf', opt.which_epoch, pretrained_path)
            if self.isTrain and not opt.load_pretrain or opt.finetune:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                if self.isTrain and self.temporal:
                    self.load_network(self.netDT, 'DT', opt.which_epoch, pretrained_path)
                if self.add_face_D:
                    self.load_network(self.netDf, 'Df', opt.which_epoch, pretrained_path)

    def update_learning_rate(self, epoch):
        new_lr = self.opt.lr * (1 - (epoch - self.opt.niter) / (self.opt.niter_decay + 1))
        if self.opt.no_TTUR:
            G_lr, D_lr = new_lr, new_lr
        else:
            G_lr, D_lr = new_lr / 2, new_lr * 2
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = D_lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = G_lr
        Visualizer.vis_print(self.opt, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def init_temporal_model(self):
        opt = self.opt
        self.temporal = True
        self.netG.init_temporal_network()
        self.netG
        if opt.isTrain:
            self.lossCollector.tD = min(opt.n_frames_D, opt.n_frames_G)
            params = list(self.netG.parameters())
            if self.refine_face:
                params += list(self.netGf.parameters())
            self.optimizer_G = self.get_optimizer(params, for_discriminator=False)
            self.netDT = networks.define_D(opt, opt.output_nc * self.lossCollector.tD, opt.ndf, opt.n_layers_D, opt.norm_D, 'n_layers', 1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            params = list(self.netD.parameters()) + list(self.netDT.parameters())
            if self.add_face_D:
                params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)
            Visualizer.vis_print(self.opt, '---------- Now start training multiple frames -------------')


class FaceRefineModel(BaseModel):

    def name(self):
        return 'FaceRefineModel'

    def initialize(self, opt, add_face_D, refine_face):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.add_face_D = add_face_D
        self.refine_face = refine_face
        self.face_size = int(opt.fineSize / opt.aspect_ratio) // 4

    def refine_face_region(self, netGf, label_valid, fake_image, label, ref_label_valid, ref_image, ref_label):
        label_face, fake_face_coarse = self.crop_face_region([label_valid, fake_image], label, crop_smaller=4)
        ref_label_face, ref_image_face = self.crop_face_region([ref_label_valid, ref_image], ref_label, crop_smaller=4)
        fake_face = netGf(label_face, ref_label_face.unsqueeze(1), ref_image_face.unsqueeze(1), img_coarse=fake_face_coarse.detach())
        fake_image = self.replace_face_region(fake_image, fake_face, label, fake_face_coarse.detach(), crop_smaller=4)
        return fake_image

    def crop_face_region(self, image, input_label, crop_smaller=0):
        if type(image) == list:
            return [self.crop_face_region(im, input_label, crop_smaller) for im in image]
        for i in range(input_label.size(0)):
            ys, ye, xs, xe = self.get_face_region(input_label[i:i + 1], crop_smaller=crop_smaller)
            output_i = F.interpolate(image[i:i + 1, -3:, ys:ye, xs:xe], size=(self.face_size, self.face_size))
            output = torch.cat([output, output_i]) if i != 0 else output_i
        return output

    def replace_face_region(self, fake_image, fake_face, input_label, fake_face_coarse=None, crop_smaller=0):
        fake_image = fake_image.clone()
        b, _, h, w = input_label.size()
        for i in range(b):
            ys, ye, xs, xe = self.get_face_region(input_label[i:i + 1], crop_smaller)
            fake_face_i = fake_face[i:i + 1] + (fake_face_coarse[i:i + 1] if fake_face_coarse is not None else 0)
            fake_face_i = F.interpolate(fake_face_i, size=(ye - ys, xe - xs), mode='bilinear')
            fake_image[i:i + 1, :, ys:ye, xs:xe] = torch.clamp(fake_face_i, -1, 1)
        return fake_image

    def get_face_region(self, pose, crop_smaller=0):
        if pose.dim() == 3:
            pose = pose.unsqueeze(0)
        elif pose.dim() == 5:
            pose = pose[(-1), -1:]
        _, _, h, w = pose.size()
        use_openpose = not self.opt.basic_point_only and not self.opt.remove_face_labels
        if use_openpose:
            face = ((pose[:, (-3)] > 0) & (pose[:, (-2)] > 0) & (pose[:, (-1)] > 0)).nonzero()
        else:
            face = (pose[:, (2)] > 0.9).nonzero()
        if face.size(0):
            y, x = face[:, (1)], face[:, (2)]
            ys, ye, xs, xe = y.min().item(), y.max().item(), x.min().item(), x.max().item()
            if use_openpose:
                xc, yc = (xs + xe) // 2, (ys * 3 + ye * 2) // 5
                ylen = int((xe - xs) * 2.5)
            else:
                xc, yc = (xs + xe) // 2, (ys + ye) // 2
                ylen = int((ye - ys) * 1.25)
            ylen = xlen = min(w, max(32, ylen))
            yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
            xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
        else:
            yc = h // 4
            xc = w // 2
            ylen = xlen = h // 32 * 8
        ys, ye, xs, xe = yc - ylen // 2, yc + ylen // 2, xc - xlen // 2, xc + xlen // 2
        if crop_smaller != 0:
            ys += crop_smaller
            xs += crop_smaller
            ye -= crop_smaller
            xe -= crop_smaller
        return ys, ye, xs, xe


class FlowNet(BaseModel):

    def name(self):
        return 'FlowNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2']()
        checkpoint = torch.load('models/networks/flownet2_pytorch/FlowNet2_checkpoint.pth.tar', map_location=torch.device('cpu'))
        self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval()
        self.resample = Resample2d()
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, data_list, epoch=0, dummy_bs=0):
        if data_list[0].get_device() == 0:
            data_list = self.remove_dummy_from_tensor(data_list, dummy_bs)
        image_now, image_ref = data_list
        image_now, image_ref = image_now[:, :, :3], image_ref[:, 0:1, :3]
        flow_gt_prev = flow_gt_ref = conf_gt_prev = conf_gt_ref = None
        with torch.no_grad():
            if not self.opt.isTrain or epoch > self.opt.niter_single:
                image_prev = torch.cat([image_now[:, 0:1], image_now[:, :-1]], dim=1)
                flow_gt_prev, conf_gt_prev = self.flowNet_forward(image_now, image_prev)
            if self.opt.warp_ref:
                flow_gt_ref, conf_gt_ref = self.flowNet_forward(image_now, image_ref.expand_as(image_now))
            flow_gt, conf_gt = [flow_gt_ref, flow_gt_prev], [conf_gt_ref, conf_gt_prev]
            return flow_gt, conf_gt

    def flowNet_forward(self, input_A, input_B):
        size = input_A.size()
        assert len(size) == 4 or len(size) == 5
        if len(size) == 5:
            b, n, c, h, w = size
            input_A = input_A.contiguous().view(-1, c, h, w)
            input_B = input_B.contiguous().view(-1, c, h, w)
            flow, conf = self.compute_flow_and_conf(input_A, input_B)
            return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
        else:
            return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert im1.size()[1] == 3
        assert im1.size() == im2.size()
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h // 64 * 64, old_w // 64 * 64
        if old_h != new_h:
            im1 = F.interpolate(im1, size=(new_h, new_w), mode='bilinear')
            im2 = F.interpolate(im2, size=(new_h, new_w), mode='bilinear')
        self.flowNet
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)
        flow1 = self.flowNet(data1)
        conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float()
        if old_h != new_h:
            flow1 = F.interpolate(flow1, size=(old_h, old_w), mode='bilinear') * old_h / new_h
            conf = F.interpolate(conf, size=(old_h, old_w), mode='bilinear')
        return flow1, conf

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def get_face_mask(pose):
    if pose.dim() == 3:
        pose = pose.unsqueeze(1)
    b, t, h, w = pose.size()
    part = (pose / 2 + 0.5) * 24
    if pose.is_cuda:
        mask = torch.ByteTensor(b, t, h, w).fill_(0)
    else:
        mask = torch.ByteTensor(b, t, h, w).fill_(0)
    for j in [23, 24]:
        mask = mask | ((part > j - 0.1) & (part < j + 0.1)).byte()
    return mask.float()


def get_fg_mask(opt, input_label, has_fg):
    if type(input_label) == list:
        return [get_fg_mask(opt, l, has_fg) for l in input_label]
    if not has_fg:
        return None
    if len(input_label.size()) == 5:
        input_label = input_label[:, (0)]
    mask = input_label[:, 2:3] if opt.label_nc == 0 else -input_label[:, 0:1]
    mask = torch.nn.MaxPool2d(15, padding=7, stride=1)(mask)
    mask = (mask > -1).float()
    return mask


def get_part_mask(pose):
    part_groups = [[0], [1, 2], [3, 4], [5, 6], [7, 9, 8, 10], [11, 13, 12, 14], [15, 17, 16, 18], [19, 21, 20, 22], [23, 24]]
    n_parts = len(part_groups)
    need_reshape = pose.dim() == 4
    if need_reshape:
        bo, t, h, w = pose.size()
        pose = pose.view(-1, h, w)
    b, h, w = pose.size()
    part = (pose / 2 + 0.5) * 24
    mask = torch.ByteTensor(b, n_parts, h, w).fill_(0)
    for i in range(n_parts):
        for j in part_groups[i]:
            mask[:, (i)] = mask[:, (i)] | ((part > j - 0.1) & (part < j + 0.1)).byte()
    if need_reshape:
        mask = mask.view(bo, t, -1, h, w)
    return mask.float()


def get_grid(batchsize, rows, cols, gpu_id=0):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)
    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False
    return t_grid


def resample(image, flow):
    b, c, h, w = image.size()
    grid = get_grid(b, h, w, gpu_id=flow.get_device())
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    try:
        output = F.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
    except:
        output = F.grid_sample(image, final_grid, mode='bilinear', padding_mode='border')
    return output


def use_valid_labels(opt, pose):
    if 'pose' not in opt.dataset_mode:
        return pose
    if pose is None:
        return pose
    if type(pose) == list:
        return [use_valid_labels(opt, p) for p in pose]
    assert pose.dim() == 4 or pose.dim() == 5
    if opt.pose_type == 'open':
        if pose.dim() == 4:
            pose = pose[:, 3:]
        elif pose.dim() == 5:
            pose = pose[:, :, 3:]
    elif opt.remove_face_labels:
        if pose.dim() == 4:
            face_mask = get_face_mask(pose[:, (2)])
            pose = torch.cat([pose[:, :3] * (1 - face_mask) - face_mask, pose[:, 3:]], dim=1)
        else:
            face_mask = get_face_mask(pose[:, :, (2)]).unsqueeze(2)
            pose = torch.cat([pose[:, :, :3] * (1 - face_mask) - face_mask, pose[:, :, 3:]], dim=2)
    return pose


class LossCollector(BaseModel):

    def name(self):
        return 'LossCollector'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.define_losses()
        self.tD = 1

    def define_losses(self):
        opt = self.opt
        if self.isTrain or opt.finetune:
            self.fake_pool = ImagePool(0)
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionFlow = networks.MaskedL1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(opt, self.gpu_ids)
            self.loss_names_G = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'Gf_GAN', 'Gf_GAN_feat', 'GT_GAN', 'GT_GAN_Feat', 'F_Flow', 'F_Warp', 'F_Mask']
            self.loss_names_D = ['D_real', 'D_fake', 'Df_real', 'Df_fake', 'DT_real', 'DT_fake']
            self.loss_names = self.loss_names_G + self.loss_names_D

    def discriminate(self, netD, tgt_label, fake_image, tgt_image, ref_image, for_discriminator):
        tgt_concat = torch.cat([fake_image, tgt_image], dim=0)
        if tgt_label is not None:
            tgt_concat = torch.cat([tgt_label.repeat(2, 1, 1, 1), tgt_concat], dim=1)
        if ref_image is not None:
            ref_image = ref_image.repeat(2, 1, 1, 1)
            if self.concat_ref_for_D:
                tgt_concat = torch.cat([ref_image, tgt_concat], dim=1)
                ref_image = None
        discriminator_out = netD(tgt_concat, ref_image)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        if for_discriminator:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            return [loss_D_real, loss_D_fake]
        else:
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_GAN_Feat = self.GAN_matching_loss(pred_real, pred_fake, for_discriminator)
            return [loss_G_GAN, loss_G_GAN_Feat]

    def discriminate_face(self, netDf, fake_image, tgt_label, tgt_image, ref_label, ref_image, faceRefiner, for_discriminator):
        losses = [self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)]
        if self.add_face_D:
            real_region, fake_region = faceRefiner.crop_face_region([tgt_image, fake_image], tgt_label)
            ref_region = faceRefiner.crop_face_region(ref_image, ref_label)
            losses = self.discriminate(netDf, ref_region, fake_region, real_region, None, for_discriminator=for_discriminator)
            losses = [(loss * self.opt.lambda_face) for loss in losses]
            if for_discriminator:
                return losses
            else:
                loss_Gf_GAN, loss_Gf_GAN_Feat = losses
                loss_Gf_GAN_Feat += self.criterionFeat(fake_region, real_region) * self.opt.lambda_feat
                loss_Gf_GAN_Feat += self.criterionVGG(fake_region, real_region) * self.opt.lambda_vgg
            return [loss_Gf_GAN, loss_Gf_GAN_Feat]
        return losses

    def compute_GAN_losses(self, nets, data_list, for_discriminator, for_temporal=False):
        if for_temporal and self.tD < 2:
            return [self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)]
        tgt_label, tgt_image, fake_image, ref_label, ref_image = data_list
        netD, netDT, netDf, faceRefiner = nets
        if isinstance(fake_image, list):
            fake_image = [x for x in fake_image if x is not None]
            losses = [self.compute_GAN_losses(nets, [tgt_label, real_i, fake_i, ref_label, ref_image], for_discriminator, for_temporal) for fake_i, real_i in zip(fake_image, tgt_image)]
            return [sum([item[i] for item in losses]) for i in range(len(losses[0]))]
        tgt_label, tgt_image, fake_image = self.reshape([tgt_label, tgt_image, fake_image], for_temporal)
        input_label = ref_concat = None
        if not for_temporal:
            t = self.opt.n_frames_per_gpu
            ref_label, ref_image = ref_label.repeat(t, 1, 1, 1), ref_image.repeat(t, 1, 1, 1)
            input_label = use_valid_labels(self.opt, tgt_label)
            if self.concat_fg_mask_for_D:
                fg_mask, ref_fg_mask = get_fg_mask(self.opt, [tgt_label, ref_label], self.has_fg)
                input_label = torch.cat([input_label, fg_mask], dim=1)
                ref_label = torch.cat([ref_label, ref_fg_mask], dim=1)
            ref_concat = torch.cat([ref_label, ref_image], dim=1)
        netD = netD if not for_temporal else netDT
        losses = self.discriminate(netD, input_label, fake_image, tgt_image, ref_concat, for_discriminator=for_discriminator)
        if for_temporal:
            if not for_discriminator:
                losses = [(loss * self.opt.lambda_temp) for loss in losses]
            return losses
        losses_face = self.discriminate_face(netDf, fake_image, tgt_label, tgt_image, ref_label, ref_image, faceRefiner, for_discriminator)
        return losses + losses_face

    def compute_VGG_losses(self, fake_image, fake_raw_image, tgt_image, fg_mask_union):
        loss_G_VGG = self.Tensor(1).fill_(0)
        opt = self.opt
        if not opt.no_vgg_loss:
            if fake_image is not None:
                loss_G_VGG = self.criterionVGG(fake_image, tgt_image)
            if fake_raw_image is not None:
                loss_G_VGG += self.criterionVGG(fake_raw_image, tgt_image * fg_mask_union)
        return loss_G_VGG * opt.lambda_vgg

    def compute_flow_losses(self, flow, warped_image, tgt_image, flow_gt, flow_conf_gt, fg_mask, tgt_label, ref_label):
        loss_F_Flow_r, loss_F_Warp_r = self.compute_flow_loss(flow[0], warped_image[0], tgt_image, flow_gt[0], flow_conf_gt[0], fg_mask)
        loss_F_Flow_p, loss_F_Warp_p = self.compute_flow_loss(flow[1], warped_image[1], tgt_image, flow_gt[1], flow_conf_gt[1], fg_mask)
        loss_F_Flow = loss_F_Flow_p + loss_F_Flow_r
        loss_F_Warp = loss_F_Warp_p + loss_F_Warp_r
        lambda_flow = self.opt.lambda_flow
        body_mask_diff = None
        if self.opt.isTrain and self.pose and flow[0] is not None:
            body_mask = get_part_mask(tgt_label[:, :, (2)])
            ref_body_mask = get_part_mask(ref_label[:, (2)].unsqueeze(1)).expand_as(body_mask)
            body_mask, ref_body_mask = self.reshape([body_mask, ref_body_mask])
            ref_body_mask_warp = resample(ref_body_mask, flow[0])
            loss_F_Warp += self.criterionFeat(ref_body_mask_warp, body_mask)
            if self.has_fg:
                fg_mask, ref_fg_mask = get_fg_mask(self.opt, [tgt_label, ref_label], True)
                ref_fg_mask_warp = resample(ref_fg_mask, flow[0])
                loss_F_Warp += self.criterionFeat(ref_fg_mask_warp, fg_mask)
            body_mask_diff = torch.sum(abs(ref_body_mask_warp - body_mask), dim=1, keepdim=True)
        return loss_F_Flow * lambda_flow, loss_F_Warp * lambda_flow, body_mask_diff

    def compute_flow_loss(self, flow, warped_image, tgt_image, flow_gt, flow_conf_gt, fg_mask):
        loss_F_Flow, loss_F_Warp = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        if self.opt.isTrain and flow is not None:
            if flow_gt is not None and self.opt.n_shot == 1:
                loss_F_Flow = self.criterionFlow(flow, flow_gt, flow_conf_gt * fg_mask)
            loss_F_Warp = self.criterionFeat(warped_image, tgt_image)
        return loss_F_Flow, loss_F_Warp

    def compute_mask_losses(self, flow_mask, fake_image, warped_image, tgt_label, tgt_image, fake_raw_image, fg_mask, ref_fg_mask, body_mask_diff):
        fake_raw_image = fake_raw_image[:, (-1)] if fake_raw_image is not None else None
        loss_mask = self.Tensor(1).fill_(0)
        loss_mask += self.compute_mask_loss(flow_mask[0], warped_image[0], tgt_image, fake_image[:, (-1)], fake_raw_image)
        loss_mask += self.compute_mask_loss(flow_mask[1], warped_image[1], tgt_image, fake_image[:, (-1)], fake_raw_image)
        opt = self.opt
        if opt.isTrain and self.pose and self.warp_ref:
            flow_mask_ref = flow_mask[0]
            b, t, _, h, w = tgt_label.size()
            dummy0, dummy1 = torch.zeros_like(flow_mask_ref), torch.ones_like(flow_mask_ref)
            face_mask = get_face_mask(tgt_label[:, :, (2)]).view(-1, 1, h, w)
            face_mask = torch.nn.AvgPool2d(15, padding=7, stride=1)(face_mask)
            loss_mask += self.criterionFlow(flow_mask_ref, dummy0, face_mask)
            if opt.spade_combine:
                loss_mask += self.criterionFlow(fake_image[:, (-1)], warped_image[0].detach(), face_mask)
            fg_mask_diff = (ref_fg_mask - fg_mask > 0).float()
            loss_mask += self.criterionFlow(flow_mask_ref, dummy1, fg_mask_diff)
            loss_mask += self.criterionFlow(flow_mask_ref, dummy1, body_mask_diff)
        return loss_mask * opt.lambda_mask

    def compute_mask_loss(self, flow_mask, warped_image, tgt_image, fake_image, fake_raw_image):
        loss_mask = 0
        if self.opt.isTrain and flow_mask is not None:
            dummy0 = torch.zeros_like(flow_mask)
            dummy1 = torch.ones_like(flow_mask)
            img_diff = torch.sum(abs(warped_image - tgt_image), dim=1, keepdim=True)
            conf = torch.clamp(1 - img_diff, 0, 1)
            loss_mask = self.criterionFlow(flow_mask, dummy0, conf)
            loss_mask += self.criterionFlow(flow_mask, dummy1, 1 - conf)
        return loss_mask

    def GAN_matching_loss(self, pred_real, pred_fake, for_discriminator=False):
        loss_G_GAN_Feat = self.Tensor(1).fill_(0)
        if not for_discriminator and not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            D_masks = 1.0 / num_D
            for i in range(num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    loss_G_GAN_Feat += D_masks * loss
        return loss_G_GAN_Feat * self.opt.lambda_feat


class DataParallel(nn.parallel.DataParallel):

    def replicate(self, module, device_ids):
        replicas = super(DataParallel, self).replicate(module, device_ids)
        replicas[0] = module
        return replicas


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class MyModel(nn.Module):

    def __init__(self, opt, model):
        super(MyModel, self).__init__()
        self.opt = opt
        model = model
        self.module = model
        self.model = DataParallelWithCallback(model, device_ids=opt.gpu_ids)
        if opt.batch_for_first_gpu != -1:
            self.bs_per_gpu = (opt.batchSize - opt.batch_for_first_gpu) // (len(opt.gpu_ids) - 1)
        else:
            self.bs_per_gpu = int(np.ceil(float(opt.batchSize) / len(opt.gpu_ids)))
        self.pad_bs = self.bs_per_gpu * len(opt.gpu_ids) - opt.batchSize

    def forward(self, *inputs, **kwargs):
        inputs = self.add_dummy_to_tensor(inputs, self.pad_bs)
        outputs = self.model(*inputs, **kwargs, dummy_bs=self.pad_bs)
        if self.pad_bs == self.bs_per_gpu:
            return self.remove_dummy_from_tensor(outputs, 1)
        return outputs

    def add_dummy_to_tensor(self, tensors, add_size=0):
        if add_size == 0 or tensors is None:
            return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.add_dummy_to_tensor(tensor, add_size) for tensor in tensors]
        if isinstance(tensors, torch.Tensor):
            dummy = torch.zeros_like(tensors)[:add_size]
            tensors = torch.cat([dummy, tensors])
        return tensors

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if remove_size == 0 or tensors is None:
            return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]
        if isinstance(tensors, torch.Tensor):
            tensors = tensors[remove_size:]
        return tensors


def actvn(x):
    out = F.leaky_relu(x, 0.2)
    return out


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


def concat(a, b, dim=0):
    if isinstance(a, list):
        return [concat(ai, bi, dim) for ai, bi in zip(a, b)]
    if a is None:
        return b
    return torch.cat([a, b], dim=dim)


def batch_conv(x, weight, bias=None, stride=1, group_size=-1):
    if weight is None:
        return x
    if isinstance(weight, list) or isinstance(weight, tuple):
        weight, bias = weight
    padding = weight.size()[-1] // 2
    groups = group_size // weight.size()[2] if group_size != -1 else 1
    if bias is None:
        bias = [None] * x.size()[0]
    y = None
    for i in range(x.size(0)):
        if stride >= 1:
            yi = F.conv2d(x[i:i + 1], weight=weight[i], bias=bias[i], padding=padding, stride=stride, groups=groups)
        else:
            yi = F.conv_transpose2d(x[i:i + 1], weight=weight[i], bias=bias[(i), :weight.size(2)], padding=padding, stride=int(1 / stride), output_padding=1, groups=groups)
        y = concat(y, yi)
    return y


class SPADE(nn.Module):

    def __init__(self, norm_nc, hidden_nc=0, norm='batch', ks=3, params_free=False):
        super().__init__()
        pw = ks // 2
        if not isinstance(hidden_nc, list):
            hidden_nc = [hidden_nc]
        for i, nhidden in enumerate(hidden_nc):
            mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            if not params_free or i != 0:
                s = str(i + 1) if i > 0 else ''
                setattr(self, 'mlp_gamma%s' % s, mlp_gamma)
                setattr(self, 'mlp_beta%s' % s, mlp_beta)
        if 'batch' in norm:
            self.norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        else:
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

    def forward(self, x, maps, weights=None):
        if not isinstance(maps, list):
            maps = [maps]
        out = self.norm(x)
        for i in range(len(maps)):
            if maps[i] is None:
                continue
            m = F.interpolate(maps[i], size=x.size()[2:])
            if weights is None or i != 0:
                s = str(i + 1) if i > 0 else ''
                gamma = getattr(self, 'mlp_gamma%s' % s)(m)
                beta = getattr(self, 'mlp_beta%s' % s)(m)
            else:
                j = min(i, len(weights[0]) - 1)
                gamma = batch_conv(m, weights[0][j])
                beta = batch_conv(m, weights[1][j])
            out = out * (1 + gamma) + beta
        return out


def generalNorm(norm):
    if 'spade' in norm:
        return SPADE

    def get_norm(norm):
        if 'instance' in norm:
            return nn.InstanceNorm2d
        elif 'syncbatch' in norm:
            return SynchronizedBatchNorm2d
        elif 'batch' in norm:
            return nn.BatchNorm2d
    norm = get_norm(norm)


    class NormalNorm(norm):

        def __init__(self, *args, hidden_nc=0, norm='', ks=1, params_free=False, **kwargs):
            super(NormalNorm, self).__init__(*args, **kwargs)

        def forward(self, input, label=None, weight=None):
            return super(NormalNorm, self).forward(input)
    return NormalNorm


class SPADEConv2d(nn.Module):

    def __init__(self, fin, fout, norm='batch', hidden_nc=0, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = sn(nn.Conv2d(fin, fout, kernel_size=kernel_size, stride=stride, padding=padding))
        Norm = generalNorm(norm)
        self.bn = Norm(fout, hidden_nc=hidden_nc, norm=norm, ks=3)

    def forward(self, x, label=None):
        x = self.conv(x)
        out = self.bn(x, label)
        out = actvn(out)
        return out


def generalConv(adaptive=False, transpose=False):


    class NormalConv2d(nn.Conv2d):

        def __init__(self, *args, **kwargs):
            super(NormalConv2d, self).__init__(*args, **kwargs)

        def forward(self, input, weight=None, bias=None, stride=1):
            return super(NormalConv2d, self).forward(input)


    class NormalConvTranspose2d(nn.ConvTranspose2d):

        def __init__(self, *args, output_padding=1, **kwargs):
            super(NormalConvTranspose2d, self).__init__(*args, **kwargs)

        def forward(self, input, weight=None, bias=None, stride=1):
            return super(NormalConvTranspose2d, self).forward(input)


    class AdaptiveConv2d(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, input, weight=None, bias=None, stride=1):
            return batch_conv(input, weight, bias, stride)
    if adaptive:
        return AdaptiveConv2d
    return NormalConv2d if not transpose else NormalConvTranspose2d


class SPADEResnetBlock(nn.Module):

    def __init__(self, fin, fout, norm='batch', hidden_nc=0, conv_ks=3, spade_ks=1, stride=1, conv_params_free=False, norm_params_free=False):
        super().__init__()
        fhidden = min(fin, fout)
        self.learned_shortcut = fin != fout
        self.stride = stride
        Conv2d = generalConv(adaptive=conv_params_free)
        sn_ = sn if not conv_params_free else lambda x: x
        self.conv_0 = sn_(Conv2d(fin, fhidden, conv_ks, stride=stride, padding=1))
        self.conv_1 = sn_(Conv2d(fhidden, fout, conv_ks, padding=1))
        if self.learned_shortcut:
            self.conv_s = sn_(Conv2d(fin, fout, 1, stride=stride, bias=False))
        Norm = generalNorm(norm)
        self.bn_0 = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=spade_ks, params_free=norm_params_free)
        self.bn_1 = Norm(fhidden, hidden_nc=hidden_nc, norm=norm, ks=spade_ks, params_free=norm_params_free)
        if self.learned_shortcut:
            self.bn_s = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=spade_ks, params_free=norm_params_free)

    def forward(self, x, label=None, conv_weights=[], norm_weights=[]):
        if not conv_weights:
            conv_weights = [None] * 3
        if not norm_weights:
            norm_weights = [None] * 3
        x_s = self._shortcut(x, label, conv_weights[2], norm_weights[2])
        dx = self.conv_0(actvn(self.bn_0(x, label, norm_weights[0])), conv_weights[0])
        dx = self.conv_1(actvn(self.bn_1(dx, label, norm_weights[1])), conv_weights[1])
        out = x_s + 1.0 * dx
        return out

    def _shortcut(self, x, label, conv_weights, norm_weights):
        if self.learned_shortcut:
            x_s = self.conv_s(self.bn_s(x, label, norm_weights), conv_weights)
        elif self.stride != 1:
            x_s = nn.AvgPool2d(3, stride=2, padding=1)(x)
        else:
            x_s = x
        return x_s


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        None
        None

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def load_pretrained_net(self, net_src, net_dst):
        source_weights = net_src.state_dict()
        target_weights = net_dst.state_dict()
        for k, v in source_weights.items():
            if k in target_weights and target_weights[k].size() == v.size():
                target_weights[k] = v
        net_dst.load_state_dict(target_weights)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return z

    def sum(self, x):
        if type(x) != list:
            return x
        return sum([self.sum(xi) for xi in x])

    def sum_mul(self, x):
        assert type(x) == list
        if type(x[0]) != list:
            return np.prod(x) + x[0]
        return [self.sum_mul(xi) for xi in x]

    def split_weights(self, weight, sizes):
        if isinstance(sizes, list):
            weights = []
            cur_size = 0
            for i in range(len(sizes)):
                next_size = cur_size + self.sum(sizes[i])
                weights.append(self.split_weights(weight[:, cur_size:next_size], sizes[i]))
                cur_size = next_size
            assert next_size == weight.size()[1]
            return weights
        return weight

    def reshape_weight(self, x, weight_size):
        if type(weight_size[0]) == list and type(x) != list:
            x = self.split_weights(x, self.sum_mul(weight_size))
        if type(x) == list:
            return [self.reshape_weight(xi, wi) for xi, wi in zip(x, weight_size)]
        weight_size = [x.size()[0]] + weight_size
        bias_size = weight_size[1]
        try:
            weight = x[:, :-bias_size].view(weight_size)
            bias = x[:, -bias_size:]
        except:
            weight = x.view(weight_size)
            bias = None
        return [weight, bias]

    def reshape_embed_input(self, x):
        if isinstance(x, list):
            return [self.reshape_embed_input(xi) for xi in zip(x)]
        b, c, _, _ = x.size()
        x = x.view(b * c, -1)
        return x


class AdaptiveDiscriminator(BaseNetwork):

    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False, adaptive_layers=1):
        super(AdaptiveDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.adaptive_layers = adaptive_layers
        self.input_nc = input_nc
        self.ndf = ndf
        self.kw = kw = 4
        self.padw = padw = int(np.ceil((kw - 1.0) / 2))
        self.actvn = actvn = nn.LeakyReLU(0.2, True)
        self.sw = opt.fineSize // 8
        self.sh = int(self.sw / opt.aspect_ratio)
        self.ch = self.sh * self.sw
        nf = ndf
        self.fc_0 = nn.Linear(self.ch, input_nc * kw ** 2)
        self.encoder_0 = nn.Sequential(*[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), actvn])
        for n in range(1, self.adaptive_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            setattr(self, 'fc_' + str(n), nn.Linear(self.ch, nf_prev * kw ** 2))
            setattr(self, 'encoder_' + str(n), nn.Sequential(*[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), actvn]))
        sequence = []
        nf = ndf * 2 ** (self.adaptive_layers - 1)
        for n in range(self.adaptive_layers, n_layers + 1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 2 if n != n_layers else 1
            item = [norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), actvn]
            sequence += [item]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n + self.adaptive_layers), nn.Sequential(*sequence[n]))

    def gen_conv_weights(self, encoded_ref):
        models = []
        b = encoded_ref[0].size()[0]
        nf = self.ndf
        actvn = self.actvn
        weight = self.fc_0(nn.AdaptiveAvgPool2d((self.sh, self.sw))(encoded_ref[0]).view(b * nf, -1))
        weight = weight.view(b, nf, self.input_nc, self.kw, self.kw)
        model0 = []
        for i in range(b):
            model0.append(self.ConvN(functools.partial(F.conv2d, weight=weight[i], stride=2, padding=self.padw), nn.InstanceNorm2d(nf), actvn))
        models.append(model0)
        for n in range(1, self.adaptive_layers):
            ch = encoded_ref[n].size()[1]
            x = nn.AdaptiveAvgPool2d((self.sh, self.sw))(encoded_ref[n]).view(b * ch, -1)
            weight = getattr(self, 'fc_' + str(n))(x)
            nf_prev = nf
            nf = min(nf * 2, 512)
            weight = weight.view(b, nf, nf_prev, self.kw, self.kw)
            model = []
            for i in range(b):
                model.append(self.ConvN(functools.partial(F.conv2d, weight=weight[i], stride=2, padding=self.padw), nn.InstanceNorm2d(nf), actvn))
            models.append(model)
        return models


    class ConvN(nn.Module):

        def __init__(self, conv, norm, actvn):
            super().__init__()
            self.conv = conv
            self.norm = norm
            self.actvn = actvn

        def forward(self, x):
            x = self.conv(x)
            out = self.norm(x)
            out = self.actvn(out)
            return out

    def encode(self, ref):
        encoded_ref = [ref]
        for n in range(self.adaptive_layers):
            encoded_ref.append(getattr(self, 'encoder_' + str(n))(encoded_ref[-1]))
        return encoded_ref[1:]

    def batch_conv(self, conv, x):
        y = conv[0](x[0:1])
        for i in range(1, x.size()[0]):
            yi = conv[i](x[i:i + 1])
            y = torch.cat([y, yi])
        return y

    def forward(self, input, ref):
        encoded_ref = self.encode(ref)
        models = self.gen_conv_weights(encoded_ref)
        res = [input]
        for n in range(self.n_layers + 2):
            if n < self.adaptive_layers:
                res.append(self.batch_conv(models[n], res[-1]))
            else:
                res.append(getattr(self, 'model' + str(n))(res[-1]))
        if self.getIntermFeat:
            return res[1:]
        else:
            return res[-1]


class NLayerDiscriminator(BaseNetwork):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False, stride=2):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, False)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            item = [norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), nn.LeakyReLU(0.2, False)]
            sequence += [item]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)), nn.LeakyReLU(0.2, False)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            x = model(res[-1])
            res.append(x)
        if self.getIntermFeat:
            return res[1:]
        else:
            return res[-1]


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, subarch='n_layers', num_D=3, getIntermFeat=False, stride=2, gpu_ids=[]):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.getIntermFeat = getIntermFeat
        self.subarch = subarch
        for i in range(num_D):
            netD = self.create_singleD(opt, subarch, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride)
            setattr(self, 'discriminator_%d' % i, netD)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def create_singleD(self, opt, subarch, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride):
        if subarch == 'adaptive':
            netD = AdaptiveDiscriminator(opt, input_nc, ndf, n_layers, norm_layer, getIntermFeat, opt.adaptive_D_layers)
        elif subarch == 'n_layers':
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride)
        else:
            raise ValueError('unrecognized discriminator sub architecture %s' % subarch)
        return netD

    def singleD_forward(self, model, input, ref):
        if self.subarch == 'adaptive':
            return model(input, ref)
        elif self.getIntermFeat:
            return model(input)
        else:
            return [model(input)]

    def forward(self, input, ref=None):
        result = []
        input_downsampled = input
        ref_downsampled = ref
        for i in range(self.num_D):
            model = getattr(self, 'discriminator_%d' % i)
            result.append(self.singleD_forward(model, input_downsampled, ref_downsampled))
            input_downsampled = self.downsample(input_downsampled)
            ref_downsampled = self.downsample(ref_downsampled) if ref is not None else None
        return result


class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):

    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target, p=2, dim=1).mean()
        return lossvalue


def EPE(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


class L1Loss(nn.Module):

    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class L2Loss(nn.Module):

    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class MultiScale(nn.Module):

    def __init__(self, args, startScale=4, numScales=5, l_weight=0.32, norm='L1'):
        super(MultiScale, self).__init__()
        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert len(self.loss_weights) == self.numScales
        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()
        self.multiScales = [nn.AvgPool2d(self.startScale * 2 ** scale, self.startScale * 2 ** scale) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-' + self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i] * EPE(output_, target_)
                lossvalue += self.loss_weights[i] * self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return [lossvalue, epevalue]


class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, _, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()
        channelnorm_cuda.forward(input1, output, norm_deg)
        ctx.save_for_backward(input1, output)
        ctx.norm_deg = norm_deg
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        channelnorm.backward(input1, output, grad_output.data, grad_input1.data, ctx.norm_deg)
        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return result


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True), nn.LeakyReLU(0.1, inplace=True))


def deconv(in_planes, out_planes):
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class FlowNetC(nn.Module):

    def __init__(self, args, batchNorm=True, div_flow=20):
        super(FlowNetC, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)
        if args.fp16:
            self.corr = nn.Sequential(tofp32(), Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1), tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:, :, :]
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)
        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias), nn.BatchNorm2d(out_planes))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias))


class FlowNetFusion(nn.Module):

    def __init__(self, args, batchNorm=True):
        super(FlowNetFusion, self).__init__()
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 11, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)
        self.inter_conv1 = i_conv(self.batchNorm, 162, 32)
        self.inter_conv0 = i_conv(self.batchNorm, 82, 16)
        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)
        return flow0


class FlowNetS(nn.Module):

    def __init__(self, args, input_channels=12, batchNorm=True):
        super(FlowNetS, self).__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class FlowNetSD(nn.Module):

    def __init__(self, args, batchNorm=True):
        super(FlowNetSD, self).__init__()
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 6, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.inter_conv5 = i_conv(self.batchNorm, 1026, 512)
        self.inter_conv4 = i_conv(self.batchNorm, 770, 256)
        self.inter_conv3 = i_conv(self.batchNorm, 386, 128)
        self.inter_conv2 = i_conv(self.batchNorm, 194, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class MyDict(dict):
    pass


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()
        resample2d_cuda.forward(input1, input2, output, kernel_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()
        input1, input2 = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())
        resample2d_cuda.backward(input1, input2, grad_output.data, grad_input1.data, grad_input2.data, ctx.kernel_size)
        return grad_input1, grad_input2, None


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)


class FlowNet2(nn.Module):

    def __init__(self, args=None, batchNorm=False, div_flow=20.0):
        super(FlowNet2, self).__init__()
        if args is None:
            args = MyDict()
            args.rgb_max = 1
            args.fp16 = False
            args.grads = {}
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        if args.fp16:
            self.resample3 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample3 = Resample2d()
        if args.fp16:
            self.resample4 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample4 = Resample2d()
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=self.batchNorm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.0)
        for i in range(min_dim):
            weight.data[(i), (i), :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        diff_flownets2_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownets2_flow)
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        diff_flownetsd_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownetsd_flow)
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)
        return flownetfusion_flow


class FlowNet2CS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow=20.0):
        super(FlowNet2CS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow=20.0):
        super(FlowNet2CSS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow


def get_nonspade_norm_layer(opt, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = sn(layer)
            subnorm_type = norm_type[len('spectral'):]
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'syncbatch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=True)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer


class FlowGenerator(BaseNetwork):

    def __init__(self, opt, n_frames_G):
        super().__init__()
        self.opt = opt
        input_nc = (opt.label_nc if opt.label_nc != 0 else opt.input_nc) * n_frames_G
        input_nc += opt.output_nc * (n_frames_G - 1)
        nf = opt.nff
        n_blocks = opt.n_blocks_F
        n_downsample_F = opt.n_downsample_F
        self.flow_multiplier = opt.flow_multiplier
        nf_max = 1024
        ch = [min(nf_max, nf * 2 ** i) for i in range(n_downsample_F + 1)]
        norm = opt.norm_F
        norm_layer = get_nonspade_norm_layer(opt, norm)
        activation = nn.LeakyReLU(0.2)
        down_flow = [norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1)), activation]
        for i in range(n_downsample_F):
            down_flow += [norm_layer(nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, padding=1, stride=2)), activation]
        res_flow = []
        ch_r = min(nf_max, nf * 2 ** n_downsample_F)
        for i in range(n_blocks):
            res_flow += [SPADEResnetBlock(ch_r, ch_r, norm=norm)]
        up_flow = []
        for i in reversed(range(n_downsample_F)):
            up_flow += [nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(ch[i + 1], ch[i], kernel_size=3, padding=1)), activation]
        conv_flow = [nn.Conv2d(nf, 2, kernel_size=3, padding=1)]
        conv_mask = [nn.Conv2d(nf, 1, kernel_size=3, padding=1), nn.Sigmoid()]
        self.down_flow = nn.Sequential(*down_flow)
        self.res_flow = nn.Sequential(*res_flow)
        self.up_flow = nn.Sequential(*up_flow)
        self.conv_flow = nn.Sequential(*conv_flow)
        self.conv_mask = nn.Sequential(*conv_mask)

    def forward(self, label, label_prev, img_prev, for_ref=False):
        label = torch.cat([label, label_prev, img_prev], dim=1)
        downsample = self.down_flow(label)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat) * self.flow_multiplier
        flow_mask = self.conv_mask(flow_feat)
        return flow, flow_mask


class LabelEmbedder(BaseNetwork):

    def __init__(self, opt, input_nc, netS=None, params_free_layers=0, first_layer_free=False):
        super().__init__()
        self.opt = opt
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_F)
        activation = nn.LeakyReLU(0.2)
        nf = opt.ngf
        nf_max = 1024
        self.netS = netS if netS is not None else opt.netS
        self.unet = 'unet' in self.netS
        self.decode = 'decoder' in self.netS or self.unet
        self.n_downsample_S = n_downsample_S = opt.n_downsample_G
        self.params_free_layers = params_free_layers if params_free_layers != -1 else n_downsample_S
        self.first_layer_free = first_layer_free
        ch = [min(nf_max, nf * 2 ** i) for i in range(n_downsample_S + 1)]
        if not first_layer_free:
            layer = [nn.Conv2d(input_nc, nf, kernel_size=3, padding=1), activation]
            self.conv_first = nn.Sequential(*layer)
        for i in range(n_downsample_S):
            layer = [nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=2, padding=1), activation]
            if i >= params_free_layers or 'decoder' in netS:
                setattr(self, 'down_%d' % i, nn.Sequential(*layer))
        if self.decode:
            for i in reversed(range(n_downsample_S)):
                ch_i = ch[i + 1] * (2 if self.unet and i != n_downsample_S - 1 else 1)
                layer = [nn.Upsample(scale_factor=2), nn.Conv2d(ch_i, ch[i], kernel_size=3, padding=1), activation]
                if i >= params_free_layers:
                    setattr(self, 'up_%d' % i, nn.Sequential(*layer))

    def forward(self, input, weights=None):
        if input is None:
            return None
        if self.first_layer_free:
            output = [actvn(batch_conv(input, weights[0]))]
            weights = weights[1:]
        else:
            output = [self.conv_first(input)]
        for i in range(self.n_downsample_S):
            if i >= self.params_free_layers or self.decode:
                conv = getattr(self, 'down_%d' % i)(output[-1])
            else:
                conv = actvn(batch_conv(output[-1], weights[i], stride=2))
            output.append(conv)
        if not self.decode:
            return output
        if not self.unet:
            output = [output[-1]]
        for i in reversed(range(self.n_downsample_S)):
            input_i = output[-1]
            if self.unet and i != self.n_downsample_S - 1:
                input_i = torch.cat([input_i, output[i + 1]], dim=1)
            if i >= self.params_free_layers:
                conv = getattr(self, 'up_%d' % i)(input_i)
            else:
                input_i = nn.Upsample(scale_factor=2)(input_i)
                conv = actvn(batch_conv(input_i, weights[i]))
            output.append(conv)
        if self.unet:
            output = output[self.n_downsample_S:]
        return output[::-1]


def pick_ref(refs, ref_idx):
    if type(refs) == list:
        return [pick_ref(r, ref_idx) for r in refs]
    if ref_idx is None:
        return refs[:, (0)]
    ref_idx = ref_idx.long().view(-1, 1, 1, 1, 1)
    ref = refs.gather(1, ref_idx.expand_as(refs)[:, 0:1])[:, (0)]
    return ref


def set_random_seed(seed):
    """Set random seeds for everything.
       Inputs:
       seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FewShotGenerator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_downsample_G = n_downsample_G = opt.n_downsample_G
        self.n_downsample_A = n_downsample_A = opt.n_downsample_A
        self.nf = nf = opt.ngf
        self.nf_max = nf_max = min(1024, nf * 2 ** n_downsample_G)
        self.ch = ch = [min(nf_max, nf * 2 ** i) for i in range(n_downsample_G + 2)]
        self.norm = norm = opt.norm_G
        self.conv_ks = conv_ks = opt.conv_ks
        self.embed_ks = embed_ks = opt.embed_ks
        self.spade_ks = spade_ks = opt.spade_ks
        self.spade_combine = opt.spade_combine
        self.n_sc_layers = opt.n_sc_layers
        self.add_raw_output_loss = opt.add_raw_output_loss and opt.spade_combine
        ch_hidden = []
        for i in range(n_downsample_G + 1):
            ch_hidden += [[ch[i]]] if not self.spade_combine or i >= self.n_sc_layers else [[ch[i]] * 3]
        self.ch_hidden = ch_hidden
        self.adap_spade = opt.adaptive_spade
        self.adap_embed = opt.adaptive_spade and not opt.no_adaptive_embed
        self.adap_conv = opt.adaptive_conv
        self.n_adaptive_layers = opt.n_adaptive_layers if opt.n_adaptive_layers != -1 else n_downsample_G
        self.concat_label_ref = 'concat' in opt.use_label_ref
        self.mul_label_ref = 'mul' in opt.use_label_ref
        self.sh_fix = self.sw_fix = 32
        self.sw = opt.fineSize // 2 ** opt.n_downsample_G
        self.sh = int(self.sw / opt.aspect_ratio)
        self.n_fc_layers = n_fc_layers = opt.n_fc_layers
        norm_ref = norm.replace('spade', '')
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        ref_nc = opt.output_nc + (0 if not self.concat_label_ref else input_nc)
        self.ref_img_first = SPADEConv2d(ref_nc, nf, norm=norm_ref)
        if self.mul_label_ref:
            self.ref_label_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
        ref_conv = SPADEConv2d if not opt.res_for_ref else SPADEResnetBlock
        for i in range(n_downsample_G):
            ch_in, ch_out = ch[i], ch[i + 1]
            setattr(self, 'ref_img_down_%d' % i, ref_conv(ch_in, ch_out, stride=2, norm=norm_ref))
            setattr(self, 'ref_img_up_%d' % i, ref_conv(ch_out, ch_in, norm=norm_ref))
            if self.mul_label_ref:
                setattr(self, 'ref_label_down_%d' % i, ref_conv(ch_in, ch_out, stride=2, norm=norm_ref))
                setattr(self, 'ref_label_up_%d' % i, ref_conv(ch_out, ch_in, norm=norm_ref))
        if self.adap_spade or self.adap_conv:
            for i in range(self.n_adaptive_layers):
                ch_in, ch_out = ch[i], ch[i + 1]
                conv_ks2 = conv_ks ** 2
                embed_ks2 = embed_ks ** 2
                spade_ks2 = spade_ks ** 2
                ch_h = ch_hidden[i][0]
                fc_names, fc_outs = [], []
                if self.adap_spade:
                    fc0_out = fcs_out = (ch_h * spade_ks2 + 1) * 2
                    fc1_out = (ch_h * spade_ks2 + 1) * (1 if ch_in != ch_out else 2)
                    fc_names += ['fc_spade_0', 'fc_spade_1', 'fc_spade_s']
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                    if self.adap_embed:
                        fc_names += ['fc_spade_e']
                        fc_outs += [ch_in * embed_ks2 + 1]
                if self.adap_conv:
                    fc0_out = ch_out * conv_ks2 + 1
                    fc1_out = ch_in * conv_ks2 + 1
                    fcs_out = ch_out + 1
                    fc_names += ['fc_conv_0', 'fc_conv_1', 'fc_conv_s']
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                for n, l in enumerate(fc_names):
                    fc_in = ch_out if self.mul_label_ref else self.sh_fix * self.sw_fix
                    activation = nn.LeakyReLU(0.2)
                    fc_layer = [sn(nn.Linear(fc_in, ch_out)), activation]
                    for k in range(1, n_fc_layers):
                        fc_layer += [sn(nn.Linear(ch_out, ch_out)), activation]
                    fc_layer += [sn(nn.Linear(ch_out, fc_outs[n]))]
                    setattr(self, '%s_%d' % (l, i), nn.Sequential(*fc_layer))
        self.label_embedding = LabelEmbedder(opt, input_nc, opt.netS, params_free_layers=self.n_adaptive_layers if self.adap_embed else 0)
        for i in reversed(range(n_downsample_G + 1)):
            setattr(self, 'up_%d' % i, SPADEResnetBlock(ch[i + 1], ch[i], norm=norm, hidden_nc=ch_hidden[i], conv_ks=conv_ks, spade_ks=spade_ks, conv_params_free=self.adap_conv and i < self.n_adaptive_layers, norm_params_free=self.adap_spade and i < self.n_adaptive_layers))
        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.up = functools.partial(F.interpolate, scale_factor=2)
        if opt.n_shot > 1:
            self.atn_query_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
            self.atn_key_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
            for i in range(n_downsample_A):
                f_in, f_out = ch[i], ch[i + 1]
                setattr(self, 'atn_key_%d' % i, SPADEConv2d(f_in, f_out, stride=2, norm=norm_ref))
                setattr(self, 'atn_query_%d' % i, SPADEConv2d(f_in, f_out, stride=2, norm=norm_ref))
        self.use_kld = opt.lambda_kld > 0
        self.z_dim = 256
        if self.use_kld:
            f_in = min(nf_max, nf * 2 ** n_downsample_G) * self.sh * self.sw
            f_out = min(nf_max, nf * 2 ** n_downsample_G) * self.sh * self.sw
            self.fc_mu_ref = nn.Linear(f_in, self.z_dim)
            self.fc_var_ref = nn.Linear(f_in, self.z_dim)
            self.fc = nn.Linear(self.z_dim, f_out)
        self.warp_prev = False
        self.warp_ref = opt.warp_ref and not opt.for_face
        if self.warp_ref:
            self.flow_network_ref = FlowGenerator(opt, 2)
            if self.spade_combine:
                self.img_ref_embedding = LabelEmbedder(opt, opt.output_nc + 1, opt.sc_arch)

    def init_temporal_network(self):
        opt = self.opt
        set_random_seed(0)
        self.warp_prev = True
        self.sep_prev_flownet = opt.sep_flow_prev or opt.n_frames_G != 2 or not opt.warp_ref
        self.sep_prev_embedding = self.spade_combine and (not opt.no_sep_warp_embed or not opt.warp_ref)
        if self.sep_prev_flownet:
            self.flow_network_temp = FlowGenerator(opt, opt.n_frames_G)
            self.flow_network_temp.init_weights(opt.init_type, opt.init_variance)
        else:
            self.flow_network_temp = self.flow_network_ref
        if self.spade_combine:
            if self.sep_prev_embedding:
                self.img_prev_embedding = LabelEmbedder(opt, opt.output_nc + 1, opt.sc_arch)
                self.img_prev_embedding.init_weights(opt.init_type, opt.init_variance)
            else:
                self.img_prev_embedding = self.img_ref_embedding
        if self.warp_ref:
            if self.sep_prev_flownet:
                self.load_pretrained_net(self.flow_network_ref, self.flow_network_temp)
            if self.sep_prev_embedding:
                self.load_pretrained_net(self.img_ref_embedding, self.img_prev_embedding)
            self.flow_temp_is_initalized = True
        set_random_seed(get_rank())

    def forward(self, label, label_refs, img_refs, prev=[None, None], t=0, img_coarse=None):
        if img_coarse is not None:
            return self.forward_face(label, label_refs, img_refs, img_coarse)
        x, encoded_label, conv_weights, norm_weights, mu, logvar, atn, atn_vis, ref_idx = self.weight_generation(img_refs, label_refs, label, t=t)
        flow, flow_mask, img_warp, ds_ref = self.flow_generation(label, label_refs, img_refs, prev, atn, ref_idx)
        flow_mask_ref, flow_mask_prev = flow_mask
        img_ref_warp, img_prev_warp = img_warp
        if self.add_raw_output_loss:
            encoded_label_raw = [encoded_label[i] for i in range(self.n_sc_layers)]
        encoded_label = self.SPADE_combine(encoded_label, ds_ref)
        for i in range(self.n_downsample_G, -1, -1):
            conv_weight = conv_weights[i] if self.adap_conv and i < self.n_adaptive_layers else None
            norm_weight = norm_weights[i] if self.adap_spade and i < self.n_adaptive_layers else None
            if self.add_raw_output_loss and i < self.n_sc_layers:
                if i == self.n_sc_layers - 1:
                    x_raw = x
                x_raw = getattr(self, 'up_' + str(i))(x_raw, encoded_label_raw[i], conv_weights=conv_weight, norm_weights=norm_weight)
                if i != 0:
                    x_raw = self.up(x_raw)
            x = getattr(self, 'up_' + str(i))(x, encoded_label[i], conv_weights=conv_weight, norm_weights=norm_weight)
            if i != 0:
                x = self.up(x)
        x = self.conv_img(actvn(x))
        img_raw = torch.tanh(x)
        if not self.spade_combine:
            if self.warp_ref:
                img_final = img_raw * flow_mask_ref + img_ref_warp * (1 - flow_mask_ref)
            else:
                img_final = img_raw
                if not self.warp_prev:
                    img_raw = None
            if self.warp_prev and prev[0] is not None:
                img_final = img_final * flow_mask_prev + img_prev_warp * (1 - flow_mask_prev)
        else:
            img_final = img_raw
            img_raw = None if not self.add_raw_output_loss else torch.tanh(self.conv_img(actvn(x_raw)))
        return img_final, flow, flow_mask, img_raw, img_warp, mu, logvar, atn_vis, ref_idx

    def forward_face(self, label, label_refs, img_refs, img_coarse):
        x, encoded_label, _, norm_weights, _, _, _, _, _ = self.weight_generation(img_refs, label_refs, label, img_coarse=img_coarse)
        for i in range(self.n_downsample_G, -1, -1):
            norm_weight = norm_weights[i] if self.adap_spade and i < self.n_adaptive_layers else None
            x = getattr(self, 'up_' + str(i))(x, encoded_label[i], norm_weights=norm_weight)
            if i != 0:
                x = self.up(x)
        x = self.conv_img(actvn(x))
        img_final = torch.tanh(x)
        return img_final

    def get_SPADE_weights(self, x, i):
        if not self.mul_label_ref:
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)
        ch_in, ch_out = self.ch[i], self.ch[i + 1]
        ch_h = self.ch_hidden[i][0]
        eks, sks = self.embed_ks, self.spade_ks
        b = x.size()[0]
        x = self.reshape_embed_input(x)
        embedding_weights = None
        if self.adap_embed:
            fc_e = getattr(self, 'fc_spade_e_' + str(i))(x).view(b, -1)
            embedding_weights = self.reshape_weight(fc_e[:, :-ch_in], [ch_in, ch_out, eks, eks])
        fc_0 = getattr(self, 'fc_spade_0_' + str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_spade_1_' + str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_spade_s_' + str(i))(x).view(b, -1)
        weight_0 = self.reshape_weight(fc_0, [[ch_out, ch_h, sks, sks]] * 2)
        weight_1 = self.reshape_weight(fc_1, [[ch_in, ch_h, sks, sks]] * 2)
        weight_s = self.reshape_weight(fc_s, [[ch_out, ch_h, sks, sks]] * 2)
        norm_weights = [weight_0, weight_1, weight_s]
        return embedding_weights, norm_weights

    def get_conv_weights(self, x, i):
        if not self.mul_label_ref:
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)
        ch_in, ch_out = self.ch[i], self.ch[i + 1]
        b = x.size()[0]
        x = self.reshape_embed_input(x)
        fc_0 = getattr(self, 'fc_conv_0_' + str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_conv_1_' + str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_conv_s_' + str(i))(x).view(b, -1)
        weight_0 = self.reshape_weight(fc_0, [ch_in, ch_out, 3, 3])
        weight_1 = self.reshape_weight(fc_1, [ch_in, ch_in, 3, 3])
        weight_s = self.reshape_weight(fc_s, [ch_in, ch_out, 1, 1])
        return [weight_0, weight_1, weight_s]

    def attention_encode(self, img, net_name):
        x = getattr(self, net_name + '_first')(img)
        for i in range(self.n_downsample_A):
            x = getattr(self, net_name + '_' + str(i))(x)
        return x

    def attention_module(self, x, label, label_ref, attention=None):
        b, c, h, w = x.size()
        n = self.opt.n_shot
        b = b // n
        if attention is None:
            atn_key = self.attention_encode(label_ref, 'atn_key')
            atn_query = self.attention_encode(label, 'atn_query')
            atn_key = atn_key.view(b, n, c, -1).permute(0, 1, 3, 2).contiguous().view(b, -1, c)
            atn_query = atn_query.view(b, c, -1)
            energy = torch.bmm(atn_key, atn_query)
            attention = nn.Softmax(dim=1)(energy)
        x = x.view(b, n, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        out = torch.bmm(x, attention).view(b, c, h, w)
        atn_vis = attention.view(b, n, h * w, h * w).sum(2).view(b, n, h, w)
        return out, attention, atn_vis[-1:, 0:1]

    def compute_kld(self, x, label, img_coarse):
        mu = logvar = None
        if img_coarse is not None:
            if self.concat_label_ref:
                img_coarse = torch.cat([img_coarse, label], dim=1)
            x_kld = self.ref_img_first(img_coarse)
            for i in range(self.n_downsample_G):
                x_kld = getattr(self, 'ref_img_down_' + str(i))(x_kld)
        elif self.use_kld:
            b, c, h, w = x.size()
            mu = self.fc_mu_ref(x.view(b, -1))
            if self.opt.isTrain:
                logvar = self.fc_var_ref(x.view(b, -1))
                z = self.reparameterize(mu, logvar)
            else:
                z = mu
            x_kld = self.fc(z).view(b, -1, h, w)
        else:
            x_kld = x
        return x_kld, mu, logvar

    def reference_encoding(self, img_ref, label_ref, label, n, t=0):
        if self.concat_label_ref:
            concat_ref = torch.cat([img_ref, label_ref], dim=1)
            x = self.ref_img_first(concat_ref)
        elif self.mul_label_ref:
            x = self.ref_img_first(img_ref)
            x_label = self.ref_label_first(label_ref)
        else:
            assert False
        atn = atn_vis = ref_idx = None
        for i in range(self.n_downsample_G):
            x = getattr(self, 'ref_img_down_' + str(i))(x)
            if self.mul_label_ref:
                x_label = getattr(self, 'ref_label_down_' + str(i))(x_label)
            if n > 1 and i == self.n_downsample_A - 1:
                x, atn, atn_vis = self.attention_module(x, label, label_ref)
                if self.mul_label_ref:
                    x_label, _, _ = self.attention_module(x_label, None, None, atn)
                atn_sum = atn.view(label.shape[0], n, -1).sum(2)
                ref_idx = torch.argmax(atn_sum, dim=1)
        encoded_ref = None
        if self.opt.isTrain or n > 1 or t == 0:
            encoded_image_ref = [x]
            if self.mul_label_ref:
                encoded_label_ref = [x_label]
            for i in reversed(range(self.n_downsample_G)):
                conv = getattr(self, 'ref_img_up_' + str(i))(encoded_image_ref[-1])
                encoded_image_ref.append(conv)
                if self.mul_label_ref:
                    conv_label = getattr(self, 'ref_label_up_' + str(i))(encoded_label_ref[-1])
                    encoded_label_ref.append(conv_label)
            if self.mul_label_ref:
                encoded_ref = []
                for i in range(len(encoded_image_ref)):
                    conv, conv_label = encoded_image_ref[i], encoded_label_ref[i]
                    b, c, h, w = conv.size()
                    conv_label = nn.Softmax(dim=1)(conv_label)
                    conv_prod = (conv.view(b, c, 1, h * w) * conv_label.view(b, 1, c, h * w)).sum(3, keepdim=True)
                    encoded_ref.append(conv_prod)
            else:
                encoded_ref = encoded_image_ref
            encoded_ref = encoded_ref[::-1]
        return x, encoded_ref, atn, atn_vis, ref_idx

    def weight_generation(self, img_ref, label_ref, label, t=0, img_coarse=None):
        b, n, c, h, w = img_ref.size()
        img_ref, label_ref = img_ref.view(b * n, -1, h, w), label_ref.view(b * n, -1, h, w)
        x, encoded_ref, atn, atn_vis, ref_idx = self.reference_encoding(img_ref, label_ref, label, n, t)
        x_kld, mu, logvar = self.compute_kld(x, label, img_coarse)
        if self.opt.isTrain or n > 1 or t == 0:
            embedding_weights, norm_weights, conv_weights = [], [], []
            for i in range(self.n_adaptive_layers):
                if self.adap_spade:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i + 1)]
                    embedding_weight, norm_weight = self.get_SPADE_weights(feat, i)
                    embedding_weights.append(embedding_weight)
                    norm_weights.append(norm_weight)
                if self.adap_conv:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i)]
                    conv_weights.append(self.get_conv_weights(feat, i))
            if not self.opt.isTrain:
                self.embedding_weights, self.conv_weights, self.norm_weights = embedding_weights, conv_weights, norm_weights
        else:
            embedding_weights, conv_weights, norm_weights = self.embedding_weights, self.conv_weights, self.norm_weights
        encoded_label = self.label_embedding(label, weights=embedding_weights if self.adap_embed else None)
        return x_kld, encoded_label, conv_weights, norm_weights, mu, logvar, atn, atn_vis, ref_idx

    def flow_generation(self, label, label_refs, img_refs, prev, atn, ref_idx):
        label_ref, img_ref = pick_ref([label_refs, img_refs], ref_idx)
        label_prev, img_prev = prev
        has_prev = label_prev is not None
        flow, flow_mask, img_warp, ds_ref = [None] * 2, [None] * 2, [None] * 2, [None] * 2
        if self.warp_ref:
            flow_ref, flow_mask_ref = self.flow_network_ref(label, label_ref, img_ref, for_ref=True)
            img_ref_warp = resample(img_ref, flow_ref)
            flow[0], flow_mask[0], img_warp[0] = flow_ref, flow_mask_ref, img_ref_warp[:, :3]
        if self.warp_prev and has_prev:
            flow_prev, flow_mask_prev = self.flow_network_temp(label, label_prev, img_prev)
            img_prev_warp = resample(img_prev[:, -3:], flow_prev)
            flow[1], flow_mask[1], img_warp[1] = flow_prev, flow_mask_prev, img_prev_warp
        if self.spade_combine:
            if self.warp_ref:
                ds_ref[0] = torch.cat([img_warp[0], flow_mask[0]], dim=1)
            if self.warp_prev and has_prev:
                ds_ref[1] = torch.cat([img_warp[1], flow_mask[1]], dim=1)
        return flow, flow_mask, img_warp, ds_ref

    def SPADE_combine(self, encoded_label, ds_ref):
        if self.spade_combine:
            encoded_image_warp = [self.img_ref_embedding(ds_ref[0]), self.img_prev_embedding(ds_ref[1]) if ds_ref[1] is not None else None]
            for i in range(self.n_sc_layers):
                encoded_label[i] = [encoded_label[i]] + [(w[i] if w is not None else None) for w in encoded_image_warp]
        return encoded_label


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
            return self.fake_label_tensor.expand_as(input)

    def loss(self, input, target_is_real, weight=None, reduce_dim=True, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            batchsize = input.size(0)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor, weight=weight)
            if not reduce_dim:
                loss = loss.view(batchsize, -1).mean(dim=1)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = input * 0 + (self.real_label if target_is_real else self.fake_label)
            if weight is None and reduce_dim:
                return F.mse_loss(input, target_tensor)
            error = (input - target_tensor) ** 2
            if weight is not None:
                error *= weight
            if reduce_dim:
                return torch.mean(error)
            else:
                return error.view(input.size(0), -1).mean(dim=1)
        elif self.gan_mode == 'hinge':
            assert weight == None
            assert reduce_dim == True
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, input * 0)
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, input * 0)
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            assert weight is None and reduce_dim
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, weight=None, reduce_dim=True, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, weight, reduce_dim, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, weight, reduce_dim, for_discriminator)


class VGG_Activations(nn.Module):

    def __init__(self, feature_idx):
        super(VGG_Activations, self).__init__()
        vgg_network = torchvision.models.vgg19(pretrained=True)
        features = list(vgg_network.features)
        self.features = nn.ModuleList(features).eval()
        self.idx_list = feature_idx

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.idx_list:
                results.append(x)
        return results


class VGGLoss(nn.Module):

    def __init__(self, opt, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG_Activations([1, 6, 11, 20, 29])
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def compute_loss(self, x_vgg, y_vgg):
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            x, y = x.view(-1, c, h, w), y.view(-1, c, h, w)
        y_vgg = self.vgg(y)
        x_vgg = self.vgg(x)
        loss = self.compute_loss(x_vgg, y_vgg)
        return loss


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        mask = mask.expand_as(input)
        loss = self.criterion(input * mask, target * mask)
        return loss


class KLDLoss(nn.Module):

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class Vgg19(nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def combine_fg_mask(fg_mask, ref_fg_mask, has_fg):
    return ((fg_mask > 0) | (ref_fg_mask > 0)).float() if has_fg else 1


def encode_label(opt, label_map):
    size = label_map.size()
    if len(size) == 5:
        bs, t, c, h, w = size
        label_map = label_map.view(-1, c, h, w)
    else:
        bs, c, h, w = size
    label_nc = opt.label_nc
    if label_nc == 0:
        input_label = label_map
    else:
        label_map = label_map
        oneHot_size = label_map.shape[0], label_nc, h, w
        input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.long(), 1.0)
    if len(size) == 5:
        return input_label.view(bs, t, -1, h, w)
    return input_label


def remove_dummy_from_tensor(opt, tensors, remove_size=0):
    if remove_size == 0:
        return tensors
    if tensors is None:
        return None
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(opt, tensor, remove_size) for tensor in tensors]
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors


def encode_input(opt, data_list, dummy_bs):
    if opt.isTrain and data_list[0].get_device() == 0:
        data_list = remove_dummy_from_tensor(opt, data_list, dummy_bs)
    tgt_label, tgt_image, flow_gt, conf_gt, ref_label, ref_image, prev_label, prev_real_image, prev_fake_image = data_list
    tgt_label = encode_label(opt, tgt_label)
    tgt_image = tgt_image
    ref_label = encode_label(opt, ref_label)
    ref_image = ref_image
    return tgt_label, tgt_image, flow_gt, conf_gt, ref_label, ref_image, [prev_label, prev_real_image, prev_fake_image]


def loss_backward(opt, losses, optimizer, loss_id):
    losses = [(torch.mean(x) if not isinstance(x, int) else x) for x in losses]
    loss = sum(losses)
    optimizer.zero_grad()
    if opt.amp != 'O0':
        with amp.scale_loss(loss, optimizer, loss_id=loss_id) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    return losses


def roll(t, ny, nx, flip):
    t = torch.cat([t[:, :, -ny:], t[:, :, :-ny]], dim=2)
    t = torch.cat([t[:, :, :, -nx:], t[:, :, :, :-nx]], dim=3)
    if flip:
        t = torch.flip(t, dims=[3])
    return t


def random_roll(tensors):
    h, w = tensors[0].shape[2:]
    ny = random.choice([random.randrange(h // 16), h - random.randrange(h // 16)])
    nx = random.choice([random.randrange(w // 16), w - random.randrange(w // 16)])
    flip = random.random() > 0.5
    return [roll(t, ny, nx, flip) for t in tensors]


class Vid2VidModel(BaseModel):

    def name(self):
        return 'Vid2VidModel'

    def initialize(self, opt, epoch=0):
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True
        set_random_seed(0)
        self.lossCollector = LossCollector()
        self.lossCollector.initialize(opt)
        self.refine_face = hasattr(opt, 'refine_face') and opt.refine_face
        self.faceRefiner = None
        if self.refine_face or self.add_face_D:
            self.faceRefiner = FaceRefineModel()
            self.faceRefiner.initialize(opt, self.add_face_D, self.refine_face)
        self.define_networks(epoch)
        self.load_networks()
        set_random_seed(get_rank())

    def forward(self, data_list, save_images=False, mode='inference', dummy_bs=0):
        tgt_label, tgt_image, flow_gt, conf_gt, ref_labels, ref_images, prevs = encode_input(self.opt, data_list, dummy_bs)
        if mode == 'generator':
            g_loss, generated, prev = self.forward_generator(tgt_label, tgt_image, ref_labels, ref_images, prevs, flow_gt, conf_gt)
            return g_loss, generated if save_images else [], prev
        elif mode == 'discriminator':
            d_loss = self.forward_discriminator(tgt_label, tgt_image, ref_labels, ref_images, prevs)
            return d_loss
        else:
            return self.inference(tgt_label, ref_labels, ref_images)

    def forward_generator(self, tgt_label, tgt_image, ref_labels, ref_images, prevs=[None] * 3, flow_gt=[None] * 2, conf_gt=[None] * 2):
        [fake_image, fake_raw_image, warped_image, flow, flow_mask], [fg_mask, ref_fg_mask], [ref_label, ref_image], prevs_new, atn_score = self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, prevs)
        nets = self.netD, self.netDT, self.netDf, self.faceRefiner
        loss_GT_GAN, loss_GT_GAN_Feat = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        if self.isTrain and self.opt.lambda_temp > 0 and prevs[0] is not None:
            tgt_image_all = torch.cat([prevs[1], tgt_image], dim=1)
            fake_image_all = torch.cat([prevs[2], fake_image], dim=1)
            data_list = [None, tgt_image_all, fake_image_all, None, None]
            loss_GT_GAN, loss_GT_GAN_Feat = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=False, for_temporal=True)
        fg_mask_union = combine_fg_mask(fg_mask, ref_fg_mask, self.has_fg)
        data_list = [tgt_label, [tgt_image, tgt_image * fg_mask_union], [fake_image, fake_raw_image], ref_label, ref_image]
        loss_G_GAN, loss_G_GAN_Feat, loss_Gf_GAN, loss_Gf_GAN_Feat = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=False)
        loss_G_VGG = self.lossCollector.compute_VGG_losses(fake_image, fake_raw_image, tgt_image, fg_mask_union)
        flow, flow_mask, flow_gt, conf_gt, fg_mask, ref_fg_mask, warped_image, tgt_image = self.reshape([flow, flow_mask, flow_gt, conf_gt, fg_mask, ref_fg_mask, warped_image, tgt_image])
        loss_F_Flow, loss_F_Warp, body_mask_diff = self.lossCollector.compute_flow_losses(flow, warped_image, tgt_image, flow_gt, conf_gt, fg_mask, tgt_label, ref_label)
        loss_F_Mask = self.lossCollector.compute_mask_losses(flow_mask, fake_image, warped_image, tgt_label, tgt_image, fake_raw_image, fg_mask, ref_fg_mask, body_mask_diff)
        loss_list = [loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_Gf_GAN, loss_Gf_GAN_Feat, loss_GT_GAN, loss_GT_GAN_Feat, loss_F_Flow, loss_F_Warp, loss_F_Mask]
        loss_list = [loss.view(1, 1) for loss in loss_list]
        return loss_list, [fake_image, fake_raw_image, warped_image, flow, flow_mask, atn_score], prevs_new

    def forward_discriminator(self, tgt_label, tgt_image, ref_labels, ref_images, prevs=[None] * 3):
        with torch.no_grad():
            [fake_image, fake_raw_image, _, _, _], [fg_mask, ref_fg_mask], [ref_label, ref_image], _, _ = self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, prevs)
        nets = self.netD, self.netDT, self.netDf, self.faceRefiner
        loss_temp = []
        if self.isTrain and self.opt.lambda_temp > 0 and prevs[0] is not None:
            tgt_image_all = torch.cat([prevs[1], tgt_image], dim=1)
            fake_image_all = torch.cat([prevs[2], fake_image], dim=1)
            data_list = [None, tgt_image_all, fake_image_all, None, None]
            loss_temp = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=True, for_temporal=True)
        fg_mask_union = combine_fg_mask(fg_mask, ref_fg_mask, self.has_fg)
        data_list = [tgt_label, [tgt_image, tgt_image * fg_mask_union], [fake_image, fake_raw_image], ref_label, ref_image]
        loss_indv = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=True)
        loss_list = list(loss_indv) + list(loss_temp)
        loss_list = [loss.view(1, 1) for loss in loss_list]
        return loss_list

    def generate_images(self, tgt_labels, tgt_images, ref_labels, ref_images, prevs=[None] * 3):
        opt = self.opt
        generated_images, atn_score = [None] * 5, None
        generated_masks = [None] * 2 if self.has_fg else [1, 1]
        ref_labels_valid = use_valid_labels(opt, ref_labels)
        for t in range(opt.n_frames_per_gpu):
            tgt_label_t, tgt_label_valid, tgt_image, prev_t = self.get_input_t(tgt_labels, tgt_images, prevs, t)
            fake_image, flow, flow_mask, fake_raw_image, warped_image, mu, logvar, atn_score, ref_idx = self.netG(tgt_label_valid, ref_labels_valid, ref_images, prev_t)
            ref_label_valid, ref_label_t, ref_image_t = pick_ref([ref_labels_valid, ref_labels, ref_images], ref_idx)
            if self.refine_face:
                fake_image = self.faceRefiner.refine_face_region(self.netGf, tgt_label_valid, fake_image, tgt_label_t, ref_label_valid, ref_image_t, ref_label_t)
            fg_mask, ref_fg_mask = get_fg_mask(self.opt, [tgt_label_t, ref_label_t], self.has_fg)
            if fake_raw_image is not None:
                fake_raw_image = fake_raw_image * combine_fg_mask(fg_mask, ref_fg_mask, self.has_fg)
            generated_images = self.concat([generated_images, [fake_image, fake_raw_image, warped_image, flow, flow_mask]], dim=1)
            generated_masks = self.concat([generated_masks, [fg_mask, ref_fg_mask]], dim=1)
            prevs = self.concat_prev(prevs, [tgt_label_valid, tgt_image, fake_image])
        return generated_images, generated_masks, [ref_label_valid, ref_image_t], prevs, atn_score

    def get_input_t(self, tgt_labels, tgt_images, prevs, t):
        b, _, _, h, w = tgt_labels.shape
        tgt_label = tgt_labels[:, (t)]
        tgt_image = tgt_images[:, (t)]
        tgt_label_valid = use_valid_labels(self.opt, tgt_label)
        prevs = [prevs[0], prevs[2]]
        prevs = [(prev.contiguous().view(b, -1, h, w) if prev is not None else None) for prev in prevs]
        return tgt_label, tgt_label_valid, tgt_image, prevs

    def concat_prev(self, prev, now):
        if type(prev) == list:
            return [self.concat_prev(p, n) for p, n in zip(prev, now)]
        if prev is None:
            prev = now.unsqueeze(1).repeat(1, self.opt.n_frames_G - 1, 1, 1, 1)
        else:
            prev = torch.cat([prev[:, 1:], now.unsqueeze(1)], dim=1)
        return prev.detach()

    def inference(self, tgt_label, ref_labels, ref_images):
        opt = self.opt
        if not hasattr(self, 'prevs') or self.prevs is None:
            None
            self.prevs = prevs = [None, None]
            self.t = 0
        else:
            b, _, _, h, w = tgt_label.shape
            prevs = [prev.view(b, -1, h, w) for prev in self.prevs]
            self.t += 1
        tgt_label_valid, ref_labels_valid = use_valid_labels(opt, [tgt_label[:, (-1)], ref_labels])
        if opt.finetune and self.t == 0:
            self.finetune(ref_labels, ref_images)
        with torch.no_grad():
            fake_image, flow, flow_mask, fake_raw_image, warped_image, _, _, atn_score, ref_idx = self.netG(tgt_label_valid, ref_labels_valid, ref_images, prevs, t=self.t)
            ref_label_valid, ref_label, ref_image = pick_ref([ref_labels_valid, ref_labels, ref_images], ref_idx)
            if self.refine_face:
                fake_image = self.faceRefiner.refine_face_region(self.netGf, tgt_label_valid, fake_image, tgt_label[:, (-1)], ref_label_valid, ref_image, ref_label)
            self.prevs = self.concat_prev(self.prevs, [tgt_label_valid, fake_image])
        return fake_image, fake_raw_image, warped_image, flow, flow_mask, atn_score

    def finetune(self, ref_labels, ref_images):
        train_names = ['fc', 'conv_img', 'up']
        params, _ = self.get_train_params(self.netG, train_names)
        self.optimizer_G = self.get_optimizer(params, for_discriminator=False)
        update_D = True
        if update_D:
            params = list(self.netD.parameters())
            if self.add_face_D:
                params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)
        iterations = 100
        for it in range(1, iterations + 1):
            idx = random.randrange(ref_labels.size(1))
            tgt_label, tgt_image = random_roll([ref_labels[:, (idx)], ref_images[:, (idx)]])
            tgt_label, tgt_image = tgt_label.unsqueeze(1), tgt_image.unsqueeze(1)
            g_losses, generated, prev = self.forward_generator(tgt_label, tgt_image, ref_labels, ref_images)
            g_losses = loss_backward(self.opt, g_losses, self.optimizer_G, 0)
            d_losses = []
            if update_D:
                d_losses = self.forward_discriminator(tgt_label, tgt_image, ref_labels, ref_images)
                d_losses = loss_backward(self.opt, d_losses, self.optimizer_D, 1)
            if it % 10 == 0:
                message = '(iters: %d) ' % it
                loss_dict = dict(zip(self.lossCollector.loss_names, g_losses + d_losses))
                for k, v in loss_dict.items():
                    if v != 0:
                        message += '%s: %.3f ' % (k, v)
                None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (DataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DataParallelWithCallback,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (FaceRefineModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (FlowNetFusion,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 11, 64, 64])], {}),
     True),
    (FlowNetS,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (FlowNetSD,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     False),
    (KLDLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Loss,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LabelEmbedder,
     lambda: ([], {'opt': _mock_config(norm_F=4, ngf=4, netS=[4, 4], n_downsample_G=4), 'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LossCollector,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (MaskedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiScale,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPADEConv2d,
     lambda: ([], {'fin': 4, 'fout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPADEResnetBlock,
     lambda: ([], {'fin': 4, 'fout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SynchronizedBatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SynchronizedBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SynchronizedBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGGLoss,
     lambda: ([], {'opt': _mock_config(), 'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_SynchronizedBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (tofp32,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVlabs_few_shot_vid2vid(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

