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
resample2d_package = _module
resample2d = _module
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
visualizer = _module

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


import copy


import torch


import torch.nn.functional as F


import numpy as np


import torch.nn as nn


import functools


import torch.nn.utils.spectral_norm as sn


from torch.nn import init


import math


from torch.utils.data import DataLoader


from torch.autograd import Variable


from torch.autograd import Function


from torch.nn.modules.module import Module


import re


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


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
            import visdom
            self.vis = visdom.Visdom()
            self.visdom_id = opt.visdom_id
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            if hasattr(opt, 'model_idx') and opt.model_idx != -1:
                self.log_name = os.path.join(opt.checkpoints_dir, opt.name,
                    'loss_log_%03d.txt' % opt.model_idx)
            else:
                self.log_name = os.path.join(opt.checkpoints_dir, opt.name,
                    'loss_log.txt')
            with open(self.log_name, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write(
                    '================ Training Loss (%s) ================\n' %
                    now)

    def display_visdom_results(self, visuals, epoch, step):
        ncols = self.ncols
        if ncols > 0:
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = (
                """<style>
                    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                    </style>"""
                 % (w, h))
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
            self.vis.images(images, nrow=ncols, win=self.visdom_id + 1,
                padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=self.visdom_id + 2,
                opts=dict(title=title + ' labels'))

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
                img_sum = self.tf.Summary.Image(encoded_image_string=s.
                    getvalue(), height=image_numpy.shape[0], width=
                    image_numpy.shape[1])
                img_summaries.append(self.tf.Summary.Value(tag=label, image
                    =img_sum))
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)
        if self.use_html:
            for label, image_numpy in visuals.items():
                if image_numpy is None:
                    continue
                ext = 'png' if 'label' in label else 'jpg'
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 
                            'epoch%03d_iter%07d_%s_%d.%s' % (epoch, step,
                            label, i, ext))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 
                        'epoch%03d_iter%07d_%s.%s' % (epoch, step, label, ext))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    util.save_image(image_numpy, img_path)
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self
                .name, refresh=300)
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
                                img_path = 'epoch%03d_iter%07d_%s_%d.%s' % (n,
                                    step, label, i, ext)
                            else:
                                img_paths = sorted(glob.glob(os.path.join(
                                    self.img_dir, 
                                    'epoch%03d_iter*_%s_%d.%s' % (n, label,
                                    i, ext))))
                                img_path = os.path.basename(img_paths[-1]
                                    ) if len(img_paths) else 'img.jpg'
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        if n == epoch:
                            img_path = 'epoch%03d_iter%07d_%s.%s' % (n,
                                step, label, ext)
                        else:
                            img_paths = sorted(glob.glob(os.path.join(self.
                                img_dir, 'epoch%03d_iter*_%s.%s' % (n,
                                label, ext))))
                            img_path = os.path.basename(img_paths[-1]) if len(
                                img_paths) else 'img.jpg'
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 6:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num],
                        width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:],
                        width=self.win_size)
            webpage.save()

    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=
                    tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)
        print(message)
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
        print(message)
        if is_master() and opt.isTrain and not opt.debug:
            log_name = os.path.join(opt.checkpoints_dir, opt.name,
                'loss_log.txt')
            with open(log_name, 'a') as log_file:
                log_file.write('%s\n' % message)


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.old_lr = opt.lr
        self.pose = 'pose' in opt.dataset_mode
        self.face = 'face' in opt.dataset_mode
        self.street = 'street' in opt.dataset_mode
        self.warp_ref = opt.warp_ref
        self.has_fg = self.pose
        self.add_face_D = opt.add_face_D
        self.concat_ref_for_D = (opt.isTrain or opt.finetune
            ) and opt.netD_subarch == 'n_layers'
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
                Visualizer.vis_print(self.opt, 'network loaded from %s' %
                    save_path)
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.
                        items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    Visualizer.vis_print(self.opt, 
                        'Pretrained network %s has excessive layers; Only loading layers that are used'
                         % network_label)
                except:
                    Visualizer.vis_print(self.opt, 
                        'Pretrained network %s has fewer layers; The following are not initialized:'
                         % network_label)
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size(
                            ) != pretrained_dict[k].size():
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
                return [self.remove_dummy_from_tensor(tensor, remove_size) for
                    tensor in tensors]
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
                    tensors_cat.append(self.concat([tensors[0][i], tensors[
                        1][i]], dim=dim))
                return tensors_cat
            return torch.cat([tensors[0], tensors[1].unsqueeze(1)], dim=dim)
        elif tensors[1] is not None:
            if isinstance(tensors[1], list):
                return [(t.unsqueeze(1) if t is not None else None) for t in
                    tensors[1]]
            return tensors[1].unsqueeze(1)
        return tensors[0]

    def reshape(self, tensors, for_temporal=False):
        if isinstance(tensors, list):
            return [self.reshape(tensor, for_temporal) for tensor in tensors]
        if tensors is None or type(tensors) != torch.Tensor or len(tensors.
            size()) <= 4:
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
                    tensors = tensors[:, -n * nD:].contiguous().view(-1, ch *
                        nD, h, w)
            else:
                tensors = tensors.contiguous().view(bs, ch * t, h, w)
        return tensors

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = [[tensor[:tensor.size(0) // 2] for tensor in p] for p in
                pred]
            real = [[tensor[tensor.size(0) // 2:] for tensor in p] for p in
                pred]
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
        input_nc = (opt.label_nc if opt.label_nc != 0 and not self.pose else
            opt.input_nc)
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
            netD_input_nc = input_nc + opt.output_nc + (1 if self.
                concat_fg_mask_for_D else 0)
            if self.concat_ref_for_D:
                netD_input_nc *= 2
            self.netD = networks.define_D(opt, netD_input_nc, opt.ndf, opt.
                n_layers_D, opt.norm_D, opt.netD_subarch, opt.num_D, not
                opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            if self.add_face_D:
                self.netDf = networks.define_D(opt, opt.output_nc * 2, opt.
                    ndf, opt.n_layers_D, opt.norm_D, 'n_layers', 1, not opt
                    .no_ganFeat_loss, gpu_ids=self.gpu_ids)
            else:
                self.netDf = None
        self.temporal = False
        self.netDT = None
        Visualizer.vis_print(self.opt,
            '---------- Networks initialized -------------')
        if self.isTrain:
            params = list(self.netG.parameters())
            if self.refine_face:
                params += list(self.netGf.parameters())
            self.optimizer_G = self.get_optimizer(params, for_discriminator
                =False)
            params = list(self.netD.parameters())
            if self.add_face_D:
                params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator
                =True)
        Visualizer.vis_print(self.opt,
            '---------- Optimizers initialized -------------')
        if (not opt.isTrain or start_epoch > opt.niter_single
            ) and opt.n_frames_G > 1:
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
            pretrained_path = ('' if not self.isTrain or opt.continue_train
                 else opt.load_pretrain)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if (self.temporal and opt.warp_ref and not self.netG.
                flow_temp_is_initalized):
                self.netG.load_pretrained_net(self.netG.flow_network_ref,
                    self.netG.flow_network_temp)
            if self.refine_face:
                self.load_network(self.netGf, 'Gf', opt.which_epoch,
                    pretrained_path)
            if self.isTrain and not opt.load_pretrain or opt.finetune:
                self.load_network(self.netD, 'D', opt.which_epoch,
                    pretrained_path)
                if self.isTrain and self.temporal:
                    self.load_network(self.netDT, 'DT', opt.which_epoch,
                        pretrained_path)
                if self.add_face_D:
                    self.load_network(self.netDf, 'Df', opt.which_epoch,
                        pretrained_path)

    def update_learning_rate(self, epoch):
        new_lr = self.opt.lr * (1 - (epoch - self.opt.niter) / (self.opt.
            niter_decay + 1))
        if self.opt.no_TTUR:
            G_lr, D_lr = new_lr, new_lr
        else:
            G_lr, D_lr = new_lr / 2, new_lr * 2
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = D_lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = G_lr
        Visualizer.vis_print(self.opt, 'update learning rate: %f -> %f' % (
            self.old_lr, new_lr))
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
            self.optimizer_G = self.get_optimizer(params, for_discriminator
                =False)
            self.netDT = networks.define_D(opt, opt.output_nc * self.
                lossCollector.tD, opt.ndf, opt.n_layers_D, opt.norm_D,
                'n_layers', 1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            params = list(self.netD.parameters()) + list(self.netDT.
                parameters())
            if self.add_face_D:
                params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator
                =True)
            Visualizer.vis_print(self.opt,
                '---------- Now start training multiple frames -------------')


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


def actvn(x):
    out = F.leaky_relu(x, 0.2)
    return out


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
            yi = F.conv2d(x[i:i + 1], weight=weight[i], bias=bias[i],
                padding=padding, stride=stride, groups=groups)
        else:
            yi = F.conv_transpose2d(x[i:i + 1], weight=weight[i], bias=bias
                [(i), :weight.size(2)], padding=padding, stride=int(1 /
                stride), output_padding=1, groups=groups)
        y = concat(y, yi)
    return y


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

    def __init__(self, fin, fout, norm='batch', hidden_nc=0, conv_ks=3,
        spade_ks=1, stride=1, conv_params_free=False, norm_params_free=False):
        super().__init__()
        fhidden = min(fin, fout)
        self.learned_shortcut = fin != fout
        self.stride = stride
        Conv2d = generalConv(adaptive=conv_params_free)
        sn_ = sn if not conv_params_free else lambda x: x
        self.conv_0 = sn_(Conv2d(fin, fhidden, conv_ks, stride=stride,
            padding=1))
        self.conv_1 = sn_(Conv2d(fhidden, fout, conv_ks, padding=1))
        if self.learned_shortcut:
            self.conv_s = sn_(Conv2d(fin, fout, 1, stride=stride, bias=False))
        Norm = generalNorm(norm)
        self.bn_0 = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=spade_ks,
            params_free=norm_params_free)
        self.bn_1 = Norm(fhidden, hidden_nc=hidden_nc, norm=norm, ks=
            spade_ks, params_free=norm_params_free)
        if self.learned_shortcut:
            self.bn_s = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=
                spade_ks, params_free=norm_params_free)

    def forward(self, x, label=None, conv_weights=[], norm_weights=[]):
        if not conv_weights:
            conv_weights = [None] * 3
        if not norm_weights:
            norm_weights = [None] * 3
        x_s = self._shortcut(x, label, conv_weights[2], norm_weights[2])
        dx = self.conv_0(actvn(self.bn_0(x, label, norm_weights[0])),
            conv_weights[0])
        dx = self.conv_1(actvn(self.bn_1(dx, label, norm_weights[1])),
            conv_weights[1])
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
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or 
                classname.find('Linear') != -1):
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
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
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
                weights.append(self.split_weights(weight[:, cur_size:
                    next_size], sizes[i]))
                cur_size = next_size
            assert next_size == weight.size()[1]
            return weights
        return weight

    def reshape_weight(self, x, weight_size):
        if type(weight_size[0]) == list and type(x) != list:
            x = self.split_weights(x, self.sum_mul(weight_size))
        if type(x) == list:
            return [self.reshape_weight(xi, wi) for xi, wi in zip(x,
                weight_size)]
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

    def __init__(self, args, startScale=4, numScales=5, l_weight=0.32, norm
        ='L1'):
        super(MultiScale, self).__init__()
        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for
            scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert len(self.loss_weights) == self.numScales
        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()
        self.multiScales = [nn.AvgPool2d(self.startScale * 2 ** scale, self
            .startScale * 2 ** scale) for scale in range(self.numScales)]
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


class MyDict(dict):
    pass


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
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=
            self.batchNorm)
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
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim
            =-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        diff_flownets2_img1 = self.channelnorm(x[:, :3, :, :] -
            diff_flownets2_flow)
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        diff_flownetsd_img1 = self.channelnorm(x[:, :3, :, :] -
            diff_flownetsd_flow)
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow,
            norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1,
            diff_flownets2_img1), dim=1)
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
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim
            =-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.
            div_flow, norm_diff_img0), dim=1)
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
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim
            =-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1,
            inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=True), nn.LeakyReLU(0.1, inplace=True))


def deconv(in_planes, out_planes):
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes,
        kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1,
        inplace=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias
        =True)


class FlowNetC(nn.Module):

    def __init__(self, args, batchNorm=True, div_flow=20):
        super(FlowNetC, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1
            )
        if args.fp16:
            self.corr = nn.Sequential(tofp32(), Correlation(pad_size=20,
                kernel_size=1, max_displacement=20, stride1=1, stride2=2,
                corr_multiply=1), tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1,
                max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
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
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
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


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True
    ):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=bias), nn.BatchNorm2d(out_planes))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=bias))


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
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7,
            stride=2)
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
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
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
        channelnorm.backward(input1, output, grad_output.data, grad_input1.
            data, ctx.norm_deg)
        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20,
        stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size,
        max_displacement, stride1, stride2, corr_multiply):
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
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.
                stride1, ctx.stride2, ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation_cuda.backward(input1, input2, rbot1, rbot2,
                grad_output, grad_input1, grad_input2, ctx.pad_size, ctx.
                kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2,
                ctx.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0,
        stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction.apply(input1, input2, self.pad_size,
            self.kernel_size, self.max_displacement, self.stride1, self.
            stride2, self.corr_multiply)
        return result


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
        resample2d_cuda.backward(input1, input2, grad_output.data,
            grad_input1.data, grad_input2.data, ctx.kernel_size)
        return grad_input1, grad_input2, None


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)


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


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=
        0.0, tensor=torch.FloatTensor, opt=None):
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

    def loss(self, input, target_is_real, weight=None, reduce_dim=True,
        for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            batchsize = input.size(0)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor,
                weight=weight)
            if not reduce_dim:
                loss = loss.view(batchsize, -1).mean(dim=1)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = input * 0 + (self.real_label if target_is_real else
                self.fake_label)
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

    def __call__(self, input, target_is_real, weight=None, reduce_dim=True,
        for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, weight,
                    reduce_dim, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, weight, reduce_dim,
                for_discriminator)


class VGGLoss(nn.Module):

    def __init__(self, opt, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG_Activations([1, 6, 11, 20, 29])
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def compute_loss(self, x_vgg, y_vgg):
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].
                detach())
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


class SPADE(nn.Module):

    def __init__(self, norm_nc, hidden_nc=0, norm='batch', ks=3,
        params_free=False):
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


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


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
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
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
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
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


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(
                input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(
                input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std *
                self.weight) + _unsqueeze_ft(self.bias)
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
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class DataParallel(nn.parallel.DataParallel):

    def replicate(self, module, device_ids):
        replicas = super(DataParallel, self).replicate(module, device_ids)
        replicas[0] = module
        return replicas


class Vgg19(nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True
            ).features
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_NVlabs_few_shot_vid2vid(_paritybench_base):
    pass
    def test_000(self):
        self._check(BaseModel(*[], **{}), [], {})

    def test_001(self):
        self._check(FlowNetFusion(*[], **{'args': _mock_config()}), [torch.rand([4, 11, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FlowNetS(*[], **{'args': _mock_config()}), [torch.rand([4, 12, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(FlowNetSD(*[], **{'args': _mock_config()}), [torch.rand([4, 6, 64, 64])], {})

    def test_004(self):
        self._check(KLDLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(L1(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(L1Loss(*[], **{'args': _mock_config()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(L2(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(L2Loss(*[], **{'args': _mock_config()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(MaskedL1Loss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(MultiScale(*[], **{'args': _mock_config()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(tofp16(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(tofp32(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

