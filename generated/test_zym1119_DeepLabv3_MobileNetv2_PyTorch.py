import sys
_module = sys.modules[__name__]
del sys
cityscapes = _module
config = _module
layers = _module
main = _module
network = _module
progressbar = _module
transfer_weights = _module
utils = _module

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


from collections import namedtuple


import numpy as np


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.checkpoint import checkpoint_sequential


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, t=6, s=1, dilation=1):
        """
        Initialization of inverted residual block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of the first convolution
        :param dilation: dilation rate of 3*3 depthwise conv
        """
        super(InvertedResidual, self).__init__()
        self.in_ = in_channels
        self.out_ = out_channels
        self.t = t
        self.s = s
        self.dilation = dilation
        self.inverted_residual_block()

    def inverted_residual_block(self):
        """
        Build Inverted Residual Block and residual connection
        """
        block = []
        block.append(nn.Conv2d(self.in_, self.in_ * self.t, 1, bias=False))
        block.append(nn.BatchNorm2d(self.in_ * self.t))
        block.append(nn.ReLU6())
        block.append(nn.Conv2d(self.in_ * self.t, self.in_ * self.t, 3,
            stride=self.s, padding=self.dilation, groups=self.in_ * self.t,
            dilation=self.dilation, bias=False))
        block.append(nn.BatchNorm2d(self.in_ * self.t))
        block.append(nn.ReLU6())
        block.append(nn.Conv2d(self.in_ * self.t, self.out_, 1, bias=False))
        block.append(nn.BatchNorm2d(self.out_))
        self.block = nn.Sequential(*block)
        if self.in_ != self.out_ and self.s != 2:
            self.res_conv = nn.Sequential(nn.Conv2d(self.in_, self.out_, 1,
                bias=False), nn.BatchNorm2d(self.out_))
        else:
            self.res_conv = None

    def forward(self, x):
        if self.s == 1:
            if self.res_conv is None:
                out = x + self.block(x)
            else:
                out = self.res_conv(x) + self.block(x)
        else:
            out = self.block(x)
        return out


class ASPP_plus(nn.Module):

    def __init__(self, params):
        super(ASPP_plus, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 1, bias=
            False), nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
            padding=params.aspp[0], dilation=params.aspp[0], bias=False),
            nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
            padding=params.aspp[1], dilation=params.aspp[1], bias=False),
            nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
            padding=params.aspp[2], dilation=params.aspp[2], bias=False),
            nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(256 * 5, 256, 1, bias=
            False), nn.BatchNorm2d(256))

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        image_pool = self.conv11(image_pool)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear',
            align_corners=True)
        upsample = upsample(image_pool)
        concate = torch.cat([conv11, conv33_1, conv33_2, conv33_3, upsample
            ], dim=1)
        return self.concate_conv(concate)


Label = namedtuple('Label', ['name', 'id', 'trainId', 'category',
    'categoryId', 'hasInstances', 'ignoreInEval', 'color'])


labels = [Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)), Label(
    'rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)), Label(
    'static', 4, 255, 'void', 0, False, True, (0, 0, 0)), Label('dynamic', 
    5, 255, 'void', 0, False, True, (111, 74, 0)), Label('ground', 6, 255,
    'void', 0, False, True, (81, 0, 81)), Label('road', 7, 0, 'flat', 1, 
    False, False, (128, 64, 128)), Label('sidewalk', 8, 1, 'flat', 1, False,
    False, (244, 35, 232)), Label('parking', 9, 255, 'flat', 1, False, True,
    (250, 170, 160)), Label('rail track', 10, 255, 'flat', 1, False, True,
    (230, 150, 140)), Label('building', 11, 2, 'construction', 2, False, 
    False, (70, 70, 70)), Label('wall', 12, 3, 'construction', 2, False, 
    False, (102, 102, 156)), Label('fence', 13, 4, 'construction', 2, False,
    False, (190, 153, 153)), Label('guard rail', 14, 255, 'construction', 2,
    False, True, (180, 165, 180)), Label('bridge', 15, 255, 'construction',
    2, False, True, (150, 100, 100)), Label('tunnel', 16, 255,
    'construction', 2, False, True, (150, 120, 90)), Label('pole', 17, 5,
    'object', 3, False, False, (153, 153, 153)), Label('polegroup', 18, 255,
    'object', 3, False, True, (153, 153, 153)), Label('traffic light', 19, 
    6, 'object', 3, False, False, (250, 170, 30)), Label('traffic sign', 20,
    7, 'object', 3, False, False, (220, 220, 0)), Label('vegetation', 21, 8,
    'nature', 4, False, False, (107, 142, 35)), Label('terrain', 22, 9,
    'nature', 4, False, False, (152, 251, 152)), Label('sky', 23, 10, 'sky',
    5, False, False, (70, 130, 180)), Label('person', 24, 11, 'human', 6, 
    True, False, (220, 20, 60)), Label('rider', 25, 12, 'human', 6, True, 
    False, (255, 0, 0)), Label('car', 26, 13, 'vehicle', 7, True, False, (0,
    0, 142)), Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)), Label(
    'caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)), Label(
    'trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)), Label(
    'train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)), Label(
    'motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)), Label(
    'bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)), Label(
    'license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142))]


def trainId2LabelId(dataset_root, train_id, name):
    """
        Transform trainId map into labelId map
        :param dataset_root: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
        :param id_map: torch tensor
        :param name: name of image, eg. 'gtFine/test/leverkusen/leverkusen_000027_000019_gtFine_labelTrainIds.png'
        """
    assert len(train_id.shape
        ) == 2, 'Id_map must be a 2-D tensor of shape (h, w) where h, w = H, W / output_stride'
    h, w = train_id.shape
    label_id = np.zeros((h, w, 3))
    train_id = train_id.cpu().numpy()
    for label in labels:
        if not label.ignoreInEval:
            label_id[train_id == label.trainId] = np.array([label.id] * 3)
    label_id = label_id.astype(np.uint8)
    name = name.replace('labelTrainIds', 'labelIds')
    cv2.imwrite(dataset_root + '/' + name, label_id)


class bar(object):

    def __init__(self):
        self.start_time = None
        self.iter_per_sec = 0
        self.time = None

    def click(self, current_idx, max_idx, total_length=40):
        """
        Each click is a draw procedure of progressbar
        :param current_idx: range from 0 to max_idx-1
        :param max_idx: maximum iteration
        :param total_length: length of progressbar
        """
        if self.start_time is None:
            self.start_time = time.time()
        else:
            self.time = time.time() - self.start_time
            self.iter_per_sec = 1 / self.time
            perc = current_idx * total_length // max_idx
            print('\r|' + '=' * perc + '>' + ' ' * (total_length - 1 - perc
                ) + '| %d/%d (%.2f iter/s)' % (current_idx + 1, max_idx,
                self.iter_per_sec), end='')
            self.start_time = time.time()

    def close(self):
        self.__init__()
        print('')


WARNING = lambda x: print('\x1b[1;31;2mWARNING: ' + x + '\x1b[0m')


LOG = lambda x: print('\x1b[0;31;2m' + x + '\x1b[0m')


def logits2trainId(logits):
    """
    Transform output of network into trainId map
    :param logits: output tensor of network, before softmax, should be in shape (#classes, h, w)
    """
    upsample = torch.nn.Upsample(size=(1024, 2048), mode='bilinear',
        align_corners=False)
    logits = upsample(logits.unsqueeze_(0))
    logits.squeeze_(0)
    logits = torch.argmax(logits, dim=0)
    return logits


def trainId2color(dataset_root, id_map, name):
    """
    Transform trainId map into color map
    :param dataset_root: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
    :param id_map: torch tensor
    :param name: name of image, eg. 'gtFine/test/leverkusen/leverkusen_000027_000019_gtFine_labelTrainIds.png'
    """
    assert len(id_map.shape
        ) == 2, 'Id_map must be a 2-D tensor of shape (h, w) where h, w = H, W / output_stride'
    h, w = id_map.shape
    color_map = np.zeros((h, w, 3))
    id_map = id_map.cpu().numpy()
    for label in labels:
        if not label.ignoreInEval:
            color_map[id_map == label.trainId] = np.array(label.color)
    color_map = color_map.astype(np.uint8)
    cv2.imwrite(dataset_root + '/' + name, id_map)
    name = name.replace('labelTrainIds', 'color')
    cv2.imwrite(dataset_root + '/' + name, color_map)
    return color_map


class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """
    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, params, datasets):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.params = params
        self.datasets = datasets
        self.pb = bar()
        self.epoch = 0
        self.init_epoch = 0
        self.ckpt_flag = False
        self.train_loss = []
        self.val_loss = []
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)
        block = []
        block.append(nn.Sequential(nn.Conv2d(3, self.params.c[0], 3, stride
            =self.params.s[0], padding=1, bias=False), nn.BatchNorm2d(self.
            params.c[0]), nn.ReLU6()))
        for i in range(6):
            block.extend(layers.get_inverted_residual_block_arr(self.params
                .c[i], self.params.c[i + 1], t=self.params.t[i + 1], s=self
                .params.s[i + 1], n=self.params.n[i + 1]))
        rate = self.params.down_sample_rate // self.params.output_stride
        block.append(layers.InvertedResidual(self.params.c[6], self.params.
            c[6], t=self.params.t[6], s=1, dilation=rate))
        for i in range(3):
            block.append(layers.InvertedResidual(self.params.c[6], self.
                params.c[6], t=self.params.t[6], s=1, dilation=rate * self.
                params.multi_grid[i]))
        block.append(layers.ASPP_plus(self.params))
        block.append(nn.Conv2d(256, self.params.num_class, 1))
        block.append(nn.Upsample(scale_factor=self.params.output_stride,
            mode='bilinear', align_corners=False))
        self.network = nn.Sequential(*block)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.opt = torch.optim.RMSprop(self.network.parameters(), lr=self.
            params.base_lr, momentum=self.params.momentum, weight_decay=
            self.params.weight_decay)
        self.initialize()
        self.load_checkpoint()
        self.load_model()
    """######################"""
    """# Train and Validate #"""
    """######################"""

    def train_one_epoch(self):
        """
        Train network in one epoch
        """
        None
        self.network.train()
        train_loss = 0
        train_loader = DataLoader(self.datasets['train'], batch_size=self.
            params.train_batch, shuffle=self.params.shuffle, num_workers=
            self.params.dataloader_workers)
        train_size = len(self.datasets['train'])
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
        else:
            total_batch = train_size // self.params.train_batch
        for batch_idx, batch in enumerate(train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image, label
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split,
                    image_cuda)
            else:
                out = self.network(image_cuda)
            loss = self.loss_fn(out, label_cuda)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            train_loss += loss.item()
            if self.train_loss == []:
                self.train_loss.append(train_loss)
                self.summary_writer.add_scalar('loss/train_loss', train_loss, 0
                    )
        self.pb.close()
        train_loss /= total_batch
        self.train_loss.append(train_loss)
        self.summary_writer.add_scalar('loss/train_loss', train_loss, self.
            epoch)

    def val_one_epoch(self):
        """
        Validate network in one epoch every m training epochs,
            m is defined in params.val_every
        """
        None
        self.network.eval()
        val_loss = 0
        val_loader = DataLoader(self.datasets['val'], batch_size=self.
            params.val_batch, shuffle=self.params.shuffle, num_workers=self
            .params.dataloader_workers)
        val_size = len(self.datasets['val'])
        if val_size % self.params.val_batch != 0:
            total_batch = val_size // self.params.val_batch + 1
        else:
            total_batch = val_size // self.params.val_batch
        for batch_idx, batch in enumerate(val_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image, label
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split,
                    image_cuda)
            else:
                out = self.network(image_cuda)
            loss = self.loss_fn(out, label_cuda)
            val_loss += loss.item()
            if self.val_loss == []:
                self.val_loss.append(val_loss)
                self.summary_writer.add_scalar('loss/val_loss', val_loss, 0)
        self.pb.close()
        val_loss /= total_batch
        self.val_loss.append(val_loss)
        self.summary_writer.add_scalar('loss/val_loss', val_loss, self.epoch)

    def Train(self):
        """
        Train network in n epochs, n is defined in params.num_epoch
        """
        self.init_epoch = self.epoch
        if self.epoch >= self.params.num_epoch:
            WARNING(
                'Num_epoch should be smaller than current epoch. Skip training......\n'
                )
        else:
            for _ in range(self.epoch, self.params.num_epoch):
                self.epoch += 1
                None
                self.train_one_epoch()
                if self.epoch % self.params.display == 0:
                    None
                if self.params.should_save:
                    if self.epoch % self.params.save_every == 0:
                        self.save_checkpoint()
                if self.params.should_val:
                    if self.epoch % self.params.val_every == 0:
                        self.val_one_epoch()
                        None
                self.adjust_lr()
            if self.params.should_save:
                self.save_checkpoint()
            self.plot_curve()

    def Test(self):
        """
        Test network on test set
        """
        None
        torch.cuda.empty_cache()
        self.network.eval()
        test_loader = DataLoader(self.datasets['test'], batch_size=self.
            params.test_batch, shuffle=False, num_workers=self.params.
            dataloader_workers)
        test_size = len(self.datasets['test'])
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch
        for batch_idx, batch in enumerate(test_loader):
            self.pb.click(batch_idx, total_batch)
            image, label, name = batch['image'], batch['label'], batch[
                'label_name']
            image_cuda, label_cuda = image, label
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split,
                    image_cuda)
            else:
                out = self.network(image_cuda)
            for i in range(self.params.test_batch):
                idx = batch_idx * self.params.test_batch + i
                id_map = logits2trainId(out[i, ...])
                color_map = trainId2color(self.params.dataset_root, id_map,
                    name=name[i])
                trainId2LabelId(self.params.dataset_root, id_map, name=name[i])
                image_orig = image[i].numpy().transpose(1, 2, 0)
                image_orig = image_orig * 255
                image_orig = image_orig.astype(np.uint8)
                self.summary_writer.add_image('test/img_%d/orig' % idx,
                    image_orig, idx)
                self.summary_writer.add_image('test/img_%d/seg' % idx,
                    color_map, idx)
    """##########################"""
    """# Model Save and Restore #"""
    """##########################"""

    def save_checkpoint(self):
        save_dict = {'epoch': self.epoch, 'train_loss': self.train_loss,
            'val_loss': self.val_loss, 'state_dict': self.network.
            state_dict(), 'optimizer': self.opt.state_dict()}
        torch.save(save_dict, self.params.ckpt_dir + 
            'Checkpoint_epoch_%d.pth.tar' % self.epoch)
        None

    def load_checkpoint(self):
        """
        Load checkpoint from given path
        """
        if self.params.resume_from is not None and os.path.exists(self.
            params.resume_from):
            try:
                LOG('Loading Checkpoint at %s' % self.params.resume_from)
                ckpt = torch.load(self.params.resume_from)
                self.epoch = ckpt['epoch']
                try:
                    self.train_loss = ckpt['train_loss']
                    self.val_loss = ckpt['val_loss']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.network.load_state_dict(ckpt['state_dict'])
                self.opt.load_state_dict(ckpt['optimizer'])
                LOG('Checkpoint Loaded!')
                LOG('Current Epoch: %d' % self.epoch)
                self.ckpt_flag = True
            except:
                WARNING(
                    'Cannot load checkpoint from %s. Start loading pre-trained model......'
                     % self.params.resume_from)
        else:
            WARNING(
                'Checkpoint do not exists. Start loading pre-trained model......'
                )

    def load_model(self):
        """
        Load ImageNet pre-trained model into MobileNetv2 backbone, only happen when
            no checkpoint is loaded
        """
        if self.ckpt_flag:
            LOG('Skip Loading Pre-trained Model......')
        elif self.params.pre_trained_from is not None and os.path.exists(self
            .params.pre_trained_from):
            try:
                LOG('Loading Pre-trained Model at %s' % self.params.
                    pre_trained_from)
                pretrain = torch.load(self.params.pre_trained_from)
                self.network.load_state_dict(pretrain)
                LOG('Pre-trained Model Loaded!')
            except:
                WARNING('Cannot load pre-trained model. Start training......')
        else:
            WARNING('Pre-trained model do not exits. Start training......')
    """#############"""
    """# Utilities #"""
    """#############"""

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * (1 - float(self.epoch) / self
            .params.num_epoch) ** self.params.power
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        None
        self.summary_writer.add_scalar('learning_rate', learning_rate, self
            .epoch)

    def plot_curve(self):
        """
        Plot train/val loss curve
        """
        x1 = np.arange(self.init_epoch, self.params.num_epoch + 1, dtype=np.int
            ).tolist()
        x2 = np.linspace(self.init_epoch, self.epoch, num=(self.epoch -
            self.init_epoch) // self.params.val_every + 1, dtype=np.int64)
        plt.plot(x1, self.train_loss, label='train_loss')
        plt.plot(x2, self.val_loss, label='val_loss')
        plt.legend(loc='best')
        plt.title('Train/Val loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zym1119_DeepLabv3_MobileNetv2_PyTorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(InvertedResidual(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

