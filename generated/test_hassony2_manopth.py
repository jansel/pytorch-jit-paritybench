import sys
_module = sys.modules[__name__]
del sys
manopth_demo = _module
manopth_mindemo = _module
mano = _module
webuser = _module
lbs = _module
posemapper = _module
serialization = _module
smpl_handpca_wrapper_HAND_only = _module
verts = _module
manopth = _module
argutils = _module
demo = _module
manolayer = _module
rodrigues_layer = _module
rot6d = _module
rotproj = _module
tensutils = _module
setup = _module
test_demo = _module

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


from torch.nn import Module


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def subtract_flat_id(rot_mats):
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device).view(
        1, 9).repeat(rot_mats.shape[0], rot_nb)
    results = rot_mats - id_flat
    return results


def th_posemap_axisang(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped = pose_vectors.contiguous().view(-1, 3)
    rot_mats = rodrigues_layer.batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hassony2_manopth(_paritybench_base):
    pass
