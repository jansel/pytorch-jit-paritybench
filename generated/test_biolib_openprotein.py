import sys
_module = sys.modules[__name__]
del sys
dashboard = _module
experiments = _module
example = _module
models = _module
my_model = _module
rrn = _module
models = _module
tmhmm3 = _module
tm_models = _module
tm_util = _module
op_cli = _module
openprotein = _module
prediction = _module
preprocessing = _module
preprocessing_cli = _module
onnx_export = _module
test_onnx_export = _module
training = _module
util = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import torch


import torch.autograd as autograd


import torch.nn as nn


import numpy as np


from torch.nn.utils.rnn import pack_padded_sequence


import time


from enum import Enum


import random


from torch.utils.data.dataset import Dataset


import torch.onnx


import re


import torch.optim as optim


import collections


import torch.utils.data


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_sequence


PI_TENSOR = torch.tensor([3.141592])


def compute_atan2(y_coord, x_coord):
    eps = 10 ** -4
    ans = torch.atan(y_coord / (x_coord + eps))
    ans = torch.where((y_coord >= 0) & (x_coord < 0), ans + PI_TENSOR, ans)
    ans = torch.where((y_coord < 0) & (x_coord < 0), ans - PI_TENSOR, ans)
    ans = torch.where((y_coord > 0) & (x_coord == 0), PI_TENSOR / 2, ans)
    ans = torch.where((y_coord < 0) & (x_coord == 0), -PI_TENSOR / 2, ans)
    return ans


class SoftToAngle(nn.Module):

    def __init__(self, mixture_size):
        super(SoftToAngle, self).__init__()
        omega_components1 = np.random.uniform(0, 1, int(mixture_size * 0.1))
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size * 0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)
        phi_components = np.genfromtxt('data/mixture_model_pfam_' + str(mixture_size) + '.txt')[:, (1)]
        psi_components = np.genfromtxt('data/mixture_model_pfam_' + str(mixture_size) + '.txt')[:, (2)]
        self.phi_components = nn.Parameter(torch.from_numpy(phi_components).contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components).contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components).view(-1, 1).float())

    def forward(self, x):
        phi_input_sin = torch.matmul(x, torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(x, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(x, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x, torch.cos(self.omega_components))
        phi = compute_atan2(phi_input_sin, phi_input_cos)
        psi = compute_atan2(psi_input_sin, psi_input_cos)
        omega = compute_atan2(omega_input_sin, omega_input_cos)
        return torch.cat((phi, psi, omega), 2)


def calc_angular_difference(values_1, values_2):
    values_1 = values_1.transpose(0, 1).contiguous()
    values_2 = values_2.transpose(0, 1).contiguous()
    acc = 0
    for idx, _ in enumerate(values_1):
        assert values_1[idx].shape[1] == 3
        assert values_2[idx].shape[1] == 3
        a1_element = values_1[idx].view(-1, 1)
        a2_element = values_2[idx].view(-1, 1)
        acc += torch.sqrt(torch.mean(torch.min(torch.abs(a2_element - a1_element), 2 * math.pi - torch.abs(a2_element - a1_element)) ** 2))
    return acc / values_1.shape[0]


def calc_pairwise_distances(chain_a, chain_b, use_gpu):
    distance_matrix = torch.Tensor(chain_a.size()[0], chain_b.size()[0]).type(torch.float)
    epsilon = 10 ** -4 * torch.ones(chain_a.size(0), chain_b.size(0))
    if use_gpu:
        distance_matrix = distance_matrix
        epsilon = epsilon
    for idx, row in enumerate(chain_a.split(1)):
        distance_matrix[idx] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)
    return torch.sqrt(distance_matrix + epsilon)


def calc_drmsd(chain_a, chain_b, use_gpu=False):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, use_gpu)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, use_gpu)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) / math.sqrt(len(chain_a) * (len(chain_a) - 1))


def transpose_atoms_to_center_of_mass(atoms_matrix):
    center_of_mass = np.matrix([[atoms_matrix[(0), :].sum() / atoms_matrix.shape[1]], [atoms_matrix[(1), :].sum() / atoms_matrix.shape[1]], [atoms_matrix[(2), :].sum() / atoms_matrix.shape[1]]])
    return atoms_matrix - center_of_mass


def calc_rmsd(chain_a, chain_b):
    chain_a_value = chain_a.cpu().numpy().transpose()
    chain_b_value = chain_b.cpu().numpy().transpose()
    X = transpose_atoms_to_center_of_mass(chain_a_value)
    Y = transpose_atoms_to_center_of_mass(chain_b_value)
    R = Y * X.transpose()
    _, S, _ = np.linalg.svd(R)
    E0 = sum(list(np.linalg.norm(x) ** 2 for x in X.transpose()) + list(np.linalg.norm(x) ** 2 for x in Y.transpose()))
    TraceS = sum(S)
    RMSD = np.sqrt(1 / len(X.transpose()) * (E0 - 2 * TraceS))
    return RMSD


def compute_cross(tensor_a, tensor_b, dim):
    result = []
    x = torch.zeros(1).long()
    y = torch.ones(1).long()
    z = torch.ones(1).long() * 2
    ax = torch.index_select(tensor_a, dim, x).squeeze(dim)
    ay = torch.index_select(tensor_a, dim, y).squeeze(dim)
    az = torch.index_select(tensor_a, dim, z).squeeze(dim)
    bx = torch.index_select(tensor_b, dim, x).squeeze(dim)
    by = torch.index_select(tensor_b, dim, y).squeeze(dim)
    bz = torch.index_select(tensor_b, dim, z).squeeze(dim)
    result.append(ay * bz - az * by)
    result.append(az * bx - ax * bz)
    result.append(ax * by - ay * bx)
    result = torch.stack(result, dim=dim)
    return result


def compute_dihedral_list(atomic_coords):
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba_normalized = ba / ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba_normalized
    n1_vec = compute_cross(ba_normalized[:-2], ba_neg[1:-1], dim=1)
    n2_vec = compute_cross(ba_neg[1:-1], ba_normalized[2:], dim=1)
    n1_vec_normalized = n1_vec / n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec_normalized = n2_vec / n2_vec.norm(dim=1).unsqueeze(1)
    m1_vec = compute_cross(n1_vec_normalized, ba_neg[1:-1], dim=1)
    x_value = torch.sum(n1_vec_normalized * n2_vec_normalized, dim=1)
    y_value = torch.sum(m1_vec * n2_vec_normalized, dim=1)
    return compute_atan2(y_value, x_value)


def calculate_dihedral_angles(atomic_coords, use_gpu):
    atomic_coords = atomic_coords.contiguous().view(-1, 3)
    zero_tensor = torch.zeros(1)
    if use_gpu:
        zero_tensor = zero_tensor
    angles = torch.cat((zero_tensor, zero_tensor, compute_dihedral_list(atomic_coords), zero_tensor)).view(-1, 3)
    return angles


def calculate_dihedral_angles_over_minibatch(atomic_coords_padded, batch_sizes, use_gpu):
    angles = []
    batch_sizes = torch.LongTensor(batch_sizes)
    atomic_coords = atomic_coords_padded.transpose(0, 1)
    for idx, coordinate in enumerate(atomic_coords.split(1, dim=0)):
        angles_from_coords = torch.index_select(coordinate.squeeze(0), 0, torch.arange(int(batch_sizes[idx].item())))
        angles.append(calculate_dihedral_angles(angles_from_coords, use_gpu))
    return torch.nn.utils.rnn.pad_sequence(angles), batch_sizes


NUM_FRAGMENTS = torch.tensor(6)


BOND_ANGLES = torch.tensor([2.124, 1.941, 2.028], dtype=torch.float32)


BOND_LENGTHS = torch.tensor([145.801, 152.326, 132.868], dtype=torch.float32)


NUM_DIHEDRALS = 3


NUM_DIMENSIONS = 3


def dihedral_to_point(dihedral, use_gpu, bond_lengths=BOND_LENGTHS, bond_angles=BOND_ANGLES):
    """
    Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points
    ready for use in reconstruction of coordinates. Bond lengths and angles
    are based on idealized averages.
    :param dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    :return: Tensor containing points of the protein's backbone atoms.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]
    r_cos_theta = bond_lengths * torch.cos(PI_TENSOR - bond_angles)
    r_sin_theta = bond_lengths * torch.sin(PI_TENSOR - bond_angles)
    if use_gpu:
        r_cos_theta = r_cos_theta
        r_sin_theta = r_sin_theta
    point_x = r_cos_theta.view(1, 1, -1).repeat(num_steps, batch_size, 1)
    point_y = torch.cos(dihedral) * r_sin_theta
    point_z = torch.sin(dihedral) * r_sin_theta
    point = torch.stack([point_x, point_y, point_z])
    point_perm = point.permute(1, 3, 2, 0)
    point_final = point_perm.contiguous().view(num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS)
    return point_final


PNERF_INIT_MATRIX = [torch.tensor([-torch.sqrt(torch.tensor([1.0 / 2.0])), torch.sqrt(torch.tensor([3.0 / 2.0])), 0]), torch.tensor([-torch.sqrt(torch.tensor([2.0])), 0, 0]), torch.tensor([0, 0, 0])]


def point_to_coordinate(points, use_gpu, num_fragments):
    """
    Takes points from dihedral_to_point and sequentially converts them into
    coordinates of a 3D structure.

    Reconstruction is done in parallel by independently reconstructing
    num_fragments and the reconstituting the chain at the end in reverse order.
    The core reconstruction algorithm is NeRF, based on
    DOI: 10.1002/jcc.20237 by Parsons et al. 2005.
    The parallelized version is described in
    https://www.biorxiv.org/content/early/2018/08/06/385450.
    :param points: Tensor containing points as returned by `dihedral_to_point`.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    :param num_fragments: Number of fragments in which the sequence is split
    to perform parallel computation.
    :return: Tensor containing correctly transformed atom coordinates.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    total_num_angles = points.size(0)
    if isinstance(total_num_angles, int):
        total_num_angles = torch.tensor(total_num_angles)
    Triplet = collections.namedtuple('Triplet', 'a, b, c')
    batch_size = points.shape[1]
    init_coords = []
    for row in PNERF_INIT_MATRIX:
        row_tensor = row.repeat([num_fragments * batch_size, 1]).view(num_fragments, batch_size, NUM_DIMENSIONS)
        if use_gpu:
            row_tensor = row_tensor
        init_coords.append(row_tensor)
    init_coords = Triplet(*init_coords)
    padding = torch.fmod(num_fragments - total_num_angles % num_fragments, num_fragments)
    padding_tensor = torch.zeros((padding, points.size(1), points.size(2)))
    points = torch.cat((points, padding_tensor))
    points = points.view(num_fragments, -1, batch_size, NUM_DIMENSIONS)
    points = points.permute(1, 0, 2, 3)

    def extend(prev_three_coords, point, multi_m):
        """
        Aligns an atom or an entire fragment depending on value of `multi_m`
        with the preceding three atoms.
        :param prev_three_coords: Named tuple storing the last three atom
        coordinates ("a", "b", "c") where "c" is the current end of the
        structure (i.e. closest to the atom/ fragment that will be added now).
        Shape NUM_DIHEDRALS x [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMENSIONS].
        First rank depends on value of `multi_m`.
        :param point: Point describing the atom that is added to the structure.
        Shape [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        First rank depends on value of `multi_m`.
        :param multi_m: If True, a single atom is added to the chain for
        multiple fragments in parallel. If False, an single fragment is added.
        Note the different parameter dimensions.
        :return: Coordinates of the atom/ fragment.
        """
        bc = F.normalize(prev_three_coords.c - prev_three_coords.b, dim=-1)
        n = F.normalize(compute_cross(prev_three_coords.b - prev_three_coords.a, bc, dim=2 if multi_m else 1), dim=-1)
        if multi_m:
            m = torch.stack([bc, compute_cross(n, bc, dim=2), n]).permute(1, 2, 3, 0)
        else:
            s = point.shape + (3,)
            m = torch.stack([bc, compute_cross(n, bc, dim=1), n]).permute(1, 2, 0)
            m = m.repeat(s[0], 1, 1).view(s)
        coord = torch.squeeze(torch.matmul(m, point.unsqueeze(3)), dim=3) + prev_three_coords.c
        return coord
    coords_list = []
    prev_three_coords = init_coords
    for point in points.split(1, dim=0):
        coord = extend(prev_three_coords, point.squeeze(0), True)
        coords_list.append(coord)
        prev_three_coords = Triplet(prev_three_coords.b, prev_three_coords.c, coord)
    coords_pretrans = torch.stack(coords_list).permute(1, 0, 2, 3)
    coords_trans = coords_pretrans[-1]
    for idx in torch.arange(end=-1, start=coords_pretrans.shape[0] - 2, step=-1).split(1, dim=0):
        transformed_coords = extend(Triplet(*[di.index_select(0, idx).squeeze(0) for di in prev_three_coords]), coords_trans, False)
        coords_trans = torch.cat([coords_pretrans.index_select(0, idx).squeeze(0), transformed_coords], 0)
    coords_to_pad = torch.index_select(coords_trans, 0, torch.arange(total_num_angles - 1))
    coords = F.pad(coords_to_pad, (0, 0, 0, 0, 1, 0))
    return coords


def get_backbone_positions_from_angles(angular_emissions, batch_sizes, use_gpu):
    points = dihedral_to_point(angular_emissions, use_gpu)
    coordinates = point_to_coordinate(points, use_gpu, num_fragments=NUM_FRAGMENTS) / 100
    return coordinates.transpose(0, 1).contiguous().view(len(batch_sizes), -1, 9).transpose(0, 1), batch_sizes


AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}


def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for protein_id in protein_id_list:
        aa_symbol = _aa_dict_inverse[protein_id.item()]
        aa_list.append(aa_symbol)
    return aa_list


def get_structure_from_angles(aa_list_encoded, angles):
    aa_list = protein_id_to_str(aa_list_encoded)
    omega_list = angles[1:, (0)]
    phi_list = angles[1:, (1)]
    psi_list = angles[:-1, (2)]
    assert len(aa_list) == len(phi_list) + 1 == len(psi_list) + 1 == len(omega_list) + 1
    structure = PeptideBuilder.make_structure(aa_list, list(map(lambda x: math.degrees(x), phi_list)), list(map(lambda x: math.degrees(x), psi_list)), list(map(lambda x: math.degrees(x), omega_list)))
    return structure


def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + str.join(' ', [str(a) for a in args]) + end
    if globals().get('experiment_id') is not None:
        with open('output/' + globals().get('experiment_id') + '.txt', 'a+') as output_file:
            output_file.write(output_string)
            output_file.flush()
    None


def write_to_pdb(structure, prot_id):
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save('output/protein_' + str(prot_id) + '.pdb')


class BaseModel(nn.Module):

    def __init__(self, use_gpu, embedding_size):
        super(BaseModel, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_size = embedding_size
        self.historical_rmsd_avg_values = list()
        self.historical_drmsd_avg_values = list()

    def get_embedding_size(self):
        return self.embedding_size

    def embed(self, original_aa_string):
        max_len = max([s.size(0) for s in original_aa_string])
        seqs = []
        for tensor in original_aa_string:
            padding_to_add = torch.zeros(max_len - tensor.size(0)).int()
            seqs.append(torch.cat((tensor, padding_to_add)))
        data = torch.stack(seqs).transpose(0, 1)
        start_compute_embed = time.time()
        arange_tensor = torch.arange(21).int().repeat(len(original_aa_string), 1).unsqueeze(0).repeat(max_len, 1, 1)
        data_tensor = data.unsqueeze(2).repeat(1, 1, 21)
        embed_tensor = (arange_tensor == data_tensor).float()
        if self.use_gpu:
            embed_tensor = embed_tensor
        end = time.time()
        write_out('Embed time:', end - start_compute_embed)
        return embed_tensor

    def compute_loss(self, minibatch):
        original_aa_string, actual_coords_list, _ = minibatch
        emissions, _backbone_atoms_padded, _batch_sizes = self._get_network_emissions(original_aa_string)
        actual_coords_list_padded = torch.nn.utils.rnn.pad_sequence(actual_coords_list)
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded
        start = time.time()
        if isinstance(_batch_sizes[0], int):
            _batch_sizes = torch.tensor(_batch_sizes)
        emissions_actual, _ = calculate_dihedral_angles_over_minibatch(actual_coords_list_padded, _batch_sizes, self.use_gpu)
        write_out('Angle calculation time:', time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual
        angular_loss = calc_angular_difference(emissions, emissions_actual)
        return angular_loss

    def forward(self, original_aa_string):
        return self._get_network_emissions(original_aa_string)

    def evaluate_model(self, data_loader):
        loss = 0
        data_total = []
        dRMSD_list = []
        RMSD_list = []
        for _, data in enumerate(data_loader, 0):
            primary_sequence, tertiary_positions, _mask = data
            start = time.time()
            predicted_angles, backbone_atoms, batch_sizes = self(primary_sequence)
            write_out('Apply model to validation minibatch:', time.time() - start)
            if predicted_angles == []:
                output_angles, _ = calculate_dihedral_angles_over_minibatch(backbone_atoms, batch_sizes, self.use_gpu)
            else:
                output_angles = predicted_angles
            cpu_predicted_angles = output_angles.transpose(0, 1).cpu().detach()
            if backbone_atoms == []:
                output_positions, _ = get_backbone_positions_from_angles(predicted_angles, batch_sizes, self.use_gpu)
            else:
                output_positions = backbone_atoms
            cpu_predicted_backbone_atoms = output_positions.transpose(0, 1).cpu().detach()
            minibatch_data = list(zip(primary_sequence, tertiary_positions, cpu_predicted_angles, cpu_predicted_backbone_atoms))
            data_total.extend(minibatch_data)
            start = time.time()
            for primary_sequence, tertiary_positions, _predicted_pos, predicted_backbone_atoms in minibatch_data:
                actual_coords = tertiary_positions.transpose(0, 1).contiguous().view(-1, 3)
                predicted_coords = predicted_backbone_atoms[:len(primary_sequence)].transpose(0, 1).contiguous().view(-1, 3).detach()
                rmsd = calc_rmsd(predicted_coords, actual_coords)
                drmsd = calc_drmsd(predicted_coords, actual_coords)
                RMSD_list.append(rmsd)
                dRMSD_list.append(drmsd)
                error = rmsd
                loss += error
                end = time.time()
            write_out('Calculate validation loss for minibatch took:', end - start)
        loss /= data_loader.dataset.__len__()
        self.historical_rmsd_avg_values.append(float(torch.Tensor(RMSD_list).mean()))
        self.historical_drmsd_avg_values.append(float(torch.Tensor(dRMSD_list).mean()))
        prim = data_total[0][0]
        pos = data_total[0][1]
        pos_pred = data_total[0][3]
        if self.use_gpu:
            pos = pos
            pos_pred = pos_pred
        angles = calculate_dihedral_angles(pos, self.use_gpu)
        angles_pred = calculate_dihedral_angles(pos_pred, self.use_gpu)
        write_to_pdb(get_structure_from_angles(prim, angles), 'test')
        write_to_pdb(get_structure_from_angles(prim, angles_pred), 'test_pred')
        data = {}
        data['pdb_data_pred'] = open('output/protein_test_pred.pdb', 'r').read()
        data['pdb_data_true'] = open('output/protein_test.pdb', 'r').read()
        data['phi_actual'] = list([math.degrees(float(v)) for v in angles[1:, (1)]])
        data['psi_actual'] = list([math.degrees(float(v)) for v in angles[:-1, (2)]])
        data['phi_predicted'] = list([math.degrees(float(v)) for v in angles_pred[1:, (1)]])
        data['psi_predicted'] = list([math.degrees(float(v)) for v in angles_pred[:-1, (2)]])
        data['rmsd_avg'] = self.historical_rmsd_avg_values
        data['drmsd_avg'] = self.historical_drmsd_avg_values
        prediction_data = None
        return loss, data, prediction_data

