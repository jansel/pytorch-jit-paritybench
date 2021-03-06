import sys
_module = sys.modules[__name__]
del sys
conf = _module
ase_interface = _module
energy_force = _module
jit = _module
load_from_neurochem = _module
neurochem_trainer = _module
nnp_training = _module
nnp_training_force = _module
vibration_analysis = _module
setup = _module
common_aev_test = _module
test_aev = _module
test_aev_benzene_md = _module
test_aev_nist = _module
test_aev_tripeptide_md = _module
test_ase = _module
test_data = _module
test_energies = _module
test_ensemble = _module
test_forces = _module
test_jit_builtin_models = _module
test_neurochem = _module
test_padding = _module
test_periodic_table_indexing = _module
test_structure_optim = _module
test_utils = _module
test_vibrational = _module
comp6 = _module
ANI1 = _module
neurochem_calculator = _module
nist = _module
torchani = _module
aev = _module
ase = _module
data = _module
_pyanitools = _module
models = _module
neurochem = _module
trainer = _module
nn = _module
units = _module
utils = _module

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


import torch


from typing import Tuple


from typing import Optional


from torch import Tensor


import math


import torch.utils.tensorboard


import random


import copy


import itertools


import numpy


import time


from typing import NamedTuple


from torch.jit import Final


import functools


from collections import Counter


import collections


from torch.optim import AdamW


from collections import OrderedDict


import torch.utils.data


from collections import defaultdict


class CustomModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchani.models.ANI1x(periodic_table_index=True).double()

    def forward(self, species: Tensor, coordinates: Tensor, return_forces: bool=False, return_hessians: bool=False) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if return_forces or return_hessians:
            coordinates.requires_grad_(True)
        energies = self.model((species, coordinates)).energies
        forces: Optional[Tensor] = None
        hessians: Optional[Tensor] = None
        if return_forces or return_hessians:
            grad = torch.autograd.grad([energies.sum()], [coordinates], create_graph=return_hessians)[0]
            assert grad is not None
            forces = -grad
            if return_hessians:
                hessians = torchani.utils.hessian(coordinates, forces=forces)
        return energies, forces, hessians


class SpeciesAEV(NamedTuple):
    species: Tensor
    aevs: Tensor


def cutoff_cosine(distances: Tensor, cutoff: float) ->Tensor:
    return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


def angular_terms(Rca: float, ShfZ: Tensor, EtaA: Tensor, Zeta: Tensor, ShfA: Tensor, vectors12: Tensor) ->Tensor:
    """Compute the angular subAEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum in the equation is not computed.
    The input tensor have shape (conformations, atoms, N), where N
    is the number of neighbor atom pairs within the cutoff radius and
    output tensor should have shape
    (conformations, atoms, ``self.angular_sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    vectors12 = vectors12.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    distances12 = vectors12.norm(2, dim=-5)
    cos_angles = 0.95 * torch.nn.functional.cosine_similarity(vectors12[0], vectors12[1], dim=-5)
    angles = torch.acos(cos_angles)
    fcj12 = cutoff_cosine(distances12, Rca)
    factor1 = ((1 + torch.cos(angles - ShfZ)) / 2) ** Zeta
    factor2 = torch.exp(-EtaA * (distances12.sum(0) / 2 - ShfA) ** 2)
    ret = 2 * factor1 * factor2 * fcj12.prod(0)
    return ret.flatten(start_dim=-4)


def neighbor_pairs(padding_mask: Tensor, coordinates: Tensor, cell: Tensor, shifts: Tensor, cutoff: float) ->Tuple[Tensor, Tensor]:
    """Compute pairs of atoms that are neighbors

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
            defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
    """
    coordinates = coordinates.detach()
    cell = cell.detach()
    num_atoms = padding_mask.shape[1]
    num_mols = padding_mask.shape[0]
    all_atoms = torch.arange(num_atoms, device=cell.device)
    p12_center = torch.triu_indices(num_atoms, num_atoms, 1, device=cell.device)
    shifts_center = shifts.new_zeros((p12_center.shape[1], 3))
    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t()
    shift_index = prod[0]
    p12 = prod[1:]
    shifts_outside = shifts.index_select(0, shift_index)
    shifts_all = torch.cat([shifts_center, shifts_outside])
    p12_all = torch.cat([p12_center, p12], dim=1)
    shift_values = shifts_all @ cell
    selected_coordinates = coordinates.index_select(1, p12_all.view(-1)).view(num_mols, 2, -1, 3)
    distances = (selected_coordinates[:, (0), (...)] - selected_coordinates[:, (1), (...)] + shift_values).norm(2, -1)
    padding_mask = padding_mask.index_select(1, p12_all.view(-1)).view(2, -1).any(0)
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, (pair_index)]
    shifts = shifts_all.index_select(0, pair_index)
    return molecule_index + atom_index12, shifts


def neighbor_pairs_nopbc(padding_mask: Tensor, coordinates: Tensor, cutoff: float) ->Tensor:
    """Compute pairs of atoms that are neighbors (doesn't use PBC)

    This function bypasses the calculation of shifts and duplication
    of atoms in order to make calculations faster

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules * atoms, 3) for atom coordinates.
        cutoff (float): the cutoff inside which atoms are considered pairs
    """
    coordinates = coordinates.detach()
    current_device = coordinates.device
    num_atoms = padding_mask.shape[1]
    num_mols = padding_mask.shape[0]
    p12_all = torch.triu_indices(num_atoms, num_atoms, 1, device=current_device)
    p12_all_flattened = p12_all.view(-1)
    pair_coordinates = coordinates.index_select(1, p12_all_flattened).view(num_mols, 2, -1, 3)
    distances = (pair_coordinates[:, (0), (...)] - pair_coordinates[:, (1), (...)]).norm(2, -1)
    padding_mask = padding_mask.index_select(1, p12_all_flattened).view(num_mols, 2, -1).any(dim=1)
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, (pair_index)] + molecule_index
    return atom_index12


def radial_terms(Rcr: float, EtaR: Tensor, ShfR: Tensor, distances: Tensor) ->Tensor:
    """Compute the radial subAEV terms of the center atom given neighbors

    This correspond to equation (3) in the `ANI paper`_. This function just
    compute the terms. The sum in the equation is not computed.
    The input tensor have shape (conformations, atoms, N), where ``N``
    is the number of neighbor atoms within the cutoff radius and output
    tensor should have shape
    (conformations, atoms, ``self.radial_sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    distances = distances.unsqueeze(-1).unsqueeze(-1)
    fc = cutoff_cosine(distances, Rcr)
    ret = 0.25 * torch.exp(-EtaR * (distances - ShfR) ** 2) * fc
    return ret.flatten(start_dim=-2)


def cumsum_from_zero(input_: Tensor) ->Tensor:
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum


def triple_by_molecule(atom_index12: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
    """Input: indices for pairs of atoms that are close to each other.
    each pair only appear once, i.e. only one of the pairs (1, 2) and
    (2, 1) exists.

    Output: indices for all central atoms and it pairs of neighbors. For
    example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
    central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
    are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
    """
    ai1 = atom_index12.view(-1)
    sorted_ai1, rev_indices = ai1.sort()
    uniqued_central_atom_index, counts = torch.unique_consecutive(sorted_ai1, return_inverse=False, return_counts=True)
    pair_sizes = counts * (counts - 1) // 2
    pair_indices = torch.repeat_interleave(pair_sizes)
    central_atom_index = uniqued_central_atom_index.index_select(0, pair_indices)
    m = counts.max().item() if counts.numel() > 0 else 0
    n = pair_sizes.shape[0]
    intra_pair_indices = torch.tril_indices(m, m, -1, device=ai1.device).unsqueeze(1).expand(-1, n, -1)
    mask = (torch.arange(intra_pair_indices.shape[2], device=ai1.device) < pair_sizes.unsqueeze(1)).flatten()
    sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, (mask)]
    sorted_local_index12 += cumsum_from_zero(counts).index_select(0, pair_indices)
    local_index12 = rev_indices[sorted_local_index12]
    n = atom_index12.shape[1]
    sign12 = (local_index12 < n) * 2 - 1
    return central_atom_index, local_index12 % n, sign12


def compute_aev(species: Tensor, coordinates: Tensor, triu_index: Tensor, constants: Tuple[float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor], sizes: Tuple[int, int, int, int, int], cell_shifts: Optional[Tuple[Tensor, Tensor]]) ->Tensor:
    Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = constants
    num_species, radial_sublength, radial_length, angular_sublength, angular_length = sizes
    num_molecules = species.shape[0]
    num_atoms = species.shape[1]
    num_species_pairs = angular_length // angular_sublength
    coordinates_ = coordinates
    coordinates = coordinates_.flatten(0, 1)
    if cell_shifts is None:
        atom_index12 = neighbor_pairs_nopbc(species == -1, coordinates_, Rcr)
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        vec = selected_coordinates[0] - selected_coordinates[1]
    else:
        cell, shifts = cell_shifts
        atom_index12, shifts = neighbor_pairs(species == -1, coordinates_, cell, shifts, Rcr)
        shift_values = shifts @ cell
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        vec = selected_coordinates[0] - selected_coordinates[1] + shift_values
    species = species.flatten()
    species12 = species[atom_index12]
    distances = vec.norm(2, -1)
    radial_terms_ = radial_terms(Rcr, EtaR, ShfR, distances)
    radial_aev = radial_terms_.new_zeros((num_molecules * num_atoms * num_species, radial_sublength))
    index12 = atom_index12 * num_species + species12.flip(0)
    radial_aev.index_add_(0, index12[0], radial_terms_)
    radial_aev.index_add_(0, index12[1], radial_terms_)
    radial_aev = radial_aev.reshape(num_molecules, num_atoms, radial_length)
    even_closer_indices = (distances <= Rca).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, even_closer_indices)
    species12 = species12.index_select(1, even_closer_indices)
    vec = vec.index_select(0, even_closer_indices)
    central_atom_index, pair_index12, sign12 = triple_by_molecule(atom_index12)
    species12_small = species12[:, (pair_index12)]
    vec12 = vec.index_select(0, pair_index12.view(-1)).view(2, -1, 3) * sign12.unsqueeze(-1)
    species12_ = torch.where(sign12 == 1, species12_small[1], species12_small[0])
    angular_terms_ = angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)
    angular_aev = angular_terms_.new_zeros((num_molecules * num_atoms * num_species_pairs, angular_sublength))
    index = central_atom_index * num_species_pairs + triu_index[species12_[0], species12_[1]]
    angular_aev.index_add_(0, index, angular_terms_)
    angular_aev = angular_aev.reshape(num_molecules, num_atoms, angular_length)
    return torch.cat([radial_aev, angular_aev], dim=-1)


def compute_shifts(cell: Tensor, pbc: Tensor, cutoff: float) ->Tensor:
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
        vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: long tensor of shifts. the center cell and
            symmetric cells are not included.
    """
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances)
    num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
    r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
    r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
    r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)
    return torch.cat([torch.cartesian_prod(r1, r2, r3), torch.cartesian_prod(r1, r2, o), torch.cartesian_prod(r1, r2, -r3), torch.cartesian_prod(r1, o, r3), torch.cartesian_prod(r1, o, o), torch.cartesian_prod(r1, o, -r3), torch.cartesian_prod(r1, -r2, r3), torch.cartesian_prod(r1, -r2, o), torch.cartesian_prod(r1, -r2, -r3), torch.cartesian_prod(o, r2, r3), torch.cartesian_prod(o, r2, o), torch.cartesian_prod(o, r2, -r3), torch.cartesian_prod(o, o, r3)])


def triu_index(num_species: int) ->Tensor:
    species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
    pair_index = torch.arange(species1.shape[0], dtype=torch.long)
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index
    return ret


class AEVComputer(torch.nn.Module):
    """The AEV computer that takes coordinates as input and outputs aevs.

    Arguments:
        Rcr (float): :math:`R_C` in equation (2) when used at equation (3)
            in the `ANI paper`_.
        Rca (float): :math:`R_C` in equation (2) when used at equation (4)
            in the `ANI paper`_.
        EtaR (:class:`torch.Tensor`): The 1D tensor of :math:`\\eta` in
            equation (3) in the `ANI paper`_.
        ShfR (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (3) in the `ANI paper`_.
        EtaA (:class:`torch.Tensor`): The 1D tensor of :math:`\\eta` in
            equation (4) in the `ANI paper`_.
        Zeta (:class:`torch.Tensor`): The 1D tensor of :math:`\\zeta` in
            equation (4) in the `ANI paper`_.
        ShfA (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (4) in the `ANI paper`_.
        ShfZ (:class:`torch.Tensor`): The 1D tensor of :math:`\\theta_s` in
            equation (4) in the `ANI paper`_.
        num_species (int): Number of supported atom types.

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    Rcr: Final[float]
    Rca: Final[float]
    num_species: Final[int]
    radial_sublength: Final[int]
    radial_length: Final[int]
    angular_sublength: Final[int]
    angular_length: Final[int]
    aev_length: Final[int]
    sizes: Final[Tuple[int, int, int, int, int]]

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species):
        super(AEVComputer, self).__init__()
        self.Rcr = Rcr
        self.Rca = Rca
        assert Rca <= Rcr, 'Current implementation of AEVComputer assumes Rca <= Rcr'
        self.num_species = num_species
        self.register_buffer('EtaR', EtaR.view(-1, 1))
        self.register_buffer('ShfR', ShfR.view(1, -1))
        self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))
        self.radial_sublength = self.EtaR.numel() * self.ShfR.numel()
        self.radial_length = self.num_species * self.radial_sublength
        self.angular_sublength = self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * self.ShfZ.numel()
        self.angular_length = self.num_species * (self.num_species + 1) // 2 * self.angular_sublength
        self.aev_length = self.radial_length + self.angular_length
        self.sizes = self.num_species, self.radial_sublength, self.radial_length, self.angular_sublength, self.angular_length
        self.register_buffer('triu_index', triu_index(num_species))
        cutoff = max(self.Rcr, self.Rca)
        default_cell = torch.eye(3, dtype=self.EtaR.dtype, device=self.EtaR.device)
        default_pbc = torch.zeros(3, dtype=torch.bool, device=self.EtaR.device)
        default_shifts = compute_shifts(default_cell, default_pbc, cutoff)
        self.register_buffer('default_cell', default_cell)
        self.register_buffer('default_shifts', default_shifts)

    def constants(self):
        return self.Rcr, self.EtaR, self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA

    def forward(self, input_: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None) ->SpeciesAEV:
        """Compute AEVs

        Arguments:
            input_ (tuple): Can be one of the following two cases:

                If you don't care about periodic boundary conditions at all,
                then input can be a tuple of two tensors: species, coordinates.
                species must have shape ``(N, A)``, coordinates must have shape
                ``(N, A, 3)`` where ``N`` is the number of molecules in a batch,
                and ``A`` is the number of atoms.

                .. warning::

                    The species must be indexed in 0, 1, 2, 3, ..., not the element
                    index in periodic table. Check :class:`torchani.SpeciesConverter`
                    if you want periodic table indexing.

                .. note:: The coordinates, and cell are in Angstrom.

                If you want to apply periodic boundary conditions, then the input
                would be a tuple of two tensors (species, coordinates) and two keyword
                arguments `cell=...` , and `pbc=...` where species and coordinates are
                the same as described above, cell is a tensor of shape (3, 3) of the
                three vectors defining unit cell:

                .. code-block:: python

                    tensor([[x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3]])

                and pbc is boolean vector of size 3 storing if pbc is enabled
                for that direction.

        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length())``
        """
        species, coordinates = input_
        if cell is None and pbc is None:
            aev = compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes, None)
        else:
            assert cell is not None and pbc is not None
            cutoff = max(self.Rcr, self.Rca)
            shifts = compute_shifts(cell, pbc, cutoff)
            aev = compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes, (cell, shifts))
        return SpeciesAEV(species, aev)


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(utils.PERIODIC_TABLE, 1)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(self, input_: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        return SpeciesCoordinates(self.conv_tensor[species], coordinates)


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


def calc():
    calc = ase_interface.ANI(False)
    calc.setnc(nc)
    return calc


conv_au_ev = 27.21138505


radial_length = 64


all_species = ['H', 'C', 'N', 'O']


species_indices = {all_species[i]: i for i in range(len(all_species))}


class NeuroChem:

    def _get_radial_part(self, fullaev):
        if len(fullaev.shape) == 3:
            return fullaev[:, :, :radial_length]
        assert len(fullaev.shape) == 2
        return fullaev[:, :radial_length]

    def _get_angular_part(self, fullaev):
        if len(fullaev.shape) == 3:
            return fullaev[:, :, radial_length:]
        assert len(fullaev.shape) == 2
        return fullaev[:, radial_length:]

    def _per_conformation(self, coordinates, species):
        atoms = coordinates.shape[0]
        mol = ase.Atoms(''.join(species), positions=coordinates)
        mol.set_calculator(calc())
        energy = mol.get_potential_energy() / conv_au_ev
        aevs = [mol.calc.nc.atomicenvironments(j) for j in range(atoms)]
        force = mol.get_forces() / conv_au_ev
        aevs = numpy.stack(aevs)
        return aevs, energy, force

    def __call__(self, args):
        if len(args) == 2:
            return self.from_coordinates_and_species(*args)
        return self.from_atoms_obj(args)

    def from_atoms_obj(self, mol):
        natoms = len(mol)
        mol = mol.copy()
        mol.set_calculator(calc())
        energy = mol.get_potential_energy() / conv_au_ev
        aevs = [mol.calc.nc.atomicenvironments(j) for j in range(natoms)]
        aevs = numpy.stack(aevs)
        force = mol.get_forces() / conv_au_ev
        cell = mol.get_cell(complete=True)
        pbc = mol.get_pbc().astype(numpy.uint8)
        coordinates = mol.get_positions()
        species = numpy.array([species_indices[i] for i in mol.get_chemical_symbols()])
        return coordinates, species, self._get_radial_part(aevs), self._get_angular_part(aevs), energy, force, cell, pbc

    def from_coordinates_and_species(self, coordinates, species):
        if len(coordinates.shape) == 2:
            coordinates = coordinates.reshape(1, -1, 3)
        conformations = coordinates.shape[0]
        results = [self._per_conformation(coordinates[i], species) for i in range(conformations)]
        aevs, energies, forces = zip(*results)
        aevs = numpy.stack(aevs)
        energies = numpy.stack(energies)
        forces = numpy.stack(forces)
        species = numpy.array([species_indices[i] for i in species])
        species = species.reshape(1, -1)
        species = numpy.broadcast_to(species, (coordinates.shape[0], species.shape[1]))
        return coordinates, species, self._get_radial_part(aevs), self._get_angular_part(aevs), energies, forces


neurochem = NeuroChem()


class BuiltinModel(torch.nn.Module):
    """Private template for the builtin ANI models """

    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):
        super(BuiltinModel, self).__init__()
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self._species_to_tensor = species_to_tensor
        self.species = consts.species
        self.periodic_table_index = periodic_table_index
        self.consts = consts
        self.sae_dict = sae_dict

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False, model_index=0):
        consts, sae_file, ensemble_prefix, ensemble_size = cls._parse_neurochem_resources(info_file_path)
        if model_index >= ensemble_size:
            raise ValueError("The ensemble size is only {}, model {} can't be loaded".format(ensemble_size, model_index))
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor
        network_dir = os.path.join('{}{}'.format(ensemble_prefix, model_index), 'networks')
        neural_networks = neurochem.load_model(consts.species, network_dir)
        return cls(species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    @staticmethod
    def _parse_neurochem_resources(info_file_path):

        def get_resource(resource_path, file_path):
            return os.path.join(resource_path, file_path)
        resource_path = os.path.join(os.path.dirname(__file__), 'resources/')
        local_dir = os.path.expanduser('~/.local/torchani/')
        repo_name = 'ani-model-zoo'
        tag_name = 'ani-2x'
        extracted_name = '{}-{}'.format(repo_name, tag_name)
        url = 'https://github.com/aiqm/{}/archive/{}.zip'.format(repo_name, tag_name)
        if not os.path.isfile(get_resource(resource_path, info_file_path)):
            if not os.path.isfile(get_resource(local_dir, info_file_path)):
                None
                resource_res = requests.get(url)
                resource_zip = zipfile.ZipFile(io.BytesIO(resource_res.content))
                try:
                    resource_zip.extractall(resource_path)
                except PermissionError:
                    resource_zip.extractall(local_dir)
                    resource_path = local_dir
            else:
                resource_path = local_dir
            files = glob.glob(os.path.join(resource_path, extracted_name, 'resources', '*'))
            for f in files:
                try:
                    shutil.move(f, resource_path)
                except shutil.Error:
                    pass
            shutil.rmtree(os.path.join(resource_path, extracted_name))
        info_file = get_resource(resource_path, info_file_path)
        with open(info_file) as f:
            lines = [x.strip() for x in f.readlines()][:4]
            const_file_path, sae_file_path, ensemble_prefix_path, ensemble_size = lines
            const_file = get_resource(resource_path, const_file_path)
            sae_file = get_resource(resource_path, sae_file_path)
            ensemble_prefix = get_resource(resource_path, ensemble_prefix_path)
            ensemble_size = int(ensemble_size)
            consts = neurochem.Constants(const_file)
        return consts, sae_file, ensemble_prefix, ensemble_size

    def forward(self, species_coordinates: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None) ->SpeciesEnergies:
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: energies for the given configurations

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor
        self.aev_computer.triu_index = self.aev_computer.triu_index

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        return self._species_to_tensor(*args, **kwargs)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        return ase.Calculator(self.species, self, **kwargs)


class BuiltinEnsemble(BuiltinModel):
    """Private template for the builtin ANI ensemble models.

    ANI ensemble models form the ANI models zoo are instances of this class.
    This class is a torch module that sequentially calculates
    AEVs, then energies from a torchani.Ensemble and then uses EnergyShifter
    to shift those energies. It is essentially a sequential

    'AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=False), or a sequential

    'SpeciesConverter -> AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=True).

    .. note::
        This class is for internal use only, avoid relying on anything from it
        except the public methods, always use ANI1x, ANI1ccx, etc to instance
        the models.
        Also, don't confuse this class with torchani.Ensemble, which is only a
        container for many ANIModel instances and shouldn't be used directly
        for calculations.

    Attributes:
        species_converter (:class:`torchani.nn.SpeciesConverter`): Converts periodic table index to
            internal indices. Only present if periodic_table_index is `True`.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with
            builtin constants
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter with
            builtin Self Atomic Energies.
        periodic_table_index (bool): Whether to use element number in periodic table
            to index species. If set to `False`, then indices must be `0, 1, 2, ..., N - 1`
            where `N` is the number of parametrized species.
    """

    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index):
        super(BuiltinEnsemble, self).__init__(species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):
        consts, sae_file, ensemble_prefix, ensemble_size = cls._parse_neurochem_resources(info_file_path)
        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        energy_shifter, sae_dict = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor
        neural_networks = neurochem.load_model_ensemble(consts.species, ensemble_prefix, ensemble_size)
        return cls(species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index)

    def __getitem__(self, index):
        """Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model

        Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model
        or
        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations. The model consists
        of a sequence AEVComputer -> ANIModel -> EnergyShifter
        and can return an ase calculator and convert species to tensor.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torchani.models.BuiltinModel`) Model ready for
                calculations
        """
        ret = BuiltinModel(self.species_converter, self.aev_computer, self.neural_networks[index], self.energy_shifter, self._species_to_tensor, self.consts, self.sae_dict, self.periodic_table_index)
        return ret

    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)


class ANIModel(torch.nn.ModuleDict):
    """ANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super(ANIModel, self).__init__(self.ensureOrderedDict(modules))

    def forward(self, species_aev: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None) ->SpeciesEnergies:
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        for i, (_, m) in enumerate(self.items()):
            mask = species_ == i
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return SpeciesEnergies(species, torch.sum(output, dim=1))


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None) ->SpeciesEnergies:
        sum_ = 0
        for x in self:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super(Sequential, self).__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""

    def forward(self, x: Tensor) ->Tensor:
        return torch.exp(-x * x)


class EnergyShifter(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    This is a subclass of :class:`torch.nn.Module`, so it can be used directly
    in a pipeline as ``[input->AEVComputer->ANIModel->EnergyShifter->output]``.

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
        fit_intercept (bool): Whether to calculate the intercept during the LSTSQ
            fit. The intercept will also be taken into account to shift energies.
    """

    def __init__(self, self_energies, fit_intercept=False):
        super(EnergyShifter, self).__init__()
        self.fit_intercept = fit_intercept
        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.double)
        self.register_buffer('self_energies', self_energies)

    def sae(self, species):
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]
        self_energies = self.self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        return self_energies.sum(dim=1) + intercept

    def forward(self, species_energies: Tuple[Tensor, Tensor], cell: Optional[Tensor]=None, pbc: Optional[Tensor]=None) ->SpeciesEnergies:
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species)
        return SpeciesEnergies(species, energies + sae)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Gaussian,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_aiqm_torchani(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

