from pymatgen import Structure
from typing import Dict, List, Tuple
from pymatgen import MPRester
from pymatgen.optimization.neighbors import find_points_in_spheres
import numpy as np
import torch
import json

def range_encode(value,min,max,steps):
    value = torch.Tensor([value])
    range_space = torch.linspace(min,max,steps)
    greater = torch.greater_equal(value,range_space).sum()
    encoded = torch.zeros_like(range_space)
    encoded[greater-1] += 1
    return encoded




class EncodedGraph:
    def __init__(self, atoms, bonds, bond_atom_1, bond_atom_2, num_atoms, num_bonds):
        self.atoms = atoms
        self.bonds = bonds
        self.bond_atom_1 = bond_atom_1
        self.bond_atom_2 = bond_atom_2
        self.num_atoms = num_atoms
        self.num_bonds = num_bonds


class CrystalGraph:
    @staticmethod
    def get_neighbors_within_cutoff(structure: Structure, cutoff: float = 5.0, tolerence: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        period_boundary_condition = np.array([1, 1, 1])
        lattice_matrix = np.array(structure.lattice.matrix)
        cart_coords = structure.cart_coords
        center_indices, neighbor_indices, _, distances = find_points_in_spheres(
            cart_coords, cart_coords, cutoff, period_boundary_condition, lattice_matrix, tolerence)
        exclude_self = (center_indices != neighbor_indices) | (
            distances > tolerence)
        return center_indices[exclude_self], neighbor_indices[exclude_self], distances[exclude_self]

    def __init__(self, structure: Structure):
        super().__init__()
        self.structure = structure
        self.atoms = structure.species
        self.num_atoms = len(self.atoms)
        self.atomic_numbers = [atom.Z for atom in self.atoms]
        self.bond_atom_1, self.bond_atom_2, self.bond_length = self.get_neighbors_within_cutoff(
            structure)
        if np.size(np.unique(self.bond_atom_1)) < len(self.atomic_numbers):
            raise RuntimeError("Isolated atoms found in the structure")
        self.state = np.array([0.0, 0.0], dtype=np.float)
    @property
    def get_atomic_periods(self):
        num_atoms = self.num_atoms
        encoded_atomic_periods = torch.zeros((num_atoms,9),dtype=torch.float)
        for i,atom in enumerate(self.atoms):
            encoded_atomic_periods[i,atom.row-1] = 1
        return encoded_atomic_periods
    @property
    def get_atomic_groups(self):
        num_atoms = self.num_atoms
        encoded_atomic_groups = torch.zeros((num_atoms,18),dtype=torch.float)
        for i, atom in enumerate(self.atoms):
            encoded_atomic_groups[i,atom.group-1] = 1
        return encoded_atomic_groups
    @staticmethod
    def Gassian_expand(value_list,min_value,max_value,intervals,expand_width):
        centers = torch.linspace(min_value,max_value,intervals)
        value_list = torch.FloatTensor(value_list)
        result  = torch.exp(-(value_list[:,None]-centers[None,:])**2/expand_width**2)
        return result
    @property
    def get_atomic_electronegativity(self):
        electronegativity = []
        meet_nan = False
        for atom in self.atoms:
            if np.isnan(atom.X)  :
                electronegativity.append(-100)
                meet_nan = True
            else:
                electronegativity.append(atom.X)
        encoded_electronegativity = self.Gassian_expand(electronegativity,0.5,4.0,10,0.35)
        return encoded_electronegativity
    @property
    def get_atomic_covalence_redius(self):
        with open("utils/covalence_radius.json","r") as f:
            covalence_radius_table = json.loads(f.read())
        cov_rad_list = [covalence_radius_table[Z-1] for Z in self.atomic_numbers]
        encoded_cov_rad = self.Gassian_expand(cov_rad_list,50,250,10,20)
        return encoded_cov_rad
    def encode_bond_length_with_Gaussian_distance(self, min_length: float = 0.0, max_length: float = 5.0, intervals: int = 100, expand_width: float = 0.5) -> np.ndarray:
        bond_length = self.bond_length
        centers = np.linspace(min_length, max_length, intervals)
        result = np.exp(-(bond_length[:, None] -
                          centers[None, :])**2/expand_width**2)
        return result
    def convert_to_model_input(self) -> Dict:
        atoms = torch.cat((self.get_atomic_groups,self.get_atomic_periods,self.get_atomic_electronegativity,self.get_atomic_covalence_redius),dim=1)
        state = torch.FloatTensor(self.state)
        bonds = torch.FloatTensor(
            self.encode_bond_length_with_Gaussian_distance())
        bond_atom_1 = torch.LongTensor(self.bond_atom_1)
        bond_atom_2 = torch.LongTensor(self.bond_atom_2)
        num_atoms = self.num_atoms
        num_bonds = bonds.shape[0]
        input = EncodedGraph(atoms, bonds, bond_atom_1,
                             bond_atom_2, num_atoms, num_bonds)
        return input
