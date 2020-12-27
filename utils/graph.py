from pymatgen import Structure
from typing import Dict, List,Tuple
from pymatgen import MPRester
from pymatgen.optimization.neighbors import find_points_in_spheres
import numpy as np
import torch



class CrystalGraph:
    @staticmethod
    def get_atomic_numbers(structure:Structure) -> List[int]:
        return [i.specie.Z for i in structure]

    @staticmethod
    def get_neighbors_within_cutoff(structure:Structure, cutoff:float=5.0, tolerence: float=1e-8) ->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        period_boundary_condition = np.array([1,1,1])
        lattice_matrix = np.array(structure.lattice.matrix)
        cart_coords = structure.cart_coords
        center_indices,neighbor_indices,_,distances = find_points_in_spheres(cart_coords,cart_coords,cutoff,period_boundary_condition,lattice_matrix,tolerence)
        exclude_self = (center_indices != neighbor_indices) | (distances > tolerence)
        return center_indices[exclude_self], neighbor_indices[exclude_self], distances[exclude_self]
    def __init__(self,structure:Structure):
        super().__init__()
        self.atomic_numbers = np.array(self.get_atomic_numbers(structure),dtype=np.int)
        self.bond_atom_1,self.bond_atom_2,self.bond_length = self.get_neighbors_within_cutoff(structure)
        if np.size(np.unique(self.bond_atom_1)) < len(self.atomic_numbers):
            raise RuntimeError("Isolated atoms found in the structure")
        self.state = np.array([0.0,0.0],dtype=np.float)
    def encode_bond_length_with_Gaussian_distance(self,min_length:float=0.0,max_length:float=5.0,intervals:int=100,expand_width:float=0.5)->np.ndarray:
        bond_length = self.bond_length
        centers = np.linspace(min_length,max_length,intervals)
        result = np.exp(-(bond_length[:,None]-centers[None,:])**2/expand_width**2)
        return result
    def convert_to_model_input(self)->Dict:
        atoms = torch.LongTensor(self.atomic_numbers)
        state = torch.FloatTensor(self.state)
        bonds = torch.FloatTensor(self.encode_bond_length_with_Gaussian_distance())
        bond_atom_1 = torch.LongTensor(self.bond_atom_1)
        bond_atom_2 = torch.LongTensor(self.bond_atom_2)
        num_atoms = atoms.shape[0]
        num_bonds = bonds.shape[0]
        input = {"atoms":atoms,"state":state,"bonds":bonds,"bond_atom_1":bond_atom_1,"bond_atom_2":bond_atom_2,"num_atoms":num_atoms,"num_bonds":num_bonds}
        return input

class CrystalGraphWithAtomicFeatures(CrystalGraph):
    def __init__(self,structure):
        super(CrystalGraphWithAtomicFeatures,self).__init__(structure)
        self.atoms = [i.specie for i in structure]
        self.num_atoms = len(self.atoms)
    def encoded_atom_periods(self):
        periods = torch.LongTensor([i.row for i in self.atoms])
        encoded = torch.zeros((self.num_atoms,9)) # (num_atoms,9)
        for i,period in enumerate(periods):
            encoded[i,period-1] = 1
        return encoded
    def encoded_atom_groups(self):
        groups = torch.LongTensor([i.group for i in self.atoms])
        encoded = torch.zeros((self.num_atoms,18)) # (num_atoms,18)
        for i,group in enumerate(groups):
            encoded[i,group-1] = 1
        return encoded
    def encoded_atom_mass(self):
        masses = torch.FloatTensor([i.atomic_mass for i in self.atoms])
        centers = torch.linspace(1,245,101)
        results = torch.exp(-(masses[:,None] - centers[None,:])**2/0.5**2)
        return results
    def convert_to_model_input(self):
        atoms = torch.cat((self.encoded_atom_groups(),self.encoded_atom_periods()),dim=1)
        state = torch.FloatTensor(self.state)
        bonds = torch.FloatTensor(self.encode_bond_length_with_Gaussian_distance())
        bond_atom_1 = torch.LongTensor(self.bond_atom_1)
        bond_atom_2 = torch.LongTensor(self.bond_atom_2)
        num_atoms = atoms.shape[0]
        num_bonds = bonds.shape[0]
        input = {"atoms":atoms,"state":state,"bonds":bonds,"bond_atom_1":bond_atom_1,"bond_atom_2":bond_atom_2,"num_atoms":num_atoms,"num_bonds":num_bonds}
        return input

