from pymatgen import Structure
from typing import Dict, List, Tuple
from pymatgen import MPRester
from pymatgen.optimization.neighbors import find_points_in_spheres
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import torch
import json
import re


def range_encode(value, min, max, steps):
    value = torch.Tensor([value])
    range_space = torch.linspace(min, max, steps)
    greater = torch.greater_equal(value, range_space).sum()
    encoded = torch.zeros_like(range_space)
    encoded[greater-1] += 1
    return encoded


class CrystalBase:
    def __init__(self, structure: Structure):
        self.structure = structure
        self.atoms = structure.species
        self.num_atoms = len(self.atoms)
        self.atomic_numbers = [atom.Z for atom in self.atoms]

    @property
    def get_atomic_groups(self):
        num_atoms = self.num_atoms
        encoded_atomic_groups = torch.zeros((num_atoms, 18), dtype=torch.float)
        for i, atom in enumerate(self.atoms):
            encoded_atomic_groups[i, atom.group-1] = 1
        return encoded_atomic_groups

    @property
    def get_atomic_periods(self):
        num_atoms = self.num_atoms
        encoded_atomic_periods = torch.zeros((num_atoms, 9), dtype=torch.float)
        for i, atom in enumerate(self.atoms):
            encoded_atomic_periods[i, atom.row-1] = 1
        return encoded_atomic_periods


class CrystalGraph(CrystalBase):
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
    def get_atomic_blocks(self):
        num_atoms = self.num_atoms
        encoded_atomic_groups = torch.zeros((num_atoms, 4), dtype=torch.float)
        for i, atom in enumerate(self.atoms):
            block = atom.block
            if block == "s":
                id = 0
            elif block == "p":
                id = 1
            elif block == "d":
                id = 2
            elif block == "f":
                id = 3
            encoded_atomic_groups[i, id] = 1
        return encoded_atomic_groups

    @staticmethod
    def Gassian_expand(value_list, min_value, max_value, intervals, expand_width):
        centers = torch.linspace(min_value, max_value, intervals)
        value_list = torch.FloatTensor(value_list)
        result = torch.exp(-(value_list[:, None] -
                             centers[None, :])**2/expand_width**2)
        return result

    @property
    def get_atomic_electronegativity(self):
        electronegativity = []
        meet_nan = False
        for atom in self.atoms:
            if np.isnan(atom.X):
                electronegativity.append(-100)
                meet_nan = True
            else:
                electronegativity.append(atom.X)
        encoded_electronegativity = self.Gassian_expand(
            electronegativity, 0.5, 4.0, 20, 0.18)
        return encoded_electronegativity

    @property
    def get_atomic_covalence_redius(self):
        with open("utils/covalence_radius.json", "r") as f:
            covalence_radius_table = json.loads(f.read())
        cov_rad_list = [covalence_radius_table[Z-1]
                        for Z in self.atomic_numbers]
        encoded_cov_rad = self.Gassian_expand(cov_rad_list, 50, 250, 20, 10)
        return encoded_cov_rad

    @property
    def get_atomic_first_ionization_energy(self):
        with open("utils/first_ionization_energy.json", "r") as f:
            first_ionization_energy_table = json.loads(f.read())
        FIE_list = [first_ionization_energy_table[Z-1]
                    for Z in self.atomic_numbers]
        encoded_FIE = self.Gassian_expand(FIE_list, 3, 25, 20, 1.1)
        return encoded_FIE

    @property
    def get_atomic_electron_affinity(self):
        with open("utils/electron_affinity.json", "r") as f:
            electron_affinity_table = json.loads(f.read())
        elec_affi_list = [electron_affinity_table[Z-1]
                          for Z in self.atomic_numbers]
        encoded_elec_affi = self.Gassian_expand(
            elec_affi_list, -3, 3.7, 20, 0.33)
        return encoded_elec_affi

    @property
    def get_valence_electron_number(self):
        patterm = re.compile(r"\d+[spdf]\d+")
        num_atoms = self.num_atoms
        encoded_valence_electron_number = torch.zeros(
            (num_atoms, 12), dtype=torch.float)
        for i, atom in enumerate(self.atoms):
            electronic_structure = atom.electronic_structure
            valence_structure = patterm.findall(electronic_structure)
            valence_electron_number = np.array(
                [int(i[2:]) for i in valence_structure]).sum()
            encoded_valence_electron_number[i, valence_electron_number-1] = 1
        return encoded_valence_electron_number

    def get_raman_mode_numbers(self):
        space_group_calculater = SpacegroupAnalyzer(self.structure)
        symmetrized_structure = space_group_calculater.get_symmetrized_structure()
        space_group = symmetrized_structure.spacegroup.int_number
        wyckoff = symmetrized_structure.wyckoff_letters
        wyckoff_letter, wyckoff_count = np.unique(wyckoff, return_counts=True)
        wyckoff = [f"{count}{letter}" for count,
                   letter in zip(wyckoff_count, wyckoff_letter)]
        with open("utils/raman_modes.json", "r") as f:
            raman_table = json.loads(f.read())
        space_group_modes = []
        for modes in raman_table:
            if modes["SpaceGroupIndex"] == space_group:
                space_group_modes.append(modes)
        num_raman = 0
        for modes in space_group_modes:
            if modes["WP"] in wyckoff:
                num_raman += modes["num_modes"]
        return num_raman

    def encode_bond_length_with_Gaussian_distance(self, min_length: float = 0.0, max_length: float = 5.0, intervals: int = 100, expand_width: float = 0.5) -> np.ndarray:
        bond_length = self.bond_length
        centers = np.linspace(min_length, max_length, intervals)
        result = np.exp(-(bond_length[:, None] -
                          centers[None, :])**2/expand_width**2)
        return result

    def convert_to_model_input(self) -> Dict:
        atoms = torch.cat((self.get_atomic_groups, self.get_atomic_periods, self.get_atomic_electronegativity, self.get_atomic_covalence_redius,
                           self.get_atomic_first_ionization_energy, self.get_atomic_electron_affinity, self.get_atomic_blocks), dim=1)
        state = torch.FloatTensor(self.state)
        bonds = torch.FloatTensor(
            self.encode_bond_length_with_Gaussian_distance())
        bond_atom_1 = torch.LongTensor(self.bond_atom_1)
        bond_atom_2 = torch.LongTensor(self.bond_atom_2)
        num_atoms = self.num_atoms
        num_bonds = bonds.shape[0]
        input = (atoms, bonds, bond_atom_1,
                 bond_atom_2, num_atoms, num_bonds)
        return input


class CrystalEmbedding(CrystalBase):
    def __init__(self, structure, max_atoms=150):
        super().__init__(structure)
        self.max_atoms = max_atoms
        if self.num_atoms > self.max_atoms:
            raise ValueError(f"Crystal exceeds max_atoms({self.max_atoms})")

    @property
    def padding(self):
        has_atoms = torch.zeros((self.num_atoms,))
        not_atoms = torch.ones((self.max_atoms-self.num_atoms,))
        padding = torch.cat((has_atoms, not_atoms)).bool()
        return padding

    @property
    def positions(self):
        sites = self.structure.sites
        for (i, site) in enumerate(sites):
            x = site.x
            y = site.y
            z = site.z
            position = torch.FloatTensor([[x, y, z]])
            # position = self.Gassian_expand(position,min_value=-3,max_value=6,intervals=20,expand_width=0.45) #(3,position_info) (3,20)
            # position = torch.reshape(position,(1,-1)) # (1, position_info) (1, 60)
            if i == 0:
                positions = position  # (1,3)
            else:
                # (num_atoms, position_info) (num_atoms, 3)
                positions = torch.cat((positions, position), dim=0)
        return positions

    @property
    def get_atomic_electronegativity(self):
        electronegativity = []
        meet_nan = False
        for atom in self.atoms:
            if np.isnan(atom.X):
                electronegativity.append(-100)
                meet_nan = True
            else:
                electronegativity.append(atom.X)
        return electronegativity

    @property
    def get_atomic_covalence_radius(self):
        with open("utils/covalence_radius.json", "r") as f:
            covalence_radius_table = json.loads(f.read())
        cov_rad_list = [covalence_radius_table[Z-1]
                        for Z in self.atomic_numbers]
        return cov_rad_list

    @property
    def get_atomic_first_ionization_energy(self):
        with open("utils/first_ionization_energy.json", "r") as f:
            first_ionization_energy_table = json.loads(f.read())
        FIE_list = [first_ionization_energy_table[Z-1]
                    for Z in self.atomic_numbers]
        return FIE_list

    @property
    def get_atomic_electron_affinity(self):
        with open("utils/electron_affinity.json", "r") as f:
            electron_affinity_table = json.loads(f.read())
        elec_affi_list = [electron_affinity_table[Z-1]
                          for Z in self.atomic_numbers]
        return elec_affi_list

    @property
    def get_atomic_weight(self):
        atomic_weight = [atom.atomic_mass for atom in self.atoms]
        return atomic_weight

    @property
    def get_mendeleev_no(self):
        mendeleev_no = [atom.mendeleev_no for atom in self.atoms]
        return mendeleev_no

    @property
    def get_valence_electrons(self):
        patterm = re.compile(r"\d+[spdf]\d+")
        num_atoms = self.num_atoms
        encoded_valence_electron_number = torch.zeros(
            (num_atoms, 32), dtype=torch.float)
        for i, atom in enumerate(self.atoms):
            electronic_structure = atom.electronic_structure
            valence_structure = patterm.findall(electronic_structure)
            for orbit in valence_structure:
                if "s" in orbit:
                    s = int(orbit[2:])
                    encoded_valence_electron_number[i, s-1] += 1
                if "p" in orbit:
                    p = int(orbit[2:])
                    encoded_valence_electron_number[i, 2+p-1] += 1
                if "d" in orbit:
                    d = int(orbit[2:])
                    encoded_valence_electron_number[i, 8+d-1] += 1
                if "f" in orbit:
                    f = int(orbit[2:])
                    encoded_valence_electron_number[i, 18+f-1] += 1
        return encoded_valence_electron_number

    def process_index_feature_input(self, value_list):
        atomic_number_like = torch.LongTensor(value_list)  # (num_atoms,)
        # (max_atoms-num_atoms,)
        padding = torch.zeros(
            (self.max_atoms-self.num_atoms,), dtype=torch.long)
        padded = torch.cat((atomic_number_like, padding))  # (max_atoms,)
        return padded

    def get_space_group_number(self):
        space_group_number = self.structure.get_space_group_info()[1]
        space_group_number = torch.LongTensor([space_group_number])  # (1,)
        return space_group_number

    def get_cell_volume(self):
        cell_volume = torch.FloatTensor([[self.structure.volume]])  # (1,1)
        return cell_volume

    def convert_to_model_input(self) -> Dict:
        atoms_fea = torch.cat((self.get_atomic_groups, self.get_atomic_periods,
                               self.get_valence_electrons), dim=1)  # (num_atoms, 59)
        embedding_dim = atoms_fea.shape[1]
        atoms_padding = torch.zeros(
            (self.max_atoms-self.num_atoms, embedding_dim))
        atoms_padded = torch.cat((atoms_fea, atoms_padding), dim=0)

        positions_dim = self.positions.shape[1]
        positions_padding = torch.ones(
            (self.max_atoms-self.num_atoms, positions_dim))*(-100)
        positions_padded = torch.cat(
            (self.positions, positions_padding), dim=0)  # (max_atoms, 3)

        elecneg = torch.FloatTensor(self.get_atomic_electronegativity).reshape(
            (-1, 1))  # (num_atoms,1)
        padding = torch.ones((self.max_atoms-self.num_atoms, 1))*(-100)
        elecneg_padded = torch.cat((elecneg, padding), dim=0)

        cov_rad = torch.FloatTensor(self.get_atomic_covalence_radius).reshape(
            (-1, 1))  # (num_atoms,1)
        covrad_padded = torch.cat((cov_rad, padding), dim=0)

        FIE = torch.FloatTensor(self.get_atomic_first_ionization_energy).reshape(
            (-1, 1))  # (num_atoms,1)
        FIE_padded = torch.cat((FIE, padding), dim=0)

        elec_affi = torch.FloatTensor(
            self.get_atomic_electron_affinity).reshape((-1, 1))  # (num_atoms,1)
        elecaffi_padded = torch.cat((elec_affi, padding), dim=0)

        atomic_weight = torch.FloatTensor(
            self.get_atomic_weight).reshape((-1, 1))
        atomic_weight_padded = torch.cat((atomic_weight, padding), dim=0)

        atomic_number_padded = self.process_index_feature_input(
            self.atomic_numbers)  # (max_atoms,)

        mendeleev_no_padded = self.process_index_feature_input(
            self.get_mendeleev_no)  # (max_atoms,)

        lattice = torch.FloatTensor(
            self.structure.lattice.matrix).reshape(-1, 1)  # (9, 1)

        space_group_number = self.get_space_group_number()

        cell_volume = self.get_cell_volume()  # (1,1)

        return {"atoms": atoms_padded, "elecneg": elecneg_padded, "covrad": covrad_padded, "FIE": FIE_padded, "elecaffi": elecaffi_padded, "AM": atomic_weight_padded, "AN": atomic_number_padded, "MN": mendeleev_no_padded, "positions": positions_padded, "padding_mask": self.padding, "lattice": lattice, "SGN": space_group_number, "CV": cell_volume}
