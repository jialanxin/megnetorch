import torch
from utils.graph import EncodedGraph


class Batch(EncodedGraph):
    def __init__(self, atoms, bonds, bond_atom_1, bond_atom_2, num_atoms, num_bonds, batch_mark_for_atoms, batch_mark_for_bonds, ramans_of_batch):
        super().__init__(atoms, bonds,
                         bond_atom_1, bond_atom_2, num_atoms, num_bonds)
        self.batch_mark_for_atoms = batch_mark_for_atoms
        self.batch_mark_for_bonds = batch_mark_for_bonds
        self.ramans_of_batch = ramans_of_batch

    def to(self, device):
        self.atoms = self.atoms.to(device)
        self.bonds = self.bonds.to(device)
        self.bond_atom_1 = self.bond_atom_1.to(device)
        self.bond_atom_2 = self.bond_atom_2.to(device)
        self.batch_mark_for_atoms = self.batch_mark_for_atoms.to(device)
        self.batch_mark_for_bonds = self.batch_mark_for_bonds.to(device)
        self.ramans_of_batch = self.ramans_of_batch.to(device)


def collate_fn(structure_list):
    num_of_structures = len(structure_list)
    # print("num_of_strucutres:",num_of_structures)
    for i in range(num_of_structures):
        graph, raman = structure_list[i]
        atoms, bonds, bond_atom_1, bond_atom_2, num_atoms, num_bonds = graph.atoms, graph.bonds, graph.bond_atom_1, graph.bond_atom_2, graph.num_atoms, graph.num_bonds
        if i == 0:
            atoms_of_batch = atoms  # (num_atoms,)
            # print("atoms_of_batch:",atoms_of_batch)
            bonds_of_batch = bonds  # (num_bonds,bond_info)
            bond_atom_1_of_batch = bond_atom_1  # (num_bonds,)
            bond_atom_2_of_batch = bond_atom_2  # (num_bonds,)
            # print("num_atoms:",num_atoms)
            batch_mark_for_atoms = torch.LongTensor(
                [i for count in range(num_atoms)])  # (num_of_atoms,)
            # print("batch_mark_for_atoms:",batch_mark_for_atoms)
            batch_mark_for_bonds = torch.LongTensor(
                [i for count in range(num_bonds)])  # (num_of_bonds,)
            ramans_of_batch = raman.unsqueeze(dim=0)  # (1,raman_size)
        else:
            atoms_of_batch = torch.cat(
                (atoms_of_batch, atoms), dim=0)  # (sum_of_num_atoms,)
            # (sum_of_num_bonds,bond_info)
            bonds_of_batch = torch.cat((bonds_of_batch, bonds), dim=0)
            bond_atom_1 = bond_atom_1+batch_mark_for_atoms.shape[0]
            bond_atom_1_of_batch = torch.cat(
                (bond_atom_1_of_batch, bond_atom_1), dim=0)  # (sum_of_num_bonds,)
            bond_atom_2 = bond_atom_2+batch_mark_for_atoms.shape[0]
            bond_atom_2_of_batch = torch.cat(
                (bond_atom_2_of_batch, bond_atom_2), dim=0)  # (sum_of_num_bonds,)
            batch_mark_for_atoms = torch.cat((batch_mark_for_atoms, torch.LongTensor(
                [i for count in range(num_atoms)])))  # (sum_of_num_atoms,)
            batch_mark_for_bonds = torch.cat((batch_mark_for_bonds, torch.LongTensor(
                [i for count in range(num_bonds)])))  # (sum_of_num_bonds,)
            # (batch_size,raman_info)
            ramans_of_batch = torch.cat(
                (ramans_of_batch, raman.unsqueeze(dim=0)), dim=0)
    num_atoms = atoms_of_batch.shape[0]
    num_bonds = bonds_of_batch.shape[0]
    return atoms_of_batch, bonds_of_batch, bond_atom_1_of_batch, bond_atom_2_of_batch, num_atoms, num_bonds, batch_mark_for_atoms, batch_mark_for_bonds, ramans_of_batch
