from os import stat
import torch.nn
from torch.nn import Embedding
from torch_geometric.nn import Set2Set, MessagePassing, BatchNorm


def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 64), torch.nn.RReLU(), torch.nn.Linear(64, 32))


def fff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 64), torch.nn.RReLU(), torch.nn.Linear(64, 64), torch.nn.RReLU(), torch.nn.Linear(64, 32))


def ff_output(input_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.RReLU(), torch.nn.Linear(128, 64), torch.nn.RReLU(), torch.nn.Linear(64, output_dim))


class EdgeUpdate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.phi_e = fff(96)

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms):
        sum_of_num_bonds = bonds.shape[0]
        atom_info = atoms.shape[1]
        # for ix_bond in range(sum_of_num_bonds):
        #     for res in range(atom_info):
        #        atom_1_to_bonds[ix_bond,res] =atoms[bond_atom_1[ix_bond,res],res]
        bond_atom_1 = bond_atom_1.unsqueeze(dim=1).repeat(
            (1, atom_info))  # (sum_of_num_bonds,atom_info)
        bond_atom_2 = bond_atom_2.unsqueeze(dim=1).repeat(
            (1, atom_info))  # (sum_of_num_bonds,atom_info)
        # (sum_of_num_bonds,atom_info)
        atom_1_to_bonds = torch.gather(input=atoms, dim=0, index=bond_atom_1)
        # (sum_of_num_bonds,atom_info)
        atom_2_to_bonds = torch.gather(input=atoms, dim=0, index=bond_atom_2)
        # (sum_of_num_bonds,atom_info*2+bond_info)
        bonds = torch.cat((atom_1_to_bonds, atom_2_to_bonds, bonds), dim=1)
        bonds = self.phi_e(bonds)  # (sum_of_num_bonds,bond_info)
        return bonds


class NodeUpdate(MessagePassing):
    def __init__(self):
        super(NodeUpdate, self).__init__(aggr="mean")
        self.phi_v = fff(64)

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms):
        bond_connection = torch.cat((bond_atom_1.unsqueeze(
            dim=0), bond_atom_2.unsqueeze(dim=0)), dim=0)  # (2,sum_of_num_bonds)
        # (sum_of_num_atoms,bond_info)
        bonds_to_atoms = self.propagate(bond_connection, edge_attr=bonds)
        # (sum_of_num_atoms,bond_info+atom_info)
        atoms = torch.cat((bonds_to_atoms, atoms), dim=1)
        atoms = self.phi_v(atoms)  # (sum_of_num_atoms,atom_info)
        return atoms

    def message(self, edge_attr):
        return edge_attr


class MegNetLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_update = EdgeUpdate()
        self.node_update = NodeUpdate()

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms):
        bonds = self.edge_update(bonds, bond_atom_1, bond_atom_2, atoms)
        atoms = self.node_update(bonds, bond_atom_1, bond_atom_2, atoms)
        return bonds, atoms


class FirstMegnetBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.megnetlayer = MegNetLayer()

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms):
        residual_bonds = bonds.clone()
        residual_atoms = atoms.clone()
        residual_bonds, residual_atoms = self.megnetlayer(
            residual_bonds, bond_atom_1, bond_atom_2, residual_atoms)
        atoms = atoms + residual_atoms
        bonds = bonds + residual_bonds
        return bonds, atoms


class FullMegnetBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.megnetlayer = MegNetLayer()
        self.atoms_ff = ff(32)
        self.bonds_ff = ff(32)

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms):
        residual_bonds = bonds.clone()
        residual_atoms = atoms.clone()
        residual_bonds = self.bonds_ff(residual_bonds)
        residual_atoms = self.atoms_ff(residual_atoms)
        residual_bonds, residual_atoms = self.megnetlayer(
            residual_bonds, bond_atom_1, bond_atom_2, residual_atoms)
        atoms = atoms + residual_atoms
        bonds = bonds + residual_bonds
        return bonds, atoms


class MegNet(torch.nn.Module):
    def __init__(self, num_of_megnetblock) -> None:
        super().__init__()
        self.atom_preblock = ff(27)
        self.bond_preblock = ff(100)
        self.firstblock = FirstMegnetBlock()
        self.fullblocks = torch.nn.ModuleList(
            [FullMegnetBlock() for i in range(num_of_megnetblock)])
        self.set2set_e = Set2Set(in_channels=32, processing_steps=3)
        self.set2set_v = Set2Set(in_channels=32, processing_steps=3)
        self.output_layer = ff_output(input_dim=128, output_dim=200)

    def forward(self, atoms, state, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds):
        # (sum_of_num_atoms,atom_info)
        atoms = self.atom_preblock(atoms)
        bonds = self.bond_preblock(bonds)  # (sum_of_num_bonds,bond_info)
        bonds, atoms = self.firstblock(
            bonds, bond_atom_1, bond_atom_2, atoms)
        for block in self.fullblocks:
            bonds, atoms = block(
                bonds, bond_atom_1, bond_atom_2, atoms)
        batch_size = batch_mark_for_bonds.max()+1
        # (batch_size,bond_info)
        bonds = self.set2set_e(bonds, batch=batch_mark_for_bonds)
        # (batch_size,atom_info)
        atoms = self.set2set_v(atoms, batch=batch_mark_for_atoms)
        # (batch_size, bond_info+atom_info)
        gather_all = torch.cat(
            (bonds, atoms), dim=1)
        output_spectrum = self.output_layer(
            gather_all)  # (batch_size, raman_info)
        return output_spectrum
