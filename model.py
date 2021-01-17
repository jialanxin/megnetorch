from os import stat
import torch.nn
from torch.nn import Embedding, RReLU, ReLU, Dropout
from torch_geometric.nn import Set2Set, MessagePassing, BatchNorm, CGConv, GINEConv, GENConv, DeepGCNLayer, LayerNorm,TransformerConv


def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))


def fff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))


def ff_output(input_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.ReLU(), Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.ReLU(), Dropout(0.1), torch.nn.Linear(64, output_dim))


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
        # conv = GENConv(32,32,norm="layer",msg_norm=True)
        # act = ReLU()
        # norm = LayerNorm(32,affine=True)
        # self.node_gcn = DeepGCNLayer(conv,norm=norm,act=act)
        self.node_update = NodeUpdate()
        self.edge_update = EdgeUpdate()

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms):
        # bond_connection = torch.cat((bond_atom_1.unsqueeze(
        # dim=0), bond_atom_2.unsqueeze(dim=0)), dim=0)  # (2,sum_of_num_bonds)
        # atoms = self.node_gcn(atoms,bond_connection,bonds)
        bonds = self.edge_update(bonds, bond_atom_1, bond_atom_2, atoms)
        atoms = self.node_update(bonds, bond_atom_1, bond_atom_2, atoms)
        return atoms, bonds


class FirstMegnetBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.megnetlayer = MegNetLayer()
        # self.bond_norm = LayerNorm(32)
        # self.atom_norm = LayerNorm(32)
    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms,batch_mark_for_atoms,batch_mark_for_bonds):
        res_atoms, res_bonds = self.megnetlayer(
            bonds, bond_atom_1, bond_atom_2, atoms)
        bonds = bonds + res_bonds
        atoms = atoms + res_atoms
        # bonds = self.bond_norm(bonds,batch_mark_for_bonds)
        # atoms = self.atom_norm(atoms,batch_mark_for_atoms)
        return atoms, bonds


class FullMegnetBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.megnetlayer = MegNetLayer()
        self.atoms_ff = ff(32)
        self.bonds_ff = ff(32)
        # self.atom_norm_cov = LayerNorm(32)
        # self.bond_norm_cov = LayerNorm(32)
        # self.atom_norm_fc = LayerNorm(32)
        # self.bond_norm_fc = LayerNorm(32)

    def forward(self, bonds, bond_atom_1, bond_atom_2, atoms,batch_mark_for_atoms,batch_mark_for_bonds):
        res_atoms = self.atoms_ff(atoms)
        res_bonds = self.bonds_ff(bonds)
        atoms = atoms+res_atoms
        bonds = bonds+res_bonds
        # atoms = self.atom_norm_fc(atoms,batch_mark_for_atoms)
        # bonds = self.bond_norm_fc(bonds,batch_mark_for_bonds)
        res_atoms, res_bonds = self.megnetlayer(
            bonds, bond_atom_1, bond_atom_2, atoms)
        atoms = atoms + res_atoms
        bonds = bonds + res_bonds
        # atoms = self.atom_norm_cov(atoms,batch_mark_for_atoms)
        # bonds = self.bond_norm_cov(bonds,batch_mark_for_bonds)
        return atoms, bonds

class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = TransformerConv(32,32,edge_dim=32)
        self.atom_fc = ff(32)
        self.edge_update = EdgeUpdate()
        self.bond_fc = ff(32)
    def forward(self,atoms,bonds,bond_atom_1,bond_atom_2):
        #Node Update (Graph Transformer Convolution)
        bond_connection = torch.cat((bond_atom_1.unsqueeze(
            dim=0), bond_atom_2.unsqueeze(dim=0)), dim=0)
        res_atoms = self.attention_layer(atoms,bond_connection,bonds)
        atoms = atoms+res_atoms
        res_atoms = self.atom_fc(atoms)
        atoms = atoms+res_atoms
        
        #Edge Update (MegNet Layer Method)
        res_bonds = self.edge_update(bonds,bond_atom_1,bond_atom_2,atoms)
        bonds = bonds+res_bonds
        res_bonds = self.bond_fc(bonds)
        bonds = bonds+res_bonds
        return atoms, bonds




class MegNet(torch.nn.Module):
    def __init__(self, num_of_megnetblock) -> None:
        super().__init__()
        self.atom_preblock = ff(71)
        self.bond_preblock = ff(100)
        # self.firstblock = FirstMegnetBlock()
        # self.fullblocks = torch.nn.ModuleList(
        #     [FullMegnetBlock() for i in range(num_of_megnetblock)])
        self.fullblocks = torch.nn.ModuleList(
        [EncoderBlock() for i in range(num_of_megnetblock)])
        self.set2set_v = Set2Set(in_channels=32, processing_steps=3)
        self.set2set_e = Set2Set(in_channels=32, processing_steps=3)
        self.output_layer = ff_output(input_dim=128, output_dim=41)

    def forward(self, atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds):
        # (sum_of_num_atoms,atom_info)
        atoms = self.atom_preblock(atoms)
        bonds = self.bond_preblock(bonds)  # (sum_of_num_bonds,bond_info)
        # atoms, bonds = self.firstblock(
        #     bonds, bond_atom_1, bond_atom_2, atoms,batch_mark_for_atoms,batch_mark_for_bonds)
        for block in self.fullblocks:
            atoms, bonds = block(
                atoms,bonds, bond_atom_1, bond_atom_2)
        batch_size = batch_mark_for_bonds.max()+1
        # print(batch_size)
        # (batch_size,bond_info)
        bonds = self.set2set_e(bonds, batch=batch_mark_for_bonds)
        atoms = self.set2set_v(atoms, batch=batch_mark_for_atoms)
        # (batch_size, bond_info+atom_info)
        gather_all = torch.cat((bonds, atoms), dim=1)
        output_spectrum = self.output_layer(
            gather_all)  # (batch_size, raman_info)
        return output_spectrum
