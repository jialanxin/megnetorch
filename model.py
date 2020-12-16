from os import stat
import torch.nn
from torch.nn import Embedding

def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim,64),torch.nn.Softplus(),torch.nn.Linear(64,32),torch.nn.Softplus())
def fff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim,64),torch.nn.Softplus(),torch.nn.Linear(64,64),torch.nn.Softplus(),torch.nn.Linear(64,32),torch.nn.Softplus())


class MegNetLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phi_e = fff(128)
        self.phi_v = fff(96)
        self.phi_u = fff(96)
    def forward(self,bonds,bond_atom_1,bond_atom_2,atoms,state):
        atom_1_to_bonds = torch.zeros_like(bonds)
        atom_2_to_bonds = torch.zeros_like(bonds)
        num_bonds = bonds.shape[0]
        for ix_bond in range(num_bonds):
            atom_1 = atoms[bond_atom_1[ix_bond],:]
            atom_2 = atoms[bond_atom_2[ix_bond],:]
            atom_1_to_bonds[ix_bond,:] = atom_1
            atom_2_to_bonds[ix_bond,:] = atom_2
        bonds = torch.cat((atom_1_to_bonds,atom_2_to_bonds,bonds,state.repeat((num_bonds,1))),dim=1)
        bonds = self.phi_e(bonds)

        bonds_to_atoms = torch.zeros_like(atoms)
        num_atoms = atoms.shape[0]
        count_bonds_to_atoms = torch.zeros((num_atoms,1))
        for ix_bond in range(num_bonds):
            bond = bonds[ix_bond,:]
            to_atom = bond_atom_1[ix_bond]
            bonds_to_atoms[to_atom,:]  += bond
            count_bonds_to_atoms[to_atom,:] += 1
        bonds_to_atoms = bonds_to_atoms/count_bonds_to_atoms
        atoms = torch.cat((bonds_to_atoms,atoms,state.repeat((num_atoms,1))),dim=1)
        atoms = self.phi_v(atoms)

        bonds_to_state = torch.mean(bonds,dim=0,keepdim=True)
        atoms_to_state = torch.mean(atoms,dim=0,keepdim=True)
        state = torch.cat((bonds_to_state,atoms_to_state,state),dim=1)
        state = self.phi_u(state)
        return bonds, atoms, state

class FirstMegnetBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.megnetlayer = MegNetLayer()
    def forward(self,bonds,bond_atom_1,bond_atom_2,atoms,state):
        residual_bonds = bonds.clone()
        residual_atoms = atoms.clone()
        residual_state = state.clone()
        residual_bonds ,residual_atoms, residual_state = self.megnetlayer(residual_bonds,bond_atom_1,bond_atom_2,residual_atoms,residual_state)
        atoms = atoms + residual_atoms
        bonds = bonds + residual_bonds
        state  = state + residual_state
        return bonds, atoms, state

class FullMegnetBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.megnetlayer = MegNetLayer()
        self.atoms_ff = ff(32)
        self.bonds_ff = ff(32)
        self.state_ff = ff(32)
    def forward(self,bonds,bond_atom_1,bond_atom_2,atoms,state):
        residual_bonds = bonds.clone()
        residual_atoms = atoms.clone()
        residual_state = state.clone()
        residual_bonds = self.bonds_ff(residual_bonds)
        residual_atoms = self.atoms_ff(residual_atoms)
        residual_state = self.state_ff(residual_state)
        residual_bonds,residual_atoms,residual_state = self.megnetlayer(residual_bonds,bond_atom_1,bond_atom_2,residual_atoms,residual_state)
        atoms = atoms + residual_atoms
        bonds = bonds + residual_bonds
        state  = state + residual_state
        return bonds, atoms, state 

class MegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.atomic_embedding = Embedding(95,16)
        self.atom_preblock = ff(16)
        self.bond_preblock = ff(100)
        self.state_preblock = ff(2)
        self.firstblock = FirstMegnetBlock()
        self.fullblocks = nn.ModuleList([FullMegnetBlock() for i in range(3)])

    def forward(self, atoms,state,bonds,bond_atom_1,bond_atom_2):
        atoms_embedded = self.atomic_embedding(torch.squeeze(atoms))
        atoms = self.atom_preblock(atoms_embedded)
        bonds = self.bond_preblock(torch.squeeze(bonds))
        state = self.state_preblock(torch.squeeze(state,dim=0))
        bond_atom_1 = torch.squeeze(bond_atom_1)
        bond_atom_2 = torch.squeeze(bond_atom_2)
        bonds,atoms,state = self.firstblock(bonds,bond_atom_1,bond_atom_2,atoms,state)
        for block in self.fullblocks:
            bonds,atoms,state = block(bonds,bond_atom_1,bond_atom_2,atoms,state)
        print("end")
        return atoms