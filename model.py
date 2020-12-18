from os import stat
import torch.nn
from torch.nn import Embedding
from torch_geometric.nn import Set2Set

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
        # batch_size = bonds.shape[0]
        num_bonds = bonds.shape[1]
        dim2 = bonds.shape[2]
        # for ix_bond in range(num_bonds):
        #     for ix_batch in range(batch_size):
        #         for res in dim2:
        #            atom_1_to_bonds[ix_batch,ix_bond,res] =atoms[ix_batch,bond_atom_1[ix_batch,ix_bond,res],res]
        bond_atom_1 = bond_atom_1.unsqueeze(dim=2).repeat((1,1,dim2))
        bond_atom_2 = bond_atom_2.unsqueeze(dim=2).repeat((1,1,dim2))
        atom_1_to_bonds = torch.gather(input=atoms,dim=1,index=bond_atom_1)
        atom_2_to_bonds = torch.gather(input=atoms,dim=1,index=bond_atom_2)
        bonds = torch.cat((atom_1_to_bonds,atom_2_to_bonds,bonds,state.repeat((1,num_bonds,1))),dim=2)
        bonds = self.phi_e(bonds)

        bonds_to_atoms = torch.zeros_like(atoms)
        num_atoms = atoms.shape[1]
        count_bonds_to_atoms = torch.zeros_like(atoms)
        # for ix_bond in range(num_bonds):
        #     for ix_batch in range(batch_size):
        #         for res in dim2:
        #             bonds_to_atoms[ix_batch,bond_atom_1[ix_batch,ix_bond,res],res] += bonds[ix_batch,ix_bond,res]
        #             count_bonds_to_atoms[ix_batch,bond_atom_1[ix_batch,ix_bond,res],res]+=1
        bonds_to_atoms = bonds_to_atoms.scatter_add(dim=1,index=bond_atom_1,src=bonds)
        count_bonds_to_atoms = count_bonds_to_atoms.scatter_add(dim=1,index=bond_atom_1,src=torch.ones_like(bonds))
        bonds_to_atoms = bonds_to_atoms/count_bonds_to_atoms
        atoms = torch.cat((bonds_to_atoms,atoms,state.repeat((1,num_atoms,1))),dim=2)
        atoms = self.phi_v(atoms)

        bonds_to_state = torch.mean(bonds,dim=1,keepdim=True)
        atoms_to_state = torch.mean(atoms,dim=1,keepdim=True)
        state = torch.cat((bonds_to_state,atoms_to_state,state),dim=2)
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
        self.fullblocks = torch.nn.ModuleList([FullMegnetBlock() for i in range(3)])
        self.set2set_e = Set2Set(in_channels=32,processing_steps=3)
        self.set2set_v = Set2Set(in_channels=32,processing_steps=3)
        self.hidden_layer = torch.nn.Linear(160,128)
        self.output_layer = torch.nn.Linear(128,200)

    def forward(self, atoms,state,bonds,bond_atom_1,bond_atom_2):
        atoms_embedded = self.atomic_embedding(atoms)
        atoms = self.atom_preblock(atoms_embedded)
        bonds = self.bond_preblock(bonds)
        state = self.state_preblock(state)
        bonds,atoms,state = self.firstblock(bonds,bond_atom_1,bond_atom_2,atoms,state)
        for block in self.fullblocks:
            bonds,atoms,state = block(bonds,bond_atom_1,bond_atom_2,atoms,state)
        batch_size = atoms.shape[0]
        num_bonds = bonds.shape[1]
        batch = torch.LongTensor([i for i in range(batch_size)]).unsqueeze(dim=1).repeat((1,num_bonds)).to("cuda")   # device specific
        bonds = self.set2set_e(bonds,batch=batch)  
        atoms = self.set2set_v(atoms,batch=batch)  
        gather_all = torch.cat((bonds,atoms,state),dim=2)
        gather_all = self.hidden_layer(gather_all)
        output_spectrum = torch.squeeze(self.output_layer(gather_all))
        return output_spectrum