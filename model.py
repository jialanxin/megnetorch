from os import stat
import torch.nn
from torch.nn import Embedding
from torch_geometric.nn import Set2Set

def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim,64),torch.nn.SELU(),torch.nn.Linear(64,32),torch.nn.SELU())
def fff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim,64),torch.nn.SELU(),torch.nn.Linear(64,64),torch.nn.SELU(),torch.nn.Linear(64,32),torch.nn.SELU())
def ff_output(input_dim,output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim,128),torch.nn.SELU(),torch.nn.Linear(128,64),torch.nn.SELU(),torch.nn.Linear(64,output_dim),torch.nn.SELU())

class MegNetLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phi_e = fff(128)
        self.phi_v = fff(96)
        self.phi_u = fff(96)
        
    def forward(self,bonds,bond_atom_1,bond_atom_2,atoms,state):
        sum_of_num_bonds = bonds.shape[0]
        atom_info = atoms.shape[1]
        # for ix_bond in range(sum_of_num_bonds):
        #     for res in range(atom_info):
        #        atom_1_to_bonds[ix_bond,res] =atoms[bond_atom_1[ix_bond,res],res]
        bond_atom_1 = bond_atom_1.unsqueeze(dim=1).repeat((1,atom_info))  # (sum_of_num_bonds,atom_info)
        bond_atom_2 = bond_atom_2.unsqueeze(dim=1).repeat((1,atom_info))  # (sum_of_num_bonds,atom_info)
        atom_1_to_bonds = torch.gather(input=atoms,dim=0,index=bond_atom_1)  #(sum_of_num_bonds,atom_info)
        atom_2_to_bonds = torch.gather(input=atoms,dim=0,index=bond_atom_2)  #(sum_of_num_bonds,atom_info)
        bonds = torch.cat((atom_1_to_bonds,atom_2_to_bonds,bonds,state.repeat((sum_of_num_bonds,1))),dim=1) #(sum_of_num_bonds,atom_info*2+bond_info+state_info)
        bonds = self.phi_e(bonds) #(sum_of_num_bonds,bond_info)

        bonds_to_atoms = torch.zeros_like(atoms) #(sum_of_num_atoms,bond_info) here because bond_info and atom_info are both 32
        sum_of_num_atoms = atoms.shape[0]
        count_bonds_to_atoms = torch.zeros_like(atoms) #(sum_of_num_atoms,bond_info)
        # for ix_bond in range(sum_of_num_bonds):
        #     for res in range(bond_info):
        #         bonds_to_atoms[bond_atom_1[ix_bond,res],res] += bonds[ix_bond,res]
        #         count_bonds_to_atoms[bond_atom_1[ix_bond,res],res]+=1
        bonds_to_atoms = bonds_to_atoms.scatter_add(dim=0,index=bond_atom_1,src=bonds)  
        count_bonds_to_atoms = count_bonds_to_atoms.scatter_add(dim=0,index=bond_atom_1,src=torch.ones_like(bonds))
        bonds_to_atoms = bonds_to_atoms/count_bonds_to_atoms #(sum_of_num_atoms,bond_info)
        atoms = torch.cat((bonds_to_atoms,atoms,state.repeat((sum_of_num_atoms,1))),dim=1) #(sum_of_num_atoms,bond_info+atom_info+state_info)
        atoms = self.phi_v(atoms) #(sum_of_num_atoms,atom_info)

        bonds_to_state = torch.mean(bonds,dim=0,keepdim=True) # (1,bond_info)
        atoms_to_state = torch.mean(atoms,dim=0,keepdim=True) # (1,atom_info)
        state = torch.cat((bonds_to_state,atoms_to_state,state),dim=1)  #(1,bond_info+atom_info+state_info)
        state = self.phi_u(state)  #(1,state_info)
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
        self.output_layer = ff_output(input_dim=160,output_dim=200)

    def forward(self, atoms,state,bonds,bond_atom_1,bond_atom_2,batch_mark_for_atoms,batch_mark_for_bonds):
        atoms_embedded = self.atomic_embedding(atoms) #(sum_of_num_atoms,atom_info)
        atoms = self.atom_preblock(atoms_embedded)    #(sum_of_num_atoms,atom_info)
        bonds = self.bond_preblock(bonds)             #(sum_of_num_bonds,bond_info)
        state = self.state_preblock(state)            #(1,state_info)
        bonds,atoms,state = self.firstblock(bonds,bond_atom_1,bond_atom_2,atoms,state)
        for block in self.fullblocks:
            bonds,atoms,state = block(bonds,bond_atom_1,bond_atom_2,atoms,state)
        batch_size = batch_mark_for_bonds.max()+1
        bonds = self.set2set_e(bonds,batch=batch_mark_for_bonds)  # (batch_size,bond_info)
        atoms = self.set2set_v(atoms,batch=batch_mark_for_atoms)  # (batch_size,atom_info)
        gather_all = torch.cat((bonds,atoms,state.repeat(batch_size,1)),dim=1) #(batch_size, bond_info+atom_info+state_info)
        output_spectrum = self.output_layer(gather_all) #(batch_size, raman_info)
        return output_spectrum