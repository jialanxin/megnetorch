from dataset import StructureRamanDataset
from torch.utils.data import DataLoader
import pickle
import json
from model import MegNet
import torch
import time

with open("Structures.pkl","rb") as f:
    structures = pickle.load(f)

with open("Raman_encoded_JVASP_90000.json","r") as f:
    ramans = json.loads(f.read())

def collate_fn(structure_list):
    num_of_structures = len(structure_list)
    atoms_of_all = []
    state_of_all = []
    bonds_of_all = []
    bond_atom_1_of_all = []
    bond_atom_2_of_all = []
    ramans_of_all = []
    for i in range(num_of_structures):
        inputs,ramans = structure_list[i]
        atoms, state, bonds,bond_atom_1,bond_atom_2 = inputs["atoms"],inputs["state"],inputs["bond_length"],inputs["bond_atom_1"],inputs["bond_atom_2"]
        atoms_of_all.append(atoms)
        state_of_all.append(state)
        bonds_of_all.append(bonds)
        bond_atom_1_of_all.append(bond_atom_1)
        bond_atom_2_of_all.append(bond_atom_2)
        ramans_of_all.append(ramans)
    return (torch.LongTensor(atoms_of_all),torch.Tensor(state_of_all),torch.Tensor(bonds_of_all),torch.LongTensor(bond_atom_1_of_all),torch.LongTensor(bond_atom_2_of_all),torch.Tensor(ramans_of_all))


device = torch.device("cuda")
dataset = StructureRamanDataset(structures,ramans)
data_loader = DataLoader(dataset=dataset,batch_size=None,collate_fn=collate_fn,sampler=dataset.data_info.keys())
net = MegNet().to(device)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for i ,data in enumerate(data_loader):
    atoms, state, bonds,bond_atom_1,bond_atom_2,ramans = data
    predicted_spectrum = net(atoms.to(device),state.to(device),bonds.to(device),bond_atom_1.to(device),bond_atom_2.to(device))
    loss = loss_func(predicted_spectrum,ramans.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))