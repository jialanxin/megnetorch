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
    for i in range(num_of_structures):
        inputs,ramans = structure_list[i]
        atoms, state, bonds,bond_atom_1,bond_atom_2,num_atoms,num_bonds = inputs["atoms"],inputs["state"],inputs["bonds"],inputs["bond_atom_1"],inputs["bond_atom_2"],inputs["num_atoms"],inputs["num_bonds"]
        if i == 0 :
            atoms_of_batch = atoms #(num_atoms,)
            state_of_batch = state.unsqueeze(dim=0) #(1,2)
            bonds_of_batch = bonds #(num_bonds,bond_info)
            bond_atom_1_of_batch = bond_atom_1 #(num_bonds,)
            bond_atom_2_of_batch = bond_atom_2 #(num_bonds,)
            batch_mark_for_atoms = torch.LongTensor([i for count in range(num_atoms)]) #(num_of_atoms,)
            batch_mark_for_bonds = torch.LongTensor([i for count in range(num_bonds)]) #(num_of_bonds,)
            ramans_of_batch = ramans.unsqueeze(dim=0) #(1,raman_size)
        else:
            atoms_of_batch = torch.cat((atoms_of_batch,atoms),dim=0)  #(sum_of_num_atoms,)
            bonds_of_batch = torch.cat((bonds_of_batch,bonds),dim=0)  #(sum_of_num_bonds,bond_info)
            bond_atom_1 = bond_atom_1+batch_mark_for_atoms.shape[0]
            bond_atom_1_of_batch = torch.cat((bond_atom_1_of_batch,bond_atom_1),dim=0) #(sum_of_num_bonds,)
            bond_atom_2 = bond_atom_2+batch_mark_for_atoms.shape[0]
            bond_atom_2_of_batch = torch.cat((bond_atom_2_of_batch,bond_atom_2),dim=0) #(sum_of_num_bonds,)
            batch_mark_for_atoms = torch.cat((batch_mark_for_atoms,torch.LongTensor([i for count in range(num_atoms)]))) #(sum_of_num_atoms,)
            batch_mark_for_bonds = torch.cat((batch_mark_for_bonds,torch.LongTensor([i for count in range(num_bonds)]))) #(sum_of_num_bonds,)
            ramans_of_batch = torch.cat((ramans_of_batch,ramans.unsqueeze(dim=0)),dim=0) #(batch_size,raman_info)

    return (atoms_of_batch,state_of_batch,bonds_of_batch,bond_atom_1_of_batch,bond_atom_2_of_batch,batch_mark_for_atoms,batch_mark_for_bonds,ramans_of_batch)


device = torch.device("cuda")
dataset = StructureRamanDataset(structures,ramans)
data_loader = DataLoader(dataset=dataset,batch_size=200,collate_fn=collate_fn,num_workers=5,shuffle=True)
net = MegNet()
net.to(device)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
for epoch in range(60):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    accumulate_loss = 0.0
    for i ,data in enumerate(data_loader):
        atoms,state,bonds,bond_atom_1,bond_atom_2,atoms_mark,bonds_mark,ramans = data
        predicted_spectrum = net(atoms.to(device),state.to(device),bonds.to(device),bond_atom_1.to(device),bond_atom_2.to(device),atoms_mark.to(device),bonds_mark.to(device))
        loss = loss_func(predicted_spectrum,ramans.to(device))
        loss.backward()
        accumulate_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    accumulate_loss = accumulate_loss/i
    end.record()
    torch.cuda.synchronize()
    print(f"epoch:{epoch}")
    print(f"Time:{start.elapsed_time(end)}")
    print(f"Loss:{accumulate_loss}")
    # if epoch % 5 == 1:
    #     checkpoint = {"model_state_dict":net.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"epoch":epoch,"loss":accumulate_loss}
    #     torch.save(checkpoint,f"./checkpoint_epoch_{epoch}_loss_{accumulate_loss}.pkl")