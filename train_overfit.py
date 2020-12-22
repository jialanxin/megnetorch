from dataset import StructureRamanDataset
from torch.utils.data import DataLoader, random_split
import pickle
import json
from model import MegNet
import torch
import time
from dataloader import collate_fn


with open("Structures.pkl","rb") as f:
    structures = pickle.load(f)

with open("Raman_encoded_JVASP_90000.json","r") as f:
    ramans = json.loads(f.read())



device = torch.device("cuda")
dataset = StructureRamanDataset(structures,ramans)
dataset_length = len(dataset)
_, dataset = random_split(dataset,[int(0.9*dataset_length)+1,int(0.1*dataset_length)])
data_loader = DataLoader(dataset=dataset,batch_size=4,collate_fn=collate_fn,num_workers=4,shuffle=True)
net = MegNet()
net.to(device)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
schedualer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=64)

prefix = "/home/jlx/v0.2.1/2.train_overfit_SELU/"
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(prefix+"runs")

for epoch in range(0,2002):
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
    writer.add_scalar("Loss vs Epoch",accumulate_loss,epoch)
    schedualer.step()
    if epoch % 100 == 1:
        checkpoint = {"model_state_dict":net.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"schedual_state_dict":schedualer.state_dict(),"epoch":epoch,"loss":accumulate_loss}
        torch.save(checkpoint,prefix+f"checkpoint_epoch_{epoch}_loss_{accumulate_loss}.pkl")
writer.close()