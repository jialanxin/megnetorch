from pymatgen.core.structure import  Structure
from dataset import StructureRamanDataset
import numpy as np
from model import MegNet
import torch
import time
from dataloader import collate_fn
from torch.utils.data import DataLoader

structure = Structure.from_file("/home/jlx/new_megnetorch/megnetorch/materials/MoS2_POSCAR")
ramans = [np.zeros((200,))]
data = StructureRamanDataset([structure],ramans)
dataloader = DataLoader(
    dataset=data, batch_size=1, collate_fn=collate_fn, num_workers=1)
device = torch.device("cuda")
net = MegNet(num_of_megnetblock=5).to(device)
check = torch.load("/home/jlx/v0.3.2/4.train_lr_2e-3/checkpoint_epoch_2001_val_loss_0.0403_val_simi_0.6026.pkl")
net.load_state_dict(check["model_state_dict"])

for i, data in enumerate(dataloader):
    atoms, state, bonds, bond_atom_1, bond_atom_2, atoms_mark, bonds_mark, ramans = data
    net.eval()
    predicted_spectrum = net(atoms.to(device), state.to(device), bonds.to(device), bond_atom_1.to(
        device), bond_atom_2.to(device), atoms_mark.to(device), bonds_mark.to(device))
predict = predicted_spectrum.cpu()
x = np.linspace(-1917.34,4354.53,200)
import plotly.graph_objects as go
fig = go.Figure(data=go.Bar(x=x,y=predict[0].detach().numpy()))
fig.write_html("view.html")