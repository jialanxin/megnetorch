from dataset import StructureRamanDataset
from torch.utils.data import DataLoader
import pickle
import json
from model import MegNet

with open("Structures.pkl","rb") as f:
    structures = pickle.load(f)

with open("Raman_encoded_JVASP_90000.json","r") as f:
    ramans = json.loads(f.read())

data_loader = DataLoader(dataset=StructureRamanDataset(structures,ramans),batch_size=1)
net = MegNet()
for i ,data in enumerate(data_loader):
    inputs, ramans = data
    atoms, state, bonds,bond_atom_1,bond_atom_2 = inputs["atoms"],inputs["state"],inputs["bond_length"],inputs["bond_atom_1"],inputs["bond_atom_2"]
    net(atoms,state,bonds,bond_atom_1,bond_atom_2)
    print("end")