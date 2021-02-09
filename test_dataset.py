import json
import pickle
from dataset import StructureRamanDataset,StructureFmtEnDataset
import torch
from torch.utils.data import DataLoader
# with open("materials/mp/Valid_data.json","r") as f:
#     data = json.loads(f.read())[:1]
# structure = StructureFmtEnDataset.get_input(data)
# print(structure)

validate_set = torch.load("./materials/mp/Valid_fmten_emd_set.pt")
validate_dataloader = DataLoader(dataset=validate_set, batch_size=1, num_workers=4)

for batch in validate_dataloader:
    encoded_graph, ramans = batch
    atoms = encoded_graph["atoms"]
    positions = encoded_graph["positions"]
    padding_mask = encoded_graph["padding_mask"]
    lattice = encoded_graph["lattice"]
    if torch.isnan(atoms).sum().bool().item():
        raise ValueError("Nan")
    if torch.isnan(positions).sum().bool().item():
        for i in range(150):
            pos = positions[0][i]
            if torch.isnan(pos).sum().bool().item():            
                raise ValueError("Nan")
    if torch.isnan(padding_mask).sum().bool().item():
        raise ValueError("Nan")
    if torch.isnan(lattice).sum().bool().item():
        raise ValueError("Nan")