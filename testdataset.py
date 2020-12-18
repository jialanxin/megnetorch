import json
import pickle
from dataset import StructureRamanDataset

with open("Structures.pkl","rb") as f:
    structures = pickle.load(f)

with open("Raman_encoded_JVASP_90000.json","r") as f:
    ramans = json.loads(f.read())

dataset = StructureRamanDataset(structures,ramans)
print(len(dataset))
count=1
for key in dataset.data_info:
    new_count = len(dataset.data_info[key])
    if new_count>count:
        count = new_count
        print(new_count)
        print(key)