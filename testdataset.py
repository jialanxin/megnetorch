import json
import pickle
from dataset import StructureRamanDataset

with open("Structures.pkl","rb") as f:
    structures = pickle.load(f)

with open("Raman_encoded_JVASP_90000.json","r") as f:
    ramans = json.loads(f.read())

dataset = StructureRamanDataset(structures,ramans)
print(len(dataset))