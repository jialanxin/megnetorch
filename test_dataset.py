import json
import pickle
from dataset import StructureRamanDataset,StructureFmtEnDataset

with open("materials/mp/Valid_data.json","r") as f:
    data = json.loads(f.read())[:1]
structure = StructureFmtEnDataset.get_input(data)
print(structure)

