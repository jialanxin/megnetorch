from utils.graph import CrystalGraph,range_encode
from torch.utils.data import Dataset
import torch
import json
from pymatgen.core.structure import Structure

class StructureRamanDataset(Dataset):
    @staticmethod
    def get_input(structures,ramans):
        length = len(structures)
        if length != len(ramans):
            raise RuntimeError("Length of Structures and Ramans do not match!")
        couples = []
        for i in range(length):
            structure = structures[i]
            raman = torch.FloatTensor(ramans[i])
            graph = CrystalGraph(structure)
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph,raman))
        return couples
    def __init__(self,structures,ramans) -> None:
        super().__init__()
        self.data_info = self.get_input(structures,ramans)
    def __getitem__(self, index: str):
        return self.data_info[index]
    def __len__(self) -> int:
        return len(self.data_info)

class StructureFmtEnDataset(Dataset):
    @staticmethod
    def get_input(data):
        couples = []
        for item in data:
            structure = Structure.from_dict(item["structure"])
            fmt_en = item["formation_energy_per_atom"]
            try:
                graph = CrystalGraph(structure)
            except RuntimeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            encoded_fmt_en = range_encode(fmt_en,-14,5,41)
            couples.append((encoded_graph,encoded_fmt_en))
        return couples
    def __init__(self,data):
        super().__init__()
        self.data_info = self.get_input(data)
    def __getitem__(self, index: str):
        return self.data_info[index]
    def __len__(self) -> int:
        return len(self.data_info)


def prepare_datesets(json_file):
    with open(json_file, "r") as f:
        data = json.loads(f.read())
    dataset = StructureFmtEnDataset(data)
    return dataset

if __name__=="__main__":
    train_set = prepare_datesets("./materials/mp/Train_data.json")
    validate_set = prepare_datesets("./materials/mp/Valid_data.json")
    torch.save(train_set,"./materials/mp/Train_set.pt")
    torch.save(validate_set,"./materials/mp/Valid_set.pt")
