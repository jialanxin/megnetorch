from utils.graph import CrystalGraph,range_encode,CrystalEmbedding
from torch.utils.data import Dataset
import torch
import json
from pymatgen.core.structure import Structure,IStructure

class StructureRamanDatasetOld(Dataset):
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

class StructureRamanDataset(Dataset):
    @staticmethod
    def get_input(data):
        couples = []
        for item in data:
            structure = IStructure.from_dict(item)
            try:
                raman = torch.FloatTensor(item["raman"])
                graph = CrystalEmbedding(structure,max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph,raman))
        return couples
    def __init__(self,data) -> None:
        super().__init__()
        self.data_info = self.get_input(data)
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
            try:
                fmt_en = torch.FloatTensor([item["formation_energy_per_atom"]])
                graph = CrystalEmbedding(structure,max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph,fmt_en))
        return couples
    def __init__(self,data):
        super().__init__()
        self.data_info = self.get_input(data)
    def __getitem__(self, index: str):
        return self.data_info[index]
    def __len__(self) -> int:
        return len(self.data_info)

class StructureSpaceGroupDataset(StructureFmtEnDataset):
    @staticmethod
    def get_input(data):
        couples = []
        for item in data:
            structure = Structure.from_dict(item["structure"])
            try:
                fmt_en = torch.FloatTensor([item["formation_energy_per_atom"]])
                graph = CrystalEmbedding(structure,max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            space_group_number = encoded_graph["SGN"]
            couples.append((encoded_graph,space_group_number))
        return couples

class StructureRamanModesDataset(Dataset):
    @staticmethod
    def get_input(data):
        couples = []
        for item in data:
            structure = Structure.from_dict(item["structure"])
            try:
                graph = CrystalGraph(structure)
                raman_modes = torch.FloatTensor([graph.get_raman_mode_numbers()])
            except RuntimeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph,raman_modes))
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
    dataset = StructureRamanDataset(data)
    return dataset

if __name__=="__main__":
    train_set = prepare_datesets("materials/JVASP/Train_set_100.json")
    print(len(train_set))
    torch.save(train_set,"materials/JVASP/Train_raman_set_100.pt")

    validate_set = prepare_datesets("materials/JVASP/Valid_set_100.json")
    print(len(validate_set))
    torch.save(validate_set,"materials/JVASP/Valid_raman_set_100.pt")
