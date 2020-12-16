from utils.graph import CrystalGraph
from torch.utils.data import Dataset


class StructureRamanDataset(Dataset):
    @staticmethod
    def get_input(structures,ramans):
        length = len(structures)
        if length != len(ramans):
            raise RuntimeError("Length of Structures and Ramans do not match!")
        couples = []
        for i in range(length):
            structure = structures[i]
            raman = ramans[i]
            graph = CrystalGraph(structure)
            input = graph.convert_to_model_input()
            couples.append((input,raman))
        return couples
    def __init__(self,structures,ramans) -> None:
        super().__init__()
        self.data_info = self.get_input(structures,ramans)
    def __getitem__(self, index: int):
        input,raman = self.data_info[index]
        return input,raman
    def __len__(self) -> int:
        return len(self.data_info)


