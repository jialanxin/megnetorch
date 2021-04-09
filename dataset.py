from utils.graph import CrystalGraph,range_encode,CrystalEmbedding
from torch.utils.data import Dataset
import torch
import json
from pymatgen.core.structure import Structure,IStructure
from multiprocessing import Pool
from pathlib import Path

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
        length = len(data)
        for i in range(length):
            item = data.pop()
            structure = Structure.from_dict(item["structure"])
            try:
                fmt_en = torch.FloatTensor([item["formation_energy_per_atom"]])
                graph = CrystalEmbedding(structure,max_atoms=30)
                encoded_graph = graph.convert_to_model_input()
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            couples.append((encoded_graph,fmt_en))
        return couples
    @staticmethod
    def split_chunks(data,num_process):
        length = len(data)
        chunk_size = length//num_process
        chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_process-1)]
        last_chunk = [data[(num_process-1)*chunk_size:]]
        chunks.extend(last_chunk)
        return chunks
    def __init__(self,data, num_process=1):
        super().__init__()
        if num_process != 1:
            chunks = self.split_chunks(data,num_process)
            with Pool(num_process) as p:
                results_list = p.map(self.get_input, chunks)
            reduced_result = []
            for i in range(num_process):
                result = results_list.pop()
                reduced_result.extend(result)
            self.data_info = reduced_result
        else:
            self.data_info = self.get_input(data)
        
    def __getitem__(self, index: str):
        return self.data_info[index]
    def __len__(self) -> int:
        return len(self.data_info)

class StructureFmtEnStreamDataset(Dataset):
    def __init__(self,json_file_path:Path, post_check=True):
        if isinstance(json_file_path, Path):
            with json_file_path.open() as f:
                if post_check:
                    self.data = json.load(f)
                else:
                    data_precheck = json.load(f)
                    self.data = self.check(data_precheck)
        else:
            raise TypeError("json_file_path only accept Path type")
    @staticmethod
    def check(data_precheck):
        post_check_data = []
        length = len(data_precheck)
        for i in range(length):
            item = data_precheck.pop()
            try:
                encoded_graph,fmt_en = StructureFmtEnStreamDataset.convert(item)
                post_check_data.append(item)
            except (ValueError,RuntimeError,TypeError):
                continue
        print(f"Length of post-check data is {len(post_check_data)}")
        return post_check_data
    @staticmethod
    def convert(item):
        structure = Structure.from_dict(item["structure"])  
        fmt_en = torch.FloatTensor([item["formation_energy_per_atom"]])
        graph = CrystalEmbedding(structure,max_atoms=30)
        encoded_graph = graph.convert_to_model_input()
        return encoded_graph, fmt_en
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        encoded_graph,fmt_en = self.convert(self.data[index])
        return (encoded_graph,fmt_en)
        
class MultiProcessChecker:
    def __init__(self,checker,data,num_process):
        length = len(data)
        chunk_size = length//num_process
        chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_process-1)]
        last_chunk = last_chunk = [data[(num_process-1)*chunk_size:]]
        chunks.extend(last_chunk)
        self.chunks=chunks
        self.num_process = num_process
        self.checker = checker
    def check(self):
        with Pool(self.num_process) as pool:
            results_list = pool.map(self.checker,self.chunks)
        reduced_results = []
        for i in range(self.num_process):
            results = results_list.pop()
            reduced_results.extend(results)
        return reduced_results

        

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
    dataset = StructureFmtEnDataset(data,num_process=4)
    return dataset

def check_dataset(in_path:Path,out_path:str):
    with in_path.open() as f:
        data_precheck = json.load(f)
    mp_checker = MultiProcessChecker(StructureFmtEnStreamDataset.check,data_precheck,3)
    data_postcheck = mp_checker.check()
    with open(out_path,"w") as f:
        json.dump(data_postcheck,f)

if __name__=="__main__":
    prefix = "gdrive/MyDrive/Raman_machine_learning/OQMD/"
    check_dataset(Path(prefix+"Valid_set.json"),prefix+"Valid_set_checked.json")
    # train_set = prepare_datesets("materials/OQMD/Train_set.json")
    # print(len(train_set))
    # torch.save(train_set,"materials/OQMD/Train_fmten_set.pt")

    # validate_set = prepare_datesets("materials/OQMD/Valid_set.json")
    # print(len(validate_set))
    # torch.save(validate_set,"materials/OQMD/Valid_fmten_set.pt")
