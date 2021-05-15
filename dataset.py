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

class StrainStructureRamanDataset(Dataset):
    @staticmethod
    def get_input(data,strain=None):
        couples = []
        for item in data:
            structure = Structure.from_dict(item)
            if strain:
                structure.apply_strain(strain)
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
    @classmethod
    def construct(cls,data,strain=None):
        couples = cls.get_input(data,strain)
        return cls(couples)
    @classmethod
    def joinsubdatasets(cls,dataset_list):
        empty_data_info = []
        for dataset in dataset_list:
            data_info = dataset.data_info
            empty_data_info.extend(data_info)
        return cls(empty_data_info)
    def __init__(self,data) -> None:
        super().__init__()
        self.data_info = data
    def __getitem__(self, index: str):
        return self.data_info[index]
    def __len__(self) -> int:
        return len(self.data_info)

class StructureFmtEnDataset(Dataset):
    @staticmethod
    def convert(item):
        structure = Structure.from_dict(item["structure"])  
        fmt_en = torch.FloatTensor([item["formation_energy_per_atom"]])
        graph = CrystalEmbedding(structure,max_atoms=30)
        encoded_graph = graph.convert_to_model_input()
        return encoded_graph, fmt_en
    @classmethod
    def get_input(cls,data):
        couples = []
        length = len(data)
        for i in range(length):
            item = data.pop()           
            try:
                encoded_graph,fmt_en = cls.convert(item)
            except (ValueError, RuntimeError, TypeError, KeyError):
                continue
            couples.append((encoded_graph,fmt_en))
        return couples
    @classmethod
    def preprocess(cls,data, num_process=1):
        if num_process != 1:
            mapreducer = MultiProcessMapReducer(cls.get_input,data,num_process)
            data_info = mapreducer.run()
        else:
            data_info = cls.get_input(data)
        return cls(data_info)
    def __init__(self,converted_data):
        super().__init__()
        self.data_info = converted_data
    def __getitem__(self, index: str):
        return self.data_info[index]
    def __len__(self) -> int:
        return len(self.data_info)

class StructureXRDDataset(StructureFmtEnDataset):
    @staticmethod
    def convert(item):
        structure = Structure.from_dict(item["structure"])  
        graph = CrystalEmbedding(structure,max_atoms=30)
        encoded_graph = graph.convert_to_model_input()
        xrd_postion, xrd_intensity = graph.get_xrd()
        return encoded_graph,{"PS":xrd_postion,"IT":xrd_intensity}

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
        
class MultiProcessMapReducer:
    def __init__(self,func,data,num_process):
        length = len(data)
        chunk_size = length//num_process
        chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_process-1)]
        last_chunk = last_chunk = [data[(num_process-1)*chunk_size:]]
        chunks.extend(last_chunk)
        self.chunks=chunks
        self.num_process = num_process
        self.func = func
    def run(self):
        with Pool(self.num_process) as pool:
            results_list = pool.map(self.func,self.chunks)
        reduced_results = []
        for i in range(self.num_process):
            results = results_list.pop()
            reduced_results.extend(results)
        return reduced_results

        

class StructureSpaceGroupDataset(StructureFmtEnDataset):
    @staticmethod
    def convert(item):
        structure = Structure.from_dict(item["structure"])  
        graph = CrystalEmbedding(structure,max_atoms=30)
        encoded_graph = graph.convert_to_model_input()
        space_group_number = encoded_graph["SGN"]
        return encoded_graph,space_group_number
    @classmethod
    def from_fmten_set(cls,dataset:StructureFmtEnDataset):
        data = dataset.data_info
        results = []
        for item in data:
            encoded_graph, fmt_en = item
            space_group_number = encoded_graph["SGN"]
            results.append((encoded_graph,space_group_number))
        return cls(results)

class StructureRamanModesDataset(StructureFmtEnDataset):
    @staticmethod
    def convert(item):
        structure = IStructure.from_dict(item["structure"])
        graph = CrystalEmbedding(structure)
        raman_modes = graph.get_raman_modes()
        encoded_graph = graph.convert_to_model_input()    
        return encoded_graph, raman_modes


def prepare_datesets(json_file,pt_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    dataset = StrainStructureRamanDataset.construct(data,strain=-0.1)
    print(len(dataset))
    torch.save(dataset, pt_file)

def convert_fmten_set_to_spsp_set(fmten_pt_file,spgp_pt_file):
    fmten_dataset = torch.load(fmten_pt_file)
    spgp_dataset = StructureSpaceGroupDataset.from_fmten_set(fmten_dataset)
    torch.save(spgp_dataset,spgp_pt_file)

def check_dataset(in_path:Path,out_path:str):
    with in_path.open() as f:
        data_precheck = json.load(f)
    mp_checker = MultiProcessMapReducer(StructureFmtEnStreamDataset.check,data_precheck,3)
    data_postcheck = mp_checker.run()
    with open(out_path,"w") as f:
        json.dump(data_postcheck,f)

def joinstains(pathlist,savepath):
    dataset_list = []
    for path in pathlist:
        dataset = torch.load(path)
        dataset_list.append(dataset)
    new_set = StrainStructureRamanDataset.joinsubdatasets(dataset_list)
    torch.save(new_set,savepath)

if __name__=="__main__":
    # prefix = "gdrive/MyDrive/Raman_machine_learning/OQMD/"
    # check_dataset(Path(prefix+"Valid_set.json"),prefix+"Valid_set_checked.json")
    # convert_fmten_set_to_spsp_set("materials/mp/Valid_fmten_set.pt","materials/mp/Valid_spgp_set.pt")
    # prepare_datesets("materials/JVASP/Train_unique_mp_id.json","materials/JVASP/Train_raman_set_strain_neg0.1.pt")
    joinstains(["materials/JVASP/Train_raman_set_strain_neg0.1.pt","materials/JVASP/Train_raman_set_strain_neg0.05.pt","materials/JVASP/Train_raman_set_strain_0.pt"],"materials/JVASP/Train_raman_set_strain_neg0.1_to_0.pt")
