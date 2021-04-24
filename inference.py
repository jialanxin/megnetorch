from evaluate import Experiment as Evaluated
from evaluate import NMS_or_not
import torch
from pymatgen.core.structure import IStructure
from utils.graph import CrystalEmbedding
from dataset import StructureRamanDataset
from torch.utils.data import DataLoader
from typing import Tuple,Dict
import plotly.graph_objects as go
import streamlit as st


class StructureOnlyDataset(StructureRamanDataset):
    @staticmethod
    def get_input(data):
        couples = []
        for item in data:
            try:
                raman = torch.zeros((25,2))
                graph = CrystalEmbedding(item, max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph, raman))
        return couples

class ToTorchScript(Evaluated):
    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]):
        predicted_spectrum = self.shared_procedure(batch)
        predicted_confidence, predicted_position, _ = NMS_or_not(
            predicted_spectrum)
        x = torch.linspace(100, 1100, 10000)
        y = self.process_y(x,predicted_confidence,predicted_position)
        return x,y
    @staticmethod
    def process_y(x, confidence, position,cut_off:float=0.5):
        y = torch.zeros_like(x)
        for conf, abs_pos in zip(confidence, position):
            if conf <= cut_off:
                continue
            else:
                y += conf*torch.exp(-(x-abs_pos)**2/2/0.5**2)
        return y




def generate_jit_module(filename):
    model = ToTorchScript()
    jit_model = model.to_torchscript()
    torch.jit.save(jit_model,filename)
def load_jit_model(model_file,crystal_file_handle):
    model = torch.jit.load(model_file)
    model.eval()
    cif_str = crystal_file_handle.read().decode("utf-8")
    struct = IStructure.from_str(cif_str,fmt="cif")
    dataset = StructureOnlyDataset([struct])
    dataloader = DataLoader(dataset,batch_size=1)
    input = next(iter(dataloader))
    x,y = model(input)
    x = x.detach().numpy()
    y = y.detach().numpy()
    return x,y

if __name__=="__main__":
    # generate_jit_module("pretrain/jit_module/v0.4.7.pt")
    uploaded_file = st.file_uploader("Upload a .cif file",type="cif")
    x,y=load_jit_model("pretrain/jit_module/v0.4.7.pt",uploaded_file)
    fig=go.Figure(data=go.Scatter(x=x,y=y))
    st.write(fig)
