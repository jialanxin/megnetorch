import json
import argparse
import torch
import torch.nn.functional as F
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.nn import Embedding, RReLU, ReLU, Dropout
from dataset import StructureRamanDataset, IStructure, CrystalEmbedding
from finetune import Experiment as Finetune
import plotly.graph_objects as go
import numpy as np
import streamlit as st

class Experiment(pl.LightningModule):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=3981-step=226973.ckpt")
        self.atom_embedding = pretrain_model.atom_embedding
        self.position_embedding = pretrain_model.position_embedding
        self.lattice_embedding = pretrain_model.lattice_embedding
        self.encoder = pretrain_model.encoder
        self.readout = pretrain_model.readout

    @staticmethod
    def Gassian_expand(value_list, min_value, max_value, intervals, expand_width, device):
        value_list = value_list.expand(-1, -1, intervals)
        centers = torch.linspace(min_value, max_value, intervals).to(device)
        result = torch.exp(-(value_list - centers)**2/expand_width**2)
        return result

    def shared_procedure(self, batch):
        encoded_graph, _ = batch
        # atoms: (batch_size,max_atoms,31)
        atoms = encoded_graph["atoms"]
        # padding_mask: (batch_size, max_atoms)
        padding_mask = encoded_graph["padding_mask"]
        # lattice: (batch_size, 9, 1)
        lattice = encoded_graph["lattice"]
        # (batch_size, max_atoms, 1)
        elecneg = encoded_graph["elecneg"]
        # (batch_size, max_atoms, 1)
        covrad = encoded_graph["covrad"]
        # (batch_size, max_atoms, 1)
        FIE = encoded_graph["FIE"]
        # (batch_size, max_atoms, 1)
        elecaffi = encoded_graph["elecaffi"]
        # (batch_size, max_atoms, 1)
        atmwht = encoded_graph["AM"]
        # (batch_size, max_atoms, 3)
        positions = encoded_graph["positions"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (batch_size, max_atoms, 20)
        elecneg = self.Gassian_expand(elecneg, 0.5, 4.0, 20, 0.18, device)
        # (batch_size, max_atoms, 20)
        covrad = self.Gassian_expand(covrad, 50, 250, 20, 10, device)
        # (batch_size, max_atoms, 20)
        FIE = self.Gassian_expand(FIE, 3, 25, 20, 1.15, device)
        # (batch_size, max_atoms, 20)
        elecaffi = self.Gassian_expand(elecaffi, -3, 3.7, 20, 0.34, device)
        # (batch_size, max_atoms, 20)
        atmwht = self.Gassian_expand(atmwht, 0, 210, 20, 10.5, device)
        # (batch_size, max_atoms, 111)
        atoms = torch.cat(
            (atoms, elecneg, covrad, FIE, elecaffi, atmwht), dim=2)

        positions = positions.unsqueeze(dim=3).expand(-1, -1, 3, 20)
        centers = torch.linspace(-15, 18, 20).to(device)
        # (batch_size, max_atoms, 3, 20)
        positions = torch.exp(-(positions - centers)**2/1.65**2)
        # (batch_size, max_atoms, 60)
        positions = torch.flatten(positions, start_dim=2)

        atoms = self.atom_embedding(atoms)  # (batch_size,max_atoms,atoms_info)
        # (batch_size,max_atoms,positions_info)
        positions = self.position_embedding(positions)
        atoms = atoms+positions  # (batch_size,max_atoms,atoms_info)

        lattice = self.Gassian_expand(
            lattice, -15, 18, 20, 1.65, device)  # (batch_size, 9, 20)
        lattice = torch.flatten(lattice, start_dim=1)  # (batch_size,180)
        lattice = self.lattice_embedding(lattice)  # (batch_size,lacttice_info)
        # (batch_size,1,lacttice_info)
        lattice = torch.unsqueeze(lattice, dim=1)
        # (batch_size,1+max_atoms,atoms_info)
        atoms = torch.cat((lattice, atoms), dim=1)
        # (1+max_atoms, batch_size, atoms_info)
        atoms = torch.transpose(atoms, dim0=0, dim1=1)
        batch_size = padding_mask.shape[0]
        cls_padding = torch.zeros((batch_size, 1)).bool().to(
            device)  # (batch_size, 1)

        # (batch_size, 1+max_atoms)
        padding_mask = torch.cat((cls_padding, padding_mask), dim=1)

        # (1+max_atoms, batch_size, atoms_info)
        atoms = self.encoder(src=atoms, src_key_padding_mask=padding_mask)

        system_out = atoms[0]  # (batch_size,atoms_info)

        output_spectrum = self.readout(system_out)  # (batch_size, raman_info)
        output_spectrum = torch.exp(output_spectrum)

        return output_spectrum

    def forward(self, batch):
        predicted_spectrum = self.shared_procedure(batch)
        spectrum_round = torch.round(predicted_spectrum)
        return spectrum_round

def load_dataset():
    with open("materials/JVASP/Valid_CrystalRamans.json","r") as f:
        data = json.loads(f.read())
    dataset = RamanFormularDataset(data)
    return dataset

class RamanFormularDataset(StructureRamanDataset):
    @staticmethod
    def get_input(data):
        couples = []
        for item in data:
            structure = IStructure.from_dict(item)
            try:
                raman = torch.FloatTensor(item["raman"])
                formula = structure.formula
                graph = CrystalEmbedding(structure,max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph,raman,formula))
        return couples

def save_loss_formula():
    dataset = load_dataset()
    dataloader = DataLoader(dataset=dataset,batch_size=1)
    model = Experiment()
    formula_list = []
    loss_list = []
    raman_list = []
    predict_list = []
    for i, data in enumerate(dataloader):
        graph, ramans, formula = data
        input = (graph,ramans)
        predicted_spectrum = model(input)
        loss = F.l1_loss(predicted_spectrum, ramans)
        formula_list.append(formula[0])
        loss_list.append(loss.item())
        raman_list.append(ramans.detach().numpy())
        predict_list.append(predicted_spectrum.detach().numpy())
    torch.save({"loss":loss_list,"formula":formula_list,"raman":raman_list,"predict":predict_list},"materials/JVASP/loss_formula.pt")
def load_loss_formula():
    data = torch.load("materials/JVASP/loss_formula.pt")
    return data["loss"],data["formula"],data["raman"],data["predict"]

if __name__ == "__main__":
    loss,formula,ramams,predicts = load_loss_formula()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=formula,y=loss,mode="markers"))
    st.write(fig)
    word = st.text_input("Insert a formula", "Te4 O8")
    index = formula.index(word)
    raman = ramams[index].flatten()
    predict = predicts[index].flatten()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=np.linspace(100,1156,25),y=raman,name="lable"))
    fig2.add_trace(go.Bar(x=np.linspace(100,1156,25),y=predict,name="predict"))
    st.write(fig2)
    st.write(f"loss:{loss[index]}")
