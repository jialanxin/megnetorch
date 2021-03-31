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


class Experiment(Finetune):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=3535-step=201551.ckpt")
        self.atom_embedding = pretrain_model.atom_embedding
        self.atomic_number_embedding = pretrain_model.atomic_number_embedding
        self.mendeleev_number_embedding = pretrain_model.mendeleev_number_embedding
        self.position_embedding = pretrain_model.position_embedding
        self.lattice_embedding = pretrain_model.lattice_embedding
        self.encoder = pretrain_model.encoder
        self.readout = pretrain_model.readout

    def forward(self, batch):
        predicted_spectrum = self.shared_procedure(batch)
        return predicted_spectrum


def load_dataset(Train_or_Valid="Valid"):
    with open(f"materials/JVASP/{Train_or_Valid}_set_25_uneq_yolov1.json", "r") as f:
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
                graph = CrystalEmbedding(structure, max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph, raman, formula))
        return couples


def save_loss_formula(Train_or_Valid="Valid"):
    dataset = load_dataset(Train_or_Valid)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    model = Experiment()
    formula_list = []
    loss_list = []
    raman_list = []
    predict_list = []
    for i, data in enumerate(dataloader):
        graph, ramans, formula = data
        input = (graph, ramans)
        predicted_spectrum = model(input)
        loss = Experiment.yolov1_loss(ramans,predicted_spectrum)
        formula_list.append(formula[0])
        loss_list.append(loss.item())
        raman_list.append(ramans.detach().numpy())
        predict_list.append(predicted_spectrum.detach().numpy())
    torch.save({"loss": loss_list, "formula": formula_list, "raman": raman_list,
                "predict": predict_list}, f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")


def load_loss_formula(Train_or_Valid="Valid"):
    data = torch.load(f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")
    return data["loss"], data["formula"], data["raman"], data["predict"]


def plot_points(Train_or_Valid="Valid"):
    loss, formula, _, _ = load_loss_formula(Train_or_Valid)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=formula, y=loss, mode="markers",
                             marker=dict(size=5 if Train_or_Valid == "Valid" else 2)))
    return fig

def process_y(x,raman,cut_off=0.5):
    edge = np.array([100, 110,  120, 131,  143,  155,  167,  180,  194,  209,  226,  241,
                  259,  277,  297,  322,  347,  375,  406,  440,  478,  528,  587,  685,  844, 1100])
    left = edge[:-1]
    right = edge[1:]
    edge_range = right-left
    confidence = raman[:,0]
    position = raman[:,1]
    absolute_position = position*edge_range+left
    y = np.zeros_like(x)
    for confidence, abs_pos in zip(confidence,absolute_position):
        if confidence <= cut_off:
            continue
        else:
            y += confidence*np.exp(-(x-abs_pos)**2/2/0.5**2)
    return y


def search_formula(formula, Train_or_Valid="Valid",cut_off=0.5):
    loss, formula_list, raman, predict = load_loss_formula(Train_or_Valid)
    index = formula_list.index(formula)
    raman = raman[index][0]
    predict = predict[index][0]
    loss = loss[index]
    fig = go.Figure()
    x = np.linspace(100,1100,10000)
    raman = process_y(x,raman)
    predict = process_y(x,predict,cut_off)
    fig.add_trace(go.Scatter(x=x, y=raman, name="lable"))
    fig.add_trace(go.Scatter(x=x, y=predict, name="predict"))
    return fig, loss


if __name__ == "__main__":
    # save_loss_formula("Train")
    # save_loss_formula("Valid")
    st.header("Loss of Validation Set")
    fig_valid = plot_points("Valid")
    st.write(fig_valid)
    formula_valid = st.text_input("Insert a formula", "Ta4 Si2")
    cut_off_valid = st.slider("ignore confidence under cutoff", min_value=0.0, max_value=1.0, value=0.5, key="Valid")
    fig_valid_search, loss_valid = search_formula(formula_valid, "Valid",cut_off_valid)
    st.write(fig_valid_search)
    st.write(f"loss:{loss_valid}")
    st.header("Loss of Train Set")
    fig_train = plot_points("Train")
    st.write(fig_train)
    formula_train = st.text_input("Insert a formula", "As4")
    cut_off_train = st.slider("ignore confidence under cutoff", min_value=0.0, max_value=1.0, value=0.5, key="Train")
    fig_train_search, loss_train = search_formula(formula_train, "Train",cut_off_train)
    st.write(fig_train_search)
    st.write(f"loss:{loss_train}")
