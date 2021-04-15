import json
from typing import overload

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from dataset import CrystalEmbedding, IStructure, StructureRamanDataset
from finetune import Experiment as Finetune


class Experiment(Finetune):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=1499-step=74999.ckpt")
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
    with open(f"materials/JVASP/{Train_or_Valid}_unique_mp_id.json", "r") as f:
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
                mp_id = item["mp_id"]
                formula = structure.formula
                graph = CrystalEmbedding(structure, max_atoms=30)
            except ValueError:
                continue
            except RuntimeError:
                continue
            except TypeError:
                continue
            encoded_graph = graph.convert_to_model_input()
            couples.append((encoded_graph, raman, formula, mp_id))
        return couples


def NMS_or_not(predicted, nms=False, cut_off=0.5):
    predicted = predicted[0]
    predicted_confidence = predicted[:, :, 0]  # (25,2)
    predicted_position = predicted[:, :, 1]   # (25,2)
    predicted_position = absolute_position(predicted_position)  # (25,2)
    predicted_confidence, selected_worker = torch.max(
        predicted_confidence, dim=-1)  # (25,)
    # predict_position[i,j=0] = predicted_postion[i, selected_worker[i,j=0]]
    selected_worker = selected_worker.view((-1, 1))
    predicted_position = torch.gather(
        predicted_position, 1, selected_worker).flatten()
    less = torch.less_equal(predicted_confidence, cut_off)
    more = torch.logical_not(less)
    predicted_confidence_round = predicted_confidence.clone()
    predicted_confidence_round[less] = 0
    predicted_confidence_round[more] = 1
    return predicted_confidence, predicted_position, predicted_confidence_round


def count_incorrects(ramans, predict_confidence_round):
    ramans = ramans[0]
    target_confidence = ramans[:, 0]
    target_position = ramans[:, 1]
    target_position = absolute_position(target_position).flatten()
    more = torch.greater(predict_confidence_round,target_confidence).float().sum().detach().item()
    less = torch.less(predict_confidence_round,target_confidence).float().sum().detach().item()
    incorrect = more+less
    return more,less,incorrect, target_confidence, target_position


def absolute_position(relative_postition):
    edge = torch.FloatTensor([100, 110,  120, 131,  143,  155,  167,  180,  194,  209,  226,  241,
                              259,  277,  297,  322,  347,  375,  406,  440,  478,  528,  587,  685,  844, 1100])
    left = edge[:-1]
    right = edge[1:]
    edge_range = right-left
    left = left.view((-1, 1))
    edge_range = edge_range.view((-1, 1))
    if relative_postition.ndim == 1:
        relative_postition = relative_postition.view((-1,1))
    abs_pos = left+edge_range*relative_postition
    return abs_pos


def save_loss_formula(Train_or_Valid="Valid"):
    dataset = load_dataset(Train_or_Valid)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    model = Experiment()
    formula_list = []
    more_list = []
    less_list = []
    incorrect_list = []
    target_confidence_list = []
    target_position_list = []
    predicted_confidence_list = []
    predicted_position_list = []
    mp_id_list = []
    for i, data in enumerate(dataloader):
        graph, ramans, formula, mp_id = data
        input = (graph, ramans)
        predicted_spectrum = model(input)
        predicted_confidence, predicted_position, predicted_confidence_round = NMS_or_not(
            predicted_spectrum)
        more,less,incorrect, target_confidence, target_position = count_incorrects(
            ramans, predicted_confidence_round)
        formula_list.append(formula[0])
        more_list.append(more)
        less_list.append(less)
        incorrect_list.append(incorrect)
        target_confidence_list.append(target_confidence)
        target_position_list.append(target_position)
        predicted_confidence_list.append(predicted_confidence)
        predicted_position_list.append(predicted_position)
        mp_id_list.append(mp_id)
    torch.save({"formula": formula_list, "more":more_list,"less":less_list,"incorrect":incorrect_list, "target_confidence": target_confidence_list, "target_position": target_position_list,
                "predicted_confidence": predicted_confidence_list, "predicted_position": predicted_position_list, "mp_id": mp_id_list}, f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")


def load_loss_formula(Train_or_Valid="Valid"):
    data = torch.load(f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")
    return  data["formula"], data["more"],data["less"],data["incorrect"], data["target_confidence"], data["target_position"], data["predicted_confidence"],data["predicted_position"],data["mp_id"]


def plot_points(Train_or_Valid="Valid"):
    formula,more,less,incorrect, _, _,_,_,_ = load_loss_formula(Train_or_Valid)
    fig = make_subplots(rows=3,cols=2,subplot_titles=("Number of modes more than target","Number of modes less than target","more hist","less hist","incorrect hist"))
    fig.add_trace(go.Scatter(x=formula, y=more, mode="markers",
                             marker=dict(size=5 if Train_or_Valid == "Valid" else 2)),col=1,row=1)
    count, value = np.histogram(more,bins=12)
    fig.add_trace(go.Scatter(x=value,y=count),col=1,row=2)

    fig.add_trace(go.Scatter(x=formula, y=less, mode="markers",
                             marker=dict(size=5 if Train_or_Valid == "Valid" else 2)),col=2,row=1)
    count, value = np.histogram(less,bins=10)
    fig.add_trace(go.Scatter(x=value,y=count),col=2,row=2)
    count,value = np.histogram(incorrect,bins=15)
    fig.add_trace(go.Scatter(x=value,y=count),col=1,row=3)
    fig.update_layout(showlegend=False)
    return fig


def process_y(x, confidence, position,cut_off=0.5):
    y = torch.zeros_like(x)
    for conf, abs_pos in zip(confidence, position):
        if conf <= cut_off:
            continue
        else:
            y += conf*torch.exp(-(x-abs_pos)**2/2/0.5**2)
    return y.detach().numpy()


def search_formula(formula, Train_or_Valid="Valid", cut_off=0.5):
    formula_list, more,less,_,target_confidence,target_position, predict_confidence,predict_position,mp_id = load_loss_formula(Train_or_Valid)
    index = formula_list.index(formula)
    target_confidence = target_confidence[index]
    target_position = target_position[index]
    predict_confidence = predict_confidence[index]
    predict_position = predict_position[index]
    more = more[index]
    less = less[index]
    mp_id = mp_id[index][0]
    fig = go.Figure()
    x = torch.linspace(100, 1100, 10000)
    raman = process_y(x, target_confidence, target_position)
    predict = process_y(x, predict_confidence,predict_position, cut_off)
    fig.add_trace(go.Scatter(x=x, y=raman, name="lable"))
    fig.add_trace(go.Scatter(x=x, y=predict, name="predict"))
    return fig, more,less, mp_id


if __name__ == "__main__":
    # save_loss_formula("Train")
    # save_loss_formula("Valid")
    st.header("Incorrectness of Validation Set")
    fig_valid = plot_points("Valid")
    st.write(fig_valid)
    formula_valid = st.text_input("Insert a formula", "Ta4 Si2")
    cut_off_valid = st.slider("ignore confidence under cutoff", min_value=0.0, max_value=1.0, value=0.5, key="Valid")
    fig_valid_search,more_valid,less_valid,mp_id_valid = search_formula(formula_valid, "Valid",cut_off_valid)
    st.write(fig_valid_search)
    st.write(f"more:{more_valid}, less:{less_valid}")
    st.write(f"mp_link: https://www.materialsproject.org/materials/mp-{mp_id_valid}")
    # st.header("Loss of Train Set")
    # fig_train = plot_points("Train")
    # st.write(fig_train)
    # formula_train = st.text_input("Insert a formula", "As4")
    # cut_off_train = st.slider("ignore confidence under cutoff", min_value=0.0, max_value=1.0, value=0.5, key="Train")
    # fig_train_search, loss_train = search_formula(formula_train, "Train",cut_off_train)
    # st.write(fig_train_search)
    # st.write(f"loss:{loss_train}")
