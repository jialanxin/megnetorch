import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from typing import Dict,Tuple
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

    def forward(self, batch:Tuple[Dict[str,torch.Tensor],torch.Tensor]):
        predicted_spectrum = self.shared_procedure(batch)
        return predicted_spectrum


def load_dataset(Train_or_Valid="Valid"):
    with open(f"materials/JVASP/{Train_or_Valid}_unique_mp_id.json", "r") as f:
        data = json.loads(f.read())
    dataset = RamanFormularDataset(data)
    print(len(dataset))
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


def NMS_or_not(predicted, nms:bool=False, cut_off:float=0.5):
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
    total = target_confidence.float().sum().detach().item()
    return int(more),int(less),int(incorrect),int(total), target_confidence, target_position


def absolute_position(relative_postition):
    edge = torch.tensor([100, 110,  120, 131,  143,  155,  167,  180,  194,  209,  226,  241,
                              259,  277,  297,  322,  347,  375,  406,  440,  478,  528,  587,  685,  844, 1100]).to(torch.float32)
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
    props = []
    for i, data in enumerate(dataloader):
        graph, ramans, formula, mp_id = data
        spacegroup = graph["SGN"].item()
        input = (graph, ramans)
        predicted_spectrum = model(input)
        predicted_confidence, predicted_position, predicted_confidence_round = NMS_or_not(
            predicted_spectrum)
        more,less,incorrect,total, target_confidence, target_position = count_incorrects(
            ramans, predicted_confidence_round)
        prop = {"FML":formula[0],"MR":more,"LS":less,"TTL":total,"ICT":incorrect,"TGT_CF":target_confidence,"TGT_PS":target_position,"PDT_CF":predicted_confidence,"PDT_PS":predicted_position,"MP":mp_id[0],"SG":spacegroup}
        props.append(prop)
    torch.save(props, f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")


def plot_points(Train_or_Valid="Valid"):
    data = torch.load(f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")
    formula = []
    more = []
    less = []
    incorrect = []
    for item in data:
        formula.append(item["FML"])
        more.append(item["MR"])
        less.append(item["LS"])
        incorrect.append(item["ICT"])
    fig = make_subplots(rows=3,cols=2,subplot_titles=("Number of modes more than target","Number of modes less than target","more hist","less hist","incorrect hist"))
    fig.add_trace(go.Scatter(x=formula, y=more, mode="markers",
                             marker=dict(size=5 if Train_or_Valid == "Valid" else 2)),col=1,row=1)
    count, value = np.histogram(more,bins=14)
    fig.add_trace(go.Scatter(x=value,y=count),col=1,row=2)

    fig.add_trace(go.Scatter(x=formula, y=less, mode="markers",
                             marker=dict(size=5 if Train_or_Valid == "Valid" else 2)),col=2,row=1)
    count, value = np.histogram(less,bins=13)
    fig.add_trace(go.Scatter(x=value,y=count),col=2,row=2)
    count,value = np.histogram(incorrect,bins=16)
    fig.add_trace(go.Scatter(x=value,y=count),col=1,row=3)
    fig.update_layout(showlegend=False)
    return fig

def plot_incorrect_dist(Train_or_Valid="Valid",spacegroup_or_total="SG"):
    data = torch.load(f"materials/JVASP/{Train_or_Valid}_loss_formula_yolo.pt")
    if spacegroup_or_total == "TTL":
        total = np.linspace(0,25,26)
        x = total
    else:
        space_groups = np.linspace(1,230,230)
        x = space_groups
    
    incorrect_0 = np.zeros_like(x)
    incorrect_1 = np.zeros_like(x)
    incorrect_2 = np.zeros_like(x)
    incorrect_3 = np.zeros_like(x)
    incorrect_4 = np.zeros_like(x)
    incorrect_more = np.zeros_like(x)
    for item in data:
        if spacegroup_or_total == "TTL":
            key = item["TTL"]
        else:
            key = item["SG"]-1
        if item["ICT"] == 0:
            incorrect_0[key] += 1
        elif item["ICT"] == 1:
            incorrect_1[key] += 1
        elif item["ICT"] == 2:
            incorrect_2[key] += 1
        elif item["ICT"] == 3:
            incorrect_3[key] += 1
        elif item["ICT"] == 4:
            incorrect_4[key] += 1
        else:
            incorrect_more[item[spacegroup_or_total]-1] += 1
    fig = go.Figure(data=[go.Bar(x=x,y=incorrect_0,name="incorrect=0"),go.Bar(x=x,y=incorrect_1,name="incorrect=1"),go.Bar(x=x,y=incorrect_2,name="incorrect=2"),go.Bar(x=x,y=incorrect_3,name="incorrect=3"),go.Bar(x=x,y=incorrect_4,name="incorrect=4"),go.Bar(x=x,y=incorrect_more,name="incorrect>=5")])
    fig.update_layout(barmode='stack')
    return fig


def process_y(x, confidence, position,cut_off=0.5):
    y = torch.zeros_like(x)
    for conf, abs_pos in zip(confidence, position):
        if conf <= cut_off:
            continue
        else:
            y += conf*torch.exp(-(x-abs_pos)**2/2/0.5**2)
    return y.detach().numpy()


def search_formula(formula, cut_off=0.5):
    data = torch.load(f"materials/JVASP/Valid_loss_formula_yolo.pt")
    for item in data:
        if formula == item["FML"]:
            target_confidence = item["TGT_CF"]
            target_position = item["TGT_PS"]
            predict_confidence = item["PDT_CF"]
            predict_position = item["PDT_PS"]
            more = item["MR"]
            less = item["LS"]
            mp_id = item["MP"]
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
    st.write(plot_points("Valid"))
    st.subheader("incorrect-total modes")
    st.write(plot_incorrect_dist("Valid","TTL"))
    st.subheader("incorrect-space group")
    st.write(plot_incorrect_dist("Valid","SG"))
    st.subheader("incorrect-space group (training set)")
    st.write(plot_incorrect_dist("Train","SG"))
    formula_valid = st.text_input("Insert a formula", "Ta4 Si2")
    cut_off_valid = st.slider("ignore confidence under cutoff", min_value=0.0, max_value=1.0, value=0.5, key="Valid")
    fig_valid_search,more_valid,less_valid,mp_id_valid = search_formula(formula_valid,cut_off_valid)
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
