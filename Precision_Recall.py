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
from dataset import StructureRamanDataset
from finetune import Experiment as Finetune
import numpy as np
import plotly.graph_objects as go

class Experiment(Finetune):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=3665-step=208961.ckpt")
        self.atom_embedding = pretrain_model.atom_embedding
        self.atomic_number_embedding = pretrain_model.atomic_number_embedding
        self.mendeleev_number_embedding = pretrain_model.mendeleev_number_embedding
        self.position_embedding = pretrain_model.position_embedding
        self.space_group_number_embedding = pretrain_model.space_group_number_embedding
        self.lattice_embedding = pretrain_model.lattice_embedding
        self.encoder = pretrain_model.encoder
        self.readout = pretrain_model.readout

    def forward(self, batch):
        predicted_spectrum = self.shared_procedure(batch)
        return predicted_spectrum
    



if __name__ == "__main__":
    # train_set = torch.load("materials/JVASP/Train_raman_set_25_uneq_yolov1.pt")
    validate_set = torch.load("materials/JVASP/Valid_raman_set_25_uneq_yolov1.pt")
    # # train_dataloader = DataLoader(
    # #     dataset=train_set, batch_size=64, num_workers=1)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=len(validate_set), num_workers=1)
    model = Experiment().eval()
    for i,data in enumerate(validate_dataloader):
        _,raman = data
        predicted_spectrum =  model(data)
        predict_confidence = predicted_spectrum[:,:,0]
        target_confidence = raman[:,:,0]
    cut_off_list = np.linspace(0.4,0.8,40)
    Ac_list = np.array([])
    Pr_list = np.array([])
    Rc_list = np.array([])
    F1_list = np.array([])
    for cut_off in cut_off_list:
        less = torch.less_equal(predict_confidence,cut_off)
        predict_confidence[less] = 0
        predict_confidence[torch.logical_not(less)] = 1
        Accuracy, Precision,Recall,F1Score = Experiment.scores(predict_confidence,target_confidence)
        Ac_list = np.append(Ac_list,Accuracy.item())
        Pr_list = np.append(Pr_list,Precision.item())
        Rc_list = np.append(Rc_list,Recall.item())
        F1_list = np.append(F1_list,F1Score.item())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cut_off_list,y=Ac_list,name="Accuracy"))
    fig.add_trace(go.Scatter(x=cut_off_list,y=Pr_list,name="Precision"))
    fig.add_trace(go.Scatter(x=cut_off_list,y=Rc_list,name="Recall"))
    fig.add_trace(go.Scatter(x=cut_off_list,y=F1_list,name="F1Score"))
    fig.show()

# Train:  loss_weight_6_sign
# Accuracy: 88% Precision: 49% Recall: 90% F1Score: 64
# Validate:
# Accuracy: 85% Precision: 44% Recall: 78% F1Score: 55

# nonobj:0.1
# Validate:
# Accuracy: 78% Precision: 49% Recall: 78% F1Score: 60