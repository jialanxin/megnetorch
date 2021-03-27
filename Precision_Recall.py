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

class Experiment(Finetune):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=3711-step=211583.ckpt")
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
        spectrum_round = torch.round(predicted_spectrum)
        return spectrum_round

def hist_count(num,raman,predicted_spectrum):
    where_num = torch.eq(raman,num*torch.ones_like(raman))
    predict_where_num = predicted_spectrum[where_num]
    count = predict_where_num.shape[0]
    if count != 0:
        hist = torch.histc(predict_where_num,bins=3,min=0,max=2)
        hist = hist/count
    else:
        hist = torch.zeros((3,))
    return hist
    



if __name__ == "__main__":
    train_set = torch.load("materials/JVASP/Train_raman_set.pt")
    validate_set = torch.load("materials/JVASP/Valid_raman_set.pt")
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=64, num_workers=1)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=64, num_workers=1)
    model = Experiment().eval()
    for i,data in enumerate(validate_dataloader):
        _,raman = data
        predicted_spectrum =  model(data)
        where_zero = torch.eq(raman,torch.zeros_like(raman))
        predict_where_zero = predicted_spectrum[where_zero]
        true_negative = torch.eq(predict_where_zero,torch.zeros_like(predict_where_zero)) #0->0
        false_positive = torch.logical_not(true_negative)                                 #0->1
        true_negative = true_negative.float().sum(dim=0,keepdim=True)
        false_positive = false_positive.float().sum(dim=0,keepdim=True)
        where_one = torch.logical_not(where_zero)
        predict_where_one = predicted_spectrum[where_one]                                 
        false_negative = torch.eq(predict_where_one,torch.zeros_like(predict_where_one))  #1->0
        true_positive =  torch.logical_not(false_negative)                                #1->1
        false_negative = false_negative.float().sum(dim=0,keepdim=True)
        true_positive = true_positive.float().sum(dim=0,keepdim=True)
        Accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
        Precision = true_positive/(true_positive+false_positive)
        Recall = true_positive/(true_positive+false_negative)
        F1Score = 2*Precision*Recall/(Precision+Recall)
        if i == 0:
            Ac_list = Accuracy
            Pr_list = Precision
            Rc_list = Recall
            F1_list = F1Score
        else:
            Ac_list = torch.cat((Ac_list,Accuracy),dim=0)
            Pr_list = torch.cat((Pr_list,Precision),dim=0)
            Rc_list = torch.cat((Rc_list,Recall),dim=0)
            F1_list = torch.cat((F1_list,F1Score),dim=0)
    Accuracy = Ac_list.mean().item()
    Precision = Pr_list.mean().item()
    Recall = Rc_list.mean().item()
    F1Score = F1_list.mean().item()
    print(f"Accuracy:{Accuracy}\nPrcision:{Precision}\nRecall:{Recall}\nF1Score:{F1Score}")

# Train:  loss_weight_6_sign
# Accuracy: 88% Precision: 49% Recall: 90% F1Score: 64
# Validate:
# Accuracy: 85% Precision: 44% Recall: 78% F1Score: 55