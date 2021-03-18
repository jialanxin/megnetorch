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

class Experiment(pl.LightningModule):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=3177-step=359113.ckpt")
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



if __name__ == "__main__":
    train_set = torch.load("materials/JVASP/Train_raman_set.pt")
    validate_set = torch.load("materials/JVASP/Valid_raman_set.pt")
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=64, num_workers=1)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=64, num_workers=1)
    model = Experiment()
    for i,data in enumerate(validate_dataloader):
        _,raman = data
        predicted_spectrum =  model(data)
        where_zero = torch.eq(raman,torch.zeros_like(raman))
        predict_where_zeros = predicted_spectrum[where_zero]
        hist = torch.histc(predict_where_zeros,bins=3,min=0,max=2)
        hist = hist/predict_where_zeros.shape[0]
        if i == 0:
            m0 = hist.unsqueeze(dim=0)
        else:
            m0 = torch.cat((m0,hist.unsqueeze(dim=0)),dim=0)

        where_zero = torch.eq(raman,torch.ones_like(raman))
        predict_where_zeros = predicted_spectrum[where_zero]
        hist = torch.histc(predict_where_zeros,bins=3,min=0,max=2)
        hist = hist/predict_where_zeros.shape[0]
        if i == 0:
            m1 = hist.unsqueeze(dim=0)
        else:
            m1 = torch.cat((m1,hist.unsqueeze(dim=0)),dim=0)

        where_zero = torch.eq(raman,2*torch.ones_like(raman))
        predict_where_zeros = predicted_spectrum[where_zero]
        hist = torch.histc(predict_where_zeros,bins=3,min=0,max=2)
        hist = hist/predict_where_zeros.shape[0]
        if i == 0:
            m2 = hist.unsqueeze(dim=0)
        else:
            m2 = torch.cat((m1,hist.unsqueeze(dim=0)),dim=0)
    m0 = m0.mean(dim=0,keepdim=True)
    m1 = m1.mean(dim=0,keepdim=True)
    m2 = m2.mean(dim=0,keepdim=True)
    m = torch.cat((m0,m1,m2),dim=0)
    print(m)


# Train:  loss+4simi                                                       
# label\predict:          0,          1,          2,                       
# 0,               9.7021e-01, 2.8971e-02, 7.7503e-04,                     
# 1,               2.7765e-01, 6.3664e-01, 8.1494e-02,                     
# 2,               2.7521e-01, 6.3836e-01, 8.2242e-02                      

# Train:  loss_weight_e
# label\predict:          0,          1,          2,
# 0,                 0.8955,     0.0620,     0.0256
# 1,                 0.2277,     0.5963,     0.1192,
# 2,                 0.2257,     0.5918,     0.1255,

# Validate:
# label\predict:          0,          1,          2,
# 0,                   0.9354,     0.0557,     0.0077,
# 1,                   0.4340,     0.4462,     0.1007,
# 2,                   0.4307,     0.4495,     0.1003

# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.8736, 0.0756, 0.0318
# 1,                   0.3053, 0.4228, 0.1872,
# 2,                   0.2981, 0.4083, 0.2023