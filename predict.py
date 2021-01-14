from pymatgen.core.structure import  Structure
from pymatgen.io.cif import CifParser
from dataset import StructureRamanDataset
import numpy as np
import pytorch_lightning as pl
from model import MegNet
import torch
import time
from dataloader import collate_fn
from torch.utils.data import DataLoader
import streamlit as st
import plotly.graph_objects as go


class Experiment(pl.LightningModule):
    def __init__(self,num_conv=3, optim_type="Adam",lr=1e-3,weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.net = MegNet(num_of_megnetblock=self.hparams.num_conv)
        

    def forward(self, batch):
        atoms, bonds, bond_atom_1, bond_atom_2, _, _, batch_mark_for_atoms, batch_mark_for_bonds, _ = batch
        predicted_spectrum = self.net(
            atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds)
        return predicted_spectrum

    def training_step(self, batch, batch_idx):
        atoms, bonds, bond_atom_1, bond_atom_2, _, _, batch_mark_for_atoms, batch_mark_for_bonds, ramans_of_batch = batch
        predicted_spectrum = self.net(
            atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds)
        loss = F.l1_loss(predicted_spectrum, ramans_of_batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        similarity = F.cosine_similarity(
            predicted_spectrum, ramans_of_batch).mean()
        self.log("train_simi", similarity, on_epoch=True, on_step=False)
        return {"loss": loss, "simi": similarity}
    def validation_step(self, batch, batch_idx):
        atoms, bonds, bond_atom_1, bond_atom_2, _, _, batch_mark_for_atoms, batch_mark_for_bonds, ramans_of_batch = batch
        predicted_spectrum = self.net(
            atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds)
        loss = F.l1_loss(predicted_spectrum, ramans_of_batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        similarity = F.cosine_similarity(
            predicted_spectrum, ramans_of_batch).mean()
        self.log("val_simi", similarity, on_epoch=True, on_step=False)
        return {"loss": loss, "simi": similarity}

    def configure_optimizers(self):
        if self.hparams.optim_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.hparams.weight_decay)
        schedualer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
        return [optimizer],[schedualer]

structure_file = st.file_uploader("Please upload a cif file")
if structure_file != None:
    structure_file = str(structure_file.read(),"utf-8")
    structure_file = CifParser.from_string(structure_file)
    structure = structure_file.get_structures()[0]
    ramans = [np.zeros((41,))]
    data = StructureRamanDataset([structure],ramans)
    dataloader = DataLoader(
        dataset=data, batch_size=1, collate_fn=collate_fn, num_workers=1)
    net = Experiment.load_from_checkpoint("/home/jlx/v0.3.10/2.3_layer_original_megnet/default/version_0/checkpoints/epoch=965-step=109157.ckpt")
    net = net.net
    net.eval()
    for i, data in enumerate(dataloader):
        atoms, bonds, bond_atom_1, bond_atom_2,_,_, atoms_mark, bonds_mark, ramans = data
        predicted_spectrum = net(atoms, bonds, bond_atom_1, bond_atom_2, atoms_mark, bonds_mark)
    predict = predicted_spectrum
    x = np.linspace(0,1200,41)
    fig = go.Figure(data=go.Bar(x=x,y=predict[0].detach().numpy()))
    st.write(fig)