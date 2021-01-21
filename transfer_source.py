from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from dataset import StructureFmtEnDataset
from torch.utils.data import DataLoader, random_split
import pickle
import json
from model import MegNet
import torch
import time
from dataloader import collate_fn
import yaml
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description="Select a train_config.yaml file")
parser.add_argument(dest="filename", metavar="/path/to/file")
arg = parser.parse_args()
path_to_file = arg.filename
with open(path_to_file, "r") as f:
    config = yaml.load(f.read(), Loader=yaml.BaseLoader)
prefix = config["prefix"]


train_set = torch.load("./materials/mp/Train_set.pt")
validate_set = torch.load("./materials/mp/Valid_set.pt")

train_dataloader = DataLoader(
        dataset=train_set, batch_size=64, collate_fn=collate_fn, num_workers=4)
validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=64, collate_fn=collate_fn, num_workers=4)

from model import ff,fff,FullMegnetBlock,Set2Set,ff_output
class Experiment(pl.LightningModule):
    def __init__(self, num_conv=3, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.atom_preblock = ff(71)
        self.bond_preblock = ff(100)
        # self.firstblock = FirstMegnetBlock()
        self.fullblocks = torch.nn.ModuleList(
            [FullMegnetBlock() for i in range(num_conv)])
        # self.fullblocks = torch.nn.ModuleList(
        # [EncoderBlock() for i in range(num_of_megnetblock)])
        self.set2set_v = Set2Set(in_channels=32, processing_steps=3)
        self.set2set_e = Set2Set(in_channels=32, processing_steps=3)
        self.output_layer = ff_output(input_dim=128, output_dim=1)
    def shared_procedure(self,atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds):
        # (sum_of_num_atoms,atom_info)
        atoms = self.atom_preblock(atoms)
        # print(f"Atoms:{atoms.shape}")
        bonds = self.bond_preblock(bonds)  # (sum_of_num_bonds,bond_info)
        # print(f"Bonds:{bonds.shape}")
        # atoms, bonds = self.firstblock(
            # bonds, bond_atom_1, bond_atom_2, atoms,batch_mark_for_atoms,batch_mark_for_bonds)
        for block in self.fullblocks:
            atoms, bonds = block(
                bonds, bond_atom_1, bond_atom_2, atoms,batch_mark_for_atoms,batch_mark_for_bonds)
        # print(f"Atoms:{atoms.shape}")
        # print(f"Bonds:{bonds.shape}")
        batch_size = batch_mark_for_bonds.max()+1
        # print(batch_size)
        # (batch_size,bond_info)
        bonds = self.set2set_e(bonds, batch=batch_mark_for_bonds)
        atoms = self.set2set_v(atoms, batch=batch_mark_for_atoms)
        # print(f"Atoms:{atoms.shape}")
        # print(f"Bonds:{bonds.shape}")
        # (batch_size, bond_info+atom_info)
        gather_all = torch.cat((bonds, atoms), dim=1)
        # print(f"Shape:{gather_all.shape}")
        output_spectrum = self.output_layer(
            gather_all)  # (batch_size, raman_info)
        # print(f"Out:{output_spectrum.shape}")
        return output_spectrum
    def forward(self, batch):
        atoms, bonds, bond_atom_1, bond_atom_2, _, _, batch_mark_for_atoms, batch_mark_for_bonds, _ = batch
        predicted_spectrum = self.shared_procedure(
            atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds)
        return predicted_spectrum

    def training_step(self, batch, batch_idx):
        atoms, bonds, bond_atom_1, bond_atom_2, _, _, batch_mark_for_atoms, batch_mark_for_bonds, ramans_of_batch = batch
        predicted_spectrum = self.shared_procedure(
            atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds)
        # print(f"Pred:{predicted_spectrum.shape}")
        # print(f"Raman:{ramans_of_batch.shape}")
        loss = F.l1_loss(predicted_spectrum, ramans_of_batch)
        # print(f"loss:{loss}")
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        # similarity = F.cosine_similarity(
        #     predicted_spectrum, ramans_of_batch).mean()
        # self.log("train_simi", similarity, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        atoms, bonds, bond_atom_1, bond_atom_2, _, _, batch_mark_for_atoms, batch_mark_for_bonds, ramans_of_batch = batch
        predicted_spectrum = self.shared_procedure(
            atoms, bonds, bond_atom_1, bond_atom_2, batch_mark_for_atoms, batch_mark_for_bonds)
        loss = F.l1_loss(predicted_spectrum, ramans_of_batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        # similarity = F.cosine_similarity(
        #     predicted_spectrum, ramans_of_batch).mean()
        # self.log("val_simi", similarity, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        if self.hparams.optim_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        schedualer = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=64)
        return [optimizer], [schedualer]


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=3,
    mode='min',
)


def model_config(model):
    params = {}
    try:
        num_conv = int(config["model"]["num_of_megnetblock"])
        params["num_conv"] = num_conv
    except:
        pass
    try:
        optim_type = config["optimizer"]["type"]
        if optim_type == "AdamW":
            params["optim_type"] = "AdamW"
    except:
        pass
    try:
        optim_lr = float(config["optimizer"]["lr"])
        params["lr"] = optim_lr
    except:
        pass
    try:
        optim_weight_decay = float(config["optimizer"]["weight_decay"])
        params["weight_decay"] = optim_weight_decay
    except:
        pass
    return params


try: 
    path = config["checkpoint"]
    experiment = Experiment.load_from_checkpoint(path)
except KeyError:
    model_hpparams = model_config(config)
    print(model_hpparams)
    experiment = Experiment(**model_hpparams)

trainer_config = config["trainer"]
logger = TensorBoardLogger(prefix)
if trainer_config == "tune":
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger, callbacks=[
                         checkpoint_callback], auto_lr_find=True)
    trainer.tune(experiment, train_dataloader, validate_dataloader)
else:
    try: 
        path = config["checkpoint"]
        trainer = pl.Trainer(resume_from_checkpoint=path,gpus=1 if torch.cuda.is_available() else 0, logger=logger, callbacks=[checkpoint_callback],max_epochs=2000)
    except KeyError:
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(experiment, train_dataloader, validate_dataloader)
