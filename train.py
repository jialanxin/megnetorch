from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from dataset import StructureRamanDataset
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


def prepare_datesets(struct_raman_file):
    with open(struct_raman_file, "r") as f:
        struct_raman_json = json.loads(f.read())
    dataset = StructureRamanDataset(struct_raman_json)
    return dataset





train_set = torch.load("materials/JVASP/Train_set.pt")
validate_set = torch.load("materials/JVASP/Valid_set.pt")
train_dataloader = DataLoader(
    dataset=train_set, batch_size=64, collate_fn=collate_fn, num_workers=4, shuffle=True)
validate_dataloader = DataLoader(
    dataset=validate_set, batch_size=64, collate_fn=collate_fn, num_workers=4)


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

checkpoint_callback = ModelCheckpoint(
    monitor='val_simi',
    save_top_k=3,
    mode='max',
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
