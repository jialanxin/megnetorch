import json
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from dataloader import collate_fn
from dataset import StructureFmtEnDataset
from torch.nn import Embedding, RReLU, ReLU, Dropout


def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))


def ff_output(input_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.RReLU(), Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.RReLU(), Dropout(0.1), torch.nn.Linear(64, output_dim))


class Experiment(pl.LightningModule):
    def __init__(self, num_enc=12, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.atom_embedding = ff(71)
        self.position_embedding = ff(30)
        self.lattice_embedding = ff(90)
        encode_layer = torch.nn.TransformerEncoderLayer(
            d_model=32, nhead=8, dim_feedforward=128)
        self.encoder = torch.nn.TransformerEncoder(
            encode_layer, num_layers=num_enc)
        self.readout = ff_output(input_dim=32, output_dim=1)

    def shared_procedure(self, atoms, positions, padding_mask, lattice):
        # atoms: (batch_size,max_atoms,atoms_info)
        # positions: (batch_size, max_atoms, position_info)
        # padding_mask: (batch_size, max_atoms)
        # lattice: (batchsize, lattice_info)
        atoms = self.atom_embedding(atoms)  # (batch_size,max_atoms,atoms_info)
        # (batch_size,max_atoms,positions_info)
        positions = self.position_embedding(positions)
        atoms = atoms+positions  # (batch_size,max_atoms,atoms_info)
        lattice = self.lattice_embedding(lattice)  # (batch_size,lacttice_info)
        # (batch_size,1,lacttice_info)
        lattice = torch.unsqueeze(lattice, dim=1)
        # (batch_size,1+max_atoms,atoms_info)
        atoms = torch.cat((lattice, atoms), dim=1)
        # (1+max_atoms, batch_size, atoms_info)
        atoms = torch.transpose(atoms, dim0=0, dim1=1)
        batch_size = padding_mask.shape[0]
        cls_padding = torch.zeros((batch_size, 1)).bool().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # (batch_size, 1)

        # (batch_size, 1+max_atoms)
        padding_mask = torch.cat((cls_padding, padding_mask), dim=1)

        # (1+max_atoms, batch_size, atoms_info)
        atoms = self.encoder(src=atoms, src_key_padding_mask=padding_mask)

        system_out = atoms[0]  # (batch_size,atoms_info)

        output_spectrum = self.readout(system_out)  # (batch_size, raman_info)

        return output_spectrum

    def forward(self, batch):
        encoded_graph, _ = batch
        atoms = encoded_graph["atoms"]
        positions = encoded_graph["positions"]
        padding_mask = encoded_graph["padding_mask"]
        lattice = encoded_graph["lattice"]
        predicted_spectrum = self.shared_procedure(
            atoms, positions, padding_mask, lattice)
        return predicted_spectrum

    def training_step(self, batch, batch_idx):
        encoded_graph, ramans = batch
        atoms = encoded_graph["atoms"]
        positions = encoded_graph["positions"]
        padding_mask = encoded_graph["padding_mask"]
        lattice = encoded_graph["lattice"]
        predicted_spectrum = self.shared_procedure(
            atoms, positions, padding_mask, lattice)
        loss = F.l1_loss(predicted_spectrum, ramans)
        # print(f"loss:{loss}")
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        # similarity = F.cosine_similarity(
        #     predicted_spectrum, ramans_of_batch).mean()
        # self.log("train_simi", similarity, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        encoded_graph, ramans = batch
        atoms = encoded_graph["atoms"]
        positions = encoded_graph["positions"]
        padding_mask = encoded_graph["padding_mask"]
        lattice = encoded_graph["lattice"]
        predicted_spectrum = self.shared_procedure(
            atoms, positions, padding_mask, lattice)
        loss = F.l1_loss(predicted_spectrum, ramans)
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


def model_config(model):
    params = {}
    try:
        num_enc = int(config["model"]["num_of_encoder"])
        params["num_conv"] = num_enc
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a train_config.yaml file")
    parser.add_argument(dest="filename", metavar="/path/to/file")
    arg = parser.parse_args()
    path_to_file = arg.filename
    with open(path_to_file, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.BaseLoader)
    prefix = config["prefix"]

    train_set = torch.load("./materials/mp/Train_fmten_emd_set.pt")
    validate_set = torch.load("./materials/mp/Valid_fmten_emd_set.pt")

    train_dataloader = DataLoader(
        dataset=train_set, batch_size=64, num_workers=4)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=64, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=3,
        mode='min',
    )
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
            trainer = pl.Trainer(resume_from_checkpoint=path, gpus=1 if torch.cuda.is_available(
            ) else 0, logger=logger, callbacks=[checkpoint_callback], max_epochs=1000)
        except KeyError:
            trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger,
                                 callbacks=[checkpoint_callback])
        trainer.fit(experiment, train_dataloader, validate_dataloader)
