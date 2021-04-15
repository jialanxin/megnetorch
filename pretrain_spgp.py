import pytorch_lightning as pl
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import StructureSpaceGroupDataset
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from pretrain_fmten import Experiment as Fmten
from cos_anneal.cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts


def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 256))


def ff_output(input_dim, output_dim):
    # , torch.nn.RReLU(), Dropout(0.2), torch.nn.Linear(128, 256), torch.nn.RReLU(), Dropout(0.2), torch.nn.Linear(256, output_dim))
    return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))


class Experiment(pl.LightningModule):
    def __init__(self, num_enc=6, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        fmten_model = Fmten.load_from_checkpoint(
            "pretrain/fmten/epoch=585-step=659835.ckpt")
        self.atom_embedding = fmten_model.atom_embedding
        self.atomic_number_embedding = fmten_model.atomic_number_embedding
        self.mendeleev_number_embedding = fmten_model.mendeleev_number_embedding
        self.position_embedding = fmten_model.position_embedding
        self.lattice_embedding = fmten_model.lattice_embedding
        self.encoder = fmten_model.encoder
        self.readout = ff_output(input_dim=256, output_dim=230)
        self.accuracy = Accuracy()

    @staticmethod
    def Gassian_expand(value_list, min_value, max_value, intervals, expand_width):
        value_list = value_list.expand(-1, -1, intervals)
        centers = torch.linspace(min_value, max_value,
                                 intervals).type_as(value_list)
        result = torch.exp(-(value_list - centers)**2/expand_width**2)
        return result

    def shared_procedure(self, batch):
        encoded_graph, _ = batch
        # atoms: (batch_size,max_atoms,59)
        atoms = encoded_graph["atoms"]
        # padding_mask: (batch_size, max_atoms)
        padding_mask = encoded_graph["padding_mask"]
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

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (batch_size, max_atoms, 80)
        elecneg = self.Gassian_expand(elecneg, 0.5, 4.0, 80, 0.04)
        # (batch_size, max_atoms, 80)
        covrad = self.Gassian_expand(covrad, 50, 250, 80, 2.5)
        # (batch_size, max_atoms, 80)
        FIE = self.Gassian_expand(FIE, 3, 25, 80, 0.28)
        # (batch_size, max_atoms, 80)
        elecaffi = self.Gassian_expand(elecaffi, -3, 3.7, 80, 0.08)
        # (batch_size, max_atoms, 80)
        atmwht = self.Gassian_expand(atmwht, 0, 210, 80, 2.63)
        atoms = torch.cat(
            (atoms, elecneg, covrad, FIE, elecaffi, atmwht), dim=2)         # (batch_size, max_atoms, 459)
        atoms = self.atom_embedding(atoms)  # (batch_size,max_atoms,atoms_info)

        positions = positions.unsqueeze(dim=3).expand(-1, -1, 3, 80)
        centers = torch.linspace(-15, 18, 80).type_as(positions)
        # (batch_size, max_atoms, 3, 80)
        positions = torch.exp(-(positions - centers)**2/0.41**2)
        # (batch_size, max_atoms, 240)
        positions = torch.flatten(positions, start_dim=2)

        # (batch_size,max_atoms,positions_info)
        positions = self.position_embedding(positions)

        atmnb = encoded_graph["AN"]  # (batch_size, max_atoms)
        atomic_numbers = self.atomic_number_embedding(atmnb)

        mennb = encoded_graph["MN"]  # (batch_size, max_atoms)
        mendeleev_numbers = self.mendeleev_number_embedding(
            mennb)  # (batch_size, max_atoms, atoms_info)

        atoms = atoms+atomic_numbers+mendeleev_numbers + \
            positions  # (batch_size,max_atoms,atoms_info)

        lattice = encoded_graph["lattice"]  # lattice: (batch_size, 9, 1)
        lattice = self.Gassian_expand(
            lattice, -15, 18, 80, 0.41)  # (batch_size, 9, 80)
        lattice = torch.flatten(lattice, start_dim=1)  # (batch_size,720)

        # lattice: (batch_size,1,1)
        cell_volume = torch.log(encoded_graph["CV"])
        cell_volume = self.Gassian_expand(
            cell_volume, 3, 8, 80, 0.06)  # (batch_size,1,80)
        cell_volume = torch.flatten(
            cell_volume, start_dim=1)  # (batch_size, 80)

        lattice = torch.cat((lattice, cell_volume), dim=1)  # (batch_size, 800)
        lattice = self.lattice_embedding(lattice)  # (batch_size,lacttice_info)
        # (batch_size,1,lacttice_info)
        lattice = torch.unsqueeze(lattice, dim=1)

        # (batch_size,1+max_atoms,atoms_info)
        atoms = torch.cat((lattice, atoms), dim=1)
        # (1+max_atoms, batch_size, atoms_info)
        atoms = torch.transpose(atoms, dim0=0, dim1=1)
        batch_size = padding_mask.shape[0]
        cls_padding = torch.zeros((batch_size, 1)).bool().type_as(
            padding_mask)  # (batch_size, 1)

        # (batch_size, 1+max_atoms)
        padding_mask = torch.cat((cls_padding, padding_mask), dim=1)

        # (1+max_atoms, batch_size, atoms_info)
        atoms = self.encoder(src=atoms, src_key_padding_mask=padding_mask)

        system_out = atoms[0]  # (batch_size,atoms_info)

        output_spectrum = self.readout(system_out)  # (batch_size, raman_info)
        return output_spectrum

    def forward(self, batch):
        predicted_spectrum = self.shared_procedure(batch)
        return predicted_spectrum

    def training_step(self, batch, batch_idx):
        _, ramans = batch
        ramans = ramans.squeeze(dim=1)
        predicted_spectrum = self.shared_procedure(batch)
        loss = F.cross_entropy(predicted_spectrum, ramans)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        probability = F.softmax(predicted_spectrum, dim=1)
        self.accuracy(probability, ramans)
        return loss

    def on_train_epoch_end(self, outputs) -> None:
        self.log("train_acc", self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        _, ramans = batch
        ramans = ramans.squeeze(dim=1)
        predicted_spectrum = self.shared_procedure(batch)
        loss = F.cross_entropy(predicted_spectrum, ramans)
        self.log("val_loss", loss, on_epoch=True,
                 on_step=False, sync_dist=True)
        probability = F.softmax(predicted_spectrum, dim=1)
        self.accuracy(probability, ramans)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.accuracy.compute())

    def configure_optimizers(self):
        if self.hparams.optim_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        schedualer = CosineAnnealingWarmupRestarts(
            optimizer=optimizer, first_cycle_steps=200, max_lr=self.hparams.lr, min_lr=0, warmup_steps=30)
        return [optimizer], [schedualer]


def model_config(optim_type, optim_lr, optim_weight_decay):
    params = {}
    if optim_type == "AdamW":
        params["optim_type"] = "AdamW"
    params["lr"] = optim_lr
    params["weight_decay"] = optim_weight_decay
    return params


if __name__ == "__main__":
    prefix = "/home/jlx/v0.4.7/9.pretrain_spgp_oqmd_layer_12/"
    trainer_config = "fit"
    checkpoint_path = None
    model_hpparams = model_config(
        optim_type="AdamW", optim_lr=1e-4, optim_weight_decay=0)
    # train_set_part = 1
    # epochs = 250*train_set_part
    epochs = 1000
    batch_size = 128
    gpus = 4 if torch.cuda.is_available() else 0
    acce = "ddp"

    train_set = torch.load("./materials/OQMD/Train_spgp_set.pt")
    validate_set = torch.load("./materials/OQMD/Valid_spgp_set.pt")
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=batch_size, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=3,
        mode='max',
    )

    experiment = Experiment(**model_hpparams)

    logger = TensorBoardLogger(prefix)
    if trainer_config == "tune":
        trainer = pl.Trainer(gpus=gpus, logger=logger, callbacks=[
            checkpoint_callback], auto_lr_find=True)
        trainer.tune(experiment, train_dataloader, validate_dataloader)
    else:
        if checkpoint_path != None:
            trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, gpus=gpus, logger=logger, callbacks=[
                                 checkpoint_callback], max_epochs=epochs)
        else:
            trainer = pl.Trainer(gpus=gpus, logger=logger, accelerator=acce, auto_select_gpus=True,
                                 callbacks=[checkpoint_callback], max_epochs=epochs)
        trainer.fit(experiment, train_dataloader, validate_dataloader)
