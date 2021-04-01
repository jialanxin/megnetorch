import json
import argparse
import torch
import torch.nn.functional as F
import yaml
import pytorch_lightning as pl
import warnings
from warnings import warn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.nn import Embedding, RReLU, ReLU, Dropout
from dataset import StructureRamanDataset
from pretrain_fmten import Experiment as FmtEn
from pretrain_spgp import Experiment as SPGP


def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128))


def ff_output(input_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.RReLU(), Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.RReLU(), Dropout(0.1), torch.nn.Linear(64, output_dim))


class Experiment(pl.LightningModule):
    def __init__(self, optim_type="Adam", lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        fmten_model = FmtEn.load_from_checkpoint(
            "pretrain/fmten/epoch=970-step=489383.ckpt")
        spgp_model = SPGP.load_from_checkpoint(
            "pretrain/spacegroup/epoch=450-step=227303.ckpt")
        self.atom_embedding = fmten_model.atom_embedding
        self.atomic_number_embedding = fmten_model.atomic_number_embedding
        self.space_group_number_embedding = fmten_model.space_group_number_embedding
        self.mendeleev_number_embedding = fmten_model.mendeleev_number_embedding
        self.position_embedding = spgp_model.position_embedding
        self.lattice_embedding = spgp_model.lattice_embedding
        self.encoder = spgp_model.encoder
        self.readout = ff_output(input_dim=128, output_dim=100)

    @staticmethod
    def Gassian_expand(value_list, min_value, max_value, intervals, expand_width, device):
        value_list = value_list.expand(-1, -1, intervals)
        centers = torch.linspace(min_value, max_value, intervals).to(device)
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (batch_size, max_atoms, 40)
        elecneg = self.Gassian_expand(elecneg, 0.5, 4.0, 40, 0.09, device)
        # (batch_size, max_atoms, 40)
        covrad = self.Gassian_expand(covrad, 50, 250, 40, 5, device)
        # (batch_size, max_atoms, 40)
        FIE = self.Gassian_expand(FIE, 3, 25, 40, 0.6, device)
        # (batch_size, max_atoms, 40)
        elecaffi = self.Gassian_expand(elecaffi, -3, 3.7, 40, 0.17, device)
        # (batch_size, max_atoms, 40)
        atmwht = self.Gassian_expand(atmwht, 0, 210, 40, 5.25, device)
        atoms = torch.cat(
            (atoms, elecneg, covrad, FIE, elecaffi, atmwht), dim=2)         # (batch_size, max_atoms, 259)
        atoms = self.atom_embedding(atoms)  # (batch_size,max_atoms,atoms_info)

        positions = positions.unsqueeze(dim=3).expand(-1, -1, 3, 40)
        centers = torch.linspace(-15, 18, 40).to(device)
        # (batch_size, max_atoms, 3, 40)
        positions = torch.exp(-(positions - centers)**2/0.83**2)
        # (batch_size, max_atoms, 120)
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
            lattice, -15, 18, 40, 0.83, device)  # (batch_size, 9, 40)
        lattice = torch.flatten(lattice, start_dim=1)  # (batch_size,360)

        # lattice: (batch_size,1,1)
        cell_volume = torch.log(encoded_graph["CV"])
        cell_volume = self.Gassian_expand(
            cell_volume, 3, 8, 40, 0.13, device)  # (batch_size,1,40)
        cell_volume = torch.flatten(
            cell_volume, start_dim=1)  # (batch_size, 40)

        lattice = torch.cat((lattice, cell_volume), dim=1)  # (batch_size, 200)
        lattice = self.lattice_embedding(lattice)  # (batch_size,lacttice_info)
        # (batch_size,1,lacttice_info)
        lattice = torch.unsqueeze(lattice, dim=1)

        space_group_number = encoded_graph["SGN"]  # (batch_size,1)
        sgn = self.space_group_number_embedding(
            space_group_number)  # (batch_size, 1,lattice_info)
        lattice = lattice+sgn

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

        # (batch_size, wavelength_clips, confidence+position)
        output_spectrum = self.readout(system_out).reshape(-1, 25, 2, 2)

        return output_spectrum

    def forward(self, batch):
        predicted_spectrum = self.shared_procedure(batch)
        return predicted_spectrum

    @staticmethod
    def scores(spectrum_sign, raman_sign):
        where_zero = torch.eq(raman_sign, torch.zeros_like(raman_sign))
        where_not_zero = torch.logical_not(where_zero)
        spectrum_should_zero = spectrum_sign[where_zero]
        spectrum_should_not_zero = spectrum_sign[where_not_zero]
        true_negative = torch.eq(
            spectrum_should_zero, torch.zeros_like(spectrum_should_zero))  # 0->0
        false_positive = torch.logical_not(
            true_negative)                                     # 0->1
        false_negative = torch.eq(spectrum_should_not_zero, torch.zeros_like(
            spectrum_should_not_zero))  # 1->0
        true_positive = torch.logical_not(
            false_negative)                                     # 1->1
        true_negative = true_negative.float().sum()
        false_positive = false_positive.float().sum()
        false_negative = false_negative.float().sum()
        true_positive = true_positive.float().sum()
        Accuracy = (true_positive+true_negative)/(true_positive +
                                                  true_negative+false_positive+false_negative)
        Precision = true_positive/(true_positive+false_positive)
        Recall = true_positive/(true_positive+false_negative)
        F1score = 2*Precision*Recall/(Precision+Recall)
        # print(f"A:{Accuracy} P:{Precision} R:{Recall} F:{F1score}")
        return Accuracy, Precision, Recall, F1score

    @staticmethod
    def yolov1_loss(raman, predict):
        # raman (batch_size, wavelength_clips, 2) # predict (batch_size, wavelength_clips, workers, 2)
        position_label = raman[:, :, 1]  # (batch_size, wavelength_clips)
        # (batch_size, wavelength_clips,workers)
        predict_position = predict[:, :, :, 1]
        # (batch_size, wavelength_clips,workers)
        position_label = position_label.unsqueeze(
            dim=-1).expand_as(predict_position)
        # (batch_size, wavelength_clips, workers)
        position_accuracy = 1 - \
            F.l1_loss(predict_position, position_label, reduction="none")
        # (batch_size, wavelength_clips, workers)
        position_prefer_mask = torch.argmax(
            position_accuracy, dim=2, keepdim=True)
        position_prefer_mask = position_prefer_mask.unsqueeze(
            dim=-1).expand_as(predict)  # (batch_size, wavelength_clips, workers, 2)
        confidence_label = raman[:, :, 0]  # (batch_size, wavelength_clips)
        object_mask = confidence_label.bool().unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand_as(predict)  # (batch_size, wavelength_clips, workers,2)
        # (batch_size, wavelength_clips, workers,2)
        object_mask = torch.logical_and(object_mask, position_prefer_mask)
        nonobject_mask = torch.logical_not(object_mask)
        # (batch_size, wavelength_clips, workers, 2)
        raman = raman.unsqueeze(dim=2).expand_as(predict)

        object_target = torch.masked_select(raman, object_mask).reshape(
            (-1, 2))  # (object_dim ,2)
        object_predict = torch.masked_select(
            predict, object_mask).reshape((-1, 2))
        nonobject_target = torch.masked_select(raman, nonobject_mask).reshape(
            (-1, 2))  # (nonobject_dim ,2)
        nonobject_predict = torch.masked_select(
            predict, nonobject_mask).reshape((-1, 2))

        nonobject_confidence_loss = F.mse_loss(
            nonobject_predict[:, 0], nonobject_target[:, 0], reduction="sum")

        object_position_loss = F.mse_loss(
            object_predict[:, 1], object_target[:, 1], reduction="sum")

        position_accuracy = 1 - \
            F.l1_loss(object_predict[:, 1],
                      object_target[:, 1], reduction="none")  # (object_dim,)
        object_confidence_target = object_target[:, 0]*position_accuracy
        object_confidence_loss = F.mse_loss(
            object_confidence_target, object_predict[:, 0], reduction="sum")

        loss = 0.2*nonobject_confidence_loss + \
            object_confidence_loss+5*object_position_loss
        batch_size = raman.shape[0]
        loss = loss/batch_size
        return loss

    def loss_scores(self, ramans, predicted_spectrum):
        # raman (batch_size, wavelength_clips, 2) # predict (batch_size, wavelength_clips, workers, 2)
        raman_confidence = ramans[:, :, 0]  # (batch_size, wavelength_clips)
        # (batch_size, wavelength_clips, workers)
        predicted_confidence = predicted_spectrum[:, :, :, 0]
        predicted_confidence, _ = torch.max(
            predicted_confidence, dim=2)  # (batch_size, wavelength_clips)
        loss_trivial = F.l1_loss(predicted_confidence,
                                 raman_confidence, reduction="none")

        loss_weight = torch.pow(3, raman_confidence)
        weight_sum = loss_weight.sum(dim=1, keepdim=True)
        loss_weight = loss_weight/weight_sum
        loss_weighed = torch.sum(loss_trivial*loss_weight, dim=1).mean()

        predicted_round = torch.round(predicted_confidence)
        acc, prc, rec, f1 = self.scores(predicted_round, raman_confidence)
        NaNcheck = torch.logical_or(prc.isnan(), rec.isnan())
        NaNcheck = torch.logical_or(NaNcheck, f1.isnan())
        return loss_trivial, loss_weighed, acc, prc, rec, f1, NaNcheck

    def training_step(self, batch, batch_idx):
        _, ramans = batch
        predicted_spectrum = self.shared_procedure(batch)
        loss_yolo = self.yolov1_loss(ramans, predicted_spectrum)
        self.log("train_yolo", loss_yolo, on_epoch=True, on_step=False)
        loss_trivial, loss_weighed, acc, prc, rec, f1, NaNcheck = self.loss_scores(
            ramans, predicted_spectrum)
        self.log("train_loss", loss_trivial.mean(),
                 on_epoch=True, on_step=False)
        self.log("train_loss_weighed", loss_weighed,
                 on_epoch=True, on_step=False)
        if NaNcheck.item() == False:
            self.log("train_acc", acc, on_epoch=True, on_step=False)
            self.log("train_prc", prc, on_epoch=True, on_step=False)
            self.log("train_rec", rec, on_epoch=True, on_step=False)
            self.log("train_f1", f1, on_epoch=True, on_step=False)
        else:
            # raise RuntimeError("NaN")
            print("Train:NaN")
        return loss_yolo

    def validation_step(self, batch, batch_idx):
        _, ramans = batch
        predicted_spectrum = self.shared_procedure(batch)
        loss_yolo = self.yolov1_loss(ramans, predicted_spectrum)
        self.log("val_yolo", loss_yolo, on_epoch=True, on_step=False)
        loss_trivial, loss_weighed, acc, prc, rec, f1, NaNcheck = self.loss_scores(
            ramans, predicted_spectrum)
        self.log("val_loss", loss_trivial.mean(), on_epoch=True, on_step=False)
        self.log("val_loss_weighed", loss_weighed,
                 on_epoch=True, on_step=False)
        if NaNcheck.item() == False:
            self.log("val_acc", acc, on_epoch=True, on_step=False)
            self.log("val_prc", prc, on_epoch=True, on_step=False)
            self.log("val_rec", rec, on_epoch=True, on_step=False)
            self.log("val_f1", f1, on_epoch=True, on_step=False)
        else:
            print("Val:NaN")
        return loss_yolo

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


def model_config(config):
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
    warnings.simplefilter('always', UserWarning)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_weighed',
        save_top_k=3,
        mode='min',
    )

    train_set = torch.load("materials/JVASP/Train_raman_set_25_uneq_yolov1.pt")
    validate_set = torch.load(
        "materials/JVASP/Valid_raman_set_25_uneq_yolov1.pt")
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=128, num_workers=2, shuffle=True)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=128, num_workers=2)

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
            ) else 0, logger=logger, callbacks=[checkpoint_callback], max_epochs=4000)
        except KeyError:
            trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger,
                                 callbacks=[checkpoint_callback],  max_epochs=4000)
        trainer.fit(experiment, train_dataloader, validate_dataloader)
