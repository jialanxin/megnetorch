import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import StructureXRDDataset
from pretrain_fmten import Experiment as FmtEn
from cos_anneal.cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts


def ff(input_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 128))


def ff_output(input_dim, output_dim):
    # , torch.nn.RReLU(), Dropout(0.3), torch.nn.Linear(128, 128), torch.nn.RReLU(), Dropout(0.3), torch.nn.Linear(128, output_dim))
    return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))


class Experiment(pl.LightningModule):
    def __init__(self, optim_type="Adam", lr=1e-3, weight_decay=0.0, nonobj=0.4, coord=5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        fmten_model = FmtEn.load_from_checkpoint(
            "pretrain/fmten/epoch=585-step=659835.ckpt")
        self.atom_embedding = fmten_model.atom_embedding
        self.atomic_number_embedding = fmten_model.atomic_number_embedding
        self.space_group_number_embedding = fmten_model.space_group_number_embedding
        self.mendeleev_number_embedding = fmten_model.mendeleev_number_embedding
        self.position_embedding = fmten_model.position_embedding
        self.lattice_embedding = fmten_model.lattice_embedding
        self.encoder = fmten_model.encoder
        self.readout = ff_output(input_dim=256, output_dim=50)

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

       # (batch_size, max_atoms, 80)
        elecneg = self.Gassian_expand(elecneg, 0.5, 4.0, 80, 0.04, device)
        # (batch_size, max_atoms, 80)
        covrad = self.Gassian_expand(covrad, 50, 250, 80, 2.5, device)
        # (batch_size, max_atoms, 80)
        FIE = self.Gassian_expand(FIE, 3, 25, 80, 0.28, device)
        # (batch_size, max_atoms, 80)
        elecaffi = self.Gassian_expand(elecaffi, -3, 3.7, 80, 0.08, device)
        # (batch_size, max_atoms, 80)
        atmwht = self.Gassian_expand(atmwht, 0, 210, 80, 2.63, device)
        atoms = torch.cat(
            (atoms, elecneg, covrad, FIE, elecaffi, atmwht), dim=2)         # (batch_size, max_atoms, 459)
        atoms = self.atom_embedding(atoms)  # (batch_size,max_atoms,atoms_info)

        positions = positions.unsqueeze(dim=3).expand(-1, -1, 3, 80)
        centers = torch.linspace(-15, 18, 80).to(device)
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
            lattice, -15, 18, 80, 0.41, device)  # (batch_size, 9, 80)
        lattice = torch.flatten(lattice, start_dim=1)  # (batch_size,720)

        # lattice: (batch_size,1,1)
        cell_volume = torch.log(encoded_graph["CV"])
        cell_volume = self.Gassian_expand(
            cell_volume, 3, 8, 80, 0.06, device)  # (batch_size,1,80)
        cell_volume = torch.flatten(
            cell_volume, start_dim=1)  # (batch_size, 80)

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
        output_spectrum = self.readout(system_out).reshape(-1, 25, 2)

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

    def yolov1_loss(self, raman, predict):
        # predict (batch_size, wavelength_clips, 2)
        xrd_position = raman["PS"]  # (batch_size, wavelength_clips)
        xrd_intensity = raman["IT"]  # (batch_size, wavelength_clips)
        predict_intensity = predict[:, :, 0]  # (batch_size, wavelength_clips)
        predict_position = predict[:, :, 1]  # (batch_size, wavelength_clips)
        nonobject_mask = torch.eq(xrd_intensity, torch.zeros_like(
            xrd_intensity))  # (batch_size, wavelength_clips)
        # (batch_size, wavelength_clips)
        object_mask = torch.logical_not(nonobject_mask)

        # (nonobj,)
        predict_intensity_where_should_zero = predict_intensity[nonobject_mask]
        non_obj_loss = F.mse_loss(predict_intensity_where_should_zero, torch.zeros_like(
            predict_intensity_where_should_zero), reduction="sum")  # (1,)

        # (obj,)
        predict_intensity_where_should_exist = predict_intensity[object_mask]
        xrd_intensity_nonzero = xrd_intensity[object_mask]  # (obj,)
        intensity_loss = F.mse_loss(
            predict_intensity_where_should_exist, xrd_intensity_nonzero, reduction="sum")

        # (obj,)
        predict_position_where_should_exist = predict_position[object_mask]
        xrd_position_nonzero = xrd_position[object_mask]  # (obj,)
        position_loss = F.mse_loss(
            predict_position_where_should_exist, xrd_position_nonzero, reduction="sum")

        loss = self.hparams.nonobj*non_obj_loss + \
            intensity_loss+self.hparams.coord*position_loss
        batch_size = xrd_intensity.shape[0]
        loss = loss/batch_size
        return loss

    def loss_scores(self, ramans, predicted_spectrum):
        # predict (batch_size, wavelength_clips, 2)
        xrd_intensity = ramans["IT"]  # (batch_size, wavelength_clips)
        xrd_intensity = torch.sign(xrd_intensity)
        # (batch_size, wavelength_clips)
        predicted_intensity = predicted_spectrum[:, :, 0]
        predicted_intensity = F.threshold(predicted_intensity, 0.1, 0)
        predicted_intensity = torch.sign(predicted_intensity)
        acc, prc, rec, f1 = self.scores(predicted_intensity, xrd_intensity)
        NaNcheck = torch.logical_or(prc.isnan(), rec.isnan())
        NaNcheck = torch.logical_or(NaNcheck, f1.isnan())
        return acc, prc, rec, f1, NaNcheck

    def training_step(self, batch, batch_idx):
        _, ramans = batch
        predicted_spectrum = self.shared_procedure(batch)
        loss_yolo = self.yolov1_loss(ramans, predicted_spectrum)
        self.log("train_yolo", loss_yolo, on_epoch=True, on_step=False)
        acc, prc, rec, f1, NaNcheck = self.loss_scores(
            ramans, predicted_spectrum)
        if NaNcheck.item() == False:
            self.log("train_acc", acc, on_epoch=True, on_step=False)
            self.log("train_prc", prc, on_epoch=True, on_step=False)
            self.log("train_rec", rec, on_epoch=True, on_step=False)
            self.log("train_f1", f1, on_epoch=True, on_step=False)
        else:
            print("Train:NaN")
        return loss_yolo

    def validation_step(self, batch, batch_idx):
        _, ramans = batch
        predicted_spectrum = self.shared_procedure(batch)
        loss_yolo = self.yolov1_loss(ramans, predicted_spectrum)
        self.log("val_yolo", loss_yolo, on_epoch=True, on_step=False)
        acc, prc, rec, f1, NaNcheck = self.loss_scores(
            ramans, predicted_spectrum)
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
        schedualer = CosineAnnealingWarmupRestarts(
            optimizer=optimizer, first_cycle_steps=200, max_lr=self.hparams.lr, min_lr=0, warmup_steps=30)
        return [optimizer], [schedualer]


def model_config(optim_type, optim_lr, optim_weight_decay, model_nonboj, model_coord):
    params = {}
    if optim_type == "AdamW":
        params["optim_type"] = "AdamW"
    params["lr"] = optim_lr
    params["weight_decay"] = optim_weight_decay
    params["nonobj"] = model_nonboj
    params["coord"] = model_coord
    return params


if __name__ == "__main__":
    prefix = "/home/jlx/v0.4.8/1.pretrain_xrd/"
    trainer_config = "fit"
    checkpoint_path = None
    model_hpparams = model_config(
        optim_type="AdamW", optim_lr=1e-4, model_nonboj=0.4, model_coord=2)
    # train_set_part = 1
    # epochs = 250*train_set_part
    epochs = 1000
    batch_size = 512

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_yolo',
        save_top_k=3,
        mode='min',
    )

    train_set = torch.load("materials/OQMD/Train_xrd_set.pt")
    validate_set = torch.load(
        "materials/OQMD/Valid_xrd_set.pt")
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=batch_size, num_workers=4)

    experiment = Experiment(**model_hpparams)

    logger = TensorBoardLogger(prefix)
    if trainer_config == "tune":
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger, callbacks=[
            checkpoint_callback], auto_lr_find=True)
        trainer.tune(experiment, train_dataloader, validate_dataloader)
    else:
        if checkpoint_path != None:
            trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, gpus=1 if torch.cuda.is_available(
            ) else 0, logger=logger, callbacks=[checkpoint_callback], max_epochs=epochs)
        else:
            trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=logger,
                                 callbacks=[checkpoint_callback],  max_epochs=epochs)
        trainer.fit(experiment, train_dataloader, validate_dataloader)
