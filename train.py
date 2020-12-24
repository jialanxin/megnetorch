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

parser = argparse.ArgumentParser(description="Select a train_config.yaml file")
parser.add_argument(dest="filename", metavar="/path/to/file")
arg = parser.parse_args()
path_to_file = arg.filename
with open(path_to_file, "r") as f:
    config = yaml.load(f.read(), Loader=yaml.BaseLoader)
prefix = config["prefix"]
model = config["model"]


with open("Structures.pkl", "rb") as f:
    structures = pickle.load(f)

with open("Raman_encoded_JVASP_90000.json", "r") as f:
    ramans = json.loads(f.read())


device = torch.device("cuda")
dataset = StructureRamanDataset(structures, ramans)
dataset_length = len(dataset)
train_set, validate_set = random_split(
    dataset, [int(0.8*dataset_length)+1, int(0.2*dataset_length)])
train_dataloader = DataLoader(
    dataset=train_set, batch_size=64, collate_fn=collate_fn, num_workers=4, shuffle=True)
validate_dataloader = DataLoader(
    dataset=validate_set, batch_size=64, collate_fn=collate_fn, num_workers=4, shuffle=True)
net = MegNet(num_of_megnetblock=int(model["num_of_megnetblock"] or 3))
net.to(device)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
schedualer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

# checkpoint = torch.load(prefix+"checkpoint_epoch_101_val_loss_0.0784395659076316.pkl")
# net.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# schedualer.load_state_dict(checkpoint["schedual_state_dict"])
# last_epoch = checkpoint["epoch"]


writer = SummaryWriter(prefix+"runs")

for epoch in range(0, 2002):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    train_loss = 0.0
    for i, data in enumerate(train_dataloader):
        atoms, state, bonds, bond_atom_1, bond_atom_2, atoms_mark, bonds_mark, ramans = data
        net.train()
        optimizer.zero_grad()
        predicted_spectrum = net(atoms.to(device), state.to(device), bonds.to(device), bond_atom_1.to(
            device), bond_atom_2.to(device), atoms_mark.to(device), bonds_mark.to(device))
        loss = loss_func(predicted_spectrum, ramans.to(device))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss = train_loss/i

    validate_loss = 0.0
    for i, data in enumerate(validate_dataloader):
        atoms, state, bonds, bond_atom_1, bond_atom_2, atoms_mark, bonds_mark, ramans = data
        net.eval()
        predicted_spectrum = net(atoms.to(device), state.to(device), bonds.to(device), bond_atom_1.to(
            device), bond_atom_2.to(device), atoms_mark.to(device), bonds_mark.to(device))
        loss = loss_func(predicted_spectrum, ramans.to(device))
        validate_loss += loss.item()
    validate_loss = validate_loss/i
    end.record()
    torch.cuda.synchronize()
    print(f"epoch:{epoch}")
    print(f"Time:{start.elapsed_time(end)}")
    print(f"train loss:{train_loss}")
    print(f"validate loss:{validate_loss}")
    writer.add_scalars("Train and Validate Loss", {
                       "train_loss": train_loss, "validate_loss": validate_loss}, epoch)
    schedualer.step()
    if epoch % 100 == 1:
        checkpoint = {"model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict(
        ), "schedualer_state_dict": schedualer.state_dict(), "epoch": epoch, "loss": validate_loss}
        torch.save(checkpoint, prefix +
                   f"/checkpoint_epoch_{epoch}_val_loss_{validate_loss}.pkl")
writer.close()
