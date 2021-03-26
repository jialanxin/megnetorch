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
            "pretrain/finetuned/epoch=3069-step=174989.ckpt")
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
    for i,data in enumerate(train_dataloader):
        _,raman = data
        predicted_spectrum =  model(data)
        hist = hist_count(0,raman,predicted_spectrum)
        if i == 0:
            m0 = hist.unsqueeze(dim=0)
        else:
            m0 = torch.cat((m0,hist.unsqueeze(dim=0)),dim=0)

        hist = hist_count(1,raman,predicted_spectrum)
        if i == 0:
            m1 = hist.unsqueeze(dim=0)
        else:
            m1 = torch.cat((m1,hist.unsqueeze(dim=0)),dim=0)

        hist = hist_count(2,raman,predicted_spectrum)
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
# label\predict:          0,       1,      2,                       
# 0,                 0.9702,  0.0289, 0.0007,                     
# 1,                 0.2776,  0.6366, 0.0881,                     
# 2,                 0.2752,  0.6383, 0.0822      
# Validate:
# label\predict:          0,          1,          2,
# 0,                   0.9354,     0.0557,     0.0077,
# 1,                   0.4340,     0.4462,     0.1007,
# 2,                   0.4307,     0.4495,     0.1003                

# Train:  loss_weight_e
# label\predict:          0,          1,          2,
# 0,                 0.8955,     0.0620,     0.0256
# 1,                 0.2277,     0.5963,     0.1192,
# 2,                 0.2257,     0.5918,     0.1255,
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.8736, 0.0756, 0.0318
# 1,                   0.3053, 0.4228, 0.1872,
# 2,                   0.2981, 0.4083, 0.2023

# Train:  loss_weight_1.5
# label\predict:          0,      1,      2,
# 0,                 0.9697, 0.0229, 0.0069
# 1,                 0.3431, 0.5947, 0.0582
# 2,                 0.3431, 0.5902, 0.0628,
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9417, 0.0395, 0.0149
# 1,                   0.4590, 0.3988, 0.1043,
# 2,                   0.4541, 0.3873, 0.1187

# Train:  loss_weight_e_sign
# label\predict:          0,      1,      2,
# 0,                 0.9570, 0.0412, 0.0017
# 1,                 0.1443, 0.8297, 0.0256
# 2,                 0.1460, 0.8246, 0.0290
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9165, 0.0696, 0.0112
# 1,                   0.3248, 0.5555, 0.0900
# 2,                   0.3238, 0.5446, 0.0963

# Train:  loss_weight_4_sign
# label\predict:          0,      1,      2,
# 0,                 0.9381, 0.0591, 0.0026
# 1,                 0.0952, 0.8891, 0.0156
# 2,                 0.0951, 0.8843, 0.0199
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9004, 0.0872, 0.0095
# 1,                   0.2691, 0.6080, 0.0988
# 2,                   0.2671, 0.5983, 0.1060

# Train:  loss_weight_4_sign_batch_128
# label\predict:          0,      1,      2,
# 0,                 0.9352, 0.0631, 0.0016
# 1,                 0.1748, 0.8111, 0.0139
# 2,                 0.1732, 0.8041, 0.0138
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9106, 0.0824, 0.0064
# 1,                   0.3825, 0.5310, 0.0730
# 2,                   0.3794, 0.5186, 0.0685

# Train:  loss_weight_4_sign_lr_1e-3
# label\predict:          0,      1,      2,
# 0,                 0.9363, 0.0619, 0.0017
# 1,                 0.1538, 0.8324, 0.0138
# 2,                 0.1524, 0.8252, 0.0137
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9117, 0.0814, 0.0060
# 1,                   0.3790, 0.5466, 0.0600
# 2,                   0.3761, 0.5332, 0.0563

# Train:  loss_weight_4_sign_batch_256
# label\predict:          0,      1,      2,
# 0,                 0.9237, 0.0715, 0.0044
# 1,                 0.1082, 0.8612, 0.0304
# 2,                 0.1072, 0.8565, 0.0360
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.8896, 0.0962, 0.0109
# 1,                   0.2537, 0.6153, 0.1021
# 2,                   0.2542, 0.6021, 0.1106

# Train:  loss_weight_3_sign
# label\predict:          0,      1,      2,
# 0,                 0.9537, 0.0445, 0.0017
# 1,                 0.1383, 0.8405, 0.0210
# 2,                 0.1371, 0.8382, 0.0244
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9153, 0.0726, 0.0095
# 1,                   0.3295, 0.5409, 0.1000
# 2,                   0.3268, 0.5204, 0.1131

# Train:  loss_weight_1_sign_pure_l1loss
# label\predict:          0,      1,      2,
# 0,                 0.9867, 0.0131, 0.0000
# 1,                 0.3077, 0.6683, 0.0235
# 2,                 0.3087, 0.6632, 0.0277
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9542, 0.0377, 0.0061
# 1,                   0.4974, 0.3980, 0.0774
# 2,                   0.4961, 0.3835, 0.0905

# Train:  loss_weight_5_sign
# label\predict:          0,      1,      2,
# 0,                 0.9370, 0.0615, 0.0015
# 1,                 0.1535, 0.8418, 0.0046
# 2,                 0.1522, 0.8344, 0.0046
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.9041, 0.0892, 0.0056
# 1,                   0.3512, 0.5763, 0.0613
# 2,                   0.3501, 0.5611, 0.0575

# Train:  loss_weight_6_sign
# label\predict:          0,      1,      2,
# 0,                 0.9014, 0.0961, 0.0024
# 1,                 0.0779, 0.9196, 0.0024
# 2,                 0.0772, 0.9116, 0.0024
# Validate:
# label\predict:            0,      1,      2,
# 0,                   0.8791, 0.1134, 0.0066
# 1,                   0.2560, 0.6621, 0.0701
# 2,                   0.2400, 0.6624, 0.0657