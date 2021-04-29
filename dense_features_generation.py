from finetune import Experiment as Finetune
from finetune import Backbone
from dataset import StructureRamanDataset
import torch
from torch.utils.data import DataLoader
import json

class FeatureNet(Backbone):
    def __init__(self):
        super().__init__()
        finetune_model = Finetune().load_from_checkpoint("pretrain/finetuned/epoch=184-step=9249.ckpt")
        backbone = finetune_model.pretrain_freezed
        self.atom_embedding = backbone.atom_embedding
        self.atomic_number_embedding = backbone.atomic_number_embedding
        self.space_group_number_embedding = backbone.space_group_number_embedding
        self.mendeleev_number_embedding = backbone.mendeleev_number_embedding
        self.position_embedding = backbone.position_embedding
        self.lattice_embedding = backbone.lattice_embedding
        encoder = backbone.encoder
        encoder.layers = encoder.layers[:2]
        self.encoder = encoder

if __name__=="__main__":
    dataset = torch.load("materials/JVASP/Train_raman_set_25_uneq_yolov1.pt")
    dataloader = DataLoader(dataset,batch_size=len(dataset))
    model = Backbone().eval()
    data = next(iter(dataloader))
    _, raman = data
    raman = raman[:,:,0]
    atoms,_ = model(data)
    feature_output = atoms[0]
    raman = raman.detach().tolist()
    feature_output = feature_output.detach().tolist()
    with open("materials/JVASP/Train_features.json","w") as f:
        json.dump({"features":feature_output,"raman":raman},f)
    