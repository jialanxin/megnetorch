import torch
from torch.utils.data import DataLoader
from dataset import StructureRamanDataset
from finetune import Experiment as Finetune
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

class Experiment(Finetune):
    def __init__(self):
        super().__init__()
        pretrain_model = Finetune.load_from_checkpoint(
            "pretrain/finetuned/epoch=1999-step=99999.ckpt")
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
        return predicted_spectrum
    
def decide_with_cut_off(model:Experiment,dataloader:DataLoader):
    for i,data in enumerate(dataloader):
        _,raman = data
        predicted_spectrum =  model(data)
        predicted_confidence = predicted_spectrum[:,:,:,0]
        predicted_confidence_backup,_ = torch.max(predicted_confidence, dim=2)
        predict_confidence = predicted_confidence_backup.clone()
        target_confidence = raman[:,:,0]
    cut_off_list = np.linspace(0.2,0.8,60)
    Ac_list = np.array([])
    Pr_list = np.array([])
    Rc_list = np.array([])
    F1_list = np.array([])
    for cut_off in cut_off_list:
        less = torch.less_equal(predict_confidence,cut_off)
        predict_confidence[less] = 0
        predict_confidence[torch.logical_not(less)] = 1
        Accuracy, Precision,Recall,F1Score = Experiment.scores(predict_confidence,target_confidence)
        predict_confidence = predicted_confidence_backup.clone()
        Ac_list = np.append(Ac_list,Accuracy.item())
        Pr_list = np.append(Pr_list,Precision.item())
        Rc_list = np.append(Rc_list,Recall.item())
        F1_list = np.append(F1_list,F1Score.item())
    fig = make_subplots(rows=1,cols=2)
    fig.add_trace(go.Scatter(x=cut_off_list,y=Ac_list,name="Accuracy"),row=1,col=1)
    fig.add_trace(go.Scatter(x=cut_off_list,y=Pr_list,name="Precision"),row=1,col=1)
    fig.add_trace(go.Scatter(x=cut_off_list,y=Rc_list,name="Recall"),row=1,col=1)
    fig.add_trace(go.Scatter(x=cut_off_list,y=F1_list,name="F1Score"),row=1,col=1)
    fig.add_trace(go.Scatter(x=Rc_list,y=Pr_list,name="Pr-Rc"),row=1,col=2)
    return fig

def decide_with_nms(model:Experiment,dataloader:DataLoader):
    for i,data in enumerate(dataloader):
        _,raman = data
        predicted_spectrum =  model(data)
        predicted_confidence = predicted_spectrum[:,:,:,0] # (batch_size, 25, 2)
        predicted_confidence = predicted_confidence.view((-1,50)) # (batch_size, 50)
        predicted_confidence_bak = predicted_confidence.clone() 
        target_confidence = raman[:,:,0] # (batch_size, 25)
        Ac_list = np.array([])
        Pr_list = np.array([])
        Rc_list = np.array([])
        F1_list = np.array([])
        for top_k in range(1,51):
            predicted_confidence = torch.argsort(predicted_confidence, dim=1, descending=True) # (batch_size, 50)
            predicted_confidence = torch.less_equal(predicted_confidence,top_k-1).float().view((-1,25,2)).sum(dim=-1).sgn() #(batch_size,25)
            Accuracy, Precision,Recall,F1Score = Experiment.scores(predicted_confidence,target_confidence)
            Ac_list = np.append(Ac_list,Accuracy.item())
            Pr_list = np.append(Pr_list,Precision.item())
            Rc_list = np.append(Rc_list,Recall.item())
            F1_list = np.append(F1_list,F1Score.item())
            predicted_confidence = predicted_confidence_bak.clone()
    fig = make_subplots(rows=1,cols=2)
    top_k_list = [i for i in range(1,51)]
    fig.add_trace(go.Scatter(x=top_k_list,y=Ac_list,name="Accuracy"),row=1,col=1)
    fig.add_trace(go.Scatter(x=top_k_list,y=Pr_list,name="Precision"),row=1,col=1)
    fig.add_trace(go.Scatter(x=top_k_list,y=Rc_list,name="Recall"),row=1,col=1)
    fig.add_trace(go.Scatter(x=top_k_list,y=F1_list,name="F1Score"),row=1,col=1)
    fig.add_trace(go.Scatter(x=Rc_list,y=Pr_list,name="Pr-Rc"),row=1,col=2)
    return fig


def NMS_or_not(model, dataloader, nms=False):
    if nms :
        return decide_with_nms(model,dataloader)
    else:
       return decide_with_cut_off(model,dataloader)



if __name__ == "__main__":
    # train_set = torch.load("materials/JVASP/Train_raman_set_25_uneq_yolov1.pt")
    validate_set = torch.load("materials/JVASP/Valid_raman_set_25_uneq_yolov1.pt")
    # # train_dataloader = DataLoader(
    # #     dataset=train_set, batch_size=64, num_workers=1)
    validate_dataloader = DataLoader(
        dataset=validate_set, batch_size=len(validate_set), num_workers=1)
    model = Experiment().eval()
    fig = NMS_or_not(model, validate_dataloader,nms=True)
    fig.show()

# Train:  loss_weight_6_sign
# Accuracy: 88% Precision: 49% Recall: 90% F1Score: 64
# Validate:
# Accuracy: 85% Precision: 44% Recall: 78% F1Score: 55

# nonobj:0.05 cut_off 0.4 L2 1e-1 workers 2
# Validate:
# Accuracy: 81% Precision: 52% Recall: 72% F1Score: 60

# nonobj:0.1 cut_off 0.42
# Validate:
# Accuracy: 79% Precision: 53% Recall: 75% F1Score: 62
# nonobj:0.1 cut_off 0.36 L2 1e-1 workers 2
# Validate:
# Accuracy: 82% Precision: 54% Recall: 72% F1Score: 62

# nonobj:0.2 cut_off 0.4
# Validate:
# Accuracy: 82% Precision: 55% Recall: 72% F1Score: 62
# nonobj:0.2 cut_off 0.41 L2 1e-2
# Validate:
# Accuracy: 83% Precision: 55% Recall: 72% F1Score: 62
# nonobj:0.2 cut_off 0.51 L2 1e-1 workers 2 max
# Validate:
# Accuracy: 83% Precision: 56% Recall: 73% F1Score: 63
# nonobj:0.2 cut_off 0.49 L2 1e-1 workers 2 max reinit 5
# Validate:
# Accuracy: 83% Precision: 55% Recall: 76% F1Score: 64
# nonobj:0.2 cut_off 0.51 L2 1e-1 workers 2 max reinit 45
# Validate:
# Accuracy: 83% Precision: 57% Recall: 72% F1Score: 63
# nonobj:0.2 cut_off 0.49 L2 1e-2 workers 2 max lr 2e-4
# Validate:
# Accuracy: 84% Precision: 57% Recall: 72% F1Score: 64
# nonobj:0.2 cut_off 0.50 L2 1e-2 workers 2 max lr 2e-4 reinit 5
# Validate:
# Accuracy: 84% Precision: 58% Recall: 72% F1Score: 64
# nonobj:0.2 cut_off 0.49 L2 1e-2 workers 2 max lr 5e-4
# Validate:
# Accuracy: 83% Precision: 55% Recall: 74% F1Score: 63

# nonobj:0.3 cut_off 0.37
# Validate:
# Accuracy: 83% Precision: 55% Recall: 70% F1Score: 62