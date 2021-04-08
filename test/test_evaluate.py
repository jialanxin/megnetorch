from torch.utils.data import dataloader
import evaluate
from torch.utils.data import DataLoader
import torch

def test_model_output():
    dataset = evaluate.load_dataset("Valid")
    dataloader = DataLoader(dataset,batch_size=1)
    model = evaluate.Experiment()
    i,data = next(enumerate(dataloader))
    graph, ramans, formula, mp_id = data
    assert ramans.shape == (1,25,2)
    input = (graph, ramans)
    predicted_spectrum = model(input)
    assert predicted_spectrum.shape == (1,25,2,2)
