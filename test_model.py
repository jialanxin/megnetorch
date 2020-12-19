from torch.nn import Embedding
from torch_geometric.nn import Set2Set

net = Embedding(95,16)
print(sum(p.numel() for p in net.parameters()))

from model import ff,MegNetLayer,ff_output

net = ff(2)
print(sum(p.numel() for p in net.parameters()))

net=MegNetLayer()
print(sum(p.numel() for p in net.parameters()))

net = ff(32)
print(sum(p.numel() for p in net.parameters()))

net = Set2Set(32,processing_steps=3)  # keras 9376
print(sum(p.numel() for p in net.parameters()))

net = ff_output(160,200)
print(sum(p.numel() for p in net.parameters()))