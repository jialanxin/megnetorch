import torch
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class Set2Set(torch.nn.Module):

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()


    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

input = torch.rand((4,2))
inputs = torch.rand((6,2))
new_input  = torch.cat([input,inputs])
batch = torch.zeros(4,dtype=torch.long)
batchs = torch.zeros(6,dtype=torch.long)
new_batch = torch.cat([batch,batchs+1])
set2set=Set2Set(in_channels=2,processing_steps=1)

output = set2set(new_input,new_batch)
print("end")

input = torch.rand((7,330,32))
input = torch.flatten(input,end_dim=1)
batch = torch.LongTensor([i for i in range(7)]).unsqueeze(dim=1).repeat((1,330)).flatten(end_dim=1)
print(batch.shape)