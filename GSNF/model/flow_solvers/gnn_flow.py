import torch
import torch.nn as nn
import torch.nn.functional as F
from model.flow_solvers.mlp import MLP


class TimeLinear(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, out_dim))
        nn.init.xavier_uniform_(self.scale)

    def forward(self, t):
        return self.scale * t


class TimeTanh(TimeLinear):
    def forward(self, t):
        return torch.tanh(self.scale * t)


class TimeFourier(nn.Module):
    def __init__(self, out_dim, hidden_dim, lmbd=0.5, bounded=False, **kwargs):
        super().__init__()
        self.bounded = bounded
        self.hidden_dim = hidden_dim
        self.shift = nn.Parameter(-torch.log(1 - torch.rand(out_dim, hidden_dim)) / lmbd)
        self.weight = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, t):
        t = t.unsqueeze(-1)
        if self.bounded:
            scale = F.softmax(self.weight, -1) / 2
        else:
            scale = self.weight / self.hidden_dim
        return (scale * torch.sin(self.shift * t)).sum(-1)


class TimeFourierBounded(TimeFourier):
    def __init__(self, out_dim, hidden_dim, lmbd=0.5, **kwargs):
        super().__init__(out_dim, hidden_dim, lmbd, bounded=True)


TIME_NETS = {
    "TimeLinear": TimeLinear,
    "TimeTanh": TimeTanh,
    "TimeFourier": TimeFourier,
    "TimeFourierBounded": TimeFourierBounded,
}

class GNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 nodes: int,
                 activation,
                 final_activation,
                 time_net: str,
                 time_hidden_dim,
                 n_power_iterations: int = 5,
                 bias:int=True,
                 conv_type: str = "basic",
                 gcn_hidden_dim: int = 32,
                 invertible: bool = True):
        super().__init__()
        # Spectral normalization keeps the graph flow contractive.
        if invertible:
            wrapper = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=n_power_iterations)
        else:
            wrapper = lambda layer: layer
        self.nodes = nodes
        self.input_size = input_size
        if conv_type not in {"basic", "residual", "gated"}:
            raise ValueError(f"Unknown GCN flow conv_type: {conv_type}")
        self.conv_type = conv_type
        gcn_hidden_dim = int(gcn_hidden_dim)
        self.net2 = MLP(input_size*self.nodes+3, hidden_size, input_size*self.nodes, activation, final_activation, wrapper_func=wrapper)
        self.net = MLP(input_size*self.nodes*2+3, hidden_size, input_size*self.nodes, activation, final_activation, wrapper_func=wrapper)
        self.lin_n = nn.Sequential(wrapper(nn.Linear(input_size, gcn_hidden_dim)))
        self.lin_r = nn.Sequential(wrapper(nn.Linear(gcn_hidden_dim, input_size)))
        if self.conv_type == "gated":
            self.gate = nn.Sequential(
                wrapper(nn.Linear(input_size * 2, gcn_hidden_dim)),
                nn.ReLU(),
                wrapper(nn.Linear(gcn_hidden_dim, input_size)),
                nn.Sigmoid(),
            )

        self.act = nn.ReLU()
        if time_net not in TIME_NETS:
            raise ValueError(f"Unknown time_net: {time_net}")
        self.time_net = TIME_NETS[time_net](input_size*self.nodes, hidden_dim=time_hidden_dim)

    def forward(self, nodes, x, h, t_start, t_end, adj):
        dt = t_end - t_start
        t_output = self.time_net(dt)
        h_self = h
        if len(adj.shape) == 4 and x.shape[0] != adj.shape[0]:
            adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
        if len(adj.shape) == 4:
            h_n = self.lin_n(torch.einsum('btajd,btji->btaid', h_self, adj))
        elif len(adj.shape) == 5:
            h_n = self.lin_n(torch.einsum('cbtjd,cbtji->cbtid', h_self, adj))
        elif len(adj.shape) == 3:
            h_n = self.lin_n(torch.einsum('cbtjd,bji->cbtid', h_self, adj))
        h_candidate = self.lin_r(self.act(h_n))
        if self.conv_type == "basic":
            h_output = h_candidate
        elif self.conv_type == "residual":
            h_output = h_candidate + h_self
        else:
            gate = self.gate(torch.cat([h_self, h_candidate], dim=-1))
            h_output = gate * h_candidate + (1.0 - gate) * h_self
        h_flat = h_output.reshape(x.shape)
        x_output = x + t_output * torch.mul(
            self.net(torch.cat([x, h_flat, dt, t_start, t_end], -1)),
            self.net2(torch.cat([x, dt, t_start, t_end], -1)))
        return x_output,h_output

class GCNFlowNF(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dims,
        num_layers,
        nodes: int,
        activation='ReLU',
        final_activation=None,
        time_net=None,
        time_hidden_dim=None,
        n_power_iterations=5,
        invertible=True,
        conv_type="basic",
        gcn_hidden_dim=32,
        **kwargs
    ):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(GNN(dim, hidden_dims, nodes, activation, final_activation, time_net,
                                          time_hidden_dim, n_power_iterations=n_power_iterations,
                                          conv_type=conv_type, gcn_hidden_dim=gcn_hidden_dim,
                                          invertible=invertible))
        self.blocks = nn.ModuleList(blocks)


    def forward(self, nodes, x, h, t_start, t_end, adj):
        for block in self.blocks:
            x,h = block(nodes, x, h, t_start, t_end, adj)
        return x,h
    
