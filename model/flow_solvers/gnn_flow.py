import torch
import torch.nn as nn

from model.flow_solvers.mlp import MLP


class TimeFourier(nn.Module):
    def __init__(self, out_dim, hidden_dim, rate=0.5):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.shift = nn.Parameter(
            -torch.log(1.0 - torch.rand(out_dim, self.hidden_dim)) / rate)
        self.weight = nn.Parameter(torch.empty(out_dim, self.hidden_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, delta):
        phase = self.shift * delta.unsqueeze(-1)
        return (self.weight * torch.sin(phase) / self.hidden_dim).sum(dim=-1)


class GraphFlowBlock(nn.Module):
    def __init__(self, input_size, hidden_size, nodes, time_hidden_dim):
        super().__init__()
        wrap = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=5)
        self.nodes = int(nodes)
        self.input_size = int(input_size)
        self.base = MLP(
            input_size * self.nodes + 3,
            hidden_size,
            input_size * self.nodes,
            "ReLU",
            None,
            wrapper_func=wrap,
        )
        self.update = MLP(
            input_size * self.nodes * 2 + 3,
            hidden_size,
            input_size * self.nodes,
            "ReLU",
            None,
            wrapper_func=wrap,
        )
        self.message_in = wrap(nn.Linear(input_size, 64))
        self.message_out = wrap(nn.Linear(64, input_size))
        self.activation = nn.ReLU()
        self.time_net = TimeFourier(input_size * self.nodes, time_hidden_dim)

    def forward(self, nodes, state, hidden, start, end, adjacency):
        delta = end - start
        time_scale = self.time_net(delta)
        adjacency = adjacency.transpose(-1, -2)
        if adjacency.dim() == 4 and state.shape[0] != adjacency.shape[0]:
            adjacency = adjacency.unsqueeze(0).expand(state.shape[0], -1, -1, -1, -1)
        if adjacency.dim() == 4:
            message = torch.einsum("btajd,btji->btaid", hidden, adjacency)
        elif adjacency.dim() == 5:
            message = torch.einsum("cbtjd,cbtji->cbtid", hidden, adjacency)
        elif adjacency.dim() == 3:
            message = torch.einsum("cbtjd,bji->cbtid", hidden, adjacency)
        else:
            raise ValueError("adjacency must have rank 3, 4, or 5")
        hidden_out = self.message_out(self.activation(self.message_in(message)))
        hidden_flat = hidden_out.reshape(state.shape)
        common = torch.cat([state, delta, start, end], dim=-1)
        state_out = state + time_scale * self.update(
            torch.cat([state, hidden_flat, delta, start, end], dim=-1)
        ) * self.base(common)
        return state_out, hidden_out


class GCNFlowNF(nn.Module):
    def __init__(self, dim, hidden_dims, num_layers, nodes, time_hidden_dim, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            GraphFlowBlock(dim, hidden_dims, nodes, time_hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, nodes, state, hidden, start, end, adjacency):
        for block in self.blocks:
            state, hidden = block(nodes, state, hidden, start, end, adjacency)
        return state, hidden
