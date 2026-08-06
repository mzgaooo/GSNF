import torch.nn as nn

from model.flow_solvers.gnn_flow import GCNFlowNF


class GNNFlow(nn.Module):
    def __init__(self, dim, nodes, n_layers, hidden_dims, time_net,
                 time_hidden_dim, **kwargs):
        super().__init__()
        if time_net != "TimeFourier":
            raise ValueError("the focused release uses TimeFourier conditioning")
        if kwargs.get("gnn_conv", "basic") != "basic":
            raise ValueError("the focused release uses basic graph propagation")
        if kwargs.get("graph_direction", "col") != "col":
            raise ValueError("the focused release uses column propagation")
        self.nodes = int(nodes)
        self.layers = nn.ModuleList([
            GCNFlowNF(
                dim,
                hidden_dims,
                n_layers,
                nodes=self.nodes,
                time_hidden_dim=time_hidden_dim,
            )
            for _ in range(n_layers)
        ])

    def forward(self, nodes, state, hidden, start, end, adjacency):
        if hidden is not None and hidden.shape[-3] == 1:
            hidden = hidden.repeat_interleave(end.shape[-2], dim=-3)
        if state.shape[-2] == 1:
            state = state.repeat_interleave(end.shape[-2], dim=-2)
        for layer in self.layers:
            state, hidden = layer(nodes, state, hidden, start, end, adjacency)
        return state
