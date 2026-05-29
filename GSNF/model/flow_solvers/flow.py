from typing import Optional

import torch.nn as nn

from model.flow_solvers.gnn_flow import GCNFlowNF


class GNNFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        nodes: int,
        n_layers: int,
        hidden_dims,
        time_net: str,
        time_hidden_dim: Optional[int],
        gnn_conv: str = "basic",
        gcn_hidden_dim: int = 32,
        invertible: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.nodes = int(nodes)
        layers = []
        for _ in range(n_layers):
            layers.append(
                GCNFlowNF(
                    dim,
                    hidden_dims,
                    n_layers,
                    nodes=self.nodes,
                    activation="ReLU",
                    final_activation=None,
                    time_net=time_net,
                    time_hidden_dim=time_hidden_dim,
                    invertible=invertible,
                    conv_type=gnn_conv,
                    gcn_hidden_dim=gcn_hidden_dim,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, nodes, x, h, t_start, t_end, adj):
        if h is not None and h.shape[-3] == 1:
            h = h.repeat_interleave(t_end.shape[-2], dim=-3)
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t_end.shape[-2], dim=-2)

        for layer in self.layers:
            x, h = layer(nodes, x, h, t_start, t_end, adj)
        return x
